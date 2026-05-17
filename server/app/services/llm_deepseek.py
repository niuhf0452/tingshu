"""DeepSeek V4 (flash, no-thinking) LLM backend.

OpenAI-compatible HTTP transport. All semantic content — prompts,
output schemas, parsers — lives in ``llm_prompts``; this module only
deals with one-shot HTTP chat completions.

Thinking is **always off** — experiments showed it bills extra tokens
without measurably improving accuracy on these tasks.

Chapter analysis is split into two independent calls (segmentation +
character updates), invoked in parallel by ``BookService``. Most calls
are single-shot; ``segment_chapter`` is the exception — it splits long
chapters into line-aligned batches (one HTTP call each) so the NDJSON
output never truncates against ``max_tokens``. Concurrency across
chapters is the caller's responsibility.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Callable, TypeVar

from ..core.models import (
    AnalyzedSentence,
    ChapterDetection,
    Character,
    ClassifiedCharacters,
)
from . import llm_prompts as P
from .llm import LLMClient


log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"

_MAX_TOKENS_DETECT_CHAPTERS = 4096
# Headroom for chapter-analysis calls. Segmentation is the longest
# (one line per reading segment); character classification + profiling
# are shorter (one line per character) but reuse the same ceiling for
# simplicity.
_MAX_TOKENS_CHAPTER = 16384
_MAX_TOKENS_CHARACTERS = 4096


T = TypeVar("T")


class DeepSeekLLMClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 600.0,
    ):
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # --- LLMClient methods ---

    def detect_chapters(
        self,
        opening_text: str,
        *,
        known_titles: list[str] | None = None,
    ) -> ChapterDetection:
        if not opening_text.strip():
            return ChapterDetection()
        messages = P.build_chapter_detection_messages(opening_text, known_titles)
        return self._call_with_parse(
            messages=messages,
            parse=P.parse_chapter_detection,
            default=ChapterDetection(),
            max_tokens=_MAX_TOKENS_DETECT_CHAPTERS,
            label="detect_chapters",
        )

    def classify_chapter_characters(
        self,
        chapter_text: str,
        known_characters: list[Character],
    ) -> ClassifiedCharacters:
        messages = P.build_classify_chapter_characters_messages(
            chapter_text, known_characters,
        )
        def parse(raw: str) -> ClassifiedCharacters:
            return P.parse_classified_characters(raw)
        return self._call_with_parse(
            messages=messages,
            parse=parse,
            default=ClassifiedCharacters(),
            max_tokens=_MAX_TOKENS_CHARACTERS,
            label="classify_chapter_characters",
        )

    def profile_new_characters(
        self,
        name_to_contexts: dict[str, list[str]],
        known_characters: list[Character],
    ) -> list[Character]:
        if not name_to_contexts:
            return []
        messages = P.build_profile_new_characters_messages(
            name_to_contexts, known_characters,
        )
        def parse(raw: str) -> list[Character]:
            return P.parse_character_updates(raw)
        return self._call_with_parse(
            messages=messages,
            parse=parse,
            default=[],
            max_tokens=_MAX_TOKENS_CHARACTERS,
            label="profile_new_characters",
        )

    def segment_chapter(
        self,
        chapter_text: str,
        known_characters: list[Character],
    ) -> list[AnalyzedSentence]:
        """Segment a chapter into reading segments.

        The NDJSON output restates the whole chapter, so a long chapter
        overruns ``_MAX_TOKENS_CHAPTER`` in a single call — the response
        truncates and the chapter's tail is silently lost. The chapter is
        therefore split into line-aligned batches whose estimated output
        fits the token budget; each batch is one LLM call and the parsed
        results are concatenated in chapter order (which keeps the
        downstream ``locate_sentences`` forward cursor monotonic).

        The LLM only cuts on speaker changes and sentence-ending
        punctuation; over-long segments are then split at clause
        boundaries by ``split_long_segments`` — a deterministic Python
        pass, since LLMs honour length caps poorly.
        """
        batches = P.split_chapter_for_segmentation(chapter_text, _MAX_TOKENS_CHAPTER)

        def parse(raw: str) -> list[AnalyzedSentence]:
            return P.parse_segmented_chapter(raw)

        raw_sentences: list[AnalyzedSentence] = []
        for i, batch in enumerate(batches, start=1):
            messages = P.build_segment_chapter_messages(batch, known_characters)
            sentences = self._call_with_parse(
                messages=messages,
                parse=parse,
                default=[],
                max_tokens=_MAX_TOKENS_CHAPTER,
                label=f"segment_chapter[{i}/{len(batches)}]",
            )
            raw_sentences.extend(sentences)

        all_sentences = P.split_long_segments(raw_sentences)
        log.info(
            "segment_chapter: %d chars -> %d batch(es) -> %d segments "
            "(%d after long-sentence split)",
            len(chapter_text), len(batches), len(raw_sentences), len(all_sentences),
        )
        return all_sentences

    # --- transport ---

    def _chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
    ) -> tuple[dict, dict]:
        """One non-streaming chat completion. Returns (assistant_msg, usage)."""
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "thinking": {"type": "disabled"},
        }
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_bytes = exc.read() if hasattr(exc, "read") else b""
            raise RuntimeError(
                f"DeepSeek HTTP {exc.code}: "
                f"{body_bytes.decode('utf-8', errors='replace')[:500]}"
            ) from exc
        choice = payload["choices"][0]
        msg = choice["message"]
        usage = payload.get("usage", {})
        # Safety net: if the backend stopped because it hit max_tokens, the
        # content is truncated mid-stream. Batching is sized to avoid this,
        # but a mis-estimate must not fail silently — surface it loudly so
        # the cause is visible instead of just "fewer sentences than lines".
        if choice.get("finish_reason") == "length":
            log.warning(
                "DeepSeek response hit max_tokens=%d — output TRUNCATED "
                "(completion_tokens=%s); downstream result is incomplete",
                max_tokens, usage.get("completion_tokens", "?"),
            )
        return msg, usage

    def _call_with_parse(
        self,
        *,
        messages: list[dict],
        parse: Callable[[str], T | None],
        default: T,
        max_tokens: int,
        label: str,
    ) -> T:
        t0 = time.monotonic()
        msg, usage = self._chat(messages, max_tokens=max_tokens)
        text = msg.get("content") or ""
        elapsed = time.monotonic() - t0
        log.info(
            "deepseek call [%s]: prompt_tokens=%s completion_tokens=%s "
            "resp_chars=%d wall=%.1fs",
            label,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
            len(text), elapsed,
        )
        parsed = parse(text)
        if parsed is None:
            log.warning("deepseek call [%s]: parse failed; returning default", label)
            return default
        return parsed
