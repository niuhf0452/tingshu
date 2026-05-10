"""TTS synthesis endpoint.

Input: ``(book_id, chapter_id, character_id, text, tone)``.

Flow:
  1. Resolve ``character_id`` to a Speaker:
     - 0..15 → predefined narrator voice (see ``core.narrator``).
     - ≥ 16 → look up the character in the book's ``characters.json``,
       then attribute-match against the voice library.
  2. Consult disk cache (key = ``sha1(speaker_id || text)`` — speed
     and tone are intentionally absent; speed is applied client-side
     via ``AVAudioUnitTimePitch``, and tone variations are accepted
     in exchange for higher cache hit rate).
  3. Synthesize via the configured backend on miss.
  4. Return audio with the matched ``X-Speaker-*`` headers.

Audio format is M4A (AAC, 48 kbps mono, 24 kHz) — see
``app/services/tts_qwen3.py`` for the rationale (~7-9× smaller than raw
WAV at perceptual transparency for speech). The stub backend still emits
WAV; only the qwen3_tts backend AAC-encodes. Either is decodable by
iOS's AVAudioFile without code changes.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from ..core.models import Speaker, TTSRequest
from ..core.storage import BookRepository
from ..core.voice import SpeakerResolutionError, resolve_speaker
from ..services.tts import NoSpeakerError, TTSService
from .deps import get_repository, get_tts_service, get_voice_library


log = logging.getLogger(__name__)


router = APIRouter(prefix="/api", tags=["tts"])


@router.post("/tts")
async def synthesize(
    req: TTSRequest,
    repo: Annotated[BookRepository, Depends(get_repository)],
    library: Annotated[list[Speaker], Depends(get_voice_library)],
    tts: Annotated[TTSService, Depends(get_tts_service)],
) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    if not library:
        raise HTTPException(
            status_code=503,
            detail="voice library is empty — populate data/voices/speakers.json",
        )

    # Look up book characters only when needed (book character path).
    # Narrator path doesn't read characters.json at all.
    book_characters = []
    if req.character_id > 15:
        try:
            book_characters = repo.load_characters(req.book_id)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=f"book {req.book_id} not found",
            ) from exc

    try:
        speaker = resolve_speaker(
            character_id=req.character_id,
            library=library,
            book_characters=book_characters,
        )
    except SpeakerResolutionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        # TTS inference is GPU/CPU-bound and can take seconds.
        # Run off the event loop so concurrent requests aren't serialised
        # at the asyncio layer (they still serialise inside the backend's
        # own lock — but at least other endpoints stay responsive).
        audio = await run_in_threadpool(
            tts.synthesize,
            text=req.text,
            speaker=speaker,
            tone=req.tone,
        )
    except NoSpeakerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(
        content=audio,
        # Sniff content — production qwen3 backend returns AAC in M4A;
        # the test stub returns WAV. iOS's AVAudioFile decodes both
        # transparently, but the right Content-Type helps clients /
        # caches / log readers identify the payload correctly.
        media_type=_sniff_media_type(audio),
        headers={
            "X-Speaker-Id": speaker.speaker_id,
            "X-Speaker-Gender": speaker.gender.value,
            "X-Speaker-Age": speaker.age.value,
            "Cache-Control": "public, max-age=31536000",  # deterministic by params
        },
    )


@router.delete("/tts/cache", status_code=204)
async def clear_tts_cache(
    tts: Annotated[TTSService, Depends(get_tts_service)],
) -> Response:
    """Wipe every server-side cached audio file.

    Used by the iOS settings page's "本地 + 服务端" clear option after
    repeated character-voice edits leave a long tail of orphan entries
    that the per-book client-side eviction can't reach.

    Filesystem-level ``.m4a`` removal — the directory itself stays
    (``main.lifespan`` creates it at startup). In-flight ``put`` calls
    racing this finish via atomic-rename, so a concurrent synth lands
    a fresh entry rather than corrupting one.

    Runs in the threadpool because a populated cache can hold tens of
    thousands of files; ``Path.iterdir`` + per-file ``unlink`` blocks
    the loop otherwise.
    """
    removed = await run_in_threadpool(tts.cache.clear)
    log.info("server tts cache cleared: removed=%d files", removed)
    return Response(status_code=204)


def _sniff_media_type(audio: bytes) -> str:
    """Inspect the first few bytes to decide the response Content-Type.
    No fancy parsing — magic-number check on RIFF (WAV) and ftyp (M4A)
    headers is enough since those are the only two formats this server
    ever produces.
    """
    if len(audio) >= 12 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return "audio/wav"
    if len(audio) >= 8 and audio[4:8] == b"ftyp":
        return "audio/mp4"  # MIME for M4A (ISO/IEC 14496-14)
    return "application/octet-stream"
