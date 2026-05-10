"""Manual validation harness for the chapter-analysis prompt changes.

Runs the **real** DeepSeek client end-to-end against a small synthetic
chapter that's been engineered to exercise the recent prompt edits:

1. ``classify_chapter_characters`` (A1)
   - Distinguish a real new named character (萧炎) from descriptor-named
     incidentals (妇人, 仆人).
   - Emit incidentals as ``{"k":"incidental","c":"妇人",...}`` with full
     inline profile.

2. ``segment_chapter`` (B1)
   - With both real characters and incidentals already on the known list,
     B1 should attribute the relevant lines to "萧炎" / "妇人" / "仆人"
     and never invent a name.
   - Long-sentence splitting: a paragraph that's > 50 characters with
     only a single sentence-end punctuation should split at commas
     (greedy ~50-char chunks), not be left as one giant segment.
   - Short comma chains (< 50 chars total) should NOT be split.

Usage::

    cd server
    source .venv/bin/activate
    DEEPSEEK_API_KEY=sk-... python -m scripts.validate_prompt_changes

Run this manually when prompt text changes — it's a one-shot ~$0.001
sanity check, not part of the pytest suite.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running both as `python scripts/validate_prompt_changes.py` and
# as `python -m scripts.validate_prompt_changes`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.enums import Age, Gender, Personality
from app.core.models import Character
from app.services.llm_deepseek import DeepSeekLLMClient


# Synthetic chapter that hits all four buckets in a small token budget.
# Line numbers in comments map to what the LLM sees.
CHAPTER_TEXT = """\
第一章 偶遇

萧炎走在青阳镇的青石板路上，远处传来叫卖声。

街角处，一名妇人提着竹篮迎面而来，看见萧炎神情萎靡，便停下脚步，关切地问道：“小哥，你脸色这么差，是不是染了风寒？我家里还有些姜汤，要不要进来喝一碗暖暖身子？”

萧炎摇了摇头，轻声答道：“不必了，多谢大娘。”

他想起当年那个清晨，自己第一次离家远行时，母亲站在门口默默看着他渐行渐远，没有挥手，也没有说话，那一刻他突然意识到这是他人生第一次真正意义上独自远行，肩上的包袱也仿佛重了几分。

萧炎深吸一口气，继续向药铺走去。途中又遇见一名仆人，正在门口扫地，见他过来，连忙退到一边，恭敬地说道：“客人请进。”
"""


# Known roster passed into A1 — only the narrator. Forces every
# character in the text to be classified as new / incidental.
KNOWN: list[Character] = [Character(id=0, name="旁白")]


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        # Fall back to config.yaml — same source the running server uses.
        from app.config import get_settings
        api_key = get_settings().llm.deepseek_api_key.strip()
    if not api_key:
        print("ERROR: no DeepSeek API key.", file=sys.stderr)
        print(
            "       set DEEPSEEK_API_KEY env or llm.deepseek_api_key in "
            "config.yaml.",
            file=sys.stderr,
        )
        sys.exit(2)

    client = DeepSeekLLMClient(api_key=api_key)

    print("=" * 72)
    print("Phase A1: classify_chapter_characters")
    print("=" * 72)
    classified = client.classify_chapter_characters(CHAPTER_TEXT, KNOWN)
    print(f"\nnew_names ({len(classified.new_names)}):")
    for n in classified.new_names:
        print(f"  - {n}")
    print(f"\nevolved ({len(classified.evolved)}):")
    for c in classified.evolved:
        print(f"  - {c.name}: {_short(c)}")
    print(f"\nincidentals ({len(classified.incidentals)}):")
    for c in classified.incidentals:
        print(f"  - {c.name}: {_short(c)}")

    # Heuristic checks the human reviewer should also eyeball.
    print("\n--- A1 expectations ---")
    _expect("萧炎 is in new_names", "萧炎" in classified.new_names)
    incidental_names = {c.name for c in classified.incidentals}
    _expect(
        "incidentals contains the woman speaker (妇人 / 大娘 / similar)",
        any(n in incidental_names for n in ("妇人", "大娘", "妇女")),
    )
    _expect(
        "incidentals contains the servant speaker (仆人)",
        any(n in incidental_names for n in ("仆人", "下人", "家仆")),
    )
    _expect(
        "incidentals do NOT include 萧炎",
        "萧炎" not in incidental_names,
    )

    print()
    print("=" * 72)
    print("Phase B1: segment_chapter (with full known list)")
    print("=" * 72)
    # Build the post-A1 known list the way service.py does: start from
    # KNOWN, append profiled new characters with placeholder ids, append
    # incidentals.
    a3_profiles = client.profile_new_characters(
        {n: [] for n in classified.new_names}, KNOWN,
    )
    full_known = list(KNOWN)
    next_id = 16
    for p in a3_profiles:
        full_known.append(_with_id(p, next_id))
        next_id += 1
    incidental_id = -1
    for c in classified.incidentals:
        full_known.append(_with_id(c, incidental_id))
        incidental_id -= 1

    segments = client.segment_chapter(CHAPTER_TEXT, full_known)
    print(f"\nsegments ({len(segments)}):")
    for s in segments:
        print(f"  [{s.tone.value:>8}] {s.speaker:<8} | {len(s.text):>3}字 | {s.text}")

    print("\n--- B1 expectations ---")
    speakers = {s.speaker for s in segments}
    _expect("at least one segment by 萧炎", "萧炎" in speakers)
    _expect(
        "at least one segment by an incidental woman",
        any(s in speakers for s in ("妇人", "大娘", "妇女")),
    )
    _expect(
        "at least one segment by an incidental servant",
        any(s in speakers for s in ("仆人", "下人", "家仆")),
    )
    invented = speakers - {c.name for c in full_known} - {"旁白"}
    _expect(
        "no invented speakers (all in known list or 旁白)",
        not invented,
        details=f"invented: {invented}" if invented else None,
    )

    # 50-char rule: the long memory paragraph should be split into
    # multiple ~50-char segments. The rest of the chapter should NOT
    # be over-split.
    long_para = (
        "他想起当年那个清晨，自己第一次离家远行时，母亲站在门口默默看着他渐行渐远，"
        "没有挥手，也没有说话，那一刻他突然意识到这是他人生第一次真正意义上独自远行，"
        "肩上的包袱也仿佛重了几分。"
    )
    long_segments = [s for s in segments if s.text and s.text in long_para]
    print(f"\nLong paragraph produced {len(long_segments)} segments "
          f"(expect 2-4 in the 30-55 char band):")
    for s in long_segments:
        print(f"  {len(s.text):>3}字 | {s.text}")
    _expect(
        "long paragraph split into >=2 segments",
        len(long_segments) >= 2,
    )
    if long_segments:
        too_long = [s for s in long_segments if len(s.text) > 70]
        _expect(
            "no long-paragraph segment exceeds ~70 chars",
            not too_long,
            details=(
                f"oversized: {[len(s.text) for s in too_long]}"
                if too_long else None
            ),
        )
        too_short = [s for s in long_segments if len(s.text) < 20]
        _expect(
            "no over-eager comma splits (<20 chars)",
            not too_short,
            details=(
                f"too short: {[(len(s.text), s.text) for s in too_short]}"
                if too_short else None
            ),
        )

    # Short comma chain shouldn't be split. The first 萧炎 narration
    # line in the chapter is short (<50 chars including punctuation).
    short_intro = "萧炎走在青阳镇的青石板路上，远处传来叫卖声。"
    short_pieces = [
        s for s in segments if s.text and s.text in short_intro
    ]
    print(f"\nShort intro paragraph produced {len(short_pieces)} segments "
          f"(expect exactly 1 — it's < 50 chars total):")
    for s in short_pieces:
        print(f"  {len(s.text):>3}字 | {s.text}")
    _expect(
        "short comma chain not split",
        len(short_pieces) == 1,
    )


def _short(c: Character) -> str:
    pers = "/".join(p.value for p in c.personality)
    return (
        f"gender={c.gender.value} age={c.age.value} "
        f"personality=[{pers}] identity={c.identity!r}"
    )


def _with_id(c: Character, new_id: int) -> Character:
    return Character(
        id=new_id, name=c.name, identity=c.identity,
        gender=c.gender, age=c.age, personality=list(c.personality),
    )


_pass_count = 0
_fail_count = 0


def _expect(description: str, ok: bool, details: str | None = None) -> None:
    global _pass_count, _fail_count
    mark = "✓" if ok else "✗"
    print(f"  {mark} {description}")
    if details:
        print(f"      → {details}")
    if ok:
        _pass_count += 1
    else:
        _fail_count += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        print()
        print("=" * 72)
        print(f"Result: {_pass_count} passed, {_fail_count} failed")
        print("=" * 72)
        sys.exit(1 if _fail_count else 0)
