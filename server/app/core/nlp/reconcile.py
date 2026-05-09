"""Character reconciliation for the chapter-meta pipeline (§2.3).

Per the post-refactor flow, the cumulative roster is **fully populated
before** segmentation runs (Phase A: classify → cross-chapter intro
context search → profile new characters → save). Reconciliation is
therefore the simpler "post-segmentation" step:

Given:
- ``known``: the cumulative character roster (already includes any
  new / evolved characters discovered earlier in this chapter).
- ``speakers``: every speaker string from ``segment_chapter``'s output.

Produce:
- ``speaker_to_id``: map every speaker name → stable character_id.
- ``chapter_characters``: snapshot of all characters that actually
  spoke in this chapter (for ``ChapterMeta.characters``).

Rules:
- Narrator is fixed at id 0 (``NARRATOR_ID``). Speaker names matching
  any narrator alias resolve to id 0.
- A speaker name in ``known`` resolves to that character's id.
- A speaker name **not** in ``known`` resolves to the narrator
  (``NARRATOR_ID``). This is the "如果仍存在无法识别的角色 / 对话归因，
  算作旁白" rule from the design — Phase A is supposed to register
  every character before we get here, so an unknown name is a glitch
  and the safe fallback is to keep the line audible (in narrator
  voice) rather than fabricate a default character with no profile.
"""
from __future__ import annotations

from ..models import Character
from ..narrator import NARRATOR_ID_MAX


NARRATOR_ID = 0
NARRATOR_NAMES = {"旁白", "narrator", "", "叙述", "叙述者"}

# Book characters start at this id. Ids 0..NARRATOR_ID_MAX (15) are
# reserved for predefined narrator voices in ``app.core.narrator``;
# allocating a book character into that range would silently route the
# character through the narrator voice table.
FIRST_BOOK_CHARACTER_ID = NARRATOR_ID_MAX + 1


def reconcile_chapter_speakers(
    *,
    known: list[Character],
    speakers: list[str],
) -> tuple[dict[str, int], list[Character]]:
    """See module docstring.

    Returns ``(speaker_to_id, chapter_characters)``.
    """
    by_name: dict[str, Character] = {c.name: c for c in known}

    speaker_to_id: dict[str, int] = {}
    speakers_in_chapter: list[str] = []  # ordered, deduped, non-narrator only
    seen_in_chapter: set[str] = set()

    for raw in speakers:
        name = (raw or "").strip()
        if name in speaker_to_id:
            continue
        if _is_narrator(name) or name not in by_name:
            speaker_to_id[name] = NARRATOR_ID
            continue
        speaker_to_id[name] = by_name[name].id
        if name not in seen_in_chapter:
            speakers_in_chapter.append(name)
            seen_in_chapter.add(name)

    chapter_characters = [by_name[n] for n in speakers_in_chapter]
    return speaker_to_id, chapter_characters


def merge_character_updates(
    *,
    known: list[Character],
    updates: list[Character],
) -> tuple[list[Character], int, int]:
    """Apply Phase-A character updates to the cumulative roster.

    ``updates`` may contain:
    - **new** characters (name not in ``known``) — appended with the
      next available id.
    - **evolved** characters (name in ``known``) — profile is overwritten
      while id is preserved (this is how character "growth" works:
      later chapters' voice matching sees the new attributes).

    Narrator entries in ``updates`` are silently skipped — narrator is
    fixed and doesn't carry a profile.

    Returns ``(updated_known, new_count, evolved_count)``.
    """
    by_name: dict[str, Character] = {c.name: c for c in known}
    # Book characters always allocate from FIRST_BOOK_CHARACTER_ID (16)
    # upwards, so a fresh roster never produces ids that collide with
    # the narrator-reserved range. Existing high ids are preserved by
    # taking the max — re-analyses don't shift ids around.
    next_id = max(
        FIRST_BOOK_CHARACTER_ID,
        max((c.id for c in known), default=NARRATOR_ID) + 1,
    )

    new_count = 0
    evolved_count = 0

    for upd in updates:
        name = upd.name.strip()
        if not name or _is_narrator(name):
            continue
        if name in by_name:
            existing = by_name[name]
            by_name[name] = Character(
                id=existing.id,
                name=name,
                identity=upd.identity,
                gender=upd.gender,
                age=upd.age,
                personality=list(upd.personality),
            )
            evolved_count += 1
        else:
            by_name[name] = Character(
                id=next_id,
                name=name,
                identity=upd.identity,
                gender=upd.gender,
                age=upd.age,
                personality=list(upd.personality),
            )
            next_id += 1
            new_count += 1

    return list(by_name.values()), new_count, evolved_count


def _is_narrator(speaker: str) -> bool:
    return speaker.lower() in NARRATOR_NAMES
