"""Narrator voice routing.

The narrator is a special role that doesn't go through attribute-based
voice matching. Instead, ``character_id``s 0..15 are reserved for
predefined narrator voices, with the speaker_id mapping configured
here. Currently 0 = male narrator (default), 1 = female narrator;
ids 2..15 are reserved for future expansion (regional narrators,
genre-specific narrators, etc.).

Why a fixed mapping instead of attribute matching: an audiobook's
narrator is a deliberate production choice, not a function of the
narrator's "personality". The user picks one voice and that voice
reads every narration line. The reserved id range keeps narrator
ids cleanly separated from book character ids (which start at 16),
so a Sentence's character_id alone tells you which voice category
it belongs to.

Both the server and the iOS app know this mapping — the iOS app's
settings page lets the user pick which narrator id to send for any
sentence whose original character_id is 0.
"""
from __future__ import annotations


# Reserved id range. character_ids in [0, NARRATOR_ID_MAX] map through
# this table; ids >= NARRATOR_ID_MAX + 1 (= 16) are book characters.
NARRATOR_ID_MAX = 15

NARRATOR_SPEAKERS: dict[int, str] = {
    0: "zs:vd_narrator_male_mature",     # 中年男旁白（默认）
    1: "zs:vd_narrator_female_adult",    # 流畅女旁白
    # 2..15 reserved for future narrators
}


def is_narrator_id(character_id: int) -> bool:
    return 0 <= character_id <= NARRATOR_ID_MAX


def speaker_id_for_narrator(character_id: int) -> str | None:
    """Return the predefined speaker_id for a narrator id, or None if
    the id is in the reserved range but not yet mapped."""
    return NARRATOR_SPEAKERS.get(character_id)
