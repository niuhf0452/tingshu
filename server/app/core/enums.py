"""Predefined enumerations shared between book metadata, voice library and TTS API.

See docs/technical-plan.md §一 for the canonical spec.
"""
from __future__ import annotations

from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class Age(str, Enum):
    CHILD = "child"
    TEEN = "teen"
    YOUTH = "youth"
    ADULT = "adult"
    ELDER = "elder"


class Personality(str, Enum):
    CALM = "calm"
    GENTLE = "gentle"
    CHEERFUL = "cheerful"
    SERIOUS = "serious"
    COLD = "cold"
    FIERCE = "fierce"
    DETERMINED = "determined"
    TIMID = "timid"
    PLAYFUL = "playful"
    MATURE = "mature"
    NAIVE = "naive"
    WISE = "wise"
    ARROGANT = "arrogant"
    KIND = "kind"
    CUNNING = "cunning"
    BRAVE = "brave"
    MELANCHOLY = "melancholy"
    PASSIONATE = "passionate"


class Tone(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    GENTLE = "gentle"
    SERIOUS = "serious"
    PLAYFUL = "playful"
    WHISPER = "whisper"


class BookStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
