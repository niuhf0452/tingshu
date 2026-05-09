"""Shared FastAPI dependencies: settings, repository, service."""
from __future__ import annotations

from functools import lru_cache

from ..config import Settings, get_settings
from ..core.models import Speaker
from ..core.service import BookService
from ..core.storage import BookRepository
from ..core.tts_cache import TTSCache
from ..core.voice import load_voice_library
from ..services.llm import LLMClient
from ..services.llm_factory import create_llm_client
from ..services.tts import TTSService
from ..services.tts_factory import create_tts_client
from .chapter_meta_stream import ChapterMetaStreamManager


@lru_cache(maxsize=1)
def get_repository() -> BookRepository:
    settings: Settings = get_settings()
    return BookRepository(settings.books_dir)


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    settings = get_settings()
    return create_llm_client(settings.llm)


@lru_cache(maxsize=1)
def get_book_service() -> BookService:
    return BookService(
        repo=get_repository(),
        llm=get_llm_client(),
    )


@lru_cache(maxsize=1)
def get_chapter_meta_stream_manager() -> ChapterMetaStreamManager:
    """Process-wide manager for SSE chapter-meta streams. Single-instance
    so concurrent requests for the same chapter share one LLM job. Tests
    can override via ``app.dependency_overrides`` to inject a fresh
    manager bound to a fixture-built service.
    """
    return ChapterMetaStreamManager(service=get_book_service())


@lru_cache(maxsize=1)
def get_voice_library() -> list[Speaker]:
    """Process-wide voice library. Loaded once from speakers.json on
    first access. Speaker resolution lives in the TTS endpoint
    (``api/tts.py``) which combines this with the per-book
    ``characters.json`` to map ``character_id`` → Speaker."""
    settings = get_settings()
    return load_voice_library(settings.voice_library_path)


@lru_cache(maxsize=1)
def get_tts_service() -> TTSService:
    settings = get_settings()
    return TTSService(
        client=create_tts_client(settings.tts),
        cache=TTSCache(settings.tts_cache_dir),
    )
