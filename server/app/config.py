"""Configuration loader: defaults -> config.yaml -> environment variables.

Env vars override YAML. Missing config.yaml is fine — defaults work for dev.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


SERVER_DIR = Path(__file__).resolve().parent.parent


class StorageSection(BaseModel):
    data_dir: str = "data"


class PreprocessSection(BaseModel):
    lookahead_chapters: int = 1


class LLMSection(BaseModel):
    # Only one supported provider — DeepSeek V4 flash (cloud, OpenAI-
    # compatible API). The ``provider`` field is kept (not removed) so
    # config files stay parseable across versions and we have a slot to
    # add new providers later.
    provider: str = "deepseek"

    # API key. Take from env (DEEPSEEK_API_KEY) — never commit a real key
    # to config.yaml.
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com/v1"


class Qwen3TTSSection(BaseModel):
    """Config specific to the Qwen3-TTS MLX backend (used only when
    ``tts.provider=qwen3_tts``). M-series Mac required."""

    model_dir: str = "pretrained_models/Qwen3-TTS-0.6B"
    prompts_dir: str = "data/voices/prompts"


class TTSSection(BaseModel):
    provider: str = "stub"
    cache_dir: str = "data/tts_cache"
    voice_library: str = "data/voices/speakers.json"
    qwen3: Qwen3TTSSection = Field(default_factory=Qwen3TTSSection)

    # Server-side TTS cache size cap. A background janitor sweeps the
    # cache every ``cache_sweep_seconds`` and, when the directory
    # exceeds ``cache_max_mb``, evicts the oldest files (by mtime, an
    # approximate LRU) until it's back under the cap. Set
    # ``cache_sweep_seconds`` to 0 to disable the janitor entirely.
    cache_max_mb: int = 500
    cache_sweep_seconds: int = 60


class AuthSection(BaseModel):
    """Bearer-token-style authentication for the public API.

    The wire format is ``Authorization: Bearer <base64(user:pass)>``.
    Functionally Basic Auth in a Bearer envelope — picked because:
    1. The single-user homelab deployment doesn't need a login flow,
       token TTL, refresh, or per-user state.
    2. Clients can compute the token offline from the same (user, pass)
       the server has in config, no /login round trip.
    3. ``Bearer`` plays nicer than ``Basic`` with some HTTP middleware
       (no popup browser auth dialog on misroute).

    When ``enabled=False`` (default) the auth dependency is a no-op so
    existing dev setups keep working without config changes. The
    ``/health`` endpoint is always public so liveness probes don't need
    creds.
    """

    enabled: bool = False
    username: str = ""
    password: str = ""


class Settings(BaseModel):
    storage: StorageSection = Field(default_factory=StorageSection)
    preprocess: PreprocessSection = Field(default_factory=PreprocessSection)
    llm: LLMSection = Field(default_factory=LLMSection)
    tts: TTSSection = Field(default_factory=TTSSection)
    auth: AuthSection = Field(default_factory=AuthSection)

    @property
    def data_dir(self) -> Path:
        p = Path(self.storage.data_dir)
        return p if p.is_absolute() else SERVER_DIR / p

    @property
    def books_dir(self) -> Path:
        return self.data_dir / "books"

    @property
    def tts_cache_dir(self) -> Path:
        p = Path(self.tts.cache_dir)
        return p if p.is_absolute() else SERVER_DIR / p

    @property
    def voice_library_path(self) -> Path:
        p = Path(self.tts.voice_library)
        return p if p.is_absolute() else SERVER_DIR / p

    @property
    def qwen3_model_dir(self) -> Path:
        p = Path(self.tts.qwen3.model_dir)
        return p if p.is_absolute() else SERVER_DIR / p

    @property
    def qwen3_prompts_dir(self) -> Path:
        p = Path(self.tts.qwen3.prompts_dir)
        return p if p.is_absolute() else SERVER_DIR / p


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _apply_env_overrides(raw: dict) -> dict:
    """Flat env overrides: ``TINGSHU_<SECTION>_<KEY>``. Example:
    ``TINGSHU_LLM_DEEPSEEK_API_KEY=sk-...`` overrides ``llm.deepseek_api_key``.
    """
    for key, value in os.environ.items():
        if not key.startswith("TINGSHU_"):
            continue
        parts = key[len("TINGSHU_"):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field = parts
        raw.setdefault(section, {})
        # crude int coercion for known numeric fields
        if value.isdigit():
            raw[section][field] = int(value)
        else:
            raw[section][field] = value
    return raw


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    cfg_path = Path(os.environ.get("TINGSHU_CONFIG", SERVER_DIR / "config.yaml"))
    raw = _load_yaml(cfg_path)
    raw = _apply_env_overrides(raw)
    return Settings.model_validate(raw)
