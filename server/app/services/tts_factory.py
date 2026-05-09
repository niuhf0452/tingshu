"""Pick a TTS backend based on config."""
from __future__ import annotations

from ..config import TTSSection, get_settings
from .tts import TTSClient
from .tts_stub import StubTTSClient


def create_tts_client(cfg: TTSSection) -> TTSClient:
    provider = cfg.provider.lower()
    if provider == "stub":
        return StubTTSClient()
    if provider == "qwen3_tts":
        from .tts_qwen3 import Qwen3TTSClient  # lazy — pulls mlx
        settings = get_settings()
        return Qwen3TTSClient(
            model_dir=str(settings.qwen3_model_dir),
            prompts_dir=str(settings.qwen3_prompts_dir),
        )
    raise ValueError(f"unknown tts provider: {cfg.provider!r}")
