"""Pick an LLM backend based on config.

Only one production provider: ``deepseek`` (DeepSeek V4 flash, no
thinking). Tests inject a ``StubLLMClient`` directly via FastAPI
``dependency_overrides``, not through this factory.
"""
from __future__ import annotations

import os

from ..config import LLMSection
from .llm import LLMClient


def create_llm_client(cfg: LLMSection) -> LLMClient:
    provider = cfg.provider.lower()
    if provider != "deepseek":
        raise ValueError(
            f"unknown llm provider: {cfg.provider!r} (only 'deepseek' is supported)"
        )
    api_key = cfg.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError(
            "DeepSeek provider needs an API key. Set the DEEPSEEK_API_KEY "
            "env var or `llm.deepseek_api_key` in config.yaml."
        )
    from .llm_deepseek import DeepSeekLLMClient  # noqa: WPS433
    return DeepSeekLLMClient(
        api_key=api_key,
        model=cfg.deepseek_model,
        base_url=cfg.deepseek_base_url,
    )
