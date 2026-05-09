"""Test-suite-wide setup.

Sets a dummy ``DEEPSEEK_API_KEY`` so the eager LLM-factory check in
``main.lifespan`` can construct a client at startup. Tests that exercise
the LLM either:
- Override ``get_llm_client`` via ``app.dependency_overrides`` to inject
  a ``StubLLMClient`` (most common), or
- Mock at a lower layer — they never make real API calls.

The dummy key is therefore safe; it won't authenticate against DeepSeek
but no test ever hits the real network.

We also point ``TINGSHU_CONFIG`` at a non-existent path so tests never
pick up the developer's local ``config.yaml``. Without this, enabling
auth (or any other non-default setting) in config.yaml would
cross-contaminate every test that uses ``create_app()``. With it, tests
get pure ``Settings`` defaults (auth disabled, etc.) — individual tests
still override via ``app.dependency_overrides[get_settings]`` when they
need a specific shape (see ``tests/test_auth.py``).
"""
from __future__ import annotations

import os


os.environ.setdefault("DEEPSEEK_API_KEY", "test-dummy-key-not-real")
os.environ.setdefault("TINGSHU_CONFIG", "/dev/null/no-such-config-for-tests.yaml")
