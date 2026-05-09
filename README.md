# TingShu (听书)

A self-hosted Chinese audiobook system with multi-character TTS narration.
Drop a TXT or EPUB into the iOS app; the server splits it into chapters,
asks an LLM to attribute each sentence to a character, then synthesises
each line with a voice picked to match that character's gender, age, and
personality.

Two parts in one repo:

- **`server/`** — FastAPI service (Python). Runs on a Mac Mini M4 in the
  homelab. Handles book ingest, LLM-driven chapter analysis, and TTS via
  Qwen3-TTS MLX. Exposes a small HTTP API + Bearer auth for the iOS
  client.
- **`ios/`** — SwiftUI iOS app (iOS 17+). Bookshelf + chapter reader +
  audio playback. Talks to the server over LAN, caches both book text
  and TTS audio locally for offline reading.
- **`docs/technical-plan.md`** — full design spec; this README is the
  operations manual.

---

# Server

## Quick start

```bash
./setup.sh
```

That's it for a fresh checkout on an Apple Silicon Mac. The script is
idempotent — re-running it is safe. It does:

1. Verifies macOS arm64 + `python3.11`.
2. Creates `server/.venv` and installs Python deps (`pip install -e ".[dev,mlx]"`).
3. Bootstraps `server/config.yaml` from `config.yaml.example` if missing,
   and warns about which fields to fill (auth, DeepSeek key, TTS provider).
4. Checks for a DeepSeek API key (env var `DEEPSEEK_API_KEY` or
   `llm.deepseek_api_key` in config).
5. Downloads the Qwen3-TTS Base 0.6B model (~1.6 GB).
6. Downloads the Qwen3-TTS VoiceDesign 1.7B model (~1.7 GB).
7. Generates the 64-voice prompt library into `server/data/voices/`.
8. Prints the command to start the server.

First run takes ~10–15 min on a decent connection (model downloads
dominate). Re-runs are seconds.

Flags:

| Flag | Effect |
|------|--------|
| (none) | Full setup |
| `--skip-models` | Stop after step 4 (venv + deps + config). Useful when iterating on Python code without needing TTS. |
| `--skip-voicedesign` | Download the Base model but skip VoiceDesign + voice library. Useful if you've brought your own `speakers.json`. |
| `--help` | Print usage |

Env overrides:

- `HF_ENDPOINT` — HuggingFace mirror for `snapshot_download`. Defaults
  to `https://hf-mirror.com` (works globally; CN-friendly). Override
  with your own mirror or unset to use the default `huggingface.co`.
- `DEEPSEEK_API_KEY` — if set, the script confirms it; otherwise warns.

## Start the server

```bash
cd server
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs> for the OpenAPI explorer.

The host *must* be `0.0.0.0` (not `127.0.0.1`) so the iOS device can
reach the server over LAN. The lifespan also pins the host awake via
`wakepy` so it doesn't suspend mid-request.

If you see `LLM provider init failed: ...` at startup, the DeepSeek
API key isn't set. Either `export DEEPSEEK_API_KEY=sk-...` or fill
`llm.deepseek_api_key` in `server/config.yaml`.

## Test

```bash
cd server
source .venv/bin/activate
pytest
```

Slow / model-loading tests are marked `@pytest.mark.slow` and skipped
by default; run them with `pytest -m slow`.

## Configuration

`server/config.yaml.example` is the template. The setup script copies
it to `config.yaml` (gitignored) on first run.

Key sections:

```yaml
llm:
  provider: deepseek                    # only supported provider
  deepseek_model: deepseek-v4-flash
  deepseek_api_key: ""                  # or use DEEPSEEK_API_KEY env var

tts:
  provider: stub                        # ← change to qwen3_tts for production
  cache_dir: data/tts_cache
  voice_library: data/voices/speakers.json
  qwen3:
    model_dir: pretrained_models/Qwen3-TTS-0.6B
    prompts_dir: data/voices/prompts

auth:
  enabled: false                        # ← turn on before exposing beyond LAN
  username: ""
  password: ""
```

Env vars override YAML using `TINGSHU_<SECTION>_<KEY>`, e.g.
`TINGSHU_TTS_PROVIDER=qwen3_tts`, `TINGSHU_AUTH_ENABLED=true`.

### Bearer auth

The iOS app sends `Authorization: Bearer <base64(username:password)>`
on every request. Server-side check is constant-time. With
`auth.enabled: false` (default), the dependency is a no-op so dev
runs work without setup. `/health` is always public regardless —
liveness probes don't need creds.

## Voice library

Production has 64 `zs:vd_*` entries in `server/data/voices/speakers.json`,
generated locally via Qwen3-TTS VoiceDesign (handled by `setup.sh`).

The matcher ranks every speaker by `(gender_distance, age_distance,
-personality_overlap, speaker_id)` and returns the best scorer — so it
degrades gracefully when the library is partial. Server responses carry
`X-Speaker-Id` / `X-Speaker-Gender` / `X-Speaker-Age` so clients and
logs can see which voice was chosen.

**Re-roll one or more voices** (VoiceDesign is non-deterministic, so a
voice you don't like can be regenerated):

```bash
cd server
source .venv/bin/activate
python -m scripts.generate_voicedesign_voices \
    --model-dir pretrained_models/Qwen3-TTS-VoiceDesign \
    --ids ancient_male_adult,fierce_female_elder --regenerate
```

**Add a one-off voice** (movie/podcast clip or real-person recording):

1. Put `<prompt_id>.wav` + `<prompt_id>.txt` (the transcript) in
   `server/data/voices/prompts/`.
2. Add an entry in `server/data/voices/speakers.json` with
   `speaker_id="zs:<prompt_id>"` plus hand-picked `gender` / `age` /
   `personality` tags.

The runtime serves `/api/tts` via Qwen3-TTS Base + zero-shot cloning
of these WAVs — VoiceDesign is *only* used at prompt-generation time,
so per-request synthesis stays deterministic.

## TTS cache

Server-side cache at `server/data/tts_cache/<sha1>.m4a`, keyed by
`text || speaker_id`. Compute-reuse only — different from the
client-side cache described in `docs/technical-plan.md` §3.6.1.

## Data layout

```
server/data/
├── books/<book_id>/
│   ├── meta.json
│   └── chapters/
│       ├── 0001.txt
│       ├── 0001.json   (lazy, generated on first chapter-meta request)
│       └── ...
├── tts_cache/
│   └── <sha1>.m4a
└── voices/
    ├── speakers.json
    └── prompts/
        ├── vd_<tag>.wav
        └── vd_<tag>.txt
```

## Manual setup (fallback if `setup.sh` doesn't fit)

```bash
# 1. venv + deps
cd server
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mlx]"

# 2. config
cp config.yaml.example config.yaml
# edit: tts.provider=qwen3_tts, llm.deepseek_api_key, auth.*

# 3. base model (~1.6 GB)
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit',
    local_dir='pretrained_models/Qwen3-TTS-0.6B',
    max_workers=4,
)"

# 4. voicedesign model (~1.7 GB) — only needed if you want to (re)generate prompts
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit',
    local_dir='pretrained_models/Qwen3-TTS-VoiceDesign',
    max_workers=4,
)"

# 5. generate the voice library (~5 min on M4)
python -m scripts.generate_voicedesign_voices \
    --model-dir pretrained_models/Qwen3-TTS-VoiceDesign

# 6. start
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

# iOS App

## Requirements

- Xcode 17+ (matches `IPHONEOS_DEPLOYMENT_TARGET = 17.0`)
- For real-device runs: a free Apple ID is enough (Personal Team signing)

## Build

```bash
cd ios
xcodebuild -project TingShu.xcodeproj -scheme TingShu \
    -destination 'platform=iOS Simulator,name=iPhone 17' \
    -configuration Debug build
```

Or open `ios/TingShu.xcodeproj` in Xcode and ⌘R.

## First-run setup on device

1. Pick the device in Xcode's run-destination dropdown and run once;
   trust the developer profile on the iPhone if prompted.
2. In the app's **设置** (Settings) page:
   - **服务端地址** — `http://<mac-lan-ip>:8000` (e.g. `http://192.168.0.148:8000`).
     `localhost` won't work; that's the device itself, not the Mac.
   - **用户名 / 密码** — must match `auth.username` / `auth.password`
     on the server (or leave blank if `auth.enabled=false`).
3. iPhone and Mac must be on the same Wi-Fi (no client-isolation segment).
4. macOS firewall: allow `python` / `uvicorn` to accept inbound, or
   disable temporarily for the test.
5. iOS will prompt **"Allow TingShu to find devices on your local
   network"** on first connection — must allow.

Verify reachability with `curl http://<mac-lan-ip>:8000/health` from
another machine on the LAN before debugging the app.

## Layout

```
ios/
├── TingShu.xcodeproj/
├── Package.swift              (Swift Package manifest, ZIPFoundation dep)
└── TingShu/
    └── Sources/
        ├── App/               (entry point, Info.plist, assets)
        ├── Models/            (data types matching server JSON)
        ├── Services/          (APIClient, BookStore, PlaybackService, …)
        └── Views/             (BookshelfView, PlayerView, SettingsView)
```

---

# Design docs

- [`docs/technical-plan.md`](docs/technical-plan.md) — the full design
  spec, end-to-end. Topics include the chapter-meta wire format, voice
  matcher, sliding-window TTS prefetch, and per-component lifecycle.
  This README is operations / quick-start; the doc is the why.
- [`docs/tts-model-evaluation.md`](docs/tts-model-evaluation.md) — TTS
  model selection record. Which models were evaluated (CosyVoice,
  Voxtral, VibeVoice, VoxCPM2, Qwen3-TTS variants), why Qwen3-TTS was
  chosen, plus sediment knowledge from the bumps we hit (ref-text
  bleed bug, EPUB TOC misclassification, etc.).
