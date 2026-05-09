# TingShu Server

FastAPI server for the TingShu audiobook app. See
[`../docs/technical-plan.md`](../docs/technical-plan.md) for the overall design.

## Requirements

- Python 3.11+
- macOS on Apple Silicon (M-series) for the MLX LLM and TTS backends
- Mac Mini M4 is the intended production host; other M-series chips work too

## Setup

```bash
cd server
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mlx]"   # mlx-audio for local TTS (Apple Silicon only)
```

Optional local overrides:

```bash
cp config.yaml.example config.yaml
# edit config.yaml
```

The DeepSeek API key comes from the `DEEPSEEK_API_KEY` env var (preferred)
or the `llm.deepseek_api_key` field in `config.yaml`:

```bash
export DEEPSEEK_API_KEY=sk-...
```

Without it, the server fails fast at startup with a clear error.

Environment variables override YAML using `TINGSHU_<SECTION>_<KEY>`
(e.g. `TINGSHU_TTS_PROVIDER=qwen3_tts`). The uvicorn host/port come from
the CLI flags below, not from config.

## Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000/docs>.

## Test

```bash
pytest
```

## Current scope (M1–M6 + M3c)

Implemented:

- FastAPI skeleton, config loading, file-system `BookRepository`.
- `GET /health`
- `POST /api/books/upload` — accepts `.txt` (LLM-driven chapter detection,
  §2.4.1) and `.epub` (direct TOC read via `toc.ncx` / `nav.xhtml`, no
  LLM). Response returns with `status=processing`; `analyze_book` marks
  the book `ready` immediately under Plan C — character discovery is
  deferred to chapter-meta generation.
- `GET /api/books` — list with status.
- `GET /api/books/{book_id}/download` — zip of `meta.json` + chapter texts.
- `GET /api/books/{book_id}/chapters/{ch}/meta` — triggers lazy LLM analysis
  on first request; caches to `chapters/<id>.json`; schedules lookahead
  generation of chapter N+1 as a background task.
- `POST /api/tts` — matches a speaker from the voice library via
  gender/age/personality, hits the on-disk cache, or delegates to the
  configured TTS backend (`stub` default; `qwen3_tts` for production).
  Runs synthesis in a thread pool so long calls don't block the event loop.
  Response headers `X-Speaker-*` report the selected voice.
- **LLM backend**: DeepSeek V4 flash via `app/services/llm_deepseek.py`
  (OpenAI-compatible HTTP, `thinking: disabled`). Tests inject a
  `StubLLMClient` directly, so they don't hit the API.
- **Chapter metadata — unified analysis** (§2.3): one
  `LLMClient.analyze_chapter` call produces two NDJSON sections —
  sentences (`{"t":"...","s":"speaker","o":"tone"}`) and character
  profile updates (`{"c":"name","g":"...","a":"...","p":[...],"i":"..."}`).
  `reconcile_chapter_characters` maps speaker strings to stable
  `character_id`s and merges profile updates into the server-only
  `characters.json` (cumulative roster). `locate_sentences` maps each
  sentence back to `(line, col)` positions via forward substring search.
- **Per-chapter character snapshot**: each `chapters/N.json` embeds the
  full profile of every character that speaks in that chapter, frozen
  at the moment of analysis. The App reads voices from this snapshot —
  no global character table to maintain client-side.

### Pending milestones

All server-side milestones (M1-M7) are complete. Remaining work is on the
iOS client (see `docs/technical-plan.md` Phase 3+).

## Voice library

The matching algorithm consults `data/voices/speakers.json` (path configurable
via `tts.voice_library`). In production this contains `zs:vd_*` entries
generated locally via Qwen3-TTS VoiceDesign (see "Seed the voice library"
below).

Matching ranks every speaker by `(gender_distance, age_distance,
-personality_overlap, speaker_id)` and returns the best scorer — so it
degrades gracefully when the library is partial. Server responses carry
`X-Speaker-Id`/`X-Speaker-Gender`/`X-Speaker-Age` so clients and logs can
see which voice was chosen.

## TTS cache

Synthesized audio is cached on disk at `data/tts_cache/<sha1>.wav`, keyed
by `text || speaker_id || tone || speed`. This is a compute-reuse cache,
not user-facing storage — different from the client-side cache described
in docs/technical-plan.md §3.6.1, which has its own lifecycle.

## Qwen3-TTS setup (production TTS backend)

Switch `tts.provider=qwen3_tts` in `config.yaml`. Before starting the
server you need (1) `mlx-audio` in your venv, (2) the model weights.

### 1. Install MLX audio dependency

```bash
cd server
source .venv/bin/activate
pip install -e ".[mlx]"   # brings in mlx-lm + mlx-audio
```

### 2. Download the model weights

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit',
    local_dir='pretrained_models/Qwen3-TTS-0.6B',
    max_workers=4,
)"
```

~1.6 GB on disk. The 4bit DWQ variant gives RTF ~0.6 on a Mac Mini M4.

### 3. Seed the voice library

The VoiceDesign variant of Qwen3-TTS synthesises a voice from a
natural-language description. Use it to generate the 64-voice prompt
library locally — fully on-device, no external account or per-character
billing. The runtime keeps using Base + zero-shot cloning of these
generated WAVs (`tts.provider=qwen3_tts` is unchanged).

```bash
# 1. Download the VoiceDesign model (~1.7 GB, one-off).
#    Note: VoiceDesign is only published for the 1.7B model — there is
#    no 0.6B-VoiceDesign and no 4bit MLX build as of 2026-04. The 8bit
#    1.7B build below is the smallest available.
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit',
    local_dir='pretrained_models/Qwen3-TTS-VoiceDesign',
    max_workers=4,
)"

# 2. Generate all prompts + register them in speakers.json:
.venv/bin/python -m scripts.generate_voicedesign_voices \
    --model-dir pretrained_models/Qwen3-TTS-VoiceDesign

# 3. Re-roll a voice you don't like (VoiceDesign is non-deterministic):
.venv/bin/python -m scripts.generate_voicedesign_voices \
    --model-dir pretrained_models/Qwen3-TTS-VoiceDesign \
    --ids ancient_male_adult,fierce_female_elder --regenerate
```

Notes:
- Output filenames use the `vd_<tag>` prefix and register as
  `zs:vd_<tag>` in speakers.json.
- The VoiceDesign model is **only** used at prompt-generation time. The
  runtime serving `/api/tts` keeps using the Base + zero-shot clone
  pipeline, so per-request synthesis stays deterministic.

**One-off** (drop a movie/podcast clip or real-person recording):

1. Put `<prompt_id>.wav` + `<prompt_id>.txt` in `data/voices/prompts/`.
2. Add an entry in `data/voices/speakers.json` with `speaker_id="zs:<prompt_id>"`
   plus hand-picked `gender` / `age` / `personality` tags.

See [`data/voices/prompts/README.md`](data/voices/prompts/README.md) for
guidelines on reference audio quality.

### 4. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Lifespan startup warms the model (logs `loading Qwen3-TTS from ...`)
so the first `/api/tts` call doesn't incur the weight-load latency.

## Data layout

```
data/books/<book_id>/
├── meta.json
└── chapters/
    ├── 0001.txt
    ├── 0001.json   (lazy, generated on first chapter-meta request)
    └── ...
```
