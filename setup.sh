#!/usr/bin/env bash
# Bootstrap the TingShu server end-to-end: venv → deps → config → models →
# voice library. Idempotent — safe to re-run; each step skips itself if
# the work is already done. The detailed manual flow is in README.md;
# this script just stitches it together so a fresh checkout is one command.
#
# Usage:
#   ./setup.sh                    # full setup
#   ./setup.sh --skip-models      # only venv + deps + config (no model downloads)
#   ./setup.sh --skip-voicedesign # download Base model only; skip VoiceDesign + library generation
#   ./setup.sh --help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/server"

# ----- pretty output -------------------------------------------------------

if [[ -t 1 ]]; then
    GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
    BOLD='\033[1m';     NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; BOLD=''; NC=''
fi
step() { printf "${BOLD}${GREEN}==>${NC} ${BOLD}%s${NC}\n" "$*"; }
note() { printf "    %s\n" "$*"; }
warn() { printf "${YELLOW}!! %s${NC}\n" "$*" >&2; }
err()  { printf "${RED}xx %s${NC}\n" "$*" >&2; exit 1; }

# ----- args ----------------------------------------------------------------

SKIP_MODELS=0
SKIP_VOICEDESIGN=0
for arg in "$@"; do
    case "$arg" in
        --skip-models)
            SKIP_MODELS=1
            SKIP_VOICEDESIGN=1
            ;;
        --skip-voicedesign)
            SKIP_VOICEDESIGN=1
            ;;
        --help|-h)
            cat <<'EOF'
Bootstrap the TingShu server end-to-end: venv → deps → config → models →
voice library. Idempotent — safe to re-run; each step skips itself if
the work is already done.

Usage:
  ./setup.sh                    full setup (default)
  ./setup.sh --skip-models      only venv + deps + config (no model downloads)
  ./setup.sh --skip-voicedesign download Base only; skip VoiceDesign + library

Env overrides:
  HF_ENDPOINT       HuggingFace mirror to use for snapshot_download.
                    Defaults to https://hf-mirror.com (works globally).
  DEEPSEEK_API_KEY  If set, the script confirms it; otherwise warns
                    you to set it before starting the server.
EOF
            exit 0
            ;;
        *)
            err "unknown arg: $arg (try --help)"
            ;;
    esac
done

# ----- prereqs -------------------------------------------------------------

step "Checking environment"
[[ "$(uname -s)" == "Darwin" ]] || err "macOS required (got $(uname -s)); the MLX TTS backend doesn't run on Linux."
[[ "$(uname -m)" == "arm64"  ]] || err "Apple Silicon required (got $(uname -m)); MLX is M-series only."
command -v python3.11 >/dev/null || err "python3.11 not found. Install via 'brew install python@3.11'."
note "macOS arm64 + python3.11 OK"

# ----- venv ----------------------------------------------------------------

step "Setting up venv at server/.venv"
if [[ ! -d .venv ]]; then
    python3.11 -m venv .venv
    note "venv created"
else
    note "venv already exists, skipping create"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip

# ----- deps ----------------------------------------------------------------

step "Installing Python deps (pip install -e \".[dev,mlx]\")"
pip install -e ".[dev,mlx]"

# ----- config --------------------------------------------------------------
#
# Interactive on first run: prompt for DeepSeek key + auth credentials
# and bake them into config.yaml. Re-runs leave the file alone — once
# config.yaml exists, edits go through the user's text editor, not us.
# Falls back to a silent copy-from-example when stdin isn't a tty
# (CI / piped install / `< /dev/null`).

step "Bootstrapping config.yaml"
if [[ -f config.yaml ]]; then
    note "config.yaml already exists, leaving alone"
elif [[ ! -t 0 ]]; then
    cp config.yaml.example config.yaml
    note "Created config.yaml from example (non-interactive mode)."
    warn "Edit server/config.yaml to set llm.deepseek_api_key, auth.*, tts.provider"
else
    echo
    note "First-run config setup. Press Enter at any prompt to skip that field"
    note "(skipped fields stay empty in config.yaml; you can edit later)."
    echo

    # DeepSeek API key — skip the prompt if the env var is already set,
    # since that path takes priority at runtime anyway.
    if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
        note "DEEPSEEK_API_KEY already set in env (won't ask again)"
        SETUP_KEY=""
    else
        printf "  DeepSeek API key (sk-...): "
        read -r SETUP_KEY
    fi

    # Auth: enter username to enable; empty username disables auth.
    # Password is read with -s so it doesn't echo to the terminal.
    printf "  Auth username (Enter to leave auth disabled): "
    read -r SETUP_USER
    SETUP_PASS=""
    if [[ -n "$SETUP_USER" ]]; then
        printf "  Auth password (input hidden): "
        read -rs SETUP_PASS
        echo
        if [[ -z "$SETUP_PASS" ]]; then
            warn "  Empty password; leaving auth disabled."
            SETUP_USER=""
        fi
    fi

    # Default to qwen3_tts when models are about to be downloaded;
    # leave as 'stub' when --skip-models so the server boots without
    # weights present.
    if [[ $SKIP_MODELS -eq 1 ]]; then
        SETUP_PROVIDER="stub"
    else
        SETUP_PROVIDER="qwen3_tts"
    fi

    # Pass values via env vars (NOT inline interpolation) so quotes /
    # backslashes / `$` in passwords don't break Python parsing.
    # `safe_dump` handles the YAML escaping.
    SETUP_KEY="$SETUP_KEY" \
    SETUP_USER="$SETUP_USER" \
    SETUP_PASS="$SETUP_PASS" \
    SETUP_PROVIDER="$SETUP_PROVIDER" \
    python <<'PYEOF'
import os
import yaml

with open('config.yaml.example') as f:
    data = yaml.safe_load(f)

api_key = os.environ.get('SETUP_KEY', '').strip()
username = os.environ.get('SETUP_USER', '').strip()
password = os.environ.get('SETUP_PASS', '')  # don't strip — could be intentional
provider = os.environ.get('SETUP_PROVIDER', 'stub')

if api_key:
    data['llm']['deepseek_api_key'] = api_key
if username and password:
    data['auth']['enabled'] = True
    data['auth']['username'] = username
    data['auth']['password'] = password
data['tts']['provider'] = provider

with open('config.yaml', 'w') as f:
    f.write("# Generated by setup.sh — see config.yaml.example for field docs.\n\n")
    yaml.safe_dump(
        data, f, allow_unicode=True, sort_keys=False, default_flow_style=False,
    )
PYEOF

    # Summary uses the captured vars directly — file-grepping `provider:`
    # is ambiguous because both `llm:` and `tts:` have one.
    auth_state=$([ -n "$SETUP_USER" ] && [ -n "$SETUP_PASS" ] && echo "auth on (user=$SETUP_USER)" || echo "auth off")
    key_state=$([ -n "$SETUP_KEY" ] && echo "DeepSeek key set" || echo "no DeepSeek key")
    note "Wrote config.yaml ($auth_state, $key_state, tts.provider=$SETUP_PROVIDER)"

    # Drop the captured creds from the shell env now that they're on disk.
    unset SETUP_KEY SETUP_USER SETUP_PASS SETUP_PROVIDER
fi

# ----- API key sanity ------------------------------------------------------

step "Checking DeepSeek API key"
if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
    note "DEEPSEEK_API_KEY is set in env"
elif grep -qE '^[[:space:]]*deepseek_api_key:[[:space:]]*\S' config.yaml 2>/dev/null \
   && ! grep -qE '^[[:space:]]*deepseek_api_key:[[:space:]]*""[[:space:]]*$' config.yaml 2>/dev/null; then
    note "deepseek_api_key set in config.yaml"
else
    warn "DEEPSEEK_API_KEY not set anywhere."
    warn "Run 'export DEEPSEEK_API_KEY=sk-...' or edit llm.deepseek_api_key in server/config.yaml."
    warn "Server will fail at startup until you do this."
fi

# ----- HF mirror -----------------------------------------------------------
# Default to the well-known China-friendly mirror; override by exporting
# `HF_ENDPOINT` before invoking this script. The mirror works globally,
# so leaving the default is harmless outside CN.

: "${HF_ENDPOINT:=https://hf-mirror.com}"
export HF_ENDPOINT

if [[ $SKIP_MODELS -eq 1 ]]; then
    warn "Skipping model downloads + voice library generation (--skip-models)"
else
    step "Downloading Qwen3-TTS Base 0.6B (~1.6 GB) — HF_ENDPOINT=$HF_ENDPOINT"
    if [[ -f pretrained_models/Qwen3-TTS-0.6B/config.json ]]; then
        note "already present, skipping"
    else
        python <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit',
    local_dir='pretrained_models/Qwen3-TTS-0.6B',
    max_workers=4,
)
PY
    fi

    if [[ $SKIP_VOICEDESIGN -eq 1 ]]; then
        warn "Skipping VoiceDesign download + voice library generation (--skip-voicedesign)"
    else
        step "Downloading Qwen3-TTS VoiceDesign 1.7B (~1.7 GB)"
        if [[ -f pretrained_models/Qwen3-TTS-VoiceDesign/config.json ]]; then
            note "already present, skipping"
        else
            python <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit',
    local_dir='pretrained_models/Qwen3-TTS-VoiceDesign',
    max_workers=4,
)
PY
        fi

        step "Generating voice library (64 voices, several minutes on M4)"
        # Skip if speakers.json already lists vd_* entries — rerun by
        # deleting the file or passing --regenerate to the script
        # directly with the tag list you want re-rolled.
        if python <<'PY' >/dev/null 2>&1
import json, pathlib, sys
p = pathlib.Path('data/voices/speakers.json')
if not p.exists():
    sys.exit(1)
data = json.loads(p.read_text())
sys.exit(0 if any(s.get('speaker_id', '').startswith('zs:vd_') for s in data) else 1)
PY
        then
            note "speakers.json already has vd_* entries, skipping"
            note "(to re-roll specific voices: python -m scripts.generate_voicedesign_voices \\"
            note "    --model-dir pretrained_models/Qwen3-TTS-VoiceDesign \\"
            note "    --ids tag1,tag2 --regenerate)"
        else
            python -m scripts.generate_voicedesign_voices \
                --model-dir pretrained_models/Qwen3-TTS-VoiceDesign
        fi
    fi
fi

# ----- finale --------------------------------------------------------------

echo
printf "${GREEN}===========================================${NC}\n"
printf "${GREEN}${BOLD}Setup complete.${NC}\n"
printf "${GREEN}===========================================${NC}\n"
echo
echo "Start the server with:"
echo
printf "  ${BOLD}cd server${NC}\n"
printf "  ${BOLD}source .venv/bin/activate${NC}\n"
printf "  ${BOLD}uvicorn app.main:app --host 0.0.0.0 --port 8000${NC}\n"
echo
echo "Then open http://localhost:8000/docs"
echo
echo "iOS app setup is in README.md → 'iOS App' section."
echo
