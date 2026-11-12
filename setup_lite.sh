#!/usr/bin/env bash
# Lightweight setup for llm-state-recovery on an Apple Silicon Mac.
# Run from inside the project folder. Safe to re-run. Downloads nothing larger than ~3 GB.
set -euo pipefail

echo "== Checking platform"
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This script is for Apple Silicon macOS (MLX needs it). Stopping."; exit 1
fi

echo "== Folder layout"
mkdir -p harness probes prompts third_party configs runs analysis
touch runs/.gitkeep
[[ -f .gitignore ]] || cat > .gitignore <<'EOF'
.venv/
runs/*
!runs/.gitkeep
__pycache__/
*.npy
*.npz
.DS_Store
EOF

echo "== Git"
[[ -d .git ]] || git init -q -b main || echo "   (git init failed; run 'git init -b main' yourself)"

echo "== Python environment"
if command -v uv >/dev/null 2>&1; then
  uv venv .venv --python 3.12
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install mlx-lm mlx-lens numpy pandas pyyaml huggingface_hub
else
  echo "uv not found; using python3 -m venv (install uv later with: brew install uv)"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install mlx-lm mlx-lens numpy pandas pyyaml huggingface_hub
fi
if command -v uv >/dev/null 2>&1; then uv pip freeze > requirements.lock.txt; else pip freeze > requirements.lock.txt; fi

echo "== Model config"
[[ -f configs/models.yaml ]] || cat > configs/models.yaml <<'EOF'
# Pilot models for Gate 0. Main grid adds gemma-3-12b, Qwen3-4B, Qwen3-14B, Llama-3.1-8B.
pilot:
  - id: mlx-community/gemma-3-4b-it-4bit
  - id: mlx-community/Qwen3-8B-4bit
decoding:
  temperature: 0.7
  max_tokens: 120
  qwen3_thinking: false
EOF

echo "== Reference prompt repos (read-only references; check licenses before copying text)"
[[ -d third_party/LLMAffect ]] || git clone --depth 1 https://github.com/WekenborgLab/LLMAffect third_party/LLMAffect
[[ -d third_party/gpt-trauma-induction ]] || git clone --depth 1 https://github.com/akjagadish/gpt-trauma-induction third_party/gpt-trauma-induction
{
  echo "# Prompt sources"
  echo
  echo "| Source | URL | Commit | License |"
  echo "| --- | --- | --- | --- |"
  for d in LLMAffect gpt-trauma-induction; do
    url=$(git -C "third_party/$d" remote get-url origin)
    sha=$(git -C "third_party/$d" rev-parse --short HEAD)
    licfile=$(ls "third_party/$d" | grep -i '^licen' | head -1 || true)
    if [[ -n "$licfile" ]]; then lic=$(head -1 "third_party/$d/$licfile"); else lic="NOT FOUND - check before copying"; fi
    echo "| $d | $url | $sha | $lic |"
  done
} > prompts/SOURCES.md
grep -qx "third_party/" .gitignore || echo "third_party/" >> .gitignore

echo "== Done. Next:"
echo "   source .venv/bin/activate && python smoke_test.py"
