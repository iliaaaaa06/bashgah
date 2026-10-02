#!/usr/bin/env bash
# Starts llama.cpp's OpenAI-compatible server. Settings come from .env (LLM_* / LLAMA_*).
# Mac (Metal) or Linux/Windows-WSL (CUDA) — llama-server picks the GPU backend it was built with.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && source .env; set +a

MODEL_HF="${LLAMA_MODEL_HF:-Qwen/Qwen2.5-7B-Instruct-GGUF:Q5_K_M}"
CTX="${LLAMA_CTX_SIZE:-32768}"          # total context, split across parallel slots
PARALLEL="${LLAMA_PARALLEL:-4}"         # concurrent requests (each gets CTX/PARALLEL tokens)
PORT="${LLAMA_PORT:-8080}"
GPU_LAYERS="${LLAMA_GPU_LAYERS:-99}"

MODEL_ARGS=(-hf "$MODEL_HF")
[ -n "${LLAMA_MODEL_PATH:-}" ] && MODEL_ARGS=(-m "$LLAMA_MODEL_PATH")

exec llama-server "${MODEL_ARGS[@]}" \
  --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" -np "$PARALLEL" -ngl "$GPU_LAYERS" \
  --flash-attn auto --jinja \
  --cache-type-k q8_0 --cache-type-v q8_0
