#!/usr/bin/env bash
# 拉取 M2 real 模式需要的模型（int8 优先，不进仓）。
#
# 源：HF（默认走 hf-mirror，本机实测 GitHub release 直连不通）。
# 用法：./scripts/download_models.sh   [可 HF_ENDPOINT=https://huggingface.co 覆盖]
set -euo pipefail

HF="${HF_ENDPOINT:-https://hf-mirror.com}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)/models"

ASR_REPO="csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
TTS_REPO="csukuangfj/kokoro-int8-multi-lang-v1_1"
VAD_REPO="onnx-community/silero-vad"

fetch() {  # fetch <repo> <remote-file> <dest-relpath>
  local repo="$1" remote="$2" dest="$ROOT/$3"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then echo "skip  $3 (已存在)"; return; fi
  echo "get   $3"
  curl -L --fail --retry 3 -o "$dest.part" "$HF/$repo/resolve/main/$remote"
  mv "$dest.part" "$dest"
}

# ASR：流式 paraformer 中英双语 int8（encoder 158MB + decoder 68MB = 226MB）
fetch "$ASR_REPO" encoder.int8.onnx asr/encoder.int8.onnx
fetch "$ASR_REPO" decoder.int8.onnx asr/decoder.int8.onnx
fetch "$ASR_REPO" tokens.txt       asr/tokens.txt

# TTS：kokoro int8 多语言（147MB，24kHz，含中文音色）
fetch "$TTS_REPO" model.int8.onnx  tts/model.int8.onnx
fetch "$TTS_REPO" tokens.txt       tts/tokens.txt
fetch "$TTS_REPO" voices.bin       tts/voices.bin
fetch "$TTS_REPO" lexicon-zh.txt   tts/lexicon-zh.txt

# VAD：silero（判停链路的 L1；L2 韵律判停模型见技术方案 §12）
fetch "$VAD_REPO" onnx/model.onnx  vad/silero_vad.onnx

echo "done: $ROOT"
du -sh "$ROOT"/* 2>/dev/null || true
