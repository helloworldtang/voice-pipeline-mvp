#!/usr/bin/env bash
# voice-pilot 发版：rsync 代码 → uv sync → 重启服务。
# 用法：./deploy/release.sh
# 前提：本机免密 ssh ubuntu@ECS 可达；首次部署见 deploy/SETUP.md。
set -euo pipefail

HOST="${VOICE_PILOT_HOST:-ubuntu@YOUR_SERVER_IP}"
# rsync 目标里 ~ 不展开，用相对 home 的路径
REMOTE_DIR=voice-pipeline-mvp

echo "==> rsync 代码（不含模型/.venv/bench）"
rsync -az --delete \
  --exclude .venv --exclude .git --exclude models --exclude bench --exclude data \
  --exclude up-wavs --exclude .ruff_cache --exclude .pytest_cache \
  ./ "$HOST:$REMOTE_DIR/"

echo "==> 远端 uv sync"
ssh "$HOST" "cd ~/voice-pipeline-mvp && ~/.local/bin/uv sync 2>&1 | tail -1"

echo "==> 修补 sherpa-onnx wheel 缺的 libonnxruntime（uv sync 可能重装包，每次都做）"
ssh "$HOST" "cd ~/voice-pipeline-mvp && SITE=\$PWD/.venv/lib/python3.12/site-packages && \
  if [ ! -f \$SITE/sherpa_onnx/lib/libonnxruntime.so.1.28.2 ]; then \
    echo '缺 libonnxruntime.so.1.28.2：从 /tmp 补（首次部署见 SETUP.md）' >&2; exit 1; \
  fi && ln -sf \$SITE/sherpa_onnx/lib/libonnxruntime.so.1.28.2 \$SITE/sherpa_onnx/lib/libonnxruntime.so"

echo "==> 重启服务"
ssh "$HOST" "sudo systemctl restart voice-pipeline && sleep 3 && systemctl is-active voice-pipeline && curl -s http://127.0.0.1:8000/healthz"
echo "==> 完成"
