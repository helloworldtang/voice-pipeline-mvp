#!/usr/bin/env bash
# 本机验收起服务（一条命令，含全部前置检查）。手册：docs/验收-本机真人-手册.md
# 用法：./scripts/dev_server.sh   （在配了 DASHSCOPE_API_KEY 的终端里跑，通常是你日常的 zsh）
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) key 检查：缺了 LLM 会静默退回 SimLlm，整轮验收作废（账单 llm 字段会如实标出，
#    但最好在起服务时就拦住）
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  echo "✗ 缺 DASHSCOPE_API_KEY——LLM 会退回替身，验收数字全部无效。" >&2
  echo "  key 在 ~/.zshrc 里：请在 zsh 终端跑本脚本，或先 source ~/.zshrc。" >&2
  exit 1
fi
echo "✓ DASHSCOPE_API_KEY 已配置"

# 2) token 提醒：本机验收不该设（设了页面 token 栏就得填）
if [ -n "${VOICE_WS_TOKEN:-}" ]; then
  echo "⚠ VOICE_WS_TOKEN 已设置——浏览器 token 栏需要填同一个值，否则 1008" >&2
fi

# 3) 验收口径默认值（已在环境里的不覆盖）
export VOICE_PIPELINE_TTS_ENGINE="${VOICE_PIPELINE_TTS_ENGINE:-dashscope}"
export VOICE_EVENTS_JSONL="${VOICE_EVENTS_JSONL:-$HOME/.voice-pilot/turns.jsonl}"
mkdir -p "$(dirname "$VOICE_EVENTS_JSONL")"

echo "✓ TTS=${VOICE_PIPELINE_TTS_ENGINE}（云流式，与线上同口径；本机 melo 会慢 3 倍）"
echo "✓ 账单落盘 $VOICE_EVENTS_JSONL"
echo
echo "下一步："
echo "  1. Chrome 打开 http://localhost:8000 —— mode=real-serial、gating=neural、token 留空"
echo "  2. 点「开麦」，按页面顶部「验收三式」说满 10 轮（5 基准 + 3 句中停顿 + 2 插话）"
echo "  3. 验完出记录：uv run python scripts/report_turns.py \"$VOICE_EVENTS_JSONL\" --out docs/验收-本机真人-\$(date +%Y%m%d).md"
echo
echo "Ctrl-C 停止服务（账单已实时落盘，随时可出记录）"
exec uv run uvicorn voice_pipeline.server.app:app --host 127.0.0.1 --port 8000
