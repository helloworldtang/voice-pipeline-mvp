#!/usr/bin/env bash
# 本机验收起服务（一条命令，含全部前置检查）。手册：docs/验收-本机真人-手册.md
# 用法：./scripts/dev_server.sh   （凭证放 ~/.voice-pilot/vendors.env 的，脚本自己加载）
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) 供应商凭证自动加载：讯飞等第三方 key 统一放 ~/.voice-pilot/vendors.env（600）。
#    2026-10-17 深夜教训：交接的命令没 source 它，用户连撞 6 次
#    「缺环境变量 XFYUN_TTS_APPID...」——起服务的脚本自己负责加载，不靠人记。
if [ -f "$HOME/.voice-pilot/vendors.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$HOME/.voice-pilot/vendors.env"
  set +a
  echo "✓ 已加载 ~/.voice-pilot/vendors.env"
fi

# 2) LLM 引擎前置检查：云端要 DASHSCOPE key（缺了静默退回 SimLlm，验收作废）；
#    本地 mlx 要 mlx-lm 装好（缺依赖时 provider 会响亮报错，但那已经浪费一轮验收）
ENGINE="${VOICE_LLM_ENGINE:-auto}"
if [ "$ENGINE" = "mlx" ]; then
  if ! uv run python -c "import mlx_lm" >/dev/null 2>&1; then
    echo "✗ VOICE_LLM_ENGINE=mlx 需要 mlx-lm：先 uv sync --extra local" >&2
    exit 1
  fi
  echo "✓ LLM=mlx（本地模型 ${VOICE_LLM_MLX_MODEL:-默认 Qwen2.5-1.5B}）"
else
  if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
    echo "✗ 缺 DASHSCOPE_API_KEY——LLM 会退回替身，验收数字全部无效。" >&2
    echo "  key 在 ~/.zshrc 里：请在 zsh 终端跑本脚本，或改用 VOICE_LLM_ENGINE=mlx 走本地模型。" >&2
    exit 1
  fi
  echo "✓ LLM=${ENGINE}（云端，DASHSCOPE_API_KEY 已配置）"
fi

# 3) TTS 引擎前置检查：讯飞的三个 key 缺一不可（缺了同样在第一轮才炸）
export VOICE_PIPELINE_TTS_ENGINE="${VOICE_PIPELINE_TTS_ENGINE:-dashscope}"
if [ "$VOICE_PIPELINE_TTS_ENGINE" = "xfyun" ]; then
  missing=""
  for v in XFYUN_TTS_APPID XFYUN_TTS_API_KEY XFYUN_TTS_API_SECRET; do
    [ -z "${!v:-}" ] && missing="$missing $v"
  done
  if [ -n "$missing" ]; then
    echo "✗ TTS=xfyun 但缺:${missing}——放 ~/.voice-pilot/vendors.env（脚本会自动加载）" >&2
    exit 1
  fi
  echo "✓ TTS=xfyun 凭证齐全"
fi

# 4) token 提醒：本机验收不该设（设了页面 token 栏就得填）
if [ -n "${VOICE_WS_TOKEN:-}" ]; then
  echo "⚠ VOICE_WS_TOKEN 已设置——浏览器 token 栏需要填同一个值，否则 1008" >&2
fi

# 5) 验收口径默认值（已在环境里的不覆盖）
export VOICE_EVENTS_JSONL="${VOICE_EVENTS_JSONL:-$HOME/.voice-pilot/turns.jsonl}"
mkdir -p "$(dirname "$VOICE_EVENTS_JSONL")"

echo "✓ TTS=${VOICE_PIPELINE_TTS_ENGINE}"
echo "✓ 账单落盘 $VOICE_EVENTS_JSONL"
echo
echo "下一步："
echo "  1. Chrome 打开 http://localhost:8000 —— mode 选 real-pipelined（验投机）或 real-serial（对照组）、gating=conservative、token 留空"
echo "     ⚠ 别选 neural：真人实测抢话 3/3，已退位为实验档（第四次定档）"
echo "  2. 点「开麦」，按页面顶部「验收三式」说满 10 轮（5 基准 + 3 句中停顿 + 2 插话）"
echo "  3. 验完出记录：uv run python scripts/report_turns.py \"$VOICE_EVENTS_JSONL\" --out docs/验收-本机真人-\$(python3 -c 'import datetime; print(datetime.datetime.now().strftime(\"%Y%m%d-%H%M%S%f\")[:-3])').md"
echo "     （文件名到毫秒：同一天多次验收互不覆盖——macOS date 无 %N，用 python 补）"
echo
echo "Ctrl-C 停止服务（账单已实时落盘，随时可出记录）"
exec uv run uvicorn voice_pipeline.server.app:app --host 127.0.0.1 --port 8000
