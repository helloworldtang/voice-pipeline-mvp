# voice-pipeline-mvp

准生产级实时语音对话链路（ASR → LLM → TTS）参考实现。

配套文章：《给AI产品接上嘴——ASR+TTS接入实战，延迟从2秒压到300毫秒》。本仓的任务是让文章里
的每个机制、每个数字都能被一条命令复现，并把评论区「300ms 是否可能」的争论做成可复现实验。

- 立项背景、评论区存档、验收基线：`背景与基线.md`
- 技术选型、架构、协议、里程碑：`技术方案.md`

## 本机准备（macOS）

```bash
uv python pin 3.12          # sherpa-onnx 的 wheel 在 3.13+ 上有坑
./scripts/download_models.sh  # ASR/TTS/VAD，约 400MB，落 models/
uv add onnxruntime            # sherpa-onnx 的 wheel 不带 libonnxruntime.dylib
ln -sf "$(pwd)/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime."*".dylib" \
        "$(pwd)/.venv/lib/python3.12/site-packages/sherpa_onnx/lib/libonnxruntime.dylib"
```

> 两个坑都已实测：① wheel 缺 `libonnxruntime.dylib`，即使 cp312/cp314 也一样，需软链补上；
> ② GitHub release 直连不通，模型走 HF 镜像。

## 当前进度

- ✅ **M0 骨架**：质量门禁 + WS echo 服务 + sim 三替身
- ✅ **M1 serial 全链**：serial 调度 + 句末切句 + TTS 分片 + waterfall 账单（`uv run python scripts/demo_serial.py`）
- ⬜ M2 real 接入 + 本机校准 + EOU 判停 / M3 barge-in + live 客户端 / M4 pipelined / M5 收尾发版

代码按 `技术方案.md` §9 的里程碑推进。每个数字都必须能被一条命令复现。

## 快速开始

```bash
uv sync                       # 安装依赖（含 dev 组）
uv run python scripts/demo_serial.py   # 跑一轮 serial，打印 waterfall + 双口径账单

uv run ruff check .           # lint
uv run mypy                   # 类型门禁（strict）
uv run pytest                 # 测试（sim 模式，零模型零网络）

uv run uvicorn voice_pipeline.server.app:app --port 8000
curl localhost:8000/healthz
```

WS 协议：`session.begin{mode: echo|serial}` → 上行 PCM16 音频帧 → `session.end` 触发一轮。
echo 模式量 WS 链路本身的开销，serial 模式走 ASR→LLM→分句→TTS→200ms 分片。

## 许可

代码 MIT（见 LICENSE）。模型不进仓，由 `scripts/download_models.sh` 拉取；模型许可证在
各自的模型仓库，商用自查。
