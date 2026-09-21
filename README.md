# voice-pipeline-mvp

准生产级实时语音对话链路（ASR → LLM → TTS）参考实现。

配套文章：《给AI产品接上嘴——ASR+TTS接入实战，延迟从2秒压到300毫秒》。本仓的任务是让文章里
的每个机制、每个数字都能被一条命令复现，并把评论区「300ms 是否可能」的争论做成可复现实验。

- 立项背景、评论区存档、验收基线：`背景与基线.md`
- 技术选型、架构、协议、里程碑：`技术方案.md`

## 当前进度

**M0 骨架**（进行中）：质量门禁（ruff + mypy strict + pytest）+ WS echo 服务 + sim 三替身。
代码按 `技术方案.md` §9 的里程碑推进，M0 → M5。

## 快速开始

```bash
uv sync                       # 安装依赖（含 dev 组）
uv run ruff check .           # lint
uv run mypy                   # 类型门禁（strict）
uv run pytest                 # 测试（sim 模式，零模型零网络）

uv run uvicorn voice_pipeline.server.app:app --port 8000
curl localhost:8000/healthz
```

## 许可

代码 MIT（见 LICENSE）。模型不进仓，由 `scripts/download_models.sh` 拉取；模型许可证在
各自的模型仓库，商用自查。
