# voice-pipeline-mvp

准生产级实时语音对话链路（ASR → LLM → TTS）参考实现。

配套文章：《给AI产品接上嘴——ASR+TTS接入实战，延迟从2秒压到300毫秒》。本仓的任务是让文章里
的每个机制、每个数字都能被一条命令复现，并把评论区「300ms 是否可能」的争论做成可复现实验。

- 立项背景、评论区存档、验收基线：`背景与基线.md`
- 技术选型、架构、协议、里程碑：`技术方案.md`
  - **§13 是本机实测**，所有延迟数字的落点

## 本机准备（macOS）

```bash
uv python pin 3.12          # sherpa-onnx 的 wheel 在 3.13+ 上有坑
brew install espeak-ng      # kokoro / melo 的中文音素词典依赖 espeak-ng-data
./scripts/download_models.sh  # ASR/TTS/VAD，约 400MB，落 models/
uv add onnxruntime            # sherpa-onnx 的 wheel 不带 libonnxruntime.dylib
ln -sf "$(pwd)/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime."*".dylib" \
        "$(pwd)/.venv/lib/python3.12/site-packages/sherpa_onnx/lib/libonnxruntime.dylib"
```

> 三个坑都已实测：① wheel 缺 `libonnxruntime.dylib`，即使 cp312/cp314 也一样，需软链补上；
> ② GitHub release 直连不通，模型走 HF 镜像；③ 中文 TTS 必须带 espeak-ng-data，brew 装最省事。

## 当前进度

- ✅ **M0 骨架**：质量门禁 + WS echo 服务 + sim 三替身
- ✅ **M1 serial 全链**：serial 调度 + 句末切句 + TTS 分片 + waterfall 账单（`scripts/demo_serial.py`）
- ✅ **M2a real 接入 + 本机校准**：真模型跑通 ASR → LLM → TTS，`real` 模式已进 WS
- ✅ **M2b EOU 判停（四路融合 + 三档门控）**：`core/eou.py` 实现 L0 能量 + L1 静音 + L2 韵律终顿 + L3 语义完整融合，conservative/semantic/aggressive 三档；real 模式 WS 自动判停、不再依赖客户端 `session.end`；抢话 `premature_eou` 与漏判 `missed_eou` 进账单。L2/L3 默认用模型无关启发式，神经网络版（Smart Turn v3 / turn-detector INT8）留作同接口可插拔升级（D2）
- ✅ **M4 pipelined 投机调度**：用户说完前把 LLM+TTS 算完，EOU 后即刻播音；命中时口径B **179ms**（melo + 250ms 静音），首音 ≤300ms 达标。作废（答错问题）与打断（不留悬挂任务）两条纪律单测覆盖（`tests/test_pipelined.py`）
- ⬜ M3 barge-in + live 客户端 / M5 收尾发版

代码按 `技术方案.md` §9 的里程碑推进。每个数字都必须能被一条命令复现。

## 本机实测（Apple M5 / 32GB，见技术方案 §13）

| 项 | 实测 |
|---|---|
| ASR 出词粒度 | paraformer **600ms** / zipformer transducer **300ms**（模型固有，改喂帧节奏无效）|
| ASR CPU 开销 | RTF 0.008（transducer）/ 0.026（paraformer）|
| TTS 首音地板 | kokoro 660ms、melo 156ms（**无增量流**：首片到手 = 整句合成完成）|
| TTS 单点收益 | 只换引擎，端到端口径B 从 1120ms → **713ms** |
| 端到端口径B（LLM 为 sim，即本机侧下界）| kokoro 964~1120ms / melo **713ms** |

**TTS 是最大的单项开销。** 300ms 在「EOU 之后才开始动」的前提下数学上不可达
（TTS 地板 156ms + EOU 判停 200~300ms + ASR 尾延迟 ~120ms 已超 500ms），
只能靠**投机执行**把 TTS 藏进用户说话的尾段——这是 M4 的主线。

| 端到端口径B（命中投机，melo + 250ms 静音窗）| **179ms** —— 首音 ≤300ms 达标（`scripts/bench_schedulers.py`）|
| 端到端口径B（未命中投机，退化串行）| 713ms（melo 地板）|

## 快速开始

```bash
uv sync                       # 安装依赖（含 dev 组）
uv run python scripts/demo_serial.py   # sim 全链：waterfall + 双口径账单

./scripts/download_models.sh  # 拉真实模型（~400MB，不进仓）
uv run python scripts/calibrate.py --silence-ms 200,300   # 本机延迟校准 → bench/*.json
uv run python scripts/demo_real.py --tts melo             # real 全链（真模型）端到端

uv run ruff check .           # lint
uv run mypy                   # 类型门禁（strict）
uv run pytest                 # 测试（sim 零模型零网络；real 用例缺模型自动跳过）

uv run uvicorn voice_pipeline.server.app:app --port 8000
curl localhost:8000/healthz
```

WS 协议：`session.begin{mode: echo|serial|real}` → 上行 PCM16 音频帧 → `session.end` 触发一轮。
echo 量 WS 链路本身的开销；serial 用 sim 三替身跑同构链路；real 用真模型（缺模型会明确报错）。

## 许可

代码 MIT（见 LICENSE）。模型不进仓，由 `scripts/download_models.sh` 拉取；模型许可证在
各自的模型仓库，商用自查。
