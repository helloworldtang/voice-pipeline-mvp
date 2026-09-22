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
- ✅ **M3 barge-in 三层 + 状态机 + live 客户端**（详见技术方案 §14）：
  - `core/turn_fsm.py`：IDLE / LISTENING / THINKING / SPEAKING / INTERRUPTED + 合法转换表，非法转换在 `strict` 下抛 `IllegalTransition`
  - `core/barge_in.py`：L1 检测（能量阈值 + 连续帧去抖），接口可插拔换成 AEC/VAD
  - L2 播放层：`scheduler.cancel()` 掐掉在飞计算，并下发 `audio.stop` 让播放端丢弃已排队未出声的分片；L3 语义层：本轮作废、状态机回 LISTENING，**等用户把插话说完再答**（不重开一轮、不一刀切静音）
  - WS 改为三路并发（接收循环 / `run_turn` / 发送 `pump`），中间用 outbox 队列解耦——AI 播放期间仍在上行麦克风帧，打断检测才跑得起来
  - 账单新增 `barge_in_count` / `barge_in_latency_ms`
  - live 麦克风客户端 `scripts/live_client.py`（pyaudio 16k 上行 / 24k 下行播放 + 本地打断检测）
- ⬜ M5 收尾发版

代码按 `技术方案.md` §9 的里程碑推进。每个数字都必须能被一条命令复现。

## 本机实测（Apple M5 / 32GB，见技术方案 §13）

| 项 | 实测 |
|---|---|
| ASR 出词粒度 | paraformer **600ms** / zipformer transducer **300ms**（模型固有，改喂帧节奏无效）|
| ASR CPU 开销 | RTF 0.008（transducer）/ 0.026（paraformer）|
| TTS 首音地板 | kokoro 660ms、melo 156ms（**无增量流**：首片到手 = 整句合成完成）|
| TTS 单点收益 | 只换引擎，端到端口径B 从 1120ms → **713ms** |
| 端到端口径B（LLM 为 sim，即本机侧下界）| kokoro 964~1120ms / melo **713ms** |
| 端到端口径B（**真 LLM = qwen-flash**，melo）| **1109ms**（账单：llm_first_token 380ms / first_sentence_done 409ms）|

**接上真 LLM 之后，端到端口径B 从 713ms 涨到 1109ms。** 713ms 是"本机侧下界"，
不是可对外报的延迟；含真 LLM 的数字才是。`scripts/probe_llm.py` 单独量 LLM 那一格。

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
uv run python scripts/demo_real.py --tts melo             # real 全链（真模型 + 真 LLM）端到端
uv run python scripts/probe_llm.py                        # 单独量 LLM 那一格：首 token / 首完整句

uv run ruff check .           # lint（含 import 排序 / 死代码 / 简化）
uv run ruff format --check .  # 格式门禁（与 CI 一致，漏跑会 push 后红）
uv run mypy                   # 类型门禁（strict）
uv run pytest                 # 测试（sim 零模型零网络；real 用例缺模型自动跳过）

uv run uvicorn voice_pipeline.server.app:app --port 8000
curl localhost:8000/healthz
```

WS 协议：`session.begin{mode, gating, grace_ms}` → 上行 PCM16 音频帧 → `session.end` 触发一轮
（real 模式由 EOU 自动判停，不必显式 `session.end`）；`interrupt` 触发打断，下行 `audio.stop`
通知播放端停播。

| `mode` | 含义 |
|---|---|
| `echo` | 原样回发音频，量 WS 链路本身的开销 |
| `serial` | 对照组：EOU 之后才启动 LLM（sim 三替身） |
| `pipelined` | **实验组**：用户说话期间投机跑 LLM+TTS（sim 三替身），唯一能逼近 300ms 的路径 |
| `real` / `real-serial` / `real-pipelined` | 同上，换真模型真 LLM（缺模型会明确报错） |

`gating` 三档 `conservative|semantic|aggressive`，**默认 `semantic`**——`conservative` 要等满
500ms 静音窗，口径B 被托底在 500ms 以上，300ms 数学上不可达。`grace_ms` 是 pipelined 的
投机等待窗（EOU 后最多再等多久让投机落地），本机实测 `grace=0` 时 6/6 全未命中、`grace=300` 才吃到那次 179ms。

live 真机试跑（M3）：

```bash
uv run python scripts/live_client.py --url ws://localhost:8000/ws   # 需 pyaudio + websockets
```

> ⚠️ **回声隔离是硬前提**：一期的打断检测是纯能量判定，外放会被 AI 自己的声音自激误触发。
> 必须戴**入耳式耳机**做物理隔离；软件 AEC 列入二期差距表（见技术方案 §14）。

## 承诺映射表（文章 / 评论区承诺 → 兑现它的代码）

> 认领标准：**表里的每个模块都真实存在于本仓**，点开能读到实现、跑通能复现数字。
> 设计意图里画过、但最终没这么做或没做的，一律列进下面的差距表，不在这里充数。

| 承诺（出处） | 兑现它的代码 | 复现证据 |
|---|---|---|
| 文章第一板：句末标点切句，不切逗号 | `core/sentence_splitter.py` | `tests/test_sentence_splitter.py` |
| 文章第三板：24kHz / 200ms / PCM16 可配置 | `audio/chunker.py`（`AudioParams`） | `tests/test_chunker.py` |
| 文章 barge-in 三层 | L1 `core/barge_in.py`、L2/L3 `server/session.py`、状态机 `core/turn_fsm.py` | `tests/test_bargein.py` |
| 评论区：滑动窗口语义提交 / 预生成 / 纠偏作废 | `core/scheduler/pipelined.py` | `tests/test_pipelined.py` |
| 质疑方时序模型（A/B 对照组） | `core/scheduler/serial.py` | `tests/test_serial_pipeline.py` |
| 双口径账单 + waterfall（回应「300ms 怎么得到的」） | `metrics/events.py`（`bill()` / `waterfall()`） | `scripts/demo_serial.py` |
| 能量门控（评论区「星🎈」） | `core/eou.py` 的 L0 能量 + L1 静音两路 | 见差距表：**ASR 前置** gate 未做 |
| EOU 四路融合判停（声学→VAD→韵律→语义） | `core/eou.py` | `tests/test_eou.py` |
| WS 会话服务 + 三路并发（收 / 跑一轮 / 发） | `server/app.py` + `server/session.py` | `tests/test_ws.py` |
| provider 插件层（sim / real 一键切） | `providers/factory.py` + `providers/base.py` | `tests/test_factory.py` |
| live 麦克风客户端（真人对讲） | `scripts/live_client.py` | 见差距表：scripted 客户端未做 |
| 本机（Apple M5）延迟校准 | `scripts/calibrate.py` | 技术方案 §13（`bench/*.json`） |

文章第二板「连接预热」是**部分兑现**：模型构造即加载、进程内常驻（省去每次调用重载），
但「预建连接 + 预合成静音帧」的 WarmPool 没做——详见差距表。

## 对比定位

**本仓不是什么**：不是通用 agent 编排框架（pipecat / LiveKit Agents 那种），不是语音 SaaS，
不提供 GPU 推理服务。

**本仓是什么**：一个**可复现的延迟实验台 + 准生产级 WS 服务骨架**，把文章的机制做成可跑代码、
把评论区的争论做成可复现实验。它比「读文章」多给三样东西：

1. **双口径账单 + waterfall**：口径A（用户开口→AI 首音）缺 `vad_onset` 时**如实留空、不填补**；
   口径B（用户说完→AI 首音）是工程目标。每个外发数字都能被一条命令复现。
2. **A/B 调度器**：`serial`（对照组，复现质疑方 1.2~1.9s 的算术）vs `pipelined`（实验组，投机流水线），
   同机同 provider、只换调度——把「300ms 是否可能」变成一个能跑的实验，而不是互相嘴仗。
3. **可靠性纪律的单测**：投机作废（答错必须作废）、打断（不留悬挂任务）、状态机合法转换、
   竞态专项（挂死要能被门禁抓住）。

**选型立场**：自研 FastAPI + WebSocket，不二开框架——因为要的是**能逐行解释的延迟账**，
不是框架黑盒。sherpa-onnx 全本地、可离线、无 GPU 依赖。

## 差距表（离「真生产」还有多远）

> 这一节是**交付物本身**，不是免责声明。生产级 = 修到位的功能 + 如实声明的差距。

| 项 | 现状 | 影响 | 后续 |
|---|---|---|---|
| **真 LLM 选型已定，代价已入账** | real 模式接 **DashScope `qwen-flash`**（`providers/llm/qwen_llm.py`），本机实测首 token 367ms / 首完整句 522ms，是候选里最快的非思考型档 | 端到端口径B 因此从 713ms（sim 下界）涨到 **1109ms**；300ms 只剩投机命中一条路，而命中率实测 1/27 | 无 key 时 `auto` 退回 SimLlm **并告警**（静默退回会让账单少算一格，护栏就是那条日志）；DeepSeek 现役模型默认开思考链，见 `scripts/probe_llm.py` |
| **回声抑制未做** | barge-in 仅 L1 能量判定 | 外放会自激误触发；一期靠入耳式耳机**物理隔离** | 文本↔ASR 重合抑制 / 软件 AEC（二期） |
| **打断尾音未量化** | `barge_in_latency_ms` 只记服务端检测时刻 | 不含客户端播放缓冲残留音频的实际停播耗时 | 需带播放队列的真机测量 |
| **scripted 客户端未做** | 只有 live（`scripts/live_client.py`） | CI 里没有「回放固定音频走 WS」的确定性回归 | 补 `client/scripted_client.py` |
| **JSONL 事件流落盘未做** | 账单在内存 / 脚本打印 | 无可观测性落盘，事后无法回溯单轮时间线 | 每轮 waterfall 落 JSONL |
| **有界队列未接线** | `DualPriorityQueue` 已实现且有单测，但 WS 主链路用的是**无界 outbox** | 背压/丢弃计数在真实链路未生效 | 接进 `server/app.py` |
| **无 Docker / compose** | 本机 `uvicorn` 可跑 | 没有容器化部署形态 | 补 Dockerfile + compose + healthcheck |
| **EOU L2/L3 是启发式** | 非神经网络版（省依赖、可离线） | 中文准确率未经神经模型实测 | D2：同接口换 turn-detector INT8 / Smart Turn v3 |
| **WarmPool「预合成静音帧」未做** | 仅模型常驻 | 首音前少一档可优化的常量延迟 | 见技术方案 §13.4 |
| 并发压测 / 鉴权 TLS / 媒体面 / 音色克隆 | 未做 | 单会话参考实现，非集群 | **范围外**，见下 |

**明确的范围外**（不要悄悄做进去）：商用抗噪与人声干扰、并发压测、硬件 AEC、鉴权/TLS、
媒体面（FreeSWITCH / WebRTC / 唇形）、音色克隆（zipvoice）。

## 发版检查单

- [x] 质量门禁四件套全绿：`ruff check` / `ruff format --check` / `mypy`(strict) / `pytest`
- [x] CI 与本地同命令同结果（py3.10 / py3.12 矩阵）
- [x] LICENSE = MIT（见 `LICENSE`）
- [x] README：本机准备 / 快速开始 / 承诺映射表 / 对比定位 / 差距表
- [x] 技术方案：§13 本机实测、§14 M3 实现与诚实边界
- [x] sim 模式零模型零网络（CI 可跑）；real 用例缺模型自动跳过
- [ ] Docker / compose（**列差距表**，非阻塞发版）
- [ ] `git push`（本轮按要求**不 push**，先在本地验收）

## 许可

代码 MIT（见 LICENSE）。模型不进仓，由 `scripts/download_models.sh` 拉取；模型许可证在
各自的模型仓库，商用自查。
