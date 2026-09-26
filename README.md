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
- ✅ **M2b EOU 判停（四路融合 + 四档门控）**：`core/eou.py` 实现 L0 能量 + L1 静音 + L2 终顿 + L3 语义完整融合，conservative/semantic/aggressive/**neural** 四档；real 模式 WS 自动判停、不再依赖客户端 `session.end`；抢话 `premature_eou` 与漏判 `missed_eou` 进账单
- ✅ **D2 神经判停器（2026-09-22）**：L2 换成 Pipecat Smart Turn v3.2（8MB INT8，BSD-2），`providers/eou/smart_turn.py`；mel 特征自己用 numpy 写（`core/eou_mel.py`，零新增运行时依赖，与 transformers 逐值核对 ~6e-6）。官方测试集受控实验（中文 n=335）：**静音 100ms 判停 → 句末检出 70.4% / 句中抢话 4.6%**；真 WS 链路判停 **513ms → 100ms**。默认档已切 `neural`（缺权重的机器自动退回 conservative 并告警）。L3 文本路（turn-detector INT8）仍未接
- ✅ **M4 pipelined 投机调度**：用户说完前把 LLM+TTS 算完，EOU 后即刻播音。作废（答错问题）与打断（不留悬挂任务）两条纪律单测覆盖（`tests/test_pipelined.py`）。⚠️ **2026-09-22 实测：投机是净负债不是优化** —— 真 LLM 下命中率 1/24 = 4%，p50 比串行慢 651ms。旧 README 写的「命中 179ms、300ms 达标」是 sim LLM + 挑最好一轮的产物，已撤下，完整账见 `docs/实验报告.md`
- ✅ **M3 barge-in 三层 + 状态机 + live 客户端**（详见技术方案 §14）：
  - `core/turn_fsm.py`：IDLE / LISTENING / THINKING / SPEAKING / INTERRUPTED + 合法转换表，非法转换在 `strict` 下抛 `IllegalTransition`
  - `core/barge_in.py`：L1 检测（能量阈值 + 连续帧去抖），接口可插拔换成 AEC/VAD
  - L2 播放层：`scheduler.cancel()` 掐掉在飞计算，并下发 `audio.stop` 让播放端丢弃已排队未出声的分片；L3 语义层：本轮作废、状态机回 LISTENING，**等用户把插话说完再答**（不重开一轮、不一刀切静音）
  - WS 改为三路并发（接收循环 / `run_turn` / 发送 `pump`），中间用 outbox 队列解耦——AI 播放期间仍在上行麦克风帧，打断检测才跑得起来
  - 账单新增 `barge_in_count` / `barge_in_latency_ms`
  - live 麦克风客户端 `scripts/live_client.py`（pyaudio 16k 上行 / 24k 下行播放 + 本地打断检测）
- ⬜ M5 收尾发版
- ✅ **B 路线：可用语音试点（2026-09-27，分支 feature/route-b-pilot）**：
  - **多轮对话**：history 贯穿 Session→调度器→LLM（最近 10 轮整对裁）；连带修掉两个单轮测试看不见的真 bug——ASR 死流（finalize 后不 reset，第 2 轮音频进死流）与账单复读（TurnMetrics 只记首达，跨轮不清账）。`tests/test_multi_turn.py` + `scripts/probe_multi_turn.py`（真模型两轮探针）钉住
  - **静态 token 鉴权**：`VOICE_WS_TOKEN` 设了才启用，错/缺回 `error(unauthorized)`+1008；QwenLlm 共享 httpx 连接（每轮省一次 TLS 握手）
  - **浏览器客户端 `web/`**：vanilla JS + AudioWorklet，浏览器 AEC 免耳机；FastAPI 挂静态（显式路由之后，防 mount 吞 /healthz）
  - **混合部署**（`docs/ecs-baseline.md`，2C/2G 腾讯 ECS 实测定档）：本地 ASR（RTF 0.044）+ 本地 EOU（SmartTurn）+ 云 LLM + **云 TTS（DashScope CosyVoice 流式，首片 0.6s——本地 melo 同句 11.9s 判死）**；调度器按 `tts_streams` 能力分流，流式 provider 边合成边出片；provider 缓存消掉 session.begin 的秒级模型重载
  - **ECS 实测端到端口径B = 1553.9ms**（判停+ASR 本地，LLM+TTS 云端），达成 <2s 试点验收线
  - 部署：systemd + caddy（`deploy/`，含 release.sh 与首次部署手册；libonnxruntime 1.28.2 需 GitHub release 中转，onnxruntime 钉上界 <1.29）

代码按 `技术方案.md` §9 的里程碑推进。每个数字都必须能被一条命令复现。

## 本机实测（Apple M5 / 32GB，见技术方案 §13）

| 项 | 实测 |
|---|---|
| ASR 出词粒度 | paraformer **600ms** / zipformer transducer **300ms**（模型固有，改喂帧节奏无效）|
| ASR CPU 开销 | RTF 0.008（transducer）/ 0.026（paraformer）|
| TTS 首音地板 | kokoro 812ms、melo 255ms（**单字**；无增量流：首片到手 = 整句合成完成）|
| TTS 边际成本 | kokoro +47ms/字、melo +37ms/字（12 字句：1689ms / 1297ms）|
| TTS 端到端收益 | 只换引擎（kokoro→melo），真 LLM 端到端口径B 1922→**1758ms**、1731→**1322ms** |
| 端到端口径B（LLM 为 sim，即本机侧下界）| kokoro 964~1120ms / melo **713ms** |
| 端到端口径B（**真 LLM = qwen-flash**，melo）| **1260ms**（20 字句）/ 1758ms（12 字句，p50）|

**接上真 LLM 之后，端到端口径B 从 713ms 涨到 1109ms。** 713ms 是"本机侧下界"，
不是可对外报的延迟；含真 LLM 的数字才是。`scripts/probe_llm.py` 单独量 LLM 那一格。

**TTS 是最大的单项开销——实测占端到端口径B 约七成。** 真 WS 账单按格拆开：
判停 100ms、LLM 首 token ~350ms、首句完成 ~36ms、**TTS 首片 964~1376ms**。
所以 300ms 不止是"差一点"，是差 4 倍：本机最好 1260ms。

TTS 引擎默认 **melo**（2026-09-22 实测定档，见 `docs/实验报告.md` 第六节）；
代价是音量偏低（RMS 峰值 0.031 vs kokoro 0.080）、44.1k 需重采样、单音色。
切回只需 `VOICE_PIPELINE_TTS_ENGINE=kokoro`。**换引擎不能救 300ms**，它只省 194~379ms。

> 历史判断更正：这里原先写「瓶颈是 LLM」，那是推断不是实测，已按格拆账推翻。
> 详见报告第六节开头。

**A/B 的完整账在 [`docs/实验报告.md`](docs/实验报告.md)**（M4 的验收物，此前缺失）。一句话：

> **300ms 不成立。** 投机执行（pipelined）在真 LLM 下命中率 **1/24 = 4%**，常态 p50 **2491ms**
> —— 比串行 1840ms **慢 651ms**。它改善的是均值、恶化的是中位数：把分布拉成双峰。

| 口径B（p50）| serial | pipelined | 命中率 | 投机净收益 |
|---|---|---|---|---|
| SimLlm（零延迟替身，不含 LLM）| 770ms | 842ms | 9/36 = 25% | **−72ms** |
| **QwenLlm（qwen-flash 真网络）** | **1840ms** | **2491ms** | **1/24 = 4%** | **−651ms** |

`179ms` 这个数字是 **sim LLM + 挑最好一轮** 的产物，已从对外表述中撤下。真 LLM 下唯一进 600ms
的那一轮是 534ms，而它需要的窗口是「静音 600ms + grace 600ms」——真实对话里不存在。
机制不难懂：投机要在「静音窗 + grace」内跑完 LLM 首完整句（522ms）+ TTS（~200ms），
而窗口只有 250ms + grace；加大 grace 能买命中率，未命中时却是纯亏损，两头一挤期望值为负。

## 快速开始

```bash
uv sync                       # 安装依赖（含 dev 组）
uv run python scripts/demo_serial.py   # sim 全链：waterfall + 双口径账单

./scripts/download_models.sh  # 拉真实模型（~400MB，不进仓）
uv run python scripts/calibrate.py --silence-ms 200,300   # 本机延迟校准 → bench/*.json
uv run python scripts/demo_real.py --tts melo             # real 全链（真模型 + 真 LLM）端到端
uv run python scripts/probe_llm.py                        # 单独量 LLM 那一格：首 token / 首完整句

uv run python scripts/fetch_eou_model.py    # 神经判停器权重（8MB，不进仓）
uv run python scripts/fetch_eou_dataset.py  # 判停评测集（439MB，不进仓；可只下 1 个分片）
uv run python scripts/bench_eou_controlled.py --langs zho,eng   # 句末检出 vs 句中抢话

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

`gating` 四档 `conservative|semantic|aggressive|neural`，**默认 `neural`**——这是 2026-09-22
第三次定档的结论（前两次：semantic 抢话 20~40% → 退回 conservative）：

| 档位 | 依据 | 判停延迟 | 抢话率 |
|---|---|---|---|
| **neural（默认）** | 官方测试集受控实验，中文 n=335 | **~100ms**（70% 轮次） | **4.6%** |
| conservative | 合成语音 + 真 ASR，句内停顿 250/350ms | 515ms | 0/5、0/5（同一场景、同一份合成数据） |
| semantic | 同上 | 330ms | 1/5、2/5 |
| aggressive | 同上 | 220ms | 5/5、5/5 |

⚠️ 四档的数字**不是同一份数据测出来的**，不能横比：neural 用的是官方数据集（带真值标签），
其余三档用的是自造合成语音（真值有缺陷，见 `docs/实验报告.md` §5.1）。neural 档需要
8MB 权重（`scripts/fetch_eou_model.py`），**没有就自动退回 conservative 并告警**，
账单 `eou_mode` 会如实写回。

semantic 用 185ms 延迟换 20~40% 抢话率；把它的静音阈值调到同等安全的 500ms 后延迟 520ms
≈ conservative 的 515ms——**同安全下没有收益**。已知代价：conservative 的 500ms 静音窗把
口径B 托底在 500ms 以上，300ms 在本判据下与「不抢话」不可兼得。`grace_ms` 是 pipelined 的
投机等待窗（EOU 后最多再等多久让投机落地）。它是赌注：真 LLM + 250ms 静音窗下
`grace=0` 命中 0/9（永远不命中），`grace=300` 命中 0/6 且未命中时白等 300ms——两头一挤期望值为负。

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
| **真 LLM 选型已定，代价已入账** | real 模式接 **DashScope `qwen-flash`**（`providers/llm/qwen_llm.py`），本机实测首 token 367ms / 首完整句 522ms，是候选里最快的非思考型档 | 端到端口径B 因此从 713ms（sim 下界）涨到 **1109ms**；300ms 原先指望投机命中，但 2026-09-22 实测命中率 **4%（1/24）** 且 p50 比串行慢 651ms —— **这条路已被实测否定**，见 `docs/实验报告.md` | 无 key 时 `auto` 退回 SimLlm **并告警**（静默退回会让账单少算一格，护栏就是那条日志）；DeepSeek 现役模型默认开思考链，见 `scripts/probe_llm.py` |
| **回声抑制未做** | barge-in 仅 L1 能量判定 | 外放会自激误触发；一期靠入耳式耳机**物理隔离** | 文本↔ASR 重合抑制 / 软件 AEC（二期） |
| **打断尾音未量化** | `barge_in_latency_ms` 只记服务端检测时刻 | 不含客户端播放缓冲残留音频的实际停播耗时 | 需带播放队列的真机测量 |
| **scripted 客户端未做** | 只有 live（`scripts/live_client.py`） | CI 里没有「回放固定音频走 WS」的确定性回归 | 补 `client/scripted_client.py` |
| **JSONL 事件流落盘未做** | 账单在内存 / 脚本打印 | 无可观测性落盘，事后无法回溯单轮时间线 | 每轮 waterfall 落 JSONL |
| **有界队列未接线** | `DualPriorityQueue` 已实现且有单测，但 WS 主链路用的是**无界 outbox** | 背压/丢弃计数在真实链路未生效 | 接进 `server/app.py` |
| **无 Docker / compose** | 本机 `uvicorn` 可跑 | 没有容器化部署形态 | 补 Dockerfile + compose + healthcheck |
| ~~EOU L2/L3 是启发式~~ → **L2 已换神经版** | L2 用 Smart Turn v3.2，L3 仍是启发式 | ① 中文样本全是 chirp3（TTS 合成），抢话率 4.6% 是乐观下界，D5 真人集未建；② L3 文本路（turn-detector INT8）未接；③ 短轮次（<2s）判停未测 | D5 中文真人集；必要时接 L3 |
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
