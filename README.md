# voice-pipeline-mvp

准生产级实时语音对话链路（ASR → LLM → TTS）参考实现。

配套文章：《给AI产品接上嘴——ASR+TTS接入实战，延迟从2秒压到300毫秒》。本仓的任务是让文章里
的每个机制、每个数字都能被一条命令复现，并把评论区「300ms 是否可能」的争论做成可复现实验。

**当前目标（2026-09-27 补充）**：把这条链路做成**成熟、可交付的 AI 语音产品**。
技术验证阶段（M0–M5）已闭环，且给出了一个否定结论——**300ms 在本架构下不成立**；
现在的重心从「证明延迟」转向「做成产品」。当前已达**可用试点**（ECS 端到端口径B 1553.9ms），
离成熟交付还差 1 项（11 项判据中 10✅/1⚠️/0❌，仅剩判据 4 TLS/公网开麦），列在本文件末的差距表与 `背景与基线.md`「一之二」节。

- 立项背景、评论区存档、验收基线（含第四层目标的 11 项成熟度判据）：`背景与基线.md`
- 技术选型、架构、协议、里程碑：`技术方案.md`
  - **§13 是本机实测**，所有延迟数字的落点
- 延迟账本（每个结论的样本量与边界）：`docs/实验报告.md`
- 为什么不挖并行、天花板在哪：`docs/流程拆解与优化空间.md`
- **耗时归因（TraceID + Span 树，残差恒等式自检）：`docs/链路追踪-根因分析.md`**
- 云上部署选型与实测：`docs/ecs-baseline.md`

## 当前目标（2026-09-27 补充第四层）

本仓目标由三层扩为**四层**（详见 `背景与基线.md`「一之二」节）：

| 层 | 目标 | 判据 | 状态 |
|---|---|---|---|
| 一 | 支撑文章 | 文章每个机制、账单每个数字可由一条命令复现 | ✅ |
| 二 | 回应评论区 | 「300ms 是否可能」做成可复现 A/B 实验 | ✅ 结论是否定的（见 `docs/实验报告.md`） |
| 三 | 超越文章 | 把承诺做实做透，不冒领 | ✅ |
| **四** | **成熟 AI 语音交付产品** | 功能完整 + 部署成型 + 可观测 + 可承受真实用户 + 差距如实成文 | ⬜ **当前主目标** |

> ⚠️ **重要转向**：`口径B 300ms` 已被实测否定（真 LLM 下本机最好 1260ms，投机是净负债），
> 不再是当前主目标。**当前主目标是第四层：把试点做成可交付产品**——这是可完成、可验收的工程量。
> 成熟度 11 项判据与现状对照见 `背景与基线.md`；**当前 10 项达标 / 1 项部分 / 0 项未做**（2026-10-03：判据 3/5/7；2026-10-08：判据 9 并发；2026-10-10：判据 8 真人验收），缺口即下节差距表。

> 🕐 **待复核（2026-10-28，未改口）**：`docs/验收-本机真人-20261027-125840468.md` 里
> real-pipelined 真人 10 轮口径B **p50 298ms**，与上面「300ms 已被否定」表面冲突。复查结论：
> 该轮 `spec_hit` **0/10**（投机未命中，记录早于修复），298ms 是**串行路径 + 本机 1.5B 模型**
> 取得的；且口径B 不含判停等待（conservative 档固定 500ms）与客户端出声（~55ms），
> 用户实际等待 ≈ **850ms**。云端 LLM 下仍是 1260ms 量级。
> ⇒ **等用修复后的 pipelined 补跑一轮真人验收再统一对外口径**，本结论暂不改。

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

> ⚠️ **2026-10-17 追踪数据要改一句话**：换引擎**能**救的不是"194~379ms"，而是
> **melo→云流式 dashscope 一刀省 ~930ms**（口径B 1966→1029ms，见
> `docs/链路追踪-根因分析.md` §三）。差别在于本地 melo 首片随字数线性增长
> （~70ms/字 × 22 字 ≈ 1.5s），而云流式首片是**固定 ~600ms**、与字数无关。
> 也就是说：**纯本机部署的下限就是 1.9s 级；混合部署（云流式 TTS）才是 1.0s 级。**
> 归因方法与逐格证据见上面那份追踪文档，探针 `scripts/probe_trace.py`。

> 🔬 **2026-10-17 深拆（第二层 span，见追踪文档 §四）**：云流式那 1029ms 的构成是
> **服务端首字节 332ms + 建连 257ms + LLM 网关 231ms + 首句 120ms + 传输 31ms**，
> 未归因残差 0ms。两个可操作结论：① **建连那 257ms 每句都白付**（合成器不复用），
> SDK 自带对象池预连后首片 604→340~375ms——当前最大的一笔可省；
> ② **换 LLM 模型救不了**：qwen-flash 已是全平台最快，新"flash"全是推理型
> （ttft 2.8~11.3s）；决策模型 Jev 不生成自然语言，顶替不了回复槽位。

> ✅ **2026-10-17 P0.5 已落地（连接预热，追踪文档 §四 4.4）**：建连那一格已去掉——
> `DashScopeTts` 自建连接池（预热走 SDK `__connect`，复用走 `__reset`+
> `__update_params` 换 task_id），判停/建会话即非阻塞预热，温连接已死且未出片时
> 冷路径重试一次。**同机同时段 A/B（`--no-prewarm` 对照）：口径B p50 724 vs 700ms
> ——差异在噪声内**；预热的真实收益是**每会话第一轮 1317→777ms** 与 `tts.connect`
> **257→38ms**（建连彻底移出口径B 窗口）。⚠️ 「1029→724ms」是**跨时段对比**
> （1029 来自 10-17 早段旧代码归档基线：每句冷建连），不是 A/B 的读数，别混着报。
> 恒等式残差仍 0ms，2 会话并发共享池验证通过（834/810ms）。

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
uv sync --extra local         # 可选：本地 LLM（MLX，仅 macOS/arm64）
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

**推荐启动（当前本机最优：口径B ≈619ms）**——本地 LLM 降级档 + 云 TTS：

```bash
VOICE_LLM_ENGINE=mlx ./scripts/dev_server.sh   # 需先 uv sync --extra local
```

- 本地 LLM 热态 TTFT ~45ms（微基准）/ ~100ms（流水线内），首句 ~75ms；比云端省 ~100ms；
- 依赖 `VOICE_ASR_THREADS` 默认 **1**（ASR RTF 0.008~0.016，1 线程远超实时；
  多出来的线程会和本地推理抢核，实测把口径B 抬回 741ms）；
- 缺云 key 时 `auto` 自动降级本地，账单 `llm` 字段如实写 `MlxLlm`——**不静默退回 SimLlm**；
- 代价：1.5B 答案更笼统（「气温适中」而不是「18 到 25 度」），降级档要明说。

WS 协议：`session.begin{mode, gating, grace_ms}` → 上行 PCM16 音频帧 → `session.end` 触发一轮
（real 模式由 EOU 自动判停，不必显式 `session.end`）；`interrupt` 触发打断，下行 `audio.stop`
通知播放端停播。**上行还有一帧 `playback.report`**：客户端把自己量到的「收到首片→出声」
时长（含 `AudioContext.outputLatency`）回传，服务端记 `rx.playback` span 与
`playback_ms` / `playback_tail_ms`——服务端追踪止于落网线，这是对端唯一的证据。
它只报**时长**不报时刻：两端时钟不同步，报时刻必然漂。

| `mode` | 含义 |
|---|---|
| `echo` | 原样回发音频，量 WS 链路本身的开销 |
| `serial` | 对照组：EOU 之后才启动 LLM（sim 三替身） |
| `pipelined` | **实验组**：用户说话期间投机跑 LLM+TTS（sim 三替身），唯一能逼近 300ms 的路径 |
| `real` / `real-serial` / `real-pipelined` | 同上，换真模型真 LLM（缺模型会明确报错） |

`gating` 四档 `conservative|semantic|aggressive|neural`，**默认 `conservative`**——第四次定档
（2026-10-04 真人验收 D5：neural 在半秒句中停顿下抢话 3/3，100ms 确认窗扛不住真人停顿，
判停 100→500ms 换 0% 抢话；neural 仍是可选项，说话不停顿的场景快 400ms。
定档史：semantic 抢话 20~40% 被否两次 → neural 上位 → 真人停顿否掉 neural 默认）：

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
| 文章第二板（连接那一半）：TTS 连接预热 / 复用 | `providers/tts/dashscope_tts.py` 连接池 + `Session`/调度器判停即预热 | `tests/test_tts_pool.py` + 追踪文档 §四 4.4 A/B |
| 文章第三板：24kHz / 200ms / PCM16 可配置 | `audio/chunker.py`（`AudioParams`） | `tests/test_chunker.py` |
| 文章 barge-in 三层 | L1 `core/barge_in.py`、L2/L3 `server/session.py`、状态机 `core/turn_fsm.py` | `tests/test_bargein.py` |
| 评论区：滑动窗口语义提交 / 预生成 / 纠偏作废 | `core/scheduler/pipelined.py` | `tests/test_pipelined.py` |
| 质疑方时序模型（A/B 对照组） | `core/scheduler/serial.py` | `tests/test_serial_pipeline.py` |
| 双口径账单 + waterfall（回应「300ms 怎么得到的」） | `metrics/events.py`（`bill()` / `waterfall()`） | `scripts/demo_serial.py` |
| 能量门控（评论区「星🎈」） | `core/eou.py` 的 L0 能量 + L1 静音两路；**ASR 前置 gate**（P1）：`core/energy_gate.py` + `server/session.py` 接线 | `tests/test_energy_gate.py` + 追踪文档 §四 4.5 A/B |
| EOU 四路融合判停（声学→VAD→韵律→语义） | `core/eou.py` | `tests/test_eou.py` |
| WS 会话服务 + 三路并发（收 / 跑一轮 / 发） | `server/app.py` + `server/session.py` | `tests/test_ws.py` |
| provider 插件层（sim / real 一键切） | `providers/factory.py` + `providers/base.py` | `tests/test_factory.py` |
| live 麦克风客户端（真人对讲） | `scripts/live_client.py` | 见差距表：scripted 客户端未做 |
| 本机（Apple M5）延迟校准 | `scripts/calibrate.py` | 技术方案 §13（`bench/*.json`） |

文章第二板「连接预热」的**连接那一半已兑现**（2026-10-17 P0.5）：`DashScopeTts`
自建连接池，判停/建会话即预热、跨句跨轮复用——**每会话第一轮 1317→777ms、
`tts.connect` 257→38ms**（同期 A/B 稳态 p50 724 vs 700ms 在噪声内；「1029→724」是
跨时段对比，见上面那条注）；**预合成 filler（语音短音，§三①）也已兑现并实测**，
差距表里的「预合成静音帧」特指 WarmPool 的静音缓冲变体，仍未做——见差距表。

**ASR 前置能量门也已兑现**（2026-10-17 P1）：`AsrEnergyGate` 把判停后客户端持续
喂进来的静音帧拦在 ASR 之前（门限 0.01 ≤ barge-in L1，只拦 ASR 不拦 EOU 判停），
同时段 A/B 口径B p50 759 vs 872ms（−113ms，两轮 −97/−113ms）。顺带修掉两枚真 bug：
L1 每帧双算 `is_speech`（去抖计数翻倍）、`finalize()` 缺尾部静音上下文丢尾字
（转写 19→18 字，自补 600ms 静音后恢复 19/19）。

**预合成 filler 也已兑现并实测**（2026-10-17 ①）：`FillerBank` + `play_filler` 在判停成立即播
预合成短音（**体感首音 6~8ms**），盖住 TTS 首片；口径B 不变（775↔788ms，在噪声内）。
**片段必须短于 TTS 首片**（现为 430~520ms；`make_filler.py` 有 600ms 硬上限）——比首片长的 filler
会沿 FIFO 把真回答的「被听见起点」推后（第一版 1190ms 实测推后 ~400ms，已修）。
见 `docs/优化空间-20261010.md` §三① 实测。

## 对比定位

**本仓不是什么**：不是通用 agent 编排框架（pipecat / LiveKit Agents 那种），不是语音 SaaS，
不提供 GPU 推理服务。

**本仓是什么**：一个**可复现的延迟实验台 + 准生产级 WS 服务骨架**，把文章的机制做成可跑代码、
把评论区的争论做成可复现实验。它比「读文章」多给三样东西：

1. **双口径账单 + waterfall**：口径A（用户开口→AI 首音）与口径B（用户说完→AI 首音）
   分开报，缺失如实写 −1、不填补。**口径A 的两个锚点 2026-10-27 才埋上**
   （`vad_onset` = 判停前第一次被 barge-in L1 判定有人声的那一帧，`asr_first_char`
   = ASR 首次出字），⚠️ 它带已知系统性偏差：L1 是「连续 3 帧超阈」的能量判定，
   100ms 帧节奏下比真实开口晚 ~300ms，所以 `caliber_a_ms` 是**偏小**的估计——
   只作参考，不与口径B 横比，也不拿去对外宣称。口径B 才是工程目标。
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
| **TLS / 公网开麦未做**（部分已补：浏览器 AEC 路径） | 静态 token 鉴权已有（`VOICE_WS_TOKEN`，`server/app.py`）；浏览器客户端在浏览器侧做 AEC，**免耳机可用**（`web/`，AudioWorklet） | 仍是 `:80` + SSH 隔离隧道，**无域名/证书，公网不可直接开麦**（`deploy/Caddyfile`） | 上域名 → caddy 自动 TLS（配置已就绪） |
| ~~无真人验收记录~~ → **已闭环（2026-10-10）** | 真人真机对讲两段会话共 25 轮（20:56 / 20:58），账单落盘并出记录：口径B p50 **1045ms**（完整 18 轮）、打断 4 次均闭环、空转写 3/25 = 12%（上一轮 67%） | 「能真正的交流」已从**断言**变成**观测** | `docs/验收-本机真人-20261010.md`；同日查出项见下表三行新增 |
| **`missed_eou` 是误报源（2026-10-10 真人验收查出）** | 置真条件（`core/eou.py` `should_end()`）=「距上次 reset >5s **且至今未出现人声**」——记的是轮间静默，不是「该判未判」；账单自证：9/21 行带它，而同轮判停都正常 500ms（自相矛盾） | 「漏判率」这个一等指标**当前不可用**（同 2026-09 `premature_eou` 恒 0 的盲区） | 条件改为「已出现人声且长时间未判停」+ 防回退单测 |
| **LLM 无日期锚定（2026-10-10）** | `DEFAULT_SYSTEM` 不含当前日期；真人问「今天几号」，模型答「2024年6月19日」且被纠正后仍坚持；同日对「我想订会议室」答「已为您预订」（本仓无工具层） | 事实性回答不可信、宣称已执行未做的事——对语音助手是内容硬伤 | 请求时注入当日日期（动态）；工具层属范围外，宣称类回答需在 system 里约束 |
| **真人空轮触发源未定位（2026-10-10）** | 25 轮里 3 次 EOU 判停在「有能量无文字」音频上（判停 1400~2100ms），两道空转写门拦住、**无 AI 输出** | 目前只损失一次判停时长；若与回声有关，则是 AEC 残余的间接证据 | 定位能量来源（疑似 AEC 残余/环境噪声） |
| 单用户、无并发 | 共享 ORT session 上推理串行 | 多人同时用会互相拖慢且无信号 | ✅ 已修（2026-10-08）：`providers/pool.py` 会话独占推理槽 + 满载明确拒绝（session_limit+1013）；ASR decode 搬线程（留在循环里则独占 recognizer 并行不起来，实测每路反慢 190ms）；A/B：每路口径B 基本持平（n=4），总墙钟 9.2→8.8s，价值在**显式背压**而非提速 |
| **回声抑制未做（降级为已知代价）** | barge-in 仅 L1 能量判定 | 脚本客户端外放会自激误触发；**浏览器端靠 AEC 免耳机，live_client 仍需入耳式耳机物理隔离** | 文本↔ASR 重合抑制 / 软件 AEC（二期） |
| **JSONL 事件流已落地**（2026-09-27，从缺口转交付；2026-10-27 部署面补齐） | `metrics/jsonl.py`：每轮一行，含 waterfall 各点 + 双份文本 + span 树 + 整轮 `root_breakdown`；`VOICE_EVENTS_JSONL` 启用（默认关，零副作用） | 首日就拦下一个静默错的账单字段（`eou_mode` 漏 `neural` 默认档）；2026-10-27 前 `deploy/` 三个模板都没设这个变量，判据 6 只是纸面达标 | **已修**：service / Dockerfile / env.example 写死路径 + compose 挂卷 + `SETUP.md` 第 6 节 logrotate（一轮一行可达百 KB，必须轮转）；多 worker 部署仍需换集中式收集 |
| 打断尾音未量化 | `barge_in_latency_ms` 只记服务端检测时刻 | 停播耗时缺客户端证据 | ✅ 已量化（2026-10-03 web 端测队列残余 + 2026-10-27 双向补齐）：服务端 `tx.stop` span 量「检测→停播帧落网线」，客户端 `playback_tail_ms` 把被丢弃的排程音频回传进账单；**OS 音频管线里已在途的毫秒级仍测不到，留白** |
| `premature_eou` 恒为 0（分析盲区） | `run_turn` 采完指标就 `eou.reset()`，把「判停后又出现人声」的证据擦掉 | 一等验收指标（抢话率）曾不可用 | ✅ 已修（2026-10-03）：观察窗跨 reset 存活 + `take_premature()` 记账时读走；真链路复验切轮账单记 1.0 |
| **默认档 `neural` 句内停顿切轮** | 受控实验：同音频下 `neural` 把「今天上海天气怎么样」切成 `今` + `海天气怎么样` 两轮，`conservative` 完整 | ⚠️ 样本是 TTS 合成语音（内部静音实测 150~240ms > 100ms 确认窗），**真人是否切轮未验**；若真人也切，默认档要退回 `conservative` | 真人验收现场复现；必要时加「最少说话时长」护栏 |
| scripted 客户端未做 | 只有 live（`scripts/live_client.py`） | 缺「回放固定音频走 WS」的确定性取证 | ✅ 已补（2026-10-03）：`scripts/scripted_client.py`（不合成不起服务，回放 WAV 出账单） |
| 有界队列未接线 | 初版 `DualPriorityQueue` 形态有顺序倒置（所有控制帧跳队会打乱 turn.end/音频序） | 背压/丢弃计数曾未生效 | ✅ 已接线（2026-10-03）：重构为 `OutboundQueue`（仅 audio.stop 跳队、其余保序、音频有界丢旧计数）接进 WS 主链路 |
| 无 Docker / compose | 本机 `uvicorn` 可跑 | 没有容器化部署形态 | ✅ 已补（2026-10-03）：`deploy/Dockerfile` + `compose.yml`，linux/amd64 构建并起容器实测 healthz/页面通过 |
| ~~EOU L2/L3 是启发式~~ → **L2 已换神经版** | L2 用 Smart Turn v3.2，L3 仍是启发式 | ① 中文样本全是 chirp3（TTS 合成），抢话率 4.6% 是乐观下界，D5 真人集未建；② L3 文本路（turn-detector INT8）未接；③ 短轮次（<2s）判停未测 | D5 中文真人集；必要时接 L3 |
| ~~JSONL 事件流落盘未做~~ → **已落地**（2026-09-27） | `metrics/jsonl.py`：每轮一行，含 waterfall 各点 + 双份文本；`VOICE_EVENTS_JSONL` 启用（默认关，零副作用） | 单进程内串行写；多 worker 部署需换集中式收集 | 真人验收与线上排查共用这条证据链 |
| ~~预合成 filler（语音短音）~~ → **已落地**（①，2026-10-17）：判停即播预合成短音，体感首音 6~8ms、口径B 不变（775↔788ms） | `core/filler.py` + `play_filler` + `scripts/make_filler.py`（片段 430~520ms，600ms 硬上限） | 见 `docs/优化空间-20261010.md` §三① | 与 P0.5/P1 同轮已闭环；片段时长是硬约束 |
| **WarmPool「预合成静音帧」变体未做**（语音 filler 已兑现，见上） | 仅语音 filler（①）；WarmPool 静音缓冲策略未做 | 静音缓冲变体：首音前少一档可优化的常量延迟（与 filler 同路线、不同素材） | 见技术方案 §13.4 |
| **媒体面 / 音色克隆未做** | 无 FreeSWITCH / WebRTC / 唇形；无 zipvoice 克隆 | 接入电话网与自定义音色不在本仓 | **范围外**，见下 |

**明确的范围外**（不要悄悄做进去）：商用抗噪与人声干扰、硬件 AEC、媒体面（FreeSWITCH / WebRTC / 唇形）、音色克隆（zipvoice）。

> ⚠️ **2026-09-27 两处修正**（原列表已过时）：
> ① **鉴权已做一半**——静态 token 鉴权已落地（`VOICE_WS_TOKEN`），仅 TLS/域名未上，故从「范围外」移到差距表；
> ② 并发曾列为范围外缺口，**2026-10-08 已落地**（判据 9 转 ✅），见差距表对应行；
> 剩余缺口收敛为**判据 4（TLS/公网开麦）**——判据 8（真人验收）已于 2026-10-10 闭环，见 `docs/验收-本机真人-20261010.md`。

## 发版检查单

- [x] 质量门禁四件套全绿：`ruff check` / `ruff format --check` / `mypy`(strict) / `pytest`（152 passed，2026-10-08 复跑）
- [x] CI 与本地同命令同结果（py3.10 / py3.12 矩阵）
- [x] LICENSE = MIT（见 `LICENSE`）
- [x] README：本机准备 / 快速开始 / 承诺映射表 / 对比定位 / 差距表
- [x] 技术方案：§13 本机实测、§14 M3 实现与诚实边界
- [x] sim 模式零模型零网络（CI 可跑）；real 用例缺模型自动跳过
- [x] **B 路线试点**：多轮对话 + 静态 token + 浏览器客户端 + 混合部署（ECS 口径B 1553.9ms）
- [x] 文档同步（2026-09-27）：技术方案 §2/§6/§9 选型与里程碑回填实际落地；背景与基线补第四层目标与 11 项成熟度判据；README 差距表补 4 项新缺口
- [x] Docker / compose（2026-10-03 实测：linux/amd64 构建通过，容器起跑 healthz/页面 200）
- [ ] `git push`（**本项目暂不 push**，只在本地验收）

## 许可

代码 MIT（见 LICENSE）。模型不进仓，由 `scripts/download_models.sh` 拉取；模型许可证在
各自的模型仓库，商用自查。
