# voice-pipeline-mvp 长期项目记忆

## 项目是什么
配套文章《给AI产品接上嘴——ASR+TTS接入实战，延迟从2秒压到300毫秒》（ailang 号，2026-09-04，文末挂本仓链接）的**准生产级参考实现**。文章此前是纸面交付，本仓把文章机制做成可跑代码、把评论区"300ms 是否可能"的争论做成可复现实验。
源文档：`技术方案.md`（方案/里程碑）、`背景与基线.md`（评论区存档/验收基线，ground truth，保持稳定）。

## 作者目标口径（2026-09-20 拍板）
**"300ms 返回" 且 "能真正的交流"**——两个都要，缺一不可。

### 关键诊断：300ms 的口径陷阱（重要，别再走回头路）
- 口径A（用户**开口**→AI首音）压到 300ms，数学上唯一手段是让 AI 在用户才说出第一个词时就出声 —— 那是抢答，不是交流。
- 所以「快」与「不抢话」**只能在口径B 下兼得**。
- **工程目标 = 口径B（用户说完→AI首音）300ms 级**；口径A 如实报告、不裁剪、不挑轮次。

## 进展
- **🔺 2026-10-17 链路追踪（TraceID + Span）落地**：`metrics/trace.py`（Tracer/Span/decompose/exclusive_times）
  + 主链路四处埋点 + `scripts/probe_trace.py`（判停后持续喂帧）+ `docs/链路追踪-根因分析.md`。
  **分格必须用 flame-graph 自有时间**（`llm.request` 嵌套 `tts.synth`，sibling 相加会 > 口径B），
  恒等式 `Σ自有 == 覆盖并集` 每轮自检。**新实测：纯本机（melo）口径B p50 1966ms、
  云流式 TTS 1029ms、kokoro 2759ms；判停 ~510ms 在口径B 外；未归因残差 0ms（无藏账）。**
  P0：默认 TTS 换云流式（省 ~930ms，抵其余优化之和）。三条踩坑（attrs 吞关键字 /
  reset_turn 清树丢判停前证据 / exclusive_times 边界未夹紧导致高估）见 2026-10-17.md，均有回归测试。
- **🔺 2026-10-17 晚 深拆（第二层 span：provider 内部时刻）**：`llm.headers`/`llm.prefill`（QwenLlm
  在 astream 记 perf 原值）+ `tts.connect`/`tts.first_byte`（DashScope SDK 回调里记），
  调度器用 `Tracer.record_perf`（**必须经 SessionClock.ms_at 时钟换算**）补记；
  `phase_span` 缺端点返回 None（宁空不填 0）。**口径B 1029ms = 服务端首字节 332 + 建连 257 +
  LLM 网关 231 + 首句 120 + 传输 31，残差 0。**
  **① 建连那 257ms 每句白付**（`DashScopeTts.synth()` 每句新建 SpeechSynthesizer，
  `_close_ws_after_use` 默认 True）：隔离探针首片 604 = 建连 259 + 服务端 338（TCP 握手仅 40ms）；
  **SDK 自带 `SpeechSynthesizerObjectPool`**（借出即预连）实测首片 340~375ms
  ⇒ **P0.5 = TTS 连接预热 −230~260ms，当时账面最大可省一笔**；或判停即预连（不依赖文本，可与 LLM 并行）。
  **✅ 2026-10-17 夜已落地（自建池，见下条）**。
  **② 换 LLM 模型救不了**：qwen-flash ttft 284ms 全平台最快；qwen3.7-flash 11342ms（2811 字思考链）、
  qwen3.8-flash 2803ms、deepseek-v4.1-flash 4059ms、step-3.7-flash 400。新"flash"全是推理型，
  只转发 content 的我们表现为"久不出声"。**llm.prefill 仅 0.1ms**（网关把等待并进响应头；
  `GET /models` ttfb 就要 208ms）。
  **③ Jev（TypeSafe 决策模型）顶替不了 LLM**：只出是/否/分类/分值，不生成自然语言；
  EOU 判停理论对口但 70~500ms + 跨洋 RTT 省不出延迟；唯一 plausible 槽 = barge-in 意图分类（二期）。
  无 key 未实测。探针：`probe_trace.py --llm-model X` / `probe_tts_firstchunk.py` /
  `report_turns.py --spans`（线上账单分格）。
- **🔺 2026-10-17 夜 P0.5 已落地：TTS 连接预热（自建池）**：`dashscope_tts.py` 重写为连接池
  （不用 SDK 的 `SpeechSynthesizerObjectPool`——cosyvoice-v1 硬编码、replenish 不受控），
  沿用其改名私有方法 `__connect/__reset/__update_params/__is_connected`，关键 `close_ws_after_use=False`。
  预热三处均非阻塞：建会话 / 判停 / 投机开跑（LLM headers ~230ms 即窗口）。
  池纪律：温连接已死且未出片→冷路径重试一次；出过片绝不重试；barge-in 事后回收轮询
  `callback.finished`（防 InvalidTask）；能力探测失败整条预热关闭退回冷路径。
  **实测（真 SDK A/B，n=6）：口径B p50 1029→724ms（−305ms），tts.connect 257→38ms，warm 6/cold 0，残差 0**；
  对照臂仅第 1 轮 1317ms（池跨轮复用）。地板 ≈**700ms**（服务端首字节 ~330 + LLM 网关 ~230 +
  首句 ~120 + 传输 ~30，判停另算在口径B 外）。
  **顺带修真 bug**：`call()` 抛 ConnectionError 时无回调投哨兵 → `queue.get()` 永久挂起
  （旧探针「偶发卡死」根因）→ `fail_sentinel` done callback。
  ⚠️ **语音 filler（§三①）已于 2026-10-17 落地+实测**（体感首音 p50 6ms、口径B 不变，但片段偏长会推迟真回答起点，需调短）；WarmPool「预合成静音帧」静音缓冲变体仍未做（README 差距表在列）——别把 P0.5 说成整个 WarmPool 兑现。
  测试 16 例（`test_tts_pool.py` 13 + `test_prewarm_wiring.py` 3）；A/B 命令见追踪文档 §八。
- **🔺 2026-10-17 P1 已落地：ASR 前置能量门**：`core/energy_gate.py`（`AsrEnergyGate`）接在
  `Session.on_audio` 的 ASR 之前——判停后客户端持续喂的静音帧不进 ASR。**门限 0.01
  （EOU ENERGY_THRESHOLD）而非 L1 的 0.02**（假阴性=丢语音更贵；不变式 gate≤L1 有单测）；
  滞后：进静音连续 3 帧、出静音任意 1 帧；**只 real 模式装门**（SimAsr 按 push 次数前进，
  拦帧会毁确定性回放）；**EOU 仍收全部帧**（静音是判停原料）；barge-in 后 reset。
  被拦帧打 span `a.gated=1`，`report_turns.py --spans` 可数。
  **实测（同机同时段 A/B n=6，两轮）：口径B p50 −97/−113ms（759 vs 872）**；判停 513ms
  两臂不变、premature/missed 全 0、残差 0。**归因诚实**：span 可直接指认的只有上行竞争
  −18~−27ms，其余是二阶调度提前（llm.headers 同向移动），不独占格不计独功。
  **顺带修两枚真 bug**：① L1 每帧双算 `is_speech`（`_above` 去抖计数翻倍，文档的 3 帧
  实际 2 帧触发）；② `finalize()` 缺尾部静音上下文丢尾字（隔离实验 19→15 字；门拦掉判停后
  追加的静音 → 转写 18/19；修法：自补 600ms 全零静音再 `input_finished()`，恢复 19/19）。
  ⚠️ **流式 ASR 的 `input_finished()` 解不出缺尾部静音的尾段**——任何「少喂静音」的优化
  （energy gate/早停）都必须配套 finalize 补静音，否则拿精度换延迟。
  测试 21 例（`test_energy_gate.py` 17 + `test_asr_finalize_tail.py` 4）；A/B 命令见追踪文档 §八。
- **M0 骨架已完成**（2026-09-21，commit 1d91f9d）：git + pyproject(uv/ruff/mypy strict/pytest) + CI(py3.10/3.12) + WS echo 服务 + sim 三替身 + 22 项测试全绿。
- **M1 serial 全链已完成**（commit d4d765f）：分句 + TTS 分片 + waterfall 双口径账单 + `scripts/demo_serial.py`；44 项测试全绿。
- **M2a real 接入 + 本机校准**（015cfd9）/ **M4 pipelined 投机调度**（33b3705）/ **M2b EOU 四路融合判停**（e07539c）均已完成。
  ⚠️ **2026-09-22 实测推翻 M4 的成绩口径**：「命中时 179ms」是 sim LLM + 挑最好一轮的产物。真 LLM（qwen-flash）下
  命中率 **1/24 = 4%**，pipelined p50 2491ms 比 serial 1840ms **慢 651ms** —— **投机是净负债，不是优化**。
  机制：投机要在「静音窗 + grace」内跑完 LLM 首完整句 522ms + TTS ~200ms，窗口不够；加大 grace 能买命中率，
  未命中时却是纯亏损。完整账在 `docs/实验报告.md`。**对外不要再报 179ms / 81ms / 534ms。**
- **M3 barge-in 三层 + 状态机 + live 客户端已完成**（本次）：`turn_fsm.py` + `barge_in.py` + `AudioStop` + WS 三路并发（接收/`run_turn`/`pump`，outbox 解耦）+ `scripts/live_client.py`；71 项测试全绿。细节与诚实边界见技术方案 **§14**。
- **M5 收尾发版已完成**（本地，未 push）：README 补齐「承诺映射表（逐项对齐真实文件）+ 对比定位 + 差距表 + 发版检查单」；LICENSE(MIT) 已在；技术方案 §5 加「设计意图≠实际落盘」差异说明、M5 行标 ✅。
- **⚠️ M5 诚实盘点发现的重要事实（勿再当"已完成"）**：① **real 模式 LLM 仍是 `SimLlm`**，真 DeepSeek 未接（端到端口径B 只是本机侧下界）；② scripted 客户端未做（§4.5 承诺 live+scripted）；③ JSONL 事件流落盘未做（§4.5 #4）；④ `DualPriorityQueue` 有实现有单测但**未接进 WS 主链路**（用无界 outbox）；⑤ 无 Dockerfile/compose；⑥ barge-in 无回声抑制（仅 L1 能量）、尾音未量化；⑦ ASR 前置 energy gate **已兑现**（2026-10-17 P1）、预合成 filler 语音版已落地（2026-10-17 ①）；WarmPool「预合成静音帧」静音缓冲变体仍未做（连接预热已兑现）；⑧ EOU L2/L3 为启发式。以上全部已列入 README 差距表。
- **全部里程碑（M0–M5）代码与文档完成**，本地 commit，**未 push**（用户要求先本地验收）。

## ⚠️ 2026-10 后续（本节最新，与上文的滞后部分冲突时以本节为准）
上文停在 2026-09-22，此后又走了很远：Docker/compose + systemd/caddy 部署成型、JSONL 账单落盘、
有界队列接进 WS 主链路、打断尾音量化、scripted 客户端、**真人开口验收**（2026-10-04）、**并发**（2026-10-08）。

- **成熟度判据：9✅ / 2⚠️ / 0❌**（判据表在 `背景与基线.md`）。原唯一 ❌ 的**并发已于 2026-10-08 闭环**。
  剩余 2 项：**判据 4 TLS/公网开麦**（需域名，caddy 配置已就绪）、**判据 8 真人验收**
  （回声根治后未重跑干净的一轮）。
- 🔺 **并发的口径（别报错）**：**并发是「多路不互相拖慢 + 显式背压」，不是「单路更快」**。
  实测 2 路：pooled 口径B p50 1628ms vs shared 1589ms（持平），总墙钟 8.76s vs 单路累加 15.6s（并行度 1.78x）。
- 🔺 **达成并发的前提不是资源独占，是把 ONNX 推理搬出事件循环**。只做独占不动线程时，
  双路每路 p50 **2163ms vs 共享 1653ms，独占净亏 510ms**（负面结果）。
  ⇒ 判据：asyncio 里跑 CPU 密集推理，独占资源救不了它，**先 `asyncio.to_thread` 再谈并行**。
- **租借池**：`providers/pool.py`，real 会话独占推理槽，满载 `session_limit` + close 1013 明确拒绝
  （**不静默退回共享**）；`VOICE_MAX_SESSIONS` 默认 2；`VOICE_SHARED_INFERENCE=1` 仅作 A/B 对照。
- **判据 8 为何是 ⚠️ 不是 ✅**：真人开口做过一次（2026-10-04 n=9）但被回声自激污染
  （6/9 轮 `user_text` 空、抢话 2/9），回声当日双门根治后**未重跑**；10-08 那 125 行账单是 scripted 跑的。
- **判停默认档 = conservative**（第四次定档）。⚠️ 该档位此前**四次**漏改过（factory.build_eou 是第四处），
  现有 `inspect` 测试钉住四层默认一致性；改档前先跑抢话三连测。
- **本仓库无 git remote**（`git remote -v` 空），所谓「上线」= 本地版本，无 push 环节。
- 门禁命令（四件套，均需全绿才可合并；漏跑 format 会 push 后 CI 红）：`uv run ruff check .` / **`uv run ruff format --check .`** / `uv run mypy` / `uv run pytest`。⚠️ 2026-09-21 M5 发现此前只跑前三项，导致 4 个文件未格式化、CI 必红；已补齐并统一修好。
- **本机（Apple M5）real 环境的两个坑（已写进 README「本机准备」）**：① sherpa-onnx wheel **不带 libonnxruntime.dylib**（cp312/cp314 皆缺）→ `uv add onnxruntime` 后软链到 `sherpa_onnx/lib/libonnxruntime.dylib`；② GitHub release 直连不通，模型一律走 **hf-mirror**（`./scripts/download_models.sh`）。
- 本机 Python 固定 **3.12**（`.python-version`）；uv 默认会拉 3.14。
- **EOU 判停**：原方案只有语义一路，2026-09-21 调研后补上**声学/韵律一路**（L2），改为四路（能量→VAD→韵律→语义）。依据与待决策点 D1–D5 见技术方案 §12。视觉/唇形属「说话人归属」层，不解决判停，不进一期。

## 已定事项（已拍板，勿回炉）
- **LICENSE = MIT**（2026-09-20 拍板）。
- **激进播放门控**：M2 接入语义判停前默认关（裸激进＝抢话），M2 之后应默认开 semantic 档。⚠️ 2026-09-21 复核推翻旧判据：它压的是**口径B**不是口径A，与「口径B 300ms」目标**一致且是必要前提**（等完整静音窗 ⇒ 口径B ≥500-700ms，300ms 不可达）。它与 EOU 语义判停是配套两端，不是二选一。配套指标 premature_eou_rate（过早判停率）文档缺失待补。**✅ 2026-09-22 已定档：默认切 `semantic`**（Session / factory.build_eou / FusedEou 三处默认同步改掉，test_protocol.py 有防回退断言）。⚠️ 换档依据是**机理不是中文实测**——D4 判错率曲线、D5 中文测试集仍未做，对外报告必须带这句。
  ⚠️ **2026-09-22 已修：semantic 档曾完全失效**（曾误记为「漏判信号/档位不稳」，归因错了，真因如下）：
  `ProsodyEou.coda_ms=250`（静音>250 就清终顿）与 `FusedEou.min_silence_short_ms=250`（静音≥250 才读终顿）
  **两个条件互斥**，100ms 帧粒度下永远采不到交集 → semantic 退化成等 2000ms 看门狗，判停延迟实测 1804ms。
  修法：终顿是**事件不是状态**，改为锁存最近一次成立时刻 + `terminal_window_ms`（默认 400）判窗口。
  修后：判停延迟 1804ms → 87~123ms（离线）/ 282~296ms（WS）；漏判 5/6 → 1/6。`scripts/probe_eou.py --trace` 可复现。
  另修：`run_turn` 在 `finish()` 前就 `eou.reset()` → `eou_latency_ms` 恒为 **0**（0 像正常值不像缺失，
  所以潜伏了整个里程碑）。改为 reset 前采集。**缺失值别写 0，写 -1。**
  ✅ **2026-09-22 抢话实测已定档（D3 真正闭环）：默认 = conservative，semantic 降级为实验档。**
  `scripts/probe_premature.py`（句内停顿 + 真 ASR partial，n=5 句）实测：
    conservative(500ms)  停顿 250/350ms → 抢话 0/5、0/5    判停延迟 515ms
    semantic(250ms)      停顿 250/350ms → 抢话 1/5、2/5    判停延迟 330ms
    aggressive(120ms)    停顿 250/350ms → 抢话 5/5、5/5    判停延迟 220ms
  把 semantic 的 `min_silence_short_ms` 扫到 500ms 能回到 0% 抢话，但延迟 520ms ≈ conservative
  → **同安全下 semantic 零收益**。⇒ **300ms 与「不抢话」在本机启发式判据下不可兼得**。
  🔺 **真因（比调参重要）**：`ProsodyEou._terminal` 是**黏性锁存**——说话中任意一帧能量跌到
  峰值 40% 以下就永久置真（只被 >250ms 静音清除），故 `is_terminal()` 几乎恒为真 →
  semantic 的「静音≥250 **且** 终顿」退化成「静音≥250」≈ aggressive。逗号结尾短语在韵律上
  本就是句末降调，只能靠语义区分，而本机 L3 是启发式分不出。**要 semantic 可用，前提是
  换掉 L2/L3（D2 神经判停器）或拿到真人测试集定标（D5）。**
  🔺 **测量教训（可复用）**：合成语音自带句尾/句首静音 160~345ms，不裁边的话「插入 250ms
  停顿」实际是 410~595ms，会把 conservative 的 500ms 阈值喂成「抢话 60%」的假结论。
  **探针必须自带自检**（脚本已输出「插入 0ms 时的实测间隙」）。
- **live 麦克风客户端 = 一期做**，依赖入耳式耳机物理隔离回声；软件 AEC 列差距表二期。
- **EOU 语义判停 = M2 主线**（LiveKit turn-detector INT8），不是升级位。
- 里程碑（编号不变，内容加重）：M2 = real 接入 + 本机校准 + EOU 判停；M3 = barge-in 三层 + live 客户端。
- **WS 并发不变量（M3 起，勿退化回串行）**：接收循环 / `run_turn` / `pump` 三路并发，中间用 outbox 队列解耦。串行模型下播音期间收不到上行帧，barge-in 永远触发不了。**关连接前必须 `await outbox.join()` 等 pump 排空**（pump 每发一项调 `task_done()`）——否则 `finally` 的 `pump_task.cancel()` 会吞掉 `turn.end`，客户端表现为**无报错挂死**。pytest 已加 `timeout=30` 把挂起变成明确失败。
- **pipelined 必须在 wire 枚举里（2026-09-22 补齐）**：`SessionBegin.mode` 原先只有 `echo|serial|real`，
  `pipelined` 只存在于 Session，导致唯一能逼近 300ms 的实验组只能跑 bench 脚本，与「A/B 打在真实服务链路上」相违。
  现枚举 = echo/serial/pipelined/real/real-serial/real-pipelined，`gating`/`grace_ms` 一并上 wire 并由 app.py 透传给 Session。
  **判据：账单里出现 `spec_hit` 键 = 走的确实是 PipelinedScheduler**（serial 的 TurnEnd 不带它），test_ws.py 据此断言。

## 文档里的坑
- 「M5 本机校准」的 M5 指 **Apple M5 芯片**，不是里程碑 M5 —— 同字冲突，已全文消歧为「本机（Apple M5）校准」。

## 技术栈定稿（勿再调研）
FastAPI+WS 自研（不做 pipecat 二开）/ sherpa-onnx：ASR 流式 Paraformer int8(226MB) + TTS kokoro-int8(147MB,24kHz) + VAD silero int8 / LLM DeepSeek 流式可插拔 / Python 3.10+ uv asyncio / M5 Mac 32GB。
 FunASR pip 在 Mac 有硬伤（依赖 2-4GB、MPS 不可用、C++服务无 macOS 版），已降为备选——不要再重新评估。

## 范围外（如实列入差距表，不要悄悄做进去）
商用抗噪/人声干扰、并发压测、硬件 AEC、鉴权/TLS、媒体面（FreeSWITCH/WebRTC/唇形）、音色克隆(zipvoice)。
