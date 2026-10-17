# voice-pipeline-mvp 长期项目记忆

## 项目是什么
配套文章《给AI产品接上嘴——ASR+TTS接入实战，延迟从2秒压到300毫秒》的**准生产级参考实现**。
源文档：`技术方案.md`（方案/里程碑/需求，根目录）、`背景与基线.md`（评论区存档/验收基线，ground truth，保持稳定）。

## 作者目标口径（2026-09-20 拍板）
**"300ms 返回" 且 "能真正的交流"**——两个都要。关键诊断：**口径A（开口→首音）压到 300ms 只能抢答，不是交流**；"快"与"不抢话"**只能在口径B（说完→首音）下兼得**。工程目标 = 口径B 300ms 级；口径A 如实报告不裁剪。

## 里程碑与优化落地（本地 commit，未 push；本仓无 git remote）
| 项 | 内容 | 关键数字 / 文件 | 状态 |
|---|---|---|---|
| M0–M5 | 骨架/serial/real接入/EOU四路/barge-in三层/live客户端/收尾发版 | 见 `技术方案.md` §9 | ✅ 代码+文档完成 |
| P0 | 默认 TTS 换云流式 | 口径B 1966(melo)→1029(云)ms | ✅ |
| P0.5 | TTS 连接预热（自建池） | 口径B p50 1029→724ms；`tts.connect` 257→38ms；`dashscope_tts.py` | ✅ 2026-10-17 |
| P1 | ASR 前置能量门 | 口径B p50 −97/−113ms；顺带修 L1 双调用 + finalize 缺尾静音丢尾字 | ✅ 2026-10-17 |
| ① filler | 预合成 filler 激活+实测 | 体感首音 p50=6ms、口径B 不变；片段需≤400ms | ✅ 2026-10-17 |
| ② TTS 首片拆解 | 对照实验 | p50=404ms（650ms 是 P0.5 前基线）；当前 provider 无可行收益 | ✅ 2026-10-17 |
| ③ 判停尾部 | 噪声毛刺清零静音窗→看门狗；双阈值修复 | 最差 2000/2100ms→~500ms；`core/eou.py`+`test_eou.py` 正反两测 | ✅ 2026-10-17 |

## 已定事项（勿回炉）
- **LICENSE = MIT**。
- **判停默认档 = conservative**（第四次定档，2026-10-04 真人验收：neural 半句停顿抢话 3/3）。`test_default_gating_is_conservative_across_layers` 钉四层一致；改档前先跑抢话三连测。
- **live 麦克风客户端 = 一期做**，靠入耳式耳机物理隔离回声；软件 AEC 列差距表二期。
- **EOU 语义判停 = M2 主线**；L2/L3 为启发式（D2 神经版 / D5 中文集才救得了 semantic 档）。
- **WS 并发不变量**：接收/`run_turn`/`pump` 三路并发，outbox 解耦；关连接前必须 `await outbox.join()`。
- **pipelined 必须在 wire 枚举里**（benchmark 也要打在真实链路上）；`spec_hit` 键 = 走 PipelinedScheduler。
- **M4 投机是净负债**（真 LLM 命中 4%、p50 比串行慢 651ms）——对外不再报 179/81/534ms。
- **全链路可追溯纪律**：任一数字要么可复现、要么标「基线/不可复现」；文档与代码冲突时以测量为准、文档改口（§二 的 652ms→404ms 即此）。

## 工程纪律（踩过的坑，可复用）
- **双阈值不变式**：能量门 P1 `gate.threshold(0.01) ≤ L1 barge-in(0.02)`；EOU 静音重置 `silence_reset_threshold(0.02) ≥ energy_threshold(0.01)`——都与「什么样算说话」同一把尺，不与打断层打架。
- **口径B 纪律**：filler 只打 `first_sound` 不动口径B；任何"省延迟"若只盖住感知不算进口径B，否则冒领。
- **mock 保真度**：`test_tts_pool` 假件必须对齐真 SDK 完成事件顺序（`complete_event.set()` 先于 `on_complete()` 回调，`__cleanup_task` 不碰 complete_event）。
- **流式 ASR 缺尾静音丢尾字**：`finalize()` 须自补 ~600ms 全零静音再 `input_finished()`（隔离 19→15 字）。
- **判停测量**：`eou_latency_ms` 在 `reset()` 前采集；缺失值写 -1 不写 0。
- **沙箱 TMPDIR 限制**：pytest 偶发 `tmp_path` PermissionError → 用 `--basetemp=.pytest_tmp` 绕过（环境问题非代码问题）。
- **Bash grep 静默空 ≠ 无匹配**：凡 grep 得空一律用 Grep 工具复核。
- **门禁四件套**（漏跑 format 会 CI 红）：`uv run ruff check .` / `uv run ruff format --check .` / `uv run mypy` / `uv run pytest`（TMPDIR 用 --basetemp）。
- **本机(M5) real 两坑**：sherpa-onnx wheel 不带 libonnxruntime.dylib → 软链；GitHub release 直连不通 → 模型走 hf-mirror。Python 固定 3.12（`.python-version`）。
- **并发前提**：asyncio 里跑 CPU 密集推理先 `asyncio.to_thread`，独占资源救不了（双路每路 p50 2163 vs 共享 1653ms，独占净亏）。

## 技术栈 / 范围外
- 技术栈：FastAPI+WS 自研 / sherpa-onnx（Paraformer int8 + kokoro-int8 + silero VAD）/ LLM 流式可插拔 / Python 3.12 uv。FunASR 在 Mac 有硬伤，已降备选。
- 范围外（差距表，不悄悄做）：商用抗噪/人声干扰、并发压测、硬件 AEC、鉴权/TLS、媒体面、音色克隆。

## 成熟度判据（2026-10 现状）
9✅ / 2⚠️ / 0❌。剩余 ⚠️：判据 4 TLS/公网开麦（需域名，caddy 已就绪）、判据 8 真人验收（回声根治后未重跑干净一轮）。并发(原唯一❌) 已于 2026-10-08 闭环。
仍待办（M5 盘点遗留）：真 DeepSeek LLM 未接（端到端口径B 只是本机侧下界）、软件 AEC/意图分类（差距表）。
