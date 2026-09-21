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
- **M0 骨架已完成**（2026-09-21，commit 1d91f9d）：git + pyproject(uv/ruff/mypy strict/pytest) + CI(py3.10/3.12) + WS echo 服务 + sim 三替身 + 22 项测试全绿。
- **M1 serial 全链已完成**（commit d4d765f）：分句 + TTS 分片 + waterfall 双口径账单 + `scripts/demo_serial.py`；44 项测试全绿。
- **M2a real 接入 + 本机校准**（015cfd9）/ **M4 pipelined 投机调度**（33b3705，命中时口径B **179ms**）/ **M2b EOU 四路融合判停**（e07539c）均已完成。
- **M3 barge-in 三层 + 状态机 + live 客户端已完成**（本次）：`turn_fsm.py` + `barge_in.py` + `AudioStop` + WS 三路并发（接收/`run_turn`/`pump`，outbox 解耦）+ `scripts/live_client.py`；71 项测试全绿。细节与诚实边界见技术方案 **§14**。
- **剩余**：仅 M5 收尾发版（README 承诺映射表/对比定位/差距表 + LICENSE + 发版检查单）。
- 门禁命令：`uv run ruff check .` / `uv run mypy` / `uv run pytest`（均需全绿才可合并）。
- **本机（Apple M5）real 环境的两个坑（已写进 README「本机准备」）**：① sherpa-onnx wheel **不带 libonnxruntime.dylib**（cp312/cp314 皆缺）→ `uv add onnxruntime` 后软链到 `sherpa_onnx/lib/libonnxruntime.dylib`；② GitHub release 直连不通，模型一律走 **hf-mirror**（`./scripts/download_models.sh`）。
- 本机 Python 固定 **3.12**（`.python-version`）；uv 默认会拉 3.14。
- **EOU 判停**：原方案只有语义一路，2026-09-21 调研后补上**声学/韵律一路**（L2），改为四路（能量→VAD→韵律→语义）。依据与待决策点 D1–D5 见技术方案 §12。视觉/唇形属「说话人归属」层，不解决判停，不进一期。

## 已定事项（已拍板，勿回炉）
- **LICENSE = MIT**（2026-09-20 拍板）。
- **激进播放门控**：M2 接入语义判停前默认关（裸激进＝抢话），M2 之后应默认开 semantic 档。⚠️ 2026-09-21 复核推翻旧判据：它压的是**口径B**不是口径A，与「口径B 300ms」目标**一致且是必要前提**（等完整静音窗 ⇒ 口径B ≥500-700ms，300ms 不可达）。它与 EOU 语义判停是配套两端，不是二选一。配套指标 premature_eou_rate（过早判停率）文档缺失待补。
- **live 麦克风客户端 = 一期做**，依赖入耳式耳机物理隔离回声；软件 AEC 列差距表二期。
- **EOU 语义判停 = M2 主线**（LiveKit turn-detector INT8），不是升级位。
- 里程碑（编号不变，内容加重）：M2 = real 接入 + 本机校准 + EOU 判停；M3 = barge-in 三层 + live 客户端。
- **WS 并发不变量（M3 起，勿退化回串行）**：接收循环 / `run_turn` / `pump` 三路并发，中间用 outbox 队列解耦。串行模型下播音期间收不到上行帧，barge-in 永远触发不了。**关连接前必须 `await outbox.join()` 等 pump 排空**（pump 每发一项调 `task_done()`）——否则 `finally` 的 `pump_task.cancel()` 会吞掉 `turn.end`，客户端表现为**无报错挂死**。pytest 已加 `timeout=30` 把挂起变成明确失败。

## 文档里的坑
- 「M5 本机校准」的 M5 指 **Apple M5 芯片**，不是里程碑 M5 —— 同字冲突，已全文消歧为「本机（Apple M5）校准」。

## 技术栈定稿（勿再调研）
FastAPI+WS 自研（不做 pipecat 二开）/ sherpa-onnx：ASR 流式 Paraformer int8(226MB) + TTS kokoro-int8(147MB,24kHz) + VAD silero int8 / LLM DeepSeek 流式可插拔 / Python 3.10+ uv asyncio / M5 Mac 32GB。
 FunASR pip 在 Mac 有硬伤（依赖 2-4GB、MPS 不可用、C++服务无 macOS 版），已降为备选——不要再重新评估。

## 范围外（如实列入差距表，不要悄悄做进去）
商用抗噪/人声干扰、并发压测、硬件 AEC、鉴权/TLS、媒体面（FreeSWITCH/WebRTC/唇形）、音色克隆(zipvoice)。
