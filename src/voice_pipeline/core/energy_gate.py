"""ASR 前置能量门（P1）：确认是静音的帧不推 ASR。

为什么需要它：判停之后客户端（真实麦克风）不会停止上行——每 100ms 一帧的静音照样
到达服务端。链路追踪实测（`docs/链路追踪-根因分析.md` §三「上行不是免费的」）：
持续喂帧时，每帧 `np.concatenate` + `asyncio.to_thread` 派发 ≈ 2.7ms 的事件循环
时间落在**口径B 窗口内**（p50 41~52ms、最差一轮 159ms）——它与 LLM/TTS 的下行
抢同一个事件循环，是唯一还能从循环里抠出来的一格。把它拦在 ASR 之前，下行就少一个
竞争者。

**阈值为什么是 0.01 而不是 barge-in L1 的 0.02**：门的假阴性代价是「丢语音」——
直接用 L1 的 0.02 会把 RMS≈0.018 的轻声整段拦在 ASR 外（本仓测试里就有这种帧，
见 `test_jsonl_metrics._frame(500)`），识别直接失败；而 L1 用 0.02 是因为它的假
阳性代价是「误打断」，两个方向的风险完全不同，阈值不该同一个数。约束取另一个方向：
**门限必须 ≤ L1 门限**——L1 认定的人声一定要能进 ASR，否则「L1 说在说、门说静音」
互相抵消（追踪文档 §六 P1 行的原话）。0.01 是 EOU L0 的「有人声」阈
（`eou.ENERGY_THRESHOLD`），本仓对人声的基准定义，复用它而不是发明第三个数字。

进出不对称（滞后）：
- **进静音要慢**：连续 `frames` 帧低于阈值才确认静音。句中停顿/换气（几百 ms 量级）
  在确认之前照常推 ASR，音频连续性不受影响。
- **出静音要快**：任意一帧超过阈值立即恢复推流（计数清零），不等人声确认——
  丢语音的代价远比多推一帧静音大。

⚠️ EOU 判停器**不受门的影响**：它仍然收到每一帧（静音正是它判「说完没有」的原料）。
门只拦 ASR 一条路。
"""

from __future__ import annotations

from voice_pipeline.core.eou import ENERGY_THRESHOLD, rms_of


class AsrEnergyGate:
    """确认静音才拦截的能量门。可插拔：生产换 AEC+VAD 输出时整个类换掉。

    threshold : RMS 归一化阈值（满幅 1.0）。默认 0.01 = EOU 人声阈，见模块 docstring。
    frames    : 连续多少帧低于阈值才确认静音（默认 3）。真实语音 RMS 通常 0.05~0.3，
                静音=0——0.01 距真实语音还有一个数量级，距静音也还有一个数量级。
    """

    def __init__(self, threshold: float = ENERGY_THRESHOLD, frames: int = 3) -> None:
        self.threshold = threshold
        self.frames = frames
        self._below = 0
        self._silent = False
        self.passed = 0
        self.blocked = 0

    @property
    def is_silent(self) -> bool:
        """当前是否处于「已确认静音」状态（诊断用；不参与判定）。"""
        return self._silent

    def allows(self, chunk: bytes) -> bool:
        """这一帧该不该推 ASR。有内部状态，同一帧只能调一次。"""
        if rms_of(chunk) >= self.threshold:
            # 出静音要快：任意一帧有人声，立刻恢复推流
            self._below = 0
            self._silent = False
            self.passed += 1
            return True
        self._below += 1
        if self._below >= self.frames:
            self._silent = True
        if self._silent:
            self.blocked += 1
            return False
        self.passed += 1
        return True

    def reset(self) -> None:
        """新一轮/一次打断后清状态。计数（passed/blocked）是会话级诊断量，不清。"""
        self._below = 0
        self._silent = False
