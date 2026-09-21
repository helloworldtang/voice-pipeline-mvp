"""barge-in L1 检测：怎么知道「用户又开口了」。

这是三层 barge-in 的第一层——只负责「信号进来，判不判开口」这一个接口，换实现
不影响上层（Session / 状态机）。

本机用短时能量 + 帧去抖（连续几帧超过阈值才算开口，避免呼吸/噪声误触）。生产环境
应替换为 AEC + 流式 VAD 的输出——因为 AI 自己播放的声音会经扬声器/空气回流进麦克风，
纯能量检测会把它当成「用户开口」而疯狂打断。

回声问题不在本里程碑解决（见技术方案差距表：软件 AEC 列二期），一期依赖**入耳式耳机**
的物理隔离。也就是说：这里的能量检测在服务端 AI 播放期间对每帧上行音频调用，真实环境
必须有回声隔离，否则会自激。
"""

from __future__ import annotations

from voice_pipeline.core.eou import rms_of


class BargeInDetector:
    """能量阈值 + 连续帧去抖。可插拔：生产换 AEC+VAD 同接口。

    threshold : RMS 归一化阈值（满幅 1.0）。默认 0.02，正常说话远高于此才触发。
    frames    : 连续多少帧超阈值才判开口（去抖，默认 3 帧 @16k/20ms ≈ 60ms）。
    """

    def __init__(self, threshold: float = 0.02, frames: int = 3) -> None:
        self.threshold = threshold
        self.frames = frames
        self._above = 0

    def is_speech(self, chunk: bytes) -> bool:
        rms = rms_of(chunk)
        if rms >= self.threshold:
            self._above += 1
        else:
            self._above = 0
        return self._above >= self.frames

    def reset(self) -> None:
        """新一轮开始前清去抖计数，避免上轮尾音污染。"""
        self._above = 0
