"""finalize 自补尾部静音：流式 ASM 解尾字需要右侧上下文（P1 接线时挖出的潜伏契约）。

背景（2026-10-17，ASR 前置能量门落地时量到）：客户端在判停那一刻就可能停止上行，
能量门也会拦掉「确认静音」的帧——缓冲区里只剩语音没有静音时，
`input_finished()` **不能**把尾字解出来。隔离实验（同音频只切换门）：19 字句子的
尾段整段丢失（19→15 字）。修法：`StreamingAsr.finalize()` 在最后一段语音之后
自补 600ms 静音；对「本来就喂过静音」的路径转写一字不变（实测）。

零网络零模型：假 recognizer 只记录进流过什么、input_finished 何时发生——
钉住契约本身（尾部静音在语音之后、在 input_finished 之前），不碰真模型。
"""

from __future__ import annotations

from typing import cast

import numpy as np
import sherpa_onnx

from voice_pipeline.providers.sherpa import StreamingAsr

TAIL_MS = 600


class FakeStream:
    """记录进流音频的假识别流。input_finished 之后还喂音频 = 违约，直接炸。"""

    def __init__(self) -> None:
        self.fed: list[np.ndarray] = []
        self.finished = False

    def accept_waveform(self, sr: int, samples: np.ndarray) -> None:
        assert not self.finished, "input_finished 之后还往流里喂音频"
        self.fed.append(np.asarray(samples, dtype=np.float32))

    def input_finished(self) -> None:
        self.finished = True


class FakeRecognizer:
    """只记录、不解码的假 recognizer（is_ready 恒假，_drain 是空转）。"""

    def __init__(self) -> None:
        self.stream = FakeStream()

    def create_stream(self) -> FakeStream:
        return self.stream

    def get_result(self, stream: FakeStream) -> str:
        return "假装转写"

    def is_ready(self, stream: FakeStream) -> bool:
        return False

    def decode_stream(self, stream: FakeStream) -> None:
        pass


class _Asr(StreamingAsr):
    """只为把抽象方法填上；recognizer 一律从构造参数注入。"""

    def _make_recognizer(self) -> sherpa_onnx.OnlineRecognizer:  # pragma: no cover
        raise AssertionError("测试不该走模型工厂")


def _asr() -> tuple[_Asr, FakeStream]:
    rec = cast("sherpa_onnx.OnlineRecognizer", FakeRecognizer())
    asr = _Asr(sample_rate=16000, window_ms=100, recognizer=rec)
    return asr, cast("FakeStream", asr._stream)


def _speech(ms: int, amp: int = 8000) -> bytes:
    n = int(16000 * ms / 1000)
    return (amp.to_bytes(2, "little", signed=True)) * n


async def test_finalize_appends_tail_silence_after_speech() -> None:
    """finalize 必须在最后一段语音之后补 ≥600ms 静音——尾字靠它解出来。"""
    asr, stream = _asr()
    await asr.push_audio(_speech(300))  # 3 个整窗，无残余
    await asr.finalize()
    tail = stream.fed[-1]
    assert len(tail) >= int(16000 * TAIL_MS / 1000), "尾部静音不足 600ms"
    assert float(np.max(np.abs(tail))) == 0.0, "尾部必须是静音（全零）"


async def test_finalize_pads_even_without_any_audio() -> None:
    """一帧没喂就 finalize 也要补尾：契约不依赖调用方喂过静音。"""
    asr, stream = _asr()
    await asr.finalize()
    assert stream.fed, "finalize 什么都没进流"
    assert float(np.max(np.abs(stream.fed[-1]))) == 0.0
    assert stream.finished, "input_finished 必须被调用"


async def test_partial_buffer_flushed_before_tail() -> None:
    """残余的部分窗先在尾静音之前进流——顺序反了等于在语音中间插空洞。"""
    asr, stream = _asr()
    await asr.push_audio(_speech(50))  # 半个窗，留在缓冲区
    await asr.finalize()
    assert len(stream.fed) >= 2, "至少两段：残余语音 + 尾部静音"
    partial, tail = stream.fed[-2], stream.fed[-1]
    assert float(np.max(np.abs(partial))) > 0, "倒数第二段应是残余语音"
    assert float(np.max(np.abs(tail))) == 0.0, "最后一段应是尾部静音"


async def test_tail_silence_precedes_input_finished() -> None:
    """input_finished 必须在尾部静音之后（FakeStream 对违约顺序会炸）。"""
    asr, stream = _asr()
    await asr.push_audio(_speech(100))
    await asr.finalize()
    assert stream.finished and len(stream.fed) >= 2
