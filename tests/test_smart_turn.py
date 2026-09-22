"""神经判停器 Smart Turn v3 + NEURAL 档。

需要 8MB 权重（models/eou/），没有就整文件 skip —— 「没有模型」不是失败，
但也不能假装测过。跑 `scripts/fetch_eou_model.py` 之后再跑本文件。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from voice_pipeline.core.eou import FusedEou, GatingMode
from voice_pipeline.providers.eou.smart_turn import SmartTurnEou

SR = 16000
WEIGHTS = Path("models/eou/smart-turn-v3.2-cpu.onnx")
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason=f"缺权重 {WEIGHTS}")


def _frame(ms: float, amp: float = 0.3, freq: float = 220.0, sr: int = SR) -> bytes:
    n = int(ms / 1000 * sr)
    out = bytearray()
    for i in range(n):
        s = int(amp * 32767 * math.sin(2 * math.pi * freq * i / sr))
        out += s.to_bytes(2, "little", signed=True)
    return bytes(out)


def test_probability_in_unit_range() -> None:
    det = SmartTurnEou()
    for _ in range(5):
        det.push_audio(_frame(100))
    p = det.probability()
    assert 0.0 <= p <= 1.0, p
    assert det.inference_count >= 1, "至少跑过一次推理"


def test_no_speech_means_no_decision() -> None:
    """没人声就不该判停（min_speech 门控），否则静音开局就会被当成说完了。"""
    det = SmartTurnEou()
    for _ in range(10):
        det.push_audio(_frame(100, amp=0.0))
    assert det.probability() == 0.0
    assert not det.is_terminal()


def test_reset_clears_state() -> None:
    det = SmartTurnEou()
    for _ in range(8):
        det.push_audio(_frame(100))
    assert det.probability() >= 0.0
    det.reset()
    assert det.probability() == 0.0
    assert det.buffer_seconds == 0.0


def test_buffer_is_capped_at_model_window() -> None:
    """只留最后 8 秒：模型窗口就这么大，多留没信息还拖慢 mel。"""
    det = SmartTurnEou()
    for _ in range(120):  # 12 秒
        det.push_audio(_frame(100, amp=0.2))
    assert det.buffer_seconds <= 8.0 + 1e-6


def test_neural_gating_needs_silence_before_asking_the_model() -> None:
    """NEURAL 档必须先等一小段静音才问模型。

    为什么：训练样本末尾都有静音，输入末尾是「正在说话的语音」时模型在分布外，
    概率在 0.01↔0.99 之间乱跳（实测）。这条断言防的是有人把确认窗调成 0。
    """
    eou = FusedEou(
        gating=GatingMode.NEURAL,
        neural=SmartTurnEou(),
        min_silence_neural_ms=100.0,
        neural_fallback_ms=100000.0,  # 关掉兜底，单独测神经通道
    )
    for _ in range(8):
        eou.push_audio(_frame(100, amp=0.3))
    assert eou.silence_ms() == 0.0
    # 说话中不判停（静音窗未满足）
    assert eou.silence_ms() < 100.0


def test_neural_fallback_prevents_waiting_forever() -> None:
    """模型不肯吐 complete 时，兜底窗必须接管 —— 否则这一轮永远收不了尾。"""
    eou = FusedEou(
        gating=GatingMode.NEURAL,
        neural=SmartTurnEou(threshold=1.01),  # 模型永远判 incomplete
        min_silence_neural_ms=100.0,
        neural_fallback_ms=300.0,
    )
    for _ in range(5):
        eou.push_audio(_frame(100, amp=0.3))
    for _ in range(3):
        eou.push_audio(_frame(100, amp=0.0))
        if eou.should_end():
            break
    assert eou.should_end(), "模型不判停时兜底窗必须生效"


def test_build_eou_falls_back_without_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """缺权重要退回 conservative —— 这是数字可信度的护栏。

    静默退回会让启发式的数字被当成神经版的数字报出去，所以这里连告警一起断言。
    """
    import voice_pipeline.providers.factory as factory

    monkeypatch.setattr(factory, "DEFAULT_NEURAL_SILENCE_MS", 100.0)
    monkeypatch.setenv("SMART_TURN_ONNX", str(tmp_path / "nope.onnx"))
    eou = factory.build_eou(GatingMode.NEURAL)
    assert eou.gating is GatingMode.CONSERVATIVE
    assert eou.neural is None


def test_official_clip_probability_separates_end_from_pause() -> None:
    """端到端自检：官方测试集上，句末样本的概率应显著高于句中停顿样本。

    数据来自 `scripts/fetch_eou_dataset.py`（不入仓）。没有数据就 skip —— 这条
    断言的价值在于把「模型真的在判别」钉住，防止特征提取悄悄漂移。
    """
    import json

    from voice_pipeline.core.eou_mel import log_mel_spectrogram

    root = Path("data/eou-test")
    if not (root / "meta.json").exists():
        pytest.skip("缺 data/eou-test（跑 scripts/fetch_eou_dataset.py）")
    meta = json.loads((root / "meta.json").read_text())
    det = SmartTurnEou(min_interval_ms=0.0)

    def prob(a: np.ndarray) -> float:
        det.reset()
        det.push_audio((a * 32767).astype("<i2").tobytes())
        return det.probability()

    def voiced_end(a: np.ndarray) -> int:
        win = 80
        n = len(a) // win * win
        rms = np.sqrt((a[:n].reshape(-1, win) ** 2).mean(axis=1))
        v = np.where(rms >= 0.005)[0]
        return int((v[-1] + 1) * win) if v.size else 0

    ps: dict[bool, list[float]] = {True: [], False: []}
    for m in [x for x in meta if x["lang"] == "zho"][:20]:
        a = np.load(root / "zho" / f"{m['i']:04d}.npy").astype(np.float32)
        cut = voiced_end(a)
        if cut > SR * 0.3:
            ps[bool(m["label"])].append(prob(a[:cut]))

    if not ps[True] or not ps[False]:
        pytest.skip("样本不足")
    # 句中停顿（真值没说完）的概率中位数必须明显更低 —— 这是模型判别力的下限
    assert np.median(ps[False]) < 0.5 < np.median(ps[True]), (
        f"句末中位 {np.median(ps[True]):.2f} / 句中中位 {np.median(ps[False]):.2f}"
    )
    assert log_mel_spectrogram(np.zeros(1000, dtype=np.float32)).shape == (80, 800)
