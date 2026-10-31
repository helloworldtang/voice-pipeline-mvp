"""real provider: sherpa-onnx models (ASR / TTS).

sim 与 real 共用 providers/base.py 的协议接口，A/B 实验只换 build 工厂，不换调用方。
模型由 scripts/download_models.sh 拉到仓库根的 models/（不进仓）。

ASR 有两个引擎，各自的「出词粒度」不同——这是延迟预算里最硬的一块，必须由
scripts/calibrate.py 实测，不能引用厂商宣传值。
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import sherpa_onnx
import soxr

from voice_pipeline.providers.base import AsrProvider, TtsProvider

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[3] / "models"
DEFAULT_ESPEAK_DATA = "/opt/homebrew/share/espeak-ng-data"

# 部署覆盖（Linux/ECS 无 Homebrew 路径；模型可挂载到任意目录）：
#   VOICE_MODEL_ROOT=/path/to/models  VOICE_ESPEAK_DATA=/usr/lib/x86_64-linux-gnu/espeak-ng-data
MODEL_ROOT_ENV = "VOICE_MODEL_ROOT"
ESPEAK_DATA_ENV = "VOICE_ESPEAK_DATA"


def _model_root(root: Path | str | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get(MODEL_ROOT_ENV)
    return Path(env) if env else DEFAULT_MODEL_ROOT


def _espeak_data(data_dir: str | None = None) -> str:
    if data_dir is not None:
        return data_dir
    return os.environ.get(ESPEAK_DATA_ENV, DEFAULT_ESPEAK_DATA)


def _models(root: Path | str | None = None) -> Path:
    path = _model_root(root)
    if not path.is_dir():
        msg = f"model dir missing: {path} -- run scripts/download_models.sh first"
        raise FileNotFoundError(msg)
    return path


def pcm16_bytes(samples: np.ndarray) -> bytes:
    # clip 防回绕（与 xfyun_tts._encode 同一纪律）：soxr 重采样过冲/高幅样本
    # ×32767 会溢出 int16——溢出是静默回绕，听感为爆音（2026-11-01 审查缺陷 8）。
    return (
        (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0)
        .astype("<i2")
        .tobytes()
    )


def pcm16_to_f32(chunk: bytes) -> np.ndarray:
    return np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0


class StreamingAsr(ABC):
    """流式 ASR 公共壳：字节流转 16k float32，按窗口喂给底层识别器。

    fed_samples / wall_ms 是校准脚本要读的量：喂了多少音频、烧了多少 CPU。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_ms: int = 100,
        *,
        recognizer: sherpa_onnx.OnlineRecognizer | None = None,
    ) -> None:
        self._sr = sample_rate
        self._window = int(sample_rate * window_ms / 1000)
        # 流操作互斥锁：push/finalize 搬到线程后，pipelined 的投机任务可能在
        # recognize 还在 worker 线程里跑时同步调 text()（get_result 同一条 C++ 流）。
        # ORT 推理会放 GIL，跨线程读写同一条流是真实的数据竞争。锁无争用时
        # 开销可忽略。
        self._stream_lock = threading.Lock()
        # recognizer 携带模型权重（重，进程内共享）；stream 是纯会话状态（轻，每连接新建）。
        # 跨连接共享 recognizer 意味着并发推理在同一 ORT session 上串行——单用户试点无碍。
        self._recognizer = recognizer or self._make_recognizer()
        self._stream = self._recognizer.create_stream()
        self._buf = np.empty(0, dtype=np.float32)
        self.frames = 0
        self.fed_samples = 0
        self.wall_ms = 0.0

    @abstractmethod
    def _make_recognizer(self) -> sherpa_onnx.OnlineRecognizer: ...

    @property
    def fed_ms(self) -> float:
        return self.fed_samples / self._sr * 1000.0

    def text(self) -> str:
        with self._stream_lock:
            return self._text_locked()

    def _text_locked(self) -> str:
        result = self._recognizer.get_result(self._stream)
        raw = result if isinstance(result, str) else result.text
        # 去掉 sentencepiece 词首标记与所有空格，便于与参考文本比对
        clean = raw.replace("\u2581", " ")
        return "".join(clean.split())

    def text_nonempty(self) -> bool:
        return bool(self.text())

    async def push_audio(self, chunk: bytes) -> None:
        """认（decode）必须在**线程**里跑。

        为什么搬出事件循环：这是 CPU 活（几十 ms 量级 × 每 100ms 一帧），留在循环里
        会把所有连接一起堵住——那时「给每个会话一份独占的 recognizer」就没有意义了：
        它根本并行不起来，只是把一份守序的排队换成了两份互相抢 CPU 的排队。
        实测依据：2026-10-08 并发 A/B，只做资源独占、不动这一行，双路并发下**每路**
        口径B 反而比共享慢 190ms（n=6，两组分布不重叠）。

        **顺序性由调用方保证**：传输层是 `await session.on_audio(...)` 一帧一帧串着
        调的，同一个会话不会有两帧同时跑进来，因此这里不需要额外加锁。
        """
        start = time.perf_counter()
        self._buf = np.concatenate([self._buf, pcm16_to_f32(chunk)])
        self.fed_samples += len(chunk) // 2

        def _recognize() -> None:
            with self._stream_lock:
                while len(self._buf) >= self._window:
                    block, self._buf = self._buf[: self._window], self._buf[self._window :]
                    self._stream.accept_waveform(self._sr, block)
                    self.frames += 1
                    self._drain()

        await asyncio.to_thread(_recognize)
        self.wall_ms += (time.perf_counter() - start) * 1000.0

    def _drain(self) -> None:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    async def finalize(self) -> str:
        """收尾：冲刷缓冲区并终结识别流。

        **自补 600ms 尾部静音**（2026-10-17 P1 接线时发现）：流式 Paraformer 解尾字
        需要右侧上下文——客户端在判停那一刻就可能停止上行（ASR 前置能量门也会拦掉
        确认静音的帧），缓冲区里只剩语音没有静音时，`input_finished` 并不能把尾字
        解出来：隔离实验（同音频只切换门）19 字句子的尾段「合出门嘛」整段丢失
        （19→15 字）。补 600ms 静音后完整解出，且对「本来就喂过静音」的路径无影响
        （转写一字不变）。成本 = 多解 6 个静音窗（~1ms），在口径B 窗口内，
        相对正确性可忽略。
        """

        def _close() -> None:
            with self._stream_lock:
                if len(self._buf) > 0:
                    self._stream.accept_waveform(self._sr, self._buf)
                    self._buf = np.empty(0, dtype=np.float32)
                    self._drain()
                # 尾部静音必须在最后一段语音**之后**进流（顺序反了等于在语音中间
                # 插空洞）。接在 input_finished 之前，让解码器带着右侧上下文收尾。
                tail = np.zeros(int(self._sr * 0.6), dtype=np.float32)
                self._stream.accept_waveform(self._sr, tail)
                self._stream.input_finished()
                self._drain()

        await asyncio.to_thread(_close)
        return self.text()

    def reset(self) -> None:
        with self._stream_lock:
            self._stream = self._recognizer.create_stream()
            self._buf = np.empty(0, dtype=np.float32)
            self.frames = 0
            self.fed_samples = 0
            self.wall_ms = 0.0


class ParaformerStreamingAsr(StreamingAsr):
    """Streaming Paraformer 中英双语 int8。

    实测（本机 M5）：partial 每 ~600ms 更新一次，与上层喂帧窗口无关（10/32/100ms
    三档喂法结果一致）—— 600ms 是模型导出时的固有块大小，运行时改不掉。
    """

    def __init__(
        self,
        model_root: Path | str | None = None,
        num_threads: int = 2,
        sample_rate: int = 16000,
        window_ms: int = 100,
        recognizer: sherpa_onnx.OnlineRecognizer | None = None,
    ) -> None:
        self._root = _models(model_root) / "asr"
        self._threads = num_threads
        super().__init__(sample_rate=sample_rate, window_ms=window_ms, recognizer=recognizer)

    def _make_recognizer(self) -> sherpa_onnx.OnlineRecognizer:
        return sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=str(self._root / "encoder.int8.onnx"),
            decoder=str(self._root / "decoder.int8.onnx"),
            tokens=str(self._root / "tokens.txt"),
            num_threads=self._threads,
            sample_rate=self._sr,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,
        )


class TransducerStreamingAsr(StreamingAsr):
    """Streaming Zipformer transducer（14M, zh, int8）。

    与 paraformer 的可比口径：出词粒度、RTF、CER，都由校准脚本实测。
    """

    def __init__(
        self,
        model_root: Path | str | None = None,
        subdir: str = "asr-zipformer",
        num_threads: int = 2,
        sample_rate: int = 16000,
        window_ms: int = 100,
        recognizer: sherpa_onnx.OnlineRecognizer | None = None,
    ) -> None:
        self._root = _models(model_root) / subdir
        self._threads = num_threads
        super().__init__(sample_rate=sample_rate, window_ms=window_ms, recognizer=recognizer)

    def _make_recognizer(self) -> sherpa_onnx.OnlineRecognizer:
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(self._root / "encoder.int8.onnx"),
            decoder=str(self._root / "decoder.int8.onnx"),
            joiner=str(self._root / "joiner.int8.onnx"),
            tokens=str(self._root / "tokens.txt"),
            num_threads=self._threads,
            sample_rate=self._sr,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,
        )


class BaseTts(ABC):
    """TTS 公共壳：上层只认 sample_rate 与 synth()，具体的声学模型由子类给定。

    为什么要把 TTS 做成可换引擎：本机实测 TTS 是整条链路最大的单项开销，
    引擎之间能差 2 倍以上，这个选择直接决定 300ms 能不能成立。
    """

    def __init__(
        self,
        chunk_ms: int = 200,
        speaker_id: int = 0,
        speed: float = 1.0,
        out_sample_rate: int = 24000,
    ) -> None:
        self._chunk_ms = chunk_ms
        self.speaker_id = speaker_id
        self.speed = speed
        self.out_sample_rate = out_sample_rate
        self._tts = self._make_engine()
        self.native_sample_rate = int(self._tts.sample_rate)
        # 下行契约固定 24k（见技术方案 §音频参数）。引擎原生采样率不是 24k 时，
        # 在 provider 内重采样，不让调用方各自为政。
        self.resample_ms = 0.0
        self.sample_rate = out_sample_rate

    @abstractmethod
    def _make_engine(self) -> sherpa_onnx.OfflineTts: ...

    def generate(self, sentence: str) -> tuple[np.ndarray, int]:
        audio = self._tts.generate(sentence, sid=self.speaker_id, speed=self.speed)
        samples = np.asarray(audio.samples, dtype=np.float32)
        sr = int(audio.sample_rate)
        if sr != self.out_sample_rate:
            start = time.perf_counter()
            samples = soxr.resample(samples, sr, self.out_sample_rate).astype(np.float32)
            self.resample_ms += (time.perf_counter() - start) * 1000.0
            sr = self.out_sample_rate
        return samples, sr

    def synth_blocking(self, sentence: str) -> bytes:
        """同步阻塞合成：**必须在线程池里调用**，否则会堵死事件循环。"""
        samples, sr = self.generate(sentence)
        return pcm16_bytes(samples)

    async def synth(self, sentence: str) -> AsyncIterator[bytes]:
        pcm = await asyncio.to_thread(self.synth_blocking, sentence)
        step = int(self.out_sample_rate * self._chunk_ms / 1000) * 2
        for off in range(0, len(pcm), step):
            yield pcm[off : off + step]

    def first_chunk_latency_ms(self, sentence: str) -> tuple[float, int, int]:
        """Measure wall time until one downlink chunk worth of audio exists.

        计时从 generate 调用开始（不是从第一次回调开始）。本机实测 sherpa-onnx
        的 kokoro 只在合成**结束时**回调一次、一次交付全部样本，因此这里拿到的
        值 ≈ 整句合成耗时：这就是「等 TTS 出第一个可播分片」的真实代价。
        返回 (latency_ms, 到手样本数, sample_rate)。
        """
        target = int(self.sample_rate * self._chunk_ms / 1000)
        state: dict[str, float] = {"samples": 0, "latency": -1.0}

        def callback(chunk: np.ndarray, _progress: float) -> int:
            if state["latency"] < 0:
                state["samples"] += int(np.asarray(chunk).size)
                if state["samples"] >= target:
                    state["latency"] = (time.perf_counter() - t0[0]) * 1000.0
                    return 1  # 够了就停，只测首片
            return 0

        t0 = [time.perf_counter()]
        self._tts.generate(sentence, sid=self.speaker_id, speed=self.speed, callback=callback)
        return state["latency"], int(state["samples"]), self.sample_rate

    def callback_profile(self, sentence: str) -> list[tuple[int, float]]:
        """回调画像：每次回调交付多少样本、在什么时刻。用于验证是否真的有增量流。"""
        calls: list[tuple[int, float]] = []
        t0 = time.perf_counter()

        def callback(chunk: np.ndarray, _progress: float) -> int:
            calls.append((int(np.asarray(chunk).size), (time.perf_counter() - t0) * 1000.0))
            return 0

        self._tts.generate(sentence, sid=self.speaker_id, speed=self.speed, callback=callback)
        return calls


class KokoroTts(BaseTts):
    """kokoro-multi-lang v1.1 int8，24kHz 原生，多音色，音质最好。

    代价：RTF ~0.6，且整句一次性产出（回调只触发一次），
    首音必须等整句合成完。本机实测一个字「好」也要 636ms。
    """

    def __init__(
        self,
        model_root: Path | str | None = None,
        data_dir: str | None = None,
        lang: str = "zh",
        speaker_id: int = 0,
        speed: float = 1.0,
        num_threads: int = 4,
        chunk_ms: int = 200,
    ) -> None:
        self._root = _models(model_root)
        self._data_dir = _espeak_data(data_dir)
        self._lang = lang
        self._threads = num_threads
        super().__init__(chunk_ms=chunk_ms, speaker_id=speaker_id, speed=speed)

    def _make_engine(self) -> sherpa_onnx.OfflineTts:
        kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(self._root / "tts" / "model.int8.onnx"),
            voices=str(self._root / "tts" / "voices.bin"),
            tokens=str(self._root / "tts" / "tokens.txt"),
            lexicon=str(self._root / "tts" / "lexicon-zh.txt"),
            data_dir=self._data_dir,
            lang=self._lang,
        )
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro, num_threads=self._threads),
            max_num_sentences=1,
        )
        if not cfg.validate():
            msg = f"invalid kokoro TTS config: data_dir={self._data_dir}"
            raise RuntimeError(msg)
        return sherpa_onnx.OfflineTts(cfg)


class MeloTts(BaseTts):
    """MeloTTS VITS int8（zh_en），44.1kHz 输出，单音色。

    本机实测明显快于 kokoro：同一句「我要...」合成 936ms vs 1416ms，
    单字「好」160ms vs 636ms。代价是采样率非 24k（需重采样）与音色单一。
    """

    def __init__(
        self,
        model_root: Path | str | None = None,
        subdir: str = "tts-melo",
        data_dir: str | None = None,
        speaker_id: int = 0,
        speed: float = 1.0,
        num_threads: int = 4,
        chunk_ms: int = 200,
    ) -> None:
        self._root = _models(model_root) / subdir
        self._data_dir = _espeak_data(data_dir)
        self._threads = num_threads
        super().__init__(chunk_ms=chunk_ms, speaker_id=speaker_id, speed=speed)

    def _make_engine(self) -> sherpa_onnx.OfflineTts:
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(self._root / "model.int8.onnx"),
            tokens=str(self._root / "tokens.txt"),
            lexicon=str(self._root / "lexicon.txt"),
            data_dir=self._data_dir,
        )
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=self._threads),
        )
        if not cfg.validate():
            msg = "invalid melo TTS config"
            raise RuntimeError(msg)
        return sherpa_onnx.OfflineTts(cfg)


# 标注成两个具体类的联合，而不是基类 `type[BaseTts]`：基类签名里没有
# `num_threads`（各引擎自己声明），写成基类会让工厂里的 `cls(num_threads=4)` 过不了
# mypy strict，逼着人用 cast 绕——那等于把类型检查关掉。
TTS_ENGINES: dict[str, type[KokoroTts] | type[MeloTts]] = {"kokoro": KokoroTts, "melo": MeloTts}


def build_asr(model_root: Path | str | None = None) -> AsrProvider:
    return ParaformerStreamingAsr(model_root)


def build_tts(model_root: Path | str | None = None, speaker_id: int = 0) -> TtsProvider:
    return KokoroTts(model_root, speaker_id=speaker_id)


def build_tts_engine(name: str = "kokoro", **kwargs: object) -> BaseTts:
    return TTS_ENGINES[name](**kwargs)  # type: ignore[arg-type]
