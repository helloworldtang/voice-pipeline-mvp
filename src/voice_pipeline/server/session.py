"""会话编排：把上行音频帧喂给调度器，把 TurnItem 流吐给传输层。

Session 不认识 WebSocket，也不认识具体 provider —— 这样同一份逻辑既能跑 WS，也能跑
scripted 回放（CI 确定性回归）与 A/B 实验。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.core.eou import EouDetector, FusedEou, GatingMode
from voice_pipeline.core.scheduler.base import AudioFragment, Scheduler, TurnItem
from voice_pipeline.core.scheduler.pipelined import PipelinedScheduler
from voice_pipeline.core.scheduler.serial import SerialScheduler
from voice_pipeline.core.sentence_splitter import SentenceSplitter
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.base import AsrProvider, LlmProvider, TtsProvider
from voice_pipeline.providers.factory import build_eou, build_providers
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts


class Session:
    """一个 WS 连接一个 Session，trace_id 贯穿日志与账单。

    mode:
      echo         —— 只回发音频帧，用来量传输层本身的开销
      serial       —— EOU 之后才启动 LLM（对照组，复现质疑方时序）
      pipelined    —— 用户说话期间就投机跑 LLM+TTS（实验组，唯一能逼近 300ms 的路径）
      real-serial / real-pipelined —— 同上，但换真模型
    """

    def __init__(
        self,
        mode: str = "serial",
        *,
        asr: AsrProvider | None = None,
        llm: LlmProvider | None = None,
        tts: TtsProvider | None = None,
        clock: SessionClock | None = None,
        grace_ms: float = 0.0,
        eou: EouDetector | None = None,
        gating: GatingMode = GatingMode.CONSERVATIVE,
    ) -> None:
        self.session_id = uuid4().hex[:12]
        self.trace_id = uuid4().hex[:16]
        self.mode = mode
        self.frames_echoed = 0
        self.interrupted = False
        self._clock = clock or SessionClock()
        self.metrics = TurnMetrics(clock=self._clock)
        self._gating = gating
        self._eou = eou

        real = mode.startswith("real-") or mode == "real"
        spec_mode = mode.removeprefix("real-")
        if spec_mode == "echo":
            self._scheduler: Scheduler | None = None
            return

        if asr is None and llm is None and tts is None:
            providers = build_providers("real" if real else "sim")
            asr, llm, tts = providers.asr, providers.llm, providers.tts
        deps: dict[str, object] = {
            "asr": asr or SimAsr(),
            "llm": llm or SimLlm(),
            "tts": tts or SimTts(),
            "metrics": self.metrics,
            "splitter": SentenceSplitter(),
            "chunker": Chunker(),
        }
        if spec_mode == "pipelined":
            self._scheduler = PipelinedScheduler(**deps, grace_ms=grace_ms)  # type: ignore[arg-type]
        else:
            self._scheduler = SerialScheduler(**deps)  # type: ignore[arg-type]

        # real 模式默认自动判停（client 不必显式发 session.end）；sim 模式留 None，
        # 仍由显式 session.end 驱动——保持原有链路测试语义不变。
        if real and self._eou is None:
            self._eou = build_eou(gating)

    async def on_audio(self, chunk: bytes) -> list[AudioFragment]:
        """上行音频帧。echo 模式立即回发；链路模式攒给 ASR，并喂给 EOU 判停器。"""
        if self._scheduler is None:
            self.frames_echoed += 1
            if self._eou is not None:
                self._eou.push_audio(chunk)
            return [AudioFragment(seq=self.frames_echoed, data=chunk)]
        await self._scheduler.on_audio(chunk)
        if self._eou is not None:
            self._eou.push_audio(chunk)
            self._eou.update_text(self._scheduler.text())
        return []

    def pending_end(self) -> bool:
        """EOU 判停器是否已判定用户说完。real 模式下由传输层轮询以自动收尾。"""
        return self._eou.should_end() if self._eou is not None else False

    async def finish(self) -> AsyncIterator[TurnItem]:
        """收到 session.end（或被 EOU 自动触发）：跑完一轮，产出事件与音频分片。"""
        if self._scheduler is None:
            return
        # 把 EOU 判停器的验收指标落进账单（仅 FusedEou 携带抢话/漏判/静音时长）
        if isinstance(self._eou, FusedEou):
            self.metrics.premature_eou = self._eou.premature_eou
            self.metrics.missed_eou = self._eou.missed_eou
            self.metrics.eou_latency_ms = self._eou.eou_latency_ms()
            self.metrics.eou_mode = self._gating.value
        async for item in self._scheduler.on_end_of_utterance():
            yield item

    def interrupt(self) -> None:
        """barge-in：用户又开口了，AI 必须闭嘴。

        这里做到的是「立刻停止**产出**」——在飞的投机与缓存产物一并丢弃。
        但**已经发出去的音频**追不回来：让声音真的停下需要播放端丢弃已排队的分片，
        那是客户端的责任（见 live 客户端），服务端这一层只能保证不再新增。
        """
        self.interrupted = True
        if self._scheduler is not None:
            self._scheduler.cancel()
