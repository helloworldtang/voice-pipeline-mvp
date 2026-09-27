"""会话编排：把上行音频帧喂给调度器，把 TurnItem 流吐给传输层。

Session 不认识 WebSocket，也不认识具体 provider —— 这样同一份逻辑既能跑 WS，
也能跑 scripted 回放（CI 确定性回归）与 A/B 实验。

M3 起，Session 内持有一台**下行 outbox 队列**：run_turn（一轮的生成协程）只负责把
TurnItem 投进队列，真正的发送由传输层统一 pump。这样做的唯一原因——barge-in：

    AI 播放期间用户仍在持续上行麦克风帧，打断检测必须能在这段时间里继续跑。
    若发送与接收挤在同一个协程里串行处理，播放期间就收不到新帧，barge-in 永远触发不了。

所以 on_audio（含打断检测）与 run_turn（含播放）是并发的两路，outbox 是它们之间唯一
的交接面；播放端停播信号（AudioStop）也走 outbox，避免与音频发送竞态。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.core.barge_in import BargeInDetector
from voice_pipeline.core.eou import EouDetector, FusedEou, GatingMode
from voice_pipeline.core.scheduler.base import AudioFragment, Scheduler, TurnItem
from voice_pipeline.core.scheduler.pipelined import PipelinedScheduler
from voice_pipeline.core.scheduler.serial import SerialScheduler
from voice_pipeline.core.sentence_splitter import SentenceSplitter
from voice_pipeline.core.turn_fsm import TurnFsm, TurnState
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.base import AsrProvider, LlmProvider, TtsProvider
from voice_pipeline.providers.factory import build_eou, build_providers
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import AsrFinal, AudioStop, TtsSentence

# 多轮上下文上限：最近 10 轮（20 条 user/assistant 消息）。语音对话的时效性强，
# 更早的轮次对当前问题几乎没有贡献，却每轮都要整包发给 LLM 付费。
MAX_HISTORY_MESSAGES = 20


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
        # 默认 neural（D2 实测后从 conservative 切过来，2026-09-22）：
        # 官方测试集受控实验（中文 n=335）静音 100ms + 神经判停 → 句末检出 70%、
        # 句中抢话 4.6%；静音窗要 500ms 才敢判停，而句中停顿一到 500ms 就 100% 抢话。
        # 神经档 70% 的轮次 ~100ms 就判停，剩下 30% 走 800ms 兜底（期望 ≈310ms）。
        # 缺权重时 build_eou 退回 conservative 并告警 —— 账单 eou_mode 会如实写回。
        gating: GatingMode = GatingMode.NEURAL,
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
        # 多轮上下文：Session 持有，调度器每次调用 LLM 时取快照。只记**实际播出去的**
        # 内容——被打断的轮次里，用户听到多少就记多少，这是有意为之（听到的才是真的），
        # 不要"修复"成整句入史。
        self._history: list[dict[str, str]] = []

        # M3：状态机 + 打断检测器 + 下行队列
        self._fsm = TurnFsm()
        self._barge = BargeInDetector()
        self.outbox: asyncio.Queue[TurnItem | AudioStop] = asyncio.Queue()
        self._turn_running = False
        # 本轮判停指标是否已抓进账单（防止被 reset 后的 0 覆盖）
        self._eou_collected = False

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
            "history": self._history,
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
        """上行音频帧。echo 模式立即回发；链路模式攒给 ASR，并喂给 EOU 判停器。

        播放期间（SPEAKING/THINKING）若检测到用户又开口，立即触发 barge-in——
        这是三层打断的第一层（L1 检测），后续的取消与停播在 barge_in() 里完成。
        """
        if self._scheduler is None:
            self.frames_echoed += 1
            if self._eou is not None:
                self._eou.push_audio(chunk)
            return [AudioFragment(seq=self.frames_echoed, data=chunk)]
        await self._scheduler.on_audio(chunk)
        if self._eou is not None:
            self._eou.push_audio(chunk)
            self._eou.update_text(self._scheduler.text())
        # L1：AI 在说/想时用户又开口 → 打断
        if (
            not self.interrupted
            and self._fsm.state in (TurnState.SPEAKING, TurnState.THINKING)
            and self._barge.is_speech(chunk)
        ):
            self.barge_in()
        return []

    def pending_end(self) -> bool:
        """EOU 判停器是否已判定用户说完。real 模式下由传输层轮询以自动收尾。"""
        return self._eou.should_end() if self._eou is not None else False

    def barge_in(self, *, source: str = "mic") -> bool:
        """三层 barge-in 的总入口（L2/L3 在这里落地）。

        前置：必须是 AI 正在说/想（SPEAKING/THINKING），且本轮还没被打断过。
        否则这是「用户在听」或「已打断」，不打断。

        L2 播放层：取消在飞的 LLM/TTS 计算（scheduler.cancel），并往 outbox 投
            AudioStop 让播放端丢弃已排队、尚未出声的分片。
        L3 暂停/重启语义：本轮作废，状态机回到 LISTENING——不重开一轮，而是等用户
            把插话说完（重新走 EOU）再答。这才是「能真正交流」而不是「一刀切静音」。
        """
        if self.interrupted:
            return False
        if self._fsm.state not in (TurnState.SPEAKING, TurnState.THINKING):
            return False
        self._fsm.move(TurnState.INTERRUPTED)
        self.interrupted = True
        if self._scheduler is not None:
            self._scheduler.cancel()
        self._barge.reset()  # 去抖计数清零，别让上一轮尾音污染下一轮
        self.metrics.barge_in_count += 1
        self.metrics.barge_in_latency_ms = self._clock.elapsed_ms()
        self.outbox.put_nowait(AudioStop())  # L2：通知播放端停播
        self._fsm.move(TurnState.LISTENING)  # L3：回到听，等用户插话说完再答
        return True

    def interrupt(self) -> bool:
        """兼容旧调用方：等价于客户端主动打断。"""
        return self.barge_in(source="client")

    def _collect_eou_metrics(self) -> None:
        """把 FusedEou 的验收指标抓进账单。**必须在本轮 reset 之前调用**。

        reset 会清掉静音计时与抢话/漏判标志，采晚了 eou_latency_ms 就是 0 ——
        而 0 看起来像个正常数字，不像缺失，所以这个失效能潜伏一整个里程碑。
        """
        if isinstance(self._eou, FusedEou):
            self.metrics.premature_eou = self._eou.premature_eou
            self.metrics.missed_eou = self._eou.missed_eou
            self.metrics.eou_latency_ms = self._eou.eou_latency_ms()
            self.metrics.eou_mode = self._gating.value
        self._eou_collected = True

    async def finish(self) -> AsyncIterator[TurnItem]:
        """跑完一轮，产出事件与音频分片（被 run_turn 包装成后台任务）。

        这里同时维护状态机的 THINKING→SPEAKING→LISTENING 流转；interrupted 为真时
        立即停止产出（已投出去的块由播放端的 AudioStop 负责停播，服务端追不回）。
        """
        if self._scheduler is None:
            return
        # 把 EOU 判停器的验收指标落进账单（仅 FusedEou 携带抢话/漏判/静音时长）。
        # ⚠️ 只在还没填过时填：run_turn 会在 reset 之前先采一次真值，这里若无条件
        # 覆盖，就会用 reset 之后的 0 把真值抹掉（eou_latency_ms 曾因此恒为 0）。
        if isinstance(self._eou, FusedEou):
            if not self._eou_collected:
                self._collect_eou_metrics()
            self.metrics.eou_mode = self._gating.value
        # 进入一轮：先到 LISTENING（已在此状态则无害跳过），再 THINKING
        self._fsm.move(TurnState.LISTENING)
        self._fsm.move(TurnState.THINKING)
        first_audio = False
        try:
            async for item in self._scheduler.on_end_of_utterance():
                if self.interrupted:
                    break  # 被打断：剩余不播
                if isinstance(item, AudioFragment) and not first_audio:
                    first_audio = True
                    self._fsm.move(TurnState.SPEAKING)
                yield item
        except asyncio.CancelledError:
            # 我们自己 cancel 在飞投机导致的，属预期，吞掉不冒泡
            pass
        finally:
            if not self.interrupted:
                self._fsm.move(TurnState.LISTENING)

    async def run_turn(self) -> None:
        """后台跑一轮：把 finish() 产出的 TurnItem 投进 outbox，传输层统一发送。

        与 on_audio 并发——这正是 barge-in 能工作的前提。新一轮开始清掉上轮的打断
        标志并 reset EOU，让「用户插话 → AI 停 → 用户继续说 → 重新判停」能闭环。
        """
        if self._turn_running:
            return
        self._turn_running = True
        self.interrupted = False
        self._eou_collected = False
        # 每轮一本新账：TurnMetrics 的 mark 只记首次到达，跨轮复用不清账的话，
        # 第 2 轮起的账单会复读第 1 轮。必须在 _collect_eou_metrics 之前清——
        # 它要把上一轮的判停指标写进本轮账里。
        self.metrics.reset_turn()
        # 必须在 reset 之前采：reset 会清掉静音计时与抢话/漏判标志，
        # 采晚了 eou_latency_ms 就恒为 0（这个指标曾因此整整失效一整个里程碑）。
        self._collect_eou_metrics()
        if isinstance(self._eou, FusedEou):
            self._eou.reset()
        user_text: str | None = None
        assistant_parts: list[str] = []
        try:
            async for item in self.finish():
                self.outbox.put_nowait(item)
                # 边播边攒多轮上下文：AsrFinal→user，TtsSentence→assistant（拼句）。
                # 只记流经 outbox 的内容，被 interrupted 截断的部分自然不入史。
                if isinstance(item, AsrFinal):
                    user_text = item.text
                elif isinstance(item, TtsSentence):
                    assistant_parts.append(item.text)
        finally:
            self._turn_running = False
            if user_text is not None:
                self._history.append({"role": "user", "content": user_text})
                if assistant_parts:
                    self._history.append({"role": "assistant", "content": "".join(assistant_parts)})
                self._trim_history()

    def _trim_history(self) -> None:
        """裁到最近 10 轮（20 条消息），从头部整对裁——保住 user 开头的轮次边界。"""
        if len(self._history) <= MAX_HISTORY_MESSAGES:
            return
        self._history[:] = self._history[-MAX_HISTORY_MESSAGES:]
        if self._history and self._history[0]["role"] != "user":
            del self._history[0]
