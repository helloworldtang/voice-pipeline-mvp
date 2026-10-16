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
from voice_pipeline.core.filler import FillerBank
from voice_pipeline.core.scheduler.base import AudioFragment, Scheduler, TurnItem, play_filler
from voice_pipeline.core.scheduler.pipelined import PipelinedScheduler
from voice_pipeline.core.scheduler.serial import SerialScheduler
from voice_pipeline.core.sentence_splitter import SentenceSplitter
from voice_pipeline.core.turn_fsm import TurnFsm, TurnState
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.metrics.jsonl import TurnRecorder
from voice_pipeline.metrics.trace import Tracer
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.base import AsrProvider, LlmProvider, TtsProvider
from voice_pipeline.providers.factory import build_eou, build_providers
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import AsrFinal, AudioStop, TtsSentence, TurnEnd

# 多轮上下文上限：最近 10 轮（20 条 user/assistant 消息）。语音对话的时效性强，
# 更早的轮次对当前问题几乎没有贡献，却每轮都要整包发给 LLM 付费。
MAX_HISTORY_MESSAGES = 20

# 空转写门的放行上限（次）：100ms 帧节奏下 ≈1s——超过它说明 ASR 不会再出字了
EMPTY_GATE_MAX_BLOCKS = 10


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
        # 默认 conservative（第四次定档，2026-10-04）：neural 在真人半秒停顿下
        # 抢话 3/3（D5 真人验收），退回。前三次定档史见下方注释。
        # 默认 neural（D2 实测后从 conservative 切过来，2026-09-22）：
        # 官方测试集受控实验（中文 n=335）静音 100ms + 神经判停 → 句末检出 70%、
        # 句中抢话 4.6%；静音窗要 500ms 才敢判停，而句中停顿一到 500ms 就 100% 抢话。
        # 神经档 70% 的轮次 ~100ms 就判停，剩下 30% 走 800ms 兜底（期望 ≈310ms）。
        # 缺权重时 build_eou 退回 conservative 并告警 —— 账单 eou_mode 会如实写回。
        gating: GatingMode = GatingMode.CONSERVATIVE,
        recorder: TurnRecorder | None = None,
        # 预合成 filler（2026-10-10，体感优化）：判停成立后、TTS 首片之前先播一段预合成
        # 短音频，把那一格 ~650ms 的固定成本盖住。None = 关闭（默认）。
        # ⚠️ 口径B 一分不省：filler 只打 first_sound，见 core/filler.py 的计量纪律。
        filler: FillerBank | None = None,
    ) -> None:
        self.session_id = uuid4().hex[:12]
        self.trace_id = uuid4().hex[:16]
        self.mode = mode
        self.frames_echoed = 0
        self.interrupted = False
        self._clock = clock or SessionClock()
        self.metrics = TurnMetrics(clock=self._clock)
        # 链路追踪：会话级 trace_id 贯穿多轮，每轮一棵 span 树（一轮 = 一个根 span）。
        # 挂在 TurnMetrics 上是为了让调度器不改签名就能埋点。
        self.metrics.tracer = Tracer(self.trace_id, self._clock)
        self._gating = gating
        self._eou = eou
        # 账单落盘（VOICE_EVENTS_JSONL）：默认关闭，见 metrics/jsonl.py。
        self._recorder = recorder or TurnRecorder.from_env()
        self._turn_index = 0  # 会话内轮次编号，从 1 起——落盘账单靠它排序
        self._recorded_this_turn = False  # 防重复记账：TurnEnd 与 finally 两条路径
        # 多轮上下文：Session 持有，调度器每次调用 LLM 时取快照。只记**实际播出去的**
        # 内容——被打断的轮次里，用户听到多少就记多少，这是有意为之（听到的才是真的），
        # 不要"修复"成整句入史。
        self._history: list[dict[str, str]] = []
        # filler 自己一个分片器：它的 seq 用 0/负数编号，不占调度器的分片序号空间
        self._filler = filler
        self._filler_chunker = Chunker()

        # M3：状态机 + 打断检测器 + 下行队列
        self._fsm = TurnFsm()
        self._barge = BargeInDetector()
        self.outbox: asyncio.Queue[TurnItem | AudioStop] = asyncio.Queue()
        self._turn_running = False
        # 传输侧那一跳（outbox → 真正下发）：Session 记"入队时刻"，由 transport 在
        # 真正发出首片时回调 note_first_audio_send() 补齐。**账单里落差记 -1 而不是 0**——
        # 没量到就是没量到（filler 开启时这一格会成为首音被压后的证据）。
        self._first_audio_enqueued_ms: float = -1.0
        self._first_audio_sent = False
        # 本轮判停指标是否已抓进账单（防止被 reset 后的 0 覆盖）
        self._eou_collected = False
        # 空转写门的连续拦截计数（见 pending_end）
        self._empty_blocks = 0
        # 空轮冷却：放行过一个空转写轮后，判停必须等新的人声再启动——否则持续
        # 的环境噪声/回声能量会以 ~1Hz 空转出账单（2026-10-04 验收实测 64 条）
        self._await_fresh_speech = False

        real = mode.startswith("real-") or mode == "real"
        spec_mode = mode.removeprefix("real-")
        if spec_mode == "echo":
            self._scheduler: Scheduler | None = None
            return

        if asr is None and llm is None and tts is None:
            providers = build_providers("real" if real else "sim")
            asr, llm, tts = providers.asr, providers.llm, providers.tts
        # 账单里标注 LLM 引擎：SimLlm 意味着该轮数字不含 LLM（缺 key 静默退回），
        # 这份验收记录就不能当真实延迟用——让 report_turns 能机器判读这一点。
        self._llm_name = type(llm).__name__ if llm is not None else "SimLlm"
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

        # 上行每帧都埋：这一格过去完全不在视野里，而它的 CPU 是在**同一个事件循环**里
        # 花掉的——下行 TTS/LLM 在 await 时会被它推迟。要看"尾部的等待是谁造成的"必须有它。
        tr = self.metrics.tracer
        if tr is not None:
            # 上一轮已闭合 → 开新一轮前先清树。**安全性**：上一轮的账在 run_turn 收到
            # TurnEnd 的那一刻就落完了（收到与落账之间没有 await），而新一轮第一帧
            # 只可能在那之后被处理——所以这里清不会抹掉任何还没写盘的证据。
            if tr.root is None or not tr.root.open:
                tr.reset()
            if tr.root is None:
                # 本轮第一帧 = 整轮 trace 的起点。没有它，"用户说话期间链路在干什么"
                # 这一段在账单里是空白（2026-10-17 实测丢过一整段）。
                tr.start_root("turn", attrs={"mode": self.mode, "turn": self._turn_index + 1})
        frame = tr.start("uplink.frame", attrs={"bytes": len(chunk)}) if tr is not None else None
        if tr is not None:
            dec = tr.start("asr.push_audio", frame)
            await self._scheduler.on_audio(chunk)
            tr.finish(dec)
        else:  # pragma: no cover - tracer 恒存在，分支只为类型收窄
            await self._scheduler.on_audio(chunk)
        if self._barge.is_speech(chunk):
            self._await_fresh_speech = False
        if self._eou is not None:
            if tr is not None:
                upd = tr.start("eou.update", frame)
                self._eou.push_audio(chunk)
                self._eou.update_text(self._scheduler.text())
                tr.finish(upd)
            else:  # pragma: no cover
                self._eou.push_audio(chunk)
                self._eou.update_text(self._scheduler.text())
        # L1：AI 在说/想时用户又开口 → 打断
        if (
            not self.interrupted
            and self._fsm.state in (TurnState.SPEAKING, TurnState.THINKING)
            and self._barge.is_speech(chunk)
        ):
            self.barge_in()
        if tr is not None and frame is not None:
            tr.finish(frame)
        return []

    def pending_end(self) -> bool:
        """EOU 判停器是否已判定用户说完。real 模式下由传输层轮询以自动收尾。

        **空转写门（2026-10-04 真人验收抓到的回声自激回环）**：外放回声/噪声
        能量够 VAD 认定「有人声」（≥0.01）、ASR 却认不出任何字——EOU 在这种
        音频上判停会触发「AI 自问自答」回环（实测一句"我想睡"引来 6 连答，
        账单 6 轮 user_text 全空）。判停作废、判停器重置；ASR 出字了再判。
        代价：超短句的判停最多晚一个 partial 周期（~300ms），换回环根治。

        **放行上限**：ASR 一直不出字（切轮残余、极怪回声）时不许无限拦——
        连续拦 EMPTY_GATE_MAX_BLOCKS 次（100ms 帧节奏下 ≈1s）后放行，由
        第二道门（调度器空转写不起轮，不出音频）兜底：回环照样断，
        而链路不会卡死在"永远判不了停"。
        """
        if self._eou is None:
            return False
        if self._await_fresh_speech:
            return False  # 空轮冷却中：等新的人声（on_audio 里清这个标志）
        if not self._eou.should_end():
            return False
        if self._scheduler is not None and not self._scheduler.text():
            self._empty_blocks += 1
            if self._empty_blocks < EMPTY_GATE_MAX_BLOCKS:
                # unlatch 而非 reset：reset 会连 has_speech 一起清掉，纯静音
                # 喂不回 VAD，EOU 永远再闩不上、计数走不到放行（实测卡死）。
                self._eou.unlatch()
                return False
            # 放行的是空轮：进入冷却，直到下一阵真实人声
            self._await_fresh_speech = True
        return True

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

        reset 会清掉静音计时与漏判标志，采晚了 eou_latency_ms 就是 0 ——
        而 0 看起来像个正常数字，不像缺失，所以这个失效能潜伏一整个里程碑。

        ⚠️ 这里**不采** premature_eou：抢话的证据窗口在判停后 ~300ms，证据落在
        reset **之后**才出现（reset 现在会把观察窗带过去，见 FusedEou.reset）。
        采的动作移到 `_record_turn`——账单定型那一刻用 `take_premature()` 读走。
        """
        if isinstance(self._eou, FusedEou):
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
        # 根 span：整轮（本轮第一帧 → turn.end）。口径B 窗口是它的孩子，
        # 账就从"一串时刻"变成"一棵可归因的树"，残差（没埋到的部分）会显式浮出来。
        tracer = self.metrics.tracer
        if tracer is not None and tracer.root is None:
            tracer.start_root("turn", attrs={"mode": self.mode, "turn": self._turn_index + 1})
        if tracer is not None and tracer.window is None:
            tracer.start_window(
                "turn.caliber_b", attrs={"mode": self.mode, "turn": self._turn_index}
            )
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
                # 预合成 filler：判停已成立、这一轮有真文本 → 紧接着把 filler 播出去，
                # 盖住 TTS 首片那一格固定成本（~650ms）。filler 也是 AI 的声音，所以
                # 状态机照常进 SPEAKING。计量纪律：只打 first_sound，**口径B 一分不省**
                # （见 core/filler.py）——关掉时（None）这段是零开销的空循环。
                if isinstance(item, AsrFinal) and item.text:
                    async for frag in play_filler(self._filler, self._filler_chunker, self.metrics):
                        if not first_audio:
                            first_audio = True
                            self._fsm.move(TurnState.SPEAKING)
                        yield frag
        except asyncio.CancelledError:
            # 我们自己 cancel 在飞投机导致的，属预期，吞掉不冒泡
            pass
        finally:
            if not self.interrupted:
                self._fsm.move(TurnState.LISTENING)
            # 整轮 trace 在这里闭合。**口径B 窗口由调度器闭合**（首音那一刻），
            # 这里只收尾外层：turn.end 之后的等待不进口径B，但要留在 trace 里可查。
            if tracer is not None:
                tracer.finish_root()

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
        self._empty_blocks = 0
        self._recorded_this_turn = False
        self._first_audio_enqueued_ms = -1.0
        self._first_audio_sent = False
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
                if (
                    isinstance(item, AudioFragment)
                    and self._first_audio_enqueued_ms < 0
                    and item.seq >= 0  # filler 用负序号：它是体感优化，不进口径B 那一跳
                ):
                    self._first_audio_enqueued_ms = self._clock.elapsed_ms()
                self.outbox.put_nowait(item)
                # 边播边攒多轮上下文：AsrFinal→user，TtsSentence→assistant（拼句）。
                # 只记流经 outbox 的内容，被 interrupted 截断的部分自然不入史。
                if isinstance(item, AsrFinal):
                    user_text = item.text
                elif isinstance(item, TtsSentence):
                    assistant_parts.append(item.text)
                elif isinstance(item, TurnEnd):
                    # 账单在这一刻定型。
                    self._record_turn(item.metrics, user_text, assistant_parts)
        finally:
            self._turn_running = False
            # 被打断的轮次产不出 TurnEnd（finish() 在 interrupted 时 break），
            # 必须在 finally 里补记——否则「打断」这个一等验收指标永远没有证据，
            # `interrupted` 会是个永远为假的死字段。（初版曾把「不落盘」当成
            # 「天然只落完整轮次」，实际上是漏掉了最需要记录的那一类。）
            if self.interrupted and not self._recorded_this_turn:
                self._record_turn(self.metrics.bill(), user_text, assistant_parts)
            if user_text is not None:
                self._history.append({"role": "user", "content": user_text})
                if assistant_parts:
                    self._history.append({"role": "assistant", "content": "".join(assistant_parts)})
                self._trim_history()

    def note_first_audio_send(self) -> None:
        """传输层回调：**真正把首片发下去的那一刻**调一次。

        量的是 outbox 排队 → 落网线那一跳。为什么单独记它：`first_audio_out` 打点发生在
        **调度器 yield 的瞬间**，从此以后（outbox 排队、JSON 封帧、WS 发送、客户端抖动缓冲）
        全都不在任何一格账单里——filler 打开后这一格会变成首音被推迟多久的直接证据。

        没能记到（尚未入队 / 尚未发送）就保持 -1，**不填 0**。
        """
        if self._first_audio_sent or self._first_audio_enqueued_ms < 0:
            return
        tracer = self.metrics.tracer
        if tracer is None:
            return
        now = self._clock.elapsed_ms()
        self._first_audio_sent = True
        tracer.record(
            "tx.first_audio",
            self._first_audio_enqueued_ms,
            now,
            parent=tracer.window or tracer.root,
            attrs={"wait_ms": round(now - self._first_audio_enqueued_ms, 3)},
        )

    def _record_turn(
        self,
        bill: dict[str, float],
        user_text: str | None,
        assistant_parts: list[str],
    ) -> None:
        """落盘一轮账单。**唯一入口**——保证一轮只落一行、不重不漏。

        两条调用路径（`TurnEnd` 正常完成 / `finally` 打断补记）共用它：
        分散写两遍就会出现「正常轮重复记」或「打断轮漏记」，两种都难查。
        被打断的轮次 `first_audio_out` 可能未打点（AI 还没出声就被掐了），
        那时口径B 如实为 -1（缺失），不做推测填补。
        """
        # 抢话取证在记账这一刻才定型：证据窗口横跨判停后的 reset（见
        # _collect_eou_metrics 的警告）。bill 是调度器在 TurnEnd 构造时就定型的
        # dict，此刻改 self.metrics 已经晚了——直接改写账单本身。take 读后即清，
        # 防上一轮幽灵污染下一轮。
        if isinstance(self._eou, FusedEou):
            premature = self._eou.take_premature()
            self.metrics.premature_eou = premature
            bill = {**bill, "premature_eou": 1.0 if premature else 0.0}
        self._recorded_this_turn = True
        self._turn_index += 1
        self._recorder.record(
            session_id=self.session_id,
            trace_id=self.trace_id,
            mode=self.mode,
            turn=self._turn_index,
            bill=bill,
            extra={
                "interrupted": self.interrupted,
                "llm": self._llm_name,
                "user_text": user_text or "",
                "assistant_text": "".join(assistant_parts),
                # span 树随账单一起落盘：事后能按 trace_id 重建一棵完整的调用树，
                # 而不是只有几个打点差。只关心数字的工具可以直接无视这一字段。
                "spans": self.metrics.tracer.to_list() if self.metrics.tracer is not None else [],
            },
        )

    def _trim_history(self) -> None:
        """裁到最近 10 轮（20 条消息），从头部整对裁——保住 user 开头的轮次边界。"""
        if len(self._history) <= MAX_HISTORY_MESSAGES:
            return
        self._history[:] = self._history[-MAX_HISTORY_MESSAGES:]
        if self._history and self._history[0]["role"] != "user":
            del self._history[0]
