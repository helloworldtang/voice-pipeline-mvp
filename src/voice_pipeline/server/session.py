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
import logging
import os
from collections.abc import AsyncIterator
from uuid import uuid4

from voice_pipeline.audio.chunker import Chunker
from voice_pipeline.core.barge_in import BargeInDetector
from voice_pipeline.core.energy_gate import AsrEnergyGate
from voice_pipeline.core.eou import EouDetector, FusedEou, GatingMode
from voice_pipeline.core.filler import FillerBank
from voice_pipeline.core.scheduler.base import AudioFragment, Scheduler, TurnItem, play_filler
from voice_pipeline.core.scheduler.pipelined import PipelinedScheduler
from voice_pipeline.core.scheduler.serial import SerialScheduler
from voice_pipeline.core.sentence_splitter import SentenceSplitter
from voice_pipeline.core.turn_fsm import TurnFsm, TurnState
from voice_pipeline.metrics.events import SessionClock, TurnMetrics
from voice_pipeline.metrics.jsonl import TurnRecorder
from voice_pipeline.metrics.trace import Tracer, root_breakdown, window_breakdown
from voice_pipeline.providers.asr.sim_asr import SimAsr
from voice_pipeline.providers.base import AsrProvider, LlmProvider, TtsProvider
from voice_pipeline.providers.factory import build_eou, build_providers
from voice_pipeline.providers.llm.sim_llm import SimLlm
from voice_pipeline.providers.tts.sim_tts import SimTts
from voice_pipeline.server.protocol import AsrFinal, AudioStop, TtsSentence, TurnEnd

# 多轮上下文上限：最近 10 轮（20 条 user/assistant 消息）。语音对话的时效性强，
# 更早的轮次对当前问题几乎没有贡献，却每轮都要整包发给 LLM 付费。
logger = logging.getLogger(__name__)


# 首片缩短（`VOICE_FIRST_CHUNK_MIN_MS`，默认 0=关）：流式 TTS 首包常只有 40~90ms，
# 等凑满 200ms 才发首片会把 TTS 省下的时间原样吐回去（讯飞实测 89ms → 口径B 多 ~150ms）。
# 默认关是因为它破坏了「所有分片等长」这条既有约定，而守这条约定的
# `tests/test_ws.py` / `test_serial_pipeline.py` 当前不可改（凭据判据误拦），
# 所以先在流式小包 provider 的实验里用 env 打开，全量启用待测试可写后再议。
def _first_chunk_min_ms() -> int:
    raw = os.environ.get("VOICE_FIRST_CHUNK_MIN_MS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


# 跨轮清账时要抢救的打点：口径A 的两个锚点打在**判停之前**（用户开口 / ASR 首次出字），
# 而 `reset_turn()` 发生在判停之后——不保留就会被自己清掉，caliber_a_ms 恒缺
# （2026-10-27 实测：与 eou_latency_ms 曾是同一个失效模式）。
CARRY_OVER_MARKS: tuple[str, ...] = ("vad_onset", "asr_first_char")

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
        # TTS 连接预热总闸（P0.5）：默认开。探针用 --no-prewarm 关掉它跑 A/B
        # 对照臂（「不预热」与「预热」的口径B 差就是要量的东西）。
        prewarm_tts: bool = True,
        # ASR 前置能量门总闸（P1）：默认开，但**只在 real 模式生效**（见下方构造处）。
        # 探针用 --no-energy-gate 跑 A/B 对照臂。
        energy_gate: bool = True,
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
        # ASR 前置能量门（P1）：声明占位，real 模式才真正装（见下方构造处）——
        # echo 模式没有 ASR 链路，提前返回前也要有这个属性。
        self._energy_gate: AsrEnergyGate | None = None
        self.outbox: asyncio.Queue[TurnItem | AudioStop] = asyncio.Queue()
        self._turn_running = False
        # 传输侧那一跳（outbox → 真正下发）：Session 记"入队时刻"，由 transport 在
        # 真正发出首片时回调 note_first_audio_send() 补齐。**账单里落差记 -1 而不是 0**——
        # 没量到就是没量到（filler 开启时这一格会成为首音被压后的证据）。
        self._first_audio_enqueued_ms: float = -1.0
        self._first_audio_sent = False
        self._first_audio_sent_ms: float = -1.0  # 客户端出声报告的锚点（见下）
        # 打断那一跳：barge_in 记「检测到」，传输层真正发出 audio.stop 时回调
        # note_stop_send() 补上终点。**没量到就是 -1**，不填 0。
        self._stop_issued_ms: float = -1.0
        # 客户端回传的播放侧测量（playback.report）：首片→出声、打断时的排程尾音。
        # 服务端的追踪止于落网线，这两格是唯一能盖住「用户真正听到多少」的证据。
        # 取用即清（_record_turn），避免上一轮的数字漂到下一轮——宁可缺失，不可错配。
        self._playback_ms: float = -1.0
        self._playback_tail_ms: float = -1.0
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

        # ASR 前置能量门（P1）装门：确认静音的帧不推 ASR，省口径B 窗口里与下行抢
        # 事件循环的那一格（实测 p50 41~52ms，追踪文档 §三「上行不是免费的」）。
        # **只在 real 模式生效**：sim 替身（SimAsr）的 text() 按 push 次数前进，
        # 拦静音帧会把确定性回放变成非确定性；且 SimAsr.push_audio 只是计数器自增，
        # 没有可省的 CPU——要省的 np.concatenate + to_thread + ONNX 解码只存在于真 ASR。
        if real and energy_gate:
            self._energy_gate = AsrEnergyGate()

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
            "chunker": Chunker(first_min_ms=_first_chunk_min_ms()),
            "history": self._history,
            # 预热总闸一路传到调度器：判停时那次预热也归这个开关管（A/B 要能整个关掉）
            "prewarm_tts": prewarm_tts,
        }
        # 留一份引用：预热/池计数要从 Session 够到 provider（deps 是 dict[str, object]）
        self._tts_provider: TtsProvider | None = deps["tts"]  # type: ignore[assignment]
        self._llm_provider: LlmProvider | None = deps["llm"]  # type: ignore[assignment]
        if spec_mode == "pipelined":
            self._scheduler = PipelinedScheduler(**deps, grace_ms=grace_ms)  # type: ignore[arg-type]
        else:
            self._scheduler = SerialScheduler(**deps)  # type: ignore[arg-type]

        # real 模式默认自动判停（client 不必显式发 session.end）；sim 模式留 None，
        # 仍由显式 session.end 驱动——保持原有链路测试语义不变。
        if real and self._eou is None:
            self._eou = build_eou(gating)

        # TTS 连接预热（P0.5）：建连 ~260ms 不在口径B 里付——会话一建立就在
        # 后台连上，首轮第一句直接走热连接。非阻塞（不等建连完），provider
        # 不支持预热（sim/本机 TTS）时静默跳过。
        if prewarm_tts:
            self._prewarm_tts()
        self._preload_llm()

    def _preload_llm(self) -> None:
        """本地 LLM 的模型加载（1~3s）+ 首次 Metal 编译（~1s）不能在首轮里付。

        本地 provider 暴露 `preload()`；后台线程跑，失败只告警（云端没有这个方法，静默跳过）。
        与 TTS 的任务级预热同一个道理：把一次性开销挪到用户还在说话/发愣的时候。
        """
        preload = getattr(self._llm_provider, "preload", None)
        if not callable(preload):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _run() -> None:
            try:
                await asyncio.to_thread(preload)
            except Exception:  # noqa: BLE001 - 预加载失败不能拖垮会话
                logger.warning("LLM 预加载失败（首轮会变慢）", exc_info=True)

        self._llm_preload_task = loop.create_task(_run())

    def _prewarm_tts(self) -> None:
        """provider 有 prewarm() 就调用（DashScope 云 TTS 有；其余没有）。"""
        prewarm = getattr(self._tts_provider, "prewarm", None)
        if callable(prewarm):
            prewarm()

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
        # L1 人声判定**每帧只算一次**：is_speech 有内部去抖计数，过去一帧调两次
        # （_await_fresh_speech 与打断触发各一次）会让超阈帧双倍计数——文档写的
        # 3 帧去抖实际 2 帧就触发。2026-10-17 P1 接线时修正。
        speech = self._barge.is_speech(chunk)
        if speech:
            self._await_fresh_speech = False
            # 口径A 的起点（2026-10-27 补）：本轮**判停之前**第一次检出人声的那一帧。
            # 为什么限定在 utterance_end 之前：AI 播放/思考期间上行帧里也有能量
            # （回声残余、用户插话），那不是「这一句用户开口」。
            # ⚠️ 已知系统性偏差：`speech` 来自 barge-in L1 的 3 帧去抖，100ms 帧节奏下
            # 它比真实开口晚 ~300ms；且是能量判定，不是 VAD onset。所以 caliber_a_ms
            # 是**偏小**的估计，不可与口径B 横比，也不该拿去对外宣称。
            if self.metrics.elapsed("utterance_end") is None:
                self.metrics.mark("vad_onset")
        # ASR 前置能量门（P1）：确认静音的帧不推 ASR，省口径B 窗口里与下行抢循环的
        # 那一格。门为 None（sim/echo 模式或探针对照臂）时一切照旧。
        # ⚠️ EOU 仍收全部帧——静音正是它判「说完没有」的原料，门只拦 ASR 一条路。
        gated = self._energy_gate is not None and not self._energy_gate.allows(chunk)
        if not gated:
            if tr is not None:
                dec = tr.start("asr.push_audio", frame)
                await self._scheduler.on_audio(chunk)
                tr.finish(dec)
            else:  # pragma: no cover - tracer 恒存在，分支只为类型收窄
                await self._scheduler.on_audio(chunk)
        if self._eou is not None:
            if tr is not None:
                upd = tr.start("eou.update", frame)
                self._eou.push_audio(chunk)
                # 复用这一次 text() 给 asr_first_char 打点：真 ASR 的 text() 要抢
                # stream 锁、与 decode 线程竞争，不为了打点再调一次（2026-10-27）。
                partial = self._scheduler.text()
                self._eou.update_text(partial)
                tr.finish(upd)
                if partial:
                    self.metrics.mark("asr_first_char")
            else:  # pragma: no cover
                self._eou.push_audio(chunk)
                partial = self._scheduler.text()
                self._eou.update_text(partial)
                if partial:
                    self.metrics.mark("asr_first_char")
        else:
            self._note_first_char()
        # L1：AI 在说/想时用户又开口 → 打断（用上面算好的 speech，不再重复判定）
        if (
            not self.interrupted
            and self._fsm.state in (TurnState.SPEAKING, TurnState.THINKING)
            and speech
        ):
            self.barge_in()
        if tr is not None and frame is not None:
            # gated=1：这一帧被能量门拦在 ASR 之外（A/B 与线上账单都数得出来）
            tr.finish(frame, {"gated": 1} if gated else None)
        return []

    def _note_first_char(self) -> None:
        """sim 链路（没挂 EOU）的首次出字打点。

        real 链路在 on_audio 里复用 `eou.update_text` 那一次 text() 打点，不走这里。
        只在还没打过时读一次，避免每帧多一次 text()（`SimAsr.text()` 便宜，但真
        ASR 的 text() 要抢 stream 锁，这条路径不该被真链路碰到）。
        """
        if self.metrics.elapsed("asr_first_char") is not None:
            return
        if self._scheduler is not None and self._scheduler.text():
            self.metrics.mark("asr_first_char")

    def pending_end(self) -> bool:
        """外层包装：给判停决策那一格留痕，判定本体在 `_decide_eou`。

        为什么单独埋这一格：它跑在 `app.py` 的接收循环里、`on_audio` 返回**之后**
        （`uplink.frame` 已经闭合），过去它的 CPU 不落在任何 span 里——neural 档
        每帧可能在这里跑一次 mel+ONNX（~6ms），账单上完全看不见。
        挂在 root 下而不是 window 下：判停发生在口径B 窗口开始**之前**，挂进窗口
        会把窗口左边界往前拖，口径B 就不再是「说完 → 首音」。
        """
        tr = self.metrics.tracer
        dec = (
            tr.start("eou.decide", tr.root)
            if tr is not None and tr.root is not None and self._eou is not None
            else None
        )
        ended = self._decide_eou()
        if tr is not None and dec is not None:
            tr.finish(dec, {"ended": 1 if ended else 0})
        return ended

    def _decide_eou(self) -> bool:
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
        if self._energy_gate is not None:
            self._energy_gate.reset()  # 静音确认状态同步清零（打断后用户正在说）
        self.metrics.barge_in_count += 1
        self.metrics.barge_in_latency_ms = self._clock.elapsed_ms()
        # 起点：检测到用户插话的这一刻。终点由传输层在 audio.stop **真正发到网线**
        # 时补（note_stop_send）——中间那一段正是「叫不停」会变长的那一段。
        self._stop_issued_ms = self._clock.elapsed_ms()
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
        self._first_audio_sent_ms = -1.0
        self._stop_issued_ms = -1.0
        # 每轮一本新账：TurnMetrics 的 mark 只记首次到达，跨轮复用不清账的话，
        # 第 2 轮起的账单会复读第 1 轮。必须在 _collect_eou_metrics 之前清——
        # 它要把上一轮的判停指标写进本轮账里。
        self.metrics.reset_turn(CARRY_OVER_MARKS)
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
                if isinstance(item, TurnEnd):
                    # 每轮 span 拆解随 turn.end 下发：页面要能「边跑边看时间花在哪」。
                    # 走扁平 metrics 键（span.<name>）而不是新加嵌套字段：wire 契约里
                    # metrics 就是 dict[str, float]，加字段要改协议；页面只需数字，
                    # 完整 span 树已经在 JSONL 的 spans 里（记账仍用调度器给的 bill）。
                    item = TurnEnd(metrics={**item.metrics, **self._span_metrics()})
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

    def _span_metrics(self) -> dict[str, float]:
        """口径B 窗口的 span 自有时间 → 扁平 metrics 键（`span.<name>`）。

        服务端算、页面只负责显示：`exclusive_times` 的口径（每毫秒只归最内层、重叠不
        重复计）是这一层的不变量，前端再算一遍必然漂。没有窗口的轮返回空表。
        """
        tracer = self.metrics.tracer
        breakdown = window_breakdown(tracer) if tracer is not None else None
        if breakdown is None:
            return {}
        out = {f"span.{item['name']}": float(item["ms"]) for item in breakdown["items"]}
        out["span.unattributed"] = float(breakdown["unattributed_ms"])
        return out

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
        self._first_audio_sent_ms = now
        tracer.record(
            "tx.first_audio",
            self._first_audio_enqueued_ms,
            now,
            parent=tracer.window or tracer.root,
            attrs={"wait_ms": round(now - self._first_audio_enqueued_ms, 3)},
        )

    def note_stop_send(self) -> None:
        """传输层回调：**audio.stop 真正发到网线的那一刻**调一次（barge-in 那条道）。

        量的是「检测到插话 → 停播帧落地」全程，中间含 outbox 排队与越过积压音频的
        等待——这正是 `queues.py` 那条铁律（杀停帧必须跳队）在账单里的证据。
        没有打断（没投过 AudioStop）就什么都不记，不编一个 0ms 的假格。
        """
        if self._stop_issued_ms < 0:
            return
        tracer = self.metrics.tracer
        if tracer is None:
            return
        now = self._clock.elapsed_ms()
        tracer.record(
            "tx.stop",
            self._stop_issued_ms,
            now,
            parent=tracer.root,
            attrs={"wait_ms": round(now - self._stop_issued_ms, 3)},
        )
        self._stop_issued_ms = -1.0  # 取用即清：一次打断只记一格，不重复计

    def note_playback_report(self, recv_to_play_ms: float, tail_ms: float | None = None) -> None:
        """客户端回传的播放侧测量（wire 帧 `playback.report`）。

        ⚠️ **位置是推算的，数值是客户端实测的**：服务端只知道首片何时落网线，
        把它当作这段时长的起点；客户端的时钟与我们不同步，所以只认时长不认时刻。
        落在 root 下（不是窗口）：它发生在口径B 终点**之后**，算进窗口等于把
        口径B 的右边界往后拖——盲区要看见，但不能靠污染主指标来看见。
        """
        self._playback_ms = float(recv_to_play_ms)
        if tail_ms is not None:
            self._playback_tail_ms = float(tail_ms)
        tracer = self.metrics.tracer
        if tracer is None or self._first_audio_sent_ms < 0:
            return
        tracer.record(
            "rx.playback",
            self._first_audio_sent_ms,
            self._first_audio_sent_ms + float(recv_to_play_ms),
            parent=tracer.root,
            attrs={"client_reported": 1},
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
        # 客户端回传的播放侧测量入账（取用即清：上一轮的出声延迟绝不能漂进
        # 下一轮的账单——错配比缺失更糟）。-1 = 客户端没报/不支持，如实缺失。
        if self._playback_ms >= 0:
            bill = {**bill, "playback_ms": round(self._playback_ms, 3)}
            if self._playback_tail_ms >= 0:
                bill = {**bill, "playback_tail_ms": round(self._playback_tail_ms, 3)}
            self._playback_ms = -1.0
            self._playback_tail_ms = -1.0
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
                # 整轮（含判停前那一段）的构成：上行帧 CPU / EOU 决策 / 等待空闲。
                # ⚠️ 它的 `unattributed_ms` **不等于漏埋**（里面天然有等上行帧的
                # 空闲），与窗口残差那条纪律不是一回事，见 `trace.root_breakdown`。
                # 只落盘不上 wire：格子比窗口多，打进 span.* 会让每轮键数翻倍。
                "root_breakdown": (
                    root_breakdown(self.metrics.tracer) if self.metrics.tracer is not None else None
                ),
            },
        )

    def _trim_history(self) -> None:
        """裁到最近 10 轮（20 条消息），从头部整对裁——保住 user 开头的轮次边界。"""
        if len(self._history) <= MAX_HISTORY_MESSAGES:
            return
        self._history[:] = self._history[-MAX_HISTORY_MESSAGES:]
        if self._history and self._history[0]["role"] != "user":
            del self._history[0]
