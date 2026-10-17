"""句末标点切句器：文章「第一板」的落点。

规则：只在句末标点（。！？；\\n）处分句，**不在逗号处切**——半句送进 TTS 会让语调塌掉，
这是「文本一出来就丢 TTS 效果不好」（评论区润兮）的工程答案。

**唯一的例外：首句护栏（2026-10-17）**。首音等的是「第一句完整」，所以首句越长，首音越晚
（实测 ≈4~8ms/字）。而 prompt 控不住字数：模型会把「今天上海多云，气温18到25度。」当成
**一句**（逗号不是句末），于是首句仍是 15~16 字。护栏的做法是：**首句超过 N 字时，退一步在
第一个逗号处切一刀**——只对首句生效，后面几句不动（切碎后面只会让语调更差，且对口径B 无益）。

默认关闭（0）。打开：`VOICE_FIRST_SENTENCE_MAX_CHARS=12`。代价是首句在逗号处收尾，语调会略塌。
"""

from __future__ import annotations

import logging
import os
import unicodedata

logger = logging.getLogger(__name__)

DEFAULT_TERMINATORS = "。！？；!?;\n"
# 首句护栏的软断点：只在首句超长时退而求其次（逗号/顿号）
SOFT_BREAKS = "，,、"
ENV_FIRST_MAX = "VOICE_FIRST_SENTENCE_MAX_CHARS"


def _has_speech(text: str) -> bool:
    """有没有可念的内容：字母/数字/汉字（Unicode L*/N*）之外的都算标点或空白。

    为什么需要它：LLM 流式吐出的第一个 token 可能就是一个裸「。」——
    2026-10-17 真人验收 turn 12 实锤，纯标点「句子」发给 DashScope 直接
    InvalidParameter("input text is valid")，异常炸穿整轮，用户一个字听不到。
    纯标点无声可言，丢弃是对的。
    """
    return any(unicodedata.category(ch)[0] in ("L", "N") for ch in text)


def _env_first_max() -> int:
    raw = os.environ.get(ENV_FIRST_MAX, "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("%s=%r 不是整数，首句护栏关闭", ENV_FIRST_MAX, raw)
        return 0


class SentenceSplitter:
    """增量切句：喂流式 token，吐完整句子。"""

    def __init__(
        self,
        terminators: str = DEFAULT_TERMINATORS,
        *,
        first_sentence_max_chars: int | None = None,
    ) -> None:
        """first_sentence_max_chars：0/None = 关闭；None 时读 `VOICE_FIRST_SENTENCE_MAX_CHARS`。

        为什么默认读环境变量而不是从调用方传：`SerialScheduler.cancel()` 会**重建**
        `SentenceSplitter()`（打断后复位）——走构造默认值，重建出来的实例才带着同一份配置，
        不会在第一次打断后就悄悄退回关闭。
        """
        self._terminators = terminators
        self._buffer = ""
        self._first_max = (
            _env_first_max()
            if first_sentence_max_chars is None
            else max(0, first_sentence_max_chars)
        )
        self._soft_breaks = SOFT_BREAKS
        self._emitted = 0  # 本轮已吐出的句数：护栏只认第 1 句

    @property
    def first_sentence_max_chars(self) -> int:
        return self._first_max

    def _soft_cut(self, text: str) -> int | None:
        """找第一个软断点，返回切点（含标点）；超过上限或没有软断点 → None。

        两条自我约束：① 切点必须 ≤ 上限——超了就宁可等句号，也不切一个长从句；
        ② 只认**第一个**软断点——挑最靠后的会让「首句」失去意义。
        """
        for i, char in enumerate(text):
            if char in self._soft_breaks:
                return i + 1 if i + 1 <= self._first_max else None
        return None

    def _scan(self) -> list[str]:
        """按句末标点切，返回本轮新吐出的句子。

        纯标点/空白的 piece 丢弃（无声可言，发给 TTS 只会换来 InvalidParameter）。
        """
        sentences: list[str] = []
        start = 0
        for i, char in enumerate(self._buffer):
            if char in self._terminators:
                piece = self._buffer[start : i + 1].strip()
                if piece and _has_speech(piece):
                    sentences.append(piece)
                start = i + 1
        if start:
            self._buffer = self._buffer[start:]
        return sentences

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text
        sentences = self._scan()
        if self._first_max and self._emitted == 0:
            if sentences:
                # 句末标点先到：整句超长时再补一刀（切完 head 立即能合成）
                cut = self._soft_cut(sentences[0])
                if cut is not None:
                    head = sentences[0][:cut].strip()
                    tail = sentences[0][cut:].strip()
                    rest = sentences[1:]
                    sentences = ([head] if head else []) + ([tail] if tail else []) + rest
            elif len(self._buffer.strip()) > self._first_max:
                # 还在流式：已经超过上限，不等句号，先在逗号处切
                cut = self._soft_cut(self._buffer)
                if cut is not None:
                    head = self._buffer[:cut].strip()
                    self._buffer = self._buffer[cut:]
                    if head:
                        sentences.append(head)
        self._emitted += len(sentences)
        return sentences

    def flush(self) -> str | None:
        """取回最后没有句末标点的残留（LLM 结束时调用）。纯标点残留丢弃。"""
        rest = self._buffer.strip()
        self._buffer = ""
        self._emitted = 0  # 轮末：下一轮重新认「第 1 句」
        return rest if _has_speech(rest) else None
