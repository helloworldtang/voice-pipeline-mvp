"""句末标点切句器：文章「第一板」的落点。

规则：只在句末标点（。！？；\\n）处分句，**不在逗号处切**——半句送进 TTS 会让语调塌掉，
这是「文本一出来就丢 TTS 效果不好」（评论区润兮）的工程答案。
"""

from __future__ import annotations

DEFAULT_TERMINATORS = "。！？；!?;\n"


class SentenceSplitter:
    """增量切句：喂流式 token，吐完整句子。"""

    def __init__(self, terminators: str = DEFAULT_TERMINATORS) -> None:
        self._terminators = terminators
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        if not text:
            return []
        self._buffer += text
        sentences: list[str] = []
        start = 0
        for i, char in enumerate(self._buffer):
            if char in self._terminators:
                piece = self._buffer[start : i + 1].strip()
                if piece:
                    sentences.append(piece)
                start = i + 1
        if start:
            self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> str | None:
        """取回最后没有句末标点的残留（LLM 结束时调用）。"""
        rest = self._buffer.strip()
        self._buffer = ""
        return rest or None
