"""首句护栏：只对「第 1 句」生效的软断点切分（默认关，`VOICE_FIRST_SENTENCE_MAX_CHARS` 开）。

为什么值得单独一组用例：这条护栏**违反**本仓「不在逗号处切」的既定决策，所以它的边界必须写死——
只切首句、只在首句确实超长时切、没有逗号就绝不切、下一轮 flush 后还能重新生效。
任何一条破了，都会变成"句子被切碎/语调塌"却没有报警。
"""

from __future__ import annotations

import pytest

from voice_pipeline.core.sentence_splitter import SentenceSplitter


def test_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICE_FIRST_SENTENCE_MAX_CHARS", raising=False)
    splitter = SentenceSplitter()
    assert splitter.first_sentence_max_chars == 0
    assert splitter.feed("今天上海多云，气温18到25度。") == ["今天上海多云，气温18到25度。"]


def test_cuts_first_sentence_at_comma() -> None:
    splitter = SentenceSplitter(first_sentence_max_chars=12)
    assert splitter.feed("今天上海多云，气温18到25度。") == ["今天上海多云，", "气温18到25度。"]


def test_cuts_before_terminator_arrives() -> None:
    """流式场景：不等句号，超长且遇到逗号就切——省下来的正是这一段时间。

    注意切点触发条件：缓冲**超过上限**才切（7 字的「今天上海多云，」还不够长，不切），
    下一块到达时缓冲 15 字 > 10 → 在逗号处吐首句，**句号还没到**。
    """
    splitter = SentenceSplitter(first_sentence_max_chars=10)
    assert splitter.feed("今天上海多云，") == []
    assert splitter.feed("气温18到25度") == ["今天上海多云，"]
    assert splitter.feed("。") == ["气温18到25度。"]


def test_no_soft_break_no_cut() -> None:
    splitter = SentenceSplitter(first_sentence_max_chars=10)
    assert splitter.feed("今天上海多云转晴气温二十五度。") == ["今天上海多云转晴气温二十五度。"]


def test_only_first_sentence() -> None:
    splitter = SentenceSplitter(first_sentence_max_chars=8)
    assert splitter.feed("好。第二句很长，带一个逗号。") == ["好。", "第二句很长，带一个逗号。"]


def test_guard_restored_after_flush() -> None:
    """轮末 flush 后要重新认「第 1 句」——否则第二轮的护栏会静默失效。"""
    splitter = SentenceSplitter(first_sentence_max_chars=10)
    assert splitter.feed("今天上海多云，气温18到25度。")
    splitter.flush()
    assert splitter.feed("明天上海小雨，气温17到22度。") == ["明天上海小雨，", "气温17到22度。"]


def test_env_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICE_FIRST_SENTENCE_MAX_CHARS", "9")
    assert SentenceSplitter().first_sentence_max_chars == 9
    monkeypatch.setenv("VOICE_FIRST_SENTENCE_MAX_CHARS", "not-a-number")
    assert SentenceSplitter().first_sentence_max_chars == 0
