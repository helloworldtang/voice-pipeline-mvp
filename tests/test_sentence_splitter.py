"""句子是怎么切的：句末标点处分句，逗号处坚决不切。

这是「文本一出来就丢 TTS 效果不好」（评论区润兮）的工程答案——半句进 TTS，语调会塌。
"""

from __future__ import annotations

from voice_pipeline.core.sentence_splitter import SentenceSplitter


def test_no_cut_on_comma() -> None:
    s = SentenceSplitter()
    assert s.feed("今天晴，") == []
    assert s.feed("气温 22 度。") == ["今天晴，气温 22 度。"]


def test_multiple_sentences_in_one_feed() -> None:
    s = SentenceSplitter()
    assert s.feed("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"]


def test_streaming_across_token_boundaries() -> None:
    """token 边界与句子边界不一致时也要拼对。"""
    s = SentenceSplitter()
    assert s.feed("上海") == []
    assert s.feed("今天") == []
    assert s.feed("多少度？") == ["上海今天多少度？"]


def test_semicolon_and_newline_are_terminators() -> None:
    s = SentenceSplitter()
    assert s.feed("先看这个；\n再看那个。") == ["先看这个；", "再看那个。"]


def test_flush_returns_trailing_fragment() -> None:
    s = SentenceSplitter()
    s.feed("还没说完")
    assert s.flush() == "还没说完"
    assert s.flush() is None


def test_english_terminators_supported() -> None:
    s = SentenceSplitter()
    assert s.feed("Hello! How are you?") == ["Hello!", "How are you?"]


def test_punctuation_only_piece_is_dropped() -> None:
    """裸标点不是句子：LLM 首 token 就吐「。」时（2026-10-17 验收 turn 12 实锤），
    纯标点 piece 发给 DashScope 直接 InvalidParameter 炸整轮。无声可言，丢弃。"""
    s = SentenceSplitter()
    assert s.feed("。") == []
    assert s.feed("。今天很冷。") == ["今天很冷。"]  # 前导裸标点被丢，句子正常
    assert s.feed("。。！") == []


def test_flush_drops_punctuation_only_remainder() -> None:
    s = SentenceSplitter()
    s.feed("今天很冷。")
    s.feed("。..")
    assert s.flush() is None  # 残留全是标点/空白：不产出
