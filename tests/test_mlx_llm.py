"""本地 LLM（MLX）：契约、线程桥、缺依赖时的失败姿态。

为什么不装 mlx 也能测：把假 `mlx_lm` 塞进 `sys.modules`，验的是**我们这层的纪律**——
1. 逐片吐出（调度器要增量，不是整段）；
2. system/history 走与云端同一套 `build_messages`（否则多轮会串味）；
3. 生成长于线程，异常必须被带回事件循环——卡在队列上就是一条永远挂住的轮次。
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

import voice_pipeline.providers.llm.mlx_llm as mlx_llm_mod
from voice_pipeline.providers.llm.mlx_llm import MlxLlm, _local_model_path


class _FakeTokenizer:
    def __init__(self) -> None:
        self.seen: list[dict[str, str]] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        self.seen = messages
        return "PROMPT"


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text


def _install(monkeypatch: pytest.MonkeyPatch, generate: Callable[..., Any]) -> _FakeTokenizer:
    tok = _FakeTokenizer()
    fake = types.ModuleType("mlx_lm")
    fake.load = lambda model: ("MODEL", tok)  # type: ignore[attr-defined]
    fake.stream_generate = generate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    # 2026-11-01 新纪律：repo id 不在本地缓存时不许联网（见 _local_model_path）。
    # 契约测试统一假装「模型已在本地」，路径解析本身的行为在下面单测。
    monkeypatch.setattr(mlx_llm_mod, "_local_model_path", lambda model: f"/fake/models/{model}")
    return tok


async def test_streams_chunks_and_keeps_system_and_history(monkeypatch: pytest.MonkeyPatch) -> None:
    def generate(model: Any, tokenizer: Any, prompt: str, max_tokens: int = 0) -> Any:
        assert prompt == "PROMPT"
        yield _FakeResp("今天")
        yield _FakeResp("晴。")

    tok = _install(monkeypatch, generate)
    llm = MlxLlm(model="test-model-a", system="SYS")
    chunks = [c async for c in llm.astream("天气", history=[{"role": "user", "content": "在吗"}])]
    assert "".join(chunks) == "今天晴。"
    # system 原文在前、日期锚定行动态追加在后（qwen_llm.build_messages 的契约）
    assert tok.seen[0]["content"].startswith("SYS\n今天的日期是")
    assert {"role": "user", "content": "在吗"} in tok.seen
    assert tok.seen[-1] == {"role": "user", "content": "天气"}


async def test_missing_mlx_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx_lm", None)  # 让 from mlx_lm import ... 抛 ImportError
    llm = MlxLlm(model="test-model-b")
    with pytest.raises(RuntimeError, match="mlx-lm"):
        async for _ in llm.astream("x"):
            pass


async def test_generation_error_surfaces_after_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """线程里炸了：已吐出的片段照常交付，异常必须抛出来（不能静默结束）。"""

    def generate(model: Any, tokenizer: Any, prompt: str, max_tokens: int = 0) -> Any:
        yield _FakeResp("半句")
        raise RuntimeError("metal oops")

    _install(monkeypatch, generate)
    llm = MlxLlm(model="test-model-c")
    got: list[str] = []
    with pytest.raises(RuntimeError, match="metal oops"):
        async for chunk in llm.astream("x"):
            got.append(chunk)
    assert got == ["半句"]


# ---- 模型解析：本地缓存优先，绝不联网（2026-11-01 门禁挂死实锤的回归钉） ----


def test_local_model_path_prefers_existing_dir(tmp_path: Any) -> None:
    assert _local_model_path(str(tmp_path)) == str(tmp_path)


def test_local_model_path_offline_cache_check_never_goes_to_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repo id 不在缓存 → 返回 None；且对 hub 只允许 `local_files_only=True` 的查询。

    修前的事故：`mlx_lm.load(repo_id)` 在未缓存时发无超时的 etag 网络请求，
    executor 线程永不退出，TestClient 收尾 join 不到 → 整套门禁 30s 超时。
    """

    class _FakeHub:
        calls: list[bool] = []

        @staticmethod
        def snapshot_download(repo_id: str, local_files_only: bool = False) -> str:
            _FakeHub.calls.append(local_files_only)
            raise RuntimeError("simulated: not in cache / offline")

    monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHub)
    assert _local_model_path("mlx-community/__never_cached_test__") is None
    assert _FakeHub.calls == [True]


def test_missing_model_fails_loudly_with_fetch_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """未缓存 → 响亮失败，报错里带 fetch 命令；绝不走到 load / 网络。"""
    fake = types.ModuleType("mlx_lm")

    def _no_load(model: Any) -> None:
        raise AssertionError("模型不在本地时不许调 load")

    fake.load = _no_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setattr(mlx_llm_mod, "_local_model_path", lambda model: None)
    llm = MlxLlm(model="test-model-not-cached")
    with pytest.raises(RuntimeError, match="fetch_model.py"):
        llm._ensure()


async def test_astream_surfaces_missing_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """astream 路径同样响亮失败：缺模型不许静默挂住首轮。"""
    fake = types.ModuleType("mlx_lm")
    fake.stream_generate = lambda *a, **k: iter(())  # type: ignore[attr-defined]  # import 要能过

    def _no_load(model: Any) -> None:
        raise AssertionError("模型不在本地时不许调 load")

    fake.load = _no_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setattr(mlx_llm_mod, "_local_model_path", lambda model: None)
    llm = MlxLlm(model="test-model-not-cached")
    with pytest.raises(RuntimeError, match="fetch_model.py"):
        async for _ in llm.astream("x"):
            pass
