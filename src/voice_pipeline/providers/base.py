"""provider 协议接口：ASR / LLM / TTS / VAD 可插拔的契约。

real 与 sim 两套实现共用这些协议，A/B 实验只换实现不换调用方。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class AsrProvider(Protocol):
    async def push_audio(self, chunk: bytes) -> None: ...
    async def finalize(self) -> str: ...


@runtime_checkable
class LlmProvider(Protocol):
    def astream(self, prompt: str) -> AsyncIterator[str]: ...


@runtime_checkable
class TtsProvider(Protocol):
    def synth(self, sentence: str) -> AsyncIterator[bytes]: ...
