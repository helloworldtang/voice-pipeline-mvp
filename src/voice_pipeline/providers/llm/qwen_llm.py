"""DashScope（阿里云百炼）qwen 流式 LLM —— real 模式的真 LLM。

选型是实测出来的，不是查文档查出来的（2026-09-22，本机 httpx 直连 OpenAI 兼容端点，
5 轮中位数，探针脚本 `scripts/probe_llm.py` 可复跑）：

| 模型            | 首 token | 首完整句 | 波动    | 思考链 |
|-----------------|---------|---------|--------|-------|
| qwen-flash      | 367ms   | 522ms   | ±121ms | 0 字  |
| qwen-turbo      | 462ms   | 524ms   | ±67ms  | 0 字  |
| qwen-plus       | 561ms   | 747ms   | ±88ms  | 0 字  |
| qwen3.8-flash   | 705ms   | 2771ms  | —      | 92 字 |

**「首完整句」才是本链路的真实成本**：TTS 要攒到句末标点才开工（`SentenceSplitter`），
首 token 早到没用。同理，名字里带 flash 不等于不思考——`qwen3.8-flash` 是推理型，
首可播 2771ms，语音场景直接出局。

**为什么只转发 `content`、不碰 `reasoning_content`**：
推理模型会先吐一整条思考链，期间 `content` 恒为 null。思考文本不能念给用户听，
null 又意味着空转。本类只取 `content`，于是「换成了推理模型」这类配置错误会表现成
**久不出声**（可观测的失败），而不是**把思考过程念出来**（静默的错误答案）。
这是刻意的选择：失败要看得见。
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Iterable, Iterator

import httpx

logger = logging.getLogger(__name__)

API_KEY_ENV = "DASHSCOPE_API_KEY"
SYSTEM_ENV = "VOICE_LLM_SYSTEM"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-flash"

# 语音助理口径：答案越短，首完整句到得越早——这直接压的是口径B 里 LLM 那一格。
# 这是流程拆解后「唯一剩下的真实杠杆」（~79ms/字），所以必须能用环境变量
# VOICE_LLM_SYSTEM 调——否则每次实验都要改代码。
DEFAULT_SYSTEM = "你是语音助手。回答要短，一句话说完，不要 markdown，不要用列表。直接给结论。"


def _resolve_system(system: str | None) -> str:
    return system or os.environ.get(SYSTEM_ENV) or DEFAULT_SYSTEM


def iter_content(lines: Iterable[str]) -> Iterator[str]:
    """从 SSE 行流里抽出可播报文本（只认 `content`，忽略 `reasoning_content`）。

    抽成同步纯函数是为了能在不联网的情况下测「思考链被丢弃」这条纪律——
    这条纪律一旦破了，用户会听到模型的自言自语。
    """
    for line in lines:
        payload = line.strip()
        if not payload.startswith("data:"):
            continue
        body = payload[len("data:") :].strip()
        if not body:
            continue
        if body == "[DONE]":
            return
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError:
            continue
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield content


def build_messages(
    system: str, history: list[dict[str, str]] | None, prompt: str
) -> list[dict[str, str]]:
    """组装 messages：system + 历史轮次 + 本轮 user。

    抽成同步纯函数是为了在不联网的情况下测「历史插在 system 与 user 之间、原样透传」
    这条契约——history 的内容属于调用方（Session），这里不许增删改。
    """
    return [
        {"role": "system", "content": system},
        *(history or []),
        {"role": "user", "content": prompt},
    ]


class QwenLlm:
    """DashScope 的 OpenAI 兼容流式接口。契约见 `providers/base.py` 的 `LlmProvider`。"""

    # 模块级共享连接：每次 astream 新开 AsyncClient 意味着每轮对话都白付一次
    # TLS 握手（~100ms+，直接落在口径B 的 LLM 格里）。共享单例把这笔开销摊到
    # 进程生命周期。注意 AsyncClient 绑定创建它的事件循环——生产是单 uvicorn
    # loop，安全；测试零网络从不真正走 astream，不受影响。试点期不 aclose。
    _shared_client: httpx.AsyncClient | None = None

    @classmethod
    def _client(cls, timeout_s: float) -> httpx.AsyncClient:
        if cls._shared_client is None:
            cls._shared_client = httpx.AsyncClient(timeout=timeout_s)
        return cls._shared_client

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        system: str | None = None,
        max_tokens: int = 256,
        timeout_s: float = 30.0,
    ) -> None:
        resolved = api_key or os.environ.get(API_KEY_ENV, "")
        if not resolved:
            msg = (
                f"缺少 {API_KEY_ENV}：real 模式要真 LLM 就必须配它，否则端到端数字会少算 LLM 那一格"
            )
            raise RuntimeError(msg)
        self.model = model
        self._api_key = resolved
        self._base_url = base_url
        self._system = _resolve_system(system)
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s

    async def astream(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "stream": True,
            "max_tokens": self._max_tokens,
            "messages": build_messages(self._system, history, prompt),
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        client = self._client(self._timeout_s)
        async with client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                # 走 iter_content 而不是另写一份解析：运行时与测试共用同一条路径，
                # 「只认 content」这条纪律才不会被两处实现悄悄改掉一边。
                for piece in iter_content([line]):
                    yield piece


def build(model: str = DEFAULT_MODEL, **kwargs: object) -> QwenLlm:
    """工厂入口，与 `sim_llm.build` 同形。"""
    return QwenLlm(model, **kwargs)  # type: ignore[arg-type]
