"""全局配置（pydantic settings，环境变量前缀 VOICE_）。"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from voice_pipeline.providers.pool import DEFAULT_MAX_SESSIONS


class Settings(BaseSettings):
    """服务与音频链路参数。所有延迟/分片参数集中在此，实验可整体覆写。"""

    model_config = SettingsConfigDict(env_prefix="VOICE_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    # 音频参数：上行 16k（ASR 习惯），下行 24k（kokoro 原生）
    in_sample_rate: int = 16000
    out_sample_rate: int = 24000
    up_chunk_ms: int = 100
    down_chunk_ms: int = 200

    # 背压：有界队列，音频积压超限时按策略丢弃并计数
    max_audio_frames: int = 200

    # WS 鉴权：设了则 session.begin 必须带匹配的 token，否则 1008 关闭；
    # 不设（默认）完全开放——本机开发 / bench / 测试零配置。
    # ⚠️ 开发机 .env 别设它，否则 bench_ws / pytest 全都要 token。
    ws_token: str | None = None

    # 浏览器客户端静态目录（默认仓根 web/；不存在的部署可指到别处或留空不挂载）
    web_dir: str = ""

    # 并发上限：同时占用本地推理（real 模式）的会话数。超出即明确拒绝
    # （error session_limit + 1013），不静默降级回共享推理——共享推理会让
    # 两个用户互相拖慢且毫无信号。为什么只卡 real：唯有 ORT session 是被独占的
    # 硬资源，sim/echo 连接不烧它。详见 providers/pool.py。
    max_sessions: int = DEFAULT_MAX_SESSIONS

    # A/B 测量开关：复现改造前「所有会话共用一份本地推理」的形态（并发即隐性串行）。
    # 存在的唯一理由是给 `scripts/bench_concurrency.py` 当对照组——同机同 provider，
    # 只拧这一个开关。生产不要开。
    shared_inference: bool = False


def get_settings() -> Settings:
    return Settings()
