"""全局配置（pydantic settings，环境变量前缀 VOICE_）。"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


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


def get_settings() -> Settings:
    return Settings()
