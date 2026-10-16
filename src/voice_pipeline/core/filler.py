"""预合成 filler：EOU 判定瞬间先出声，把 TTS 首片那段固定成本盖住。

为什么需要它：TTS 首片 ~650ms 是口径B 里最大的一格（真人 26 轮占 63%），而实测它与
首句字数**无关**（两段独立样本斜率 +3 / −8 ms/字，见 `docs/优化空间-20261010.md` §一#6）
——它是固定成本。压不动就绕开：把预合成的短音频在判停成立那一刻先播出去，用户听到的
第一个声音从 ~判停+口径B（~1.5s）提前到 ~判停（~0.5s）。

⚠️ **计量纪律（本仓最忌讳的一类错）**：filler **不改口径B**。它打 `first_sound`，而
`first_audio_out` 仍然只指「真实回答的首片」——否则 `caliber_b_ms` 会从「说完→回答首音」
静默变成「说完→任意声音」，历史账与本轮的账就不可比了。对外必须把「体感首音」
（`first_sound_ms`）与「口径B」（`caliber_b_ms`）分开报。

格式：一个目录，里面 index.json + 若干 .pcm（24k PCM16 mono，与下行同格式、零转换）：

    {"sample_rate": 24000, "clips": [{"text": "嗯，我想一下。", "file": "abc123.pcm"}]}

用 `scripts/make_filler.py` 生成（同一个 TTS 引擎、同一个音色）。打开开关是
`VOICE_FILLER=1`（可选 `VOICE_FILLER_DIR` 指目录）。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 与账单文件同处一个家目录：不进仓、不随代码走
DEFAULT_FILLER_DIR = Path.home() / ".voice-pilot" / "filler"


@dataclass(slots=True, frozen=True)
class FillerClip:
    text: str
    pcm: bytes


class FillerBank:
    """读盘后的 filler 集合。`next_clip()` 轮转，避免每条都用同一句。"""

    def __init__(self, clips: list[FillerClip], directory: Path | None = None) -> None:
        self.clips = clips
        self.directory = directory
        self._next = 0

    def __len__(self) -> int:
        return len(self.clips)

    def next_clip(self) -> FillerClip | None:
        if not self.clips:
            return None
        clip = self.clips[self._next % len(self.clips)]
        self._next += 1
        return clip

    @classmethod
    def from_env(cls) -> FillerBank | None:
        """按环境变量加载：`VOICE_FILLER=1` 开，`VOICE_FILLER_DIR` 指定目录（默认见上）。

        为什么直接读 env 而不走 Settings：与 `metrics/jsonl.py` 的 `VOICE_EVENTS_JSONL`
        同一个理由——这是**运维/实验开关**，跟着自己的模块走，
        `VOICE_FILLER=1 ./scripts/dev_server.sh` 一条命令就能生效。
        """
        if os.environ.get("VOICE_FILLER", "").strip().lower() not in ("1", "true", "yes", "on"):
            return None
        return cls.load(os.environ.get("VOICE_FILLER_DIR") or DEFAULT_FILLER_DIR)

    @classmethod
    def load(cls, directory: str | Path) -> FillerBank | None:
        """按目录加载。**任何缺项都不抛异常**——没配、目录不在、index 坏了、clip 丢了，
        一律 warning + 返回 None：filler 是增益项，不该把会话拖垮（或者更糟：静默播一半）。
        """
        path = Path(directory).expanduser()
        index = path / "index.json"
        if not index.exists():
            logger.warning("filler 目录里没有 index.json（%s）——filler 关闭", path)
            return None
        try:
            meta = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("filler 的 index.json 读不了（%s）：%s——filler 关闭", index, exc)
            return None
        clips: list[FillerClip] = []
        for item in meta.get("clips", []):
            pcm_path = path / str(item.get("file", ""))
            try:
                pcm = pcm_path.read_bytes()
            except OSError as exc:
                logger.warning("filler 的 clip 读不了（%s）：%s——跳过该条", pcm_path, exc)
                continue
            if pcm:
                clips.append(FillerClip(text=str(item.get("text", "")), pcm=pcm))
        if not clips:
            logger.warning("filler 目录里没有可用 clip（%s）——filler 关闭", path)
            return None
        return cls(clips, directory=path)
