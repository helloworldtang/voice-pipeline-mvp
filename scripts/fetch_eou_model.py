"""下载 Smart Turn v3.2（神经判停器权重，8MB）到 models/eou/。

为什么单独一个脚本：权重 8MB 不入仓（.gitignore 忽略 models/），但「本机怎么拿到它」
必须可复现 —— 否则下一个人跑 bench 时会静默退回 conservative，得到启发式的数字
还以为是神经档的（`build_eou` 会告警，但告警不是复现步骤）。

用法：
  uv run python scripts/fetch_eou_model.py
  uv run python scripts/fetch_eou_model.py --mirror https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "pipecat-ai/smart-turn-v3"
FILE = "smart-turn-v3.2-cpu.onnx"
DEFAULT_MIRROR = "https://hf-mirror.com"  # huggingface.co 在本机不可达
EXPECTED_BYTES = 8679182  # 下载后自检：大小不对就是拿到了 LFS 指针或错误页


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default=DEFAULT_MIRROR)
    ap.add_argument("--out", default="models/eou")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / FILE
    url = f"{args.mirror}/{REPO}/resolve/main/{FILE}"

    print(f"下载 {url}\n  → {dest}")
    with urllib.request.urlopen(url, timeout=300) as r:  # noqa: S310
        data = r.read()
    if len(data) != EXPECTED_BYTES:
        msg = f"大小 {len(data)} != 期望 {EXPECTED_BYTES}：多半拿到了重定向页或 LFS 指针"
        raise SystemExit(msg)
    dest.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()[:16]
    print(f"OK {len(data)} bytes  sha256[:16]={sha}")
    print("\n验证：uv run python scripts/bench_eou_controlled.py --langs zho")
    return 0


if __name__ == "__main__":
    sys.exit(main())
