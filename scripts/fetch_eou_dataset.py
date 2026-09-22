"""下载官方判停测试集并解码成 npy，供 `bench_eou_controlled.py` / `bench_eou_official.py` 用。

为什么不在 bench 脚本里直接读 parquet：数据集 4.3GB，解码（flac → 16k float32）之后
每片只剩零头，落盘一次后面反复跑实验才划算。产物 **不入仓**（439MB，.gitignore 忽略
data/），所以「怎么拿到它」必须写成脚本。

依赖：pyarrow + soundfile + soxr（dev 组，评测专用，运行时不需要）。

用法：
  uv run python scripts/fetch_eou_dataset.py                # 3 个分片（约 1.4GB 下载）
  uv run python scripts/fetch_eou_dataset.py --shards 0     # 只要一个分片
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import soxr

REPO = "pipecat-ai/smart-turn-data-v3.1-test"
DEFAULT_MIRROR = "https://hf-mirror.com"
SR = 16000
LANGS = ("zho", "eng")  # 中文是本项目目标语言；英文作「模型没坏」的对照


def download(shard: int, mirror: str, cache: Path) -> Path:
    name = f"train-{shard:05d}-of-00009.parquet"
    dest = cache / name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  复用缓存 {dest.name}")
        return dest
    url = f"{mirror}/datasets/{REPO}/resolve/main/data/{name}"
    print(f"  下载 {url}")
    with urllib.request.urlopen(url, timeout=600) as r:  # noqa: S310
        dest.write_bytes(r.read())
    if dest.stat().st_size < 1_000_000:
        msg = f"{dest.name} 只有 {dest.stat().st_size} 字节 —— 多半拿到了重定向页"
        raise SystemExit(msg)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="0,1,2")
    ap.add_argument("--mirror", default=DEFAULT_MIRROR)
    ap.add_argument("--cache", default="data/parquet-cache")
    ap.add_argument("--out", default="data/eou-test")
    ap.add_argument("--eng-per-shard", type=int, default=150)
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta: list[dict] = []
    for shard in [int(s) for s in args.shards.split(",")]:
        path = download(shard, args.mirror, cache)
        t = pq.read_table(
            str(path), columns=["audio", "language", "endpoint_bool", "dataset", "synthetic"]
        )
        lang = t.column("language").to_pylist()
        labels = t.column("endpoint_bool").to_pylist()
        dsets = t.column("dataset").to_pylist()
        synth = t.column("synthetic").to_pylist()
        picked: dict[str, list[int]] = {
            "zho": [i for i, x in enumerate(lang) if x == "zho"],
            "eng": [i for i, x in enumerate(lang) if x == "eng"][: args.eng_per_shard],
        }
        print(f"  分片 {shard}: " + ", ".join(f"{k}={len(v)}" for k, v in picked.items()))
        for target, idxs in picked.items():
            (out / target).mkdir(parents=True, exist_ok=True)
            for i in idxs:
                a = t.column("audio")[i].as_py()
                d, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                if d.ndim > 1:
                    d = d.mean(axis=1)
                if sr != SR:
                    d = soxr.resample(d, sr, SR).astype(np.float32)
                n = len(meta)
                np.save(out / target / f"{n:04d}.npy", d.astype(np.float32))
                meta.append(
                    {
                        "lang": target,
                        "i": n,
                        "shard": shard,
                        "label": bool(labels[i]),
                        "dur_s": round(len(d) / SR, 2),
                        "dataset": dsets[i],
                        "synthetic": bool(synth[i]),
                    }
                )

    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"\n{len(meta)} 条 -> {out}")
    print(collections.Counter((m["lang"], m["label"]) for m in meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
