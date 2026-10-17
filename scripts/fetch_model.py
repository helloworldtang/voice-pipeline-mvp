"""把模型拉到本地：**ModelScope 优先**，HF 兜底。

为什么：本机实测 HF 镜像的两个坑——
1. Xet 存储后端在镜像端返回 401（`cas-server.xethub.hf.co`），必须 `HF_HUB_DISABLE_XET=1`；
2. 关掉 Xet 后仍然慢，625MB 拉了很久。

ModelScope 上是**同名镜像**（`mlx-community/Qwen3.5-0.8B-4bit` 在两边是同一个仓库 id），
实测 100s / 14.5MB/s 下完，且 `model.safetensors` 的 sha256 与 HF 版**逐字节一致**，
所以不存在"国内源是另一个模型"的风险。

用法：
    uv run --with modelscope python scripts/fetch_model.py --repo mlx-community/Qwen3.5-0.8B-4bit
    # 打印本地目录，直接喂给链路：
    VOICE_LLM_MLX_MODEL=<打印出的路径> ./scripts/dev_server.sh
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo", required=True, help="仓库 id，两边同名，如 mlx-community/Qwen3.5-0.8B-4bit"
    )
    ap.add_argument("--no-hf", action="store_true", help="只用 ModelScope（HF 网络不通时）")
    args = ap.parse_args()

    try:
        from modelscope import snapshot_download as ms_download

        path = ms_download(args.repo)
        print(path)
        return
    except Exception as exc:  # noqa: BLE001 - 换源是正常路径，不是异常处理洁癖
        print(f"[modelscope] 失败 {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
        if args.no_hf:
            raise SystemExit(2) from exc

    from huggingface_hub import snapshot_download as hf_download

    path = hf_download(args.repo)
    print(path)


if __name__ == "__main__":
    main()
