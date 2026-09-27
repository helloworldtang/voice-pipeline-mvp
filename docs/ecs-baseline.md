# ECS x86 基线实测与部署选型（2026-09-27）

机器：腾讯云广州 2C/2G（Xeon 8255C @2.5GHz）/ Ubuntu 24.04 / 50G 盘。
对照机：Apple M5 / 32GB。同一份代码与模型（transducer int8 + melo int8）。

## 逐项实测

| 项 | Apple M5 | ECS 2C x86 | 结论 |
|---|---|---|---|
| sherpa import（依赖链） | README 软链 dylib | 同坑 + 更深：需 libonnxruntime 1.28.2（PyPI 无 1.28.2，从 GitHub release 取 so），onnxruntime 钉上界 <1.29 | 已修（pyproject + 部署步骤） |
| ASR transducer 加载 | ~1s | **0.89s** | 可用 |
| ASR 流式 RTF | 0.008 | **0.044** | 可用（1.6s 音频 0.07s 喂完） |
| EOU SmartTurn | ~6ms/判 | 判停 100ms 档正常触发 | 可用 |
| TTS melo 12 字 | ~0.7s | **11.92s（RTF≈14.5）** | **判死** |
| TTS melo 2 字 | ~0.26s | 1.7s | 判死 |
| 内存 | — | 满载 available 最低 1.32G | 富余（瓶颈不在内存） |
| LLM qwen-flash | 首 token ~350ms | 云端不受影响 | — |

## 附带发现

1. **session.begin 阻塞接收循环 ~2.4-2.9s**（同步加载模型）。首轮 ready 迟到，
   喂进来的音频在 socket 排队。Provider 缓存（跨 begin 复用模型实例）能消掉
   重复加载；首次加载仍会阻塞，属已知代价。
2. bench_ws 全部 eou漏判/口径B=-1 的表象 = TTS 卡死（tts.sentence 出了、
   9s 无音频帧、无 turn.end），不是 EOU 的问题——EOU 实际正常触发。

## 选型判定（按计划 Phase 0 gate）

- 本地全模型：**不过**（TTS 单项就超出一切预算）
- 全上云（ASR+TTS 都换 DashScope）：**不必要**——ASR 本地 RTF 0.044 又快又免费
- **拍板：混合部署** —— 本地 ASR + 本地 EOU（SmartTurn）+ 云 LLM（qwen-flash，
  既有）+ **云 TTS（DashScope CosyVoice 流式，新增 provider）**
- 需要新写：`providers/tts/dashscope_tts.py`（流式）+ 调度器 streaming 分流 +
  factory 接线；`VOICE_MODEL_ROOT`/`VOICE_ESPEAK_DATA` env 接线

## 复现

```bash
# ECS 上（模型已就位）
uv run python - <<'EOF'   # 直接计时 melo
import time
from voice_pipeline.providers.sherpa import build_tts_engine
tts = build_tts_engine("melo", num_threads=4)
t0 = time.time(); tts.generate("今天天气晴朗，气温适宜。")
print(time.time() - t0)   # ≈ 11.9s
EOF
```
