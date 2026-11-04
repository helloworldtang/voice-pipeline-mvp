# ECS 首次部署（腾讯云 2C/2G / Ubuntu 24.04）

一次性的环境准备，之后的发版走 `deploy/release.sh`。

## 1. 系统依赖

```bash
sudo apt update && sudo apt install -y espeak-ng espeak-ng-data
curl -LsSf https://astral.sh/uv/install.sh | sh   # ~/.local/bin/uv
```

## 2. 代码与模型

```bash
# 本机执行（模型从 Mac 带过去，ECS 不走 HF）
rsync -az --exclude .venv --exclude .git --exclude data --exclude bench \
  --exclude models/tts --exclude models/tts-matcha \
  ./ ubuntu@ECS:~/voice-pipeline-mvp/
```

需要的模型目录：`asr-zipformer/`（默认 ASR 引擎，别漏）、`eou/`、`vad/`。
`asr/`（paraformer）可选；本地 TTS 模型不需要（走 DashScope）。

## 3. libonnxruntime（sherpa-onnx Linux wheel 的坑）

sherpa-onnx 的 Linux wheel 链接 `VERS_1.28.2` 符号版本，但 PyPI 上
onnxruntime 1.28 系只有 1.28.0（符号版本不符）。解法：从 onnxruntime
GitHub release 取 1.28.2 的 so 放进 sherpa 的 lib 目录（Mac 中转）：

```bash
# 本机
curl -sL -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.28.2/onnxruntime-linux-x64-1.28.2.tgz
tar xzf /tmp/ort.tgz -C /tmp onnxruntime-linux-x64-1.28.2/lib/libonnxruntime.so.1.28.2
scp /tmp/onnxruntime-linux-x64-1.28.2/lib/libonnxruntime.so.1.28.2 ubuntu@ECS:/tmp/

# ECS（uv sync 之后）
cd ~/voice-pipeline-mvp && ~/.local/bin/uv sync
SITE=.venv/lib/python3.12/site-packages
cp /tmp/libonnxruntime.so.1.28.2 $SITE/sherpa_onnx/lib/
ln -sf $SITE/sherpa_onnx/lib/libonnxruntime.so.1.28.2 $SITE/sherpa_onnx/lib/libonnxruntime.so
~/.local/bin/uv run python -c "import sherpa_onnx; print('OK')"
```

注意：`uv sync` 重装 sherpa-onnx 包会清掉这个文件，`release.sh` 每次会重做软链，
但 `.so.1.28.2` 本体被清掉时需重新 cp（release.sh 会报错提醒）。

## 4. 密钥与 token

```bash
sudo tee /etc/voice-pilot.env >/dev/null <<'EOF'
DASHSCOPE_API_KEY=sk-...
VOICE_WS_TOKEN=自定一个随机串
EOF
sudo chmod 600 /etc/voice-pilot.env
```

（`VOICE_EVENTS_JSONL` 不走这里——它在 `voice-pipeline.service` 里已经写死，
改路径要改 service 并 `daemon-reload`。）

## 5. systemd + caddy

```bash
sudo cp ~/voice-pipeline-mvp/deploy/voice-pipeline.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now voice-pipeline

# caddy 已在跑（Ubuntu 自带配置），换成反代：
sudo tee /etc/caddy/Caddyfile >/dev/null < ~/voice-pipeline-mvp/deploy/Caddyfile
sudo systemctl reload caddy
```

## 6. 账单日志（可观测性，判据 6）

`VOICE_EVENTS_JSONL` 不在 service 里设的话**一行都不落**——判据 6 就只是纸面达标。
service 里已写死 `/home/ubuntu/voice-pipeline-mvp/logs/turns.jsonl`，落盘侧要配轮转：

```bash
sudo tee /etc/logrotate.d/voice-pilot >/dev/null <<'EOF'
/home/ubuntu/voice-pipeline-mvp/logs/turns.jsonl {
    daily
    rotate 7
    missingok
    notifempty
    copytruncate
    su ubuntu ubuntu
}
EOF
sudo logrotate -f /etc/logrotate.d/voice-pilot   # 立刻验证一次，别等明天
```

为什么必须轮转：一轮一行，行内含完整 span 树（每个上行帧 3~4 个 span × 250~300 帧），
单行可达百 KB；真人连续对讲一天能写几百 MB，`copytruncate` 是因为进程只做追加写、
不重开文件（切走 inode 会继续写旧文件）。

验它真的在落：

```bash
# 跑一轮之后
jq -c '{turn, caliber_b_ms, eou_latency_ms, spans: (.spans|length)}' \
  ~/voice-pipeline-mvp/logs/turns.jsonl | tail -3
```

## 7. 验收

```bash
# 下文用 $HOST 指代你的服务器（例：export HOST=ubuntu@1.2.3.4）
curl -s http://127.0.0.1:8000/healthz          # {"status":"ok"}
curl -s "http://${HOST#*@}/"                   # voice-pilot 页面 HTML
# 本机开麦（secure context）：浏览器访问 http://localhost:8888
ssh -L 8888:localhost:8000 "$HOST"
```
