# voice-pilot 容器镜像（差距表「无 Docker / compose」的兑现）。
# 构建（GitHub 可达的机器上；ort so 走 GitHub release，国内网络用 --build-arg 换镜像）：
#   docker build -f deploy/Dockerfile -t voice-pilot .
# 运行见 deploy/compose.yml；模型不进镜像，挂载 /models。
# 基础镜像可换国内镜像源（示例）：
#   docker build --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.12-slim ...
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# espeak-ng-data：melo/kokoro 中文音素词典依赖（VOICE_ESPEAK_DATA 指到包路径）
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng-data curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV VOICE_ESPEAK_DATA=/usr/lib/x86_64-linux-gnu/espeak-ng-data

# sherpa-onnx 的 Linux wheel 链接 VERS_1.28.2 符号版本，PyPI 无此件（见 deploy/SETUP.md）
ARG ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.28.2/onnxruntime-linux-x64-1.28.2.tgz
ARG ORT_SHA=SKIP

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY web ./web
COPY scripts ./scripts
# 本地预置的 libonnxruntime（deploy/ort/，gitignore 不进仓；没有则回退 GitHub release）
COPY deploy/ort/ /tmp/ortlocal/

# uv 走 pip 安装（可指国内 PyPI 镜像）；GitHub 直连不可达的环境靠 ortlocal 免下载
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
RUN pip install --index-url "$PIP_INDEX_URL" uv \
    && if [ -f /tmp/ortlocal/libonnxruntime.so.1.28.2 ]; then \
           cp /tmp/ortlocal/libonnxruntime.so.1.28.2 /tmp/; \
       else \
           curl -sL "$ORT_URL" -o /tmp/ort.tgz \
           && ([ "$ORT_SHA" = SKIP ] || echo "$ORT_SHA  /tmp/ort.tgz" | sha256sum -c -) \
           && tar xzf /tmp/ort.tgz -C /tmp --strip-components=2 \
              onnxruntime-linux-x64-1.28.2/lib/libonnxruntime.so.1.28.2; \
       fi \
    && UV_DEFAULT_INDEX="$PIP_INDEX_URL" uv sync --frozen --no-dev \
    && SITE=/app/.venv/lib/python3.12/site-packages \
    && cp /tmp/libonnxruntime.so.1.28.2 "$SITE/sherpa_onnx/lib/" \
    && ln -sf "$SITE/sherpa_onnx/lib/libonnxruntime.so.1.28.2" \
              "$SITE/sherpa_onnx/lib/libonnxruntime.so" \
    && uv run python -c "import sherpa_onnx; print('sherpa OK')"

ENV PATH="/app/.venv/bin:$PATH" \
    VOICE_MODEL_ROOT=/models \
    VOICE_PIPELINE_TTS_ENGINE=dashscope \
    VOICE_EVENTS_JSONL=/var/log/voice-pipeline/turns.jsonl
# 账单写在容器层会被 `docker rm` 带走：compose.yml 把 ../logs 挂进来做持久化。

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=2)"
CMD ["uvicorn", "voice_pipeline.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
