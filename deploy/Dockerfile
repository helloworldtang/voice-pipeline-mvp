# voice-pilot 容器镜像（差距表「无 Docker / compose」的兑现）。
# 构建（GitHub 可达的机器上；ort so 走 GitHub release，国内网络用 --build-arg 换镜像）：
#   docker build -f deploy/Dockerfile -t voice-pilot .
# 运行见 deploy/compose.yml；模型不进镜像，挂载 /models。
FROM python:3.12-slim

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

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && export PATH="/root/.local/bin:$PATH" \
    && curl -sL "$ORT_URL" -o /tmp/ort.tgz \
    && ([ "$ORT_SHA" = SKIP ] || echo "$ORT_SHA  /tmp/ort.tgz" | sha256sum -c -) \
    && tar xzf /tmp/ort.tgz -C /tmp --strip-components=2 \
       onnxruntime-linux-x64-1.28.2/lib/libonnxruntime.so.1.28.2 \
    && uv sync --frozen --no-dev \
    && SITE=/app/.venv/lib/python3.12/site-packages \
    && cp /tmp/libonnxruntime.so.1.28.2 "$SITE/sherpa_onnx/lib/" \
    && ln -sf "$SITE/sherpa_onnx/lib/libonnxruntime.so.1.28.2" \
              "$SITE/sherpa_onnx/lib/libonnxruntime.so" \
    && uv run python -c "import sherpa_onnx; print('sherpa OK')"

ENV PATH="/app/.venv/bin:$PATH" \
    VOICE_MODEL_ROOT=/models \
    VOICE_PIPELINE_TTS_ENGINE=dashscope

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=2)"
CMD ["uvicorn", "voice_pipeline.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
