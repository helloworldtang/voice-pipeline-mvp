// 采集侧降采样 worklet：浏览器原生采样率 → 16k PCM16，攒满 100ms（1600 样本）一帧。
// 线性插值足够——上行只是给 ASR 认字，不是听感场景。
// 额外职责：顺手算每帧 RMS（本地打断检测在主线程做，需要能量）。

class DownsampleWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = (options && options.processorOptions && options.processorOptions.targetRate) || 16000;
    this.frameSamples = Math.round(this.targetRate * 0.1); // 100ms/帧，与服务端 up_chunk_ms 对齐
    this._acc = new Float32Array(this.frameSamples);
    this._accLen = 0;
    this._pos = 0.0; // 源流重采样游标
    this._prev = null; // 上一个源样本（插值用）
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;
    const ch = input[0];

    if (this._prev === null) this._prev = ch[0];
    const step = sampleRate / this.targetRate;
    while (this._pos < ch.length) {
      const i = Math.floor(this._pos);
      const frac = this._pos - i;
      const cur = ch[i];
      const sample = this._prev + (cur - this._prev) * frac;
      this._acc[this._accLen++] = sample;
      if (this._accLen === this.frameSamples) {
        this._emit();
      }
      this._pos += step;
      this._prev = cur;
    }
    this._pos -= ch.length;
    this._prev = ch[ch.length - 1];
    return true;
  }

  _emit() {
    // Float32 → Int16 PCM，RMS 附带（打断检测用）
    const pcm = new Int16Array(this.frameSamples);
    let sumSq = 0;
    for (let i = 0; i < this.frameSamples; i++) {
      const s = Math.max(-1, Math.min(1, this._acc[i]));
      sumSq += s * s;
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    const rms = Math.sqrt(sumSq / this.frameSamples);
    this.port.postMessage({ pcm: pcm.buffer, rms }, [pcm.buffer]);
    this._accLen = 0;
  }
}

registerProcessor("downsample-worklet", DownsampleWorklet);
