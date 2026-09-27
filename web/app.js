// voice-pilot 浏览器客户端：vanilla JS，无构建步骤。
// 协议契约对齐 scripts/live_client.py：二进制=PCM16 音频帧，JSON=事件/控制帧。
// 握手时序（必须遵守）：连上 → 收首个 session.ready → 发 session.begin
// → 等 rebound session.ready → 才推音频（服务端 begin 时会重建会话，
// 抢跑的帧会落进旧会话的 outbox 丢失）。

"use strict";

const $ = (id) => document.getElementById(id);
const OUT_RATE = 24000;
const INTERRUPT_RMS = 0.05; // 与 live_client.py 一致；浏览器 AEC 后可再调
const INTERRUPT_FRAMES = 3;

const state = {
  ws: null,
  micOn: false,
  micCtx: null,
  playCtx: null,
  playSources: [],
  nextStart: 0,
  recentRms: [],
  interruptArmed: true,
  reconnectAttempt: 0,
  beginSent: false,
};

// ---- 转写区 -----------------------------------------------------------------

function addLine(cls, text) {
  const li = document.createElement("li");
  li.className = cls;
  li.textContent = text;
  $("log").appendChild(li);
  return li;
}

function setPartial(text) {
  let el = document.querySelector("#log .partial:last-of-type");
  if (!el) el = addLine("partial", "");
  el.textContent = text;
}

function setChips(metrics) {
  const chips = [];
  if (metrics.caliber_b_ms >= 0) chips.push(["口径B", Math.round(metrics.caliber_b_ms) + "ms"]);
  if (metrics.eou_latency_ms >= 0) chips.push(["判停", Math.round(metrics.eou_latency_ms) + "ms"]);
  const llm = metrics.llm_first_token - metrics.utterance_end;
  if (Number.isFinite(llm) && metrics.llm_first_token >= 0) chips.push(["LLM首token", Math.round(llm) + "ms"]);
  const tts = metrics.tts_first_chunk - metrics.first_sentence_done;
  if (Number.isFinite(tts) && metrics.tts_first_chunk >= 0) chips.push(["TTS首片", Math.round(tts) + "ms"]);
  $("chips").innerHTML = chips.map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join("");
}

// ---- 播放：24k AudioContext 背靠背调度 ---------------------------------------

function playPcm16(arrayBuffer) {
  const pcm = new Int16Array(arrayBuffer);
  if (pcm.length === 0) return;
  const ctx = state.playCtx;
  const buf = ctx.createBuffer(1, pcm.length, OUT_RATE);
  const data = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  // 落后（积压清空后重启）就贴着 now 重新起排，否则永远追不上
  if (state.nextStart < ctx.currentTime) state.nextStart = ctx.currentTime + 0.02;
  src.start(state.nextStart);
  state.nextStart += buf.duration;
  state.playSources.push(src);
  src.onended = () => { state.playSources = state.playSources.filter((s) => s !== src); };
}

function stopPlayback() {
  state.playSources.forEach((s) => { try { s.stop(); } catch { /* already ended */ } });
  state.playSources = [];
  state.nextStart = 0;
}

// ---- 采集：getUserMedia(AEC) + worklet 降采样 16k -----------------------------

async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const ctx = new AudioContext(); // 采集用原生采样率，worklet 里降到 16k
  state.micCtx = ctx;
  await ctx.audioWorklet.addModule("/downsample-worklet.js");
  const node = new AudioWorkletNode(ctx, "downsample-worklet");
  node.port.onmessage = (ev) => {
    const { pcm, rms } = ev.data;
    if (state.ws && state.ws.readyState === WebSocket.OPEN && state.beginSent) {
      state.ws.send(pcm);
    }
    // 本地打断：AI 播放中连续 3 帧 RMS 超阈值 → interrupt（服务端 L1 兜底）
    state.recentRms.push(rms);
    if (state.recentRms.length > INTERRUPT_FRAMES) state.recentRms.shift();
    if (
      state.interruptArmed &&
      state.playSources.length > 0 &&
      state.recentRms.length === INTERRUPT_FRAMES &&
      state.recentRms.every((r) => r >= INTERRUPT_RMS)
    ) {
      state.interruptArmed = false;
      state.recentRms = [];
      stopPlayback();
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ type: "interrupt", reason: "client-barge-in" }));
      }
      addLine("partial", "（打断）").className = "user";
    }
    if (state.recentRms.length === 0 || rms < INTERRUPT_RMS) state.interruptArmed = true;
  };
  ctx.createMediaStreamSource(stream).connect(node);
  state.micOn = true;
  $("mic").textContent = "关麦";
  $("mic").classList.add("on");
}

function stopMic() {
  if (state.micCtx) { state.micCtx.close().catch(() => {}); state.micCtx = null; }
  state.micOn = false;
  $("mic").textContent = "开麦";
  $("mic").classList.remove("on");
}

// ---- WS 事件 -----------------------------------------------------------------

function handleEvent(ev) {
  switch (ev.type) {
    case "session.ready":
      if (!state.beginSent) {
        // 首个 ready（默认会话）→ 发 begin，等 rebound ready 才推音频
        state.ws.send(JSON.stringify({
          type: "session.begin",
          mode: $("mode").value,
          gating: $("gating").value,
          token: $("token").value || null,
        }));
        state.beginSent = true;
        $("status").textContent = "已连接，等待会话…";
      } else {
        $("status").textContent = "就绪";
        $("mic").disabled = false;
      }
      break;
    case "asr.partial": setPartial(ev.text); break;
    case "asr.final": addLine("user", ev.text); break;
    case "tts.sentence": addLine("assistant", ev.text); break;
    case "turn.end": setChips(ev.metrics || {}); break;
    case "audio.stop": stopPlayback(); break;
    case "error":
      addLine("err", `错误 ${ev.code}: ${ev.message}`);
      break;
    case "session.closed":
      $("status").textContent = "会话已关闭";
      break;
    default: break;
  }
}

// audio.chunk 事件帧后面跟一帧二进制音频——binaryType=arraybuffer 时
// onmessage 的 data 是 ArrayBuffer，文本帧走 JSON。注意：audio.chunk 的 JSON
// 与其音频二进制之间可能插入其他事件？不会——服务端 pump 串行发送且紧邻。

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws.binaryType = "arraybuffer";
  state.beginSent = false;

  state.ws.onopen = () => { $("status").textContent = "已连接"; };
  state.ws.onmessage = (msg) => {
    if (typeof msg.data === "string") {
      let ev;
      try { ev = JSON.parse(msg.data); } catch { return; }
      handleEvent(ev);
    } else {
      playPcm16(msg.data);
    }
  };
  state.ws.onclose = () => {
    $("status").textContent = "连接断开";
    $("mic").disabled = true;
    stopPlayback();
    // 自动重连：指数退避，上限 15s。手动关麦不重连。
    const delay = Math.min(15000, 1000 * 2 ** state.reconnectAttempt++);
    $("status").textContent = `断开，${Math.round(delay / 1000)}s 后重连…`;
    if (state.micOn) setTimeout(connect, delay);
  };
  state.ws.onerror = () => { /* onclose 会接手 */ };
}

// ---- UI 绑定 -------------------------------------------------------------------

$("mic").addEventListener("click", async () => {
  if (state.micOn) { stopMic(); return; }
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.playCtx = state.playCtx || new AudioContext({ sampleRate: OUT_RATE });
  await state.playCtx.resume();
  try {
    await startMic();
    state.reconnectAttempt = 0;
  } catch (err) {
    addLine("err", `开麦失败: ${err.message}`);
  }
});

connect();
