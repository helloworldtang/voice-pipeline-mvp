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

// ---- 验收剧本（句子是实测选定的：kokoro 合成→真 ASR 回灌打分，满分句优先，
// 抢话句要求两段都可识别——2026-10-04 测量） ------------------------------------
const SCRIPT = [
  { type: "基准", text: "今天上海天气怎么样" },
  { type: "追问", text: "那明天呢" },
  { type: "基准", text: "明天要下雨吗" },
  { type: "基准", text: "给我讲个笑话" },
  { type: "基准", text: "你会说英语吗" },
  { type: "抢话", text: "我想订……（停半秒）……明天下午的会议室" },
  { type: "抢话", text: "帮我查一下……（停半秒）……去北京的火车票" },
  { type: "抢话", text: "我等会儿……（停半秒）……还要开个会" },
  { type: "打断", text: "随便问一句，AI 播报中直接说「等一下」" },
  { type: "打断", text: "再问一句，播报中插话「停一下」" },
];
let scriptIdx = 0;

function renderScript() {
  const ol = $("script-steps");
  ol.innerHTML = "";
  SCRIPT.forEach((step, i) => {
    const li = document.createElement("li");
    li.className = "step" + (i === scriptIdx ? " current" : "");
    li.innerHTML = `<span class="tag">${step.type}</span>${step.text}`;
    li.onclick = () => { scriptIdx = i; renderScript(); };
    ol.appendChild(li);
  });
}
function advanceScript() {
  if (scriptIdx < SCRIPT.length - 1) { scriptIdx += 1; renderScript(); }
}

// ---- 状态指示 -----------------------------------------------------------------
// 绿=聆听 黄=思考 蓝=播报：抢话/打断测试都要求用户知道 AI 此刻在干什么，
// 没有状态点就只能靠耳朵猜——验收体验的第一刚需。
function setState(kind, text) {
  const dot = $("dot");
  dot.className = kind || "";
  if (text) $("status").textContent = text;
}

// ---- 转写区 -----------------------------------------------------------------

function addLine(cls, text) {
  const li = document.createElement("li");
  li.className = cls;
  li.textContent = text;
  $("log").appendChild(li);
  return li;
}

let pendingPartial = null;
let lastAssistantLi = null;

function setPartial(text) {
  if (!text) return; // 空 partial 不占行
  if (!pendingPartial || !pendingPartial.isConnected) pendingPartial = addLine("partial", "");
  pendingPartial.textContent = text;
}

function settlePartial(finalText) {
  // final 落定：灰行升级为正式行（同一句话不再显示两遍）
  if (pendingPartial && pendingPartial.isConnected && finalText) {
    pendingPartial.className = "user";
    pendingPartial.textContent = finalText;
    pendingPartial = null;
    return;
  }
  pendingPartial = null;
  if (finalText) addLine("user", finalText);
}

let lastTailMs = null; // 停播队列残余（打断取证），跟随下一轮 chips 一起呈现

function setChips(metrics) {
  const chips = [];
  if (metrics.caliber_b_ms >= 0) chips.push(["口径B", Math.round(metrics.caliber_b_ms) + "ms"]);
  else if (metrics.barge_in_count > 0) chips.push(["打断轮", "无首音"]);
  if (metrics.eou_latency_ms >= 0) chips.push(["判停", Math.round(metrics.eou_latency_ms) + "ms"]);
  if (metrics.premature_eou === 1.0) chips.push(["抢话", "已记"]);
  if (metrics.barge_in_count > 0) chips.push(["打断", `×${metrics.barge_in_count}`]);
  const llm = metrics.llm_first_token - metrics.utterance_end;
  if (Number.isFinite(llm) && metrics.llm_first_token >= 0) chips.push(["LLM首token", Math.round(llm) + "ms"]);
  const tts = metrics.tts_first_chunk - metrics.first_sentence_done;
  if (Number.isFinite(tts) && metrics.tts_first_chunk >= 0) chips.push(["TTS首片", Math.round(tts) + "ms"]);
  if (lastTailMs !== null) chips.push(["丢弃队列", lastTailMs + "ms"]);
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
  // 尾音量化（差距表「打断尾音未量化」，2026-10-03）：停播时刻播放队列里
  // 还排着多少未出声的音频。nextStart 是背靠背排程的末端，与 currentTime
  // 之差 = 打断发生时「已被服务端下发、但还没轮到出声」的总量——被丢弃的
  // 正是这一段。真正的尾音（OS 音频管线里已在途的毫秒级）测不到，如实留白。
  const ctx = state.playCtx;
  if (ctx && state.nextStart > ctx.currentTime) {
    const tailMs = Math.round((state.nextStart - ctx.currentTime) * 1000);
    console.info(`[voice-pilot] 停播：队列残余 ${tailMs}ms（未出声即被丢弃）`);
    lastTailMs = tailMs;
  }
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
  setState("listening", "聆听中——请说话");
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
        setState("", "就绪——点「开麦」开始");
        $("mic").disabled = false;
      }
      break;
    case "turn.start": setState("thinking", "思考中…"); lastAssistantLi = null; break;
    case "asr.partial": setPartial(ev.text); break;
    case "asr.final":
      settlePartial(ev.text);
      if (ev.text) advanceScript(); // 念完一句（转写非空）自动亮下一句
      break;
    case "tts.sentence": lastAssistantLi = addLine("assistant", ev.text); break;
    case "turn.end": {
      const m = ev.metrics || {};
      // 耗时是验收指标，跟着对话行走（一眼可查），chips 仍是最近一轮的完整账
      if (lastAssistantLi) {
        const lat = document.createElement("span");
        lat.className = "lat";
        lat.title = "口径B：EOU 判定说完 → AI 首音到达（不含说话时长与判停等待）";
        lat.textContent = m.caliber_b_ms >= 0
          ? `⏱${Math.round(m.caliber_b_ms)}ms`
          : m.barge_in_count > 0 ? "⏱—（被打断）" : "";
        lastAssistantLi.appendChild(lat);
      }
      setChips(m);
      setState("listening", "聆听中");
      break;
    }
    case "audio.stop":
      stopPlayback();
      // 打断轮服务端不补 turn.end（轮次作废），耗时标注挂这里
      if (lastAssistantLi) {
        const lat = document.createElement("span");
        lat.className = "lat";
        lat.textContent = "（被打断）";
        lastAssistantLi.appendChild(lat);
      }
      setState("listening", "已打断，聆听中");
      break;
    case "error":
      addLine("err", `错误 ${ev.code}: ${ev.message}`);
      break;
    case "session.closed":
      setState("", "会话已关闭");
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
      setState("speaking", "播报中");
      playPcm16(msg.data);
    }
  };
  state.ws.onclose = () => {
    setState("", "连接断开");
    $("mic").disabled = true;
    stopPlayback();
    // 自动重连：指数退避，上限 15s。手动关麦不重连。
    const delay = Math.min(15000, 1000 * 2 ** state.reconnectAttempt++);
    setState("", `断开，${Math.round(delay / 1000)}s 后重连…`);
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

renderScript();
connect();
