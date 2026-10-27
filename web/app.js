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
  firstRecvCtx: null, // 本轮首个下行帧到达时刻（AudioContext 时钟；null=已测完/新轮）
  lastRecvToPlayMs: null, // 本轮已回传的「收到→可听」实测（打断轮复用）
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

// 最新一轮常驻可见：对话区是整页唯一的滚动容器（见 index.html 的 #log-wrap），
// 内容一多就会掉出视口。两条规则：
//   ① 追加前先量「用户是否贴着底部」，只有贴底才自动跟随——否则用户往上翻看
//      历史时会被强行拽回底部，回看根本没法用（验收时要回查前几轮的账）。
//   ② 滚到底要等浏览器排完版（rAF），否则量到的是加内容之前的高度。
function logWrap() { return $("log-wrap"); }

function nearBottom(el, threshold = 60) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function stickToBottom() {
  const el = logWrap();
  el.scrollTop = el.scrollHeight;
  $("to-latest").classList.remove("show");
}

function followLatest() {
  const el = logWrap();
  if (nearBottom(el)) {
    requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
  }
}

function addLine(cls, text) {
  const stick = nearBottom(logWrap()); // 必须在 appendChild 之前量
  const li = document.createElement("li");
  li.className = cls;
  li.textContent = text;
  $("log").appendChild(li);
  if (stick) requestAnimationFrame(() => { logWrap().scrollTop = logWrap().scrollHeight; });
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
  // 体感首音（含预合成 filler）：与口径B **分开报**——它省的是「以为在等」的那段，
  // 真实回答的首片一秒没早（见 core/filler.py 的计量纪律）。
  if (metrics.first_sound_ms >= 0) chips.push(["体感首音", Math.round(metrics.first_sound_ms) + "ms"]);
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

// ---- 每轮 span 拆解 ---------------------------------------------------------

// 服务端把口径B 窗口的自有时间摊成 `span.<name>` 键随 turn.end 下发（见
// server/session.py:_span_metrics）。页面只负责排序与展示——自有时间的口径
//（每毫秒只归最内层、重叠不重复计）不在前端重算，否则两边一定漂。
function spanRows(metrics) {
  return Object.keys(metrics)
    .filter((k) => k.startsWith("span.") && k !== "span.unattributed")
    .map((k) => ({ name: k.slice(5), ms: metrics[k] }))
    .filter((r) => Number.isFinite(r.ms))
    .sort((a, b) => b.ms - a.ms);
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function appendBreakdown(li, metrics) {
  const rows = spanRows(metrics);
  if (!rows.length) return;
  const total = metrics.caliber_b_ms >= 0 ? metrics.caliber_b_ms : rows.reduce((s, r) => s + r.ms, 0);
  const body = rows
    .map((r) => {
      const pct = total > 0 ? Math.round((r.ms / total) * 100) : 0;
      return `<tr><td>${esc(r.name)}</td><td class="num">${Math.round(r.ms)}ms</td><td class="num">${pct}%</td></tr>`;
    })
    .join("");
  const unattr = metrics["span.unattributed"];
  const unattrRow = Number.isFinite(unattr) && unattr >= 0.5
    ? `<tr class="unattr"><td>未归因</td><td class="num">${Math.round(unattr)}ms</td><td class="num">—</td></tr>`
    : "";
  const d = document.createElement("details");
  d.className = "breakdown";
  d.innerHTML =
    `<summary>拆解 ${Math.round(total)}ms · ${rows.length} 格</summary>` +
    `<table><thead><tr><th>span</th><th>耗时</th><th>占比</th></tr></thead>` +
    `<tbody>${body}${unattrRow}</tbody></table>`;
  li.appendChild(d);
}

// ---- 播放：24k AudioContext 背靠背调度 ---------------------------------------

function playPcm16(arrayBuffer) {
  const pcm = new Int16Array(arrayBuffer);
  if (pcm.length === 0) return;
  const ctx = state.playCtx;
  // 本轮首个下行帧到达的时刻（AudioContext 时钟，与服务端只对时长不对时刻）：
  // 排到出声的那一刻就能确定地算出 recv→play，不用等真的响
  if (state.firstRecvCtx === null) state.firstRecvCtx = ctx.currentTime;
  const buf = ctx.createBuffer(1, pcm.length, OUT_RATE);
  const data = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  // 落后（积压清空后重启）就贴着 now 重新起排，否则永远追不上
  if (state.nextStart < ctx.currentTime) state.nextStart = ctx.currentTime + 0.02;
  const startAt = state.nextStart;
  src.start(startAt);
  state.nextStart += buf.duration;
  state.playSources.push(src);
  src.onended = () => { state.playSources = state.playSources.filter((s) => s !== src); };
  // 首片的「收到→可听」实测（含 outputLatency）：值在排程那一刻就定了，
  // 立即回传，不等真响。浏览器不给 outputLatency 就只报排程差（如实偏小）。
  if (state.firstRecvCtx !== null && state.lastRecvToPlayMs === null) {
    const outLat = ctx.outputLatency || ctx.baseLatency || 0;
    const ms = Math.max(0, Math.round((startAt + outLat - state.firstRecvCtx) * 1000));
    state.lastRecvToPlayMs = ms;
    sendPlaybackReport(ms);
  }
}

function sendPlaybackReport(recvToPlayMs, tailMs) {
  // 播放侧回传（wire: playback.report）。发完即忘：服务端只是账单里的一格，
  // 丢了下一轮会重测，不值得为它造重试。
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  const frame = { type: "playback.report", recv_to_play_ms: recvToPlayMs };
  if (tailMs !== undefined && tailMs !== null) frame.tail_ms = tailMs;
  state.ws.send(JSON.stringify(frame));
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
    // 打断轮的完整回传：出声延迟（本轮已测过就复用）+ 被丢弃的排程尾音
    if (state.lastRecvToPlayMs !== null) sendPlaybackReport(state.lastRecvToPlayMs, tailMs);
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
    case "turn.start":
      setState("thinking", "思考中…"); lastAssistantLi = null;
      // 播放侧测量按轮清零：上一轮的出声延迟绝不带到下一轮（错配比缺失糟）
      state.firstRecvCtx = null; state.lastRecvToPlayMs = null;
      break;
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
        appendBreakdown(lastAssistantLi, m);
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
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
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
    // 换 mode/gating 主动重连时，旧 socket 的 close 不能砸掉新连接的状态
    if (state.ws !== ws) return;
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

// mode/gating 是 session.begin 的一次性参数：连接建立后再改下拉，对已建会话无效
// （2026-10-18 验收实锤：用户切到 real-pipelined，账单里三段会话仍全是 real-serial，
// 切换被静默吞掉）。未开麦时改选 → 重连换会话；开麦中改选 → 明说本轮仍用旧值。
for (const id of ["mode", "gating"]) {
  $(id).addEventListener("change", () => {
    if (state.micOn) {
      addLine("partial", `（${id === "mode" ? "模式" : "判停档"}已锁定本轮，关麦后再切换）`).className = "user";
      return;
    }
    state.reconnectAttempt = 0;
    state.ws?.close(1000, "mode-change");
    connect();
    $("status").textContent = `切换中… 目标 ${$("mode").value} / ${$("gating").value}`;
  });
}

renderScript();
connect();
