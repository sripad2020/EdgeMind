"use strict";
// AIR-SDK Chat UI. Security: all dynamic text is inserted with textContent (never innerHTML),
// so model output and server data cannot inject markup or script.

const $ = (id) => document.getElementById(id);
const state = { history: [], sessionReady: false, busy: false, hasGpu: false, conversationId: newId() };

// The server keeps this conversation's sectioned memory under this random id.
function newId() {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

// ---------- helpers ----------
function el(tag, opts = {}, children = []) {
  const node = document.createElement(tag);
  if (opts.cls) node.className = opts.cls;
  if (opts.text !== undefined && opts.text !== null) node.textContent = String(opts.text);
  if (opts.title) node.title = opts.title;
  for (const c of children) if (c) node.appendChild(c);
  return node;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
function fmtBytes(b) {
  if (b === null || b === undefined) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; let v = Number(b);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}
function fmtNum(n, d = 2) { return n === null || n === undefined ? "—" : Number(n).toFixed(d); }
function kv(target, pairs) {
  clear(target);
  for (const [k, v] of pairs) {
    target.appendChild(el("dt", { text: k }));
    target.appendChild(el("dd", { text: v === null || v === undefined || v === "" ? "—" : v }));
  }
}
function chip(text, cls = "") { return el("span", { cls: `chip ${cls}`.trim(), text }); }
function deviceOf(cfg) { return cfg && cfg.n_gpu_layers > 0 ? `GPU (${cfg.n_gpu_layers} layers)` : "CPU"; }

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) {
    const detail = data && (data.error || (Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail));
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return data;
}

function setStatus(text, mode) {
  $("status-text").textContent = text;
  const dot = $("session-status").querySelector(".dot");
  dot.className = `dot ${mode}`;
}
function setBusy(b) {
  state.busy = b;
  $("send-btn").disabled = b || !state.sessionReady;
  $("start-btn").disabled = b;
  $("prompt").disabled = !state.sessionReady;
}

// ---------- hardware & models ----------
async function loadHardware() {
  try {
    const hw = await api("/api/hardware");
    const gpus = hw.gpus || [];
    state.hasGpu = gpus.length > 0;
    const pairs = [
      ["CPU", hw.cpu], ["Cores", `${hw.physical_cores} physical / ${hw.logical_cores} logical`],
      ["RAM", `${hw.available_ram_gib} / ${hw.total_ram_gib} GiB free`], ["OS", hw.os],
    ];
    if (gpus.length) for (const g of gpus) pairs.push([`GPU ${g.index}`, `${g.name} · ${g.free_vram_gib}/${g.total_vram_gib} GiB`]);
    else pairs.push(["GPU", "None supported (NVIDIA/NVML)"]);
    kv($("hardware"), pairs);
    $("device-gpu").disabled = !state.hasGpu;
    $("gpu-hint").textContent = state.hasGpu
      ? "GPU layers are chosen by the SDK; 'GPU' forces the maximum safe offload."
      : "No supported NVIDIA GPU detected: models run on the CPU. GPU mode activates automatically on machines with an NVIDIA GPU.";
  } catch (e) {
    kv($("hardware"), [["Error", e.message]]);
  }
}

async function loadModels() {
  const sel = $("model");
  clear(sel);
  try {
    const data = await api("/api/models");
    if (!data.models.length) {
      sel.appendChild(el("option", { text: "No chat models in Ollama" }));
      $("start-btn").disabled = true;
      return;
    }
    for (const m of data.models) {
      const o = el("option", { text: `${m.name}  ·  ${m.parameters} ${m.quantization}` });
      o.value = m.name;
      sel.appendChild(o);
    }
    const small = data.models.find((m) => /0\.5b|1b|1\.5b/i.test(m.name));
    if (small) sel.value = small.name;
  } catch (e) {
    sel.appendChild(el("option", { text: "Ollama not reachable" }));
    $("session-error").textContent = `Cannot list Ollama models: ${e.message}. Is Ollama running?`;
  }
}

// ---------- session ----------
async function startSession(ev) {
  ev.preventDefault();
  $("session-error").textContent = "";
  const device = document.querySelector("input[name=device]:checked").value;
  const body = {
    model: $("model").value,
    device,
    objective: $("objective").value,
    rl_enabled: $("rl_enabled").checked,
    auto_adapt: $("auto_adapt").checked,
    n_ctx: parseInt($("n_ctx").value, 10),
    adapt_every: Math.max(1, Math.min(100, parseInt($("adapt_every").value, 10) || 1)),
    probe: $("probe").checked,
    auto_context: $("auto_context").checked,
  };
  setBusy(true);
  setStatus(`Loading ${body.model}…`, "busy");
  try {
    const s = await api("/api/session", body);
    state.sessionReady = true;
    renderPlan(s.plan);
    renderProfile(s.profile, s.variants);
    renderQuantPlan(s.quantization);
    renderAdaptation(s.performance);
    resetChat();
    setStatus(`${body.model} · ${deviceOf(s.plan.config)} · ${body.objective}`, "ready");
    await refreshInsights();
  } catch (e) {
    state.sessionReady = false;
    $("session-error").textContent = e.message;
    setStatus("No model loaded", "idle");
  } finally {
    setBusy(false);
  }
}

function renderPlan(plan) {
  if (!plan) return;
  const c = plan.config;
  kv($("plan"), [
    ["Chosen by", { learned: "past learning", heuristic: "hardware prediction", probe: "speed test", conservative: "safety fallback" }[plan.source] || plan.source],
    ["Precision", c.precision], ["KV cache", c.kv_type],
    ["Device", deviceOf(c)], ["Threads", c.n_threads], ["Batch", c.n_batch], ["Context", c.n_ctx],
    ["Est. RAM", fmtBytes(plan.estimated_ram_bytes)], ["Est. VRAM", fmtBytes(plan.estimated_vram_bytes)],
    ["Quality-loss prior", fmtNum(plan.quality_loss_prior, 4)], ["Candidates", plan.candidate_count],
  ]);
  const f = plan.fallback;
  kv($("fallback"), [["Precision", f.precision], ["Device", deviceOf(f)], ["Threads", f.n_threads], ["Batch", f.n_batch], ["KV cache", f.kv_type]]);
}

function renderProfile(p, variants) {
  kv($("profile"), [
    ["Name", p.name], ["Architecture", p.architecture], ["Precision", p.precision],
    ["Parameters", Number(p.parameters).toLocaleString()], ["Layers", p.layers], ["Hidden size", p.hidden_size],
    ["Attention / KV heads", `${p.attention_heads} / ${p.kv_heads}`], ["Feed-forward", p.feed_forward],
    ["Vocabulary", Number(p.vocab).toLocaleString()], ["Max context", Number(p.max_context).toLocaleString()],
    ["Weights", fmtBytes(p.weight_bytes)], ["KV / token (f16)", fmtBytes(p.kv_bytes_per_token_f16)], ["Size class", p.size_class],
  ]);
  const ul = $("variants");
  clear(ul);
  for (const v of variants) ul.appendChild(el("li", { text: `${v.precision} · ${fmtBytes(v.weight_bytes)}` }));
}

// ---------- chat ----------
function resetChat() {
  const old = state.conversationId;
  if (state.history.length) fetch(`/api/conversations/${old}`, { method: "DELETE" }).catch(() => {});
  state.conversationId = newId();
  state.history = [];
  clear($("messages"));
  const empty = el("div", { cls: "empty" }, [
    el("p", { cls: "big", text: "Ready." }),
    el("p", { text: "Each reply is one reinforcement-learning episode: the agent picks a configuration, measures speed, latency and memory, and updates its Q-table." }),
  ]);
  $("messages").appendChild(empty);
  $("prompt").disabled = !state.sessionReady;
  $("send-btn").disabled = !state.sessionReady;
  if (state.sessionReady) $("prompt").focus();
}

function addMessage(role, text) {
  const empty = $("messages").querySelector(".empty");
  if (empty) empty.remove();
  const bubble = el("div", { cls: "bubble", text });
  const wrap = el("div", { cls: `msg ${role}` }, [bubble]);
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
  return { wrap, bubble };
}

// ---------- verified skills ----------
const SKILL_CONTEXT = {
  auto: ["Facts (optional)", "e.g. CREATE TABLE … — remembered for this conversation"],
  sql: ["Database schema", "CREATE TABLE customers (id INTEGER, name TEXT, city TEXT); …"],
  json: ["Required keys", "name, age, email   (or an example JSON object)"],
  math: ["Expected answer (optional)", "e.g. 391"],
};

function updateSkillContext() {
  const info = SKILL_CONTEXT[$("skill").value];
  $("ctx-wrap").hidden = !info;
  if (info) {
    $("ctx-label").textContent = info[0];
    $("skill-context").placeholder = info[1];
  }
}

function skillChips(s) {
  const chips = el("div", { cls: "chips" });
  if (s.verified === true) chips.appendChild(chip(`✓ ${s.name} check passed`, "good"));
  else if (s.verified === false) chips.appendChild(chip(`✗ ${s.name} check failed: ${s.check}`, "bad"));
  else chips.appendChild(chip(`${s.name} (no automatic check)`));
  if (s.escalated) chips.appendChild(chip(`path: ${s.attempts.map((a) => `${a.precision}${a.ok ? " ✓" : " ✗"}`).join(" → ")}`, "warn"));
  return chips;
}

function contextChips(ctx, skill) {
  const chips = el("div", { cls: "chips" });
  chips.appendChild(chip(`Section: ${ctx.section}`, "strong"));
  chips.appendChild(chip(`task ${skill.name}${skill.auto ? " (detected)" : ""} · ${ctx.route}`));
  const b = ctx.budget;
  chips.appendChild(chip(`${b.facts_sent} facts · ${b.turns_sent} turns · ~${b.prompt_tokens}/${b.budget} tokens`));
  if (b.rung !== "full") chips.appendChild(chip(`memory trimmed: ${b.rung}`, "warn"));
  if (ctx.correction) chips.appendChild(chip("correction noted", "warn"));
  if (b.notice) chips.appendChild(chip(b.notice, "bad"));
  return chips;
}

function metricChips(m) {
  const c = m.config;
  const chips = el("div", { cls: "chips" });
  chips.appendChild(chip(`${fmtNum(m.tokens_per_second, 1)} tok/s`, "strong"));
  chips.appendChild(chip(`TTFT ${fmtNum(m.ttft_s, 2)}s`));
  chips.appendChild(chip(`${c.precision} · ${deviceOf(c)}`));
  chips.appendChild(chip(`${c.n_threads} threads · batch ${c.n_batch}`));
  chips.appendChild(chip(`RAM ${fmtBytes(m.peak_ram_bytes)}`));
  if (m.peak_vram_bytes) chips.appendChild(chip(`VRAM ${fmtBytes(m.peak_vram_bytes)}`));
  chips.appendChild(chip(`${m.action}${m.explored ? " (explore)" : ""}`, m.explored ? "warn" : ""));
  if (m.relief) chips.appendChild(chip("switched: memory got tight", "warn"));
  if (m.prefetched && m.switch_s < 0.05) chips.appendChild(chip("config prepared in background", "good"));
  if (m.switch_s >= 0.05) chips.appendChild(chip(`waited ${fmtNum(m.switch_s, 2)}s for reload`, "warn"));
  if (m.reward !== null) chips.appendChild(chip(`reward ${m.reward >= 0 ? "+" : ""}${fmtNum(m.reward, 3)}`, m.reward >= 0 ? "good" : "bad"));
  else if (!m.learned) chips.appendChild(chip("not scored (short reply or RL off)"));
  if (m.rolled_back) chips.appendChild(chip("rolled back after failure", "bad"));
  return chips;
}

async function sendMessage(ev) {
  ev.preventDefault();
  const text = $("prompt").value.trim();
  if (!text || state.busy || !state.sessionReady) return;
  $("prompt").value = "";
  addMessage("user", text);
  state.history.push({ role: "user", content: text });
  const { wrap, bubble } = addMessage("assistant", "");
  bubble.classList.add("streaming");
  setBusy(true);
  setStatus("Generating…", "busy");
  let answer = "";
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        // The server remembers the conversation (by id): send only the new message.
        messages: [{ role: "user", content: text }],
        conversation_id: state.conversationId,
        max_tokens: parseInt($("max_tokens").value, 10) || 512,
        temperature: $("temperature").value === "" ? null
          : Math.max(0, Math.min(2, parseFloat($("temperature").value) || 0)),
        skill: $("skill").value,
        context: $("skill-context").value.trim() || null,
      }),
    });
    if (!res.ok || !res.body) {
      let msg = `HTTP ${res.status}`;
      try { const d = await res.json(); msg = d.error || (d.detail && JSON.stringify(d.detail)) || msg; } catch { /* ignore */ }
      throw new Error(msg);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let finished = false;
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
        const evName = (raw.match(/^event: (.*)$/m) || [])[1];
        const dataLine = (raw.match(/^data: (.*)$/m) || [])[1];
        if (!evName || dataLine === undefined) continue;
        const data = JSON.parse(dataLine);
        if (evName === "token") {
          answer += data; bubble.textContent = answer;
          $("messages").scrollTop = $("messages").scrollHeight;
        } else if (evName === "reset") {
          answer = ""; bubble.textContent = "";
        } else if (evName === "escalate") {
          // The answer failed its check: discard it and show what happens next.
          answer = "";
          bubble.textContent = "";
          wrap.insertBefore(el("div", { cls: "chips" }, [
            chip(`check failed on ${data.from}: ${data.why}`, "bad"),
            chip(`retrying on ${data.to}`, "warn"),
          ]), bubble);
        } else if (evName === "done") {
          answer = data.text; bubble.textContent = answer;
          if (data.context) wrap.appendChild(contextChips(data.context, data.skill));
          if (data.skill && (data.skill.name !== "general" || data.skill.verified !== null)) wrap.appendChild(skillChips(data.skill));
          wrap.appendChild(metricChips(data.metrics));
          renderLastMetrics(data.metrics);
          finished = true;
        } else if (evName === "error") {
          throw new Error(data.message);
        }
      }
    }
    state.history.push({ role: "assistant", content: answer });
  } catch (e) {
    wrap.classList.add("error");
    bubble.textContent = `Error: ${e.message}`;
    state.history.pop(); // drop the unanswered user turn so the history stays valid
  } finally {
    bubble.classList.remove("streaming");
    setBusy(false);
    setStatus($("status-text").textContent.startsWith("Generating") ? "Ready" : $("status-text").textContent, "ready");
    refreshInsights();
  }
}

function renderLastMetrics(m) {
  const c = m.config;
  kv($("last-metrics"), [
    ["Decode speed", `${fmtNum(m.tokens_per_second, 1)} tok/s`], ["Time to first token", `${fmtNum(m.ttft_s, 3)} s`],
    ["Total time", `${fmtNum(m.total_s, 2)} s`], ["Prompt / completion tokens", `${m.prompt_tokens} / ${m.completion_tokens}`],
    ["Peak RAM / VRAM", `${fmtBytes(m.peak_ram_bytes)} / ${fmtBytes(m.peak_vram_bytes)}`],
    ["Action", `${m.action}${m.explored ? " (exploration)" : " (exploitation)"}`],
    ["Reward", m.reward === null ? (m.learned ? "—" : "not scored") : fmtNum(m.reward, 4)],
    ["Reload before reply", m.switch_s > 0 ? `${fmtNum(m.switch_s, 2)} s` : "none"],
    ["Config", `${c.precision} · kv ${c.kv_type} · ${deviceOf(c)} · ${c.n_threads}t · b${c.n_batch}`],
    ["Rolled back", m.rolled_back ? "yes" : "no"],
  ]);
}

// ---------- insights ----------
async function refreshInsights() {
  if (!state.sessionReady) return;
  try {
    const [rep, pol] = await Promise.all([api("/api/reports"), api("/api/policy")]);
    if (rep.plan) renderPlan(rep.plan);
    const perf = rep.performance;
    if (perf.current_config) {
      const c = perf.current_config;
      kv($("plan"), [
        ["Source", rep.plan ? rep.plan.source : "—"], ["Precision", c.precision], ["KV cache", c.kv_type],
        ["Device", deviceOf(c)], ["Threads", c.n_threads], ["Batch", c.n_batch], ["Context", c.n_ctx],
        ["Est. RAM", fmtBytes(perf.estimated_memory && perf.estimated_memory.ram_bytes)],
        ["Est. VRAM", fmtBytes(perf.estimated_memory && perf.estimated_memory.vram_bytes)],
        ["Candidates", rep.plan ? rep.plan.candidate_count : "—"],
      ]);
    }
    renderLearning(rep);
    renderPolicy(pol);
    renderAdaptation(perf);
  } catch (e) {
    /* insights are best-effort; the chat keeps working */
  }
}

function renderLearning(rep) {
  const l = rep.learning, p = rep.performance;
  kv($("learning"), [
    ["Episodes", l.episodes], ["Exploration ε", fmtNum(l.epsilon, 3)], ["States learned", l.q_states],
    ["Mean reward (earlier)", fmtNum(l.mean_reward_early, 4)], ["Mean reward (recent)", fmtNum(l.mean_reward_recent, 4)],
    ["Blacklisted configs", l.blacklisted_configs],
  ]);
  kv($("performance"), [
    ["Requests", p.requests], ["Mean decode speed", `${fmtNum(p.mean_tokens_per_second, 1)} tok/s`],
    ["Mean TTFT", `${fmtNum(p.mean_ttft_s, 3)} s`], ["Failures", p.failures], ["Rollbacks", p.rollbacks],
  ]);
  const ol = $("top-configs");
  clear(ol);
  for (const t of l.top_configs || []) {
    let c = {};
    try { c = JSON.parse(t.config); } catch { /* keep empty */ }
    ol.appendChild(el("li", { text: `${fmtNum(t.mean_reward, 3)} (${t.samples}×) · ${c.precision} · kv ${c.kv_type} · ${deviceOf(c)} · ${c.n_threads}t · b${c.n_batch}` }));
  }
  drawSparkline(rep.rewards || []);
}

function drawSparkline(values) {
  const host = $("sparkline");
  clear(host);
  if (!values.length) { host.appendChild(el("p", { cls: "hint", text: "No episodes yet." })); return; }
  const NS = "http://www.w3.org/2000/svg";
  const W = 340, H = 90, P = 6;
  const lo = Math.min(0, ...values), hi = Math.max(0, ...values);
  const span = hi - lo || 1;
  const x = (i) => P + (values.length === 1 ? (W - 2 * P) / 2 : (i * (W - 2 * P)) / (values.length - 1));
  const y = (v) => H - P - ((v - lo) * (H - 2 * P)) / span;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Reward over the last ${values.length} episodes`);
  const zero = document.createElementNS(NS, "line");
  zero.setAttribute("x1", P); zero.setAttribute("x2", W - P);
  zero.setAttribute("y1", y(0)); zero.setAttribute("y2", y(0));
  zero.setAttribute("class", "zero");
  svg.appendChild(zero);
  const path = document.createElementNS(NS, "polyline");
  path.setAttribute("points", values.map((v, i) => `${x(i)},${y(v)}`).join(" "));
  path.setAttribute("class", "line");
  svg.appendChild(path);
  values.forEach((v, i) => {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", x(i)); c.setAttribute("cy", y(v)); c.setAttribute("r", 2.2);
    c.setAttribute("class", v <= -4.99 ? "pt fail" : "pt");
    const t = document.createElementNS(NS, "title"); t.textContent = `episode ${i + 1}: ${v}`;
    c.appendChild(t); svg.appendChild(c);
  });
  host.appendChild(svg);
}

function renderPolicy(p) {
  const head = $("policy-head"), table = $("policy-table");
  clear(head); clear(table);
  if (!p || !p.actions) { head.appendChild(chip("not optimized")); return; }
  for (const part of p.state.split("|")) {
    if (/^(v|kv|g|t|b)=/.test(part)) continue;
    head.appendChild(chip(part.replace("=", ": ")));
  }
  head.appendChild(chip(`ε ${fmtNum(p.epsilon, 3)}`, "strong"));
  head.appendChild(chip(`greedy → ${p.greedy_action}`, "strong"));
  const thead = el("tr", {}, [el("th", { text: "Action" }), el("th", { text: "Q-value" }), el("th", { text: "Visits" })]);
  table.appendChild(el("thead", {}, [thead]));
  const tbody = el("tbody");
  for (const a of p.actions) {
    const name = el("td", {}, [el("span", { text: a.action.replaceAll("_", " ").toLowerCase() })]);
    if (a.allowed && a.target) {
      const t = a.target;
      name.appendChild(el("span", { cls: "tgt", text: `${t.precision} · kv ${t.kv_type} · ${deviceOf(t)} · ${t.n_threads}t · b${t.n_batch}` }));
    } else if (!a.allowed) {
      name.appendChild(el("span", { cls: "why", text: `blocked: ${(a.blocked_reason || "unsafe").replaceAll("_", " ")}` }));
    }
    const q = el("td", { cls: "num", text: a.allowed ? `${fmtNum(a.q_value, 3)}${a.learned ? "" : "*"}` : "—", title: a.learned ? "learned Q-value" : "prior (from observed config rewards)" });
    const tr = el("tr", {}, [name, q, el("td", { cls: "num", text: a.visits })]);
    if (!a.allowed) tr.className = "blocked";
    if (a.action === p.greedy_action) tr.className = "greedy";
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  table.appendChild(el("caption", { cls: "hint", text: "* prior estimate (action not yet tried in this state)" }));
}

// ---------- adaptation & quantization ----------
function renderAdaptation(p) {
  if (!p) return;
  kv($("adaptation"), [
    ["Background switching", p.background_switching ? "on" : "off"],
    ["Replies with config prepared in background", p.background_prepared],
    ["Total time waiting for reloads", `${fmtNum(p.total_wait_for_reloads_s, 2)} s`],
    ["Memory-pressure switches", p.memory_pressure_reliefs],
    ["Memory estimate calibration", p.memory_estimate_scale === null ? "—" : `× ${fmtNum(p.memory_estimate_scale, 2)} on overhead`],
    ["Context in use / max", `${p.context_active ?? "—"} / ${p.context_max ?? "—"} tokens${p.auto_context ? " (grows when needed)" : ""}`],
    ["Context grown", `${p.context_growths} time(s)`],
  ]);
  for (const r of p.probe || []) {
    $("adaptation").appendChild(el("dt", { text: `Speed test: ${r.precision}` }));
    $("adaptation").appendChild(el("dd", { text: `${fmtNum(r.tokens_per_second, 1)} tok/s${r.chosen ? "  ← chosen" : ""}` }));
  }
  const ul = $("events");
  clear(ul);
  for (const e of (p.events || []).slice().reverse()) {
    const t = new Date(e.time * 1000).toLocaleTimeString();
    const text = e.event === "context_grow"
      ? `${t} · conversation grew: context ${e.from} → ${e.to} tokens`
      : `${t} · memory got tight (${e.when}): ${e.from} → ${e.to}`;
    ul.appendChild(el("li", { text }));
  }
}

function renderQuality(q) {
  const dl = $("quality");
  clear(dl);
  $("quality-error").textContent = "";
  if (!q) return;
  if (q.note) { kv(dl, [["Note", q.note]]); return; }
  const pairs = [["Reference (original)", q.reference]];
  for (const r of q.results) {
    pairs.push([r.model, `${Math.round(r.agreement * 100)}% same words · loss ${fmtNum(r.measured_loss, 4)} ${r.within_limit ? "✓ within limit" : "✗ over limit (excluded)"}`]);
  }
  for (const s of q.skipped || []) pairs.push([s, "skipped: does not fit free RAM now"]);
  kv(dl, pairs);
}

function sessionEnded(msg) {
  state.sessionReady = false;
  setStatus(msg, "idle");
  $("prompt").disabled = true;
  $("send-btn").disabled = true;
}

async function runQuality() {
  const model = quantModel || $("model").value;
  if (!model) return;
  const btn = $("quality-btn");
  btn.disabled = true; setBusy(true);
  btn.textContent = "Measuring… (about a minute)";
  try {
    renderQuality(await api("/api/quality", { model }));
    sessionEnded("Measurement done: load the model again to chat");
  } catch (e) {
    $("quality-error").textContent = e.message;
  } finally {
    btn.disabled = false; btn.textContent = "Measure answer quality"; setBusy(false);
  }
}

let quantModel = null;
function renderQuantPlan(q) {
  $("quant-error").textContent = "";
  const btn = $("quantize-btn");
  if (!q) { btn.hidden = true; return; }
  quantModel = q.model;
  const pairs = [["Source precision", q.source_precision], ["Free RAM for a model now", fmtBytes(q.free_ram_budget_bytes)]];
  for (const t of q.targets) {
    const status = t.installed ? "installed" : (t.fits_free_ram_now ? "can create · fits now" : "can create · needs more free RAM");
    pairs.push([`${t.quant.toUpperCase()} (${fmtBytes(t.estimated_bytes)})`, status]);
  }
  if (q.recommended) pairs.push(["Recommended for this machine now", q.recommended]);
  kv($("quant-plan"), pairs);
  const missing = q.targets.filter((t) => !t.installed);
  btn.hidden = !q.eligible || missing.length === 0;
  $("quant-note").textContent = q.eligible
    ? (missing.length ? "Creates smaller, faster versions of this model on your machine (Q8_0 ≈ same quality, Q4_0 ≈ 2–2.5× faster on CPU). Reload the session afterwards so the RL engine can use them."
                      : "All quantized versions exist. Load the session again to let the RL engine switch between them.")
    : (q.note || "");
}

async function runQuantize() {
  if (!quantModel) return;
  const btn = $("quantize-btn");
  btn.disabled = true; setBusy(true);
  $("quant-error").textContent = "";
  btn.textContent = "Quantizing… (this can take a minute)";
  try {
    const r = await api("/api/quantize", { model: quantModel });
    renderQuantPlan(r.plan);
    renderQuality(r.quality);
    const made = r.results.filter((x) => x.created).map((x) => `${x.name} (${fmtBytes(x.size_bytes)}, ${x.seconds}s)`);
    $("quant-note").textContent = made.length ? `Created: ${made.join(", ")}. Quality measured below. Click "Load & optimise" again to use them.` : "Nothing new to create.";
    sessionEnded("Quantization done: load the model again to chat");
    await loadModels();
  } catch (e) {
    $("quant-error").textContent = e.message;
  } finally {
    btn.disabled = false; btn.textContent = "Create quantized versions"; setBusy(false);
  }
}

// ---------- adding any model ----------
async function readSSE(res, handlers) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
      const ev = (raw.match(/^event: (.*)$/m) || [])[1];
      const data = (raw.match(/^data: (.*)$/m) || [])[1];
      if (ev && data !== undefined && handlers[ev]) handlers[ev](JSON.parse(data));
    }
  }
}

async function pullModel() {
  const name = $("pull-name").value.trim()
    .replace(/^https?:\/\/huggingface\.co\//i, "hf.co/");
  const status = $("pull-status");
  if (!name) { status.textContent = "Type a model name first."; return; }
  $("pull-btn").disabled = true; setBusy(true);
  status.textContent = `Starting download of ${name}…`;
  let failed = null;
  try {
    const res = await fetch("/api/models/pull", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: name }),
    });
    if (!res.ok || !res.body) {
      let msg = `HTTP ${res.status}`;
      try { const d = await res.json(); msg = d.error || (d.detail && "invalid model name") || msg; } catch { /* ignore */ }
      throw new Error(msg);
    }
    await readSSE(res, {
      progress: (p) => {
        status.textContent = p.total ? `${p.status}: ${Math.round((100 * p.done) / p.total)}% of ${fmtBytes(p.total)}` : p.status;
      },
      done: () => { status.textContent = `${name} downloaded. Select it above and click "Load & optimise".`; },
      error: (e) => { failed = e.message; },
    });
    if (failed) throw new Error(failed);
    await loadModels();
    const match = [...$("model").options].find((o) => o.value.toLowerCase().startsWith(name.toLowerCase()));
    if (match) $("model").value = match.value;
  } catch (e) {
    status.textContent = `Download failed: ${e.message}`;
  } finally {
    $("pull-btn").disabled = false; setBusy(false);
  }
}

// ---------- learning toggles ----------
async function pushLearningFlags() {
  if (!state.sessionReady) return;
  try {
    await api("/api/learning", { rl_enabled: $("rl_enabled").checked, auto_adapt: $("auto_adapt").checked });
  } catch (e) {
    $("session-error").textContent = e.message;
  }
}

// ---------- wiring ----------
function initTabs() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab.dataset.tab}`));
      if (tab.dataset.tab !== "model") refreshInsights();
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  $("session-form").addEventListener("submit", startSession);
  $("chat-form").addEventListener("submit", sendMessage);
  $("clear-btn").addEventListener("click", resetChat);
  $("quantize-btn").addEventListener("click", runQuantize);
  $("quality-btn").addEventListener("click", runQuality);
  $("pull-btn").addEventListener("click", pullModel);
  $("skill").addEventListener("change", updateSkillContext);
  updateSkillContext();
  $("pull-name").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); pullModel(); } });
  $("rl_enabled").addEventListener("change", pushLearningFlags);
  $("auto_adapt").addEventListener("change", pushLearningFlags);
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chat-form").requestSubmit(); }
  });
  loadHardware();
  loadModels();
});
