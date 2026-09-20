const $ = id => document.getElementById(id);
const LLM_POOL_KEY = "llmPool.v2";
const LEGACY_LLM_SETTINGS_KEY = "llmSettings.v1";

const state = {
  view: localStorage.getItem("activeView") || "agent",
  sessions: [],
  session: null,
  sessionId: localStorage.getItem("agentSessionId") || "",
  activeRunId: localStorage.getItem("agentRunId") || "",
  sessionToken: 0,
  agentPoll: null,
  agentStream: null,
  streamFrame: null,
  renderedTurns: new Set(),
  jobPoll: null,
  selectedVideos: new Set(),
  category: "",
  lastQuery: "",
  libraryMode: "library",
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function normalizeLlmEndpoint(item, index = 0) {
  return {
    id: String(item.id || crypto.randomUUID?.() || `endpoint-${Date.now()}-${index}`),
    name: String(item.name || `模型配置 ${index + 1}`),
    base_url: String(item.base_url || item.baseUrl || "").trim().replace(/\/$/, ""),
    model: String(item.model || "").trim(),
    api_key: String(item.api_key || item.apiKey || "").trim(),
    enabled: item.enabled !== false,
    priority: Math.max(1, Number(item.priority) || 1),
    weight: Math.max(1, Number(item.weight) || 1),
  };
}

function loadLlmPool() {
  try {
    const stored = JSON.parse(localStorage.getItem(LLM_POOL_KEY) || "null");
    if (Array.isArray(stored?.endpoints)) return { version: 2, endpoints: stored.endpoints.map(normalizeLlmEndpoint) };
    const legacy = JSON.parse(localStorage.getItem(LEGACY_LLM_SETTINGS_KEY) || "null");
    if (legacy?.apiKey && legacy?.baseUrl && legacy?.model) {
      return { version: 2, endpoints: [normalizeLlmEndpoint({ ...legacy, name: "迁移的模型配置" })] };
    }
  } catch { /* 损坏的本地配置按空池处理 */ }
  return { version: 2, endpoints: [] };
}

function saveLlmPool(pool) {
  const clean = { version: 2, endpoints: pool.endpoints.map(normalizeLlmEndpoint) };
  localStorage.setItem(LLM_POOL_KEY, JSON.stringify(clean));
  return clean;
}

function activeLlmPool() {
  return loadLlmPool().endpoints.filter(endpoint => endpoint.enabled && endpoint.api_key && endpoint.base_url && endpoint.model);
}

function llmPoolPayload(endpoints = activeLlmPool()) {
  return endpoints.map(endpoint => ({
    id: endpoint.id, name: endpoint.name, base_url: endpoint.base_url,
    model: endpoint.model, api_key: endpoint.api_key, enabled: endpoint.enabled,
    priority: endpoint.priority, weight: endpoint.weight,
  }));
}

function requestUsesLlm(url) {
  return ["/api/ask", "/api/summarize", "/api/classify", "/api/llm/test", "/api/agent/ask"].includes(url)
    || /\/api\/agent\/[^/]+\/(approval|continue)$/.test(url);
}

async function fetchJson(url, options = {}, poolOverride = null) {
  const headers = new Headers(options.headers || {});
  let body = options.body;
  if (requestUsesLlm(url)) {
    const pool = poolOverride || activeLlmPool();
    if (body && typeof body === "string" && headers.get("Content-Type")?.includes("application/json")) {
      try { body = JSON.stringify({ ...JSON.parse(body), llm_pool: llmPoolPayload(pool) }); } catch { /* 后端返回格式错误 */ }
    } else if (!body) {
      body = JSON.stringify({ llm_pool: llmPoolPayload(pool) });
      headers.set("Content-Type", "application/json");
    }
  }
  const response = await fetch(url, { ...options, body, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `请求失败（${response.status}）`);
  return data;
}

function isSecureCredentialContext() {
  return location.protocol === "https:" || ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
}

function readLlmForm() {
  return {
    id: $("llmEndpointId").value.trim(),
    name: $("llmName").value.trim(),
    baseUrl: $("llmBaseUrl").value.trim().replace(/\/$/, ""),
    model: $("llmModel").value.trim(),
    apiKey: $("llmApiKey").value.trim(),
    enabled: $("llmEnabled").checked,
    priority: Math.max(1, Number($("llmPriority").value) || 1),
    weight: Math.max(1, Number($("llmWeight").value) || 1),
  };
}

function validateLlmSettings(settings) {
  if (!settings.name || !settings.baseUrl || !settings.model || !settings.apiKey) throw new Error("请完整填写配置名称、API 地址、模型名称和 API Key");
  let parsed;
  try { parsed = new URL(settings.baseUrl); } catch { throw new Error("API 地址格式不正确"); }
  const local = ["localhost", "127.0.0.1", "[::1]"].includes(parsed.hostname);
  if (parsed.protocol !== "https:" && !(local && parsed.protocol === "http:")) throw new Error("远程 API 地址必须使用 HTTPS");
  if (!isSecureCredentialContext()) throw new Error("当前页面不是 HTTPS，不能安全保存 API Key");
  return settings;
}

function resetLlmEditor() {
  $("llmEndpointId").value = "";
  $("llmName").value = "";
  $("llmBaseUrl").value = "";
  $("llmModel").value = "";
  $("llmApiKey").value = "";
  $("llmEnabled").checked = true;
  $("llmPriority").value = "1";
  $("llmWeight").value = "1";
  $("llmEditorEyebrow").textContent = "ADD ENDPOINT";
  $("llmEditorTitle").textContent = "添加模型配置";
  $("saveLlmEndpointButton").textContent = "添加到模型池";
  $("cancelLlmEditButton").hidden = true;
}

function editLlmEndpoint(id) {
  const endpoint = loadLlmPool().endpoints.find(item => item.id === id);
  if (!endpoint) return;
  $("llmEndpointId").value = endpoint.id;
  $("llmName").value = endpoint.name;
  $("llmBaseUrl").value = endpoint.base_url;
  $("llmModel").value = endpoint.model;
  $("llmApiKey").value = endpoint.api_key;
  $("llmEnabled").checked = endpoint.enabled;
  $("llmPriority").value = endpoint.priority;
  $("llmWeight").value = endpoint.weight;
  $("llmEditorEyebrow").textContent = "EDIT ENDPOINT";
  $("llmEditorTitle").textContent = "编辑模型配置";
  $("saveLlmEndpointButton").textContent = "保存此配置";
  $("cancelLlmEditButton").hidden = false;
  $("llmName").focus();
}

function renderLlmPool() {
  const pool = loadLlmPool();
  const list = $("llmPoolList");
  if (!pool.endpoints.length) {
    list.innerHTML = '<div class="pool-empty">还没有模型配置。添加至少一个 Provider 后，Agent 才能开始工作。</div>';
  } else {
    list.innerHTML = pool.endpoints
      .sort((a, b) => a.priority - b.priority)
      .map(endpoint => `<article class="pool-item ${endpoint.enabled ? "" : "disabled"}">
        <div class="pool-item-main"><span class="pool-dot"></span><div><h3>${esc(endpoint.name)}</h3><p>${esc(endpoint.model)} · ${esc(endpoint.base_url)}</p></div></div>
        <div class="pool-item-meta"><span>${endpoint.enabled ? "已启用" : "已停用"}</span><span>P${endpoint.priority} · W${endpoint.weight}</span><span>••••${esc(endpoint.api_key.slice(-4))}</span></div>
        <div class="pool-item-actions"><button class="text-button" data-llm-edit="${esc(endpoint.id)}">编辑</button><button class="text-button" data-llm-toggle="${esc(endpoint.id)}">${endpoint.enabled ? "停用" : "启用"}</button><button class="text-button danger-text" data-llm-delete="${esc(endpoint.id)}">删除</button></div>
      </article>`).join("");
  }
  const status = $("llmConfigStatus");
  const active = activeLlmPool();
  status.textContent = active.length ? `已配置 · ${active.length} 个可用` : "尚未配置";
  status.classList.toggle("configured", Boolean(active.length));
}

function saveLlmEndpoint(event) {
  event?.preventDefault();
  try {
    const input = validateLlmSettings(readLlmForm());
    const pool = loadLlmPool();
    const endpoint = normalizeLlmEndpoint({ ...input, base_url: input.baseUrl, api_key: input.apiKey });
    const index = pool.endpoints.findIndex(item => item.id === endpoint.id);
    if (index >= 0) pool.endpoints[index] = endpoint;
    else pool.endpoints.push(endpoint);
    saveLlmPool(pool);
    renderLlmPool();
    resetLlmEditor();
    toast("模型配置已加入此浏览器的模型池");
    return endpoint;
  } catch (error) { toast(error.message, "error"); return null; }
}

function ensureLlmConfigured() {
  if (activeLlmPool().length) return true;
  showView("settings");
  toast("请先在模型设置中添加至少一个可用配置", "error");
  return false;
}

async function testLlmPool(endpoints = activeLlmPool(), button = $("testLlmPoolButton")) {
  if (!endpoints.length) { toast("请先添加至少一个模型配置", "error"); return; }
  button.disabled = true;
  button.textContent = "测试中…";
  try {
    const result = await fetchJson("/api/llm/test", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ llm_pool: llmPoolPayload(endpoints) }) }, endpoints);
    toast(result.ok ? `连接成功：${result.used?.name || "模型配置"}` : "模型响应异常", result.ok ? "ok" : "error");
  } catch (error) { toast(error.message, "error"); }
  finally { button.disabled = false; button.textContent = endpoints.length > 1 ? "测试整个模型池" : "测试此配置"; }
}

function clearLlmSettings() {
  localStorage.removeItem(LLM_POOL_KEY);
  localStorage.removeItem(LEGACY_LLM_SETTINGS_KEY);
  renderLlmPool();
  resetLlmEditor();
  toast("已清除此浏览器中的模型配置");
}

let toastTimer;
function toast(message, kind = "ok") {
  const element = $("toast");
  element.textContent = message;
  element.className = `toast show${kind === "error" ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("show"), 2800);
}

function formatTime(seconds) {
  if (!seconds) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(new Date(seconds * 1000));
}

function formatDuration(seconds) {
  return seconds >= 60 ? `${Math.floor(seconds / 60)}分${seconds % 60}秒` : `${seconds}秒`;
}

function statusLabel(status) {
  return ({
    running: "研究中", completed: "已完成", waiting_approval: "等待确认",
    interrupted: "已中断", paused: "已暂停", failed: "失败", cancelled: "已取消",
  })[status] || "等待任务";
}

function showView(name) {
  state.view = name;
  localStorage.setItem("activeView", name);
  document.querySelectorAll("[data-view-panel]").forEach(panel => {
    panel.classList.toggle("active", panel.dataset.viewPanel === name);
  });
  document.querySelectorAll("[data-view]").forEach(button => {
    const active = button.dataset.view === name;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  document.body.classList.remove("sidebar-open");
  if (name === "library" && !$("results").children.length) loadLibrary();
}

function openSidebar() { document.body.classList.add("sidebar-open"); }
function closeSidebar() { document.body.classList.remove("sidebar-open"); }
function openInspector() { document.body.classList.add("inspector-open"); }
function closeInspector() { document.body.classList.remove("inspector-open"); }

function setFooterStatus(online) {
  const footer = document.querySelector(".app-footer");
  const label = $("footerStatus");
  if (!footer || !label) return;
  footer.classList.toggle("offline", !online);
  label.textContent = online ? "本地服务已连接 · 数据仅存本机" : "服务未连接 · 请确认本地服务已启动";
}

async function loadStats() {
  try {
    const stats = await fetchJson("/api/stats");
    $("stTotal").textContent = stats.total;
    $("stTr").textContent = stats.transcribed;
    $("stSum").textContent = stats.summarized;
    setFooterStatus(true);
  } catch (error) {
    $("stTotal").textContent = "?";
    setFooterStatus(false);
    toast("无法连接本地服务，请确认服务已经启动。", "error");
  }
}

// ---------- Agent 会话与对话 ----------
function latestRun() {
  const runs = state.session?.runs || [];
  return runs.length ? runs[runs.length - 1] : null;
}

function canAskAgent(status) {
  return !status || ["completed", "failed", "cancelled"].includes(status);
}

function stopAgentWatch() {
  clearTimeout(state.agentPoll);
  state.agentPoll = null;
  if (state.agentStream) state.agentStream.close();
  state.agentStream = null;
  if (state.streamFrame) cancelAnimationFrame(state.streamFrame);
  state.streamFrame = null;
}

function renderSessions() {
  const query = $("sessionSearch").value.trim().toLowerCase();
  const sessions = state.sessions.filter(item => {
    const haystack = `${item.title || ""} ${item.last_question || ""}`.toLowerCase();
    return !query || haystack.includes(query);
  });
  $("sessionCount").textContent = state.sessions.length;
  if (!sessions.length) {
    $("agentSessions").innerHTML = `<div class="session-empty">${query ? "没有匹配的会话" : "还没有研究会话"}</div>`;
    return;
  }
  $("agentSessions").innerHTML = sessions.map(item => `
    <div class="session-item${item.session_id === state.sessionId ? " active" : ""}"
         role="listitem" tabindex="0" data-session-id="${esc(item.session_id)}">
      <span class="session-title">${esc(item.title || "未命名研究")}</span>
      <span class="status-dot ${esc(item.latest_status || "")}" title="${esc(statusLabel(item.latest_status))}"></span>
      <span class="session-preview">${esc(item.last_question || `${item.run_count} 个研究轮次`)}</span>
    </div>`).join("");
}

async function loadAgentSessions() {
  try {
    const data = await fetchJson("/api/agent/sessions?limit=50");
    state.sessions = data.items || [];
    renderSessions();
  } catch (error) {
    $("agentSessions").innerHTML = `<div class="session-empty">会话加载失败</div>`;
  }
}

function emptyConversation() {
  return `<div class="conversation-empty">
    <div class="empty-mark">✦</div>
    <h2>从收藏里找到真正有用的线索</h2>
    <p>Agent 会自己拆分任务、搜索收藏、读取转写，并在需要修改数据时向你确认。</p>
    <div class="starter-grid">
      <button class="starter-prompt">比较我收藏里关于个人成长的不同观点</button>
      <button class="starter-prompt">整理最近收藏的 AI 学习路线</button>
      <button class="starter-prompt">找出值得进一步转写的高价值视频</button>
      <button class="starter-prompt">总结收藏中反复出现的核心主题</button>
    </div>
  </div>`;
}

function runAnswerHtml(run) {
  if (run.status === "running") {
    if (run.answer) {
      return `<div class="streaming-answer">${esc(run.answer)}<span class="stream-caret" aria-hidden="true"></span></div>`;
    }
    return `<div class="typing"><span class="typing-dots"><i></i><i></i><i></i></span><span>正在研究收藏并整理证据…</span></div>`;
  }
  if (run.status === "waiting_approval") return `<div class="typing">执行暂停，等待你的确认。</div>`;
  if (["paused", "interrupted"].includes(run.status)) {
    return `<div class="message-error">${esc(run.error || "研究暂时中断，可以从最近一步继续。")}</div>`;
  }
  if (run.status === "failed") {
    return `<div class="message-error">${esc(run.error || "这次研究未能完成。你可以修改问题后重新发送。")}</div>`;
  }
  if (run.status === "cancelled") {
    return `<div class="message-error">${esc(run.error || "这次研究已取消。")}</div>`;
  }
  return esc(run.answer || "研究完成，但没有生成文字回答。");
}

function renderConversation() {
  const runs = state.session?.runs || [];
  if (!runs.length) {
    $("conversation").innerHTML = emptyConversation();
    bindStarterPrompts();
    return;
  }
  $("conversation").innerHTML = runs.map((run, index) => {
    const fresh = run.run_id && !state.renderedTurns.has(run.run_id);
    if (run.run_id) state.renderedTurns.add(run.run_id);
    return `
    <article class="turn${fresh ? " turn-enter" : ""}" data-run-id="${esc(run.run_id)}" data-status="${esc(run.status)}">
      <div class="message user">
        <div class="message-avatar" aria-hidden="true">你</div>
        <div class="message-content">${esc(run.question)}</div>
      </div>
      <div class="message agent">
        <div class="message-avatar" aria-hidden="true">浮</div>
        <div class="message-content">
          ${runAnswerHtml(run)}
          <div class="message-meta">${esc(statusLabel(run.status))}${run.created_at ? ` · ${formatTime(run.created_at)}` : ""}</div>
        </div>
      </div>
      ${index < runs.length - 1 ? '<div class="turn-divider"></div>' : ""}
    </article>`;
  }).join("");
  requestAnimationFrame(() => { $("conversation").scrollTop = $("conversation").scrollHeight; });
}

function toolLabel(tool) {
  return ({
    search_favorites: "关键词搜索收藏", search_semantic: "语义检索收藏",
    get_video: "读取视频内容", transcribe_videos: "转写视频",
    summarize_videos: "生成视频概要", compare_videos: "准备比较材料",
  })[tool] || tool || "处理任务";
}

function renderInspector(run) {
  const status = run?.status || "";
  const statusElement = $("agentRunStatus");
  statusElement.className = `run-status ${esc(status)}`;
  const used = run?.llm?.last_used;
  statusElement.textContent = used?.name ? `${statusLabel(status)} · ${used.name}` : statusLabel(status);

  const plan = run?.plan || [];
  $("agentPlan").innerHTML = plan.length ? plan.map(task => {
    const icon = ({ completed: "✓", running: "→", waiting_approval: "!", failed: "×", skipped: "–", pending: "·" })[task.status] || "·";
    const cls = task.status === "completed" ? "done" : (["running", "waiting_approval"].includes(task.status) ? "active" : (task.status === "failed" ? "failed" : ""));
    return `<div class="trace-row ${cls}"><span class="trace-icon">${icon}</span><span>${esc(task.title)}</span></div>`;
  }).join("") : `<div class="inspector-empty">开始研究后，这里会显示 Agent 的任务拆分。</div>`;

  const steps = run?.steps || [];
  const toolProgress = run?.tool_progress;
  const progressHtml = toolProgress?.kind === "transcription" ? (() => {
    const total = Math.max(1, Number(toolProgress.total) || 1);
    const done = Math.min(total, Math.max(0, Number(toolProgress.done) || 0));
    const percent = Math.round(done / total * 100);
    const eta = Number.isFinite(toolProgress.eta_seconds) && toolProgress.eta_seconds >= 0
      ? ` · 预计 ${formatDuration(toolProgress.eta_seconds)}` : "";
    return `<div class="trace-progress"><div class="trace-progress-head"><span>转写进度</span><b>${done}/${total} · ${percent}%</b></div><div class="trace-progress-bar"><i style="width:${percent}%"></i></div><div class="trace-progress-note">${esc(toolProgress.title || "正在处理视频")}${eta}</div></div>`;
  })() : "";
  $("agentSteps").innerHTML = progressHtml + (steps.length ? steps.map(step => {
    const label = step.type === "final" ? "整理最终回答" : (step.type === "plan" ? "制定研究计划" : (step.type === "replan" ? "调整研究计划" : toolLabel(step.tool)));
    const cls = step.status === "completed" ? "done" : (step.status === "failed" || step.status === "rejected" ? "failed" : "waiting");
    const icon = step.status === "completed" ? "✓" : (step.status === "failed" ? "×" : (step.status === "rejected" ? "–" : "·"));
    return `<div class="trace-row ${cls}"><span class="trace-icon">${icon}</span><span>${esc(label)}</span></div>`;
  }).join("") : `<div class="inspector-empty">工具调用会按时间出现在这里。</div>`);

  const sources = run?.sources || [];
  $("agentSources").innerHTML = sources.length ? sources.map(source => `
    <a class="source-link" href="https://www.douyin.com/video/${encodeURIComponent(source.aweme_id)}" target="_blank" rel="noopener">
      ${esc(source.title || source.aweme_id)} ↗
    </a>`).join("") : `<div class="inspector-empty">检索到的引用来源会出现在这里。</div>`;
}

function renderApproval(run) {
  const container = $("agentApproval");
  if (!run) { container.innerHTML = ""; return; }
  if (run.status === "running") {
    container.innerHTML = `<div class="approval-card"><div class="approval-actions"><button class="danger-button compact" data-agent-action="cancel">停止研究</button></div></div>`;
    return;
  }
  if (["paused", "interrupted"].includes(run.status)) {
    container.innerHTML = `<div class="approval-card"><h3>研究暂时中断</h3><p>${esc(run.error || "可以从最近保存的步骤继续。")}</p><div class="approval-actions"><button class="secondary-button compact" data-agent-action="continue">继续研究</button></div></div>`;
    return;
  }
  if (run.status === "failed" && run.error) {
    const loginHint = /抖音|登录|cookie/i.test(run.error)
      ? "请先到“数据维护”扫码登录抖音，再重新发起转写。" : "请根据错误信息处理后重新发起任务。";
    container.innerHTML = `<div class="approval-card"><h3>研究失败</h3><p>${esc(run.error)}</p><p class="field-help">${loginHint}</p></div>`;
    return;
  }
  if (run.status === "waiting_approval" && run.pending_tool) {
    const pending = run.pending_tool;
    const count = Array.isArray(pending.args?.ids) ? `，涉及 ${pending.args.ids.length} 个视频` : "";
    container.innerHTML = `<div class="approval-card needs-confirm"><h3>需要确认</h3><p>Agent 准备${esc(toolLabel(pending.tool))}${count}。${esc(pending.reason || "这项操作会修改本地数据。")}</p><div class="approval-actions"><button class="secondary-button compact" data-agent-action="approve">批准操作</button><button class="danger-button compact" data-agent-action="reject">拒绝并结束</button></div></div>`;
    return;
  }
  container.innerHTML = "";
}

function setComposerState(run) {
  const allowed = canAskAgent(run?.status);
  $("question").disabled = !allowed;
  $("btnAgentAsk").disabled = !allowed;
  $("question").placeholder = allowed ? "继续追问，或开始一个新的研究主题…" : "当前研究完成这一步后才能继续提问";
}

function renderAgentWorkspace() {
  const run = latestRun();
  $("agentSessionTitle").textContent = state.session?.title || "新建研究";
  const count = state.session?.runs?.length || 0;
  $("agentSessionMeta").textContent = count
    ? `${count} 个研究轮次 · 最近更新 ${formatTime(state.session.updated_at)}`
    : "提出问题，Agent 会规划任务并检索你的收藏。";
  renderConversation();
  renderInspector(run);
  renderApproval(run);
  setComposerState(run);
}

function bindStarterPrompts() {
  document.querySelectorAll(".starter-prompt").forEach(button => {
    button.addEventListener("click", () => {
      $("question").value = button.textContent.trim();
      resizeComposer();
      $("question").focus();
    });
  });
}

async function openAgentSession(sessionId) {
  const token = ++state.sessionToken;
  stopAgentWatch();
  state.sessionId = sessionId;
  localStorage.setItem("agentSessionId", sessionId);
  showView("agent");
  renderSessions();
  $("conversation").innerHTML = `<div class="empty-state">正在恢复研究会话…</div>`;
  try {
    const detail = await fetchJson(`/api/agent/sessions/${encodeURIComponent(sessionId)}`);
    if (token !== state.sessionToken) return;
    state.session = detail;
    const run = latestRun();
    state.activeRunId = run?.run_id || "";
    if (state.activeRunId) localStorage.setItem("agentRunId", state.activeRunId);
    else localStorage.removeItem("agentRunId");
    state.renderedTurns.clear(); // 会话切换后重新播放入场动画
    renderAgentWorkspace();
    if (run?.status === "running") watchAgent(run.run_id, sessionId);
  } catch (error) {
    if (token !== state.sessionToken) return;
    toast(error.message, "error");
    newAgentSession(false);
  }
}

function newAgentSession(focus = true) {
  state.sessionToken += 1;
  stopAgentWatch();
  state.session = null;
  state.sessionId = "";
  state.activeRunId = "";
  localStorage.removeItem("agentSessionId");
  localStorage.removeItem("agentRunId");
  $("question").value = "";
  resizeComposer();
  showView("agent");
  renderSessions();
  renderAgentWorkspace();
  if (focus) $("question").focus();
}

async function beginRenameSession() {
  if (!state.sessionId || !state.session) { toast("请先创建或选择一个会话。", "error"); return; }
  const title = $("agentSessionTitle");
  const oldTitle = title.textContent.trim();
  title.contentEditable = "true";
  title.setAttribute("role", "textbox");
  title.focus();
  document.getSelection()?.selectAllChildren(title);

  const save = async () => {
    title.contentEditable = "false";
    title.removeAttribute("role");
    const nextTitle = title.textContent.trim().slice(0, 80);
    if (!nextTitle || nextTitle === oldTitle) { title.textContent = oldTitle; return; }
    try {
      await fetchJson(`/api/agent/sessions/${encodeURIComponent(state.sessionId)}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: nextTitle }),
      });
      state.session.title = nextTitle;
      await loadAgentSessions();
      toast("会话名称已更新");
    } catch (error) {
      title.textContent = oldTitle;
      toast(error.message, "error");
    }
  };
  title.onblur = save;
  title.onkeydown = event => {
    if (event.key === "Enter") { event.preventDefault(); title.blur(); }
    if (event.key === "Escape") { title.textContent = oldTitle; title.blur(); }
  };
}

async function deleteCurrentSession() {
  if (!state.sessionId) { toast("当前没有可删除的会话。", "error"); return; }
  if (!confirm(`删除“${state.session?.title || "这个会话"}”及全部研究记录？此操作不可撤销。`)) return;
  try {
    await fetchJson(`/api/agent/sessions/${encodeURIComponent(state.sessionId)}`, { method: "DELETE" });
    newAgentSession(false);
    await loadAgentSessions();
    toast("会话已删除");
  } catch (error) { toast(error.message, "error"); }
}

function mergeRun(run) {
  if (!state.session) return;
  const index = state.session.runs.findIndex(item => item.run_id === run.run_id || item.optimistic);
  if (index >= 0) state.session.runs[index] = { ...state.session.runs[index], ...run, optimistic: false };
  else state.session.runs.push(run);
  state.session.updated_at = Date.now() / 1000;
  state.activeRunId = run.run_id;
  localStorage.setItem("agentRunId", run.run_id);
}

function scheduleStreamRender() {
  if (state.streamFrame) return;
  state.streamFrame = requestAnimationFrame(() => {
    state.streamFrame = null;
    renderConversation();
  });
}

function pollAgent(runId, sessionId = state.sessionId) {
  state.agentPoll = setTimeout(async () => {
    if (sessionId !== state.sessionId || runId !== state.activeRunId) return;
    try {
      const run = await fetchJson(`/api/agent/${encodeURIComponent(runId)}`);
      if (sessionId !== state.sessionId || runId !== state.activeRunId) return;
      mergeRun(run);
      renderAgentWorkspace();
      if (run.status === "running") pollAgent(runId, sessionId);
      else await loadAgentSessions();
    } catch (error) {
      toast(`无法更新研究状态：${error.message}`, "error");
      setComposerState(null);
    }
  }, 700);
}

function watchAgent(runId, sessionId = state.sessionId) {
  stopAgentWatch();
  if (!("EventSource" in window)) {
    pollAgent(runId, sessionId);
    return;
  }
  const stream = new EventSource(`/api/agent/${encodeURIComponent(runId)}/stream`);
  state.agentStream = stream;
  stream.onmessage = async event => {
    if (sessionId !== state.sessionId || runId !== state.activeRunId) {
      stopAgentWatch();
      return;
    }
    let payload;
    try { payload = JSON.parse(event.data); }
    catch { return; }
    if (payload.type === "text_delta") {
      const run = latestRun();
      if (!run || run.run_id !== runId) return;
      run.answer = `${run.answer || ""}${payload.delta || ""}`;
      run.status = "running";
      scheduleStreamRender();
      return;
    }
    if (payload.run) {
      mergeRun(payload.run);
      renderAgentWorkspace();
    }
    if (payload.type === "done") {
      stopAgentWatch();
      await loadAgentSessions();
    }
  };
  stream.onerror = () => {
    if (state.agentStream !== stream) return;
    stream.close();
    state.agentStream = null;
    pollAgent(runId, sessionId);
  };
}

async function askAgent(event) {
  event?.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  if (!ensureLlmConfigured()) return;
  const token = state.sessionToken;
  const originalSessionId = state.sessionId;
  const optimistic = {
    run_id: `pending-${Date.now()}`, question, status: "running", plan: [], steps: [],
    sources: [], created_at: Date.now() / 1000, optimistic: true,
  };
  if (!state.session) {
    state.session = { session_id: "", title: question.slice(0, 40), runs: [], updated_at: Date.now() / 1000 };
  }
  state.session.runs.push(optimistic);
  $("question").value = "";
  resizeComposer();
  renderAgentWorkspace();

  try {
    const data = await fetchJson("/api/agent/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, ...(originalSessionId ? { session_id: originalSessionId } : {}) }),
    });
    if (token !== state.sessionToken) return;
    state.sessionId = data.session_id;
    state.session.session_id = data.session_id;
    localStorage.setItem("agentSessionId", data.session_id);
    mergeRun({ ...optimistic, ...data, run_id: data.run_id, optimistic: false });
    renderAgentWorkspace();
    await loadAgentSessions();
    watchAgent(data.run_id, data.session_id);
  } catch (error) {
    if (token !== state.sessionToken) return;
    optimistic.status = "failed";
    optimistic.error = error.message;
    renderAgentWorkspace();
    toast(error.message, "error");
  }
}

async function agentAction(action) {
  const run = latestRun();
  if (!run) return;
  if (["approve", "reject", "continue"].includes(action) && !ensureLlmConfigured()) return;
  let url = `/api/agent/${encodeURIComponent(run.run_id)}`;
  let options = { method: "POST" };
  if (action === "approve" || action === "reject") {
    url += "/approval";
    options = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ approved: action === "approve", llm_pool: llmPoolPayload() }) };
  } else if (action === "continue") {
    url += "/continue";
    options = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ llm_pool: llmPoolPayload() }) };
  }
  else if (action === "cancel") url += "/cancel";
  else return;

  try {
    const updated = await fetchJson(url, options);
    mergeRun(updated);
    renderAgentWorkspace();
    await loadAgentSessions();
    if (updated.status === "running") watchAgent(updated.run_id, state.sessionId);
  } catch (error) { toast(error.message, "error"); }
}

function resizeComposer() {
  const textarea = $("question");
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
}

// ---------- 收藏库 ----------
function tagsOf(raw) {
  try { return JSON.parse(raw || "[]").slice(0, 8); } catch { return []; }
}

function highlight(text) {
  let safe = esc(text);
  if (!state.lastQuery || !safe) return safe;
  for (const token of state.lastQuery.split(/\s+/).filter(Boolean)) {
    const pattern = new RegExp(token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
    safe = safe.replace(pattern, match => `<mark>${match}</mark>`);
  }
  return safe;
}

function videoCard(video) {
  const hasTranscript = video.transcript && !video.transcript.startsWith("【");
  const summary = video.summary ? `<div class="video-summary">${esc(video.summary)}<button class="inline-action copy-summary" data-summary="${esc(video.summary)}">复制概要</button></div>` : "";
  const transcript = hasTranscript ? `<div class="video-transcript clamp" id="tr-${esc(video.aweme_id)}">${highlight(video.transcript)}</div><div class="video-transcript"><button class="inline-action toggle-transcript" data-id="${esc(video.aweme_id)}">展开转写</button></div>` : "";
  return `<article class="video-card">
    <label class="video-select" title="选择后转写"><input type="checkbox" data-video-select="${esc(video.aweme_id)}" ${hasTranscript ? "disabled" : ""}></label>
    <h3>${highlight(video.title)}</h3>
    <div class="video-flags">${hasTranscript ? '<span class="badge">已转写</span>' : ""}${video.category ? `<span class="badge">${esc(video.category)}</span>` : ""}</div>
    <div class="video-meta"><span>${esc(video.author || "未知作者")}</span><a href="https://www.douyin.com/video/${encodeURIComponent(video.aweme_id)}" target="_blank" rel="noopener">打开视频 ↗</a></div>
    <div class="video-tags">${tagsOf(video.tags).map(tag => `<span class="tag">#${highlight(tag)}</span>`).join("")}</div>
    ${summary}${transcript}
  </article>`;
}

function librarySkeleton() {
  return Array(4).fill('<div class="skeleton-row"></div>').join("");
}

async function loadCategories() {
  try {
    const data = await fetchJson("/api/categories");
    const items = [
      `<button class="chip${state.category === "" ? " active" : ""}" data-category="">全部</button>`,
      ...(data.counts || []).map(item => `<button class="chip${item.category === state.category ? " active" : ""}" data-category="${esc(item.category)}">${esc(item.category)}<b>${item.n}</b></button>`),
    ];
    if (data.unclassified) items.push(`<button class="chip${state.category === "__none__" ? " active" : ""}" data-category="__none__">未分类<b>${data.unclassified}</b></button>`);
    $("catChips").innerHTML = items.join("");
  } catch { $("catChips").innerHTML = ""; }
}

async function loadLibrary() {
  state.libraryMode = "library";
  state.lastQuery = "";
  $("results").innerHTML = librarySkeleton();
  try {
    const data = await fetchJson(`/api/videos?limit=30&category=${encodeURIComponent(state.category)}`);
    $("results").innerHTML = data.hits.length ? data.hits.map(videoCard).join("") : '<div class="empty-state">这个分类还没有收藏内容。</div>';
    $("searchNote").textContent = state.category ? `分类「${state.category === "__none__" ? "未分类" : state.category}」的最近内容` : "最近采集的 30 条收藏";
  } catch (error) {
    $("results").innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
  }
}

async function searchLibrary(event) {
  event?.preventDefault();
  const query = $("q").value.trim();
  if (!query) return;
  state.libraryMode = "search";
  state.lastQuery = query;
  $("results").innerHTML = librarySkeleton();
  try {
    const data = await fetchJson(`/api/videos?q=${encodeURIComponent(query)}`);
    $("results").innerHTML = data.hits.length ? data.hits.map(videoCard).join("") : '<div class="empty-state">没有找到相关收藏，试试更短的关键词。</div>';
    $("searchNote").textContent = `找到 ${data.hits.length} 条相关收藏`;
  } catch (error) { $("results").innerHTML = `<div class="empty-state">${esc(error.message)}</div>`; }
}

function updateSelectedButton() {
  $("btnSel").textContent = `转写勾选 · ${state.selectedVideos.size}`;
  $("btnSel").disabled = state.selectedVideos.size === 0;
}

async function transcribeSelected() {
  if (!state.selectedVideos.size) return;
  try {
    await fetchJson("/api/transcribe", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: [...state.selectedVideos] }),
    });
    state.selectedVideos.clear();
    updateSelectedButton();
    toast("已开始转写选中的视频");
    showView("maintenance");
    watchJob();
  } catch (error) { toast(error.message, "error"); }
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast("已复制"); }
  catch { toast("复制失败，请检查浏览器权限。", "error"); }
}

// ---------- 后台任务 ----------
const jobButtonIds = ["btnTr", "btnTrAll", "btnSum", "btnSumAll", "btnCls", "btnReCls", "btnReIdx", "btnLogin", "btnCrawl"];
function setJobButtons(disabled) { jobButtonIds.forEach(id => { $(id).disabled = disabled; }); }

async function runJob(url, n = 0, all = false) {
  if (["/api/summarize", "/api/classify"].includes(url) && !ensureLlmConfigured()) return;
  try {
    await fetchJson(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n, all }),
    });
    toast("后台任务已开始");
    watchJob();
  } catch (error) { toast(error.message, "error"); }
}

function watchJob() {
  clearInterval(state.jobPoll);
  setJobButtons(true);
  $("barwrap").hidden = false;
  state.jobPoll = setInterval(pollJob, 1800);
  pollJob();
}

async function pollJob() {
  let job;
  try { job = await fetchJson("/api/job"); }
  catch (error) { toast(error.message, "error"); return; }
  if (job.running) {
    $("jobState").textContent = "运行中";
    const eta = Number.isFinite(job.eta_seconds) && job.eta_seconds >= 0
      ? ` · 预计剩余 ${formatDuration(job.eta_seconds)}` : "";
    $("jobLine").textContent = `${job.name}：${job.progress}（已进行 ${formatDuration(job.elapsed)}${eta}）`;
    $("btnCancel").hidden = false;
    if (job.total > 0) {
      const percent = Math.round(job.done / job.total * 100);
      $("bar").style.width = `${percent}%`;
      const eta = Number.isFinite(job.eta_seconds) && job.eta_seconds >= 0
        ? ` · 预计 ${formatDuration(job.eta_seconds)}` : "";
      $("bartext").textContent = `${job.done} / ${job.total} · ${percent}%${eta}`;
    } else {
      $("bar").style.width = "18%";
      $("bartext").textContent = "处理中";
    }
  } else {
    clearInterval(state.jobPoll);
    state.jobPoll = null;
    setJobButtons(false);
    $("jobState").textContent = "空闲";
    $("btnCancel").hidden = true;
    $("barwrap").hidden = true;
    if (job.error) $("jobLine").textContent = `${job.name} 失败：${job.error}`;
    else if (["完成", "已取消"].includes(job.progress)) {
      $("jobLine").textContent = `${job.name} · ${job.progress}`;
      loadStats(); loadCategories();
      if (state.libraryMode === "library") loadLibrary();
    } else $("jobLine").textContent = "当前没有运行中的任务。";
  }
  renderJobHistory(job.history || []);
}

function renderJobHistory(items) {
  if (!items.length) { $("history").innerHTML = ""; return; }
  $("history").innerHTML = `<div class="history-head"><span>最近任务</span><button class="text-button" id="clearHistoryButton">清空记录</button></div>${items.map(item => `
    <div class="history-row">
      <span class="status-dot ${item.status === "完成" ? "completed" : (item.status === "失败" ? "failed" : "")}"></span>
      <span>${esc(item.name)} · ${esc(item.status)}</span>
      <span class="history-time">${item.done}/${item.total} · ${formatDuration(item.seconds)}</span>
      <button data-history-delete="${item.id}" aria-label="删除任务记录">×</button>
    </div>`).join("")}`;
  $("clearHistoryButton").addEventListener("click", clearHistory);
}

async function cancelJob() {
  try { await fetchJson("/api/cancel", { method: "POST" }); toast("已请求取消，当前条目结束后停止。", "error"); }
  catch (error) { toast(error.message, "error"); }
}

async function deleteHistory(id) {
  await fetchJson(`/api/history/${id}`, { method: "DELETE" });
  const job = await fetchJson("/api/job");
  renderJobHistory(job.history || []);
}

async function clearHistory() {
  await fetchJson("/api/history", { method: "DELETE" });
  $("history").innerHTML = "";
  toast("任务记录已清空");
}

// ---------- 事件绑定与初始化 ----------
function bindEvents() {
  document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => showView(button.dataset.view)));
  $("sidebarOpen").addEventListener("click", openSidebar);
  $("sidebarClose").addEventListener("click", closeSidebar);
  $("sidebarBackdrop").addEventListener("click", closeSidebar);
  $("inspectorOpen").addEventListener("click", openInspector);
  $("inspectorClose").addEventListener("click", closeInspector);
  $("newSessionButton").addEventListener("click", () => newAgentSession());
  $("renameSessionButton").addEventListener("click", beginRenameSession);
  $("deleteSessionButton").addEventListener("click", deleteCurrentSession);
  $("sessionSearch").addEventListener("input", renderSessions);
  $("agentComposer").addEventListener("submit", askAgent);
  $("question").addEventListener("input", resizeComposer);
  $("question").addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); askAgent(event); }
  });
  $("agentSessions").addEventListener("click", event => {
    const item = event.target.closest("[data-session-id]");
    if (item) openAgentSession(item.dataset.sessionId);
  });
  $("agentSessions").addEventListener("keydown", event => {
    if (!["Enter", " "].includes(event.key)) return;
    const item = event.target.closest("[data-session-id]");
    if (item) { event.preventDefault(); openAgentSession(item.dataset.sessionId); }
  });
  $("agentApproval").addEventListener("click", event => {
    const button = event.target.closest("[data-agent-action]");
    if (button) agentAction(button.dataset.agentAction);
  });

  $("llmEndpointForm").addEventListener("submit", saveLlmEndpoint);
  $("testLlmEndpointButton").addEventListener("click", () => {
    try {
      const endpoint = validateLlmSettings(readLlmForm());
      testLlmPool([normalizeLlmEndpoint({ ...endpoint, base_url: endpoint.baseUrl, api_key: endpoint.apiKey })], $("testLlmEndpointButton"));
    } catch (error) { toast(error.message, "error"); }
  });
  $("testLlmPoolButton").addEventListener("click", () => testLlmPool());
  $("cancelLlmEditButton").addEventListener("click", resetLlmEditor);
  $("clearLlmSettingsButton").addEventListener("click", clearLlmSettings);
  $("llmShowKey").addEventListener("click", () => {
    const input = $("llmApiKey");
    const visible = input.type === "text";
    input.type = visible ? "password" : "text";
    $("llmShowKey").textContent = visible ? "显示" : "隐藏";
    $("llmShowKey").setAttribute("aria-pressed", String(!visible));
  });
  $("llmPoolList").addEventListener("click", event => {
    const edit = event.target.closest("[data-llm-edit]");
    if (edit) return editLlmEndpoint(edit.dataset.llmEdit);
    const toggle = event.target.closest("[data-llm-toggle]");
    if (toggle) {
      const pool = loadLlmPool();
      const endpoint = pool.endpoints.find(item => item.id === toggle.dataset.llmToggle);
      if (endpoint) { endpoint.enabled = !endpoint.enabled; saveLlmPool(pool); renderLlmPool(); }
      return;
    }
    const remove = event.target.closest("[data-llm-delete]");
    if (remove) {
      const pool = loadLlmPool();
      pool.endpoints = pool.endpoints.filter(item => item.id !== remove.dataset.llmDelete);
      saveLlmPool(pool); renderLlmPool(); toast("模型配置已删除");
    }
  });

  $("librarySearchForm").addEventListener("submit", searchLibrary);
  $("showLibraryButton").addEventListener("click", loadLibrary);
  $("btnSel").addEventListener("click", transcribeSelected);
  $("catChips").addEventListener("click", event => {
    const chip = event.target.closest("[data-category]");
    if (!chip) return;
    state.category = chip.dataset.category;
    loadCategories(); loadLibrary();
  });
  $("results").addEventListener("change", event => {
    const checkbox = event.target.closest("[data-video-select]");
    if (!checkbox) return;
    checkbox.checked ? state.selectedVideos.add(checkbox.dataset.videoSelect) : state.selectedVideos.delete(checkbox.dataset.videoSelect);
    updateSelectedButton();
  });
  $("results").addEventListener("click", event => {
    const copy = event.target.closest(".copy-summary");
    if (copy) copyText(copy.dataset.summary);
    const toggle = event.target.closest(".toggle-transcript");
    if (toggle) {
      const transcript = $(`tr-${toggle.dataset.id}`);
      transcript.classList.toggle("clamp");
      toggle.textContent = transcript.classList.contains("clamp") ? "展开转写" : "收起转写";
    }
  });

  $("btnLogin").addEventListener("click", () => runJob("/api/login"));
  $("btnCrawl").addEventListener("click", () => runJob("/api/crawl"));
  $("btnTr").addEventListener("click", () => runJob("/api/transcribe", 10, false));
  $("btnTrAll").addEventListener("click", () => runJob("/api/transcribe", 0, true));
  $("btnSum").addEventListener("click", () => runJob("/api/summarize", 10, false));
  $("btnSumAll").addEventListener("click", () => runJob("/api/summarize", 0, true));
  $("btnCls").addEventListener("click", () => runJob("/api/classify", 0, false));
  $("btnReCls").addEventListener("click", () => runJob("/api/classify", 0, true));
  $("btnReIdx").addEventListener("click", () => runJob("/api/reindex"));
  $("btnCancel").addEventListener("click", cancelJob);
  $("history").addEventListener("click", event => {
    const button = event.target.closest("[data-history-delete]");
    if (button) deleteHistory(button.dataset.historyDelete);
  });

  document.addEventListener("keydown", event => {
    if (event.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
      event.preventDefault(); showView("library"); $("q").focus();
    }
    if (event.key === "Escape") { closeSidebar(); closeInspector(); }
  });
}

async function initialize() {
  bindEvents();
  renderLlmPool();
  resetLlmEditor();
  showView(state.view);
  renderAgentWorkspace();
  await Promise.all([loadStats(), loadCategories(), loadAgentSessions()]);
  if (state.sessionId && state.sessions.some(item => item.session_id === state.sessionId)) {
    await openAgentSession(state.sessionId);
  } else newAgentSession(false);
  loadLibrary();
  try {
    const job = await fetchJson("/api/job");
    if (job.running) watchJob();
    else renderJobHistory(job.history || []);
  } catch { /* 页面其他功能仍可使用 */ }
}

initialize();
