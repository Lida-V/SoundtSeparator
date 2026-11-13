"use strict";

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const uploadView = $("upload-view");
const progressView = $("progress-view");
const resultView = $("result-view");
const errorBox = $("error-box");
const dropzone = $("dropzone");
const fileInput = $("file-input");
const tracksEl = $("tracks");
const playBtn = $("play-btn");

// ---------- 多言語UI ----------
const t = (key, params) => I18N.t(key, params);
const langParam = () => "lang=" + encodeURIComponent(I18N.current);
I18N.init();

// 言語セレクタの選択肢は読み込み済みlocaleから構築する
const langSelect = $("lang-select");
for (const locale of I18N.supported()) {
  const option = document.createElement("option");
  option.value = locale;
  option.textContent = I18N.labels[locale] || locale;
  langSelect.append(option);
}
langSelect.value = I18N.current;
langSelect.addEventListener("change", () => I18N.setLocale(langSelect.value));
document.addEventListener("i18nchange", () => {
  langSelect.value = I18N.current;
  renderRecentJobs();
  refreshModelCatalog(); // カタログの説明文はサーバー側で ?lang= 翻訳される
  refreshModelDownloads(); // モデル管理の一覧もサーバー側で ?lang= 翻訳される
});

// モデル一覧はサーバーを正とし、対応する目的だけを表示する。
let modelCatalog = { models: [], defaults: {} };
let lastSubmittedFile = null;
let currentJobMetadata = null;
let selectedModelByMode = {};
let selectedStemNamesByModel = {};
let downloadGroups = [];
const modelDownloadJobs = new Map();
const MODE_KEYS = {
  dme: { title: "mode.dme.title", description: "mode.dme.desc" },
  music: { title: "mode.music.title", description: "mode.music.desc" },
  vocals: { title: "mode.vocals.title", description: "mode.vocals.desc" },
  mega: { title: "mode.mega.title", description: "mode.mega.desc" },
};

function uiElement(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function currentMode() { return document.querySelector('input[name="mode"]:checked')?.value || "dme"; }
function currentModel() {
  return modelCatalog.models.find((m) => m.id === document.querySelector('input[name="model_id"]:checked')?.value);
}

function renderModeOptions(preferredMode) {
  const modes = new Set(modelCatalog.models.flatMap((m) => m.modes || []));
  const activeMode = modes.has(preferredMode) ? preferredMode : [...modes][0];
  const container = $("mode-options");
  container.replaceChildren();
  for (const [mode, info] of Object.entries(MODE_KEYS)) {
    if (!modes.has(mode)) continue;
    const label = uiElement("label", "mode-card");
    const input = document.createElement("input");
    input.type = "radio"; input.name = "mode"; input.value = mode; input.checked = mode === activeMode;
    const body = uiElement("div", "mode-body");
    body.append(uiElement("div", "mode-title", t(info.title)), uiElement("div", "mode-desc", t(info.description)));
    label.append(input, body);
    input.addEventListener("change", () => renderModelOptions());
    container.append(label);
  }
  renderModelOptions();
}

function renderModelOptions() {
  const mode = currentMode();
  const models = modelCatalog.models.filter((m) => m.modes?.includes(mode));
  const preferred = selectedModelByMode[mode] || modelCatalog.defaults[mode];
  const selected = models.find((m) => m.id === preferred) || models[0];
  $("model-options").replaceChildren();
  models.forEach((m) => {
    const label = uiElement("label", "model-card");
    const radio = document.createElement("input");
    radio.type = "radio"; radio.name = "model_id"; radio.value = m.id; radio.checked = m === selected;
    const body = uiElement("div", "model-card-body");
    const heading = uiElement("div", "model-card-heading");
    heading.append(uiElement("strong", "", m.label), uiElement("span", "model-state" + (m.downloaded ? " ready" : ""), m.downloaded ? t("model.downloaded") : t("model.not_downloaded")));
    body.append(heading, uiElement("p", "", m.description || ""));
    label.append(radio, body);
    radio.addEventListener("change", () => { selectedModelByMode[mode] = m.id; renderModelDetails(); });
    $("model-options").append(label);
  });
  renderModelDetails();
}

function renderModelDetails() {
  const model = currentModel();
  const detail = $("model-detail");
  detail.replaceChildren();
  $("stem-selector").classList.toggle("hidden", currentMode() !== "mega" || !model);
  if (!model) return;
  const facts = uiElement("div", "model-facts");
  if (model.download_mb) facts.append(uiElement("span", "", model.download_mb >= 1000
    ? t("model.download_size_gb", { gb: (model.download_mb / 1000).toFixed(1) })
    : t("model.download_size_mb", { mb: model.download_mb })));
  if (model.vram_gb && !model.gpu_summary) facts.append(uiElement("span", "", t("model.vram", { gb: model.vram_gb })));
  facts.append(uiElement("span", "", model.cached ? t("model.cached") : t("model.load_on_use")));
  if (model.source_url && /^https?:\/\//.test(model.source_url)) {
    const link = uiElement("a", "model-source", t("model.source"));
    link.href = model.source_url; link.target = "_blank"; link.rel = "noopener noreferrer";
    facts.append(link);
  }
  detail.append(facts);
  if (model.gpu_summary) detail.append(uiElement("p", "settings-note", model.gpu_summary));
  if (!model.downloaded) {
    const help = uiElement("button", "text-button", t("model.get_first"));
    help.type = "button";
    help.addEventListener("click", openModelManager);
    detail.append(help);
  }
  const notes = Array.isArray(model.notes) ? model.notes : model.notes ? [model.notes] : [];
  notes.forEach((note) => detail.append(uiElement("p", "settings-note", note)));
  if (currentMode() === "mega") renderStemOptions(model.stems || []);
}

function renderStemOptions(stems) {
  $("stem-options").replaceChildren();
  $("stem-search").value = "";
  const common = new Set(["vocals", "vocal", "drums", "bass", "guitar", "piano", "other"]);
  const defaults = stems.filter((s) => common.has(s.name));
  const modelId = currentModel()?.id;
  const initialNames = new Set(selectedStemNamesByModel[modelId] || (defaults.length ? defaults : stems.slice(0, 6)).map((s) => s.name));
  stems.forEach((s) => {
    const label = uiElement("label", "stem-choice");
    const input = document.createElement("input");
    input.type = "checkbox"; input.value = s.name; input.checked = initialNames.has(s.name);
    input.addEventListener("change", updateStemCount);
    label.append(input, uiElement("span", "", s.label || s.name));
    label.dataset.search = `${s.label || ""} ${s.name}`.toLowerCase();
    $("stem-options").append(label);
  });
  updateStemCount();
}

function updateStemCount() {
  const inputs = [...$("stem-options").querySelectorAll("input:checked")];
  const n = inputs.length;
  const modelId = currentModel()?.id;
  if (currentMode() === "mega" && modelId) selectedStemNamesByModel[modelId] = inputs.map((input) => input.value);
  $("stem-count").textContent = t("stem.count", { n, total: $("stem-options").children.length });
}

function renderRuntime(runtime) {
  if (!runtime) return;
  const gpu = runtime.gpu_name || (String(runtime.device).startsWith("cuda") ? "CUDA GPU" : t("runtime.cpu_mode"));
  $("runtime-status").textContent = `${runtime.busy ? t("runtime.busy") : t("runtime.ready")} · ${gpu}`;
  $("runtime-status").classList.toggle("cpu-mode", !String(runtime.device).startsWith("cuda"));
  $("runtime-memory").textContent = runtime.vram_total_gb
    ? t("runtime.vram_use", { used: Number(runtime.vram_allocated_gb || 0).toFixed(1), total: Number(runtime.vram_total_gb).toFixed(1) })
    : t("runtime.cpu_slow_note");
  $("release-models").disabled = !!runtime.busy || !runtime.cached_models?.length;
}

async function refreshModelCatalog() {
  try {
    const response = await fetch(`/api/models?${langParam()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(t("error.catalog_fetch"));
    const catalog = await response.json();
    if (!Array.isArray(catalog.models) || !catalog.models.length) throw new Error(t("error.no_models"));
    const mode = currentMode();
    const selected = currentModel();
    if (selected) selectedModelByMode[mode] = selected.id;
    modelCatalog = catalog;
    renderModeOptions(mode);
    renderRuntime(catalog.runtime);
    $("catalog-status").textContent = "";
  } catch (error) {
    if (!modelCatalog.models.length) {
      modelCatalog = { models: [
        { id: "bandit", label: "BandIt Plus", modes: ["dme"], description: t("catalog.bandit.desc") },
        { id: "bs_roformer", label: "BS-Roformer + Demucs", modes: ["music"], description: t("catalog.bs_roformer.desc") },
      ], defaults: { dme: "bandit", music: "bs_roformer" } };
      renderModeOptions("dme");
    }
    $("catalog-status").textContent = t("error.catalog_offline");
    $("runtime-status").textContent = t("error.runtime_info");
    $("release-models").disabled = true;
  }
}

// サーバーは detail (ja正本) に detail_key を併設する。キーがあれば翻訳を優先する。
async function responseError(response, fallbackKey) {
  let body = null;
  try { body = await response.json(); } catch (_) {}
  if (body) {
    if (body.detail_key && I18N.has(body.detail_key)) return new Error(t(body.detail_key, body.detail_params));
    if (typeof body.detail === "string" && body.detail) return new Error(body.detail);
    if (body.message) return new Error(body.message);
  }
  return new Error(`${t(fallbackKey)} (${response.status})`);
}

async function responseDetail(response, fallbackKey) {
  const body = await response.json().catch(() => ({}));
  if (body.detail_key && I18N.has(body.detail_key)) return t(body.detail_key, body.detail_params);
  return body.detail || t(fallbackKey);
}

// ジョブメッセージは status_key + status_params があれば翻訳、なければ ja の message をそのまま表示
function jobMessage(job) {
  if (job && job.status_key && I18N.has(job.status_key)) return t(job.status_key, job.status_params);
  return (job && job.message) || "";
}

function openModelManager() {
  const manager = $("model-manager");
  manager.open = true;
  manager.scrollIntoView({ behavior: "smooth", block: "start" });
  refreshModelDownloads();
}

function renderModelDownloads() {
  const list = $("download-list");
  list.replaceChildren();
  for (const group of downloadGroups) {
    const state = modelDownloadJobs.get(group.id);
    const active = state && ["starting", "queued", "running"].includes(state.status);
    const row = uiElement("div", "download-item");
    const top = uiElement("div", "download-item-top");
    const copy = uiElement("div", "download-copy");
    const title = uiElement("strong", "", group.label || group.id);
    const badge = uiElement("span", "model-state" + (group.downloaded ? " ready" : ""),
      group.downloaded ? t("dl.state_ready") : active ? t("dl.state_downloading") : t("dl.state_missing"));
    copy.append(title, badge);
    const details = [];
    if (group.description) details.push(group.description);
    if (group.download_mb) details.push(group.download_mb >= 1000
      ? t("dl.size_hint_gb", { gb: (group.download_mb / 1000).toFixed(1) })
      : t("dl.size_hint_mb", { mb: group.download_mb }));
    if (details.length) copy.append(uiElement("p", "download-description", details.join(" · ")));
    if (group.gpu_summary) copy.append(uiElement("p", "download-requirements", group.gpu_summary));
    const instruction = group.instructions || group.instruction;
    if (instruction) copy.append(uiElement("p", "download-instructions", Array.isArray(instruction) ? instruction.join("\n") : String(instruction)));
    const actions = uiElement("div", "download-actions");
    if (group.source_url && /^https?:\/\//.test(group.source_url)) {
      const link = uiElement("a", "", t("model.source"));
      link.href = group.source_url; link.target = "_blank"; link.rel = "noopener noreferrer";
      actions.append(link);
    }
    const button = uiElement("button", "btn" + (!group.downloaded && group.downloadable !== false ? " btn-primary" : ""),
      group.downloaded ? t("dl.state_ready") : active ? t("dl.state_downloading_ellipsis")
        : group.downloadable === false ? t("dl.check_source") : state?.status === "error" ? t("dl.retry") : t("dl.download"));
    button.type = "button";
    button.disabled = !!group.downloaded || !!active || group.downloadable === false;
    button.addEventListener("click", () => startModelDownload(group.id));
    actions.append(button);
    top.append(copy, actions); row.append(top);
    if (active) {
      const progress = document.createElement("progress");
      progress.className = "download-progress"; progress.max = 100;
      if (state.progress !== null && state.progress !== undefined && Number.isFinite(Number(state.progress))) progress.value = Math.max(0, Math.min(100, Number(state.progress)));
      progress.setAttribute("aria-label", t("dl.aria_progress", { label: group.label || group.id }));
      row.append(progress);
    }
    // サーバーは ja正本 + message_key/error_key を併設する。キーがあれば翻訳を優先する
    let message = null;
    if (state) {
      if (state.error) {
        message = (state.error_key && I18N.has(state.error_key)) ? t(state.error_key, state.error_params) : String(state.error);
      } else if (state.message_key && I18N.has(state.message_key)) {
        message = t(state.message_key, state.message_params);
      } else if (state.message) {
        message = String(state.message);
      }
    }
    if (message) {
      const status = uiElement("p", "download-message" + (state.status === "error" ? " error" : ""), message);
      status.setAttribute("role", "status"); row.append(status);
    }
    list.append(row);
  }
}

async function refreshModelDownloads() {
  try {
    const response = await fetch(`/api/model-downloads?${langParam()}`, { cache: "no-store" });
    if (!response.ok) throw await responseError(response, "error.dl_fetch_status");
    const data = await response.json();
    if (!Array.isArray(data.groups)) throw new Error(t("error.dl_bad_payload"));
    downloadGroups = data.groups;
    $("download-list-status").textContent = data.groups.length ? t("dl.hint") : t("dl.empty");
    renderModelDownloads();
  } catch (error) {
    $("download-list-status").textContent = t("dl.fetch_failed", { error: error.message });
  }
}

async function startModelDownload(id) {
  if (["starting", "queued", "running"].includes(modelDownloadJobs.get(id)?.status)) return;
  modelDownloadJobs.set(id, { status: "starting", message_key: "dl.preparing" });
  renderModelDownloads();
  try {
    const response = await fetch(`/api/model-downloads/${encodeURIComponent(id)}`, { method: "POST" });
    if (!response.ok) throw await responseError(response, "error.dl_start");
    const job = await response.json();
    if (!job.job_id) throw new Error(t("error.dl_no_job"));
    while (true) {
      const progressResponse = await fetch(`/api/model-downloads/jobs/${encodeURIComponent(job.job_id)}`, { cache: "no-store" });
      if (!progressResponse.ok) throw await responseError(progressResponse, "error.dl_progress");
      const state = await progressResponse.json();
      modelDownloadJobs.set(id, state);
      renderModelDownloads();
      if (["done", "completed"].includes(state.status)) {
        await Promise.all([refreshModelDownloads(), refreshModelCatalog()]);
        return;
      }
      if (["error", "failed"].includes(state.status)) {
        if (state.error_key && I18N.has(state.error_key)) throw new Error(t(state.error_key, state.error_params));
        throw new Error(state.error || state.message || t("error.dl_failed"));
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  } catch (error) {
    modelDownloadJobs.set(id, { status: "error", error: error.message });
    await refreshModelDownloads();
  }
}

function separationOptions() {
  const options = {
    mode: currentMode(),
    quality: document.querySelector('input[name="quality"]:checked').value,
    model_id: currentModel()?.id,
  };
  if ($("preview-enabled").checked) {
    const start = $("preview-start").valueAsNumber;
    const length = $("preview-duration").valueAsNumber;
    if (!Number.isFinite(start) || start < 0) throw new Error(t("error.preview_start"));
    if (!Number.isFinite(length) || length < 5 || length > 60) throw new Error(t("error.preview_length"));
    options.preview_start = start;
    options.preview_duration = length;
  }
  if (options.mode === "mega") {
    options.selected_stems = [...$("stem-options").querySelectorAll("input:checked")].map((input) => input.value);
    if (!options.selected_stems.length) throw new Error(t("error.no_stems"));
  }
  return options;
}

function jobForm(file, options) {
  const form = new FormData();
  form.append("file", file);
  Object.entries(options).forEach(([key, value]) => {
    if (value !== undefined) form.append(key, Array.isArray(value) ? JSON.stringify(value) : value);
  });
  return form;
}

function optionsDescription(options) {
  const quality = { fast: t("quality.fast"), standard: t("quality.standard"), high: t("quality.high") }[options.quality] || options.quality;
  const model = modelCatalog.models.find((m) => m.id === options.model_id);
  const title = model?.label || options.model_id || (MODE_KEYS[options.mode] ? t(MODE_KEYS[options.mode].title) : "");
  return `${title} · ${quality}${options.preview_duration
    ? ` · ${t("desc.preview", { start: options.preview_start, length: options.preview_duration })}`
    : ` · ${t("desc.full")}`}`;
}

// 履歴にはジョブ情報だけを保存する。音声はサーバー内にあり、ブラウザには複製しない。
function readRecentJobs() {
  try { const items = JSON.parse(sessionStorage.getItem("ssRecentJobs") || "[]"); return Array.isArray(items) ? items.slice(0, 8) : []; }
  catch (_) { return []; }
}

function rememberJob(job, filename) {
  const items = readRecentJobs();
  const old = items.find((item) => item.id === job.id) || {};
  const item = { ...old, id: job.id, filename: filename || old.filename, status: job.status, model_label: job.model_label || old.model_label, elapsed: job.elapsed, zip_url: job.zip_url, is_preview: job.is_preview };
  try { sessionStorage.setItem("ssRecentJobs", JSON.stringify([item, ...items.filter((entry) => entry.id !== job.id)].slice(0, 8))); } catch (_) { /* ストレージ禁止時も分離は利用できる */ }
  renderRecentJobs();
}

function renderRecentJobs() {
  const items = readRecentJobs();
  $("recent-jobs").classList.toggle("hidden", !items.length);
  $("recent-list").replaceChildren();
  items.forEach((item) => {
    const row = uiElement("div", "recent-row");
    const text = uiElement("div", "recent-label");
    const bits = [];
    if (item.model_label) bits.push(item.model_label);
    if (item.is_preview) bits.push(t("recent.preview_badge"));
    if (item.elapsed !== undefined) bits.push(t("recent.elapsed", { sec: item.elapsed }));
    text.append(uiElement("strong", "", item.filename || t("recent.audio_file")), uiElement("span", "", bits.join(" · ")));
    row.append(text);
    const open = uiElement("button", "btn", item.status === "done" ? t("recent.open") : item.status === "error" ? t("recent.error") : t("recent.check"));
    open.disabled = item.status === "error";
    open.addEventListener("click", async () => {
      hideError();
      try {
        const response = await fetch(`/api/jobs/${encodeURIComponent(item.id)}?${langParam()}`);
        if (!response.ok) throw new Error(t("error.recent_gone"));
        const job = await response.json();
        rememberJob(job, item.filename);
        if (job.status === "error") throw new Error(job.message);
        uploadView.classList.add("hidden"); progressView.classList.remove("hidden");
        baseFilename = (item.filename || "audio").replace(/\.[^.]+$/, "");
        $("progress-filename").textContent = item.filename;
        if (job.status === "done") await showResult(job, item.filename);
        else pollJob(item.id, item.filename);
      } catch (error) { showError(error.message); resetToUpload(); }
    });
    row.append(open);
    $("recent-list").append(row);
  });
}

$("preview-enabled").addEventListener("change", () => $("preview-fields").classList.toggle("hidden", !$("preview-enabled").checked));
$("stem-search").addEventListener("input", () => {
  const term = $("stem-search").value.trim().toLowerCase();
  [...$("stem-options").children].forEach((el) => el.classList.toggle("hidden", !el.dataset.search.includes(term)));
});
$("stems-all").addEventListener("click", () => { $("stem-options").querySelectorAll("input").forEach((input) => input.checked = true); updateStemCount(); });
$("stems-clear").addEventListener("click", () => { $("stem-options").querySelectorAll("input").forEach((input) => input.checked = false); updateStemCount(); });
$("clear-history").addEventListener("click", () => { try { sessionStorage.removeItem("ssRecentJobs"); } catch (_) {} renderRecentJobs(); });
$("release-models").addEventListener("click", async () => {
  $("release-models").disabled = true;
  try {
    const response = await fetch("/api/models/release", { method: "POST" });
    if (!response.ok) throw await responseError(response, "error.release_failed");
    const state = await response.json();
    renderRuntime(state.runtime || state);
    await refreshModelCatalog();
    flashInfo(t("toast.models_released"));
  } catch (error) { showError(error.message); await refreshModelCatalog(); }
});
$("refresh-downloads").addEventListener("click", refreshModelDownloads);
$("model-manager").addEventListener("toggle", () => { if ($("model-manager").open) refreshModelDownloads(); });
$("retry-job").addEventListener("click", () => { if (lastSubmittedFile) startJob(lastSubmittedFile); });
const modelCatalogReady = refreshModelCatalog();
refreshModelDownloads();
renderRecentJobs();

// ---------- 状態 ----------
let audioCtx = null;
let masterGain = null;
let masterAnalyser = null;
let meterBuf = null;
let tracks = []; // {name,label,color,isOriginal,chans:[Float32Array],buffer,gainNode,source,muted,solo,volume,...}
let sampleRate = 44100;
let lengthSamples = 0;
let isPlaying = false;
let playOffset = 0; // 秒
let playStartCtxTime = 0;
let rafId = null;
let baseFilename = "audio";
let loopMode = false;

// エディタ状態
let selection = null; // {start, end} 秒
let activeTrackIdx = -1;
let clipboard = null; // {start(サンプル), length, chans: [Float32Array], sourceLabel} 単一トラックの内容
let undoStack = [];
let redoStack = [];
const UNDO_LIMIT = 8;
let editorBusy = false;
const FADE_SAMPLES = 441; // 編集境界のクリックノイズ防止フェード (10ms @44.1kHz)
let autoScale = localStorage.getItem("ssAutoScale") !== "0"; // 波形表示の自動スケール
let viewMode = localStorage.getItem("ssViewMode") === "spec" ? "spec" : "wave"; // 波形 or スペクトログラム

const duration = () => lengthSamples / sampleRate;

// 表示ズーム状態
let viewStart = 0; // 表示開始 (秒)
let viewDur = null; // 表示幅 (秒)。null = 全体表示
const getViewDur = () => (viewDur === null ? duration() : viewDur);

function setView(start, dur) {
  const d = duration();
  dur = clamp(dur, 0.02, d);
  if (dur >= d - 1e-6) {
    viewStart = 0;
    viewDur = null;
  } else {
    viewStart = clamp(start, 0, d - dur);
    viewDur = dur;
  }
  requestViewRefresh();
}

function zoomAt(factor, centerT) {
  const vd = getViewDur();
  const nd = vd * factor;
  setView(centerT - (centerT - viewStart) * (nd / vd), nd);
}

let viewRefreshPending = false;
function requestViewRefresh() {
  if (viewRefreshPending) return;
  viewRefreshPending = true;
  requestAnimationFrame(() => {
    viewRefreshPending = false;
    tracks.forEach((t) => layoutTrackCanvas(t));
    renderSelection();
    updateOverview();
    render();
  });
}

function updateOverview() {
  const ov = $("overview-window");
  if (!ov) return;
  ov.style.left = (viewStart / duration()) * 100 + "%";
  ov.style.width = (getViewDur() / duration()) * 100 + "%";
  $("overview").classList.toggle("zoomed", viewDur !== null);
}

// ============================================================
// アップロード
// ============================================================
function handleFile(f) {
  if (f.name.toLowerCase().endsWith(".ssproj")) loadProject(f);
  else startJob(f);
}

function handleFiles(fileList) {
  const files = [...fileList];
  if (files.length === 1) return handleFile(files[0]);
  const media = files.filter((f) => !f.name.toLowerCase().endsWith(".ssproj"));
  if (media.length) startBatch(media);
}

// ---------- 一括分離 (バッチ処理) ----------
async function startBatch(files) {
  hideError();
  await modelCatalogReady;
  let options;
  try { options = separationOptions(); } catch (error) { showError(error.message); return; }
  uploadView.classList.add("hidden");
  $("batch-view").classList.remove("hidden");
  const list = $("batch-list");
  list.innerHTML = "";

  const rows = files.map((f) => {
    const row = document.createElement("div");
    row.className = "batch-row";
    row.innerHTML =
      `<span class="batch-name"></span>` +
      `<span class="batch-status">${t("batch.waiting")}</span>` +
      `<div class="batch-bar"><div class="batch-fill"></div></div>` +
      `<span class="batch-link"></span>`;
    row.querySelector(".batch-name").textContent = f.name;
    list.appendChild(row);
    return row;
  });

  let ok = 0;
  for (let i = 0; i < files.length; i++) {
    const row = rows[i];
    const status = row.querySelector(".batch-status");
    const fill = row.querySelector(".batch-fill");
    try {
      status.textContent = t("status.uploading");
      const form = jobForm(files[i], options);
      const res = await fetch("/api/jobs", { method: "POST", body: form });
      if (!res.ok) throw await responseError(res, "error.upload_failed");
      const { id } = await res.json();
      rememberJob({ id, status: "queued", model_label: currentModel()?.label }, files[i].name);

      while (true) {
        await new Promise((r) => setTimeout(r, 1200));
        const jr = await fetch(`/api/jobs/${id}?${langParam()}`);
        if (!jr.ok) throw new Error(t("error.job_fetch"));
        const job = await jr.json();
        status.textContent = jobMessage(job);
        fill.style.width = (job.progress || 0) + "%";
        if (job.status === "error") { rememberJob(job, files[i].name); throw new Error(jobMessage(job)); }
        if (job.status === "done") {
          rememberJob(job, files[i].name);
          status.textContent = t("batch.done");
          fill.style.width = "100%";
          const a = document.createElement("a");
          a.href = job.zip_url;
          a.className = "btn";
          a.textContent = "⬇ ZIP";
          a.download = "";
          row.querySelector(".batch-link").appendChild(a);
          ok++;
          break;
        }
      }
    } catch (err) {
      status.textContent = "⚠ " + err.message;
      row.classList.add("batch-error");
    }
  }
  flashInfo(t("toast.batch_done", { ok, total: files.length }));
}

$("batch-back").addEventListener("click", () => location.reload());

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.target === dropzone && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); fileInput.click(); }
});
$("browse-btn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFiles(fileInput.files); });

["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});

async function startJob(file) {
  hideError();
  await modelCatalogReady;
  lastSubmittedFile = file;
  let options;
  try { options = separationOptions(); } catch (error) { showError(error.message); return; }
  uploadView.classList.add("hidden");
  progressView.classList.remove("hidden");
  $("progress-filename").textContent = file.name;
  $("progress-message").textContent = t("status.uploading");
  $("progress-percent").textContent = "0%";
  $("progress-bar").style.width = "0%";
  $("progress-settings").textContent = optionsDescription(options);
  baseFilename = file.name.replace(/\.[^.]+$/, "");

  const form = jobForm(file, options);

  try {
    const res = await fetch("/api/jobs", { method: "POST", body: form });
    if (!res.ok) throw await responseError(res, "error.upload_failed");
    const { id } = await res.json();
    rememberJob({ id, status: "queued", model_label: currentModel()?.label }, file.name);
    pollJob(id, file.name);
  } catch (err) {
    showError(err.message);
    resetToUpload();
  }
}

async function pollJob(id, filename) {
  try {
    const res = await fetch(`/api/jobs/${id}?${langParam()}`);
    if (!res.ok) throw new Error(t("error.job_fetch"));
    const job = await res.json();

    $("progress-message").textContent = jobMessage(job);
    $("progress-percent").textContent = (job.progress || 0) + "%";
    $("progress-bar").style.width = (job.progress || 0) + "%";

    if (job.status === "error") { rememberJob(job, filename); throw new Error(jobMessage(job)); }
    if (job.status === "done") {
      rememberJob(job, filename);
      $("progress-message").textContent = t("status.loading_stems");
      await showResult(job, filename);
      return;
    }
    setTimeout(() => pollJob(id, filename), 800);
  } catch (err) {
    showError(err.message);
    resetToUpload();
  }
}

// ============================================================
// 結果表示
// ============================================================
function initAudio() {
  // 推論エンジンと同じ44.1kHzに保ち、ブラウザ既定48kHzへの自動変換を防ぐ。
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
  if (!masterGain) {
    masterGain = audioCtx.createGain();
    masterGain.connect(audioCtx.destination);
    masterAnalyser = audioCtx.createAnalyser();
    masterAnalyser.fftSize = 1024;
    masterGain.connect(masterAnalyser);
  }
}

async function showResult(job, filename) {
  initAudio();
  currentJobMetadata = {
    mode: job.mode, model_id: job.model_id, model_label: job.model_label || job.model_id,
    quality: job.quality, elapsed: job.elapsed, is_preview: job.is_preview,
    preview_start: job.preview_start, preview_duration: job.preview_duration,
    output_channels: job.output_channels, selected_stems: job.selected_stems,
    model_notes: job.model_notes, duration: job.duration, sample_rate: job.sample_rate || 44100,
  };
  baseFilename = filename.replace(/\.[^.]+$/, "") + (job.is_preview ? `_preview_${Number(job.preview_start || 0)}s` : "");

  const defs = [
    ...job.stems.map((s, i) => ({ ...s, defaultMuted: job.mode === "mega" && i > 0, isOriginal: false })),
    { name: "original", label: t("track.original"), color: "#94a3b8", url: job.original_url, defaultMuted: true, isOriginal: true },
  ];

  const buffers = await Promise.all(
    defs.map(async (d) => {
      const res = await fetch(d.url);
      if (!res.ok) throw new Error(t("error.stem_load", { label: d.label, status: res.status }));
      const buf = await res.arrayBuffer();
      return audioCtx.decodeAudioData(buf);
    })
  );

  sampleRate = buffers[0].sampleRate;
  lengthSamples = Math.max(...buffers.map((b) => b.length));
  tracks = defs.map((d, i) => makeTrack(d, buffers[i]));

  progressView.classList.add("hidden");
  resultView.classList.remove("hidden");
  $("result-filename").textContent = filename;
  renderResultMetadata();
  $("time-total").textContent = fmtTime(duration());
  $("time-current").textContent = fmtTime(0);
  if (!localStorage.getItem("ssGuideDismissed")) $("guide-card").classList.remove("hidden");
  updateToolbar();
  updateZipBtn();

  requestAnimationFrame(() => refreshAll());
}

function renderResultMetadata() {
  const meta = currentJobMetadata;
  const el = $("result-meta");
  el.replaceChildren();
  el.classList.toggle("hidden", !meta);
  if (!meta) return;
  if (meta.model_label) el.append(uiElement("span", "result-model", meta.model_label));
  const quality = { fast: t("quality.fast"), standard: t("quality.standard"), high: t("quality.high") }[meta.quality];
  if (quality) el.append(uiElement("span", "", quality));
  if (meta.elapsed !== undefined) el.append(uiElement("span", "", t("result.elapsed", { sec: meta.elapsed })));
  if (meta.output_channels === 1) el.append(uiElement("span", "preview-badge", t("result.mono")));
  if (meta.is_preview) {
    const actual = Number(meta.duration || duration());
    const requested = Number(meta.preview_duration || actual);
    const parts = [t("result.preview_from", { start: Number(meta.preview_start || 0), dur: actual.toFixed(1) })];
    if (Math.abs(actual - requested) > 0.1) parts.push(t("result.preview_short", { requested }));
    parts.push(t("result.preview_zero"));
    el.append(uiElement("span", "preview-badge", parts.join(" · ")));
  }
  if (meta.mode === "mega") el.append(uiElement("p", "result-warning", t("result.mega_warning")));
}

function bufToChans(buffer) {
  const chans = [];
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const src = buffer.getChannelData(c);
    const arr = new Float32Array(lengthSamples);
    arr.set(src.subarray(0, Math.min(src.length, lengthSamples)));
    chans.push(arr);
  }
  if (chans.length === 1) chans.push(new Float32Array(chans[0])); // mono→stereo
  return chans;
}

function makeTrack(def, bufferOrChans) {
  const track = {
    ...def,
    chans: null,
    buffer: null,
    muted: def.defaultMuted,
    solo: false,
    volume: 1,
    source: null,
    gainNode: audioCtx.createGain(),
  };
  track.volumeDb = 0;
  track.chans = Array.isArray(bufferOrChans) ? bufferOrChans : bufToChans(bufferOrChans);
  track.gainNode.connect(masterGain || audioCtx.destination);
  track.analyser = audioCtx.createAnalyser();
  track.analyser.fftSize = 1024;
  track.gainNode.connect(track.analyser);

  const el = document.createElement("div");
  el.className = "track";
  el.innerHTML = `
    <div class="track-side">
      <div class="track-title">
        <span class="track-dot" style="color:${def.color};background:${def.color}"></span>
        <span class="track-label">${def.label}</span>
        <span class="gain-badge" title="${t("track.gain_title")}"></span>
      </div>
      <div class="track-controls">
        <button class="ms-btn btn-m" title="${t("track.mute")}">M</button>
        <button class="ms-btn btn-s" title="${t("track.solo")}">S</button>
        <label class="exp-chk" title="${t("track.export_title")}"><input type="checkbox" class="exp-input"${def.isOriginal ? "" : " checked"}>💾</label>
        <button class="dl-btn" title="${t("track.download_title")}">⬇</button>
        ${def.isOriginal ? "" : `<button class="dl-btn sub-btn" title="${t("track.subsep_title")}">🔬</button>`}
        ${def.isOriginal ? "" : `<button class="dl-btn del-btn" title="${t("track.delete_title")}">✕</button>`}
      </div>
      <div class="track-vol">
        <input type="range" class="vol-slider" min="-36" max="12" step="0.5" value="0" title="${t("track.vol_title")}">
        <input type="number" class="vol-num" min="-60" max="12" step="0.5" value="0" title="${t("track.vol_num_title")}">
        <span class="vol-unit">dB</span>
      </div>
      <div class="track-meter" title="${t("track.meter_title")}">
        <div class="meter-mask"></div>
        <span class="meter-clip">●</span>
      </div>
    </div>
    <div class="track-wave">
      <canvas></canvas>
      <div class="selection-overlay hidden"></div>
      <div class="playhead" style="left:0"></div>
    </div>`;
  tracksEl.appendChild(el);

  track.el = el;
  track.waveEl = el.querySelector(".track-wave");
  track.canvas = el.querySelector("canvas");
  track.playheadEl = el.querySelector(".playhead");
  track.selectionEl = el.querySelector(".selection-overlay");
  track.btnM = el.querySelector(".btn-m");
  track.btnS = el.querySelector(".btn-s");

  track.btnM.addEventListener("click", () => { track.muted = !track.muted; applyGains(); });
  track.btnS.addEventListener("click", () => { track.solo = !track.solo; applyGains(); });

  // 音量: dBスライダー + 数値入力を同期
  track.volSlider = el.querySelector(".vol-slider");
  track.volNum = el.querySelector(".vol-num");
  track.meterMask = el.querySelector(".meter-mask");
  track.clipEl = el.querySelector(".meter-clip");
  const setVol = (db) => {
    track.volumeDb = clamp(isNaN(db) ? 0 : db, -60, 12);
    track.volume = Math.pow(10, track.volumeDb / 20);
    track.volSlider.value = clamp(track.volumeDb, -36, 12);
    track.volNum.value = track.volumeDb;
    applyGains();
  };
  track.volSlider.addEventListener("input", (e) => setVol(parseFloat(e.target.value)));
  track.volNum.addEventListener("change", (e) => setVol(parseFloat(e.target.value)));
  track.exportChecked = !def.isOriginal;
  el.querySelector(".exp-input").addEventListener("change", (e) => {
    track.exportChecked = e.target.checked;
    updateZipBtn();
  });
  el.querySelector(".dl-btn").addEventListener("click", () => downloadTrack(track));
  const delBtn = el.querySelector(".del-btn");
  if (delBtn) delBtn.addEventListener("click", () => deleteTrack(track));
  const subBtn = el.querySelector(".sub-btn");
  if (subBtn) subBtn.addEventListener("click", () => openSubsepPanel(track));

  // ラベルのダブルクリックでリネーム
  track.labelEl = el.querySelector(".track-label");
  track.gainBadgeEl = el.querySelector(".gain-badge");
  track.labelEl.title = t("track.rename_title");
  track.labelEl.addEventListener("dblclick", () => renameTrack(track));

  attachSelectionHandlers(track);
  if (track.muted) syncTrackUI(track);
  return track;
}

// ============================================================
// トラック管理 (追加・リネーム・削除)
// ============================================================
const TRACK_PALETTE = ["#f87171", "#22d3ee", "#a3e635", "#e879f9", "#fdba74", "#93c5fd"];
let customTrackCount = 0;

function addTrack() {
  if (!tracks.length || editorBusy) return;
  customTrackCount++;
  const def = {
    name: `custom${customTrackCount}`,
    label: t("track.new", { n: customTrackCount }),
    color: TRACK_PALETTE[(customTrackCount - 1) % TRACK_PALETTE.length],
    defaultMuted: false,
    isOriginal: false,
    custom: true,
  };
  const chans = [new Float32Array(lengthSamples), new Float32Array(lengthSamples)];
  const track = makeTrack(def, chans);
  // 元音源トラックの上に配置する
  const orig = tracks.find((t) => t.isOriginal);
  if (orig) {
    tracksEl.insertBefore(track.el, orig.el);
    tracks.splice(tracks.indexOf(orig), 0, track);
  } else {
    tracks.push(track);
  }
  // トラック構成の変更で取り消し履歴は無効になる
  undoStack = [];
  redoStack = [];
  setActiveTrack(tracks.indexOf(track));
  layoutTrackCanvas(track);
  refreshAll();
  updateToolbar();
  updateZipBtn();
  flashInfo(t("toast.track_added", { label: track.label }));
}

function renameTrack(track) {
  const input = document.createElement("input");
  input.type = "text";
  input.value = track.label;
  input.className = "rename-input";
  input.maxLength = 30;
  track.labelEl.replaceWith(input);
  input.focus();
  input.select();
  const commit = () => {
    const v = input.value.trim();
    if (v) track.label = v;
    track.labelEl.textContent = track.label;
    input.replaceWith(track.labelEl);
    renderSelection();
    updateToolbar();
  };
  input.addEventListener("blur", commit);
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") input.blur();
    if (e.key === "Escape") { input.value = track.label; input.blur(); }
  });
}

function deleteTrack(track) {
  if (editorBusy) return;
  if (tracks.filter((t) => !t.isOriginal).length <= 1) { showError(t("error.last_track")); return; }
  if (!confirm(t("confirm.delete_track", { label: track.label }))) return;
  if (isPlaying) pause();
  try { track.gainNode.disconnect(); } catch (e) {}
  track.el.remove();
  tracks.splice(tracks.indexOf(track), 1);
  if (clipboard && clipboard.sourceLabel === track.label) clipboard = null;
  undoStack = [];
  redoStack = [];
  activeTrackIdx = -1;
  tracks.forEach((t) => t.el.classList.remove("active-track"));
  clearSelection();
  refreshAll();
  updateToolbar();
  updateZipBtn();
}

// ============================================================
// 波形描画
// ============================================================
function computePeaks(track, width) {
  // 表示範囲 (viewStart〜viewStart+getViewDur) のピークを計算
  const chans = track.chans;
  const startSample = viewStart * sampleRate;
  const spp = (getViewDur() * sampleRate) / width;
  const peaks = new Float32Array(width * 2);
  for (let x = 0; x < width; x++) {
    let min = Infinity, max = -Infinity;
    let s0 = Math.floor(startSample + x * spp);
    let s1 = Math.floor(startSample + (x + 1) * spp);
    if (s1 <= s0) s1 = s0 + 1; // 高倍率時: 最低1サンプル
    s0 = Math.max(0, Math.min(s0, lengthSamples - 1));
    s1 = Math.min(s1, lengthSamples);
    const step = Math.max(1, Math.floor((s1 - s0) / 200));
    for (const ch of chans) {
      for (let s = s0; s < s1; s += step) {
        const v = ch[s];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min)) { min = 0; max = 0; }
    peaks[x * 2] = min;
    peaks[x * 2 + 1] = max;
  }
  return peaks;
}

function layoutTrackCanvas(track) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = track.waveEl.clientWidth;
  const h = track.waveEl.clientHeight;
  track.canvas.width = Math.round(w * dpr);
  track.canvas.height = Math.round(h * dpr);

  const off = document.createElement("canvas");
  off.width = track.canvas.width;
  off.height = track.canvas.height;

  if (viewMode === "spec") {
    if (track.gainBadgeEl) track.gainBadgeEl.textContent = "";
    renderSpectrogram(track, off);
    track.baseCanvas = off;
    return;
  }

  const peaks = computePeaks(track, track.canvas.width);
  const ctx = off.getContext("2d");
  const H = off.height, mid = H / 2;
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, track.color);
  grad.addColorStop(0.5, shade(track.color, 1.25));
  grad.addColorStop(1, track.color);
  if (track.displayGain == null) track.displayGain = computeDisplayGain(track);
  const gain = autoScale ? track.displayGain : 1;
  if (track.gainBadgeEl) {
    track.gainBadgeEl.textContent = gain > 1.05 ? t("track.display_gain", { gain: gain < 10 ? gain.toFixed(1) : Math.round(gain) }) : "";
  }
  const amp = (v) => clamp(v * gain, -1, 1) * mid * 0.92;

  const spp = (getViewDur() * sampleRate) / off.width;
  if (spp < 3) {
    // 高倍率: サンプル値を折れ線で描く (波形の形が正確に見える)
    ctx.strokeStyle = grad;
    ctx.lineWidth = Math.max(1.2, off.height / 70);
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let x = 0; x < off.width; x++) {
      const y = mid + amp((peaks[x * 2] + peaks[x * 2 + 1]) / 2);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  } else {
    ctx.fillStyle = grad;
    for (let x = 0; x < off.width; x++) {
      const y0 = mid + amp(peaks[x * 2]);
      const y1 = mid + amp(peaks[x * 2 + 1]);
      ctx.fillRect(x, Math.min(y0, y1), 1, Math.max(1, Math.abs(y1 - y0)));
    }
  }
  ctx.fillStyle = "rgba(255,255,255,0.10)";
  ctx.fillRect(0, mid - 0.5, off.width, 1);
  track.baseCanvas = off;
}

// ---------- スペクトログラム描画 ----------
const SPEC_LUT = (() => {
  // inferno風カラーマップ (暗→紫→赤→橙→黄→白)
  const stops = [
    [0.0, [10, 8, 24]],
    [0.25, [64, 18, 96]],
    [0.5, [168, 44, 92]],
    [0.75, [242, 122, 42]],
    [0.92, [252, 212, 84]],
    [1.0, [255, 255, 224]],
  ];
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    let k = 0;
    while (k < stops.length - 2 && x > stops[k + 1][0]) k++;
    const [x0, c0] = stops[k], [x1, c1] = stops[k + 1];
    const t = clamp((x - x0) / (x1 - x0), 0, 1);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(c0[c] + (c1[c] - c0[c]) * t);
  }
  return lut;
})();

let _hannWin = null;
function fftRadix2(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const vr = re[b] * cr - im[b] * ci;
        const vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr; im[a] += vi;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

function renderSpectrogram(track, off) {
  const FFT = 1024, HALF = FFT / 2;
  const W = Math.min(off.width, 1200);
  const H = Math.min(off.height, 200);
  if (!_hannWin || _hannWin.length !== FFT) {
    _hannWin = new Float32Array(FFT);
    for (let i = 0; i < FFT; i++) _hannWin[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (FFT - 1));
  }
  const ch = track.chans[0];
  const startS = viewStart * sampleRate;
  const spp = (getViewDur() * sampleRate) / W;
  const re = new Float32Array(FFT), im = new Float32Array(FFT);

  // 対数周波数軸 (上=高域): 40Hz〜ナイキスト
  const fmin = 40, fmax = sampleRate / 2;
  const rowBin = new Uint16Array(H);
  for (let y = 0; y < H; y++) {
    const f = fmin * Math.pow(fmax / fmin, 1 - y / (H - 1));
    rowBin[y] = clamp(Math.round((f * FFT) / sampleRate), 1, HALF - 1);
  }

  const tmp = document.createElement("canvas");
  tmp.width = W;
  tmp.height = H;
  const tctx = tmp.getContext("2d");
  const img = tctx.createImageData(W, H);
  const px = img.data;
  const norm = 1 / (FFT / 4); // Hann窓の振幅補正

  for (let x = 0; x < W; x++) {
    const center = Math.floor(startS + x * spp);
    const s0 = center - HALF;
    for (let i = 0; i < FFT; i++) {
      const s = s0 + i;
      re[i] = s >= 0 && s < lengthSamples ? ch[s] * _hannWin[i] : 0;
      im[i] = 0;
    }
    fftRadix2(re, im);
    for (let y = 0; y < H; y++) {
      const b = rowBin[y];
      const mag = Math.sqrt(re[b] * re[b] + im[b] * im[b]) * norm;
      const db = 20 * Math.log10(mag + 1e-7);
      const v = clamp((db + 90) / 70, 0, 1); // -90〜-20dB を表示レンジに
      const li = Math.round(v * 255) * 3;
      const p = (y * W + x) * 4;
      px[p] = SPEC_LUT[li];
      px[p + 1] = SPEC_LUT[li + 1];
      px[p + 2] = SPEC_LUT[li + 2];
      px[p + 3] = 255;
    }
  }
  tctx.putImageData(img, 0, 0);
  const ctx = off.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, off.width, off.height);
}

function shade(hex, factor) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.min(255, Math.round(((n >> 16) & 255) * factor));
  const g = Math.min(255, Math.round(((n >> 8) & 255) * factor));
  const b = Math.min(255, Math.round((n & 255) * factor));
  return `rgb(${r},${g},${b})`;
}

function drawTrack(track, timeSec) {
  if (!track.baseCanvas) layoutTrackCanvas(track); // 非表示タブ等でrAFが飛ばされた場合の保険
  const ctx = track.canvas.getContext("2d");
  const W = track.canvas.width, H = track.canvas.height;
  ctx.clearRect(0, 0, W, H);
  const vd = getViewDur();
  const frac = (timeSec - viewStart) / vd;
  const playedX = clamp(Math.round(frac * W), 0, W);
  ctx.globalAlpha = viewMode === "spec" ? 0.6 : 0.38;
  ctx.drawImage(track.baseCanvas, 0, 0);
  if (playedX > 0) {
    ctx.globalAlpha = 1;
    ctx.drawImage(track.baseCanvas, 0, 0, playedX, H, 0, 0, playedX, H);
  }
  ctx.globalAlpha = 1;
  if (frac >= 0 && frac <= 1) {
    track.playheadEl.style.display = "";
    track.playheadEl.style.left = frac * 100 + "%";
  } else {
    track.playheadEl.style.display = "none";
  }
}

function refreshAll() {
  // 編集で尺が変わった場合に表示範囲を補正
  if (viewDur !== null) {
    if (viewDur >= duration() - 1e-6) { viewStart = 0; viewDur = null; }
    else viewStart = clamp(viewStart, 0, duration() - viewDur);
  }
  tracks.forEach((t) => layoutTrackCanvas(t));
  $("time-total").textContent = fmtTime(duration());
  renderSelection();
  updateOverview();
  render();
}

// ============================================================
// 再生制御
// ============================================================
function currentTime() {
  return isPlaying ? Math.min(playOffset + audioCtx.currentTime - playStartCtxTime, duration()) : playOffset;
}

function applyGains() {
  const anySolo = tracks.some((t) => t.solo);
  tracks.forEach((t) => {
    const silent = t.muted || (anySolo && !t.solo);
    t.gainNode.gain.setTargetAtTime(silent ? 0 : t.volume, audioCtx.currentTime, 0.01);
    syncTrackUI(t, silent);
  });
}

function syncTrackUI(track, silent) {
  if (silent === undefined) silent = track.muted;
  track.btnM.classList.toggle("active-m", track.muted);
  track.btnS.classList.toggle("active-s", track.solo);
  track.el.classList.toggle("inactive", silent);
}

function trackBuffer(track) {
  if (!track.buffer || track.buffer.length !== lengthSamples) {
    const buf = audioCtx.createBuffer(track.chans.length, Math.max(lengthSamples, 1), sampleRate);
    track.chans.forEach((ch, i) => buf.copyToChannel(ch, i));
    track.buffer = buf;
  }
  return track.buffer;
}

function invalidateBuffers() {
  tracks.forEach((t) => { t.buffer = null; t.displayGain = null; });
}

function computeDisplayGain(track) {
  // トラック全体のピークから表示倍率を決める (ズームしても倍率が安定するよう表示範囲ではなく全体基準)
  let peak = 0;
  for (const c of track.chans) {
    for (let i = 0; i < c.length; i += 16) {
      const v = Math.abs(c[i]);
      if (v > peak) peak = v;
    }
  }
  if (peak < 1e-4) return 1;
  return clamp(0.95 / peak, 1, 30);
}

function startSources(offset) {
  tracks.forEach((t) => {
    const src = audioCtx.createBufferSource();
    src.buffer = trackBuffer(t);
    src.connect(t.gainNode);
    if (offset < duration()) src.start(0, offset);
    t.source = src;
  });
}

function stopSources() {
  tracks.forEach((t) => {
    if (t.source) { try { t.source.stop(); } catch (e) {} t.source = null; }
  });
}

async function play() {
  if (editorBusy) return;
  if (audioCtx.state === "suspended") await audioCtx.resume();
  // ループ再生: 選択範囲の外にいたら範囲の先頭から
  if (loopMode && selection && (playOffset < selection.start || playOffset >= selection.end - 0.01)) {
    playOffset = selection.start;
  }
  if (playOffset >= duration() - 0.01) playOffset = 0;
  startSources(playOffset);
  playStartCtxTime = audioCtx.currentTime;
  isPlaying = true;
  playBtn.textContent = "⏸";
  applyGains();
  tick();
}

function pause() {
  playOffset = currentTime();
  stopSources();
  isPlaying = false;
  playBtn.textContent = "▶";
  cancelAnimationFrame(rafId);
  render();
}

function seek(t) {
  t = Math.max(0, Math.min(t, duration()));
  if (isPlaying) {
    stopSources();
    playOffset = t;
    startSources(t);
    playStartCtxTime = audioCtx.currentTime;
  } else {
    playOffset = t;
    render();
  }
}

function tick() {
  render();
  if (isPlaying && loopMode && selection && currentTime() >= selection.end) {
    seek(selection.start);
  } else if (currentTime() >= duration()) {
    pause();
    playOffset = 0;
    render();
    return;
  }
  if (isPlaying) rafId = requestAnimationFrame(tick);
}

function render() {
  const t = currentTime();
  $("time-current").textContent = fmtTime(t);
  // 再生追従: ズーム中に再生位置が画面外へ出たらページ送り
  if (isPlaying && viewDur !== null && (t > viewStart + viewDur || t < viewStart)) {
    setView(t - viewDur * 0.08, viewDur);
  }
  tracks.forEach((tr) => drawTrack(tr, t));
  const op = $("overview-playhead");
  if (op) op.style.left = (t / duration()) * 100 + "%";
  updateMeters();
}

// ---------- レベルメーター ----------
function readPeak(analyser) {
  if (!meterBuf) meterBuf = new Float32Array(1024);
  analyser.getFloatTimeDomainData(meterBuf);
  let p = 0;
  for (let i = 0; i < meterBuf.length; i++) {
    const v = Math.abs(meterBuf[i]);
    if (v > p) p = v;
  }
  return p;
}

function setMeter(maskEl, clipEl, peak) {
  maskEl.style.width = (1 - clamp(peak, 0, 1)) * 100 + "%";
  if (peak >= 0.99) {
    clipEl.classList.add("clipping");
    clearTimeout(clipEl._t);
    clipEl._t = setTimeout(() => clipEl.classList.remove("clipping"), 1800);
  }
}

function updateMeters() {
  tracks.forEach((t) => {
    if (!t.analyser || !t.meterMask) return;
    setMeter(t.meterMask, t.clipEl, isPlaying ? readPeak(t.analyser) : 0);
  });
  const mm = $("master-meter-mask");
  if (mm && masterAnalyser) {
    setMeter(mm, $("master-clip"), isPlaying ? readPeak(masterAnalyser) : 0);
  }
}

playBtn.addEventListener("click", () => (isPlaying ? pause() : play()));
$("new-btn").addEventListener("click", () => location.reload());

function toggleLoop() {
  loopMode = !loopMode;
  $("loop-btn").classList.toggle("active", loopMode);
  if (loopMode && !selection) flashInfo(t("toast.loop_on_no_sel"));
  else flashInfo(loopMode ? t("toast.loop_on") : t("toast.loop_off"));
}
$("loop-btn").addEventListener("click", toggleLoop);

// ============================================================
// 範囲選択
// ============================================================
function attachSelectionHandlers(track) {
  const xToTime = (clientX, rect) =>
    viewStart + ((clientX - rect.left) / rect.width) * getViewDur();

  track.waveEl.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const rect = track.waveEl.getBoundingClientRect();
    const t0 = xToTime(e.clientX, rect);
    let moved = false;

    setActiveTrack(tracks.indexOf(track));

    const onMove = (ev) => {
      const t1 = clamp(xToTime(ev.clientX, rect), 0, duration());
      if (Math.abs(ev.clientX - e.clientX) > 4) moved = true;
      if (moved) {
        selection = { start: Math.min(t0, t1), end: Math.max(t0, t1) };
        renderSelection();
        updateToolbar();
      }
    };
    const onUp = (ev) => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      if (!moved) {
        selection = null;
        renderSelection();
        updateToolbar();
        seek(clamp(t0, 0, duration()));
      }
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });

  // Ctrl+ホイール = カーソル位置を中心にズーム / ホイール = 横スクロール (ズーム中のみ)
  track.waveEl.addEventListener("wheel", (e) => {
    const rect = track.waveEl.getBoundingClientRect();
    if (e.ctrlKey) {
      e.preventDefault();
      zoomAt(e.deltaY > 0 ? 1.35 : 1 / 1.35, xToTime(e.clientX, rect));
    } else if (viewDur !== null) {
      e.preventDefault();
      const delta = (e.deltaY || e.deltaX) > 0 ? 1 : -1;
      setView(viewStart + delta * getViewDur() * 0.15, getViewDur());
    }
  }, { passive: false });
}

function setActiveTrack(idx) {
  activeTrackIdx = idx;
  tracks.forEach((t, i) => t.el.classList.toggle("active-track", i === idx));
  renderSelection();
  updateToolbar();
}

function renderSelection() {
  const vd = getViewDur();
  tracks.forEach((t) => {
    if (!selection) { t.selectionEl.classList.add("hidden"); return; }
    const l = ((selection.start - viewStart) / vd) * 100;
    const r = ((selection.end - viewStart) / vd) * 100;
    if (r <= 0 || l >= 100) { t.selectionEl.classList.add("hidden"); return; }
    t.selectionEl.classList.remove("hidden");
    const cl = clamp(l, 0, 100);
    t.selectionEl.style.left = cl + "%";
    t.selectionEl.style.width = (clamp(r, 0, 100) - cl) + "%";
  });
  const info = $("selection-info");
  if (selection) {
    const label = activeTrackIdx >= 0 ? tracks[activeTrackIdx].label : "-";
    info.textContent = t("info.selection", { start: fmtTime(selection.start), end: fmtTime(selection.end), dur: (selection.end - selection.start).toFixed(2), label });
  } else if (activeTrackIdx >= 0) {
    info.textContent = t("info.target", { label: tracks[activeTrackIdx].label }) +
      (clipboard ? t("info.clipboard", { label: clipboard.sourceLabel, dur: (clipboard.length / sampleRate).toFixed(2) }) : "");
  } else {
    info.textContent = "";
  }
}

function clearSelection() {
  selection = null;
  renderSelection();
  updateToolbar();
}

// ============================================================
// 編集操作
// ============================================================
function pushUndo() {
  undoStack.push(snapshot());
  if (undoStack.length > UNDO_LIMIT) undoStack.shift();
  redoStack = [];
  updateToolbar();
}

function snapshot() {
  return {
    lengthSamples,
    tracks: tracks.map((t) => t.chans.map((c) => new Float32Array(c))),
  };
}

function restore(snap) {
  lengthSamples = snap.lengthSamples;
  tracks.forEach((t, i) => { t.chans = snap.tracks[i].map((c) => new Float32Array(c)); });
  invalidateBuffers();
  if (isPlaying) pause();
  playOffset = Math.min(playOffset, duration());
  clearSelection();
  refreshAll();
}

function undo() {
  if (!undoStack.length || editorBusy) return;
  redoStack.push(snapshot());
  restore(undoStack.pop());
  updateToolbar();
}

function redo() {
  if (!redoStack.length || editorBusy) return;
  undoStack.push(snapshot());
  restore(redoStack.pop());
  updateToolbar();
}

function selSamples() {
  if (!selection) return null;
  const s = Math.floor(selection.start * sampleRate);
  const e = Math.min(Math.ceil(selection.end * sampleRate), lengthSamples);
  return e - s > 0 ? [s, e] : null;
}

function fadeZero(chans, s, e) {
  // [s,e) を無音化。境界に短いフェードを入れてクリックノイズを防ぐ
  const f = Math.min(FADE_SAMPLES, Math.floor((e - s) / 2));
  chans.forEach((c) => {
    for (let i = 0; i < f; i++) {
      c[s + i] *= 1 - (i + 1) / f;      // 先頭: フェードアウト (1→0)
      c[e - 1 - i] *= 1 - (i + 1) / f;  // 末尾: フェードイン (0→1) ※逆順で適用
    }
    c.fill(0, s + f, e - f);
  });
}

function fadeEdges(clipChans) {
  // クリップの端に短いフェードを入れて貼り付け時のクリックノイズを防ぐ
  const len = clipChans[0].length;
  const f = Math.min(FADE_SAMPLES, Math.floor(len / 2));
  clipChans.forEach((c) => {
    for (let i = 0; i < f; i++) {
      const g = (i + 1) / f;
      c[i] *= g;
      c[len - 1 - i] *= g;
    }
  });
}

function grabSelection() {
  // 選択範囲 (アクティブトラック) をクリップボードへ
  const se = selSamples();
  if (!se || activeTrackIdx < 0) return null;
  const [s, e] = se;
  const t = tracks[activeTrackIdx];
  const chans = t.chans.map((c) => c.slice(s, e));
  fadeEdges(chans);
  clipboard = {
    start: s,
    length: e - s,
    chans,
    sourceLabel: t.label,
  };
  return [s, e];
}

function doCut() {
  if (editorBusy) return;
  if (activeTrackIdx < 0) { showError(t("error.select_track")); return; }
  if (isPlaying) pause();
  pushUndo();
  const se = grabSelection();
  if (!se) { undoStack.pop(); return; }
  const [s, e] = se;
  fadeZero(tracks[activeTrackIdx].chans, s, e);
  invalidateBuffers();
  refreshAll();
  updateToolbar();
  flashInfo(t("toast.cut", { label: clipboard.sourceLabel, dur: (clipboard.length / sampleRate).toFixed(2) }));
}

function doCopy() {
  if (activeTrackIdx < 0) { showError(t("error.select_track")); return; }
  if (!grabSelection()) return;
  updateToolbar();
  flashInfo(t("toast.copied", { label: clipboard.sourceLabel, dur: (clipboard.length / sampleRate).toFixed(2) }));
}

function doPaste() {
  if (!clipboard || editorBusy) return;
  if (activeTrackIdx < 0) { showError(t("error.pick_paste_target")); return; }
  if (isPlaying) pause();
  pushUndo();
  mixInto(tracks[activeTrackIdx], clipboard);
  const s = Math.min(clipboard.start, Math.max(lengthSamples - 1, 0));
  const e = Math.min(clipboard.start + clipboard.length, lengthSamples);
  invalidateBuffers();
  selection = { start: s / sampleRate, end: e / sampleRate };
  renderSelection();
  refreshAll();
  updateToolbar();
  flashInfo(t("toast.pasted", { label: tracks[activeTrackIdx].label, time: fmtTime(s / sampleRate) }));
}

function mixInto(track, clip) {
  // クリップ内容を元と同じ時間位置に加算合成する (尺は変えない)
  const s = Math.min(clip.start, Math.max(lengthSamples - 1, 0));
  const n = Math.min(clip.length, lengthSamples - s);
  if (n <= 0) return;
  track.chans.forEach((c, ci) => {
    const src = clip.chans[Math.min(ci, clip.chans.length - 1)];
    for (let i = 0; i < n; i++) c[s + i] += src[i];
  });
}

function doMove(targetIdx) {
  if (editorBusy || targetIdx < 0 || targetIdx === activeTrackIdx) return;
  if (activeTrackIdx < 0) { showError(t("error.select_source_track")); return; }
  const se = selSamples();
  if (!se) return;
  if (isPlaying) pause();
  pushUndo();
  const [s, e] = se;
  const src = tracks[activeTrackIdx];
  const clipChans = src.chans.map((c) => c.slice(s, e));
  fadeEdges(clipChans);
  const clip = { start: s, length: e - s, chans: clipChans };
  fadeZero(src.chans, s, e);
  mixInto(tracks[targetIdx], clip);
  invalidateBuffers();
  refreshAll();
  updateToolbar();
  flashInfo(t("toast.moved", { from: src.label, to: tracks[targetIdx].label, dur: (clip.length / sampleRate).toFixed(2) }));
}

function doRippleDelete() {
  const se = selSamples();
  if (!se || editorBusy) return;
  const [s, e] = se;
  if (e - s >= lengthSamples) { showError(t("error.cannot_delete_all")); return; }
  if (isPlaying) pause();
  pushUndo();
  tracks.forEach((t) => {
    t.chans = t.chans.map((c) => {
      const out = new Float32Array(lengthSamples - (e - s));
      out.set(c.subarray(0, s), 0);
      out.set(c.subarray(e), s);
      return out;
    });
  });
  lengthSamples -= e - s;
  invalidateBuffers();
  playOffset = Math.min(s / sampleRate, duration());
  clearSelection();
  refreshAll();
  flashInfo(t("toast.rippled", { dur: ((e - s) / sampleRate).toFixed(2) }));
}

function doSilence() {
  const se = selSamples();
  if (!se || editorBusy) return;
  if (activeTrackIdx < 0) { showError(t("error.pick_silence_target")); return; }
  const [s, e] = se;
  if (isPlaying) pause();
  pushUndo();
  fadeZero(tracks[activeTrackIdx].chans, s, e);
  invalidateBuffers();
  refreshAll();
  flashInfo(t("toast.silenced", { label: tracks[activeTrackIdx].label, dur: ((e - s) / sampleRate).toFixed(2) }));
}

// ---------- 無音カット ----------
function doSilenceCut(thresholdDb, minLenSec, padSec) {
  if (editorBusy) return;
  const thr = Math.pow(10, thresholdDb / 20);
  const win = 1024;
  const minLen = Math.floor(minLenSec * sampleRate);
  const pad = Math.floor(padSec * sampleRate);

  // 元音源トラックがあればそれで判定、なければ全ステムの最大振幅
  const source = tracks.find((t) => t.isOriginal) || null;
  const judgeTracks = source ? [source] : tracks;

  // 窓ごとの最大振幅
  const nWin = Math.ceil(lengthSamples / win);
  const env = new Float32Array(nWin);
  for (const t of judgeTracks) {
    for (const c of t.chans) {
      for (let w = 0; w < nWin; w++) {
        let m = env[w];
        const s0 = w * win, s1 = Math.min(s0 + win, lengthSamples);
        for (let s = s0; s < s1; s += 4) {
          const v = Math.abs(c[s]);
          if (v > m) m = v;
        }
        env[w] = m;
      }
    }
  }

  // 無音区間の検出
  const regions = [];
  let start = -1;
  for (let w = 0; w <= nWin; w++) {
    const silent = w < nWin && env[w] < thr;
    if (silent && start < 0) start = w * win;
    if (!silent && start >= 0) {
      let s = start + pad;
      let e = Math.min(w * win, lengthSamples) - pad;
      if (e - s >= minLen) regions.push([s, e]);
      start = -1;
    }
  }

  if (!regions.length) { showError(t("error.no_silence")); return; }

  const total = regions.reduce((a, [s, e]) => a + (e - s), 0);
  if (total >= lengthSamples) { showError(t("error.all_silent")); return; }

  if (isPlaying) pause();
  pushUndo();
  // 後ろから順に削除
  for (let i = regions.length - 1; i >= 0; i--) {
    const [s, e] = regions[i];
    tracks.forEach((t) => {
      t.chans = t.chans.map((c) => {
        const out = new Float32Array(c.length - (e - s));
        out.set(c.subarray(0, s), 0);
        out.set(c.subarray(e), s);
        return out;
      });
    });
    lengthSamples -= e - s;
  }
  invalidateBuffers();
  playOffset = 0;
  clearSelection();
  refreshAll();
  flashInfo(t("toast.silence_cut", { count: regions.length, dur: (total / sampleRate).toFixed(2) }));
}

// ---------- ノイズ除去 ----------
async function doDenoise(scope, strength) {
  const se = selSamples();
  if (!se) { showError(t("error.pick_noise_range")); return; }
  if (editorBusy) return;

  let targets;
  if (scope === "all") {
    targets = tracks.filter((t) => !t.isOriginal);
  } else {
    if (activeTrackIdx < 0) { showError(t("error.pick_target")); return; }
    targets = [tracks[activeTrackIdx]];
  }

  if (isPlaying) pause();
  editorBusy = true;
  updateToolbar();
  pushUndo();

  try {
    for (let i = 0; i < targets.length; i++) {
      const t = targets[i];
      flashInfo(t("status.denoising_track", { label: t.label, index: i + 1, total: targets.length }));
      const wav = encodeWav(t.chans, sampleRate);
      const form = new FormData();
      form.append("file", new Blob([wav], { type: "audio/wav" }), "track.wav");
      form.append("noise_start", String(selection.start));
      form.append("noise_end", String(selection.end));
      form.append("strength", String(strength));
      const res = await fetch("/api/denoise", { method: "POST", body: form });
      if (!res.ok) {
        throw new Error(await responseDetail(res, "error.denoise_failed"));
      }
      const buf = await audioCtx.decodeAudioData(await res.arrayBuffer());
      const chans = [];
      for (let c = 0; c < buf.numberOfChannels; c++) {
        const arr = new Float32Array(lengthSamples);
        arr.set(buf.getChannelData(c).subarray(0, Math.min(buf.length, lengthSamples)));
        chans.push(arr);
      }
      if (chans.length === 1) chans.push(new Float32Array(chans[0]));
      t.chans = chans;
    }
    invalidateBuffers();
    refreshAll();
    flashInfo(t("toast.denoise_done"));
  } catch (err) {
    showError(err.message);
    // 失敗時は undo で戻せる状態を維持
  } finally {
    editorBusy = false;
    updateToolbar();
  }
}

// ---------- 修復・エフェクト ----------
const REPAIR_NOTES = {
  declick: "repair.note.declick",
  dereverb: "repair.note.dereverb",
  nemotron_reuse: "repair.note.nemotron_reuse",
  dehum: "repair.note.dehum",
  fadein: "repair.note.fadein",
  fadeout: "repair.note.fadeout",
  normalize: "repair.note.normalize",
  loudnorm: "repair.note.loudnorm",
  pitchshift: "repair.note.pitchshift",
  tempo: "repair.note.tempo",
};

async function applyWholeTrackServerEffect(track, effect, extraForm, errorKey) {
  // トラック全体をサーバー処理して差し替える (長さ変化も許容)
  const form = new FormData();
  form.append("file", new Blob([encodeWav(track.chans, sampleRate)], { type: "audio/wav" }), "track.wav");
  form.append("effect", effect);
  for (const [k, v] of Object.entries(extraForm)) form.append(k, v);
  const res = await fetch("/api/effect", { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(await responseDetail(res, errorKey));
  }
  const buf = await audioCtx.decodeAudioData(await res.arrayBuffer());
  const chans = [];
  for (let c = 0; c < buf.numberOfChannels; c++) chans.push(new Float32Array(buf.getChannelData(c)));
  if (chans.length === 1) chans.push(new Float32Array(chans[0]));
  return chans;
}

async function applyPitchShift() {
  if (activeTrackIdx < 0) { showError(t("error.click_target")); return; }
  const steps = parseFloat($("rp-pitch").value) || 0;
  if (steps === 0) { showError(t("error.enter_semitones")); return; }
  if (editorBusy) return;
  const track = tracks[activeTrackIdx];
  if (isPlaying) pause();
  editorBusy = true;
  updateToolbar();
  pushUndo();
  try {
    const sign = steps > 0 ? "+" : "";
    flashInfo(t("status.pitch_shifting", { label: track.label, steps: `${sign}${steps}` }));
    const chans = await applyWholeTrackServerEffect(track, "pitchshift", { param: steps }, "error.pitch_failed");
    // 長さは不変のはずだが念のため合わせる
    track.chans = chans.map((c) => {
      const out = new Float32Array(lengthSamples);
      out.set(c.subarray(0, Math.min(c.length, lengthSamples)));
      return out;
    });
    invalidateBuffers();
    refreshAll();
    flashInfo(t("toast.pitch_done", { label: track.label, steps: `${sign}${steps}` }));
  } catch (err) {
    showError(err.message);
  } finally {
    editorBusy = false;
    updateToolbar();
  }
}

async function applyTempo() {
  const percent = parseFloat($("rp-tempo").value) || 100;
  if (percent === 100) { showError(t("error.tempo_not_100")); return; }
  if (editorBusy) return;
  const rate = clamp(percent / 100, 0.5, 2.0);
  if (isPlaying) pause();
  editorBusy = true;
  updateToolbar();
  pushUndo();
  try {
    const targets = tracks;
    const newChans = [];
    for (let i = 0; i < targets.length; i++) {
      flashInfo(t("status.tempo_changing", { label: targets[i].label, index: i + 1, total: targets.length }));
      newChans.push(await applyWholeTrackServerEffect(targets[i], "tempo", { param: rate }, "error.tempo_failed"));
    }
    lengthSamples = Math.max(...newChans.map((cs) => cs[0].length));
    targets.forEach((t, i) => {
      t.chans = newChans[i].map((c) => {
        const out = new Float32Array(lengthSamples);
        out.set(c.subarray(0, Math.min(c.length, lengthSamples)));
        return out;
      });
    });
    invalidateBuffers();
    playOffset = 0;
    clearSelection();
    refreshAll();
    flashInfo(t("toast.tempo_done", { percent }));
  } catch (err) {
    showError(err.message);
  } finally {
    editorBusy = false;
    updateToolbar();
  }
}

function applyClientEffect(track, s, e, effect) {
  const n = e - s;
  if (effect === "fadein") {
    track.chans.forEach((c) => { for (let i = 0; i < n; i++) c[s + i] *= i / n; });
  } else if (effect === "fadeout") {
    track.chans.forEach((c) => { for (let i = 0; i < n; i++) c[s + i] *= 1 - i / n; });
  } else if (effect === "normalize") {
    let peak = 0;
    track.chans.forEach((c) => { for (let i = s; i < e; i++) peak = Math.max(peak, Math.abs(c[i])); });
    if (peak < 1e-6) return false;
    const g = Math.pow(10, -1 / 20) / peak;
    track.chans.forEach((c) => { for (let i = s; i < e; i++) c[i] *= g; });
  }
  return true;
}

async function applyLoudnorm() {
  if (activeTrackIdx < 0) { showError(t("error.click_target")); return; }
  if (editorBusy) return;
  const track = tracks[activeTrackIdx];
  const platform = $("rp-platform").value;
  const target = platform === "custom" ? parseFloat($("rp-lufs").value) : parseFloat(platform);
  if (isNaN(target)) { showError(t("error.bad_lufs")); return; }
  if (isPlaying) pause();

  editorBusy = true;
  updateToolbar();
  pushUndo();
  try {
    flashInfo(t("status.loudness_measuring", { label: track.label }));
    const form = new FormData();
    form.append("file", new Blob([encodeWav(track.chans, sampleRate)], { type: "audio/wav" }), "track.wav");
    form.append("effect", "loudnorm");
    form.append("target_lufs", String(target));
    const res = await fetch("/api/effect", { method: "POST", body: form });
    if (!res.ok) {
      throw new Error(await responseDetail(res, "error.loudnorm_failed"));
    }
    const measured = res.headers.get("X-Measured-LUFS");
    const gainDb = res.headers.get("X-Applied-Gain-DB");
    const buf = await audioCtx.decodeAudioData(await res.arrayBuffer());
    const chans = [];
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const arr = new Float32Array(lengthSamples);
      arr.set(buf.getChannelData(c).subarray(0, Math.min(buf.length, lengthSamples)));
      chans.push(arr);
    }
    if (chans.length === 1) chans.push(new Float32Array(chans[0]));
    track.chans = chans;
    invalidateBuffers();
    refreshAll();
    flashInfo(t("toast.loudnorm_done", { label: track.label, measured, target, gain: `${gainDb > 0 ? "+" : ""}${gainDb}dB` }));
  } catch (err) {
    showError(err.message);
  } finally {
    editorBusy = false;
    updateToolbar();
  }
}

async function applyRepair() {
  const effect = $("rp-effect").value;
  if (effect === "loudnorm") return applyLoudnorm();
  if (effect === "pitchshift") return applyPitchShift();
  if (effect === "tempo") return applyTempo();

  const se = selSamples();
  if (!se) { showError(t("error.pick_repair_range")); return; }
  if (activeTrackIdx < 0) { showError(t("error.select_track")); return; }
  if (editorBusy) return;

  const mix = effect === "nemotron_reuse" ? 1 : parseFloat($("rp-mix").value);
  const track = tracks[activeTrackIdx];
  const [s, e] = se;
  if (isPlaying) pause();

  // クライアントサイドで完結するエフェクト
  if (effect === "fadein" || effect === "fadeout" || effect === "normalize") {
    pushUndo();
    applyClientEffect(track, s, e, effect);
    invalidateBuffers();
    refreshAll();
    flashInfo(t("toast.effect_applied", { label: track.label, effect: $("rp-effect").selectedOptions[0].textContent }));
    return;
  }

  // サーバー処理 (前後にコンテキストを付けて送り、選択範囲だけ書き戻す)
  editorBusy = true;
  updateToolbar();
  pushUndo();
  try {
    const ctx = Math.floor(1.5 * sampleRate);
    const cs = Math.max(0, s - ctx);
    const ce = Math.min(lengthSamples, e + ctx);
    const clip = track.chans.map((c) => c.slice(cs, ce));

    const form = new FormData();
    form.append("file", new Blob([encodeWav(clip, sampleRate)], { type: "audio/wav" }), "clip.wav");
    form.append("effect", effect);
    form.append("sensitivity", $("rp-sensitivity").value);
    form.append("base_freq", $("rp-humfreq").value);

    flashInfo(effect === "dereverb" ? t("status.dereverb_first") : t("status.processing"));
    const res = await fetch("/api/effect", { method: "POST", body: form });
    if (!res.ok) {
      throw new Error(await responseDetail(res, "error.repair_failed"));
    }
    const buf = await audioCtx.decodeAudioData(await res.arrayBuffer());

    const off = s - cs;
    const n = e - s;
    const f = Math.min(FADE_SAMPLES, Math.floor(n / 2));
    track.chans.forEach((c, ci) => {
      const p = buf.getChannelData(Math.min(ci, buf.numberOfChannels - 1));
      for (let i = 0; i < n; i++) {
        let g = mix;
        if (i < f) g *= (i + 1) / f;
        if (i >= n - f) g *= (n - i) / f;
        c[s + i] = c[s + i] * (1 - g) + p[off + i] * g;
      }
    });
    invalidateBuffers();
    refreshAll();
    flashInfo(t("toast.repaired", { label: track.label, dur: (n / sampleRate).toFixed(2) }));
  } catch (err) {
    showError(err.message);
  } finally {
    editorBusy = false;
    updateToolbar();
  }
}

// ---------- トラックの追加分離 (ドラム細分化 / リード・バック) ----------
let subsepTrack = null;
let subsepBusy = false;

function openSubsepPanel(track) {
  if (editorBusy) return;
  subsepTrack = track;
  $("ss-desc").textContent = t("panel.ss.desc", { label: track.label });
  // トラック名から種類を推定 (ラベルは表示言語に応じてja/en/zh両方を判定する)
  if (track.name === "vocals" || track.name === "dialog" || /ボーカル|声|人声|对白|说话|vocal|voice|dialog/i.test(track.label)) {
    $("ss-kind").value = "vocals";
  } else if (track.name === "drums" || /ドラム|鼓|drum/i.test(track.label)) {
    $("ss-kind").value = "drums";
  }
  togglePanel("panel-subsep");
}

async function runSubsep() {
  if (editorBusy || subsepBusy || !subsepTrack || !tracks.includes(subsepTrack)) return;
  const src = subsepTrack;
  const kind = $("ss-kind").value;
  subsepBusy = true;
  $("ss-run").disabled = true;
  try {
    flashInfo(t("status.uploading_track", { label: src.label }));
    const form = new FormData();
    form.append("file", new Blob([encodeWav(src.chans, sampleRate)], { type: "audio/wav" }), "track.wav");
    form.append("kind", kind);
    const res = await fetch("/api/subseparate", { method: "POST", body: form });
    if (!res.ok) throw await responseError(res, "error.subsep_start");
    const { id } = await res.json();

    let job;
    while (true) {
      await new Promise((r) => setTimeout(r, 1500));
      const jr = await fetch(`/api/jobs/${id}?${langParam()}`);
      if (!jr.ok) throw new Error(t("error.job_fetch"));
      job = await jr.json();
      if (job.status === "error") throw new Error(jobMessage(job));
      if (job.status === "done") break;
      flashInfo(t("toast.subsep_progress", { message: jobMessage(job) }));
    }

    flashInfo(t("status.loading_stems"));
    const buffers = await Promise.all(
      job.stems.map(async (s) => audioCtx.decodeAudioData(await (await fetch(s.url)).arrayBuffer()))
    );

    // 元トラックの直後に新トラックを挿入し、元はミュート
    if (isPlaying) pause();
    let insertAt = tracks.indexOf(src) + 1;
    job.stems.forEach((spec, i) => {
      const track = makeTrack(
        { name: spec.name, label: spec.label, color: spec.color, defaultMuted: false, isOriginal: false, custom: true },
        buffers[i]
      );
      tracksEl.insertBefore(track.el, tracks[insertAt] ? tracks[insertAt].el : null);
      tracks.splice(insertAt, 0, track);
      insertAt++;
    });
    src.muted = true;
    applyGains();
    undoStack = [];
    redoStack = [];
    clearSelection();
    refreshAll();
    updateToolbar();
    updateZipBtn();
    togglePanel("panel-subsep");
    flashInfo(t("toast.subsep_done", { label: src.label, count: job.stems.length }));
  } catch (err) {
    showError(err.message);
  } finally {
    subsepBusy = false;
    $("ss-run").disabled = false;
  }
}

// ---------- 話者ごとの単独発話と重なり区間 ----------
let speakerStemsBusy = false;
let speakerStemsSources = [];

function openSpeakerStemsPanel() {
  if (editorBusy || speakerStemsBusy) return;
  const selector = $("sp-track");
  selector.replaceChildren();
  speakerStemsSources = [...tracks];
  speakerStemsSources.forEach((track, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = track.isOriginal ? t("sp.option_original", { label: track.label }) : track.label;
    selector.append(option);
  });
  // 選択中のトラック、次に話し声トラックを候補にする。ステム表示名は言語ごとに異なるため全言語の語を拾う。
  const preferred = activeTrackIdx >= 0 ? activeTrackIdx : tracks.findIndex((t) =>
    t.name === "dialog" || t.name === "vocals" || /話し声|ボーカル|声|人声|对白|说话|voice|vocal|dialog/i.test(t.label));
  if (preferred >= 0) selector.value = String(preferred);
  $("sp-status").textContent = "";
  togglePanel("panel-speaker-stems");
}

async function runSpeakerStems() {
  if (speakerStemsBusy || editorBusy || subsepBusy || dialogueBusy || midiBusy) return;
  const src = speakerStemsSources[Number.parseInt($("sp-track").value, 10)];
  if (!src || !tracks.includes(src)) { showError(t("error.sp_track_missing")); return; }
  if (isPlaying) pause();
  speakerStemsBusy = true;
  editorBusy = true;
  $("sp-run").disabled = true;
  $("sp-track").disabled = true;
  updateToolbar();
  try {
    $("sp-status").textContent = t("status.preparing_audio");
    const form = new FormData();
    form.append("file", new Blob([encodeWav(src.chans, sampleRate)], { type: "audio/wav" }), "track.wav");
    const res = await fetch("/api/speaker-stems", { method: "POST", body: form });
    if (!res.ok) throw await responseError(res, "error.sp_start");
    const { id } = await res.json();
    let job;
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      const response = await fetch("/api/jobs/" + id);
      if (!response.ok) throw await responseError(response, "error.job_fetch");
      job = await response.json();
      if (job.status === "error") throw new Error(jobMessage(job) || t("error.sp_failed"));
      if (job.status === "done") break;
      $("sp-status").textContent = `${jobMessage(job) || t("status.sp_analyzing")} (${Math.round(job.progress || 0)}%)`;
    }
    if (!Array.isArray(job.stems) || !job.stems.length) throw new Error(t("error.sp_no_stems"));
    $("sp-status").textContent = t("status.loading_stems");
    const buffers = await Promise.all(job.stems.map(async (stem) => {
      const response = await fetch(stem.url);
      if (!response.ok) throw new Error(t("error.stem_load", { label: stem.label, status: response.status }));
      return audioCtx.decodeAudioData(await response.arrayBuffer());
    }));
    if (!tracks.includes(src)) throw new Error(t("error.sp_track_lost"));
    if (buffers.some((buffer) => buffer.length !== src.chans[0].length)) {
      throw new Error(t("error.sp_length_mismatch"));
    }

    // すべて読み込めた後に一括で追加。元トラックと試聴音量を揃える。
    let insertAt = tracks.indexOf(src) + 1;
    job.stems.forEach((spec, i) => {
      const track = makeTrack(
        { name: spec.name, label: spec.label, color: spec.color, defaultMuted: false, isOriginal: false, custom: true },
        buffers[i]
      );
      track.volumeDb = src.volumeDb;
      track.volume = src.volume;
      track.volSlider.value = clamp(track.volumeDb, -36, 12);
      track.volNum.value = track.volumeDb;
      track.solo = src.solo;
      tracksEl.insertBefore(track.el, tracks[insertAt] ? tracks[insertAt].el : null);
      tracks.splice(insertAt, 0, track);
      insertAt++;
    });
    src.solo = false;
    src.muted = true;
    applyGains();
    undoStack = [];
    redoStack = [];
    clearSelection();
    setActiveTrack(tracks.indexOf(src));
    refreshAll();
    updateZipBtn();
    // warnings は ja正本、warning_keys があれば翻訳を優先する
    const warningKeys = Array.isArray(job.warning_keys) ? job.warning_keys : [];
    const warnings = warningKeys.length
      ? warningKeys.map((key) => I18N.has(key) ? t(key) : "").filter(Boolean)
      : Array.isArray(job.warnings) ? job.warnings.filter((warning) => typeof warning === "string") : [];
    const message = t("sp.done_message", { label: src.label, count: job.stems.length });
    $("sp-status").textContent = [message, ...warnings].join(" ");
    flashInfo(warnings.length ? `${message} ${warnings.join(" ")}` : message);
  } catch (err) {
    $("sp-status").textContent = err.message;
    showError(err.message);
  } finally {
    speakerStemsBusy = false;
    editorBusy = false;
    $("sp-run").disabled = false;
    $("sp-track").disabled = false;
    updateToolbar();
  }
}

// ---------- MIDI書き出し ----------
let midiBusy = false;

function openMidiPanel() {
  const sel = $("md-track");
  sel.innerHTML = "";
  tracks.forEach((t, i) => {
    if (t.isOriginal) return;
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = t.label;
    sel.appendChild(opt);
  });
  const preferred = tracks.findIndex((t) => ["piano", "guitar", "bass", "vocals"].includes(t.name));
  if (preferred >= 0) sel.value = preferred;
  togglePanel("panel-midi");
}

async function runMidiExport() {
  if (midiBusy) return;
  const track = tracks[parseInt($("md-track").value, 10)];
  if (!track) { showError(t("error.pick_target")); return; }
  midiBusy = true;
  $("md-run").disabled = true;
  try {
    flashInfo(t("status.transcribing_midi", { label: track.label }));
    const form = new FormData();
    form.append("file", new Blob([encodeWav(track.chans, sampleRate)], { type: "audio/wav" }), "track.wav");
    form.append("onset", $("md-sens").value);
    form.append("min_note_ms", $("md-minnote").value);
    const res = await fetch("/api/midi", { method: "POST", body: form });
    if (!res.ok) {
      throw new Error(await responseDetail(res, "error.midi_failed"));
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${baseFilename}_${track.label}.mid`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    flashInfo(t("toast.midi_done", { label: track.label }));
  } catch (err) {
    showError(err.message);
  } finally {
    midiBusy = false;
    $("md-run").disabled = false;
  }
}

// ---------- セリフ書き出し (①解析 → ②一覧で調整 → ③書き出し) ----------
let dialogueBusy = false;
let dialogueSegments = []; // 編集可能なセグメント一覧
let dialogueTrackIdx = -1;
let dialogueSpeakersEnabled = false;

function openDialoguePanel() {
  // 対象トラックの選択肢を構築 (話し声/ボーカルがあれば初期選択)
  const sel = $("dg-track");
  sel.innerHTML = "";
  tracks.forEach((t, i) => {
    if (t.isOriginal) return;
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = t.label;
    sel.appendChild(opt);
  });
  const preferred = tracks.findIndex((t) => t.name === "dialog" || t.name === "vocals");
  if (preferred >= 0) sel.value = preferred;
  togglePanel("panel-dialogue");
}

function dialogueForm(track, extra) {
  const form = new FormData();
  form.append("file", new Blob([encodeWav(track.chans, sampleRate)], { type: "audio/wav" }), "track.wav");
  form.append("model_size", $("dg-model").value);
  form.append("language", $("dg-lang").value);
  form.append("base_name", baseFilename);
  for (const [k, v] of Object.entries(extra)) form.append(k, v);
  return form;
}

async function pollDialogueJob(id) {
  while (true) {
    await new Promise((r) => setTimeout(r, 1500));
    const jr = await fetch(`/api/jobs/${id}?${langParam()}`);
    if (!jr.ok) throw new Error(t("error.job_fetch"));
    const job = await jr.json();
    if (job.status === "error") throw new Error(jobMessage(job));
    if (job.status === "done") return job;
    flashInfo(t("toast.dialogue_progress", { message: jobMessage(job) }));
  }
}

const SPEAKER_COLORS = { "話者A": "#38bdf8", "話者B": "#f472b6", "話者C": "#a3e635", "話者D": "#fbbf24", "話者E": "#c084fc", "話者F": "#fb923c", "話者G": "#34d399", "話者H": "#f87171" };

// サーバー側の話者名 (話者A...) は正本のまま扱い、表示だけ現在の言語に合わせる
function speakerDisplayName(name) {
  const match = /^話者([A-Z])$/.exec(name);
  return match ? t("speaker.name", { letter: match[1] }) : name;
}

async function runDialogueAnalyze() {
  if (dialogueBusy) return;
  const trackIdx = parseInt($("dg-track").value, 10);
  const track = tracks[trackIdx];
  if (!track) { showError(t("error.pick_target")); return; }

  dialogueBusy = true;
  const requestDiarize = $("dg-diarize").checked;
  $("dg-run").disabled = true;
  $("dg-result").classList.add("hidden");
  try {
    flashInfo(t("status.analyzing", { label: track.label }));
    const form = dialogueForm(track, {
      action: "analyze",
      auto_adjust: $("dg-adjust").checked,
      min_len: $("dg-min").value,
      max_len: $("dg-max").value,
      diarize: requestDiarize,
      diarization_model: $("dg-diarization-model").value,
      num_speakers: $("dg-spknum").value,
    });
    const res = await fetch("/api/dialogue-export", { method: "POST", body: form });
    if (!res.ok) throw await responseError(res, "error.analyze_start");
    const { id } = await res.json();
    const job = await pollDialogueJob(id);

    dialogueSegments = job.segments.map((s) => ({ start: s.start, end: s.end, text: s.text, speaker: s.speaker || "" }));
    dialogueSpeakersEnabled = requestDiarize;
    dialogueTrackIdx = trackIdx;
    renderDialogueList(job.language);
    $("dg-result").classList.remove("hidden");
    flashInfo(t("toast.dialogue_detected", { count: dialogueSegments.length }));
  } catch (err) {
    showError(err.message);
  } finally {
    dialogueBusy = false;
    $("dg-run").disabled = false;
  }
}

function renderDialogueList(language) {
  const minL = parseFloat($("dg-min").value) || 0;
  const maxL = parseFloat($("dg-max").value) || Infinity;
  const durs = dialogueSegments.map((s) => s.end - s.start);
  const total = durs.reduce((a, b) => a + b, 0);
  $("dg-summary").textContent = language
    ? t("dialogue.summary_lang", { count: dialogueSegments.length, total: total.toFixed(1), lang: language, min: Math.min(...durs).toFixed(1), max: Math.max(...durs).toFixed(1) })
    : t("dialogue.summary", { count: dialogueSegments.length, total: total.toFixed(1), min: Math.min(...durs).toFixed(1), max: Math.max(...durs).toFixed(1) });

  const list = $("dg-segments");
  list.innerHTML = "";
  const hasSpeakers = dialogueSpeakersEnabled || dialogueSegments.some((s) => s.speaker);
  dialogueSegments.forEach((seg, i) => {
    const dur = seg.end - seg.start;
    const row = document.createElement("div");
    row.className = "dg-seg" + ($("dg-adjust").checked && (dur < minL || dur > maxL) ? " dg-warn" : "");
    const spkHtml = hasSpeakers
      ? `<select class="dg-spk" title="${t("dialogue.speaker_title")}"><option value=""${!seg.speaker ? " selected" : ""}>${t("dialogue.speaker_unassigned")}</option>${Object.keys(SPEAKER_COLORS).map((n) => `<option value="${n}"${n === seg.speaker ? " selected" : ""}>${speakerDisplayName(n)}</option>`).join("")}</select>`
      : "";
    row.innerHTML =
      `<button class="dg-play" title="${t("dialogue.play_title")}">▶</button>` +
      `<span class="dg-idx">${String(i + 1).padStart(3, "0")}</span>` +
      spkHtml +
      `<input class="dg-in dg-start" type="number" step="0.05" min="0" value="${seg.start.toFixed(2)}" title="${t("dialogue.start_title")}">` +
      `<span class="dg-sep">〜</span>` +
      `<input class="dg-in dg-end" type="number" step="0.05" min="0" value="${seg.end.toFixed(2)}" title="${t("dialogue.end_title")}">` +
      `<span class="dg-dur">${dur.toFixed(1)}s</span>` +
      `<input class="dg-text-in" type="text" value="" title="${t("dialogue.text_title")}">` +
      `<button class="dg-del" title="${t("dialogue.exclude_title")}">✕</button>`;
    row.querySelector(".dg-text-in").value = seg.text;
    const spkSel = row.querySelector(".dg-spk");
    if (spkSel) {
      spkSel.style.color = SPEAKER_COLORS[seg.speaker] || "#e8ecf5";
      spkSel.addEventListener("change", (e) => {
        seg.speaker = e.target.value;
        spkSel.style.color = SPEAKER_COLORS[seg.speaker] || "#e8ecf5";
      });
    }

    row.querySelector(".dg-play").addEventListener("click", () => {
      selection = { start: seg.start, end: seg.end };
      setActiveTrack(dialogueTrackIdx);
      renderSelection();
      updateToolbar();
      seek(seg.start);
    });
    row.querySelector(".dg-start").addEventListener("change", (e) => {
      seg.start = clamp(parseFloat(e.target.value) || 0, 0, seg.end - 0.05);
      renderDialogueList(language);
    });
    row.querySelector(".dg-end").addEventListener("change", (e) => {
      seg.end = clamp(parseFloat(e.target.value) || 0, seg.start + 0.05, duration());
      renderDialogueList(language);
    });
    row.querySelector(".dg-text-in").addEventListener("input", (e) => { seg.text = e.target.value; });
    row.querySelector(".dg-del").addEventListener("click", () => {
      dialogueSegments.splice(i, 1);
      renderDialogueList(language);
    });
    list.appendChild(row);
  });
}

async function runDialogueExportFiles() {
  if (dialogueBusy) return;
  const track = tracks[dialogueTrackIdx];
  if (!track || !dialogueSegments.length) { showError(t("error.analyze_first")); return; }

  dialogueBusy = true;
  $("dg-export").disabled = true;
  try {
    flashInfo(t("status.exporting"));
    const form = dialogueForm(track, {
      action: "export",
      segments: JSON.stringify(dialogueSegments),
      make_clips: $("dg-clips").checked,
      make_srt: $("dg-srt").checked,
      make_tts: $("dg-tts").checked,
      mono_clips: $("dg-mono").checked,
    });
    const res = await fetch("/api/dialogue-export", { method: "POST", body: form });
    if (!res.ok) throw await responseError(res, "error.export_start");
    const { id } = await res.json();
    const job = await pollDialogueJob(id);

    const a = document.createElement("a");
    a.href = job.zip_url;
    a.download = `${baseFilename}_dialogue.zip`;
    a.click();
    flashInfo(t("toast.dialogue_exported", { count: job.segments.length }));
  } catch (err) {
    showError(err.message);
  } finally {
    dialogueBusy = false;
    $("dg-export").disabled = false;
  }
}

// ============================================================
// ツールバー
// ============================================================
function updateToolbar() {
  const hasSel = !!selection;
  const hasTarget = activeTrackIdx >= 0;
  $("tool-cut").disabled = !hasSel || !hasTarget || editorBusy;
  $("tool-copy").disabled = !hasSel || !hasTarget || editorBusy;
  $("tool-paste").disabled = !clipboard || !hasTarget || editorBusy;
  $("tool-silence").disabled = !hasSel || !hasTarget || editorBusy;
  $("tool-ripple").disabled = !hasSel || editorBusy;
  $("tool-deselect").disabled = !hasSel;
  $("tool-silence-cut").disabled = editorBusy;
  $("tool-denoise").disabled = editorBusy;
  $("tool-repair").disabled = editorBusy;
  $("tool-speaker-stems").disabled = editorBusy;
  $("tool-dialogue").disabled = editorBusy;
  $("tool-midi").disabled = editorBusy;
  $("new-btn").disabled = editorBusy;
  $("tool-addtrack").disabled = editorBusy;
  $("tool-undo").disabled = !undoStack.length || editorBusy;
  $("tool-redo").disabled = !redoStack.length || editorBusy;
  $("tool-zoom-sel").disabled = !hasSel;
  updateMoveSelect();
}

function updateMoveSelect() {
  const sel = $("tool-move");
  const hasSel = !!selection && activeTrackIdx >= 0 && !editorBusy;
  sel.disabled = !hasSel;
  let html = `<option value="">${t("tool.move_target")}</option>`;
  if (hasSel) {
    tracks.forEach((track, i) => {
      if (i === activeTrackIdx || track.isOriginal) return;
      html += `<option value="${i}">${t("tool.move_option", { label: track.label })}</option>`;
    });
  }
  if (sel.innerHTML !== html) sel.innerHTML = html;
}

$("tool-cut").addEventListener("click", doCut);
$("tool-copy").addEventListener("click", doCopy);
$("tool-paste").addEventListener("click", doPaste);
$("tool-silence").addEventListener("click", doSilence);
$("tool-ripple").addEventListener("click", doRippleDelete);
$("tool-deselect").addEventListener("click", clearSelection);
$("tool-undo").addEventListener("click", undo);
$("tool-redo").addEventListener("click", redo);
$("tool-move").addEventListener("change", (e) => {
  const idx = parseInt(e.target.value, 10);
  e.target.value = "";
  if (!isNaN(idx)) doMove(idx);
});

// ---------- ズーム ----------
function zoomToSelection() {
  if (!selection || selection.end - selection.start < 0.001) return;
  const len = selection.end - selection.start;
  setView(selection.start - len * 0.15, len * 1.3);
}

$("tool-zoom-in").addEventListener("click", () => zoomAt(1 / 1.6, viewCenterT()));
$("tool-zoom-out").addEventListener("click", () => zoomAt(1.6, viewCenterT()));
$("tool-zoom-sel").addEventListener("click", zoomToSelection);
$("tool-zoom-full").addEventListener("click", () => setView(0, duration()));

$("tool-autoscale").addEventListener("click", () => {
  autoScale = !autoScale;
  localStorage.setItem("ssAutoScale", autoScale ? "1" : "0");
  $("tool-autoscale").classList.toggle("active", autoScale);
  requestViewRefresh();
  flashInfo(autoScale ? t("toast.autoscale_on") : t("toast.autoscale_off"));
});
$("tool-autoscale").classList.toggle("active", autoScale);

$("guide-close").addEventListener("click", () => {
  $("guide-card").classList.add("hidden");
  localStorage.setItem("ssGuideDismissed", "1");
});

$("tool-specview").addEventListener("click", () => {
  viewMode = viewMode === "spec" ? "wave" : "spec";
  localStorage.setItem("ssViewMode", viewMode);
  $("tool-specview").classList.toggle("active", viewMode === "spec");
  requestViewRefresh();
  flashInfo(viewMode === "spec" ? t("toast.spec_on") : t("toast.wave_back"));
});
$("tool-specview").classList.toggle("active", viewMode === "spec");

function viewCenterT() {
  // 再生位置が画面内ならそこを、外なら表示中央を基準にズーム
  const t = currentTime();
  if (t >= viewStart && t <= viewStart + getViewDur()) return t;
  return viewStart + getViewDur() / 2;
}

// 全体ミニマップ: クリック/ドラッグで表示位置を移動
$("overview").addEventListener("mousedown", (e) => {
  e.preventDefault();
  const rect = $("overview").getBoundingClientRect();
  const pan = (clientX) => {
    const t = ((clientX - rect.left) / rect.width) * duration();
    if (viewDur === null) seek(clamp(t, 0, duration()));
    else setView(t - viewDur / 2, viewDur);
  };
  pan(e.clientX);
  const onMove = (ev) => pan(ev.clientX);
  const onUp = () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  };
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
});

function togglePanel(id) {
  ["panel-silence-cut", "panel-denoise", "panel-repair", "panel-dialogue", "panel-guide", "panel-subsep", "panel-midi", "panel-speaker-stems"].forEach((p) => {
    if (p === id) $(p).classList.toggle("hidden");
    else $(p).classList.add("hidden");
  });
}
$("tool-guide").addEventListener("click", () => togglePanel("panel-guide"));
$("ss-run").addEventListener("click", runSubsep);
$("tool-midi").addEventListener("click", openMidiPanel);
$("md-run").addEventListener("click", runMidiExport);
$("md-sens").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  $("md-sens-val").textContent = v < 0.4 ? t("midi.sens_high") : v > 0.6 ? t("midi.sens_low") : t("midi.sens_standard");
});
$("ss-kind").addEventListener("change", () => {
  $("ss-note").textContent = t($("ss-kind").value === "drums" ? "panel.ss.note_drums" : "panel.ss.note_vocals");
});
// ガイド内のボタンは data-i18n-html で再構築されるため、document 宛の委譲で拾う
document.addEventListener("click", (e) => {
  const b = e.target.closest(".guide-open");
  if (!b) return;
  const target = b.dataset.panel;
  if (target === "panel-dialogue") openDialoguePanel();
  else togglePanel(target);
});
$("tool-silence-cut").addEventListener("click", () => togglePanel("panel-silence-cut"));
$("tool-denoise").addEventListener("click", () => togglePanel("panel-denoise"));
$("tool-repair").addEventListener("click", () => togglePanel("panel-repair"));
$("tool-dialogue").addEventListener("click", openDialoguePanel);
$("tool-speaker-stems").addEventListener("click", openSpeakerStemsPanel);
$("sp-run").addEventListener("click", runSpeakerStems);
$("dg-run").addEventListener("click", runDialogueAnalyze);
$("dg-export").addEventListener("click", runDialogueExportFiles);
$("tool-addtrack").addEventListener("click", addTrack);

$("rp-effect").addEventListener("change", () => {
  const v = $("rp-effect").value;
  $("rp-sens-row").classList.toggle("hidden", v !== "declick");
  $("rp-hum-row").classList.toggle("hidden", v !== "dehum");
  $("rp-mix-row").classList.toggle("hidden", !["declick", "dereverb", "dehum"].includes(v));
  $("rp-platform-row").classList.toggle("hidden", v !== "loudnorm");
  $("rp-lufs-row").classList.toggle("hidden", v !== "loudnorm" || $("rp-platform").value !== "custom");
  $("rp-pitch-row").classList.toggle("hidden", v !== "pitchshift");
  $("rp-tempo-row").classList.toggle("hidden", v !== "tempo");
  $("rp-note").textContent = REPAIR_NOTES[v] ? t(REPAIR_NOTES[v]) : "";
});
function updateDiarizationOptions() {
  const enabled = $("dg-diarize").checked;
  const nemotron = $("dg-diarization-model").value === "nemotron";
  $("dg-diarization-model-row").classList.toggle("hidden", !enabled);
  $("dg-spk-row").classList.toggle("hidden", !enabled || nemotron);
  $("dg-nemotron-note").classList.toggle("hidden", !enabled || !nemotron);
}
$("dg-diarize").addEventListener("change", updateDiarizationOptions);
$("dg-diarization-model").addEventListener("change", updateDiarizationOptions);
$("rp-platform").addEventListener("change", () => {
  const custom = $("rp-platform").value === "custom";
  $("rp-lufs-row").classList.toggle("hidden", !custom);
  if (!custom) $("rp-lufs").value = $("rp-platform").value;
});
$("rp-sensitivity").addEventListener("input", (e) => {
  $("rp-sens-val").textContent = Math.round(parseFloat(e.target.value) * 100) + "%";
});
$("rp-mix").addEventListener("input", (e) => {
  $("rp-mix-val").textContent = Math.round(parseFloat(e.target.value) * 100) + "%";
});
$("rp-run").addEventListener("click", applyRepair);
document.querySelectorAll(".panel-close").forEach((b) =>
  b.addEventListener("click", (e) => e.target.closest(".panel").classList.add("hidden"))
);

$("sc-run").addEventListener("click", () => {
  const thr = parseFloat($("sc-threshold").value);
  const minLen = parseFloat($("sc-minlen").value);
  const pad = parseFloat($("sc-pad").value);
  if (isNaN(thr) || isNaN(minLen) || isNaN(pad)) return;
  doSilenceCut(thr, minLen, pad);
});

$("dn-strength").addEventListener("input", (e) => {
  $("dn-strength-val").textContent = Math.round(parseFloat(e.target.value) * 100) + "%";
});
$("dn-run").addEventListener("click", () => {
  doDenoise($("dn-scope").value, parseFloat($("dn-strength").value));
});

// ============================================================
// キーボードショートカット
// ============================================================
document.addEventListener("keydown", (e) => {
  if (resultView.classList.contains("hidden")) return;
  const tag = e.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;

  if (e.code === "Space") {
    e.preventDefault();
    isPlaying ? pause() : play();
  } else if (e.ctrlKey && e.code === "KeyZ") {
    e.preventDefault(); undo();
  } else if (e.ctrlKey && (e.code === "KeyY" || (e.shiftKey && e.code === "KeyZ"))) {
    e.preventDefault(); redo();
  } else if (e.ctrlKey && e.code === "KeyX") {
    e.preventDefault(); doCut();
  } else if (e.ctrlKey && e.code === "KeyC") {
    e.preventDefault(); doCopy();
  } else if (e.ctrlKey && e.code === "KeyV") {
    e.preventDefault(); doPaste();
  } else if ((e.code === "Delete" || e.code === "Backspace") && e.shiftKey) {
    e.preventDefault(); doRippleDelete();
  } else if (e.code === "Delete" || e.code === "Backspace") {
    e.preventDefault(); doSilence();
  } else if (e.code === "Escape") {
    clearSelection();
    document.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
  } else if (e.key === "+" || e.key === "=") {
    e.preventDefault(); zoomAt(1 / 1.6, viewCenterT());
  } else if (e.key === "-") {
    e.preventDefault(); zoomAt(1.6, viewCenterT());
  } else if (e.key === "0") {
    e.preventDefault(); setView(0, duration());
  } else if (e.key === "z" && !e.ctrlKey) {
    e.preventDefault(); zoomToSelection();
  } else if (e.key === "l" && !e.ctrlKey) {
    e.preventDefault(); toggleLoop();
  }
});

// ============================================================
// 書き出し (編集内容を反映したクライアントサイドエンコード)
// ============================================================
function encodeWav(chans, sr, format = "float32") {
  const nCh = chans.length;
  const n = chans[0].length;
  const floatOutput = format === "float32";
  const bytesPerSample = floatOutput ? 4 : 2;
  const dataSize = n * nCh * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataSize);
  const v = new DataView(buf);
  const wstr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  wstr(0, "RIFF"); v.setUint32(4, 36 + dataSize, true); wstr(8, "WAVE");
  wstr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, floatOutput ? 3 : 1, true);
  v.setUint16(22, nCh, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * nCh * bytesPerSample, true);
  v.setUint16(32, nCh * bytesPerSample, true); v.setUint16(34, bytesPerSample * 8, true);
  wstr(36, "data"); v.setUint32(40, dataSize, true);
  let o = 44;
  for (let s = 0; s < n; s++) {
    for (let c = 0; c < nCh; c++) {
      if (floatOutput) v.setFloat32(o, chans[c][s], true);
      else {
        const x = Math.max(-1, Math.min(1, chans[c][s]));
        v.setInt16(o, x < 0 ? x * 0x8000 : x * 0x7fff, true);
      }
      o += bytesPerSample;
    }
  }
  return new Uint8Array(buf);
}

function downloadTrack(track) {
  const wav = encodeWav(track.chans, sampleRate, $("export-format").value);
  const url = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `${baseFilename}_${track.label}.wav`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

function updateZipBtn() {
  const n = tracks.filter((track) => track.exportChecked).length;
  const btn = $("zip-btn");
  btn.textContent = t("editor.zip_btn_count", { n });
  btn.disabled = n === 0;
  btn.title = n === 1 ? t("editor.zip_title_one") : t("editor.zip_title_multi");
}

$("zip-btn").addEventListener("click", () => {
  const targets = tracks.filter((t) => t.exportChecked);
  if (!targets.length) return;
  if (targets.length === 1) { downloadTrack(targets[0]); return; }
  const format = $("export-format").value;
  const files = targets.map((t) => ({ name: `${baseFilename}_${t.label}.wav`, data: encodeWav(t.chans, sampleRate, format) }));
  files.push({ name: "separation.json", data: new TextEncoder().encode(JSON.stringify({ ...currentJobMetadata, sample_rate: sampleRate, edited_duration: duration(), export_format: format, tracks: targets.map((t) => ({ name: t.name, label: t.label })) }, null, 2)) });
  const zip = makeZip(files);
  const url = URL.createObjectURL(new Blob([zip], { type: "application/zip" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `${baseFilename}_stems.zip`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
});

// ---------- プロジェクト保存 / 再開 (.ssproj = 無圧縮ZIP) ----------
function saveProject() {
  if (!tracks.length) return;
  const meta = {
    version: 1,
    app: "SoundtSeparator",
    sampleRate,
    baseFilename,
    sourceName: $("result-filename").textContent,
    separation: currentJobMetadata,
    tracks: tracks.map((t) => ({
      name: t.name,
      label: t.label,
      color: t.color,
      muted: t.muted,
      isOriginal: !!t.isOriginal,
      custom: !!t.custom,
      volumeDb: t.volumeDb,
      exportChecked: !!t.exportChecked,
    })),
  };
  const files = [
    { name: "project.json", data: new TextEncoder().encode(JSON.stringify(meta, null, 1)) },
    ...tracks.map((t, i) => ({ name: `track_${i}.wav`, data: encodeWav(t.chans, sampleRate, "float32") })),
  ];
  const zip = makeZip(files);
  const url = URL.createObjectURL(new Blob([zip], { type: "application/octet-stream" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `${baseFilename}.ssproj`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  flashInfo(t("toast.project_saved"));
}
$("save-proj-btn").addEventListener("click", saveProject);

function parseZip(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 22 - 65536); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error(t("error.proj_unrecognized"));
  const count = dv.getUint16(eocd + 10, true);
  let off = dv.getUint32(eocd + 16, true);
  const entries = new Map();
  const dec = new TextDecoder();
  for (let n = 0; n < count; n++) {
    if (dv.getUint32(off, true) !== 0x02014b50) break;
    const method = dv.getUint16(off + 10, true);
    const compSize = dv.getUint32(off + 20, true);
    const nameLen = dv.getUint16(off + 28, true);
    const extraLen = dv.getUint16(off + 30, true);
    const commentLen = dv.getUint16(off + 32, true);
    const lho = dv.getUint32(off + 42, true);
    const name = dec.decode(bytes.subarray(off + 46, off + 46 + nameLen));
    if (method !== 0) throw new Error(t("error.proj_wrong_app"));
    const lNameLen = dv.getUint16(lho + 26, true);
    const lExtraLen = dv.getUint16(lho + 28, true);
    const dataOff = lho + 30 + lNameLen + lExtraLen;
    entries.set(name, bytes.subarray(dataOff, dataOff + compSize));
    off += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

async function loadProject(file) {
  hideError();
  uploadView.classList.add("hidden");
  progressView.classList.remove("hidden");
  $("progress-filename").textContent = file.name;
  $("progress-message").textContent = t("status.loading_project");
  $("progress-percent").textContent = "";
  try {
    const entries = parseZip(new Uint8Array(await file.arrayBuffer()));
    const projRaw = entries.get("project.json");
    if (!projRaw) throw new Error(t("error.proj_broken"));
    const proj = JSON.parse(new TextDecoder().decode(projRaw));
    baseFilename = proj.baseFilename || file.name.replace(/\.[^.]+$/, "");
    currentJobMetadata = proj.separation || null;
    initAudio();

    const buffers = await Promise.all(
      proj.tracks.map(async (t, i) => {
        const data = entries.get(`track_${i}.wav`);
        if (!data) throw new Error(t("error.proj_track_missing", { file: `track_${i}.wav` }));
        return audioCtx.decodeAudioData(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
      })
    );
    sampleRate = buffers[0].sampleRate;
    lengthSamples = Math.max(...buffers.map((b) => b.length));
    tracksEl.innerHTML = "";
    tracks = proj.tracks.map((t, i) =>
      makeTrack(
        { name: t.name, label: t.label, color: t.color, defaultMuted: !!t.muted, isOriginal: !!t.isOriginal, custom: !!t.custom },
        buffers[i]
      )
    );
    // 音量・保存チェックを復元
    proj.tracks.forEach((t, i) => {
      const tr = tracks[i];
      if (typeof t.volumeDb === "number" && t.volumeDb !== 0) {
        tr.volNum.value = t.volumeDb;
        tr.volNum.dispatchEvent(new Event("change"));
      }
      if (t.exportChecked === false && !tr.isOriginal) tr.el.querySelector(".exp-input").click();
    });

    progressView.classList.add("hidden");
    resultView.classList.remove("hidden");
    $("result-filename").textContent = proj.sourceName || file.name;
    renderResultMetadata();
    $("time-total").textContent = fmtTime(duration());
    $("time-current").textContent = fmtTime(0);
    updateToolbar();
    updateZipBtn();
    requestAnimationFrame(() => refreshAll());
    flashInfo(t("toast.project_loaded", { count: tracks.length }));
  } catch (err) {
    showError(err.message);
    resetToUpload();
  }
}

// ---------- 最小ZIPライター (無圧縮 + UTF-8ファイル名) ----------
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[i] = c >>> 0;
  }
  return t;
})();

function crc32(data) {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function makeZip(files) {
  const enc = new TextEncoder();
  const parts = [];
  const central = [];
  let offset = 0;

  for (const f of files) {
    const name = enc.encode(f.name);
    const crc = crc32(f.data);
    const local = new ArrayBuffer(30);
    const v = new DataView(local);
    v.setUint32(0, 0x04034b50, true);
    v.setUint16(4, 20, true);
    v.setUint16(6, 0x0800, true); // UTF-8 フラグ
    v.setUint16(8, 0, true); // 無圧縮
    v.setUint32(14, crc, true);
    v.setUint32(18, f.data.length, true);
    v.setUint32(22, f.data.length, true);
    v.setUint16(26, name.length, true);

    const c = new ArrayBuffer(46);
    const cv = new DataView(c);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0x0800, true);
    cv.setUint16(10, 0, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, f.data.length, true);
    cv.setUint32(24, f.data.length, true);
    cv.setUint16(28, name.length, true);
    cv.setUint32(42, offset, true);
    central.push(new Uint8Array(c), name);

    parts.push(new Uint8Array(local), name, f.data);
    offset += 30 + name.length + f.data.length;
  }

  let centralSize = 0;
  central.forEach((p) => (centralSize += p.length));
  const end = new ArrayBuffer(22);
  const ev = new DataView(end);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);

  const totalSize = offset + centralSize + 22;
  const out = new Uint8Array(totalSize);
  let pos = 0;
  for (const p of [...parts, ...central, new Uint8Array(end)]) {
    out.set(p, pos);
    pos += p.length;
  }
  return out;
}

// ============================================================
// util
// ============================================================
let resizeTimer = null;
window.addEventListener("resize", () => {
  if (resultView.classList.contains("hidden")) return;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => refreshAll(), 150);
});

function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

// 操作結果・エラーは画面中央下のトーストで大きく表示する
let toastTimer = null;
function toast(msg, isError) {
  const el = $("toast");
  el.textContent = (isError ? "⚠ " : "") + msg;
  el.classList.remove("hidden", "toast-error", "toast-info");
  el.classList.add(isError ? "toast-error" : "toast-info");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), isError ? 6000 : 4500);
}
function flashInfo(msg) { toast(msg, false); }
function showError(msg) {
  toast(msg, true);
  $("error-message").textContent = msg;
  errorBox.classList.remove("hidden");
  $("retry-job").classList.toggle("hidden", !lastSubmittedFile || uploadView.classList.contains("hidden"));
}
function hideError() { $("toast").classList.add("hidden"); errorBox.classList.add("hidden"); }
function resetToUpload() {
  progressView.classList.add("hidden");
  resultView.classList.add("hidden");
  uploadView.classList.remove("hidden");
  fileInput.value = "";
  if (!errorBox.classList.contains("hidden")) $("retry-job").classList.toggle("hidden", !lastSubmittedFile);
  refreshModelCatalog();
}
