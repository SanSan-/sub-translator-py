"use strict";

const picker = document.getElementById("picker");
const pickerFileBtn = document.getElementById("pickerFileBtn");
const pickerFolderBtn = document.getElementById("pickerFolderBtn");
const translateBtn = document.getElementById("translateBtn");
const unloadBtn = document.getElementById("unloadBtn");
const fileList = document.getElementById("fileList");
const selectedPath = document.getElementById("selectedPath");
const settingsForm = document.getElementById("settingsForm");
const logConsole = document.getElementById("logConsole");
const clearLogs = document.getElementById("clearLogs");
const apiSelect = settingsForm.querySelector("[data-setting=\"api\"]");
const sourceLangSelect = settingsForm.querySelector("[data-setting=\"source_lang\"]");
const targetLangSelect = settingsForm.querySelector("[data-setting=\"target_lang\"]");
const batchSizeInput = settingsForm.querySelector("[data-setting=\"batch_size\"]");
const threadsInput = settingsForm.querySelector("[data-setting=\"threads\"]");
const googleSections = settingsForm.querySelectorAll("[data-section=\"google\"]");
const agentSections = settingsForm.querySelectorAll("[data-section=\"agent\"]");
const localSections = settingsForm.querySelectorAll("[data-section=\"local\"]");
const localWorkerSections = settingsForm.querySelectorAll("[data-section=\"local-worker\"]");
const localModelMeta = document.getElementById("localModelMeta");
const agentModelInput = settingsForm.querySelector("[data-setting=\"agent_model\"]");
const systemPromptBtn = document.getElementById("editSystemPrompt");
const userPromptBtn = document.getElementById("editUserPrompt");
const systemPromptModal = document.getElementById("systemPromptModal");
const userPromptModal = document.getElementById("userPromptModal");
const systemPromptArea = document.getElementById("systemPromptArea");
const userPromptArea = document.getElementById("userPromptArea");
const systemPromptStatus = document.getElementById("systemPromptStatus");
const userPromptStatus = document.getElementById("userPromptStatus");
const openAiKeyStatus = document.getElementById("openAiKeyStatus");
const smartSplitInput = settingsForm.querySelector("[data-setting=\"smart_split\"]");
const smartSplitControls = document.getElementById("smartSplitControls");
const smartSplitSettingsBtn = document.getElementById("smartSplitSettingsBtn");
const smartSplitModal = document.getElementById("smartSplitModal");
const smartSplitStatus = document.getElementById("smartSplitStatus");
const smartSplitInputs = smartSplitModal
  ? Array.from(smartSplitModal.querySelectorAll("[data-setting]"))
  : [];

const SETTINGS_STORAGE_KEY = "subTranslateSettings";
const JOB_ID_PATTERN = /^[a-f0-9]{32}$/;
const volatileSettingKeys = new Set([
  "agent_system_prompt",
  "agent_prompt",
  "agent_system_prompt_file",
  "agent_prompt_file",
]);
const allowedSettingKeys = new Set(
  Array.from(settingsForm.querySelectorAll("[data-setting]"))
    .map((input) => input.dataset.setting)
    .filter((key) => key && !volatileSettingKeys.has(key))
);
const allowCpuFallbackInput = settingsForm.querySelector("[data-setting=\"allow_cpu_fallback\"]");
const autoDownloadModelInput = settingsForm.querySelector("[data-setting=\"auto_download_model\"]");
const forceCacheInput = settingsForm.querySelector("[data-setting=\"force\"]");
const timeoutInput = settingsForm.querySelector("[data-setting=\"timeout\"]");

const statusLabels = {
  idle: "ожидание",
  started: "в работе",
  cached: "кеш",
  done: "готово",
  error: "ошибка",
};

const finalStates = new Set(["cached", "done", "error"]);
const googleApis = new Set(["google"]);
const agentApis = new Set(["agent"]);

const state = {
  items: [],
  itemMap: new Map(),
  mode: null,
  jobId: null,
  jobTotal: 0,
  jobDone: 0,
  completed: new Set(),
  uiConfig: null,
  lastThreadValue: null,
  agentConfigured: false,
};

function appendLog(message) {
  const line = message.endsWith("\n") ? message : `${message}\n`;
  logConsole.textContent += line;
  logConsole.scrollTop = logConsole.scrollHeight;
}

function setTranslateBusy(isBusy) {
  translateBtn.disabled = isBusy || state.items.length === 0;
  translateBtn.textContent = isBusy ? "Перевод: 0/0" : "Перевести";
  if (unloadBtn) {
    unloadBtn.disabled = isBusy;
  }
}

function updateTranslateProgress() {
  if (!state.jobId) {
    translateBtn.textContent = "Перевести";
    return;
  }
  translateBtn.textContent = `Перевод: ${state.jobDone}/${state.jobTotal}`;
}

function debounce(fn, delay) {
  let timer = null;
  return (...args) => {
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => fn(...args), delay);
  };
}

function setInputDefaults(input, value) {
  if (!input) {
    return;
  }
  const normalized = String(value);
  input.value = normalized;
  input.defaultValue = normalized;
}

function setSectionVisibility(sections, isVisible) {
  sections.forEach((section) => {
    section.hidden = !isVisible;
  });
}

function setSelectOptions(select, options, fallbackValue) {
  if (!select) {
    return;
  }
  const current = select.value;
  select.innerHTML = "";
  const values = [];
  options.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = item.label;
    select.append(option);
    values.push(item.value);
  });
  let nextValue = current && values.includes(current) ? current : fallbackValue;
  if (!values.includes(nextValue)) {
    nextValue = values[0] || "";
  }
  select.value = nextValue;
  select.defaultValue = nextValue;
}

function getTranslatorProfile(api) {
  const translators = state.uiConfig?.translators;
  if (!Array.isArray(translators)) {
    return null;
  }
  return translators.find((profile) => profile.id === api) || null;
}

function setTargetLanguageOptions(profile, languageConfig, fallbackValue) {
  let options = languageConfig?.target || [];
  const directions = Array.isArray(profile?.supported_directions)
    ? profile.supported_directions
    : [];
  if (directions.length > 0 && sourceLangSelect) {
    const allowedTargets = new Set(
      directions
        .filter((direction) => direction.source === sourceLangSelect.value)
        .map((direction) => direction.target)
    );
    options = options.filter((item) => allowedTargets.has(item.value));
  }
  setSelectOptions(targetLangSelect, options, fallbackValue);
}

function loadStoredSettings() {
  if (!window.localStorage) {
    return null;
  }
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    const sanitized = Object.fromEntries(
      Object.entries(parsed).filter(([key]) => allowedSettingKeys.has(key))
    );
    if (Object.keys(sanitized).length !== Object.keys(parsed).length) {
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(sanitized));
    }
    return sanitized;
  } catch {
    console.warn("Сохранённые настройки повреждены и будут проигнорированы.");
    return null;
  }
}

function persistSettings() {
  if (!window.localStorage) {
    return;
  }
  try {
    const settings = Object.fromEntries(
      Object.entries(readSettings()).filter(([key]) => allowedSettingKeys.has(key))
    );
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    console.warn("Браузер не разрешил сохранить настройки.");
  }
}

function setSelectValueIfExists(select, value) {
  if (!select || value === undefined || value === null) {
    return;
  }
  const hasOption = Array.from(select.options).some((item) => item.value === value);
  if (hasOption) {
    select.value = value;
    select.defaultValue = value;
  }
}

function applyStoredSettings(stored) {
  if (!stored || typeof stored !== "object") {
    return;
  }
  if (apiSelect && stored.api) {
    setSelectValueIfExists(apiSelect, stored.api);
  }
  updateApiDependentUi();
  const api = apiSelect ? apiSelect.value : "google";
  const profile = getTranslatorProfile(api);
  settingsForm.querySelectorAll("[data-setting]").forEach((input) => {
    const key = input.dataset.setting;
    if (key === "api" || key === "source_lang" || key === "target_lang") {
      return;
    }
    if (!Object.hasOwn(stored, key)) {
      return;
    }
    const value = stored[key];
    if (input.type === "checkbox") {
      input.checked = Boolean(value);
      input.defaultChecked = Boolean(value);
      return;
    }
    if (input.type === "number") {
      const parsed = Number.parseInt(value, 10);
      if (Number.isFinite(parsed)) {
        if (key === "threads") {
          state.lastThreadValue = String(parsed);
          if (profile && !profile.thread_safe) {
            return;
          }
        }
        setInputDefaults(input, parsed);
      }
      return;
    }
    setInputDefaults(input, value == null ? "" : value);
  });
  setSelectValueIfExists(sourceLangSelect, stored.source_lang);
  const languageConfig = state.uiConfig?.languages?.[api];
  setTargetLanguageOptions(profile, languageConfig, stored.target_lang || "ru");
  setSelectValueIfExists(targetLangSelect, stored.target_lang);
  updatePromptStatus();
  updateSmartSplitUi();
}

function updatePromptStatus() {
  if (!systemPromptStatus || !userPromptStatus) {
    return;
  }
  const systemText = systemPromptArea ? systemPromptArea.value.trim() : "";
  const userText = userPromptArea ? userPromptArea.value.trim() : "";
  systemPromptStatus.textContent = systemText
    ? `Системный промпт: ${systemText.length} символов.`
    : "Системный промпт не задан.";
  userPromptStatus.textContent = userText
    ? `Промпт агента: ${userText.length} символов.`
    : "Промпт агента не задан.";
  if (openAiKeyStatus) {
    openAiKeyStatus.textContent = state.agentConfigured
      ? "Ключ задан в окружении сервера."
      : "Ключ не задан в окружении сервера.";
  }
}

function readNumericValue(input) {
  if (!input) {
    return 0;
  }
  const parsed = Number.parseInt(input.value, 10);
  if (Number.isFinite(parsed)) {
    return parsed;
  }
  const fallback = Number.parseInt(input.defaultValue, 10);
  return Number.isFinite(fallback) ? fallback : 0;
}

function updateSmartSplitStatus() {
  if (!smartSplitStatus || smartSplitInputs.length === 0) {
    return;
  }
  const values = {
    smart_split_max_lines: readNumericValue(settingsForm.querySelector("[data-setting=\"smart_split_max_lines\"]")),
    smart_split_max_words: readNumericValue(settingsForm.querySelector("[data-setting=\"smart_split_max_words\"]")),
    smart_split_max_chars: readNumericValue(settingsForm.querySelector("[data-setting=\"smart_split_max_chars\"]")),
    smart_split_max_gap_ms: readNumericValue(settingsForm.querySelector("[data-setting=\"smart_split_max_gap_ms\"]")),
    smart_split_max_duration_ms: readNumericValue(
      settingsForm.querySelector("[data-setting=\"smart_split_max_duration_ms\"]")
    ),
  };
  smartSplitStatus.textContent = (
    `Лимиты: строки=${values.smart_split_max_lines}, слова=${values.smart_split_max_words}, ` +
    `символы=${values.smart_split_max_chars}, пауза=${values.smart_split_max_gap_ms} мс, ` +
    `длительность=${values.smart_split_max_duration_ms} мс.`
  );
}

function updateSmartSplitUi() {
  if (!smartSplitControls || !smartSplitInput) {
    return;
  }
  const isEnabled = smartSplitInput.checked;
  smartSplitControls.hidden = !isEnabled;
  if (!isEnabled && smartSplitModal) {
    smartSplitModal.hidden = true;
  }
  updateSmartSplitStatus();
}

function openSmartSplitModal() {
  if (!smartSplitModal) {
    return;
  }
  const snapshot = {};
  smartSplitInputs.forEach((input) => {
    snapshot[input.dataset.setting] = input.value;
  });
  smartSplitModal.dataset.prevValue = JSON.stringify(snapshot);
  smartSplitModal.hidden = false;
  const firstInput = smartSplitInputs[0];
  if (firstInput) {
    firstInput.focus();
  }
}

function closeSmartSplitModal(restore) {
  if (!smartSplitModal) {
    return;
  }
  if (restore) {
    try {
      const snapshot = JSON.parse(smartSplitModal.dataset.prevValue || "{}");
      smartSplitInputs.forEach((input) => {
        if (Object.hasOwn(snapshot, input.dataset.setting)) {
          input.value = snapshot[input.dataset.setting];
        }
      });
    } catch {
      console.warn("Снимок настроек объединения повреждён и будет проигнорирован.");
    }
  }
  smartSplitModal.hidden = true;
  updateSmartSplitStatus();
  persistSettings();
  refreshDebounced();
}

function openPromptModal(modal, area) {
  if (!modal || !area) {
    return;
  }
  modal.dataset.prevValue = area.value;
  modal.hidden = false;
  area.focus();
}

function closePromptModal(modal, area, restore) {
  if (!modal || !area) {
    return;
  }
  if (restore) {
    area.value = modal.dataset.prevValue || "";
  }
  modal.hidden = true;
  updatePromptStatus();
  if (!restore) {
    persistSettings();
  }
}

function updateApiDependentUi() {
  const api = apiSelect ? apiSelect.value : "google";
  const profile = getTranslatorProfile(api);
  const config = state.uiConfig || {};
  const langConfig = config.languages ? config.languages[api] : null;
  const defaults = config.defaults || {};
  if (langConfig) {
    setSelectOptions(sourceLangSelect, langConfig.source || [], defaults.source_lang || "en");
    setTargetLanguageOptions(profile, langConfig, defaults.target_lang || "ru");
  }

  if (threadsInput) {
    const requiresSingleThread = Boolean(profile && !profile.thread_safe);
    if (requiresSingleThread) {
      if (!threadsInput.disabled) {
        state.lastThreadValue = threadsInput.value || threadsInput.defaultValue;
      }
      setInputDefaults(threadsInput, 1);
      threadsInput.disabled = true;
    } else {
      const restore = state.lastThreadValue || defaults.threads || 3;
      threadsInput.disabled = false;
      setInputDefaults(threadsInput, restore);
    }
  }

  setSectionVisibility(googleSections, googleApis.has(api));
  setSectionVisibility(agentSections, agentApis.has(api));
  setSectionVisibility(localSections, Boolean(profile?.local));
  setSectionVisibility(localWorkerSections, profile?.runtime_kind === "isolated_worker");
  const profileTimeout = profile?.default_timeout_seconds || defaults.timeout || 30;
  setInputDefaults(timeoutInput, profileTimeout);
  updateLocalModelMeta(profile);
}

function updateLocalModelMeta(profile) {
  if (!localModelMeta || !profile?.local) {
    return;
  }
  const model = profile.model || {};
  const parts = [];
  if (model.id) {
    parts.push(`Модель: ${model.id}.`);
  }
  if (model.revision) {
    parts.push(`Ревизия: ${model.revision}.`);
  }
  if (model.quantization) {
    parts.push(`Квантование: ${model.quantization}.`);
  }
  if (model.requires_access_token) {
    parts.push(
      model.access_token_configured
        ? "Токен Hugging Face задан на сервере."
        : "Для загрузки нужен токен Hugging Face в окружении сервера."
    );
  }
  localModelMeta.textContent = parts.join(" ") || "Для профиля не задана модель.";
}

async function loadUiConfig() {
  const response = await fetch("/api/ui-config");
  if (!response.ok) {
    throw new Error("Не удалось получить настройки UI.");
  }
  const config = await response.json();
  state.uiConfig = config;
  const defaults = config.defaults || {};
  const translatorOptions = Array.isArray(config.translators)
    ? config.translators.map((profile) => ({ value: profile.id, label: profile.label }))
    : [];
  setSelectOptions(apiSelect, translatorOptions, defaults.api || "google");
  setInputDefaults(batchSizeInput, defaults.batch_size || 21);
  setInputDefaults(threadsInput, defaults.threads || 3);
  setInputDefaults(settingsForm.querySelector("[data-setting=\"tld\"]"), defaults.tld || "com");
  setInputDefaults(settingsForm.querySelector("[data-setting=\"timeout\"]"), defaults.timeout || 30);
  setInputDefaults(settingsForm.querySelector("[data-setting=\"request_delay_ms\"]"), defaults.request_delay_ms || 350);
  setInputDefaults(
    settingsForm.querySelector("[data-setting=\"smart_split_max_lines\"]"),
    defaults.smart_split_max_lines || 4
  );
  setInputDefaults(
    settingsForm.querySelector("[data-setting=\"smart_split_max_words\"]"),
    defaults.smart_split_max_words || 40
  );
  setInputDefaults(
    settingsForm.querySelector("[data-setting=\"smart_split_max_chars\"]"),
    defaults.smart_split_max_chars || 220
  );
  setInputDefaults(
    settingsForm.querySelector("[data-setting=\"smart_split_max_gap_ms\"]"),
    defaults.smart_split_max_gap_ms || 800
  );
  setInputDefaults(
    settingsForm.querySelector("[data-setting=\"smart_split_max_duration_ms\"]"),
    defaults.smart_split_max_duration_ms || 6000
  );
  if (allowCpuFallbackInput) {
    const allowFallback = Boolean(defaults.allow_cpu_fallback);
    allowCpuFallbackInput.checked = allowFallback;
    allowCpuFallbackInput.defaultChecked = allowFallback;
  }
  if (autoDownloadModelInput) {
    const autoDownload = Boolean(defaults.auto_download_model);
    autoDownloadModelInput.checked = autoDownload;
    autoDownloadModelInput.defaultChecked = autoDownload;
  }
  if (forceCacheInput) {
    const forceEnabled = Boolean(defaults.force);
    forceCacheInput.checked = forceEnabled;
    forceCacheInput.defaultChecked = forceEnabled;
  }
  if (systemPromptArea && config.agent_prompts) {
    systemPromptArea.value = config.agent_prompts.system || "";
  }
  if (userPromptArea && config.agent_prompts) {
    userPromptArea.value = config.agent_prompts.user || "";
  }
  const agentSettings = config.agent_settings || {};
  if (agentModelInput) {
    setInputDefaults(agentModelInput, agentSettings.model || "");
  }
  state.agentConfigured = Boolean(agentSettings.configured);
  updatePromptStatus();
  const stored = loadStoredSettings();
  if (stored) {
    applyStoredSettings(stored);
  } else {
    updateApiDependentUi();
    updateSmartSplitUi();
  }
}

function readSettings() {
  const data = {};
  settingsForm.querySelectorAll("[data-setting]").forEach((input) => {
    const key = input.dataset.setting;
    if (input.type === "checkbox") {
      data[key] = input.checked;
      return;
    }
    if (input.type === "number") {
      const parsed = Number.parseInt(input.value, 10);
      const fallback = Number.parseInt(input.defaultValue, 10);
      data[key] = Number.isFinite(parsed) ? parsed : fallback;
      return;
    }
    const rawValue = String(input.value || "");
    const value = rawValue.trim();
    if (key === "agent_system_prompt" || key === "agent_prompt") {
      data[key] = value === "" ? null : rawValue;
      return;
    }
    if (key === "agent_system_prompt_file" || key === "agent_prompt_file") {
      data[key] = value === "" ? null : value;
      return;
    }
    data[key] = value === "" ? input.defaultValue || "" : value;
  });
  return data;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      if (data?.detail) {
        detail = data.detail;
      }
    } catch {
      detail = response.statusText || "Сервер вернул некорректный ответ.";
    }
    throw new Error(detail);
  }

  return response.json();
}

function buildFileCard(item, index) {
  const card = document.createElement("div");
  const status = item.cached ? "cached" : "idle";
  card.className = "file-card";
  card.dataset.path = item.path;
  card.dataset.status = status;
  card.style.setProperty("--delay", `${Math.min(index, 6) * 0.05}s`);

  const row = document.createElement("div");
  row.className = "file-row";

  const dot = document.createElement("div");
  dot.className = "file-dot";

  const info = document.createElement("div");
  info.className = "file-info";

  const name = document.createElement("div");
  name.className = "file-name";
  name.textContent = item.name;

  const meta = document.createElement("div");
  meta.className = "file-meta";

  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = item.format.toUpperCase();

  const output = document.createElement("span");
  output.className = "file-output";
  output.textContent = `-> ${item.output}`;

  meta.append(badge, output);
  info.append(name, meta);

  const statusEl = document.createElement("div");
  statusEl.className = "file-status";
  statusEl.textContent = statusLabels[status];

  row.append(dot, info, statusEl);

  const progress = document.createElement("div");
  progress.className = "file-progress";

  const bar = document.createElement("span");
  bar.className = "file-progress-bar";
  bar.style.width = item.cached ? "100%" : "0%";
  progress.append(bar);

  card.append(row, progress);

  return { card, statusEl, bar, outputEl: output };
}

function renderItems(items, modeLabel) {
  state.items = items;
  state.itemMap = new Map();
  fileList.innerHTML = "";

  if (!items.length) {
    fileList.classList.add("empty");
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Список пуст - выберите файл или папку.";
    fileList.append(empty);
    translateBtn.disabled = true;
    selectedPath.textContent = modeLabel || "Папка не выбрана";
    return;
  }

  fileList.classList.remove("empty");
  const fragment = document.createDocumentFragment();
  items.forEach((item, index) => {
    const entry = buildFileCard(item, index);
    state.itemMap.set(item.path, entry);
    fragment.append(entry.card);
    if (
      item.state ||
      item.progress !== undefined ||
      item.output ||
      item.error
    ) {
      updateFileState(item.path, {
        status: item.state,
        progress: item.progress,
        output: item.output,
        error: item.error,
      });
    }
  });
  fileList.append(fragment);

  translateBtn.disabled = state.jobId !== null;
  selectedPath.textContent = modeLabel || "Выбрано файлов: " + items.length;
}

function updateFileState(path, payload) {
  const entry = state.itemMap.get(path);
  if (!entry) {
    return;
  }
  if (payload.status) {
    entry.card.dataset.status = payload.status;
    entry.statusEl.textContent = statusLabels[payload.status] || payload.status;
  }
  if (payload.progress !== undefined) {
    entry.bar.style.width = `${payload.progress}%`;
    const status = payload.status || entry.card.dataset.status;
    if (status === "started") {
      entry.statusEl.textContent = `${statusLabels.started} ${payload.progress}%`;
    }
  }
  if (payload.output) {
    entry.outputEl.textContent = `-> ${payload.output}`;
  }
  if (payload.error) {
    entry.card.title = payload.error;
  }
}

async function pick(kind) {
  const settings = readSettings();
  const result = await postJson("/api/pick", { kind, settings });
  if (result.mode === "folder") {
    renderItems(result.items || [], result.path || "Папка не выбрана");
    return;
  }
  renderItems(result.items || [], "Выбрано файлов: " + (result.items || []).length);
}

async function runPick(kind) {
  try {
    await pick(kind);
  } catch (err) {
    appendLog(`Ошибка выбора: ${err?.message || "неизвестная ошибка"}`);
  }
}

async function refreshCache() {
  if (state.items.length === 0 || state.jobId) {
    return;
  }
  const settings = readSettings();
  const paths = state.items.map((item) => item.path);
  const result = await postJson("/api/refresh", { paths, settings });
  renderItems(result.items || [], selectedPath.textContent);
}

async function loadActiveJob() {
  let result;
  try {
    const response = await fetch("/api/active-job");
    if (!response.ok) {
      return;
    }
    result = await response.json();
  } catch (err) {
    appendLog(`Ошибка синхронизации: ${err.message}`);
    return;
  }
  if (!result?.active) {
    return;
  }
  const items = result.items || [];
  renderItems(items, `Перевод: ${items.length} файлов`);
  state.jobId = result.job_id;
  state.jobTotal = Number.isFinite(result.total) ? result.total : items.length;
  state.jobDone = Number.isFinite(result.done) ? result.done : 0;
  state.completed = new Set();
  items.forEach((item) => {
    if (finalStates.has(item.state)) {
      state.completed.add(item.path);
    }
  });
  if (Array.isArray(result.logs)) {
    logConsole.textContent = "";
    result.logs.forEach((line) => appendLog(line));
  }
  setTranslateBusy(true);
  updateTranslateProgress();
  listenJob(result.job_id);
}

async function translate() {
  if (state.items.length === 0 || state.jobId) {
    return;
  }
  setTranslateBusy(true);
  state.jobDone = 0;
  state.jobTotal = 0;
  state.completed = new Set();
  updateTranslateProgress();

  try {
    const settings = readSettings();
    const paths = state.items.map((item) => item.path);
    const result = await postJson("/api/translate", { paths, settings });
    state.jobId = result.job_id;
    appendLog("Запущен перевод.\n");
    listenJob(result.job_id);
  } catch (err) {
    if (String(err.message || "").includes("Перевод уже выполняется")) {
      await loadActiveJob();
      return;
    }
    appendLog(`Ошибка запуска: ${err.message}`);
    state.jobId = null;
    setTranslateBusy(false);
  }
}

function listenJob(jobId) {
  if (!JOB_ID_PATTERN.test(jobId)) {
    appendLog("Сервер вернул некорректный идентификатор задачи.\n");
    state.jobId = null;
    setTranslateBusy(false);
    return;
  }
  const streamUrl = new URL(`/api/stream/${encodeURIComponent(jobId)}`, window.location.origin);
  const stream = new EventSource(streamUrl);

  stream.onmessage = (event) => {
    if (!event.data) {
      return;
    }
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      appendLog("Получено повреждённое событие задачи.\n");
      return;
    }
    if (payload.type === "log") {
      appendLog(payload.message);
      return;
    }
    if (payload.type === "job") {
      state.jobTotal = payload.total;
      updateTranslateProgress();
      return;
    }
    if (payload.type === "file") {
      updateFileState(payload.path, {
        status: payload.state,
        progress: payload.progress,
        output: payload.output,
        error: payload.error,
      });
      if (finalStates.has(payload.state) && !state.completed.has(payload.path)) {
        state.completed.add(payload.path);
        state.jobDone += 1;
        updateTranslateProgress();
      }
      return;
    }
    if (payload.type === "done") {
      const message = payload.status === "ok"
        ? "Перевод завершён.\n"
        : "Перевод завершён с ошибками.\n";
      appendLog(message);
      stream.close();
      state.jobId = null;
      setTranslateBusy(false);
    }
  };

  stream.onerror = () => {
    appendLog("Поток логов оборвался.\n");
    stream.close();
    state.jobId = null;
    setTranslateBusy(false);
  };
}

if (pickerFileBtn) {
  pickerFileBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    runPick("file");
  });
}

if (pickerFolderBtn) {
  pickerFolderBtn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    runPick("folder");
  });
}

picker.addEventListener("dragover", (event) => {
  event.preventDefault();
  picker.classList.add("drag-over");
});

picker.addEventListener("dragleave", () => {
  picker.classList.remove("drag-over");
});

picker.addEventListener("drop", async (event) => {
  event.preventDefault();
  event.stopPropagation();
  picker.classList.remove("drag-over");
  appendLog("Перетаскивание отключено. Используйте выбор файла/папки.\n");
});

translateBtn.addEventListener("click", () => {
  translate();
});

if (apiSelect) {
  apiSelect.addEventListener("change", () => {
    updateApiDependentUi();
    persistSettings();
    refreshCache().catch((err) => {
      appendLog(`Ошибка обновления: ${err.message}`);
    });
  });
}

if (sourceLangSelect) {
  sourceLangSelect.addEventListener("change", () => {
    const api = apiSelect ? apiSelect.value : "google";
    const profile = getTranslatorProfile(api);
    const languageConfig = state.uiConfig?.languages?.[api];
    const fallback = state.uiConfig?.defaults?.target_lang || "ru";
    setTargetLanguageOptions(profile, languageConfig, fallback);
    persistSettings();
    refreshDebounced();
  });
}

if (unloadBtn) {
  unloadBtn.addEventListener("click", async () => {
    if (state.jobId) return;
    try {
      await postJson("/api/unload", {});
      appendLog("Модели выгружены из памяти.\n");
    } catch (err) {
      appendLog(`Ошибка выгрузки: ${err.message}`);
    }
  });
}

clearLogs.addEventListener("click", () => {
  logConsole.textContent = "";
});

if (systemPromptBtn) {
  systemPromptBtn.addEventListener("click", () => openPromptModal(systemPromptModal, systemPromptArea));
}
if (userPromptBtn) {
  userPromptBtn.addEventListener("click", () => openPromptModal(userPromptModal, userPromptArea));
}
if (smartSplitSettingsBtn) {
  smartSplitSettingsBtn.addEventListener("click", () => openSmartSplitModal());
}
if (smartSplitInput) {
  smartSplitInput.addEventListener("change", () => updateSmartSplitUi());
}

function handlePromptModalAction(name, restore) {
  const modal = name === "system" ? systemPromptModal : userPromptModal;
  const area = name === "system" ? systemPromptArea : userPromptArea;
  closePromptModal(modal, area, restore);
}

function handleModalAction(name, restore) {
  if (name === "smart-split") {
    closeSmartSplitModal(restore);
    return;
  }
  if (name === "system" || name === "user") {
    handlePromptModalAction(name, restore);
  }
}

document.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  if (target.dataset.modalClose) {
    handleModalAction(target.dataset.modalClose, true);
    return;
  }
  if (target.dataset.modalSave) {
    handleModalAction(target.dataset.modalSave, false);
  }
});

if (systemPromptModal) {
  systemPromptModal.addEventListener("click", (event) => {
    if (event.target === systemPromptModal) {
      closePromptModal(systemPromptModal, systemPromptArea, true);
    }
  });
}

if (userPromptModal) {
  userPromptModal.addEventListener("click", (event) => {
    if (event.target === userPromptModal) {
      closePromptModal(userPromptModal, userPromptArea, true);
    }
  });
}
if (smartSplitModal) {
  smartSplitModal.addEventListener("click", (event) => {
    if (event.target === smartSplitModal) {
      closeSmartSplitModal(true);
    }
  });
}

const refreshDebounced = debounce(() => {
  refreshCache().catch((err) => {
    appendLog(`Ошибка обновления: ${err.message}`);
  });
}, 450);

settingsForm.addEventListener("input", (event) => {
  const target = event.target;
  if (target instanceof HTMLElement && target.classList.contains("prompt-area")) {
    return;
  }
  if (target instanceof HTMLElement && target.dataset.setting === "smart_split") {
    updateSmartSplitUi();
  }
  if (target instanceof HTMLElement && String(target.dataset.setting || "").startsWith("smart_split_")) {
    updateSmartSplitStatus();
  }
  persistSettings();
  refreshDebounced();
});

setTranslateBusy(false);

try {
  await loadUiConfig();
  await loadActiveJob();
} catch (err) {
  appendLog(`Ошибка загрузки настроек UI: ${err.message}`);
  updateApiDependentUi();
}
