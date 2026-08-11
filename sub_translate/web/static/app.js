"use strict";

const picker = document.getElementById("picker");
const pickerHint = document.getElementById("pickerHint");
const pickerFileBtn = document.getElementById("pickerFileBtn");
const pickerFolderBtn = document.getElementById("pickerFolderBtn");
const refreshBtn = document.getElementById("refreshBtn");
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
const BROWSER_LOG_LIMIT = 500;
const JOB_ID_PATTERN = /^[a-f0-9]{32}$/;
const PREPARATION_STATUS_PATH = "/api/preparation-status";
const PREPARATION_POLL_INTERVAL_MS = 400;
const PREPARATION_STATUS_TIMEOUT_MS = 4000;
const PREPARATION_STALE_WARNING_MS = 12000;
const CARD_PAGE_SIZE = 200;
const JOB_SYNC_RETRY_DELAY_MS = 500;
const DEFAULT_PICKER_HINT = "ASS, SRT и VTT; папка включает все подпапки";
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
  itemDataMap: new Map(),
  itemPage: 0,
  mode: null,
  sourcePaths: [],
  selectionLabel: "Папка не выбрана",
  jobId: null,
  jobTotal: 0,
  jobDone: 0,
  completed: new Set(),
  uiConfig: null,
  lastThreadValue: null,
  agentConfigured: false,
  isBusy: false,
  isPreparing: false,
  itemRequestGeneration: 0,
  itemRequestController: null,
  preparationGeneration: 0,
  preparationOperationId: null,
  preparationLogKeys: new Set(),
  logLines: [],
  eventSource: null,
  jobSnapshotTimer: null,
  jobSnapshotInProgress: false,
  jobSnapshotErrorShown: false,
  jobSnapshotReason: "disconnect",
  activeJobRetryTimer: null,
  activeJobRetryInProgress: false,
  activeJobSyncErrorShown: false,
  activeJobRecoveryContext: null,
  jobAuthorityTimer: null,
  jobAuthorityInProgress: false,
  jobAuthorityErrorShown: false,
  jobUnlockContext: null,
};

function appendLog(message) {
  if (message === undefined || message === null) {
    return;
  }
  const lines = String(message).replaceAll("\r\n", "\n").split("\n");
  if (lines.at(-1) === "") {
    lines.pop();
  }
  state.logLines.push(...lines);
  if (state.logLines.length > BROWSER_LOG_LIMIT) {
    state.logLines.splice(0, state.logLines.length - BROWSER_LOG_LIMIT);
  }
  logConsole.textContent = state.logLines.length > 0 ? `${state.logLines.join("\n")}\n` : "";
  logConsole.scrollTop = logConsole.scrollHeight;
}

function setTranslateBusy(isBusy) {
  state.isBusy = isBusy;
  translateBtn.textContent = isBusy ? "Перевод: 0/0" : "Перевести";
  updateInteractionState();
}

function setPickerHint(message, title = "") {
  if (!pickerHint) {
    return;
  }
  pickerHint.textContent = message || DEFAULT_PICKER_HINT;
  pickerHint.title = title;
}

function updateInteractionState() {
  const locked = state.isBusy || state.isPreparing;
  picker.classList.toggle("disabled", locked);
  picker.setAttribute("aria-busy", String(state.isPreparing));
  picker.setAttribute("aria-disabled", String(locked));
  pickerFileBtn.disabled = locked;
  pickerFolderBtn.disabled = locked;
  if (refreshBtn) {
    refreshBtn.disabled = locked || (state.items.length === 0 && state.sourcePaths.length === 0);
  }
  translateBtn.disabled = locked || state.items.length === 0;
  if (unloadBtn) {
    unloadBtn.disabled = locked;
  }
}

function setPreparing(isPreparing) {
  state.isPreparing = isPreparing;
  updateInteractionState();
}

function abortItemRequest() {
  state.itemRequestGeneration += 1;
  if (state.itemRequestController) {
    state.itemRequestController.abort();
    state.itemRequestController = null;
  }
}

function beginItemRequest() {
  abortItemRequest();
  const controller = new AbortController();
  state.itemRequestController = controller;
  return { controller, generation: state.itemRequestGeneration };
}

function completeItemRequest(controller) {
  if (state.itemRequestController === controller) {
    state.itemRequestController = null;
  }
}

function isAbortError(error) {
  return error?.name === "AbortError";
}

function updateTranslateProgress() {
  if (!state.jobId) {
    translateBtn.textContent = "Перевести";
    return;
  }
  translateBtn.textContent = `Перевод: ${state.jobDone}/${state.jobTotal}`;
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

function applyStoredCheckbox(input, key, value) {
  if (key === "allow_cpu_fallback" && input.disabled) {
    input.checked = false;
    input.defaultChecked = false;
    return;
  }
  input.checked = Boolean(value);
  input.defaultChecked = Boolean(value);
}

function applyStoredNumber(input, key, value, profile) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return;
  }
  if (key === "threads") {
    state.lastThreadValue = String(parsed);
    if (profile && !profile.thread_safe) {
      return;
    }
  }
  setInputDefaults(input, parsed);
}

function applyStoredInput(input, stored, profile) {
  const key = input.dataset.setting;
  if (key === "api" || key === "source_lang" || key === "target_lang") {
    return;
  }
  if (!Object.hasOwn(stored, key)) {
    return;
  }
  const value = stored[key];
  if (input.type === "checkbox") {
    applyStoredCheckbox(input, key, value);
    return;
  }
  if (input.type === "number") {
    applyStoredNumber(input, key, value, profile);
    return;
  }
  setInputDefaults(input, value == null ? "" : value);
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
    applyStoredInput(input, stored, profile);
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
  if (allowCpuFallbackInput) {
    const supportsCpuFallback = Boolean(profile?.supports_cpu_fallback);
    allowCpuFallbackInput.disabled = !supportsCpuFallback;
    if (!supportsCpuFallback) {
      allowCpuFallbackInput.checked = false;
    }
  }
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

async function getJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    cache: "no-store",
    signal: options.signal,
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function postJson(url, payload, options = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Accept": "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal: options.signal,
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
    const error = new Error(detail);
    error.status = response.status;
    throw error;
  }

  return response.json();
}

function preparationStatusGeneration(status) {
  const generation = Number(status?.generation);
  return Number.isSafeInteger(generation) && generation >= 0 ? generation : 0;
}

async function getPreparationStatus(signal) {
  const controller = new AbortController();
  let timedOut = false;
  const forwardAbort = () => controller.abort();
  if (signal?.aborted) {
    controller.abort();
  } else if (signal) {
    signal.addEventListener("abort", forwardAbort, { once: true });
  }
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, PREPARATION_STATUS_TIMEOUT_MS);
  try {
    return await getJson(PREPARATION_STATUS_PATH, { signal: controller.signal });
  } catch (error) {
    if (timedOut) {
      throw new Error("Локальный сервис не ответил на запрос состояния подготовки.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", forwardAbort);
  }
}

function resetPreparationLogCursor(status) {
  const operationId = status?.operation_id ? String(status.operation_id) : null;
  if (operationId !== state.preparationOperationId) {
    state.preparationOperationId = operationId;
    state.preparationLogKeys = new Set();
  }
}

function preparationCounter(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(0, Math.trunc(parsed)) : 0;
}

function preparationTotal(value) {
  if (value === null || value === undefined) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(0, Math.trunc(parsed)) : null;
}

function preparationSummary(status) {
  const discovered = preparationCounter(status?.discovered);
  const processed = preparationCounter(status?.processed);
  const total = preparationTotal(status?.total);
  const boundedProcessed = total === null ? processed : Math.min(processed, total);
  if (status?.status === "error") {
    return "Ошибка подготовки списка";
  }
  if (!status?.active && status?.status === "done") {
    return `Список готов: ${total ?? Math.max(processed, discovered)} файлов`;
  }
  if (status?.phase === "dialog") {
    return "Ожидается выбор в системном диалоге…";
  }
  if (status?.phase === "collecting") {
    return `Рекурсивный поиск: найдено ${discovered}`;
  }
  if (status?.phase === "probing") {
    return total === null ? `Подготовлено: ${processed}` : `Подготовлено: ${boundedProcessed}/${total}`;
  }
  if (status?.phase === "queueing") {
    return total === null
      ? `Формируется список: ${processed}`
      : `Формируется список: ${boundedProcessed}/${total}`;
  }
  return "Подготавливается список…";
}

function appendPreparationStatus(status, baselineGeneration, expectedOperation) {
  const generation = preparationStatusGeneration(status);
  if (generation <= baselineGeneration || status?.operation !== expectedOperation) {
    return false;
  }
  state.preparationGeneration = Math.max(state.preparationGeneration, generation);
  resetPreparationLogCursor(status);
  const entries = Array.isArray(status.logs) ? status.logs : [];
  entries.forEach((entry) => {
    const message = String(entry?.message || "").trim();
    const key = `log:${String(entry?.id ?? message)}`;
    if (message && !state.preparationLogKeys.has(key)) {
      state.preparationLogKeys.add(key);
      appendLog(message);
    }
  });
  const message = String(status?.message || "").trim();
  setPickerHint(preparationSummary(status), message);
  const messageKey = `message:${message}`;
  if (message && entries.length === 0 && !state.preparationLogKeys.has(messageKey)) {
    state.preparationLogKeys.add(messageKey);
    appendLog(message);
  }
  const error = String(status?.error || "").trim();
  const errorKey = `error:${error}`;
  if (error && !state.preparationLogKeys.has(errorKey)) {
    state.preparationLogKeys.add(errorKey);
    appendLog(`Ошибка подготовки: ${error}`);
  }
  return true;
}

function preparationDelay(signal) {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve(false);
      return;
    }
    const finish = (continued) => {
      clearTimeout(timer);
      signal.removeEventListener("abort", stop);
      resolve(continued);
    };
    const stop = () => finish(false);
    const timer = setTimeout(() => finish(true), PREPARATION_POLL_INTERVAL_MS);
    signal.addEventListener("abort", stop, { once: true });
  });
}

function isPreparationRequestActive(request) {
  return request.generation === state.itemRequestGeneration && !request.controller.signal.aborted;
}

function createPreparationMonitorState() {
  return {
    lastUpdatedAt: null,
    lastProgressAt: Date.now(),
    staleWarningShown: false,
    connectionWarningShown: false,
  };
}

function recordPreparationProgress(status, monitor) {
  const updatedAt = String(status.updated_at || "");
  if (!updatedAt || updatedAt === monitor.lastUpdatedAt) {
    return;
  }
  monitor.lastUpdatedAt = updatedAt;
  monitor.lastProgressAt = Date.now();
  monitor.staleWarningShown = false;
}

function handleMonitoredPreparationStatus(status, monitor, baselineGeneration, expectedOperation) {
  monitor.connectionWarningShown = false;
  if (!appendPreparationStatus(status, baselineGeneration, expectedOperation)) {
    return true;
  }
  recordPreparationProgress(status, monitor);
  return Boolean(status.active);
}

function handlePreparationConnectionError(error, request, monitor) {
  if (request.controller.signal.aborted) {
    return false;
  }
  if (!monitor.connectionWarningShown) {
    appendLog(`Потеряна связь с локальным сервисом во время подготовки: ${error.message}`);
    monitor.connectionWarningShown = true;
  }
  return true;
}

function reportStalePreparation(monitor) {
  if (
    monitor.staleWarningShown
    || Date.now() - monitor.lastProgressAt < PREPARATION_STALE_WARNING_MS
  ) {
    return;
  }
  appendLog("Подготовка не сообщает новый прогресс более 12 секунд; ожидание продолжается.");
  monitor.staleWarningShown = true;
}

async function monitorPreparation(request, baselineGeneration, expectedOperation, isComplete) {
  const monitor = createPreparationMonitorState();
  while (isPreparationRequestActive(request) && !isComplete()) {
    if (!(await preparationDelay(request.controller.signal))) {
      return;
    }
    try {
      const status = await getPreparationStatus(request.controller.signal);
      if (!handleMonitoredPreparationStatus(
        status,
        monitor,
        baselineGeneration,
        expectedOperation,
      )) {
        return;
      }
    } catch (error) {
      if (!handlePreparationConnectionError(error, request, monitor)) {
        return;
      }
    }
    reportStalePreparation(monitor);
  }
}

async function captureFinalPreparationStatus(request, baselineGeneration, expectedOperation) {
  if (request.controller.signal.aborted) {
    return;
  }
  try {
    const status = await getPreparationStatus(request.controller.signal);
    appendPreparationStatus(status, baselineGeneration, expectedOperation);
  } catch (error) {
    if (!request.controller.signal.aborted) {
      appendLog(`Не удалось получить итог подготовки: ${error.message}`);
    }
  }
}

async function postWithPreparationStatus(url, payload, request, expectedOperation) {
  let baselineGeneration = state.preparationGeneration;
  try {
    const baseline = await getPreparationStatus(request.controller.signal);
    baselineGeneration = Math.max(baselineGeneration, preparationStatusGeneration(baseline));
    state.preparationGeneration = baselineGeneration;
  } catch (error) {
    if (request.controller.signal.aborted) {
      throw error;
    }
    appendLog(`Состояние подготовки пока недоступно: ${error.message}`);
  }
  let complete = false;
  const responsePromise = postJson(url, payload, { signal: request.controller.signal });
  const monitorPromise = monitorPreparation(
    request,
    baselineGeneration,
    expectedOperation,
    () => complete,
  );
  try {
    return await responsePromise;
  } finally {
    complete = true;
    await monitorPromise;
    await captureFinalPreparationStatus(request, baselineGeneration, expectedOperation);
  }
}

function normalizedItemStatus(item) {
  if (item?.state && Object.hasOwn(statusLabels, item.state)) {
    return item.state;
  }
  return item?.cached ? "cached" : "idle";
}

function itemProgress(item, status) {
  const progress = Number(item?.progress);
  if (Number.isFinite(progress)) {
    return Math.max(0, Math.min(100, Math.round(progress)));
  }
  return finalStates.has(status) ? 100 : 0;
}

function applyCardState(entry, item) {
  const status = normalizedItemStatus(item);
  const progress = itemProgress(item, status);
  entry.card.dataset.status = status;
  entry.statusEl.textContent = status === "started" && progress > 0
    ? `${statusLabels.started} ${progress}%`
    : statusLabels[status];
  entry.bar.style.width = `${progress}%`;
  entry.outputEl.textContent = `-> ${item.output || ""}`;
  entry.card.title = item.error || "";
}

function buildFileCard(item, index) {
  const card = document.createElement("div");
  card.className = "file-card";
  card.dataset.path = String(item.path || "");
  card.style.setProperty("--delay", `${Math.min(index, 6) * 0.05}s`);

  const row = document.createElement("div");
  row.className = "file-row";

  const dot = document.createElement("div");
  dot.className = "file-dot";

  const info = document.createElement("div");
  info.className = "file-info";

  const name = document.createElement("div");
  name.className = "file-name";
  name.textContent = item.name || "Без имени";

  const meta = document.createElement("div");
  meta.className = "file-meta";

  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = String(item.format || "sub").toUpperCase();

  const output = document.createElement("span");
  output.className = "file-output";
  output.textContent = `-> ${item.output}`;

  meta.append(badge, output);
  info.append(name, meta);

  const statusEl = document.createElement("div");
  statusEl.className = "file-status";

  row.append(dot, info, statusEl);

  const progress = document.createElement("div");
  progress.className = "file-progress";

  const bar = document.createElement("span");
  bar.className = "file-progress-bar";
  progress.append(bar);

  card.append(row, progress);

  const entry = { card, statusEl, bar, outputEl: output };
  applyCardState(entry, item);
  return entry;
}

function buildPagination(page, pageCount, startIndex, endIndex) {
  const pagination = document.createElement("div");
  pagination.className = "file-pagination";

  const previous = document.createElement("button");
  previous.className = "btn ghost file-page-button";
  previous.type = "button";
  previous.textContent = "Назад";
  previous.disabled = page === 0;
  previous.addEventListener("click", () => renderItemPage(page - 1));

  const label = document.createElement("span");
  label.className = "file-page-label";
  label.textContent = `${startIndex + 1}–${endIndex} из ${state.items.length}`;

  const next = document.createElement("button");
  next.className = "btn ghost file-page-button";
  next.type = "button";
  next.textContent = "Далее";
  next.disabled = page >= pageCount - 1;
  next.addEventListener("click", () => renderItemPage(page + 1));

  pagination.append(previous, label, next);
  return pagination;
}

function renderItemPage(requestedPage) {
  const pageCount = Math.max(1, Math.ceil(state.items.length / CARD_PAGE_SIZE));
  const page = Math.max(0, Math.min(pageCount - 1, requestedPage));
  const startIndex = page * CARD_PAGE_SIZE;
  const endIndex = Math.min(startIndex + CARD_PAGE_SIZE, state.items.length);
  state.itemPage = page;
  state.itemMap = new Map();
  fileList.innerHTML = "";

  const fragment = document.createDocumentFragment();
  for (let index = startIndex; index < endIndex; index += 1) {
    const item = state.items[index];
    const entry = buildFileCard(item, index - startIndex);
    state.itemMap.set(String(item.path || ""), entry);
    fragment.append(entry.card);
  }
  fileList.append(fragment);
  if (pageCount > 1) {
    fileList.append(buildPagination(page, pageCount, startIndex, endIndex));
  }
  fileList.dataset.renderedCount = String(endIndex - startIndex);
  fileList.dataset.totalCount = String(state.items.length);
  fileList.scrollTop = 0;
}

function renderItems(items, modeLabel) {
  state.items = Array.isArray(items) ? items.map((item) => ({ ...item })) : [];
  state.itemMap = new Map();
  state.itemDataMap = new Map(
    state.items.map((item) => [String(item.path || ""), item])
  );
  fileList.innerHTML = "";
  fileList.dataset.renderedCount = "0";
  fileList.dataset.totalCount = String(state.items.length);

  if (modeLabel) {
    state.selectionLabel = String(modeLabel);
  }
  selectedPath.textContent = state.selectionLabel;
  selectedPath.title = state.selectionLabel;

  if (!state.items.length) {
    fileList.classList.add("empty");
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = state.sourcePaths.length > 0
      ? "Поддерживаемые субтитры не найдены."
      : "Список пуст — выберите файл или папку.";
    fileList.append(empty);
    updateInteractionState();
    return;
  }

  fileList.classList.remove("empty");
  renderItemPage(0);
  updateInteractionState();
}

function updateFileState(path, payload) {
  const normalizedPath = String(path || "");
  const item = state.itemDataMap.get(normalizedPath);
  if (item) {
    Object.entries(payload).forEach(([key, value]) => {
      if (value !== undefined) {
        item[key] = value;
      }
    });
    if (payload.status) {
      item.state = payload.status;
    }
  }
  const entry = state.itemMap.get(normalizedPath);
  if (!entry) {
    return;
  }
  applyCardState(entry, item || payload);
}

async function pick(kind) {
  if (state.isBusy || state.isPreparing) {
    return;
  }
  const request = beginItemRequest();
  const settings = readSettings();
  const isFolder = kind === "folder";
  appendLog(
    isFolder
      ? "Открываю системный диалог выбора папки; все подпапки будут включены автоматически."
      : "Открываю системный диалог выбора файлов."
  );
  setPickerHint(isFolder ? "Ожидается выбор папки…" : "Ожидается выбор файлов…");
  setPreparing(true);
  let result;
  try {
    result = await postWithPreparationStatus(
      "/api/pick",
      { kind, settings },
      request,
      "pick",
    );
  } finally {
    completeItemRequest(request.controller);
    if (request.generation === state.itemRequestGeneration) {
      setPreparing(false);
    }
  }
  if (request.generation !== state.itemRequestGeneration || state.isBusy) {
    return;
  }
  if (result.cancelled) {
    appendLog("Выбор отменён.");
    setPickerHint("Выбор отменён", DEFAULT_PICKER_HINT);
    return;
  }
  const items = Array.isArray(result.items) ? result.items : [];
  state.mode = result.mode || kind;
  if (state.mode === "folder" && result.path) {
    state.sourcePaths = [String(result.path)];
  } else if (Array.isArray(result.paths)) {
    state.sourcePaths = result.paths.map(String);
  } else {
    state.sourcePaths = items.map((item) => String(item.path || "")).filter(Boolean);
  }
  const label = state.mode === "folder" && result.path
    ? String(result.path)
    : `Выбрано файлов: ${items.length}`;
  renderItems(items, label);
  if (items.length === 0) {
    appendLog("Поддерживаемые субтитры ASS, SRT или VTT не найдены.");
  } else {
    appendLog(`Список подготовлен: ${items.length} файлов.`);
  }
  setPickerHint(`Список готов: ${items.length} файлов`, DEFAULT_PICKER_HINT);
}

async function runPick(kind) {
  try {
    await pick(kind);
  } catch (err) {
    if (isAbortError(err)) {
      return;
    }
    appendLog(`Ошибка выбора: ${err?.message || "неизвестная ошибка"}`);
    setPickerHint("Ошибка выбора", err?.message || "Неизвестная ошибка");
  }
}

async function refreshCache() {
  if (
    (state.items.length === 0 && state.sourcePaths.length === 0)
    || state.isBusy
    || state.isPreparing
  ) {
    return;
  }
  const request = beginItemRequest();
  const settings = readSettings();
  const recursive = state.mode === "folder";
  const paths = state.sourcePaths.length > 0
    ? [...state.sourcePaths]
    : state.items.map((item) => item.path);
  appendLog(recursive ? "Повторно собираю субтитры из папки и всех подпапок." : "Обновляю выбранные файлы.");
  setPickerHint(recursive ? "Повторный рекурсивный поиск…" : "Обновляется список…");
  setPreparing(true);
  let result;
  try {
    result = await postWithPreparationStatus(
      "/api/refresh",
      { paths, settings, recursive },
      request,
      "refresh",
    );
  } finally {
    completeItemRequest(request.controller);
    if (request.generation === state.itemRequestGeneration) {
      setPreparing(false);
    }
  }
  if (request.generation !== state.itemRequestGeneration || state.isBusy) {
    return;
  }
  const items = Array.isArray(result.items) ? result.items : [];
  if (!recursive && Array.isArray(result.paths)) {
    state.sourcePaths = result.paths.map(String);
  }
  renderItems(items, result.path || state.selectionLabel);
  setPickerHint(`Список готов: ${items.length} файлов`, DEFAULT_PICKER_HINT);
}

async function fetchActiveJobSnapshot() {
  const response = await fetch("/api/active-job", {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const result = await response.json();
  if (!result || typeof result.active !== "boolean") {
    throw new TypeError("Сервер вернул некорректное состояние задачи.");
  }
  return result;
}

async function fetchJobSnapshot(jobId) {
  if (!JOB_ID_PATTERN.test(jobId)) {
    throw new Error("Некорректный идентификатор задачи.");
  }
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const result = await response.json();
  const allowedStatuses = new Set(["running", "ok", "partial", "error"]);
  if (
    !result
    || typeof result !== "object"
    || typeof result.job_id !== "string"
    || typeof result.completed !== "boolean"
    || typeof result.status !== "string"
    || !Array.isArray(result.items)
    || !Array.isArray(result.logs)
  ) {
    throw new TypeError("Сервер вернул некорректную структуру снимка задачи.");
  }
  if (result.job_id !== jobId || !allowedStatuses.has(result.status)) {
    throw new Error("Сервер вернул некорректный снимок задачи.");
  }
  return result;
}

function replaceJobLogs(logs) {
  state.logLines = [];
  logConsole.textContent = "";
  if (Array.isArray(logs)) {
    logs.forEach((line) => appendLog(line));
  }
}

function appendLogOnce(message) {
  if (!state.logLines.includes(message)) {
    appendLog(message);
  }
}

function normalizedJobCounter(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(0, Math.trunc(parsed)) : fallback;
}

function applyJobSnapshot(result) {
  if (!JOB_ID_PATTERN.test(result.job_id)) {
    throw new Error("Сервер вернул некорректный идентификатор активной задачи.");
  }
  if (!Array.isArray(result.items) || !Array.isArray(result.logs)) {
    throw new TypeError("Сервер вернул некорректное содержимое задачи.");
  }
  const items = result.items;
  const preserveFolderContext = (
    state.jobId === result.job_id
    && state.mode === "folder"
    && state.sourcePaths.length === 1
  );
  const selectionLabel = preserveFolderContext
    ? state.selectionLabel
    : `Перевод: ${items.length} файлов`;
  if (!preserveFolderContext) {
    state.mode = "files";
    state.sourcePaths = items.map((item) => String(item.path || "")).filter(Boolean);
  }
  renderItems(items, selectionLabel);
  state.jobId = result.job_id;
  state.completed = new Set();
  items.forEach((item) => {
    if (finalStates.has(item.state)) {
      state.completed.add(item.path);
    }
  });
  state.jobTotal = Math.max(items.length, normalizedJobCounter(result.total, items.length));
  state.jobDone = Math.min(
    state.jobTotal,
    Math.max(state.completed.size, normalizedJobCounter(result.done)),
  );
  replaceJobLogs(result.logs);
  setTranslateBusy(true);
  updateTranslateProgress();
}

async function loadActiveJob(options = {}) {
  const reportError = options.reportError !== false;
  let result;
  try {
    result = await fetchActiveJobSnapshot();
  } catch (error) {
    if (reportError) {
      appendLog(`Ошибка синхронизации: ${error.message}`);
    }
    return { outcome: "error", error };
  }
  if (!result.active) {
    return { outcome: "inactive" };
  }
  try {
    applyJobSnapshot(result);
  } catch (error) {
    if (reportError) {
      appendLog(`Ошибка синхронизации: ${error.message}`);
    }
    return { outcome: "error", error };
  }
  listenJob(result.job_id);
  return { outcome: "active" };
}

function reportRefreshError(error) {
  if (isAbortError(error)) {
    return;
  }
  const message = error?.message || "неизвестная ошибка";
  appendLog(`Ошибка обновления: ${message}`);
  setPickerHint("Ошибка обновления списка", message);
}

async function translate() {
  if (state.items.length === 0 || state.jobId || state.isBusy) {
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
    if (err.status === 409 || String(err.message || "").includes("Перевод уже выполняется")) {
      appendLog("Сервер сообщил о выполняющемся переводе; синхронизирую состояние.");
      await beginActiveJobRecovery("conflict");
      return;
    }
    appendLog(`Ошибка запуска: ${err.message}`);
    const status = Number(err.status);
    if (!Number.isInteger(status) || status >= 500) {
      appendLog("Результат запуска неоднозначен; проверяю активную задачу на сервере.");
      await beginActiveJobRecovery("ambiguous");
      return;
    }
    state.jobId = null;
    setTranslateBusy(false);
  }
}

function clearJobSnapshotTimer() {
  if (state.jobSnapshotTimer !== null) {
    clearTimeout(state.jobSnapshotTimer);
    state.jobSnapshotTimer = null;
  }
}

function clearActiveJobRetryTimer() {
  if (state.activeJobRetryTimer !== null) {
    clearTimeout(state.activeJobRetryTimer);
    state.activeJobRetryTimer = null;
  }
}

function clearJobAuthorityTimer() {
  if (state.jobAuthorityTimer !== null) {
    clearTimeout(state.jobAuthorityTimer);
    state.jobAuthorityTimer = null;
  }
}

function closeJobStream(stream = state.eventSource) {
  if (!stream) {
    return;
  }
  stream.onopen = null;
  stream.onmessage = null;
  stream.onerror = null;
  stream.close();
  if (state.eventSource === stream) {
    state.eventSource = null;
  }
}

function releaseJobLock(message, hint = DEFAULT_PICKER_HINT, hintTitle = "") {
  clearJobSnapshotTimer();
  clearActiveJobRetryTimer();
  clearJobAuthorityTimer();
  closeJobStream();
  state.jobId = null;
  state.jobSnapshotErrorShown = false;
  state.activeJobSyncErrorShown = false;
  state.activeJobRecoveryContext = null;
  state.jobAuthorityErrorShown = false;
  state.jobUnlockContext = null;
  if (message) {
    appendLogOnce(message);
  }
  setPickerHint(hint, hintTitle);
  setTranslateBusy(false);
}

function completedJobMessage(status) {
  if (status === "ok") {
    return "Перевод завершён.";
  }
  if (status === "partial") {
    return "Перевод завершён с ошибками.";
  }
  return "Перевод завершился с ошибкой.";
}

function beginJobUnlock(jobId, message, hint = DEFAULT_PICKER_HINT, hintTitle = "") {
  state.jobUnlockContext = { jobId, message, hint, hintTitle };
  scheduleJobAuthorityReconcile(jobId, 0);
}

function applyFileJobEvent(payload) {
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
}

function applyJobEvent(stream, payload, jobId) {
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
    applyFileJobEvent(payload);
    return;
  }
  if (payload.type === "done") {
    closeJobStream(stream);
    appendLogOnce("Получен итог задачи; загружаю окончательный снимок.");
    setPickerHint("Синхронизируется итог задачи", "Кнопки останутся заблокированными до итогового снимка.");
    scheduleJobSnapshotSync(jobId, 0, "terminal");
  }
}

function handleJobStreamMessage(stream, event, jobId) {
  if (state.eventSource !== stream || !event.data) {
    return;
  }
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch {
    appendLog("Получено повреждённое событие задачи.");
    return;
  }
  applyJobEvent(stream, payload, jobId);
}

function scheduleJobSnapshotSync(jobId, delay = JOB_SYNC_RETRY_DELAY_MS, reason = "disconnect") {
  if (
    state.jobSnapshotTimer !== null
    || state.jobSnapshotInProgress
    || !state.isBusy
    || state.jobId !== jobId
  ) {
    return;
  }
  state.jobSnapshotReason = reason;
  state.jobSnapshotTimer = setTimeout(() => {
    state.jobSnapshotTimer = null;
    synchronizeJobSnapshot(jobId);
  }, delay);
}

function reportJobSnapshotFailure(jobId, error) {
  if (!state.isBusy || state.jobId !== jobId) {
    return;
  }
  if (!state.jobSnapshotErrorShown) {
    appendLog(`Не удалось получить снимок задачи: ${error.message}. Повторяю проверку.`);
    state.jobSnapshotErrorShown = true;
  }
  scheduleJobSnapshotSync(jobId, JOB_SYNC_RETRY_DELAY_MS, state.jobSnapshotReason);
}

function reportRunningJobSnapshot() {
  if (state.jobSnapshotReason === "terminal") {
    appendLogOnce("Сервер ещё формирует итоговый снимок задачи; повторяю проверку.");
    setPickerHint("Синхронизируется итог задачи", "Кнопки останутся заблокированными до итогового снимка.");
    return;
  }
  appendLogOnce("Поток событий оборвался. Состояние задачи восстановлено по снимку.");
  setPickerHint("Связь с задачей восстанавливается", "Состояние обновляется по снимкам сервера.");
}

async function synchronizeJobSnapshot(jobId) {
  if (state.jobSnapshotInProgress || !state.isBusy || state.jobId !== jobId) {
    return;
  }
  state.jobSnapshotInProgress = true;
  let result;
  try {
    result = await fetchJobSnapshot(jobId);
  } catch (error) {
    state.jobSnapshotInProgress = false;
    reportJobSnapshotFailure(jobId, error);
    return;
  }
  state.jobSnapshotInProgress = false;
  if (!state.isBusy || state.jobId !== jobId) {
    return;
  }
  if (result === null) {
    beginJobUnlock(
      jobId,
      "Снимок задачи недоступен или уже удалён; интерфейс разблокирован.",
      "Состояние задачи недоступно",
      "Повторный запуск безопасен после проверки выходных файлов.",
    );
    return;
  }
  try {
    applyJobSnapshot(result);
  } catch (error) {
    reportJobSnapshotFailure(jobId, error);
    return;
  }
  state.jobSnapshotErrorShown = false;
  if (result.completed) {
    beginJobUnlock(jobId, completedJobMessage(result.status));
    return;
  }
  reportRunningJobSnapshot();
  scheduleJobSnapshotSync(jobId, JOB_SYNC_RETRY_DELAY_MS, state.jobSnapshotReason);
}

function handleJobStreamError(stream, jobId) {
  if (state.eventSource !== stream || state.jobId !== jobId) {
    return;
  }
  closeJobStream(stream);
  appendLog("Поток событий оборвался; проверяю состояние задачи на сервере.");
  setPickerHint("Связь с задачей прервана", "Интерфейс остаётся заблокированным до синхронизации.");
  setTranslateBusy(true);
  updateTranslateProgress();
  scheduleJobSnapshotSync(jobId);
}

function scheduleJobAuthorityReconcile(jobId, delay = JOB_SYNC_RETRY_DELAY_MS) {
  if (
    state.jobAuthorityTimer !== null
    || state.jobAuthorityInProgress
    || !state.isBusy
    || state.jobId !== jobId
    || state.jobUnlockContext?.jobId !== jobId
  ) {
    return;
  }
  state.jobAuthorityTimer = setTimeout(() => {
    state.jobAuthorityTimer = null;
    reconcileJobAuthority(jobId);
  }, delay);
}

function reportJobAuthorityFailure(jobId, error) {
  if (!state.isBusy || state.jobId !== jobId || state.jobUnlockContext?.jobId !== jobId) {
    return;
  }
  if (!state.jobAuthorityErrorShown) {
    appendLog(`Не удалось проверить активную задачу: ${error.message}. Повторяю проверку.`);
    setPickerHint("Проверяется активная задача", "Интерфейс остаётся заблокированным до ответа сервера.");
    state.jobAuthorityErrorShown = true;
  }
  scheduleJobAuthorityReconcile(jobId);
}

function followAuthoritativeJob(result, previousJobId) {
  applyJobSnapshot(result);
  state.jobUnlockContext = null;
  state.jobAuthorityErrorShown = false;
  appendLogOnce(`Обнаружена новая активная задача ${result.job_id}; продолжаю синхронизацию.`);
  setPickerHint("Переключено на активную задачу", `Задача ${previousJobId} завершена; отслеживается ${result.job_id}.`);
  if (result.completed) {
    beginJobUnlock(result.job_id, completedJobMessage(result.status));
    return;
  }
  listenJob(result.job_id);
}

async function reconcileJobAuthority(jobId) {
  if (state.jobAuthorityInProgress || !state.isBusy || state.jobId !== jobId) {
    return;
  }
  state.jobAuthorityInProgress = true;
  let result;
  try {
    result = await fetchActiveJobSnapshot();
  } catch (error) {
    state.jobAuthorityInProgress = false;
    reportJobAuthorityFailure(jobId, error);
    return;
  }
  state.jobAuthorityInProgress = false;
  if (!state.isBusy || state.jobId !== jobId || state.jobUnlockContext?.jobId !== jobId) {
    return;
  }
  if (!result.active) {
    const context = state.jobUnlockContext;
    releaseJobLock(context.message, context.hint, context.hintTitle);
    return;
  }
  if (result.job_id === jobId) {
    appendLogOnce("Завершённая задача ещё отмечена активной; повторяю проверку.");
    scheduleJobAuthorityReconcile(jobId);
    return;
  }
  try {
    followAuthoritativeJob(result, jobId);
  } catch (error) {
    reportJobAuthorityFailure(jobId, error);
  }
}

function scheduleActiveJobRetry() {
  if (
    state.activeJobRetryTimer !== null
    || state.activeJobRetryInProgress
    || !state.isBusy
    || state.jobId
  ) {
    return;
  }
  state.activeJobRetryTimer = setTimeout(() => {
    state.activeJobRetryTimer = null;
    recoverConflictingJob();
  }, JOB_SYNC_RETRY_DELAY_MS);
}

function activeJobRecoveryErrorPrefix(context) {
  if (context === "startup") {
    return "Не удалось проверить активную задачу при запуске";
  }
  if (context === "ambiguous") {
    return "Не удалось проверить результат запуска";
  }
  return "Не удалось синхронизировать конфликт запуска";
}

function inactiveRecoveryMessage(context) {
  if (context === "startup") {
    return "";
  }
  if (context === "ambiguous") {
    return "Активная задача после неоднозначного ответа запуска не найдена; интерфейс разблокирован.";
  }
  return "Активная задача после проверки конфликта запуска не найдена; интерфейс разблокирован.";
}

async function beginActiveJobRecovery(context) {
  clearActiveJobRetryTimer();
  state.activeJobRecoveryContext = context;
  state.activeJobSyncErrorShown = false;
  setTranslateBusy(true);
  setPickerHint("Проверяется активная задача", "Интерфейс остаётся заблокированным до ответа сервера.");
  await recoverConflictingJob();
}

async function recoverConflictingJob() {
  if (state.activeJobRetryInProgress || !state.isBusy || state.jobId) {
    return;
  }
  state.activeJobRetryInProgress = true;
  const result = await loadActiveJob({ reportError: false });
  state.activeJobRetryInProgress = false;
  if (result.outcome === "active") {
    state.activeJobRecoveryContext = null;
    state.activeJobSyncErrorShown = false;
    return;
  }
  if (!state.isBusy || state.jobId) {
    return;
  }
  const context = state.activeJobRecoveryContext || "conflict";
  if (result.outcome === "inactive") {
    releaseJobLock(
      inactiveRecoveryMessage(context),
      context === "startup" ? DEFAULT_PICKER_HINT : "Активная задача не найдена",
      context === "startup" ? "" : "Можно безопасно повторить запуск.",
    );
    return;
  }
  if (!state.activeJobSyncErrorShown) {
    appendLog(`${activeJobRecoveryErrorPrefix(context)}: ${result.error.message}. Повторяю проверку.`);
    setPickerHint("Проверяется активная задача", "Интерфейс остаётся заблокированным до ответа сервера.");
    state.activeJobSyncErrorShown = true;
  }
  scheduleActiveJobRetry();
}

function listenJob(jobId) {
  if (!JOB_ID_PATTERN.test(jobId)) {
    releaseJobLock(
      "Сервер вернул некорректный идентификатор задачи; восстановление невозможно.",
      "Некорректный идентификатор задачи",
    );
    return;
  }
  clearJobSnapshotTimer();
  clearJobAuthorityTimer();
  closeJobStream();
  const streamUrl = new URL(`/api/stream/${encodeURIComponent(jobId)}`, window.location.origin);
  const stream = new EventSource(streamUrl);
  state.eventSource = stream;
  stream.onmessage = (event) => handleJobStreamMessage(stream, event, jobId);
  stream.onerror = () => handleJobStreamError(stream, jobId);
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
  });
}

if (refreshBtn) {
  refreshBtn.addEventListener("click", () => {
    refreshCache().catch(reportRefreshError);
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
  state.logLines = [];
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
});

function clientErrorMessage(value) {
  if (value instanceof Error && value.message) {
    return value.message;
  }
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  return "причина не указана";
}

window.addEventListener("error", (event) => {
  const message = clientErrorMessage(event.error || event.message);
  appendLog(`Необработанная ошибка интерфейса: ${message}`);
  setPickerHint("Ошибка интерфейса", message);
});

window.addEventListener("unhandledrejection", (event) => {
  const message = clientErrorMessage(event.reason);
  appendLog(`Необработанная ошибка операции: ${message}`);
  setPickerHint("Ошибка операции", message);
});

window.addEventListener("beforeunload", () => {
  abortItemRequest();
  clearJobSnapshotTimer();
  clearActiveJobRetryTimer();
  clearJobAuthorityTimer();
  closeJobStream();
});

setTranslateBusy(false);

const uiConfigPromise = loadUiConfig().catch((err) => {
  appendLog(`Ошибка загрузки настроек UI: ${err.message}`);
  updateApiDependentUi();
});
await beginActiveJobRecovery("startup");
await uiConfigPromise;
