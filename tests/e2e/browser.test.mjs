import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright-core";

const testDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(testDirectory, "../..");
const windowsPython = path.join(repositoryRoot, ".venv", "Scripts", "python.exe");
const posixPython = path.join(repositoryRoot, ".venv", "bin", "python");
const browserCandidates = [
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].filter(Boolean);

let serverProcess;
let browser;
let baseUrl;
let serverOutput = "";

function findExecutable(candidates) {
  return candidates.find((candidate) => existsSync(candidate));
}

function findPython() {
  const configured = process.env.SUB_TRANSLATOR_PYTHON;
  if (configured) {
    return configured;
  }
  if (existsSync(windowsPython)) {
    return windowsPython;
  }
  if (existsSync(posixPython)) {
    return posixPython;
  }
  return "python";
}

async function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close((error) => error ? reject(error) : resolve(port));
    });
  });
}

async function waitForServer(url) {
  for (let attempt = 0; attempt < 300; attempt += 1) {
    try {
      const response = await fetch(`${url}/api/health`);
      if (response.ok) {
        return;
      }
    } catch {
      // Сервер ещё запускается.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Веб-сервер не запустился. ${serverOutput}`);
}

before(async () => {
  const executablePath = findExecutable(browserCandidates);
  assert.ok(executablePath, "Не найден Chrome или Edge для браузерной проверки.");
  const port = await findFreePort();
  baseUrl = `http://127.0.0.1:${port}`;
  serverProcess = spawn(
    findPython(),
    [
      "-B",
      "-m",
      "uvicorn",
      "sub_translate.web.app:app",
      "--host",
      "127.0.0.1",
      "--port",
      String(port),
    ],
    {
      cwd: repositoryRoot,
      env: process.env,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  serverProcess.stdout.on("data", (chunk) => {
    serverOutput += chunk.toString();
  });
  serverProcess.stderr.on("data", (chunk) => {
    serverOutput += chunk.toString();
  });
  await waitForServer(baseUrl);
  browser = await chromium.launch({ executablePath, headless: true });
});

after(async () => {
  if (browser) {
    await browser.close();
  }
  if (serverProcess && serverProcess.exitCode === null) {
    serverProcess.kill();
  }
});

for (const scenario of [
  { name: "ошибка ui-config не мешает восстановить активную задачу", configError: true, activeFailures: 0, jobChar: "3" },
  { name: "ошибка active-job на старте сохраняет блокировку и повторяет запрос", configError: false, activeFailures: 1, jobChar: "4" },
]) {
  test(scenario.name, async () => {
    const page = await browser.newPage();
    const jobId = scenario.jobChar.repeat(32);
    const item = {
      name: `стартовая задача ${scenario.jobChar}.srt`,
      path: `D:\\Тесты\\стартовая задача ${scenario.jobChar}.srt`,
      format: "srt",
      cached: false,
      output: `D:\\Тесты\\стартовая задача ${scenario.jobChar}.google.ru.srt`,
    };
    let activeJobCalls = 0;

    if (scenario.configError) {
      await page.route("**/api/ui-config", async (route) => {
        await new Promise((resolve) => setTimeout(resolve, 200));
        await route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Временная ошибка конфигурации." }),
        });
      });
    }
    await page.route("**/api/active-job", (route) => {
      activeJobCalls += 1;
      if (activeJobCalls <= scenario.activeFailures) {
        route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Временная ошибка состояния." }),
        });
        return;
      }
      if (activeJobCalls === scenario.activeFailures + 1) {
        route.fulfill({
          json: {
            active: true,
            job_id: jobId,
            completed: false,
            status: "running",
            items: [{ ...item, state: "started", progress: 30 }],
            logs: ["Активная задача восстановлена при загрузке страницы."],
            total: 1,
            done: 0,
          },
        });
        return;
      }
      route.fulfill({ json: { active: false } });
    });
    await page.route("**/api/stream/**", async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 900));
      const events = [
        { type: "file", path: item.path, state: "done", progress: 100, output: item.output },
        { type: "done", status: "ok" },
      ];
      await route.fulfill({
        status: 200,
        headers: {
          "Cache-Control": "no-cache",
          "Content-Type": "text/event-stream",
        },
        body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
      });
    });
    await page.route("**/api/jobs/**", (route) => route.fulfill({
      json: {
        job_id: jobId,
        completed: true,
        status: "ok",
        items: [{ ...item, state: "done", progress: 100 }],
        logs: ["Финальный снимок стартовой задачи."],
        total: 1,
        done: 1,
      },
    }));

    await page.goto(baseUrl);
    if (scenario.activeFailures > 0) {
      await page.waitForFunction(() => (
        document.querySelector("#logConsole")?.textContent.includes("Не удалось проверить активную задачу при запуске")
      ));
      assert.equal(await page.locator("#pickerFileBtn").isDisabled(), true);
      assert.equal(await page.locator("#unloadBtn").isDisabled(), true);
    }
    await page.waitForFunction(() => document.querySelector(".file-status")?.textContent.includes("30%"));
    if (scenario.configError) {
      await page.waitForFunction(() => (
        document.querySelector("#logConsole")?.textContent.includes("Ошибка загрузки настроек UI")
      ));
    }
    assert.equal(await page.locator("#translateBtn").isDisabled(), true);
    assert.equal(await page.locator("#refreshBtn").isDisabled(), true);
    assert.equal(await page.locator("#pickerFileBtn").isDisabled(), true);
    assert.equal(await page.locator("#unloadBtn").isDisabled(), true);

    await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("Перевод завершён."));
    assert.equal(activeJobCalls, scenario.activeFailures + 2);
    assert.equal(await page.locator("#translateBtn").isDisabled(), false);
    assert.equal(await page.locator("#pickerFileBtn").isDisabled(), false);
    assert.equal(await page.locator(".file-card").getAttribute("data-status"), "done");
    await page.close();
  });
}

for (const failure of [
  { name: "потерянный ответ запуска", mode: "network", jobChar: "5" },
  { name: "ошибка 500 после принятия запуска", mode: "server", jobChar: "6" },
]) {
  test(`${failure.name} восстанавливает принятую задачу`, async () => {
    const page = await browser.newPage();
    const jobId = failure.jobChar.repeat(32);
    const item = {
      name: `неоднозначный запуск ${failure.jobChar}.srt`,
      path: `D:\\Тесты\\неоднозначный запуск ${failure.jobChar}.srt`,
      format: "srt",
      cached: false,
      output: `D:\\Тесты\\неоднозначный запуск ${failure.jobChar}.google.ru.srt`,
    };
    let activeJobCalls = 0;
    let streamCalls = 0;

    await page.route("**/api/active-job", (route) => {
      activeJobCalls += 1;
      if (activeJobCalls === 1 || activeJobCalls >= 3) {
        route.fulfill({ json: { active: false } });
        return;
      }
      route.fulfill({
        json: {
          active: true,
          job_id: jobId,
          completed: false,
          status: "running",
          items: [{ ...item, state: "started", progress: 35 }],
          logs: ["Принятая задача найдена после неоднозначного ответа."],
          total: 1,
          done: 0,
        },
      });
    });
    await page.route("**/api/pick", (route) => route.fulfill({
      json: { mode: "files", paths: [item.path], items: [item], cancelled: false, recursive: false },
    }));
    await page.route("**/api/translate", (route) => {
      if (failure.mode === "network") {
        route.abort("connectionreset");
        return;
      }
      route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Ответ потерян после принятия задачи." }),
      });
    });
    await page.route("**/api/stream/**", async (route) => {
      streamCalls += 1;
      await new Promise((resolve) => setTimeout(resolve, 800));
      const events = [
        { type: "file", path: item.path, state: "done", progress: 100, output: item.output },
        { type: "done", status: "ok" },
      ];
      await route.fulfill({
        status: 200,
        headers: {
          "Cache-Control": "no-cache",
          "Content-Type": "text/event-stream",
        },
        body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
      });
    });
    await page.route("**/api/jobs/**", (route) => route.fulfill({
      json: {
        job_id: jobId,
        completed: true,
        status: "ok",
        items: [{ ...item, state: "done", progress: 100 }],
        logs: ["Финальный снимок восстановленной задачи."],
        total: 1,
        done: 1,
      },
    }));

    await page.goto(baseUrl);
    await page.waitForFunction(() => document.querySelector("#sourceLang")?.options.length > 0);
    await page.click("#pickerFileBtn");
    await page.waitForFunction(() => document.querySelectorAll(".file-card").length === 1);
    await page.click("#translateBtn");
    await page.waitForFunction(() => (
      document.querySelector("#logConsole")?.textContent.includes("Принятая задача найдена после неоднозначного ответа")
    ));
    await page.waitForFunction(() => document.querySelector(".file-status")?.textContent.includes("35%"));
    assert.equal(activeJobCalls, 2);
    assert.equal(streamCalls, 1);
    assert.equal(await page.locator("#translateBtn").isDisabled(), true);
    assert.equal(await page.locator("#pickerFileBtn").isDisabled(), true);
    assert.equal(await page.locator("#unloadBtn").isDisabled(), true);

    await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("Перевод завершён."));
    assert.equal(activeJobCalls, 3);
    assert.equal(await page.locator("#translateBtn").isDisabled(), false);
    assert.equal(await page.locator("#pickerFileBtn").isDisabled(), false);
    assert.equal(await page.locator(".file-card").getAttribute("data-status"), "done");
    await page.close();
  });
}

test("выбор, прогресс, ошибка, кеш и выгрузка модели", async () => {
  const page = await browser.newPage();
  const jobId = "b".repeat(32);
  const items = [
    {
      name: "эпизод 1.srt",
      path: "D:\\Тесты\\эпизод 1.srt",
      format: "srt",
      cached: false,
      output: "D:\\Тесты\\эпизод 1.ru.srt",
    },
    {
      name: "broken.vtt",
      path: "D:\\Тесты\\broken.vtt",
      format: "vtt",
      cached: false,
      output: "D:\\Тесты\\broken.ru.vtt",
    },
  ];
  let translateAttempts = 0;

  await page.route("**/api/active-job", (route) => route.fulfill({ json: { active: false } }));
  await page.route("**/api/pick", (route) => route.fulfill({
    json: { mode: "files", items },
  }));
  await page.route("**/api/refresh", (route) => route.fulfill({
    json: { items: [{ ...items[0], cached: true }, items[1]] },
  }));
  await page.route("**/api/unload", (route) => route.fulfill({
    json: { status: "ok", message: "Модели выгружены." },
  }));
  await page.route("**/api/translate", (route) => {
    translateAttempts += 1;
    if (translateAttempts > 1) {
      return route.fulfill({
        status: 400,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Тестовая ошибка запуска." }),
      });
    }
    return route.fulfill({ json: { job_id: jobId } });
  });
  await page.route("**/api/stream/**", (route) => {
    assert.match(route.request().url(), new RegExp(`/api/stream/${jobId}$`));
    const events = [
      { type: "job", total: 2 },
      { type: "file", path: items[0].path, state: "started", progress: 50 },
      { type: "file", path: items[0].path, state: "done", progress: 100, output: items[0].output },
      { type: "file", path: items[1].path, state: "error", progress: 100, error: "Ошибка файла." },
      { type: "done", status: "partial" },
    ];
    const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
    return route.fulfill({
      status: 200,
      headers: {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
      },
      body,
    });
  });
  await page.route("**/api/jobs/**", (route) => {
    assert.equal(new URL(route.request().url()).pathname, `/api/jobs/${jobId}`);
    return route.fulfill({
      json: {
        job_id: jobId,
        completed: true,
        status: "partial",
        items: [
          { ...items[0], state: "done", progress: 100 },
          { ...items[1], state: "error", progress: 100, error: "Ошибка файла." },
        ],
        logs: ["Итоговый снимок перевода получен."],
        total: 2,
        done: 2,
      },
    });
  });

  await page.goto(baseUrl);
  await page.waitForFunction(() => document.querySelector("#sourceLang")?.options.length > 0);
  const translatorIds = await page.locator("#translatorApi option").evaluateAll((options) =>
    options.map((option) => option.value),
  );
  assert.deepEqual(translatorIds, [
    "google", "agent", "nllb-600m", "translategemma", "translategemma-12b", "seedx",
  ]);
  await page.selectOption("#translatorApi", "translategemma");
  assert.equal(await page.locator("#threadCount").isDisabled(), true);
  assert.match(await page.locator("#localModelMeta").innerText(), /google\/translategemma-4b-it/);
  assert.equal(await page.locator("#modelPath").isVisible(), true);
  assert.equal(await page.locator("#modelRevision").isVisible(), true);
  assert.equal(await page.locator("#workerPythonPath").isVisible(), true);
  await page.selectOption("#translatorApi", "translategemma-12b");
  assert.match(await page.locator("#localModelMeta").innerText(), /google\/translategemma-12b-it/);
  assert.match(await page.locator("#localModelMeta").innerText(), /bitsandbytes-nf4-double/);
  assert.equal(await page.locator("#requestTimeout").inputValue(), "3600");
  assert.equal(await page.locator("#allowCpuFallback").isDisabled(), true);
  await page.selectOption("#translatorApi", "seedx");
  assert.match(await page.locator("#localModelMeta").innerText(), /ByteDance-Seed\/Seed-X-PPO-7B/);
  assert.equal(await page.locator("#requestTimeout").inputValue(), "3600");
  assert.equal(await page.locator("#autoDownloadModel").isChecked(), false);
  assert.equal(await page.locator("#allowCpuFallback").isDisabled(), true);
  await page.selectOption("#translatorApi", "nllb-600m");
  assert.equal(await page.locator("#allowCpuFallback").isDisabled(), false);
  await page.selectOption("#translatorApi", "seedx");
  assert.deepEqual(
    await page.locator("#targetLang option").evaluateAll((options) => options.map((option) => option.value)),
    ["ru"],
  );
  await page.selectOption("#sourceLang", "ru");
  assert.deepEqual(
    await page.locator("#targetLang option").evaluateAll((options) => options.map((option) => option.value)),
    ["en"],
  );
  await page.evaluate(() => {
    localStorage.setItem(
      "subTranslateSettings",
      JSON.stringify({api: "seedx", allow_cpu_fallback: true}),
    );
  });
  await page.reload();
  await page.waitForFunction(() => document.querySelector("#translatorApi")?.value === "seedx");
  assert.equal(await page.locator("#allowCpuFallback").isDisabled(), true);
  assert.equal(await page.locator("#allowCpuFallback").isChecked(), false);
  await page.evaluate(() => {
    localStorage.setItem(
      "subTranslateSettings",
      JSON.stringify({
        api: "google",
        openai_api_key: "legacy-secret",
        agent_system_prompt: "private system prompt",
        agent_prompt: "private user prompt",
        agent_system_prompt_file: "C:\\private-system.txt",
        agent_prompt_file: "C:\\private-user.txt",
      }),
    );
  });
  await page.reload();
  await page.waitForFunction(() => document.querySelector("#sourceLang")?.options.length > 0);
  const stored = await page.evaluate(() => localStorage.getItem("subTranslateSettings"));
  assert.equal(stored.includes("openai_api_key"), false);
  assert.equal(stored.includes("legacy-secret"), false);
  assert.equal(stored.includes("agent_system_prompt"), false);
  assert.equal(stored.includes("agent_prompt"), false);
  assert.equal(stored.includes("private"), false);

  await page.click("#pickerFileBtn");
  await page.waitForFunction(() => document.querySelectorAll(".file-card").length === 2);
  assert.match(await page.locator("#fileList").innerText(), /эпизод 1\.srt/);

  await page.click("#translateBtn");
  await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("с ошибками"));
  const statuses = await page.locator(".file-card").evaluateAll((cards) =>
    cards.map((card) => card.dataset.status),
  );
  assert.deepEqual(statuses, ["done", "error"]);

  await page.selectOption("#translatorApi", "agent");
  await page.click("#refreshBtn");
  await page.waitForFunction(() => document.querySelector(".file-card")?.dataset.status === "cached");
  await page.click("#translateBtn");
  await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("Тестовая ошибка запуска"));

  await page.click("#unloadBtn");
  await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("Модели выгружены"));
  assert.equal(await page.locator("#openAiKeyStatus").count(), 1);
  assert.equal(await page.locator('[data-setting="openai_api_key"]').count(), 0);
  await page.close();
});

test("разрыв SSE сохраняет блокировку и восстанавливает задачу по снимкам", async () => {
  const page = await browser.newPage();
  const jobId = "c".repeat(32);
  const item = {
    name: "активная серия.srt",
    path: "D:\\Тесты\\активная серия.srt",
    format: "srt",
    cached: false,
    output: "D:\\Тесты\\активная серия.google.ru.srt",
  };
  let activeJobCalls = 0;
  let streamCalls = 0;
  let jobSnapshotCalls = 0;

  await page.route("**/api/active-job", (route) => {
    activeJobCalls += 1;
    route.fulfill({ json: { active: false } });
  });
  await page.route("**/api/pick", (route) => route.fulfill({
    json: { mode: "files", paths: [item.path], items: [item], cancelled: false, recursive: false },
  }));
  await page.route("**/api/translate", (route) => {
    route.fulfill({ json: { job_id: jobId } });
  });
  await page.route("**/api/stream/**", (route) => {
    streamCalls += 1;
    assert.equal(new URL(route.request().url()).pathname, `/api/stream/${jobId}`);
    const events = [
      { type: "job", total: 1 },
      { type: "file", path: item.path, state: "started", progress: 25 },
    ];
    return route.fulfill({
      status: 200,
      headers: {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
      },
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });
  await page.route("**/api/jobs/**", async (route) => {
    jobSnapshotCalls += 1;
    assert.equal(new URL(route.request().url()).pathname, `/api/jobs/${jobId}`);
    if (jobSnapshotCalls === 1) {
      await route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Временная ошибка снимка." }),
      });
      return;
    }
    if (jobSnapshotCalls === 2) {
      await route.fulfill({
        json: {
          job_id: jobId,
          completed: false,
          status: "running",
          items: [{ ...item, state: "started", progress: 60 }],
          logs: ["Снимок активной задачи после разрыва."],
          total: 1,
          done: 0,
        },
      });
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 800));
    await route.fulfill({
      json: {
        job_id: jobId,
        completed: true,
        status: "ok",
        items: [{ ...item, state: "done", progress: 100, output: item.output }],
        logs: ["Финальный снимок задачи после разрыва."],
        total: 1,
        done: 1,
      },
    });
  });

  await page.goto(baseUrl);
  await page.waitForFunction(() => document.querySelector("#sourceLang")?.options.length > 0);
  await page.click("#pickerFileBtn");
  await page.waitForFunction(() => document.querySelectorAll(".file-card").length === 1);
  await page.click("#translateBtn");
  await page.waitForFunction(() => (
    document.querySelector("#logConsole")?.textContent.includes("Не удалось получить снимок задачи: HTTP 503")
  ));
  assert.equal(await page.locator("#translateBtn").isDisabled(), true);
  assert.equal(await page.locator("#refreshBtn").isDisabled(), true);
  await page.waitForFunction(() => (
    document.querySelector("#logConsole")?.textContent.includes("Состояние задачи восстановлено по снимку")
  ));
  await page.waitForFunction(() => document.querySelector(".file-status")?.textContent.includes("60%"));
  assert.equal(activeJobCalls, 1);
  assert.equal(streamCalls, 1);
  assert.equal(jobSnapshotCalls >= 2, true);
  assert.equal(await page.locator("#translateBtn").isDisabled(), true);
  assert.equal(await page.locator("#refreshBtn").isDisabled(), true);
  assert.equal(await page.locator("#pickerFileBtn").isDisabled(), true);
  assert.equal(await page.locator("#unloadBtn").isDisabled(), true);
  assert.match(await page.locator("#logConsole").innerText(), /Поток событий оборвался/);
  assert.equal(await page.locator(".file-card").getAttribute("data-status"), "started");

  await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("Перевод завершён."));
  assert.equal(streamCalls, 1);
  assert.equal(activeJobCalls, 2);
  assert.equal(jobSnapshotCalls >= 3, true);
  assert.equal(await page.locator("#translateBtn").isDisabled(), false);
  assert.equal(await page.locator("#refreshBtn").isDisabled(), false);
  assert.equal(await page.locator("#pickerFileBtn").isDisabled(), false);
  assert.equal(await page.locator("#unloadBtn").isDisabled(), false);
  assert.equal(await page.locator(".file-card").getAttribute("data-status"), "done");
  const finalLogs = await page.locator("#logConsole").innerText();
  assert.equal(finalLogs.split("Финальный снимок задачи после разрыва.").length - 1, 1);
  await page.close();
});

test("terminal snapshot переключается на новую активную задачу до разблокировки", async () => {
  const page = await browser.newPage();
  const firstJobId = "d".repeat(32);
  const secondJobId = "e".repeat(32);
  const firstItem = {
    name: "первая задача.srt",
    path: "D:\\Тесты\\первая задача.srt",
    format: "srt",
    cached: false,
    output: "D:\\Тесты\\первая задача.google.ru.srt",
  };
  const secondItem = {
    name: "вторая задача.srt",
    path: "D:\\Тесты\\вторая задача.srt",
    format: "srt",
    cached: false,
    output: "D:\\Тесты\\вторая задача.google.ru.srt",
  };
  let activeJobCalls = 0;
  const streamCalls = [];
  const snapshotCalls = [];

  await page.route("**/api/active-job", (route) => {
    activeJobCalls += 1;
    if (activeJobCalls === 2) {
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Временная ошибка активной задачи." }),
      });
      return;
    }
    if (activeJobCalls === 1 || activeJobCalls >= 4) {
      route.fulfill({ json: { active: false } });
      return;
    }
    route.fulfill({
      json: {
        active: true,
        job_id: secondJobId,
        completed: false,
        status: "running",
        items: [{ ...secondItem, state: "started", progress: 40 }],
        logs: ["Вторая задача уже запущена в другой вкладке."],
        total: 1,
        done: 0,
      },
    });
  });
  await page.route("**/api/pick", (route) => route.fulfill({
    json: {
      mode: "files",
      paths: [firstItem.path],
      items: [firstItem],
      cancelled: false,
      recursive: false,
    },
  }));
  await page.route("**/api/translate", (route) => route.fulfill({ json: { job_id: firstJobId } }));
  await page.route("**/api/jobs/**", (route) => {
    const jobId = new URL(route.request().url()).pathname.split("/").at(-1);
    snapshotCalls.push(jobId);
    const item = jobId === firstJobId ? firstItem : secondItem;
    return route.fulfill({
      json: {
        job_id: jobId,
        completed: true,
        status: "ok",
        items: [{ ...item, state: "done", progress: 100, output: item.output }],
        logs: [`Финальный снимок ${jobId === firstJobId ? "первой" : "второй"} задачи.`],
        total: 1,
        done: 1,
      },
    });
  });
  await page.route("**/api/stream/**", async (route) => {
    const jobId = new URL(route.request().url()).pathname.split("/").at(-1);
    streamCalls.push(jobId);
    if (jobId === secondJobId) {
      await new Promise((resolve) => setTimeout(resolve, 800));
    }
    const item = jobId === firstJobId ? firstItem : secondItem;
    const events = [
      { type: "job", total: 1 },
      { type: "file", path: item.path, state: "done", progress: 100, output: item.output },
      { type: "done", status: "ok" },
    ];
    return route.fulfill({
      status: 200,
      headers: {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
      },
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });

  await page.goto(baseUrl);
  await page.waitForFunction(() => document.querySelector("#sourceLang")?.options.length > 0);
  await page.click("#pickerFileBtn");
  await page.waitForFunction(() => document.querySelectorAll(".file-card").length === 1);
  await page.click("#translateBtn");
  await page.waitForFunction(() => (
    document.querySelector("#logConsole")?.textContent.includes("Не удалось проверить активную задачу: HTTP 503")
  ));
  assert.equal(await page.locator("#translateBtn").isDisabled(), true);
  assert.equal(await page.locator("#pickerFileBtn").isDisabled(), true);
  await page.waitForFunction(() => (
    document.querySelector("#logConsole")?.textContent.includes("Обнаружена новая активная задача")
  ));
  await page.waitForFunction(() => document.querySelector(".file-status")?.textContent.includes("40%"));

  assert.deepEqual(streamCalls, [firstJobId, secondJobId]);
  assert.deepEqual(snapshotCalls, [firstJobId]);
  assert.equal(activeJobCalls, 3);
  assert.equal(await page.locator("#translateBtn").isDisabled(), true);
  assert.equal(await page.locator("#refreshBtn").isDisabled(), true);
  assert.equal(await page.locator("#pickerFileBtn").isDisabled(), true);
  assert.equal(await page.locator("#unloadBtn").isDisabled(), true);
  assert.match(await page.locator("#fileList").innerText(), /вторая задача\.srt/);

  await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("Перевод завершён."));
  assert.deepEqual(snapshotCalls, [firstJobId, secondJobId]);
  assert.equal(activeJobCalls, 4);
  assert.equal(await page.locator("#translateBtn").isDisabled(), false);
  assert.equal(await page.locator("#refreshBtn").isDisabled(), false);
  assert.equal(await page.locator("#pickerFileBtn").isDisabled(), false);
  assert.equal(await page.locator("#unloadBtn").isDisabled(), false);
  assert.equal(await page.locator(".file-card").getAttribute("data-status"), "done");
  const finalLogs = await page.locator("#logConsole").innerText();
  assert.equal(finalLogs.split("Финальный снимок второй задачи.").length - 1, 1);
  await page.close();
});

test("конфликт запуска без активной задачи безопасно снимает блокировку", async () => {
  const page = await browser.newPage();
  const item = {
    name: "конфликт.srt",
    path: "D:\\Тесты\\конфликт.srt",
    format: "srt",
    cached: false,
    output: "D:\\Тесты\\конфликт.google.ru.srt",
  };
  let activeJobCalls = 0;
  let streamCalls = 0;

  await page.route("**/api/active-job", (route) => {
    activeJobCalls += 1;
    route.fulfill({ json: { active: false } });
  });
  await page.route("**/api/pick", (route) => route.fulfill({
    json: { mode: "files", paths: [item.path], items: [item], cancelled: false, recursive: false },
  }));
  await page.route("**/api/translate", (route) => route.fulfill({
    status: 409,
    contentType: "application/json",
    body: JSON.stringify({ detail: "Перевод уже выполняется." }),
  }));
  await page.route("**/api/stream/**", (route) => {
    streamCalls += 1;
    route.abort();
  });

  await page.goto(baseUrl);
  await page.waitForFunction(() => document.querySelector("#sourceLang")?.options.length > 0);
  await page.click("#pickerFileBtn");
  await page.waitForFunction(() => document.querySelectorAll(".file-card").length === 1);
  await page.click("#translateBtn");
  await page.waitForFunction(() => (
    document.querySelector("#logConsole")?.textContent.includes("Активная задача после проверки конфликта запуска не найдена")
  ));

  assert.equal(activeJobCalls, 2);
  assert.equal(streamCalls, 0);
  assert.equal(await page.locator("#translateBtn").isDisabled(), false);
  assert.equal(await page.locator("#refreshBtn").isDisabled(), false);
  assert.equal(await page.locator("#pickerFileBtn").isDisabled(), false);
  assert.equal(await page.locator("#unloadBtn").isDisabled(), false);
  await page.close();
});

test("рекурсивная папка показывает подготовку и сохраняет корень", async () => {
  const page = await browser.newPage();
  const rootPath = "D:\\Субтитры\\Курс";
  const emptyRootPath = "D:\\Субтитры\\Пустая папка";
  const jobId = "f".repeat(32);
  const items = Array.from({ length: 10000 }, (_unused, index) => {
    const number = String(index + 1).padStart(5, "0");
    const folder = index % 2 === 0 ? "Верхний уровень" : "Вложенная папка\\Ещё глубже";
    const source = `${rootPath}\\${folder}\\Урок ${number}.srt`;
    return {
      name: `Урок ${number}.srt`,
      path: source,
      format: "srt",
      cached: false,
      output: source.replace(/\.srt$/, ".google.ru.srt"),
    };
  });
  const newNestedItem = {
    name: "000 Новый вложенный урок.srt",
    path: `${rootPath}\\Новая подпапка\\000 Новый вложенный урок.srt`,
    format: "srt",
    cached: false,
    output: `${rootPath}\\Новая подпапка\\000 Новый вложенный урок.google.ru.srt`,
  };
  const refreshedItems = [newNestedItem, ...items];
  let pickCalls = 0;
  let firstPickPending = false;
  let activePreparationStatusRequests = 0;
  const refreshPayloads = [];

  await page.route("**/api/active-job", (route) => route.fulfill({ json: { active: false } }));
  await page.route("**/api/preparation-status", (route) => {
    const generation = pickCalls;
    const active = firstPickPending;
    if (active) {
      activePreparationStatusRequests += 1;
    }
    const collecting = active && activePreparationStatusRequests === 1;
    const queueingProgress = active ? Math.min(2, activePreparationStatusRequests - 1) : 0;
    route.fulfill({
      json: {
        generation,
        operation_id: generation > 0 ? `pick-${generation}` : null,
        operation: generation > 0 ? "pick" : null,
        status: active ? "running" : generation > 0 ? "done" : "idle",
        phase: collecting ? "collecting" : active ? "queueing" : null,
        active,
        discovered: active ? items.length : generation > 0 ? items.length : 0,
        processed: collecting ? 0 : active ? queueingProgress : generation > 0 ? items.length : 0,
        total: collecting ? null : generation > 0 ? items.length : null,
        message: active ? "Рекурсивный поиск: найдено 10000 файлов." : "",
        error: null,
        updated_at: active
          ? `2026-08-11T12:00:0${Math.min(3, activePreparationStatusRequests)}Z`
          : "2026-08-11T12:00:04Z",
        logs: active
          ? [{ id: 1, message: "Рекурсивный поиск: найдено 10000 файлов." }]
          : [],
      },
    });
  });
  await page.route("**/api/pick", async (route) => {
    pickCalls += 1;
    if (pickCalls === 1) {
      firstPickPending = true;
      await new Promise((resolve) => setTimeout(resolve, 1800));
      firstPickPending = false;
      await route.fulfill({
        json: {
          mode: "folder",
          path: rootPath,
          paths: items.map((item) => item.path),
          items,
          cancelled: false,
          recursive: true,
        },
      });
      return;
    }
    if (pickCalls === 2) {
      await route.fulfill({
        json: {
          mode: "folder",
          path: emptyRootPath,
          paths: [],
          items: [],
          cancelled: false,
          recursive: true,
        },
      });
      return;
    }
    if (pickCalls === 3) {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Контрольная ошибка рекурсивного обхода." }),
      });
      return;
    }
    await route.fulfill({
      json: {
        mode: "folder",
        path: "",
        paths: [],
        items: [],
        cancelled: true,
        recursive: true,
      },
    });
  });
  await page.route("**/api/refresh", (route) => {
    refreshPayloads.push(route.request().postDataJSON());
    route.fulfill({
      json: {
        path: rootPath,
        paths: refreshedItems.map((item) => item.path),
        items: refreshedItems,
        recursive: true,
      },
    });
  });
  await page.route("**/api/translate", (route) => route.fulfill({ json: { job_id: jobId } }));
  await page.route("**/api/stream/**", (route) => {
    assert.equal(new URL(route.request().url()).pathname, `/api/stream/${jobId}`);
    const events = [
      { type: "job", total: 2 },
      { type: "file", path: items[0].path, state: "done", progress: 100, output: items[0].output },
      { type: "file", path: items[1].path, state: "done", progress: 100, output: items[1].output },
      { type: "done", status: "ok" },
    ];
    return route.fulfill({
      status: 200,
      headers: {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream",
      },
      body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
    });
  });
  await page.route("**/api/jobs/**", (route) => route.fulfill({
    json: {
      job_id: jobId,
      completed: true,
      status: "ok",
      items: items.slice(0, 2).map((item) => ({ ...item, state: "done", progress: 100 })),
      logs: ["Финальный снимок перевода папки."],
      total: 2,
      done: 2,
    },
  }));

  await page.goto(baseUrl);
  await page.waitForFunction(() => document.querySelector("#sourceLang")?.options.length > 0);
  await page.click("#pickerFolderBtn");
  await page.waitForFunction(() => document.querySelector("#pickerFolderBtn")?.disabled);
  await page.evaluate(() => document.querySelector("#pickerFolderBtn")?.click());
  await page.waitForFunction(() => (
    document.querySelector("#logConsole")?.textContent.includes("Рекурсивный поиск: найдено 10000")
  ));
  await page.waitForFunction(() => document.querySelector("#pickerHint")?.textContent.includes("1/10000"));
  assert.equal(await page.locator("#pickerHint").innerText(), "Формируется список: 1/10000");
  await page.waitForFunction(() => document.querySelector("#pickerHint")?.textContent.includes("2/10000"));
  assert.equal(await page.locator("#pickerHint").innerText(), "Формируется список: 2/10000");
  assert.equal(pickCalls, 1);
  assert.equal(await page.locator("#picker").getAttribute("aria-busy"), "true");
  assert.equal(await page.locator("#refreshBtn").isDisabled(), true);
  assert.match(await page.locator("#logConsole").innerText(), /все подпапки будут включены/);

  await page.waitForFunction(() => document.querySelectorAll(".file-card").length === 200);
  assert.equal(await page.locator("#picker").getAttribute("aria-busy"), "false");
  assert.equal(await page.locator("#selectedPath").getAttribute("title"), rootPath);
  assert.equal(await page.locator("#fileList").getAttribute("data-total-count"), "10000");
  assert.equal(await page.locator("#fileList").getAttribute("data-rendered-count"), "200");
  assert.deepEqual(
    await page.locator(".file-name").evaluateAll((nodes) => [nodes[0].textContent, nodes.at(-1).textContent]),
    ["Урок 00001.srt", "Урок 00200.srt"],
  );
  await page.getByRole("button", { name: "Далее" }).click();
  assert.deepEqual(
    await page.locator(".file-name").evaluateAll((nodes) => [nodes[0].textContent, nodes.at(-1).textContent]),
    ["Урок 00201.srt", "Урок 00400.srt"],
  );
  const progressLog = await page.locator("#logConsole").innerText();
  assert.equal(progressLog.split("Рекурсивный поиск: найдено 10000 файлов.").length - 1, 1);

  await page.locator("#batchSize").fill("19");
  await page.selectOption("#translatorApi", "google");
  await page.selectOption("#sourceLang", "ru");
  await new Promise((resolve) => setTimeout(resolve, 700));
  assert.equal(refreshPayloads.length, 0);
  await page.click("#translateBtn");
  await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("Перевод завершён."));
  assert.equal(await page.locator("#selectedPath").getAttribute("title"), rootPath);
  await page.click("#refreshBtn");
  for (let attempt = 0; attempt < 50 && refreshPayloads.length === 0; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  assert.equal(refreshPayloads.length > 0, true);
  assert.deepEqual(refreshPayloads[0].paths, [rootPath]);
  assert.equal(refreshPayloads[0].recursive, true);
  await page.waitForFunction(() => document.querySelector("#fileList")?.dataset.totalCount === "10001");
  assert.match(await page.locator("#fileList").innerText(), /000 Новый вложенный урок\.srt/);
  assert.equal(await page.locator("#selectedPath").getAttribute("title"), rootPath);
  const stored = JSON.parse(await page.evaluate(() => localStorage.getItem("subTranslateSettings")) || "{}");
  assert.equal(Object.hasOwn(stored, "recursive"), false);
  assert.equal(Object.hasOwn(stored, "path"), false);
  assert.equal(Object.hasOwn(stored, "paths"), false);

  await page.click("#pickerFolderBtn");
  await page.waitForFunction(() => document.querySelector(".empty-state")?.textContent.includes("не найдены"));
  assert.equal(await page.locator("#selectedPath").getAttribute("title"), emptyRootPath);

  await page.click("#pickerFolderBtn");
  await page.waitForFunction(() => (
    document.querySelector("#logConsole")?.textContent.includes("Контрольная ошибка рекурсивного обхода")
  ));
  assert.equal(await page.locator("#selectedPath").getAttribute("title"), emptyRootPath);
  assert.equal(await page.locator("#pickerFolderBtn").isDisabled(), false);

  await page.click("#pickerFolderBtn");
  await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("Выбор отменён"));
  assert.equal(await page.locator("#selectedPath").getAttribute("title"), emptyRootPath);

  await page.evaluate(() => {
    window.dispatchEvent(new ErrorEvent("error", {
      message: "Контрольная ошибка интерфейса",
      error: new Error("Контрольная ошибка интерфейса"),
    }));
    const rejection = new Event("unhandledrejection");
    Object.defineProperty(rejection, "reason", {
      value: new Error("Контрольное отклонение операции"),
    });
    window.dispatchEvent(rejection);
  });
  assert.match(await page.locator("#logConsole").innerText(), /Необработанная ошибка интерфейса/);
  assert.match(await page.locator("#logConsole").innerText(), /Необработанная ошибка операции/);
  await page.close();
});
