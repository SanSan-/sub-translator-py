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
  await page.selectOption("#translatorApi", "seedx");
  assert.match(await page.locator("#localModelMeta").innerText(), /ByteDance-Seed\/Seed-X-PPO-7B-AWQ-Int4/);
  assert.equal(await page.locator("#requestTimeout").inputValue(), "3600");
  assert.equal(await page.locator("#autoDownloadModel").isChecked(), false);
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
  await page.waitForFunction(() => document.querySelector(".file-card")?.dataset.status === "cached");
  await page.click("#translateBtn");
  await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("Тестовая ошибка запуска"));

  await page.click("#unloadBtn");
  await page.waitForFunction(() => document.querySelector("#logConsole")?.textContent.includes("Модели выгружены"));
  assert.equal(await page.locator("#openAiKeyStatus").count(), 1);
  assert.equal(await page.locator('[data-setting="openai_api_key"]').count(), 0);
  await page.close();
});
