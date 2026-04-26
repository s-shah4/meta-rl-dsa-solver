from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel

from env.adapt_env import AdaptEnvironment
from env.test_cases import load_problem_bank
from models import AdaptAction, AdaptObservation, AdaptState
from server.runtime import SpaceTrainingManager

ENV_NAME = "adapt-dsa-tutor"
ENV_DESCRIPTION = (
    "Adversarial DSA Programming Tutor - RL environment for training LLMs to solve "
    "algorithmic problems through adaptive curriculum and self-repair."
)
ENV_VERSION = "0.4.0"
SESSION_TTL = timedelta(minutes=30)
SESSIONS: dict[str, AdaptEnvironment] = {}
SESSION_LAST_ACCESSED: dict[str, datetime] = {}
TRAINING_MANAGER = SpaceTrainingManager()
TASKS = [
    {
        "name": problem["problem_id"],
        "difficulty": problem["difficulty"],
        "description": problem["problem"],
    }
    for problem in load_problem_bank()
]

app = FastAPI(title="ADAPT DSA Tutor OpenEnv", version=ENV_VERSION)


class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    problem_id: Optional[str] = None
    difficulty: Optional[str] = None


class TrainRequest(BaseModel):
    preset: str = "overnight"
    model_name: Optional[str] = None
    output_dir: Optional[str] = None
    dataset_size: Optional[int] = None
    max_steps: Optional[int] = None
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    num_generations: Optional[int] = None
    max_seq_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    learning_rate: Optional[float] = None
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    load_in_4bit: Optional[bool] = None
    gradient_checkpointing: Optional[bool] = None
    bf16: Optional[bool] = None
    evaluation_episodes: Optional[int] = None
    eval_max_new_tokens: Optional[int] = None
    baseline_eval: Optional[bool] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    generator_mode: Optional[str] = None
    non_deterministic_generator: Optional[bool] = None
    use_dataset: bool = False
    dataset_name: str = "deepmind/code_contests"
    dataset_max_problems: int = 5000
    disable_wandb: Optional[bool] = None
    trace_logging_enabled: Optional[bool] = None
    checkpoint_log_interval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = None
    upload_checkpoints_to_hub: Optional[bool] = None
    save_merged_model: Optional[bool] = None


class RunTrainedPolicyRequest(BaseModel):
    problem_id: Optional[str] = None
    difficulty: Optional[str] = None
    max_new_tokens: int = 512


class GenerateCodeRequest(BaseModel):
    problem: str
    input_format: str
    constraints: str
    feedback: Optional[str] = None
    problem_id: str = "custom_problem"
    problem_type: str = "custom"
    difficulty: str = "custom"
    attempt_number: int = 1
    max_steps: int = 1
    max_new_tokens: int = 512


class RunCodeRequest(BaseModel):
    code: str
    stdin: str = ""


DEMO_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ADAPT Judge Demo</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    :root {
      color-scheme: dark;
      --bg: #071018;
      --panel: rgba(15, 23, 42, 0.92);
      --panel-soft: rgba(18, 28, 45, 0.82);
      --border: rgba(148, 163, 184, 0.18);
      --text: #e5eefb;
      --muted: #8fa6c2;
      --accent: #64d2ff;
      --accent-2: #7cf29a;
      --warn: #facc15;
      --danger: #fb7185;
      --shadow: 0 24px 60px rgba(2, 6, 23, 0.45);
      --radius: 20px;
      --mono: "SFMono-Regular", "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(100, 210, 255, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(124, 242, 154, 0.12), transparent 24%),
        linear-gradient(180deg, #09111b 0%, #050a11 100%);
      color: var(--text);
    }

    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 32px 20px 40px;
    }

    .hero {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 24px;
      flex-wrap: wrap;
    }

    .hero h1 {
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3rem);
      letter-spacing: -0.04em;
    }

    .hero p {
      margin: 0;
      max-width: 720px;
      color: var(--muted);
      line-height: 1.6;
    }

    .status-pill {
      border: 1px solid var(--border);
      background: rgba(10, 16, 28, 0.85);
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 0.92rem;
      color: var(--muted);
      white-space: nowrap;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
      gap: 20px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 22px;
      backdrop-filter: blur(18px);
    }

    .panel h2 {
      margin: 0;
      font-size: 1.25rem;
    }

    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }

    .subtle {
      color: var(--muted);
      font-size: 0.95rem;
      line-height: 1.5;
    }

    .stack {
      display: grid;
      gap: 14px;
    }

    label {
      display: block;
      font-size: 0.9rem;
      margin-bottom: 8px;
      color: #d7e3f3;
    }

    textarea,
    pre,
    table {
      width: 100%;
    }

    textarea {
      resize: vertical;
      min-height: 180px;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.18);
      background: rgba(6, 12, 22, 0.88);
      color: var(--text);
      padding: 14px 16px;
      font: 0.95rem/1.6 var(--mono);
      outline: none;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    textarea:focus {
      border-color: rgba(100, 210, 255, 0.55);
      box-shadow: 0 0 0 3px rgba(100, 210, 255, 0.12);
    }

    button {
      appearance: none;
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      font-weight: 700;
      cursor: pointer;
      background: linear-gradient(135deg, #64d2ff 0%, #47b4ff 100%);
      color: #04111c;
      transition: transform 0.15s ease, filter 0.15s ease, opacity 0.15s ease;
    }

    button.secondary {
      background: rgba(148, 163, 184, 0.12);
      color: var(--text);
      border: 1px solid rgba(148, 163, 184, 0.18);
    }

    button:hover:not(:disabled) {
      transform: translateY(-1px);
      filter: brightness(1.05);
    }

    button:disabled {
      cursor: wait;
      opacity: 0.72;
    }

    .button-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }

    .code-block,
    .output-block {
      min-height: 180px;
      margin: 0;
      padding: 16px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      background:
        linear-gradient(180deg, rgba(17, 24, 39, 0.96) 0%, rgba(9, 14, 24, 0.98) 100%);
      font: 0.93rem/1.55 var(--mono);
      color: #d8e9fb;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }

    .stat-card {
      padding: 16px;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      background: var(--panel-soft);
    }

    .stat-label {
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 6px;
    }

    .stat-value {
      font-size: 1.5rem;
      font-weight: 800;
      letter-spacing: -0.03em;
    }

    .chart-shell {
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      background: rgba(6, 12, 22, 0.76);
      padding: 16px;
      min-height: 320px;
      position: relative;
    }

    .chart-empty {
      position: absolute;
      inset: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      color: var(--muted);
      border: 1px dashed rgba(148, 163, 184, 0.18);
      border-radius: 12px;
      padding: 24px;
    }

    canvas {
      width: 100% !important;
      height: 280px !important;
    }

    table {
      border-collapse: collapse;
      margin-top: 18px;
      overflow: hidden;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      background: rgba(6, 12, 22, 0.8);
    }

    th,
    td {
      text-align: left;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(148, 163, 184, 0.12);
      vertical-align: top;
    }

    th {
      color: var(--muted);
      width: 42%;
      font-weight: 600;
    }

    tr:last-child th,
    tr:last-child td {
      border-bottom: 0;
    }

    .error {
      color: var(--danger);
    }

    @media (max-width: 1024px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 720px) {
      .shell {
        padding: 24px 14px 28px;
      }

      .panel {
        padding: 18px;
      }

      .summary-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div>
        <h1>ADAPT Judge Demo</h1>
        <p>
          Explore the model, run generated Python against custom stdin, and inspect the latest training results
          from the same Space deployment.
        </p>
      </div>
      <div class="status-pill" id="page-status">Ready</div>
    </div>

    <div class="layout">
      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Section 1 - Model Playground</h2>
            <div class="subtle">Paste a DSA problem statement, generate a solution, then run it with your own input.</div>
          </div>
        </div>

        <div class="stack">
          <div>
            <label for="problem-input">Problem Statement</label>
            <textarea id="problem-input" placeholder="Example: Given an array of integers, print the maximum subarray sum. The first line contains n, the second line contains n integers..."></textarea>
          </div>

          <div class="button-row">
            <button id="generate-button" type="button">Generate Solution</button>
          </div>

          <div>
            <label for="generated-code">Generated Code</label>
            <pre id="generated-code" class="code-block"># Generated code will appear here.</pre>
          </div>

          <div>
            <label for="stdin-input">Custom Input</label>
            <textarea id="stdin-input" style="min-height: 140px;" placeholder="Paste stdin here exactly as the program should receive it."></textarea>
          </div>

          <div class="button-row">
            <button id="run-button" type="button" class="secondary">Run Code</button>
          </div>

          <div>
            <label for="run-output">Execution Output</label>
            <pre id="run-output" class="output-block"># Stdout and stderr will appear here.</pre>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <div>
            <h2>Section 2 - Training Results</h2>
            <div class="subtle">Live status, charted reward data, and rollout metrics fetched directly from this Space.</div>
          </div>
          <button id="refresh-button" type="button" class="secondary">Refresh</button>
        </div>

        <div class="summary-grid">
          <div class="stat-card">
            <div class="stat-label">Overall Accuracy</div>
            <div id="overall-accuracy" class="stat-value">--</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">Training Status</div>
            <div id="training-status" class="stat-value">Loading</div>
          </div>
        </div>

        <div class="chart-shell">
          <canvas id="reward-chart"></canvas>
          <div id="chart-empty" class="chart-empty" hidden>No reward curve is available yet. Start a run or refresh after logs are written.</div>
        </div>

        <table>
          <tbody id="metrics-body">
            <tr><th>Loading</th><td>Fetching training metrics...</td></tr>
          </tbody>
        </table>
      </section>
    </div>
  </div>

  <script>
    let currentCode = "";
    let rewardChart = null;

    const pageStatus = document.getElementById("page-status");
    const problemInput = document.getElementById("problem-input");
    const stdinInput = document.getElementById("stdin-input");
    const generateButton = document.getElementById("generate-button");
    const runButton = document.getElementById("run-button");
    const refreshButton = document.getElementById("refresh-button");
    const generatedCodeBlock = document.getElementById("generated-code");
    const runOutputBlock = document.getElementById("run-output");
    const overallAccuracy = document.getElementById("overall-accuracy");
    const trainingStatus = document.getElementById("training-status");
    const metricsBody = document.getElementById("metrics-body");
    const chartEmpty = document.getElementById("chart-empty");
    const rewardChartCanvas = document.getElementById("reward-chart");

    function setStatus(message, isError = false) {
      pageStatus.textContent = message;
      pageStatus.classList.toggle("error", isError);
    }

    function setButtonLoading(button, loading, idleText, loadingText) {
      button.disabled = loading;
      button.textContent = loading ? loadingText : idleText;
    }

    function formatPercent(value) {
      if (value === null || value === undefined || value === "") {
        return "--";
      }
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        return "--";
      }
      return `${(numeric * 100).toFixed(1)}%`;
    }

    function formatValue(value) {
      if (value === null || value === undefined || value === "") {
        return "-";
      }
      if (typeof value === "number") {
        if (!Number.isFinite(value)) {
          return "-";
        }
        if (Math.abs(value) >= 100 || Number.isInteger(value)) {
          return String(value);
        }
        return value.toFixed(4).replace(/0+$/, "").replace(/\\.$/, "");
      }
      return String(value);
    }

    function titleize(value) {
      return String(value)
        .replace(/_/g, " ")
        .replace(/\\b\\w/g, (match) => match.toUpperCase());
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, options);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = payload && typeof payload.detail === "string" ? payload.detail : `Request failed (${response.status})`;
        throw new Error(detail);
      }
      return payload;
    }

    function renderMetricsTable(status) {
      const rows = [];

      const addRow = (label, value) => {
        if (value === null || value === undefined || value === "") {
          return;
        }
        rows.push([label, formatValue(value)]);
      };

      const addNestedRows = (prefix, data) => {
        if (!data || typeof data !== "object" || Array.isArray(data)) {
          return;
        }
        Object.entries(data).forEach(([key, value]) => {
          if (value === null || value === undefined || typeof value === "object") {
            return;
          }
          addRow(`${prefix} ${titleize(key)}`, value);
        });
      };

      addRow("Status", status.status);
      addRow("Phase", status.phase);
      addRow("Completed Steps", status.completed_steps);
      addRow("Total Steps", status.total_steps);
      addRow("Remaining Steps", status.remaining_steps);
      addRow("Progress Ratio", status.progress_ratio);
      addRow("Last Execution Status", status.last_execution_status);
      addRow("Current Difficulty", status.current_difficulty);
      addRow("Train Episode Index", status.train_episode_index);
      addRow("Last Problem ID", status.last_problem_id);
      addRow("Elapsed Hours", status.elapsed_hours);

      addNestedRows("Rolling", status.rolling_metrics);
      addNestedRows("Timing", status.timing_summary);
      addNestedRows("Baseline", status.baseline_summary);
      addNestedRows("Trained", status.trained_summary);

      metricsBody.replaceChildren();

      if (!rows.length) {
        const row = document.createElement("tr");
        const key = document.createElement("th");
        key.textContent = "Metrics";
        const value = document.createElement("td");
        value.textContent = "No metrics available yet.";
        row.append(key, value);
        metricsBody.appendChild(row);
        return;
      }

      rows.forEach(([label, value]) => {
        const row = document.createElement("tr");
        const key = document.createElement("th");
        key.textContent = label;
        const val = document.createElement("td");
        val.textContent = value;
        row.append(key, val);
        metricsBody.appendChild(row);
      });
    }

    function renderRewardChart(curve) {
      const points = Array.isArray(curve) ? curve : [];
      const hasData = points.length > 0;

      chartEmpty.hidden = hasData;
      rewardChartCanvas.hidden = !hasData;

      if (rewardChart) {
        rewardChart.destroy();
        rewardChart = null;
      }

      if (!hasData) {
        return;
      }

      rewardChart = new Chart(rewardChartCanvas, {
        type: "line",
        data: {
          labels: points.map((item) => item.step),
          datasets: [
            {
              label: "Episode Reward",
              data: points.map((item) => item.episode_reward),
              borderColor: "#64d2ff",
              backgroundColor: "rgba(100, 210, 255, 0.12)",
              yAxisID: "reward",
              pointRadius: 0,
              borderWidth: 2,
              tension: 0.18,
            },
            {
              label: "Pass Rate",
              data: points.map((item) => item.pass_rate),
              borderColor: "#7cf29a",
              backgroundColor: "rgba(124, 242, 154, 0.12)",
              yAxisID: "rate",
              pointRadius: 0,
              borderWidth: 2,
              tension: 0.18,
            },
            {
              label: "Visible Pass Rate",
              data: points.map((item) => item.visible_pass_rate),
              borderColor: "#facc15",
              backgroundColor: "rgba(250, 204, 21, 0.12)",
              yAxisID: "rate",
              pointRadius: 0,
              borderWidth: 2,
              tension: 0.18,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: "index",
            intersect: false,
          },
          plugins: {
            legend: {
              labels: {
                color: "#d8e9fb",
              },
            },
          },
          scales: {
            x: {
              ticks: { color: "#8fa6c2" },
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              title: { display: true, text: "Step", color: "#8fa6c2" },
            },
            reward: {
              ticks: { color: "#8fa6c2" },
              grid: { color: "rgba(148, 163, 184, 0.08)" },
              title: { display: true, text: "Reward", color: "#8fa6c2" },
            },
            rate: {
              position: "right",
              min: 0,
              max: 1,
              ticks: { color: "#8fa6c2" },
              grid: { drawOnChartArea: false },
              title: { display: true, text: "Pass Rate", color: "#8fa6c2" },
            },
          },
        },
      });
    }

    async function loadTrainingStatus() {
      setStatus("Refreshing training status...");
      try {
        const status = await requestJson("./train/status");
        overallAccuracy.textContent = formatPercent(status.overall_accuracy);
        trainingStatus.textContent = titleize(status.status || "unknown");
        renderRewardChart(status.reward_curve);
        renderMetricsTable(status);
        setStatus("Training results loaded");
      } catch (error) {
        overallAccuracy.textContent = "--";
        trainingStatus.textContent = "Unavailable";
        renderRewardChart([]);
        metricsBody.innerHTML = "<tr><th>Error</th><td>Unable to fetch training metrics.</td></tr>";
        setStatus(`Training status error: ${error.message}`, true);
      }
    }

    async function handleGenerateCode() {
      const problem = problemInput.value.trim();
      if (!problem) {
        generatedCodeBlock.textContent = "# Enter a DSA problem first.";
        setStatus("Problem statement required", true);
        return;
      }

      setButtonLoading(generateButton, true, "Generate Solution", "Generating...");
      generatedCodeBlock.textContent = "# Generating solution...";
      setStatus("Generating solution...");

      try {
        const result = await requestJson("./generate-code", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            problem,
            input_format: "Infer stdin structure from the problem statement and examples.",
            constraints: "Prefer a correct and reasonably efficient Python solution.",
          }),
        });

        currentCode = (result.code || result.completion || "").trim();
        generatedCodeBlock.textContent = currentCode || "# No code returned.";
        setStatus("Solution ready");
      } catch (error) {
        currentCode = "";
        generatedCodeBlock.textContent = `# Error\\n${error.message}`;
        setStatus(`Generation failed: ${error.message}`, true);
      } finally {
        setButtonLoading(generateButton, false, "Generate Solution", "Generating...");
      }
    }

    async function handleRunCode() {
      if (!currentCode.trim()) {
        runOutputBlock.textContent = "# Generate a solution before running code.";
        setStatus("No generated code to run", true);
        return;
      }

      setButtonLoading(runButton, true, "Run Code", "Running...");
      runOutputBlock.textContent = "# Running code...";
      setStatus("Running generated code...");

      try {
        const result = await requestJson("./run-code", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            code: currentCode,
            stdin: stdinInput.value,
          }),
        });

        const chunks = [];
        if (result.stdout) {
          chunks.push(`STDOUT\\n${result.stdout}`);
        }
        if (result.stderr) {
          chunks.push(`STDERR\\n${result.stderr}`);
        }
        runOutputBlock.textContent = chunks.length ? chunks.join("\\n\\n") : "Program finished with no output.";
        setStatus("Execution complete");
      } catch (error) {
        runOutputBlock.textContent = `# Error\\n${error.message}`;
        setStatus(`Execution failed: ${error.message}`, true);
      } finally {
        setButtonLoading(runButton, false, "Run Code", "Running...");
      }
    }

    generateButton.addEventListener("click", handleGenerateCode);
    runButton.addEventListener("click", handleRunCode);
    refreshButton.addEventListener("click", loadTrainingStatus);

    loadTrainingStatus();
  </script>
</body>
</html>
"""


def _metadata() -> dict[str, Any]:
    return {
        "name": ENV_NAME,
        "description": ENV_DESCRIPTION,
        "version": ENV_VERSION,
        "tasks": TASKS,
        "mode": "simulation",
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _cleanup_sessions() -> None:
    now = _utc_now()
    expired = [
        session_id
        for session_id, last_seen in SESSION_LAST_ACCESSED.items()
        if now - last_seen > SESSION_TTL
    ]
    for session_id in expired:
        SESSIONS.pop(session_id, None)
        SESSION_LAST_ACCESSED.pop(session_id, None)


def _touch_session(session_id: str) -> None:
    SESSION_LAST_ACCESSED[session_id] = _utc_now()


def _require_session(session_id: str) -> AdaptEnvironment:
    _cleanup_sessions()
    env = SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown or expired session_id: {session_id}")
    _touch_session(session_id)
    return env


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _read_json_dict(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    try:
        path = Path(str(path_value))
        if not path.exists() or not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_reward_curve(csv_path_value: Any) -> list[dict[str, Any]]:
    if not csv_path_value:
        return []
    try:
        csv_path = Path(str(csv_path_value))
        if not csv_path.exists() or not csv_path.is_file():
            return []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            reward_curve: list[dict[str, Any]] = []
            for row in reader:
                step = _int_or_none(row.get("step"))
                if step is None:
                    continue
                reward_curve.append(
                    {
                        "step": step,
                        "episode_reward": _float_or_none(row.get("episode_reward")),
                        "pass_rate": _float_or_none(row.get("pass_rate")),
                        "visible_pass_rate": _float_or_none(row.get("visible_pass_rate")),
                    }
                )
            return reward_curve
    except (OSError, csv.Error):
        return []


def _stringify_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _enriched_train_status() -> dict[str, Any]:
    payload = TRAINING_MANAGER.status_payload()
    run_summary = _read_json_dict(payload.get("run_summary_path"))

    if run_summary:
        rolling_metrics = run_summary.get("rolling_metrics")
        if isinstance(rolling_metrics, dict) and not payload.get("rolling_metrics"):
            payload["rolling_metrics"] = rolling_metrics

        final_metrics = run_summary.get("final_metrics")
        if isinstance(final_metrics, dict):
            if not payload.get("baseline_summary") and isinstance(final_metrics.get("baseline_summary"), dict):
                payload["baseline_summary"] = final_metrics["baseline_summary"]
            if not payload.get("trained_summary") and isinstance(final_metrics.get("trained_summary"), dict):
                payload["trained_summary"] = final_metrics["trained_summary"]
            if not payload.get("timing_summary") and isinstance(final_metrics.get("timing_summary"), dict):
                payload["timing_summary"] = final_metrics["timing_summary"]

    payload["reward_curve"] = _parse_reward_curve(payload.get("reward_curve_csv"))

    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    trained_summary = payload.get("trained_summary") if isinstance(payload.get("trained_summary"), dict) else {}
    rolling_metrics = payload.get("rolling_metrics") if isinstance(payload.get("rolling_metrics"), dict) else {}

    overall_accuracy = None
    if config.get("baseline_eval"):
        overall_accuracy = _float_or_none(trained_summary.get("overall"))
    if overall_accuracy is None:
        overall_accuracy = _float_or_none(rolling_metrics.get("avg_pass_rate"))
    if overall_accuracy is None:
        overall_accuracy = _float_or_none(payload.get("last_pass_rate"))

    payload["overall_accuracy"] = overall_accuracy
    return payload


@app.on_event("startup")
def startup() -> None:
    TRAINING_MANAGER.load_latest_model()


@app.get("/")
def root() -> HTMLResponse:
    _cleanup_sessions()
    return HTMLResponse(content=DEMO_PAGE_HTML)


@app.get("/web", include_in_schema=False)
def web_root() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@app.get("/web/", include_in_schema=False)
def web_root_slash() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, Any]:
    _cleanup_sessions()
    return {
        "status": "healthy",
        "active_sessions": len(SESSIONS),
        "training": TRAINING_MANAGER.status_payload()["status"],
        "model_loaded": TRAINING_MANAGER.model_status_payload()["loaded"],
    }


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    _cleanup_sessions()
    return _metadata()


@app.get("/tasks")
def list_tasks() -> dict[str, Any]:
    _cleanup_sessions()
    return {"tasks": TASKS}


@app.get("/schema")
def schema() -> dict[str, Any]:
    _cleanup_sessions()
    return {
        "action": AdaptAction.model_json_schema(),
        "observation": AdaptObservation.model_json_schema(),
        "state": AdaptState.model_json_schema(),
    }


@app.get("/train/status")
def train_status() -> dict[str, Any]:
    return _enriched_train_status()


@app.get("/model/status")
def model_status() -> dict[str, Any]:
    return TRAINING_MANAGER.model_status_payload()


@app.post("/train")
def train(request: Optional[TrainRequest] = None) -> dict[str, Any]:
    try:
        return TRAINING_MANAGER.start_training((request or TrainRequest()).model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/run-trained-policy")
def run_trained_policy(request: Optional[RunTrainedPolicyRequest] = None) -> dict[str, Any]:
    effective_request = request or RunTrainedPolicyRequest()
    try:
        return TRAINING_MANAGER.run_trained_policy(
            problem_id=effective_request.problem_id,
            difficulty=effective_request.difficulty,
            max_new_tokens=effective_request.max_new_tokens,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"run-trained-policy failed: {exc}") from exc


@app.post("/run-code")
def run_code(request: RunCodeRequest) -> dict[str, str]:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
            handle.write(request.code)
            temp_path = Path(handle.name)

        completed = subprocess.run(
            [sys.executable, str(temp_path)],
            input=request.stdin,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        stderr = _stringify_subprocess_output(exc.stderr)
        timeout_message = "Execution timed out after 5 seconds."
        stderr = f"{stderr.rstrip()}\n{timeout_message}".strip() if stderr else timeout_message
        return {
            "stdout": _stringify_subprocess_output(exc.stdout),
            "stderr": stderr,
        }
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"run-code failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"run-code failed: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/generate-code")
def generate_code(request: GenerateCodeRequest) -> dict[str, Any]:
    try:
        return TRAINING_MANAGER.generate_code(
            problem=request.problem,
            input_format=request.input_format,
            constraints=request.constraints,
            feedback=request.feedback,
            problem_id=request.problem_id,
            problem_type=request.problem_type,
            difficulty=request.difficulty,
            attempt_number=request.attempt_number,
            max_steps=request.max_steps,
            max_new_tokens=request.max_new_tokens,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generate-code failed: {exc}") from exc


@app.post("/mcp")
def mcp(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    _cleanup_sessions()
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "error": {
            "code": -32601,
            "message": "MCP methods are not implemented for this environment.",
        },
    }


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> dict[str, Any]:
    _cleanup_sessions()
    effective_request = request or ResetRequest()
    session_id = effective_request.session_id or str(uuid4())
    env = AdaptEnvironment(session_id=session_id)
    SESSIONS[session_id] = env
    _touch_session(session_id)
    observation = env.reset(
        session_id=session_id,
        seed=effective_request.seed,
        episode_id=effective_request.episode_id,
        problem_id=effective_request.problem_id,
        difficulty=effective_request.difficulty,
    )
    return observation.model_dump()


@app.post("/step")
async def step(request: Request) -> dict[str, Any]:
    _cleanup_sessions()
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")

    raw_action = payload.get("action", payload)
    try:
        effective_action = AdaptAction.model_validate(raw_action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action payload: {exc}") from exc

    if not effective_action.session_id:
        raise HTTPException(status_code=422, detail="`session_id` is required in the /step request body.")

    env = _require_session(effective_action.session_id)
    observation = env.step(effective_action)
    return {
        "observation": observation.model_dump(),
        "reward": float(observation.reward),
        "done": bool(observation.done),
        "info": {
            "session_id": observation.session_id,
            "feedback": observation.feedback,
            "pass_rate": observation.pass_rate,
            "visible_pass_rate": observation.visible_pass_rate,
            "execution_status": observation.execution_status,
        },
    }


@app.get("/state")
def state(session_id: str = Query(..., description="Session id returned from /reset.")) -> dict[str, Any]:
    env = _require_session(session_id)
    if not env.problem:
        env.reset(session_id=session_id)
    return env.state.model_dump()


def main(host: Optional[str] = None, port: Optional[int] = None) -> None:
    if host is None or port is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=7860)
        args = parser.parse_args()
        host = args.host if host is None else host
        port = args.port if port is None else port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
