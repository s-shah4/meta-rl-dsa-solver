"""
STDOUT FORMAT (must match exactly):
[START] task=<task_name> env=adapt_dsa_tutor model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
from typing import Any

from env.adapt_env import AdaptEnvironment
from env.test_cases import load_problem_bank
from models import AdaptAction

BENCHMARK = "adapt_dsa_tutor"
TASKS = [problem["problem_id"] for problem in load_problem_bank()]
SYSTEM_PROMPT = """You are solving a programming problem in Python.

You will receive:
- a problem statement
- input format
- constraints
- feedback from previous attempts

Reply with ONLY runnable Python code. The code must read from stdin and print to stdout.
Do not include markdown fences or explanations."""


def require_env(name: str, value: str | None) -> str:
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def safe_log_value(value: str | None) -> str:
    if not value:
        return "null"
    return str(value).replace("\n", "_").replace("\r", "_").replace("\t", "_").replace(" ", "_")


def extract_code(response_text: str) -> str:
    text = response_text.strip()
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if text.startswith("python"):
        text = text[6:].strip()
    return text


def build_user_prompt(observation: dict[str, Any]) -> str:
    payload = {
        "problem_id": observation["problem_id"],
        "problem_type": observation["problem_type"],
        "difficulty": observation["difficulty"],
        "problem": observation["problem"],
        "input_format": observation["input_format"],
        "constraints": observation["constraints"],
        "feedback": observation["feedback"],
    }
    return json.dumps(payload, indent=2)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={safe_log_value(action_str)} reward={reward:.2f} "
        f"done={str(done).lower()} error={safe_log_value(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def run_task(task_name: str) -> float:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` package is required for inference runs. Install it before running inference.py."
        ) from exc

    api_key = require_env("HF_TOKEN", os.getenv("HF_TOKEN"))
    base_url = require_env("API_BASE_URL", os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
    model_name = require_env("MODEL_NAME", os.getenv("MODEL_NAME", "openai/gpt-oss-120b"))

    client = OpenAI(base_url=base_url, api_key=api_key)
    env = AdaptEnvironment()
    observation = env.reset(problem_id=task_name)

    log_start(task_name, BENCHMARK, model_name)
    rewards = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    max_steps = 3
    for step_index in range(1, max_steps + 1):
        messages.append({"role": "user", "content": build_user_prompt(observation.model_dump())})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            response_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": response_text})

            code = extract_code(response_text)
            observation = env.step(AdaptAction(code=code))
            rewards.append(float(observation.reward))
            log_step(step_index, "submit_code", float(observation.reward), bool(observation.done), None)

            if observation.pass_rate == 1.0 or observation.done:
                break

        except Exception as exc:
            rewards.append(0.0)
            log_step(step_index, "parse_error", 0.0, False, str(exc))
            messages.append(
                {
                    "role": "user",
                    "content": f"Your last response failed. Error: {exc}. Reply with only Python code.",
                }
            )

    success = observation.pass_rate == 1.0
    score = float(observation.reward)
    log_end(success, len(rewards), score, rewards)
    return score


def main() -> dict[str, float]:
    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)
    return scores


if __name__ == "__main__":
    main()
