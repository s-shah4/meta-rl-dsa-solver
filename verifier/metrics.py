from __future__ import annotations

from typing import Any

STEP_DISCOUNTS = {1: 1.0, 2: 0.85, 3: 0.70}
TERMINAL_ZERO_STATUSES = {"syntax_error", "safety_violation", "timeout"}


def step_discount(step_number: int) -> float:
    return STEP_DISCOUNTS.get(int(step_number), 0.70)


def compute_reward(
    pass_rate: float,
    step_number: int,
    execution_status: str,
    format_compliance: float,
) -> float:
    del format_compliance
    correctness = max(0.0, min(float(pass_rate), 1.0))
    if execution_status in TERMINAL_ZERO_STATUSES:
        return 0.0
    reward = correctness * step_discount(step_number)
    return round(min(max(reward, 0.0), 1.0), 4)


def compute_pass_rate(
    results: list[dict[str, Any]],
    step_number: int = 1,
    *,
    syntax_ok: bool = True,
    safety_ok: bool = True,
    precheck_status: str | None = None,
) -> tuple[float, dict[str, Any]]:
    total = len(results)
    hidden_results = [result for result in results if result.get("visibility") == "hidden"]
    visible_results = [result for result in results if result.get("visibility") == "visible"]

    hidden_total = len(hidden_results)
    visible_total = len(visible_results)
    hidden_passed = sum(1 for result in hidden_results if result.get("passed"))
    visible_passed = sum(1 for result in visible_results if result.get("passed"))
    passed = sum(1 for result in results if result.get("passed"))

    timeout_count = sum(1 for result in results if result.get("status") == "timeout")
    runtime_error_count = sum(1 for result in results if result.get("status") == "runtime_error")
    invalid_output_count = sum(1 for result in results if result.get("status") == "invalid_output_format")
    wrong_answer_count = sum(1 for result in results if result.get("status") == "wrong_answer")
    format_ok_count = sum(1 for result in results if result.get("format_ok", False))

    hidden_pass_rate = hidden_passed / hidden_total if hidden_total else 0.0
    visible_pass_rate = visible_passed / visible_total if visible_total else 0.0
    pass_rate = hidden_pass_rate if hidden_total else (passed / total if total else 0.0)
    format_compliance = format_ok_count / total if total else 0.0

    if not syntax_ok:
        execution_status = "syntax_error"
    elif not safety_ok:
        execution_status = "safety_violation"
    elif precheck_status and precheck_status not in {"ready", "completed"}:
        execution_status = precheck_status
    elif timeout_count:
        execution_status = "timeout"
    elif runtime_error_count:
        execution_status = "runtime_error"
    elif invalid_output_count:
        execution_status = "invalid_output_format"
    elif wrong_answer_count:
        execution_status = "wrong_answer"
    else:
        execution_status = "completed"

    reward = compute_reward(
        pass_rate=pass_rate,
        step_number=step_number,
        execution_status=execution_status,
        format_compliance=format_compliance,
    )

    verifier_components = {
        "hidden_correctness": round(hidden_pass_rate, 4),
        "visible_correctness": round(visible_pass_rate, 4),
        "format_compliance": round(format_compliance, 4),
        "runtime_reliability": round(0.0 if timeout_count or runtime_error_count else 1.0, 4),
        "anti_cheat_compliance": round(1.0 if syntax_ok and safety_ok else 0.0, 4),
        "step_discount": round(step_discount(step_number), 4),
    }

    return reward, {
        "passed": passed,
        "total": total,
        "hidden_passed": hidden_passed,
        "hidden_total": hidden_total,
        "visible_passed": visible_passed,
        "visible_total": visible_total,
        "pass_rate": round(pass_rate, 4),
        "hidden_pass_rate": round(hidden_pass_rate, 4),
        "visible_pass_rate": round(visible_pass_rate, 4),
        "timeout_count": timeout_count,
        "runtime_error_count": runtime_error_count,
        "invalid_output_count": invalid_output_count,
        "wrong_answer_count": wrong_answer_count,
        "format_compliance": round(format_compliance, 4),
        "syntax_valid": bool(syntax_ok),
        "safety_valid": bool(safety_ok),
        "execution_status": execution_status,
        "reward_components": {
            "correctness": round(float(pass_rate), 4),
            "step_discount": round(step_discount(step_number), 4),
            "reward": reward,
        },
        "verifier_components": verifier_components,
    }


def compute_episode_reward(
    *,
    pass_rate: float,
    step_number: int,
    execution_status: str,
    previous_pass_rate: float,
    done: bool,
    efficiency_score: float,
    optimization_target_met: bool,
) -> tuple[float, dict[str, float]]:
    discount = step_discount(step_number)
    clipped_pass_rate = max(0.0, min(float(pass_rate), 1.0))
    clipped_efficiency = max(0.0, min(float(efficiency_score), 1.0))
    progress_delta = max(0.0, clipped_pass_rate - max(0.0, min(float(previous_pass_rate), 1.0)))

    if execution_status in TERMINAL_ZERO_STATUSES:
        reward = 0.0
    elif clipped_pass_rate == 1.0:
        reward = round(discount * (0.6 + 0.4 * clipped_efficiency), 4)
        if not optimization_target_met and not done:
            reward = min(reward, 0.94)
    elif done:
        reward = 0.0
    else:
        reward = round(0.1 * progress_delta, 4)

    return reward, {
        "correctness": round(clipped_pass_rate, 4),
        "efficiency_score": round(clipped_efficiency, 4),
        "step_discount": round(discount, 4),
        "progress_delta": round(progress_delta, 4),
        "reward": round(float(reward), 4),
    }
