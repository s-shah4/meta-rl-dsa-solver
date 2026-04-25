from __future__ import annotations

from typing import Any


def compute_reward(
    pass_rate: float,
    step_number: int,
    execution_status: str,
    format_compliance: float,
) -> float:
    """
    Clean, interpretable reward signal for GRPO training.
    """
    del format_compliance

    step_discount = 1.0 if step_number == 1 else (0.85 if step_number == 2 else 0.70)
    correctness = pass_rate

    if execution_status == "timeout":
        return 0.0
    if execution_status == "syntax_error":
        return 0.0

    reward = correctness * step_discount
    return round(min(max(reward, 0.0), 1.0), 4)


def compute_pass_rate(
    results: list[dict[str, Any]],
    step_number: int = 1,
) -> tuple[float, dict[str, Any]]:
    total = len(results)
    hidden_results = [result for result in results if result.get("visibility") == "hidden"]
    visible_results = [result for result in results if result.get("visibility") == "visible"]

    hidden_total = len(hidden_results)
    visible_total = len(visible_results)

    hidden_passed = sum(1 for result in hidden_results if result["passed"])
    visible_passed = sum(1 for result in visible_results if result["passed"])
    passed = sum(1 for result in results if result["passed"])

    timeout_count = sum(1 for result in results if result["status"] == "timeout")
    runtime_error_count = sum(1 for result in results if result["status"] == "runtime_error")
    invalid_output_count = sum(1 for result in results if result["status"] == "invalid_output_format")
    wrong_answer_count = sum(1 for result in results if result["status"] == "wrong_answer")
    format_ok_count = sum(1 for result in results if result.get("format_ok", False))

    hidden_pass_rate = hidden_passed / hidden_total if hidden_total else 0.0
    visible_pass_rate = visible_passed / visible_total if visible_total else 0.0
    pass_rate = hidden_pass_rate if hidden_total else (passed / total if total else 0.0)
    format_compliance = format_ok_count / total if total else 0.0

    if timeout_count:
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
        "execution_status": execution_status,
        "reward_components": {
            "correctness": round(float(pass_rate), 4),
            "step_discount": 1.0 if step_number == 1 else (0.85 if step_number == 2 else 0.70),
            "reward": reward,
        },
    }
