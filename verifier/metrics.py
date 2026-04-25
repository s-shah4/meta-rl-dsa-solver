from __future__ import annotations

from typing import Any


def compute_pass_rate(results: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    timeout_count = sum(1 for result in results if result["status"] == "timeout")
    runtime_error_count = sum(1 for result in results if result["status"] == "runtime_error")
    invalid_output_count = sum(1 for result in results if result["status"] == "invalid_output_format")
    wrong_answer_count = sum(1 for result in results if result["status"] == "wrong_answer")
    format_ok_count = sum(1 for result in results if result.get("format_ok", False))

    pass_rate = passed / total if total else 0.0
    format_compliance = format_ok_count / total if total else 0.0
    timeout_rate = timeout_count / total if total else 0.0
    runtime_error_rate = runtime_error_count / total if total else 0.0
    invalid_output_rate = invalid_output_count / total if total else 0.0

    reward_components = {
        "correctness": 0.8 * pass_rate,
        "format": 0.1 * format_compliance,
        "execution": 0.1 if timeout_count == 0 and runtime_error_count == 0 else 0.0,
        "timeout_penalty": -0.2 * timeout_rate,
        "runtime_penalty": -0.1 * runtime_error_rate,
        "invalid_output_penalty": -0.1 * invalid_output_rate,
    }
    reward = max(0.0, min(1.0, sum(reward_components.values())))

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

    return reward, {
        "passed": passed,
        "total": total,
        "pass_rate": round(pass_rate, 4),
        "timeout_count": timeout_count,
        "runtime_error_count": runtime_error_count,
        "invalid_output_count": invalid_output_count,
        "wrong_answer_count": wrong_answer_count,
        "format_compliance": round(format_compliance, 4),
        "execution_status": execution_status,
        "reward_components": {
            key: round(float(value), 4) for key, value in reward_components.items()
        },
    }
