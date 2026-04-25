from __future__ import annotations

from typing import Any

from verifier.complexity import analyze_code_complexity
from verifier.metrics import compute_pass_rate
from verifier.sandbox import run_code


def verify(code: str, test_cases: list[dict[str, Any]] | list[tuple[str, str]]) -> tuple[float, dict[str, Any]]:
    results: list[dict[str, Any]] = []
    complexity = analyze_code_complexity(code)

    for index, test_case in enumerate(test_cases):
        if isinstance(test_case, dict):
            stdin = str(test_case.get("input", ""))
            expected = str(test_case.get("output", ""))
            is_visible = bool(test_case.get("is_visible", False))
        else:
            stdin, expected = test_case
            is_visible = False

        execution = run_code(code, stdin)
        actual = str(execution.get("stdout", "")).strip()
        stderr = str(execution.get("stderr", "")).strip()
        timed_out = bool(execution.get("timed_out", False))
        exit_code = int(execution.get("exit_code", 0))
        expected_text = expected.strip()
        format_ok = exit_code == 0 and (actual != "" or expected_text == "")

        if timed_out:
            status = "timeout"
        elif exit_code != 0:
            status = "runtime_error"
        elif not format_ok:
            status = "invalid_output_format"
        elif actual != expected_text:
            status = "wrong_answer"
        else:
            status = "passed"

        results.append(
            {
                "index": index,
                "status": status,
                "passed": status == "passed",
                "format_ok": format_ok,
                "stdout": actual if is_visible else "",
                "stderr": stderr,
                "expected": expected_text if is_visible else "",
                "input": stdin if is_visible else "",
                "timed_out": timed_out,
                "exit_code": exit_code,
                "visibility": "visible" if is_visible else "hidden",
            }
        )

    reward, metrics = compute_pass_rate(results)
    feedback = _build_feedback(metrics)

    return reward, {
        **metrics,
        **complexity,
        "feedback": feedback,
        "results": results,
    }


def _build_feedback(metrics: dict[str, Any]) -> str:
    if metrics["execution_status"] == "timeout":
        return "Submission timed out on one or more hidden evaluation tests."
    if metrics["execution_status"] == "runtime_error":
        return "Submission raised a runtime error on one or more hidden evaluation tests."
    if metrics["execution_status"] == "invalid_output_format":
        return "Submission completed but produced invalid output format on one or more tests."
    if metrics["execution_status"] == "wrong_answer":
        return "Submission ran successfully but returned an incorrect answer on one or more hidden tests."
    return f"All hidden tests passed. Pass rate: {metrics['pass_rate']:.2f}"
