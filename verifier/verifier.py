from __future__ import annotations

from typing import Any

from verifier.complexity import analyze_code_complexity
from verifier.metrics import compute_pass_rate
from verifier.sandbox import run_code, validate_code


def verify(
    code: str,
    test_cases: list[dict[str, Any]] | list[tuple[str, str]],
    *,
    step_number: int = 1,
) -> tuple[float, dict[str, Any]]:
    precheck = validate_code(code)
    complexity = analyze_code_complexity(code)

    if not precheck["syntax_ok"] or not precheck["safety_ok"]:
        reward, metrics = compute_pass_rate(
            [],
            step_number=step_number,
            syntax_ok=bool(precheck["syntax_ok"]),
            safety_ok=bool(precheck["safety_ok"]),
            precheck_status=str(precheck["execution_status"]),
        )
        feedback = _build_feedback(metrics, error=str(precheck["error"]))
        return reward, {
            **metrics,
            **complexity,
            "feedback": feedback,
            "results": [],
            "error": str(precheck["error"]),
        }

    results: list[dict[str, Any]] = []
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
                "duration_ms": execution.get("duration_ms", 0.0),
                "sandboxed": bool(execution.get("sandboxed", False)),
                "sandbox_mode": execution.get("sandbox_mode", "portable"),
                "visibility": "visible" if is_visible else "hidden",
            }
        )

    reward, metrics = compute_pass_rate(results, step_number=step_number)
    feedback = _build_feedback(metrics)
    return reward, {
        **metrics,
        **complexity,
        "feedback": feedback,
        "results": results,
    }


def _build_feedback(metrics: dict[str, Any], *, error: str = "") -> str:
    execution_status = str(metrics.get("execution_status", "unknown"))
    if execution_status == "syntax_error":
        return f"Submission has a syntax error. {error}".strip()
    if execution_status == "safety_violation":
        return f"Submission violated the sandbox policy. {error}".strip()
    if execution_status == "timeout":
        return "Submission timed out on one or more hidden evaluation tests."
    if execution_status == "runtime_error":
        return "Submission raised a runtime error on one or more hidden evaluation tests."
    if execution_status == "invalid_output_format":
        return "Submission completed but produced invalid output format on one or more tests."
    if execution_status == "wrong_answer":
        return "Submission ran successfully but returned an incorrect answer on one or more hidden tests."
    return f"All hidden tests passed. Pass rate: {metrics['pass_rate']:.2f}"
