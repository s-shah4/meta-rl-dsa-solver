from __future__ import annotations

from env.executor import run_code as execute_submission


def run_code(code: str, stdin: str, timeout: int = 1) -> dict[str, object]:
    return execute_submission(code, stdin, timeout_seconds=timeout)
