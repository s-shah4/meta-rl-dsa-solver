from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TIMEOUT_SECONDS = 1.0
MEMORY_LIMIT_MB = 512
OUTPUT_LIMIT_BYTES = 256 * 1024


def _sandbox_env(tmpdir: Path) -> dict[str, str]:
    # Run user code with a tightly scoped environment to reduce hidden state.
    return {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONNOUSERSITE": "1",
        "HOME": str(tmpdir),
        "TMPDIR": str(tmpdir),
        "TEMP": str(tmpdir),
        "TMP": str(tmpdir),
    }


def _linux_preexec_fn(timeout_seconds: float) -> Any:
    if platform.system().lower() != "linux":
        return None

    def _apply_limits() -> None:
        import resource

        memory_limit = MEMORY_LIMIT_MB * 1024 * 1024
        cpu_limit = max(1, int(timeout_seconds) + 1)
        file_limit = OUTPUT_LIMIT_BYTES

        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))

    return _apply_limits


def run_code(code: str, input_data: str, timeout_seconds: int | float | None = None) -> dict[str, Any]:
    timeout_value = float(TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds)
    temp_parent = Path(os.getenv("ADAPT_TMP_DIR", ".adapt_tmp")).resolve()
    temp_parent.mkdir(parents=True, exist_ok=True)

    tmpdir_path = Path(tempfile.mkdtemp(prefix="run_", dir=str(temp_parent)))
    submission_path = tmpdir_path / "submission.py"
    submission_path.write_text(code, encoding="utf-8")

    started = time.perf_counter()
    try:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(submission_path)],
                input=input_data,
                text=True,
                capture_output=True,
                timeout=timeout_value,
                cwd=str(tmpdir_path),
                env=_sandbox_env(tmpdir_path),
                preexec_fn=_linux_preexec_fn(timeout_value),
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            return {
                "stdout": str(exc.stdout or ""),
                "stderr": "Execution timed out",
                "exit_code": -1,
                "timed_out": True,
                "duration_ms": duration_ms,
                "sandboxed": True,
                "sandbox_mode": "linux_limited" if platform.system().lower() == "linux" else "portable",
            }

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        stdout = result.stdout[:OUTPUT_LIMIT_BYTES]
        stderr = result.stderr[:OUTPUT_LIMIT_BYTES]
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": int(result.returncode),
            "timed_out": False,
            "duration_ms": duration_ms,
            "sandboxed": True,
            "sandbox_mode": "linux_limited" if platform.system().lower() == "linux" else "portable",
        }
    finally:
        shutil.rmtree(tmpdir_path, ignore_errors=True)
