from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


TIMEOUT_SECONDS = 2


def run_code(code: str, input_data: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "submission.py"
        file_path.write_text(code, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, str(file_path)],
                input=input_data,
                text=True,
                capture_output=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout or "",
                "stderr": "Execution timed out",
                "exit_code": -1,
            }

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
