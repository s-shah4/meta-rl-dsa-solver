from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


TIMEOUT_SECONDS = 2


def run_code(code: str, input_data: str) -> dict:
    temp_parent = Path(os.getenv("ADAPT_TMP_DIR", ".adapt_tmp")).resolve()
    temp_parent.mkdir(parents=True, exist_ok=True)
    tmpdir = temp_parent / f"run_{uuid4().hex}"
    tmpdir.mkdir()

    try:
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
                "timed_out": True,
            }

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "timed_out": False,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
