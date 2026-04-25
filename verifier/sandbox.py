import os
import subprocess
import sys
import tempfile


def run_code(code: str, stdin: str, timeout: int = 2):
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name

        result = subprocess.run(
            [sys.executable, path],
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return False, result.stderr.strip()

        return True, result.stdout.strip()

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"

    except Exception as e:
        return False, f"ERROR: {e}"

    finally:
        if path and os.path.exists(path):
            os.remove(path)
