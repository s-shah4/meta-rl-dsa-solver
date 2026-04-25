import subprocess
import tempfile
import os


def run_code(code: str, stdin: str, timeout: int = 2):
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name

        result = subprocess.run(
            ["python3", path],
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

        os.remove(path)

        if result.returncode != 0:
            return False, result.stderr.strip()

        return True, result.stdout.strip()

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"

    except Exception as e:
        return False, f"ERROR: {e}"