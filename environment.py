from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable


class AdaptEnv:
    def __init__(
        self,
        verifier: Callable[[str, str], tuple[float, dict[str, Any]]] | None = None,
    ):
        self.verifier = verifier
        self.problem = ""
        self.current_input = ""
        self.expected_output = ""
        self.step_count = 0
        self.last_output = ""

    def reset(self) -> dict[str, str]:
        self.problem = "Given an integer n, return n * 2"
        self.current_input = "5"
        self.expected_output = "10"
        self.step_count = 0
        self.last_output = ""

        return {
            "problem": self.problem,
            "input": self.current_input,
        }

    def step(self, action: str) -> dict[str, Any]:
        if not self.problem:
            self.reset()

        self.step_count += 1
        output = self._run_code(action)
        reward = self._compute_reward(output)
        self.last_output = output

        return {
            "observation": output,
            "reward": reward,
            "done": True,
            "info": {},
        }

    def _run_code(self, code: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "submission.py"
            file_path.write_text(code, encoding="utf-8")

            try:
                result = subprocess.run(
                    ["python3", str(file_path)],
                    input=self.current_input,
                    text=True,
                    capture_output=True,
                    timeout=2,
                )
            except subprocess.TimeoutExpired:
                return "ERROR: timeout"
            except Exception as exc:
                return f"ERROR: {exc}"

            if result.returncode != 0:
                stderr = result.stderr.strip()
                return f"ERROR: {stderr or 'runtime error'}"

            return result.stdout.strip()

    def _compute_reward(self, output: str) -> float:
        if self.verifier is not None:
            reward, _info = self.verifier(output, self.expected_output)
            return reward

        if output == self.expected_output:
            return 1.0
        return 0.0
