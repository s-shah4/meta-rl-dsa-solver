from __future__ import annotations

from typing import Any

from env.executor import run_code
from env.test_cases import VISIBLE_TEST_COUNT, load_problem, load_test_cases


class AdaptEnv:
    def __init__(self) -> None:
        self.problem: dict[str, Any] = {}
        self.test_cases: list[dict[str, str]] = []
        self.visible_tests: list[dict[str, str]] = []
        self.hidden_tests: list[dict[str, str]] = []
        self.step_count = 0

    def reset(self) -> dict:
        self.problem = self._load_problem()
        self.test_cases = load_test_cases()
        self.visible_tests, self.hidden_tests = self._split_test_cases(self.test_cases)
        self.step_count = 0
        return self._build_observation()

    def step(self, code: str) -> dict:
        if not self.test_cases:
            self.reset()

        self.step_count += 1
        run_results = self._run_all_tests(code)
        reward, metadata = self._verify_code(code)
        metadata = metadata or {}

        pass_rate = float(metadata.get("pass_rate", self._compute_pass_rate(run_results)))
        feedback = str(metadata.get("feedback") or self._build_feedback(run_results, pass_rate))

        return {
            "reward": float(reward),
            "done": True,
            "feedback": feedback,
            "pass_rate": pass_rate,
        }

    def _load_problem(self) -> dict:
        return load_problem()

    def _split_test_cases(
        self,
        test_cases: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        visible_tests = test_cases[:VISIBLE_TEST_COUNT]
        hidden_tests = test_cases[VISIBLE_TEST_COUNT:]
        return visible_tests, hidden_tests

    def _build_observation(self) -> dict:
        return {
            "problem": self.problem["problem"],
            "input_format": self.problem["input_format"],
            "constraints": self.problem["constraints"],
            "examples": self.problem["examples"],
            "visible_tests": self.visible_tests,
        }

    def _run_all_tests(self, code: str) -> list[dict[str, Any]]:
        results = []
        for test_case in self.test_cases:
            execution = run_code(code, test_case["input"])
            actual = execution["stdout"].strip()
            expected = test_case["output"].strip()
            results.append(
                {
                    "input": test_case["input"],
                    "expected": expected,
                    "actual": actual,
                    "stderr": execution["stderr"].strip(),
                    "exit_code": execution["exit_code"],
                    "passed": execution["exit_code"] == 0 and actual == expected,
                }
            )
        return results

    def _verify_code(self, code: str) -> tuple[float, dict[str, Any]]:
        from verifier.verifier import verify

        return verify(code, self.test_cases)

    def _compute_pass_rate(self, run_results: list[dict[str, Any]]) -> float:
        if not run_results:
            return 0.0
        passed = sum(1 for result in run_results if result["passed"])
        return passed / len(run_results)

    def _build_feedback(self, run_results: list[dict[str, Any]], pass_rate: float) -> str:
        for result in run_results:
            if result["exit_code"] != 0:
                error = result["stderr"] or "runtime error"
                return f"Runtime error on input {result['input'].strip()}: {error}"

            if not result["passed"]:
                return (
                    f"Failed on input {result['input'].strip()}: "
                    f"expected {result['expected']}, got {result['actual']}"
                )

        return f"All tests passed. Pass rate: {pass_rate:.2f}"
