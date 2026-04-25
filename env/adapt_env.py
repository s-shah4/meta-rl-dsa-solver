from __future__ import annotations

import ast
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from env.executor import run_code
from env.test_cases import get_test_cases, load_problem, split_test_cases
from models import AdaptAction, AdaptObservation, AdaptState


FORBIDDEN_IMPORTS = {"os", "pathlib", "shutil", "socket", "subprocess"}
DIFFICULTY_LABELS = {1: "easy", 2: "medium", 3: "hard"}


class AdaptEnvironment(Environment[AdaptAction, AdaptObservation, AdaptState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = AdaptState(episode_id=str(uuid4()), step_count=0)
        self.problem: dict[str, Any] = {}
        self.test_cases: list[dict[str, str]] = []
        self.visible_tests: list[dict[str, str]] = []
        self.hidden_tests: list[dict[str, str]] = []
        self.last_results: list[dict[str, Any]] = []
        self.history: list[float] = []
        self.max_history = 20
        self.difficulty: int = 1
        self.min_difficulty = 1
        self.max_difficulty = 3

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        problem_id: str | None = None,
        difficulty: str | None = None,
        **_: Any,
    ) -> AdaptObservation:
        del seed
        if difficulty is not None:
            self.difficulty = self._difficulty_to_tier(difficulty)
        elif len(self.history) >= 5:
            success_rate = self._get_success_rate()
            if success_rate > 0.7:
                self.difficulty = min(self.difficulty + 1, self.max_difficulty)
            elif success_rate < 0.3:
                self.difficulty = max(self.difficulty - 1, self.min_difficulty)

        difficulty_label = self._tier_to_difficulty(self.difficulty)
        self.problem = load_problem(problem_id=problem_id, difficulty=difficulty_label)
        self.test_cases = get_test_cases(self.problem, self.difficulty)
        self.visible_tests, self.hidden_tests = split_test_cases(self.test_cases)
        self.last_results = []
        self._state = AdaptState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            problem_id=self.problem["problem_id"],
            difficulty=difficulty_label,
        )
        return self._build_observation(
            reward=0.0,
            done=False,
            feedback="Submit Python code that reads stdin and prints the required answer.",
        )

    def step(
        self,
        action: AdaptAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> AdaptObservation:
        del timeout_s
        if not self.problem:
            self.reset()

        self._state.step_count += 1
        syntax_ok, syntax_error = self._check_syntax(action.code)
        if not syntax_ok:
            observation = self._build_observation(
                reward=0.0,
                done=True,
                feedback=f"Syntax error: {syntax_error}",
                syntax_valid=False,
                execution_status="syntax_error",
            )
            self._update_history(observation.reward)
            self._record_metrics(observation)
            return observation

        safety_ok, safety_error = self._check_safety(action.code)
        if not safety_ok:
            observation = self._build_observation(
                reward=0.0,
                done=True,
                feedback=safety_error,
                syntax_valid=True,
                execution_status="safety_violation",
            )
            self._update_history(observation.reward)
            self._record_metrics(observation)
            return observation

        run_results = self._run_all_tests(action.code)
        self.last_results = run_results
        metrics = self._score_results(run_results)
        verifier_reward, verifier_metadata = self._try_verify(action.code)
        if verifier_reward is not None:
            metrics["reward"] = max(metrics["reward"], verifier_reward)
            if verifier_metadata.get("feedback"):
                metrics["feedback"] = str(verifier_metadata["feedback"])

        observation = self._build_observation(
            reward=metrics["reward"],
            done=True,
            feedback=metrics["feedback"],
            pass_rate=metrics["pass_rate"],
            visible_pass_rate=metrics["visible_pass_rate"],
            hidden_pass_rate=metrics["hidden_pass_rate"],
            syntax_valid=True,
            execution_status=metrics["execution_status"],
            timeout_count=metrics["timeout_count"],
            runtime_error_count=metrics["runtime_error_count"],
            format_compliance=metrics["format_compliance"],
            reward_components=metrics["reward_components"],
        )
        self._update_history(observation.reward)
        self._record_metrics(observation)
        return observation

    @property
    def state(self) -> AdaptState:
        return self._state

    def _build_observation(
        self,
        reward: float,
        done: bool,
        feedback: str,
        pass_rate: float = 0.0,
        visible_pass_rate: float = 0.0,
        hidden_pass_rate: float = 0.0,
        syntax_valid: bool = True,
        execution_status: str = "not_run",
        timeout_count: int = 0,
        runtime_error_count: int = 0,
        format_compliance: float = 0.0,
        reward_components: dict[str, float] | None = None,
    ) -> AdaptObservation:
        return AdaptObservation(
            problem_id=self.problem["problem_id"],
            difficulty=self._tier_to_difficulty(self.difficulty),
            problem=self.problem["problem"],
            input_format=self.problem["input_format"],
            constraints=self.problem["constraints"],
            examples=self.problem["examples"],
            visible_tests=self.visible_tests,
            feedback=feedback,
            pass_rate=pass_rate,
            visible_pass_rate=visible_pass_rate,
            hidden_pass_rate=hidden_pass_rate,
            syntax_valid=syntax_valid,
            execution_status=execution_status,
            timeout_count=timeout_count,
            runtime_error_count=runtime_error_count,
            format_compliance=format_compliance,
            reward_components=reward_components or {},
            reward=round(max(0.0, min(1.0, reward)), 4),
            done=done,
        )

    def _run_all_tests(self, code: str) -> list[dict[str, Any]]:
        results = []
        visible_count = len(self.visible_tests)
        for index, test_case in enumerate(self.test_cases):
            execution = run_code(code, test_case["input"])
            actual = str(execution["stdout"]).strip()
            expected = test_case["output"].strip()
            results.append(
                {
                    "index": index,
                    "split": "visible" if index < visible_count else "hidden",
                    "input": test_case["input"] if index < visible_count else None,
                    "expected": expected if index < visible_count else None,
                    "actual": actual if index < visible_count else None,
                    "stderr": str(execution["stderr"]).strip(),
                    "exit_code": int(execution["exit_code"]),
                    "timed_out": bool(execution.get("timed_out", False)),
                    "passed": execution["exit_code"] == 0 and actual == expected,
                    "format_ok": execution["exit_code"] == 0 and actual != "",
                }
            )
        return results

    def _score_results(self, run_results: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(run_results)
        visible = [result for result in run_results if result["split"] == "visible"]
        hidden = [result for result in run_results if result["split"] == "hidden"]
        pass_rate = self._pass_rate(run_results)
        visible_pass_rate = self._pass_rate(visible)
        hidden_pass_rate = self._pass_rate(hidden)
        timeout_count = sum(1 for result in run_results if result["timed_out"])
        runtime_error_count = sum(
            1
            for result in run_results
            if result["exit_code"] != 0 and not result["timed_out"]
        )
        format_compliance = (
            sum(1 for result in run_results if result["format_ok"]) / total
            if total
            else 0.0
        )
        timeout_rate = timeout_count / total if total else 0.0
        runtime_error_rate = runtime_error_count / total if total else 0.0
        reward_components = {
            "correctness": 0.8 * pass_rate,
            "syntax": 0.05,
            "execution": 0.05 if runtime_error_count == 0 and timeout_count == 0 else 0.0,
            "format": 0.1 * format_compliance,
            "timeout_penalty": -0.2 * timeout_rate,
            "runtime_penalty": -0.1 * runtime_error_rate,
        }
        reward = max(0.0, min(1.0, sum(reward_components.values())))

        if timeout_count:
            status = "timeout"
        elif runtime_error_count:
            status = "runtime_error"
        else:
            status = "completed"

        return {
            "reward": round(reward, 4),
            "feedback": self._build_feedback(run_results, pass_rate),
            "pass_rate": round(pass_rate, 4),
            "visible_pass_rate": round(visible_pass_rate, 4),
            "hidden_pass_rate": round(hidden_pass_rate, 4),
            "timeout_count": timeout_count,
            "runtime_error_count": runtime_error_count,
            "format_compliance": round(format_compliance, 4),
            "execution_status": status,
            "reward_components": {
                key: round(value, 4) for key, value in reward_components.items()
            },
        }

    def _build_feedback(self, run_results: list[dict[str, Any]], pass_rate: float) -> str:
        for result in run_results:
            if result["timed_out"]:
                label = self._safe_test_label(result)
                return f"Timed out on {label}."

            if result["exit_code"] != 0:
                label = self._safe_test_label(result)
                error = result["stderr"] or "runtime error"
                return f"Runtime error on {label}: {error}"

            if not result["passed"] and result["split"] == "visible":
                return (
                    f"Failed on visible input {str(result['input']).strip()}: "
                    f"expected {result['expected']}, got {result['actual']}"
                )

            if not result["passed"]:
                return f"Failed on hidden test {result['index'] + 1}."

        return f"All tests passed. Pass rate: {pass_rate:.2f}"

    def _get_success_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def _update_history(self, reward: float) -> None:
        self.history.append(float(reward))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def _record_metrics(self, observation: AdaptObservation) -> None:
        self._state.last_reward = float(observation.reward or 0.0)
        self._state.last_pass_rate = observation.pass_rate
        self._state.last_feedback = observation.feedback
        self._state.recent_metrics = {
            "difficulty_tier": self.difficulty,
            "difficulty_label": self._tier_to_difficulty(self.difficulty),
            "moving_success_rate": round(self._get_success_rate(), 4),
            "history_size": len(self.history),
            "visible_pass_rate": observation.visible_pass_rate,
            "hidden_pass_rate": observation.hidden_pass_rate,
            "execution_status": observation.execution_status,
            "timeout_count": observation.timeout_count,
            "runtime_error_count": observation.runtime_error_count,
            "format_compliance": observation.format_compliance,
            "reward_components": dict(observation.reward_components),
        }

    def _try_verify(self, code: str) -> tuple[float | None, dict[str, Any]]:
        try:
            from verifier.verifier import verify
        except ImportError:
            return None, {}

        try:
            reward, metadata = verify(code, self.test_cases)
        except Exception as exc:
            return None, {"feedback": f"Verifier unavailable: {exc}"}

        return float(reward), metadata or {}

    def _check_syntax(self, code: str) -> tuple[bool, str]:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, str(exc)
        return True, ""

    def _check_safety(self, code: str) -> tuple[bool, str]:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".", 1)[0]
                    if root_name in FORBIDDEN_IMPORTS:
                        return False, f"Forbidden import: {root_name}"

            if isinstance(node, ast.ImportFrom):
                root_name = (node.module or "").split(".", 1)[0]
                if root_name in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {root_name}"

        return True, ""

    def _pass_rate(self, results: list[dict[str, Any]]) -> float:
        if not results:
            return 0.0
        return sum(1 for result in results if result["passed"]) / len(results)

    def _safe_test_label(self, result: dict[str, Any]) -> str:
        if result["split"] == "visible":
            return f"visible input {str(result['input']).strip()}"
        return f"hidden test {result['index'] + 1}"

    def _tier_to_difficulty(self, tier: int) -> str:
        return DIFFICULTY_LABELS.get(tier, "easy")

    def _difficulty_to_tier(self, difficulty: str) -> int:
        normalized = str(difficulty).strip().lower()
        if normalized.isdigit():
            try:
                return max(
                    self.min_difficulty,
                    min(self.max_difficulty, int(normalized)),
                )
            except ValueError:
                return self.difficulty
        for tier, label in DIFFICULTY_LABELS.items():
            if normalized == label:
                return tier
        return self.difficulty
