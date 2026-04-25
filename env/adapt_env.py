from __future__ import annotations

import ast
from typing import Any, Generic, TypeVar
from uuid import uuid4

from env.generator import DIFFICULTY_LABELS, GeneratorAgent, generator_reward, validate_problem
from models import AdaptAction, AdaptObservation, AdaptState

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    ActionT = TypeVar("ActionT")
    ObservationT = TypeVar("ObservationT")
    StateT = TypeVar("StateT")

    class Environment(Generic[ActionT, ObservationT, StateT]):
        SUPPORTS_CONCURRENT_SESSIONS = False

        def __init__(self) -> None:
            pass


FORBIDDEN_IMPORTS = {"os", "pathlib", "shutil", "socket", "subprocess"}


class AdaptEnvironment(Environment[AdaptAction, AdaptObservation, AdaptState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        generator: GeneratorAgent | None = None,
        generator_mode: str = "heuristic",
    ) -> None:
        super().__init__()
        self.generator = generator or GeneratorAgent()
        self.generator_mode = generator_mode
        self.problem: dict[str, Any] = {}
        self.test_cases: list[dict[str, str]] = []
        self.last_results: list[dict[str, Any]] = []
        self.max_history = 20
        self.min_difficulty = 1
        self.max_difficulty = 3
        self.difficulty: int = 1
        self.history: dict[str, Any] = {
            "recent_pass_rates": [],
            "problem_types": [],
            "generator_rewards": [],
            "problem_signatures": [],
            "episode_index": 0,
        }
        self._state = AdaptState(episode_id=str(uuid4()), step_count=0, generator_mode=self.generator_mode)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        problem_id: str | None = None,
        difficulty: str | None = None,
        generated_problem: dict[str, Any] | None = None,
        generator_mode: str | None = None,
        **_: Any,
    ) -> AdaptObservation:
        del seed
        if generator_mode is not None:
            self.generator_mode = generator_mode
        if difficulty is not None:
            self.difficulty = self._difficulty_to_tier(difficulty)
        elif self.history["recent_pass_rates"]:
            self.difficulty = self._recommend_next_difficulty()

        self.problem = self._load_problem(
            generated_problem=generated_problem,
            problem_id=problem_id,
        )
        self.test_cases = [dict(test_case) for test_case in self.problem["test_cases"]]
        self.last_results = []
        self._state = AdaptState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            problem_id=self.problem["problem_id"],
            problem_type=self.problem.get("problem_type", ""),
            difficulty=self.problem.get("difficulty_label", self._tier_to_difficulty(self.difficulty)),
            generator_mode=self.generator_mode,
            generated_problem=self._public_problem_view(),
        )
        return self._build_observation(
            reward=0.0,
            done=False,
            feedback="Submit Python code that reads stdin and prints the required answer.",
            execution_status="ready",
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
                reward_components={"correctness": 0.0, "format": 0.0},
            )
            self._finalize_episode(observation)
            return observation

        safety_ok, safety_error = self._check_safety(action.code)
        if not safety_ok:
            observation = self._build_observation(
                reward=0.0,
                done=True,
                feedback=safety_error,
                syntax_valid=True,
                execution_status="safety_violation",
                reward_components={"correctness": 0.0, "format": 0.0},
            )
            self._finalize_episode(observation)
            return observation

        reward, metadata = self._verify_submission(action.code)
        self.last_results = list(metadata.get("results", []))
        observation = self._build_observation(
            reward=reward,
            done=True,
            feedback=str(metadata.get("feedback", "Evaluation complete.")),
            pass_rate=float(metadata.get("pass_rate", 0.0)),
            visible_pass_rate=0.0,
            hidden_pass_rate=float(metadata.get("pass_rate", 0.0)),
            syntax_valid=True,
            execution_status=str(metadata.get("execution_status", "completed")),
            timeout_count=int(metadata.get("timeout_count", 0)),
            runtime_error_count=int(metadata.get("runtime_error_count", 0)),
            invalid_output_count=int(metadata.get("invalid_output_count", 0)),
            wrong_answer_count=int(metadata.get("wrong_answer_count", 0)),
            format_compliance=float(metadata.get("format_compliance", 0.0)),
            reward_components={
                key: round(float(value), 4)
                for key, value in dict(metadata.get("reward_components", {})).items()
            },
            generator_reward_signal=float(metadata.get("generator_reward", 0.0)),
        )
        self._finalize_episode(observation)
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
        invalid_output_count: int = 0,
        wrong_answer_count: int = 0,
        format_compliance: float = 0.0,
        reward_components: dict[str, float] | None = None,
        generator_reward_signal: float = 0.0,
    ) -> AdaptObservation:
        public_problem = self._public_problem_view()
        return AdaptObservation(
            problem_id=self.problem.get("problem_id", ""),
            problem_type=self.problem.get("problem_type", ""),
            difficulty=self.problem.get("difficulty_label", self._tier_to_difficulty(self.difficulty)),
            problem=public_problem.get("problem", ""),
            input_format=public_problem.get("input_format", ""),
            constraints=public_problem.get("constraints", ""),
            feedback=feedback,
            pass_rate=pass_rate,
            visible_pass_rate=visible_pass_rate,
            hidden_pass_rate=hidden_pass_rate,
            syntax_valid=syntax_valid,
            execution_status=execution_status,
            timeout_count=timeout_count,
            runtime_error_count=runtime_error_count,
            invalid_output_count=invalid_output_count,
            wrong_answer_count=wrong_answer_count,
            format_compliance=format_compliance,
            reward_components=reward_components or {},
            generator_reward_signal=round(float(generator_reward_signal), 4),
            reward=round(max(0.0, min(1.0, reward)), 4),
            done=done,
        )

    def _load_problem(
        self,
        generated_problem: dict[str, Any] | None,
        problem_id: str | None,
    ) -> dict[str, Any]:
        candidate = generated_problem or self.generator.generate(
            self.difficulty,
            self.history,
            problem_id=problem_id,
        )
        if validate_problem(candidate):
            return candidate
        fallback = self.generator.generate(self.difficulty, self.history, problem_id=problem_id)
        if not validate_problem(fallback):
            raise ValueError("Generator produced an invalid problem twice in a row.")
        return fallback

    def _verify_submission(self, code: str) -> tuple[float, dict[str, Any]]:
        try:
            from verifier.verifier import verify
        except ImportError as exc:
            return 0.0, {"feedback": f"Verifier unavailable: {exc}", "execution_status": "verifier_error"}

        try:
            reward, metadata = verify(code, self.test_cases)
        except Exception as exc:
            return 0.0, {"feedback": f"Verifier crashed: {exc}", "execution_status": "verifier_error"}

        metadata = dict(metadata or {})
        diversity_bonus = self._diversity_bonus(self.problem.get("problem_type", ""))
        validity_bonus = float(self.problem.get("validity_bonus", 0.0))
        metadata["generator_reward"] = generator_reward(
            float(metadata.get("pass_rate", 0.0)),
            diversity_bonus=diversity_bonus,
            validity_bonus=validity_bonus,
        )
        return float(reward), metadata

    def _finalize_episode(self, observation: AdaptObservation) -> None:
        self._update_history(observation.pass_rate, observation.generator_reward_signal)
        self._record_metrics(observation)

    def _update_history(self, pass_rate: float, generator_signal: float) -> None:
        self.history["recent_pass_rates"].append(round(float(pass_rate), 4))
        self.history["problem_types"].append(self.problem.get("problem_type", ""))
        self.history["problem_signatures"].append(self.problem.get("problem_id", ""))
        self.history["generator_rewards"].append(round(float(generator_signal), 4))
        self.history["episode_index"] = int(self.history.get("episode_index", 0)) + 1

        for key in ("recent_pass_rates", "problem_types", "problem_signatures", "generator_rewards"):
            values = self.history[key]
            if len(values) > self.max_history:
                del values[:-self.max_history]

    def _record_metrics(self, observation: AdaptObservation) -> None:
        self._state.last_reward = float(observation.reward or 0.0)
        self._state.last_pass_rate = observation.pass_rate
        self._state.last_feedback = observation.feedback
        self._state.generator_reward_signal = observation.generator_reward_signal
        self._state.history = {
            "recent_pass_rates": list(self.history["recent_pass_rates"]),
            "problem_types": list(self.history["problem_types"]),
            "generator_rewards": list(self.history["generator_rewards"]),
        }
        self._state.recent_metrics = {
            "difficulty_tier": self.difficulty,
            "difficulty_label": self.problem.get("difficulty_label", self._tier_to_difficulty(self.difficulty)),
            "history_size": len(self.history["recent_pass_rates"]),
            "pass_rate": observation.pass_rate,
            "execution_status": observation.execution_status,
            "timeout_count": observation.timeout_count,
            "runtime_error_count": observation.runtime_error_count,
            "invalid_output_count": observation.invalid_output_count,
            "wrong_answer_count": observation.wrong_answer_count,
            "format_compliance": observation.format_compliance,
            "reward_components": dict(observation.reward_components),
        }

    def _recommend_next_difficulty(self) -> int:
        recent = [float(value) for value in self.history["recent_pass_rates"][-5:]]
        if not recent:
            return self.difficulty
        moving_average = sum(recent) / len(recent)
        if moving_average > 0.75:
            return min(self.max_difficulty, self.difficulty + 1)
        if moving_average < 0.25:
            return max(self.min_difficulty, self.difficulty - 1)
        return self.difficulty

    def _public_problem_view(self) -> dict[str, str]:
        visible = dict(self.problem.get("visible_problem", {}))
        return {
            "problem": visible.get("problem", self.problem.get("problem", "")),
            "input_format": visible.get("input_format", self.problem.get("input_format", "")),
            "constraints": visible.get("constraints", self.problem.get("constraints", "")),
        }

    def _diversity_bonus(self, problem_type: str) -> float:
        recent_types = list(self.history.get("problem_types", [])[-4:])
        if not recent_types:
            return 0.1
        if problem_type in recent_types:
            return 0.0
        return 0.1

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

    def _tier_to_difficulty(self, tier: int) -> str:
        return DIFFICULTY_LABELS.get(tier, "easy")

    def _difficulty_to_tier(self, difficulty: str) -> int:
        normalized = str(difficulty).strip().lower()
        if normalized.isdigit():
            try:
                return max(self.min_difficulty, min(self.max_difficulty, int(normalized)))
            except ValueError:
                return self.difficulty
        for tier, label in DIFFICULTY_LABELS.items():
            if normalized == label:
                return tier
        try:
            numeric = float(normalized)
        except ValueError:
            return self.difficulty
        if numeric < 0.34:
            return 1
        if numeric < 0.67:
            return 2
        return 3
