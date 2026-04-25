from __future__ import annotations

import ast
from typing import Any, Generic, TypeVar
from uuid import uuid4

from env.generator import DIFFICULTY_LABELS, GeneratorAgent, generator_reward, validate_problem
from models import AdaptAction, AdaptObservation, AdaptState
from verifier.metrics import compute_reward

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
MAX_STEPS_PER_EPISODE = 3


class AdaptEnvironment(Environment[AdaptAction, AdaptObservation, AdaptState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        generator: GeneratorAgent | None = None,
        generator_mode: str = "heuristic",
        session_id: str | None = None,
    ) -> None:
        super().__init__()
        self.generator = generator or GeneratorAgent()
        self.generator_mode = generator_mode
        self.session_id = session_id or str(uuid4())
        self.problem: dict[str, Any] = {}
        self.test_cases: list[dict[str, Any]] = []
        self.last_results: list[dict[str, Any]] = []
        self.max_history = 50
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
        self.attempt_history: list[dict[str, Any]] = []
        self.previous_execution_status = "ready"
        self.episode_done = False
        self._state = AdaptState(
            session_id=self.session_id,
            episode_id=str(uuid4()),
            step_count=0,
            generator_mode=self.generator_mode,
            max_steps=MAX_STEPS_PER_EPISODE,
            history={"attempts": []},
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        problem_id: str | None = None,
        difficulty: str | None = None,
        generated_problem: dict[str, Any] | None = None,
        generator_mode: str | None = None,
        session_id: str | None = None,
        family_weights: dict[str, float] | None = None,
        **_: Any,
    ) -> AdaptObservation:
        del seed

        if session_id:
            self.session_id = session_id
        if generator_mode is not None:
            self.generator_mode = generator_mode
        if difficulty is not None:
            self.difficulty = self._difficulty_to_tier(difficulty)
        elif generated_problem is not None:
            generated_label = str(generated_problem.get("difficulty_label", "")).strip().lower()
            if generated_label:
                self.difficulty = self._difficulty_to_tier(generated_label)

        self.problem = self._load_problem(
            generated_problem=generated_problem,
            problem_id=problem_id,
            family_weights=family_weights,
        )
        self.test_cases = [dict(test_case) for test_case in self.problem["test_cases"]]
        self.last_results = []
        self.attempt_history = []
        self.previous_execution_status = "ready"
        self.episode_done = False
        self._state = AdaptState(
            session_id=self.session_id,
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            problem_id=self.problem["problem_id"],
            problem_type=self.problem.get("problem_type", ""),
            difficulty=self.problem.get("difficulty_label", self._tier_to_difficulty(self.difficulty)),
            generator_mode=self.generator_mode,
            max_steps=MAX_STEPS_PER_EPISODE,
            generated_problem=self._public_problem_view(),
            history={"attempts": []},
        )
        return self._build_observation(
            reward=0.0,
            done=False,
            feedback=(
                "You have up to 3 attempts. Submit Python code that reads stdin and prints the required answer. "
                "Use the examples to infer the expected behavior."
            ),
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
            self.reset(session_id=action.session_id or self.session_id)

        if self.episode_done:
            return self._build_observation(
                reward=float(self._state.last_reward or 0.0),
                done=True,
                feedback="This episode is finished. Call reset() to start a new problem.",
                pass_rate=float(self._state.last_pass_rate or 0.0),
                visible_pass_rate=float(self._state.recent_metrics.get("visible_pass_rate", 0.0)),
                hidden_pass_rate=float(self._state.last_pass_rate or 0.0),
                syntax_valid=self._state.last_execution_status != "syntax_error",
                execution_status=self._state.last_execution_status or "completed",
                timeout_count=int(self._state.recent_metrics.get("timeout_count", 0)),
                runtime_error_count=int(self._state.recent_metrics.get("runtime_error_count", 0)),
                invalid_output_count=int(self._state.recent_metrics.get("invalid_output_count", 0)),
                wrong_answer_count=int(self._state.recent_metrics.get("wrong_answer_count", 0)),
                format_compliance=float(self._state.recent_metrics.get("format_compliance", 0.0)),
                reward_components=dict(self._state.recent_metrics.get("reward_components", {})),
                generator_reward_signal=float(self._state.generator_reward_signal or 0.0),
            )

        self._state.step_count += 1
        attempt_number = self._state.step_count
        previous_status = self.previous_execution_status
        previous_pass_rate = float(self._state.last_pass_rate or 0.0)

        syntax_ok, syntax_error = self._check_syntax(action.code)
        if not syntax_ok:
            done = attempt_number >= MAX_STEPS_PER_EPISODE
            observation = self._build_observation(
                reward=0.0,
                done=done,
                feedback=self._format_static_feedback(
                    attempt_number=attempt_number,
                    previous_status=previous_status,
                    execution_status="syntax_error",
                    details=f"Syntax error: {syntax_error}",
                ),
                syntax_valid=False,
                execution_status="syntax_error",
                reward_components={
                    "correctness": 0.0,
                    "step_discount": 1.0 if attempt_number == 1 else (0.85 if attempt_number == 2 else 0.70),
                    "progress_delta": 0.0,
                },
            )
            self.last_results = []
            self.previous_execution_status = observation.execution_status
            self._record_metrics(observation)
            if done:
                self._finalize_episode(observation)
            return observation

        safety_ok, safety_error = self._check_safety(action.code)
        if not safety_ok:
            done = attempt_number >= MAX_STEPS_PER_EPISODE
            observation = self._build_observation(
                reward=0.0,
                done=done,
                feedback=self._format_static_feedback(
                    attempt_number=attempt_number,
                    previous_status=previous_status,
                    execution_status="safety_violation",
                    details=safety_error,
                ),
                syntax_valid=True,
                execution_status="safety_violation",
                reward_components={
                    "correctness": 0.0,
                    "step_discount": 1.0 if attempt_number == 1 else (0.85 if attempt_number == 2 else 0.70),
                    "progress_delta": 0.0,
                },
            )
            self.last_results = []
            self.previous_execution_status = observation.execution_status
            self._record_metrics(observation)
            if done:
                self._finalize_episode(observation)
            return observation

        _, metadata = self._verify_submission(action.code)
        self.last_results = list(metadata.get("results", []))
        hidden_pass_rate = float(metadata.get("hidden_pass_rate", metadata.get("pass_rate", 0.0)))
        visible_pass_rate = float(metadata.get("visible_pass_rate", 0.0))
        execution_status = str(metadata.get("execution_status", "completed"))
        done = hidden_pass_rate == 1.0 or attempt_number >= MAX_STEPS_PER_EPISODE
        reward, reward_components = self._shape_reward(
            pass_rate=hidden_pass_rate,
            step_number=attempt_number,
            execution_status=execution_status,
            previous_pass_rate=previous_pass_rate,
            done=done,
        )
        feedback = self._format_feedback(
            results=self.last_results,
            attempt_number=attempt_number,
            previous_status=previous_status,
            execution_status=execution_status,
            hidden_pass_rate=hidden_pass_rate,
            visible_pass_rate=visible_pass_rate,
        )
        observation = self._build_observation(
            reward=reward,
            done=done,
            feedback=feedback,
            pass_rate=hidden_pass_rate,
            visible_pass_rate=visible_pass_rate,
            hidden_pass_rate=hidden_pass_rate,
            syntax_valid=True,
            execution_status=execution_status,
            timeout_count=int(metadata.get("timeout_count", 0)),
            runtime_error_count=int(metadata.get("runtime_error_count", 0)),
            invalid_output_count=int(metadata.get("invalid_output_count", 0)),
            wrong_answer_count=int(metadata.get("wrong_answer_count", 0)),
            format_compliance=float(metadata.get("format_compliance", 0.0)),
            reward_components=reward_components,
            generator_reward_signal=float(metadata.get("generator_reward", 0.0)),
        )
        self.previous_execution_status = observation.execution_status
        self._record_metrics(observation)
        if done:
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
            session_id=self.session_id,
            problem_id=self.problem.get("problem_id", ""),
            problem_type=self.problem.get("problem_type", ""),
            difficulty=self.problem.get("difficulty_label", self._tier_to_difficulty(self.difficulty)),
            attempt_number=self._state.step_count,
            max_steps=MAX_STEPS_PER_EPISODE,
            problem=public_problem.get("problem", ""),
            input_format=public_problem.get("input_format", ""),
            constraints=public_problem.get("constraints", ""),
            feedback=feedback,
            pass_rate=round(float(pass_rate), 4),
            visible_pass_rate=round(float(visible_pass_rate), 4),
            hidden_pass_rate=round(float(hidden_pass_rate), 4),
            syntax_valid=syntax_valid,
            execution_status=execution_status,
            timeout_count=timeout_count,
            runtime_error_count=runtime_error_count,
            invalid_output_count=invalid_output_count,
            wrong_answer_count=wrong_answer_count,
            format_compliance=round(float(format_compliance), 4),
            reward_components=reward_components or {},
            generator_reward_signal=round(float(generator_reward_signal), 4),
            reward=round(max(0.0, min(1.0, reward)), 4),
            done=done,
        )

    def _load_problem(
        self,
        generated_problem: dict[str, Any] | None,
        problem_id: str | None,
        family_weights: dict[str, float] | None,
    ) -> dict[str, Any]:
        candidate = generated_problem or self.generator.generate_problem(
            self.difficulty,
            self.history,
            problem_id=problem_id,
            family_weights=family_weights,
        )
        if validate_problem(candidate):
            return candidate
        fallback = self.generator.generate_problem(
            self.difficulty,
            self.history,
            problem_id=problem_id,
            family_weights=family_weights,
        )
        if not validate_problem(fallback):
            raise ValueError("Generator produced an invalid problem twice in a row.")
        return fallback

    def _verify_submission(self, code: str) -> tuple[float, dict[str, Any]]:
        try:
            from verifier.verifier import verify
        except ImportError as exc:
            return 0.0, {
                "feedback": f"Verifier unavailable: {exc}",
                "execution_status": "verifier_error",
                "results": [],
            }

        try:
            reward, metadata = verify(code, self.test_cases)
        except Exception as exc:
            return 0.0, {
                "feedback": f"Verifier crashed: {exc}",
                "execution_status": "verifier_error",
                "results": [],
            }

        metadata = dict(metadata or {})
        diversity_bonus = self._diversity_bonus(self.problem.get("problem_type", ""))
        validity_bonus = float(self.problem.get("validity_bonus", 0.0))
        hidden_pass_rate = float(metadata.get("hidden_pass_rate", metadata.get("pass_rate", 0.0)))
        metadata["generator_reward"] = generator_reward(
            hidden_pass_rate,
            diversity_bonus=diversity_bonus,
            validity_bonus=validity_bonus,
        )
        return float(reward), metadata

    def _shape_reward(
        self,
        pass_rate: float,
        step_number: int,
        execution_status: str,
        previous_pass_rate: float,
        done: bool,
    ) -> tuple[float, dict[str, float]]:
        step_discount = 1.0 if step_number == 1 else (0.85 if step_number == 2 else 0.70)
        progress_delta = max(0.0, float(pass_rate) - float(previous_pass_rate))

        if execution_status in {"timeout", "syntax_error", "safety_violation"}:
            reward = 0.0
        elif pass_rate == 1.0:
            reward = compute_reward(
                pass_rate=pass_rate,
                step_number=step_number,
                execution_status=execution_status,
                format_compliance=0.0,
            )
        elif done:
            reward = 0.0
        else:
            reward = round(0.1 * progress_delta, 4)

        return reward, {
            "correctness": round(float(pass_rate), 4),
            "step_discount": round(step_discount, 4),
            "progress_delta": round(progress_delta, 4),
            "reward": round(float(reward), 4),
        }

    def _format_feedback(
        self,
        results: list[dict[str, Any]],
        attempt_number: int,
        previous_status: str,
        execution_status: str,
        hidden_pass_rate: float,
        visible_pass_rate: float,
    ) -> str:
        lines = [
            f"Attempt {attempt_number}/{MAX_STEPS_PER_EPISODE}.",
            f"Previous attempt status: {previous_status}.",
            f"Current execution status: {execution_status}.",
            f"Hidden pass rate: {hidden_pass_rate:.2f}. Visible pass rate: {visible_pass_rate:.2f}.",
        ]

        failed_tests = self._summarize_failed_tests(results)
        if failed_tests:
            lines.append("Failed tests:")
            lines.extend(failed_tests)
        elif hidden_pass_rate == 1.0:
            lines.append("All hidden tests passed.")
        else:
            lines.append("No failing test details were available.")

        return "\n".join(lines)

    def _format_static_feedback(
        self,
        attempt_number: int,
        previous_status: str,
        execution_status: str,
        details: str,
    ) -> str:
        return "\n".join(
            [
                f"Attempt {attempt_number}/{MAX_STEPS_PER_EPISODE}.",
                f"Previous attempt status: {previous_status}.",
                f"Current execution status: {execution_status}.",
                details,
            ]
        )

    def _summarize_failed_tests(self, results: list[dict[str, Any]]) -> list[str]:
        summaries: list[str] = []
        for result in results:
            if result.get("passed", False):
                continue
            visibility = str(result.get("visibility", "hidden"))
            label = f"{visibility.title()} test #{int(result.get('index', 0)) + 1}"
            status = str(result.get("status", "unknown"))
            if visibility == "visible":
                actual = str(result.get("stdout", "")).strip()
                expected = str(result.get("expected", "")).strip()
                details = []
                if expected:
                    details.append(f"expected={expected}")
                if actual:
                    details.append(f"got={actual}")
                if result.get("stderr"):
                    details.append("stderr_present")
                if details:
                    summaries.append(f"- {label}: {status} ({', '.join(details)})")
                else:
                    summaries.append(f"- {label}: {status}")
            else:
                summaries.append(f"- {label}: {status}")
        return summaries

    def _record_metrics(self, observation: AdaptObservation) -> None:
        attempt_record = {
            "attempt_number": observation.attempt_number,
            "reward": float(observation.reward or 0.0),
            "pass_rate": float(observation.pass_rate),
            "visible_pass_rate": float(observation.visible_pass_rate),
            "execution_status": observation.execution_status,
            "feedback": observation.feedback,
            "done": bool(observation.done),
        }
        self.attempt_history.append(attempt_record)
        self._state.last_reward = float(observation.reward or 0.0)
        self._state.last_pass_rate = observation.pass_rate
        self._state.last_feedback = observation.feedback
        self._state.last_execution_status = observation.execution_status
        self._state.generator_reward_signal = observation.generator_reward_signal
        self._state.history = {"attempts": list(self.attempt_history)}
        self._state.generated_problem = self._public_problem_view()
        self._state.recent_metrics = {
            "difficulty_tier": self.difficulty,
            "difficulty_label": self.problem.get("difficulty_label", self._tier_to_difficulty(self.difficulty)),
            "visible_pass_rate": observation.visible_pass_rate,
            "pass_rate": observation.pass_rate,
            "execution_status": observation.execution_status,
            "timeout_count": observation.timeout_count,
            "runtime_error_count": observation.runtime_error_count,
            "invalid_output_count": observation.invalid_output_count,
            "wrong_answer_count": observation.wrong_answer_count,
            "format_compliance": observation.format_compliance,
            "reward_components": dict(observation.reward_components),
        }

    def _finalize_episode(self, observation: AdaptObservation) -> None:
        self.episode_done = True
        self._update_history(observation.pass_rate, observation.generator_reward_signal)

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

    def _public_problem_view(self) -> dict[str, str]:
        visible = dict(self.problem.get("visible_problem", {}))
        base_problem = visible.get("problem", self.problem.get("problem", ""))
        examples = self._format_examples()
        if examples:
            base_problem = f"{base_problem}\n\nExamples:\n{examples}"
        return {
            "problem": base_problem,
            "input_format": visible.get("input_format", self.problem.get("input_format", "")),
            "constraints": visible.get("constraints", self.problem.get("constraints", "")),
        }

    def _format_examples(self) -> str:
        visible_cases = [test_case for test_case in self.test_cases if test_case.get("is_visible", False)]
        if not visible_cases:
            return ""
        chunks = []
        for test_case in visible_cases:
            chunks.append(
                f"Input:\n{test_case['input']}Expected Output:\n{test_case['output']}\n"
            )
        return "\n".join(chunks).rstrip()

    def _diversity_bonus(self, problem_type: str) -> float:
        recent_types = list(self.history.get("problem_types", [])[-6:])
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
