from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import dataset_loader
from env.adapt_env import AdaptEnvironment
from env.generator import GeneratorAgent, ProblemTemplate, validate_problem
from env.test_cases import load_problem_bank
from models import AdaptAction


class FakeBank:
    def __init__(self, problem: dict) -> None:
        self.problem = problem

    def sample(self, difficulty: str, rng, recent_types: list[str]) -> dict:
        del difficulty, rng, recent_types
        return _copy_problem(self.problem)

    def all_problem_ids(self) -> list[str]:
        return [self.problem["problem_id"]]

    def get_by_id(self, problem_id: str) -> dict:
        if problem_id != self.problem["problem_id"]:
            raise KeyError(problem_id)
        return _copy_problem(self.problem)

    def problem_types_for_difficulty(self, difficulty: str) -> list[str]:
        del difficulty
        return [self.problem["problem_type"]]


def _copy_problem(problem: dict) -> dict:
    copied = dict(problem)
    copied["test_cases"] = [dict(test_case) for test_case in problem.get("test_cases", [])]
    copied["visible_problem"] = dict(problem.get("visible_problem", {}))
    examples = copied["visible_problem"].get("examples")
    if isinstance(examples, list):
        copied["visible_problem"]["examples"] = [dict(example) for example in examples]
    return copied


def main() -> None:
    template_problem = GeneratorAgent().generate_problem(1, {}, problem_id="sum_even_numbers")
    dataset_problem = _copy_problem(template_problem)
    dataset_problem["problem_id"] = "cc_stub_sum_even_numbers"
    dataset_problem["generation_mode"] = "dataset"
    dataset_problem["validity_bonus"] = 1.0

    fake_bank = FakeBank(dataset_problem)
    original_bank = dataset_loader._BANK
    original_config = dataset_loader._BANK_CONFIG
    dataset_loader._BANK = fake_bank
    dataset_loader._BANK_CONFIG = ("deepmind/code_contests", "train", 5000)

    try:
        loaded_bank = load_problem_bank(use_dataset=True)
        assert loaded_bank
        assert validate_problem(loaded_bank[0])

        generated = GeneratorAgent(use_dataset=True).generate_problem("easy", {})
        assert generated["problem_id"] == dataset_problem["problem_id"]
        assert generated["generation_mode"] == "dataset"
        assert validate_problem(generated)

        env = AdaptEnvironment(use_dataset=True)
        observation = env.reset(difficulty="easy")
        assert env.problem["generation_mode"] == "dataset"
        assert observation.problem_type == "sum_even_numbers"

        result = env.step(
            AdaptAction(
                code=(
                    "n=int(input())\n"
                    "nums=list(map(int,input().split()))\n"
                    "print(sum(x for x in nums if x % 2 == 0))"
                )
            )
        )
        assert result.pass_rate == 1.0
        assert result.execution_status == "completed"
        assert result.reward > 0.0
        assert 0.0 <= result.reward_components.get("efficiency_score", -1.0) <= 1.0

        injected_env = AdaptEnvironment()
        injected_env.reset(
            difficulty="easy",
            generated_problem=dataset_problem,
            session_id="dataset-smoke",
        )
        injected_result = injected_env.step(
            AdaptAction(
                session_id="dataset-smoke",
                code=(
                    "n=int(input())\n"
                    "nums=list(map(int,input().split()))\n"
                    "print(sum(x for x in nums if x % 2 == 0))"
                )
            )
        )
        assert injected_result.pass_rate == 1.0
        assert injected_result.execution_status == "completed"
        assert injected_result.reward > 0.0
        assert 0.0 <= injected_result.reward_components.get("efficiency_score", -1.0) <= 1.0

        resilient_agent = GeneratorAgent()
        good_template = next(template for template in resilient_agent.templates if template.problem_type == "sum_even_numbers")
        bad_template = ProblemTemplate(
            problem_type="always_bad",
            difficulty_tier=good_template.difficulty_tier,
            title="Always Bad",
            input_format=good_template.input_format,
            constraints=good_template.constraints,
            statement_builder=lambda: "Bad template used for fallback testing.",
            solver=good_template.solver,
            case_builder=lambda rng: ["1\n1\n"] * 10,
        )
        resilient_agent.templates = [bad_template, *resilient_agent.templates]

        with patch.object(resilient_agent, "_choose_template", return_value=bad_template):
            resilient_problem = resilient_agent.generate_problem("easy", {})
        assert resilient_problem["problem_type"] != "always_bad"
        assert validate_problem(resilient_problem)

        print("Dataset mode smoke tests passed")
    finally:
        dataset_loader._BANK = original_bank
        dataset_loader._BANK_CONFIG = original_config


if __name__ == "__main__":
    main()
