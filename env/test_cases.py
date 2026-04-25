from __future__ import annotations

from typing import Any

from env.generator import DIFFICULTY_LABELS, GeneratorAgent, VISIBLE_TEST_COUNT


def load_problem_bank() -> list[dict[str, Any]]:
    generator = GeneratorAgent()
    bank = []
    for template in generator.templates:
        generated = generator.generate(template.difficulty_tier, {}, problem_id=template.problem_type)
        bank.append(
            {
                "problem_id": template.problem_type,
                "difficulty": DIFFICULTY_LABELS[template.difficulty_tier],
                "problem": generated["problem"],
                "input_format": generated["input_format"],
                "constraints": generated["constraints"],
                "test_cases": [dict(test_case) for test_case in generated["test_cases"]],
            }
        )
    return bank


def load_problem(problem_id: str | None = None, difficulty: str | None = None) -> dict[str, Any]:
    generator = GeneratorAgent()
    history: dict[str, Any] = {}
    if problem_id is not None:
        for template in generator.templates:
            if template.problem_type == problem_id:
                return generator.generate(template.difficulty_tier, history, problem_id=problem_id)

    tier = 1
    if difficulty is not None:
        normalized = str(difficulty).strip().lower()
        for candidate_tier, label in DIFFICULTY_LABELS.items():
            if normalized == label:
                tier = candidate_tier
                break
    return generator.generate(tier, history)


def get_test_cases(problem: dict[str, Any], difficulty: int) -> list[dict[str, str]]:
    del difficulty
    return [dict(test_case) for test_case in problem.get("test_cases", [])]


def split_test_cases(
    test_cases: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    visible = [dict(test_case) for test_case in test_cases if test_case.get("is_visible", False)]
    hidden = [dict(test_case) for test_case in test_cases if not test_case.get("is_visible", False)]
    return visible, hidden
