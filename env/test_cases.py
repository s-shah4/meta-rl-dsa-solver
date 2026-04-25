from __future__ import annotations

import random
from typing import Any


VISIBLE_TEST_COUNT = 3


PROBLEM_BANK: list[dict[str, Any]] = [
    {
        "problem_id": "easy_double",
        "difficulty": "easy",
        "problem": "Given an integer n, print n * 2.",
        "input_format": "A single integer n.",
        "constraints": "-10^9 <= n <= 10^9",
        "examples": [
            {"input": "2\n", "output": "4"},
            {"input": "5\n", "output": "10"},
        ],
        "test_cases": [
            {"input": "2\n", "output": "4"},
            {"input": "5\n", "output": "10"},
            {"input": "0\n", "output": "0"},
            {"input": "1\n", "output": "2"},
            {"input": "-3\n", "output": "-6"},
            {"input": "10\n", "output": "20"},
            {"input": "999999\n", "output": "1999998"},
            {"input": "-1000000000\n", "output": "-2000000000"},
        ],
    },
    {
        "problem_id": "easy_sum_two",
        "difficulty": "easy",
        "problem": "Given two integers a and b, print their sum.",
        "input_format": "Two space-separated integers a and b.",
        "constraints": "-10^9 <= a, b <= 10^9",
        "examples": [
            {"input": "2 3\n", "output": "5"},
            {"input": "-4 7\n", "output": "3"},
        ],
        "test_cases": [
            {"input": "2 3\n", "output": "5"},
            {"input": "-4 7\n", "output": "3"},
            {"input": "0 0\n", "output": "0"},
            {"input": "1000000000 1\n", "output": "1000000001"},
            {"input": "-8 -9\n", "output": "-17"},
            {"input": "42 -42\n", "output": "0"},
        ],
    },
    {
        "problem_id": "medium_maximum",
        "difficulty": "medium",
        "problem": "Given n integers, print the maximum value.",
        "input_format": "First line contains n. Second line contains n space-separated integers.",
        "constraints": "1 <= n <= 200000; -10^9 <= values <= 10^9",
        "examples": [
            {"input": "5\n1 7 3 2 5\n", "output": "7"},
            {"input": "3\n-5 -2 -9\n", "output": "-2"},
        ],
        "test_cases": [
            {"input": "5\n1 7 3 2 5\n", "output": "7"},
            {"input": "3\n-5 -2 -9\n", "output": "-2"},
            {"input": "1\n42\n", "output": "42"},
            {"input": "6\n0 0 0 0 0 0\n", "output": "0"},
            {"input": "4\n1000000000 -1 5 999999999\n", "output": "1000000000"},
            {"input": "7\n-10 -20 -30 -1 -40 -50 -60\n", "output": "-1"},
        ],
    },
    {
        "problem_id": "medium_count_even",
        "difficulty": "medium",
        "problem": "Given n integers, print how many of them are even.",
        "input_format": "First line contains n. Second line contains n space-separated integers.",
        "constraints": "1 <= n <= 200000; -10^9 <= values <= 10^9",
        "examples": [
            {"input": "5\n1 2 3 4 5\n", "output": "2"},
            {"input": "4\n2 4 6 8\n", "output": "4"},
        ],
        "test_cases": [
            {"input": "5\n1 2 3 4 5\n", "output": "2"},
            {"input": "4\n2 4 6 8\n", "output": "4"},
            {"input": "3\n1 3 5\n", "output": "0"},
            {"input": "1\n0\n", "output": "1"},
            {"input": "6\n-2 -3 -4 -5 -6 -7\n", "output": "3"},
            {"input": "8\n10 11 12 13 14 15 16 17\n", "output": "4"},
        ],
    },
    {
        "problem_id": "hard_reverse_words",
        "difficulty": "hard",
        "problem": "Given a sentence, print its words in reverse order.",
        "input_format": "A single line containing words separated by one or more spaces.",
        "constraints": "1 <= sentence length <= 10000",
        "examples": [
            {"input": "hello world\n", "output": "world hello"},
            {"input": "openenv rewards matter\n", "output": "matter rewards openenv"},
        ],
        "test_cases": [
            {"input": "hello world\n", "output": "world hello"},
            {"input": "openenv rewards matter\n", "output": "matter rewards openenv"},
            {"input": "single\n", "output": "single"},
            {"input": "  trim   extra   spaces  \n", "output": "spaces extra trim"},
            {"input": "a b c d e\n", "output": "e d c b a"},
            {"input": "adaptive dsa tutor\n", "output": "tutor dsa adaptive"},
        ],
    },
]


def load_problem_bank() -> list[dict[str, Any]]:
    return [_copy_problem(problem) for problem in PROBLEM_BANK]


def load_problem(problem_id: str | None = None, difficulty: str | None = None) -> dict[str, Any]:
    problems = load_problem_bank()
    if problem_id is not None:
        matches = [problem for problem in problems if problem["problem_id"] == problem_id]
        if matches:
            return random.choice(matches)

    if difficulty is not None:
        matches = [problem for problem in problems if problem["difficulty"] == difficulty]
        if matches:
            return random.choice(matches)

    return random.choice(problems)


def get_test_cases(problem: dict[str, Any], difficulty: int) -> list[dict[str, str]]:
    base_cases = [dict(test_case) for test_case in problem.get("test_cases", [])]
    if not base_cases:
        return []

    tier = max(1, min(3, int(difficulty)))
    selected = _select_cases_for_tier(base_cases, tier)
    random.shuffle(selected)
    return selected


def split_test_cases(
    test_cases: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return test_cases[:VISIBLE_TEST_COUNT], test_cases[VISIBLE_TEST_COUNT:]


def _copy_problem(problem: dict[str, Any]) -> dict[str, Any]:
    copied = dict(problem)
    copied["examples"] = [dict(example) for example in problem["examples"]]
    copied["test_cases"] = [dict(test_case) for test_case in problem["test_cases"]]
    return copied


def _select_cases_for_tier(
    test_cases: list[dict[str, str]],
    tier: int,
) -> list[dict[str, str]]:
    total = len(test_cases)
    if total <= VISIBLE_TEST_COUNT:
        return [dict(test_case) for test_case in test_cases]

    simple_end = max(1, total // 3)
    moderate_end = max(simple_end + 1, (2 * total) // 3)

    simple_cases = test_cases[:simple_end]
    moderate_cases = test_cases[simple_end:moderate_end]
    stress_cases = test_cases[moderate_end:]

    if tier == 1:
        candidate_pool = simple_cases + _sample_subset(moderate_cases, 1)
        target_size = min(
            len(candidate_pool),
            max(VISIBLE_TEST_COUNT + 1, len(simple_cases)),
        )
        return _sample_subset(candidate_pool, target_size)

    if tier == 2:
        candidate_pool = simple_cases + moderate_cases + _sample_subset(stress_cases, 1)
        target_size = min(
            len(candidate_pool),
            max(VISIBLE_TEST_COUNT + 2, len(simple_cases) + len(moderate_cases)),
        )
        return _sample_subset(candidate_pool, target_size)

    return [dict(test_case) for test_case in test_cases]


def _sample_subset(
    test_cases: list[dict[str, str]],
    count: int,
) -> list[dict[str, str]]:
    if not test_cases or count <= 0:
        return []
    if count >= len(test_cases):
        return [dict(test_case) for test_case in test_cases]
    return [dict(test_case) for test_case in random.sample(test_cases, count)]
