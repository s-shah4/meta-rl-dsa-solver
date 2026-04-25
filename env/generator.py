from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Callable

VISIBLE_TEST_COUNT = 0
MIN_TEST_CASES = 5


@dataclass(frozen=True)
class ProblemTemplate:
    problem_type: str
    difficulty_tier: int
    title: str
    input_format: str
    constraints: str
    statement_builder: Callable[[dict[str, Any]], str]
    solver: Callable[[str], str]
    case_builder: Callable[[random.Random, float], list[str]]


def generator_reward(
    pass_rate: float,
    *,
    diversity_bonus: float = 0.0,
    validity_bonus: float = 0.0,
) -> float:
    """Reward problems that live near the target difficulty sweet spot."""
    clipped = max(0.0, min(1.0, float(pass_rate)))
    target_gap = abs(clipped - 0.5)
    base = max(0.0, 1.0 - (target_gap / 0.5) ** 2)
    reward = base + float(diversity_bonus) + float(validity_bonus)
    return round(max(0.0, min(1.5, reward)), 4)


def validate_problem(problem_dict: dict[str, Any]) -> bool:
    required_keys = {"problem", "input_format", "constraints", "test_cases", "difficulty"}
    if not required_keys.issubset(problem_dict):
        return False

    if any(not str(problem_dict[key]).strip() for key in ("problem", "input_format", "constraints")):
        return False

    difficulty = problem_dict.get("difficulty")
    try:
        difficulty_value = float(difficulty)
    except (TypeError, ValueError):
        return False

    if not 0.0 <= difficulty_value <= 1.0:
        return False

    test_cases = problem_dict.get("test_cases")
    if not isinstance(test_cases, list) or len(test_cases) < MIN_TEST_CASES:
        return False

    seen_inputs: set[str] = set()
    distinct_outputs: set[str] = set()
    visible_count = 0

    for test_case in test_cases:
        if not isinstance(test_case, dict):
            return False

        raw_input = test_case.get("input")
        raw_output = test_case.get("output")
        is_visible = bool(test_case.get("is_visible", False))
        if not isinstance(raw_input, str) or not isinstance(raw_output, str):
            return False
        if not raw_input.endswith("\n"):
            return False
        if raw_input in seen_inputs:
            return False
        seen_inputs.add(raw_input)
        distinct_outputs.add(raw_output.strip())
        visible_count += 1 if is_visible else 0

    if visible_count != VISIBLE_TEST_COUNT:
        return False

    if len(distinct_outputs) < max(3, len(test_cases) // 3):
        return False

    return True


class GeneratorAgent:
    """Deterministic, dependency-free generator for DSA-style problems."""

    def __init__(self, deterministic: bool = True) -> None:
        self.deterministic = deterministic
        self.templates = _build_templates()

    def generate(
        self,
        difficulty_level: int | float | str,
        history: dict[str, Any] | None,
        problem_id: str | None = None,
    ) -> dict[str, Any]:
        history = history or {}
        target_tier = _difficulty_to_tier(difficulty_level)
        adjusted_tier = self._adjust_tier(target_tier, history)
        rng = self._rng_for(adjusted_tier, history, problem_id)
        template = self._choose_template(adjusted_tier, history, rng, forced_problem_type=problem_id)

        for attempt in range(10):
            params = {
                "window": 3 + adjusted_tier,
                "modulus": 10 + 5 * adjusted_tier,
                "max_n": 8 + adjusted_tier * 4,
                "attempt": attempt,
            }
            raw_cases = template.case_builder(rng, 0.2 + adjusted_tier * 0.25)
            test_cases = [
                {
                    "input": case_input,
                    "output": template.solver(case_input),
                    "is_visible": False,
                }
                for case_input in raw_cases
            ]
            signature = self._problem_signature(template.problem_type, test_cases)
            problem = {
                "problem_id": f"{template.problem_type}_{signature[:8]}",
                "problem_type": template.problem_type,
                "difficulty": round(self._tier_to_scalar(adjusted_tier), 4),
                "difficulty_label": DIFFICULTY_LABELS[adjusted_tier],
                "problem": template.statement_builder(params),
                "input_format": template.input_format,
                "constraints": template.constraints,
                "test_cases": test_cases,
                "visible_problem": {
                    "problem": template.statement_builder(params),
                    "input_format": template.input_format,
                    "constraints": template.constraints,
                },
                "generation_mode": "deterministic_fallback" if self.deterministic else "local_rule_based",
                "validity_bonus": 0.15,
            }
            if validate_problem(problem):
                return problem

        raise ValueError(f"Unable to generate a valid problem for template {template.problem_type}")

    def _adjust_tier(self, target_tier: int, history: dict[str, Any]) -> int:
        recent_pass_rates = [float(value) for value in history.get("recent_pass_rates", [])[-5:]]
        if not recent_pass_rates:
            return target_tier

        moving_average = sum(recent_pass_rates) / len(recent_pass_rates)
        if moving_average > 0.8:
            return min(3, target_tier + 1)
        if moving_average < 0.2:
            return max(1, target_tier - 1)
        return target_tier

    def _choose_template(
        self,
        tier: int,
        history: dict[str, Any],
        rng: random.Random,
        forced_problem_type: str | None = None,
    ) -> ProblemTemplate:
        eligible = [template for template in self.templates if template.difficulty_tier == tier]
        if not eligible:
            eligible = list(self.templates)

        if forced_problem_type:
            for template in eligible:
                if template.problem_type == forced_problem_type:
                    return template
            for template in self.templates:
                if template.problem_type == forced_problem_type:
                    return template

        recent_types = list(history.get("problem_types", [])[-4:])
        weighted: list[tuple[float, ProblemTemplate]] = []
        for template in eligible:
            repetition_penalty = 0.35 if template.problem_type in recent_types else 0.0
            jitter = rng.random() * 0.2
            weighted.append((1.0 - repetition_penalty + jitter, template))
        weighted.sort(key=lambda item: item[0], reverse=True)
        return weighted[0][1]

    def _rng_for(
        self,
        tier: int,
        history: dict[str, Any],
        problem_id: str | None,
    ) -> random.Random:
        seed_material = {
            "tier": tier,
            "problem_id": problem_id or "",
            "pass_rates": [round(float(value), 4) for value in history.get("recent_pass_rates", [])[-8:]],
            "problem_types": list(history.get("problem_types", [])[-8:]),
            "episode_index": int(history.get("episode_index", 0)),
        }
        digest = hashlib.sha256(repr(seed_material).encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _problem_signature(self, problem_type: str, test_cases: list[dict[str, str]]) -> str:
        body = "|".join(f"{case['input']}=>{case['output']}" for case in test_cases)
        digest = hashlib.sha256(f"{problem_type}|{body}".encode("utf-8")).hexdigest()
        return digest

    def _tier_to_scalar(self, tier: int) -> float:
        return {1: 0.25, 2: 0.5, 3: 0.75}.get(tier, 0.5)


def _difficulty_to_tier(difficulty_level: int | float | str) -> int:
    if isinstance(difficulty_level, str):
        normalized = difficulty_level.strip().lower()
        if normalized in DIFFICULTY_NAME_TO_TIER:
            return DIFFICULTY_NAME_TO_TIER[normalized]
        try:
            difficulty_level = float(normalized)
        except ValueError:
            return 1

    if isinstance(difficulty_level, float) and difficulty_level <= 1.0:
        if difficulty_level < 0.34:
            return 1
        if difficulty_level < 0.67:
            return 2
        return 3

    try:
        numeric = int(difficulty_level)
    except (TypeError, ValueError):
        return 1
    return max(1, min(3, numeric))


DIFFICULTY_LABELS = {1: "easy", 2: "medium", 3: "hard"}
DIFFICULTY_NAME_TO_TIER = {value: key for key, value in DIFFICULTY_LABELS.items()}


def _build_templates() -> list[ProblemTemplate]:
    return [
        ProblemTemplate(
            problem_type="sum_even_numbers",
            difficulty_tier=1,
            title="Sum Even Numbers",
            input_format="The first line contains n. The second line contains n space-separated integers.",
            constraints="1 <= n <= 12; -100 <= values[i] <= 100",
            statement_builder=lambda _: (
                "Given a list of integers, print the sum of the numbers that are even. "
                "If no number is even, print 0."
            ),
            solver=_solve_sum_even_numbers,
            case_builder=_build_sum_even_cases,
        ),
        ProblemTemplate(
            problem_type="range_span",
            difficulty_tier=1,
            title="Range Span",
            input_format="The first line contains n. The second line contains n space-separated integers.",
            constraints="2 <= n <= 12; -100 <= values[i] <= 100",
            statement_builder=lambda _: (
                "Given a list of integers, print the difference between the maximum and minimum value."
            ),
            solver=_solve_range_span,
            case_builder=_build_range_span_cases,
        ),
        ProblemTemplate(
            problem_type="count_local_peaks",
            difficulty_tier=2,
            title="Count Local Peaks",
            input_format="The first line contains n. The second line contains n space-separated integers.",
            constraints="3 <= n <= 14; -100 <= values[i] <= 100",
            statement_builder=lambda _: (
                "Count how many indices i are local peaks, meaning values[i] is strictly greater than both "
                "values[i-1] and values[i+1]. The first and last element can never be peaks."
            ),
            solver=_solve_count_local_peaks,
            case_builder=_build_peak_cases,
        ),
        ProblemTemplate(
            problem_type="longest_non_decreasing_run",
            difficulty_tier=2,
            title="Longest Non-Decreasing Run",
            input_format="The first line contains n. The second line contains n space-separated integers.",
            constraints="1 <= n <= 16; -100 <= values[i] <= 100",
            statement_builder=lambda _: (
                "Find the length of the longest contiguous subarray whose values are non-decreasing."
            ),
            solver=_solve_longest_non_decreasing_run,
            case_builder=_build_run_cases,
        ),
        ProblemTemplate(
            problem_type="smallest_most_frequent",
            difficulty_tier=3,
            title="Smallest Most Frequent",
            input_format="The first line contains n. The second line contains n space-separated integers.",
            constraints="1 <= n <= 18; -30 <= values[i] <= 30",
            statement_builder=lambda _: (
                "Print the value that appears most often in the array. If several values have the same highest "
                "frequency, print the smallest of them."
            ),
            solver=_solve_smallest_most_frequent,
            case_builder=_build_frequency_cases,
        ),
        ProblemTemplate(
            problem_type="reverse_words",
            difficulty_tier=3,
            title="Reverse Words",
            input_format="A single line containing one or more words separated by spaces.",
            constraints="1 <= line length <= 80",
            statement_builder=lambda _: (
                "Read a line of text and print the words in reverse order. Multiple spaces in the input should "
                "be treated as a single separator."
            ),
            solver=_solve_reverse_words,
            case_builder=_build_reverse_word_cases,
        ),
    ]


def _build_sum_even_cases(rng: random.Random, difficulty_scalar: float) -> list[str]:
    size = 5 + math.ceil(difficulty_scalar * 5)
    cases = set()
    while len(cases) < 6:
        numbers = [rng.randint(-25, 25) for _ in range(size + rng.randint(0, 3))]
        if all(number % 2 for number in numbers):
            numbers[0] = 0
        cases.add(_array_case(numbers))
    return list(cases)


def _build_range_span_cases(rng: random.Random, difficulty_scalar: float) -> list[str]:
    size = 4 + math.ceil(difficulty_scalar * 6)
    cases = set()
    while len(cases) < 6:
        numbers = [rng.randint(-40, 40) for _ in range(size + rng.randint(0, 3))]
        if len(set(numbers)) == 1:
            numbers[-1] += 3
        cases.add(_array_case(numbers))
    return list(cases)


def _build_peak_cases(rng: random.Random, difficulty_scalar: float) -> list[str]:
    size = 5 + math.ceil(difficulty_scalar * 6)
    cases = set()
    while len(cases) < 6:
        numbers = []
        current = rng.randint(-10, 10)
        for index in range(size + rng.randint(0, 4)):
            delta = rng.randint(-6, 6)
            if index % 2 == 1:
                delta = abs(delta) + 1
            current += delta
            numbers.append(current)
        numbers[0] -= 5
        numbers[-1] -= 5
        cases.add(_array_case(numbers))
    return list(cases)


def _build_run_cases(rng: random.Random, difficulty_scalar: float) -> list[str]:
    size = 6 + math.ceil(difficulty_scalar * 6)
    cases = set()
    while len(cases) < 6:
        numbers = [rng.randint(-20, 20)]
        for _ in range(size + rng.randint(0, 4) - 1):
            numbers.append(numbers[-1] + rng.randint(-5, 5))
        cases.add(_array_case(numbers))
    return list(cases)


def _build_frequency_cases(rng: random.Random, difficulty_scalar: float) -> list[str]:
    size = 8 + math.ceil(difficulty_scalar * 6)
    cases = set()
    while len(cases) < 6:
        numbers = [rng.randint(-6, 6) for _ in range(size + rng.randint(0, 5))]
        numbers.extend([rng.choice(numbers), rng.choice(numbers)])
        cases.add(_array_case(numbers))
    return list(cases)


def _build_reverse_word_cases(rng: random.Random, difficulty_scalar: float) -> list[str]:
    vocabulary = [
        "graph",
        "queue",
        "stack",
        "array",
        "tree",
        "hash",
        "search",
        "sort",
        "path",
        "heap",
        "node",
        "edge",
    ]
    word_count = 4 + math.ceil(difficulty_scalar * 4)
    cases = set()
    while len(cases) < 6:
        words = [rng.choice(vocabulary) for _ in range(word_count + rng.randint(0, 2))]
        spacer = " " * rng.randint(1, 3)
        prefix = " " * rng.randint(0, 2)
        suffix = " " * rng.randint(0, 2)
        cases.add(f"{prefix}{spacer.join(words)}{suffix}\n")
    return list(cases)


def _array_case(numbers: list[int]) -> str:
    return f"{len(numbers)}\n{' '.join(str(number) for number in numbers)}\n"


def _solve_sum_even_numbers(stdin: str) -> str:
    _, numbers = _parse_int_array(stdin)
    return str(sum(number for number in numbers if number % 2 == 0))


def _solve_range_span(stdin: str) -> str:
    _, numbers = _parse_int_array(stdin)
    return str(max(numbers) - min(numbers))


def _solve_count_local_peaks(stdin: str) -> str:
    _, numbers = _parse_int_array(stdin)
    peaks = 0
    for index in range(1, len(numbers) - 1):
        if numbers[index] > numbers[index - 1] and numbers[index] > numbers[index + 1]:
            peaks += 1
    return str(peaks)


def _solve_longest_non_decreasing_run(stdin: str) -> str:
    _, numbers = _parse_int_array(stdin)
    best = 1
    current = 1
    for index in range(1, len(numbers)):
        if numbers[index] >= numbers[index - 1]:
            current += 1
        else:
            current = 1
        best = max(best, current)
    return str(best)


def _solve_smallest_most_frequent(stdin: str) -> str:
    _, numbers = _parse_int_array(stdin)
    counts: dict[int, int] = {}
    for number in numbers:
        counts[number] = counts.get(number, 0) + 1
    best_count = max(counts.values())
    best_value = min(number for number, count in counts.items() if count == best_count)
    return str(best_value)


def _solve_reverse_words(stdin: str) -> str:
    words = stdin.strip().split()
    return " ".join(reversed(words))


def _parse_int_array(stdin: str) -> tuple[int, list[int]]:
    lines = [line.strip() for line in stdin.strip().splitlines() if line.strip()]
    n = int(lines[0])
    numbers = [int(part) for part in lines[1].split()]
    if len(numbers) != n:
        raise ValueError(f"Expected {n} integers, received {len(numbers)}")
    return n, numbers
