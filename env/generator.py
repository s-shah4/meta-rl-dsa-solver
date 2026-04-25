from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable

VISIBLE_TEST_COUNT = 2
HIDDEN_TEST_COUNT = 8
TOTAL_TEST_CASES = VISIBLE_TEST_COUNT + HIDDEN_TEST_COUNT
MIN_TEST_CASES = TOTAL_TEST_CASES


@dataclass(frozen=True)
class ProblemTemplate:
    problem_type: str
    difficulty_tier: int
    title: str
    input_format: str
    constraints: str
    statement_builder: Callable[[], str]
    solver: Callable[[str], str]
    case_builder: Callable[[random.Random], list[str]]


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
    if not isinstance(test_cases, list) or len(test_cases) != TOTAL_TEST_CASES:
        return False

    seen_inputs: set[str] = set()
    distinct_outputs: set[str] = set()
    visible_count = 0
    hidden_count = 0

    for index, test_case in enumerate(test_cases):
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

        if index < VISIBLE_TEST_COUNT and not is_visible:
            return False
        if index >= VISIBLE_TEST_COUNT and is_visible:
            return False

        if is_visible:
            visible_count += 1
        else:
            hidden_count += 1

    if visible_count != VISIBLE_TEST_COUNT or hidden_count != HIDDEN_TEST_COUNT:
        return False

    normalized_outputs = {output.strip().lower() for output in distinct_outputs}
    min_output_diversity = 2 if normalized_outputs.issubset({"yes", "no", "true", "false", "0", "1"}) else max(
        3,
        len(test_cases) // 3,
    )
    if len(distinct_outputs) < min_output_diversity:
        return False

    return True


class GeneratorAgent:
    """Deterministic, dependency-free generator for DSA-style problems."""

    def __init__(self, deterministic: bool = True) -> None:
        self.deterministic = deterministic
        self.templates = _build_templates()

    def generate_problem(
        self,
        difficulty_level: int | float | str,
        history: dict[str, Any] | None,
        problem_id: str | None = None,
        family_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        history = history or {}
        target_tier = _difficulty_to_tier(difficulty_level)
        rng = self._rng_for(target_tier, history, problem_id, family_weights or {})
        template = self._choose_template(
            target_tier,
            history,
            rng,
            forced_problem_type=problem_id,
            family_weights=family_weights or {},
        )

        for _ in range(20):
            raw_cases = template.case_builder(rng)
            test_cases = [
                {
                    "input": case_input,
                    "output": template.solver(case_input),
                    "is_visible": index < VISIBLE_TEST_COUNT,
                }
                for index, case_input in enumerate(raw_cases)
            ]
            signature = self._problem_signature(template.problem_type, test_cases)
            problem = {
                "problem_id": f"{template.problem_type}_{signature[:8]}",
                "problem_type": template.problem_type,
                "difficulty": round(self._tier_to_scalar(target_tier), 4),
                "difficulty_label": DIFFICULTY_LABELS[target_tier],
                "problem": template.statement_builder(),
                "input_format": template.input_format,
                "constraints": template.constraints,
                "test_cases": test_cases,
                "visible_problem": {
                    "problem": template.statement_builder(),
                    "input_format": template.input_format,
                    "constraints": template.constraints,
                },
                "generation_mode": "deterministic_fallback" if self.deterministic else "local_rule_based",
                "validity_bonus": 0.15,
            }
            if validate_problem(problem):
                return problem

        raise ValueError(f"Unable to generate a valid problem for template {template.problem_type}")

    def generate(
        self,
        difficulty_level: int | float | str,
        history: dict[str, Any] | None,
        problem_id: str | None = None,
        family_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return self.generate_problem(
            difficulty_level=difficulty_level,
            history=history,
            problem_id=problem_id,
            family_weights=family_weights,
        )

    def _choose_template(
        self,
        tier: int,
        history: dict[str, Any],
        rng: random.Random,
        forced_problem_type: str | None = None,
        family_weights: dict[str, float] | None = None,
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

        recent_types = list(history.get("problem_types", [])[-6:])
        weights: list[float] = []
        for template in eligible:
            base_weight = float((family_weights or {}).get(template.problem_type, 1.0))
            base_weight = max(base_weight, 1e-6)
            if template.problem_type in recent_types:
                base_weight *= 0.35
            weights.append(base_weight)

        return rng.choices(eligible, weights=weights, k=1)[0]

    def _rng_for(
        self,
        tier: int,
        history: dict[str, Any],
        problem_id: str | None,
        family_weights: dict[str, float],
    ) -> random.Random:
        if not self.deterministic:
            return random.Random()

        seed_material = {
            "tier": tier,
            "problem_id": problem_id or "",
            "pass_rates": [round(float(value), 4) for value in history.get("recent_pass_rates", [])[-8:]],
            "problem_types": list(history.get("problem_types", [])[-8:]),
            "episode_index": int(history.get("episode_index", 0)),
            "family_weights": {
                key: round(float(value), 4)
                for key, value in sorted(family_weights.items())
            },
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
            statement_builder=lambda: (
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
            statement_builder=lambda: (
                "Given a list of integers, print the difference between the maximum and minimum value."
            ),
            solver=_solve_range_span,
            case_builder=_build_range_span_cases,
        ),
        ProblemTemplate(
            problem_type="count_vowels",
            difficulty_tier=1,
            title="Count Vowels",
            input_format="A single line containing lowercase or uppercase letters and spaces.",
            constraints="1 <= line length <= 80",
            statement_builder=lambda: (
                "Count how many vowels appear in the input line. Treat a, e, i, o, u as vowels "
                "and ignore case."
            ),
            solver=_solve_count_vowels,
            case_builder=_build_count_vowels_cases,
        ),
        ProblemTemplate(
            problem_type="max_consecutive_ones",
            difficulty_tier=1,
            title="Max Consecutive Ones",
            input_format="A single line containing a binary string.",
            constraints="1 <= string length <= 40",
            statement_builder=lambda: (
                "Print the length of the longest contiguous block of '1' characters in the binary string."
            ),
            solver=_solve_max_consecutive_ones,
            case_builder=_build_max_consecutive_ones_cases,
        ),
        ProblemTemplate(
            problem_type="fizzbuzz_variant",
            difficulty_tier=1,
            title="FizzBuzz Variant",
            input_format="The first line contains n a b. The second line contains label_a and label_b.",
            constraints="1 <= n <= 25; 2 <= a, b <= 9; labels contain only letters",
            statement_builder=lambda: (
                "For each integer from 1 to n, print label_a if the number is divisible by a, "
                "label_b if it is divisible by b, and the concatenation label_a+label_b if it is divisible "
                "by both. Otherwise print the number itself. Output all tokens on one line separated by spaces."
            ),
            solver=_solve_fizzbuzz_variant,
            case_builder=_build_fizzbuzz_variant_cases,
        ),
        ProblemTemplate(
            problem_type="running_total",
            difficulty_tier=1,
            title="Running Total",
            input_format="The first line contains n. The second line contains n space-separated integers.",
            constraints="1 <= n <= 14; -50 <= values[i] <= 50",
            statement_builder=lambda: (
                "Print the running total after each element of the array. Output the cumulative sums on one line "
                "separated by spaces."
            ),
            solver=_solve_running_total,
            case_builder=_build_running_total_cases,
        ),
        ProblemTemplate(
            problem_type="count_local_peaks",
            difficulty_tier=2,
            title="Count Local Peaks",
            input_format="The first line contains n. The second line contains n space-separated integers.",
            constraints="3 <= n <= 16; -100 <= values[i] <= 100",
            statement_builder=lambda: (
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
            constraints="1 <= n <= 18; -100 <= values[i] <= 100",
            statement_builder=lambda: (
                "Find the length of the longest contiguous subarray whose values are non-decreasing."
            ),
            solver=_solve_longest_non_decreasing_run,
            case_builder=_build_run_cases,
        ),
        ProblemTemplate(
            problem_type="two_sum_count",
            difficulty_tier=2,
            title="Two Sum Count",
            input_format="The first line contains n and target. The second line contains n space-separated integers.",
            constraints="2 <= n <= 16; -50 <= values[i] <= 50",
            statement_builder=lambda: (
                "Count how many index pairs (i, j) with i < j have values[i] + values[j] equal to target."
            ),
            solver=_solve_two_sum_count,
            case_builder=_build_two_sum_count_cases,
        ),
        ProblemTemplate(
            problem_type="max_subarray_sum",
            difficulty_tier=2,
            title="Maximum Subarray Sum",
            input_format="The first line contains n. The second line contains n space-separated integers.",
            constraints="1 <= n <= 18; -50 <= values[i] <= 50",
            statement_builder=lambda: (
                "Print the maximum possible sum of a contiguous subarray."
            ),
            solver=_solve_max_subarray_sum,
            case_builder=_build_max_subarray_sum_cases,
        ),
        ProblemTemplate(
            problem_type="group_anagrams_count",
            difficulty_tier=2,
            title="Group Anagrams Count",
            input_format="The first line contains n. The second line contains n space-separated lowercase words.",
            constraints="1 <= n <= 12; each word length is between 1 and 8",
            statement_builder=lambda: (
                "Group words that are anagrams of each other. Print the number of distinct anagram groups."
            ),
            solver=_solve_group_anagrams_count,
            case_builder=_build_group_anagrams_cases,
        ),
        ProblemTemplate(
            problem_type="balanced_brackets",
            difficulty_tier=2,
            title="Balanced Brackets",
            input_format="A single line containing only the characters ()[]{}.",
            constraints="1 <= line length <= 50",
            statement_builder=lambda: (
                "Print YES if the bracket string is balanced and NO otherwise."
            ),
            solver=_solve_balanced_brackets,
            case_builder=_build_balanced_brackets_cases,
        ),
        ProblemTemplate(
            problem_type="matrix_diagonal_sum",
            difficulty_tier=2,
            title="Matrix Diagonal Sum",
            input_format="The first line contains n. The next n lines each contain n space-separated integers.",
            constraints="2 <= n <= 6; -20 <= matrix[i][j] <= 20",
            statement_builder=lambda: (
                "For the square matrix, print the sum of the primary diagonal and secondary diagonal. "
                "If n is odd, count the center element only once."
            ),
            solver=_solve_matrix_diagonal_sum,
            case_builder=_build_matrix_diagonal_sum_cases,
        ),
        ProblemTemplate(
            problem_type="smallest_most_frequent",
            difficulty_tier=3,
            title="Smallest Most Frequent",
            input_format="The first line contains n. The second line contains n space-separated integers.",
            constraints="1 <= n <= 20; -30 <= values[i] <= 30",
            statement_builder=lambda: (
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
            constraints="1 <= line length <= 120",
            statement_builder=lambda: (
                "Read a line of text and print the words in reverse order. Multiple spaces in the input should "
                "be treated as a single separator."
            ),
            solver=_solve_reverse_words,
            case_builder=_build_reverse_word_cases,
        ),
        ProblemTemplate(
            problem_type="longest_common_subsequence",
            difficulty_tier=3,
            title="Longest Common Subsequence",
            input_format="The first line contains string s. The second line contains string t.",
            constraints="1 <= len(s), len(t) <= 18; strings contain lowercase letters",
            statement_builder=lambda: (
                "Print the length of the longest common subsequence of the two strings."
            ),
            solver=_solve_longest_common_subsequence,
            case_builder=_build_lcs_cases,
        ),
        ProblemTemplate(
            problem_type="word_ladder_steps",
            difficulty_tier=3,
            title="Word Ladder Steps",
            input_format="The first line contains start and target. The second line contains n. The third line contains n space-separated words.",
            constraints="All words have the same length between 3 and 5; 1 <= n <= 14",
            statement_builder=lambda: (
                "You may change one character at a time. Every intermediate word and the target word must appear "
                "in the given word list. Print the minimum number of single-character changes needed to transform "
                "start into target, or -1 if it is impossible."
            ),
            solver=_solve_word_ladder_steps,
            case_builder=_build_word_ladder_cases,
        ),
        ProblemTemplate(
            problem_type="merge_intervals",
            difficulty_tier=3,
            title="Merge Intervals",
            input_format="The first line contains n. The next n lines each contain start and end.",
            constraints="1 <= n <= 12; -20 <= start <= end <= 30",
            statement_builder=lambda: (
                "Merge all overlapping intervals and print how many intervals remain after merging."
            ),
            solver=_solve_merge_intervals,
            case_builder=_build_merge_intervals_cases,
        ),
        ProblemTemplate(
            problem_type="min_coins",
            difficulty_tier=3,
            title="Minimum Coins",
            input_format="The first line contains n and target. The second line contains n distinct positive coin values.",
            constraints="1 <= n <= 8; 1 <= target <= 40; 1 <= coin values <= 20",
            statement_builder=lambda: (
                "Print the minimum number of coins needed to make exactly target using unlimited copies of the given "
                "coin values. Print -1 if it is impossible."
            ),
            solver=_solve_min_coins,
            case_builder=_build_min_coins_cases,
        ),
        ProblemTemplate(
            problem_type="rotate_matrix_90",
            difficulty_tier=3,
            title="Rotate Matrix 90 Degrees",
            input_format="The first line contains n. The next n lines each contain n space-separated integers.",
            constraints="2 <= n <= 5; -20 <= matrix[i][j] <= 20",
            statement_builder=lambda: (
                "Rotate the square matrix 90 degrees clockwise and print the rotated matrix flattened in row-major "
                "order on one line separated by spaces."
            ),
            solver=_solve_rotate_matrix_90,
            case_builder=_build_rotate_matrix_cases,
        ),
    ]


def _build_sum_even_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _array_case([2, 3, 4]),
        _array_case([1, 3, 5, 7]),
        _array_case([0, -2, 5, 8]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _sum_even_hidden_case)


def _sum_even_hidden_case(rng: random.Random) -> str:
    length = rng.randint(5, 12)
    numbers = [rng.randint(-50, 50) for _ in range(length)]
    if all(number % 2 for number in numbers):
        numbers[rng.randrange(length)] = rng.choice([-8, -2, 0, 6, 14])
    return _array_case(numbers)


def _build_range_span_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _array_case([1, 4, 9]),
        _array_case([-2, -2, -2, 1]),
        _array_case([8, 3]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _range_span_hidden_case)


def _range_span_hidden_case(rng: random.Random) -> str:
    length = rng.randint(4, 12)
    numbers = [rng.randint(-60, 60) for _ in range(length)]
    if len(set(numbers)) == 1:
        numbers[-1] += 5
    return _array_case(numbers)


def _build_count_vowels_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        "hello world\n",
        "sky\n",
        "AEIOU\n",
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _count_vowels_hidden_case)


def _count_vowels_hidden_case(rng: random.Random) -> str:
    word_bank = [
        "algorithm",
        "queue",
        "stack",
        "binary",
        "graph",
        "open env",
        "unit test",
        "dynamic programming",
        "vowel heavy area",
        "crypt rhythm",
    ]
    parts = [rng.choice(word_bank) for _ in range(rng.randint(1, 3))]
    text = " ".join(parts)
    return f"{text[:80]}\n"


def _build_max_consecutive_ones_cases(rng: random.Random) -> list[str]:
    visible_pool = ["1101110\n", "00000\n", "1\n"]
    return _cases_from_pool_and_factory(rng, visible_pool, _max_consecutive_ones_hidden_case)


def _max_consecutive_ones_hidden_case(rng: random.Random) -> str:
    length = rng.randint(8, 40)
    chars = [rng.choice(["0", "1"]) for _ in range(length)]
    if "1" not in chars:
        start = rng.randint(0, max(0, length - 3))
        run_length = rng.randint(1, min(5, length - start))
        for index in range(start, start + run_length):
            chars[index] = "1"
    return f"{''.join(chars)}\n"


def _build_fizzbuzz_variant_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _fizzbuzz_case(8, 3, 5, "Fizz", "Buzz"),
        _fizzbuzz_case(6, 2, 4, "Hop", "Pop"),
        _fizzbuzz_case(10, 2, 3, "Up", "Go"),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _fizzbuzz_hidden_case)


def _fizzbuzz_hidden_case(rng: random.Random) -> str:
    labels = [
        ("Fizz", "Buzz"),
        ("Ping", "Pong"),
        ("Hop", "Skip"),
        ("Alpha", "Beta"),
        ("Red", "Blue"),
    ]
    label_a, label_b = rng.choice(labels)
    a = rng.randint(2, 6)
    b = rng.randint(2, 6)
    while b == a:
        b = rng.randint(2, 6)
    n = rng.randint(10, 25)
    return _fizzbuzz_case(n, a, b, label_a, label_b)


def _build_running_total_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _array_case([1, 2, 3, 4]),
        _array_case([5, -2, 7]),
        _array_case([0, 0, 1]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _running_total_hidden_case)


def _running_total_hidden_case(rng: random.Random) -> str:
    length = rng.randint(5, 14)
    numbers = [rng.randint(-20, 20) for _ in range(length)]
    return _array_case(numbers)


def _build_peak_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _array_case([1, 3, 2, 4, 1]),
        _array_case([5, 4, 3, 2, 1]),
        _array_case([2, 5, 1, 5, 2]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _peak_hidden_case)


def _peak_hidden_case(rng: random.Random) -> str:
    length = rng.randint(6, 16)
    numbers = [rng.randint(-20, 20)]
    for _ in range(length - 1):
        numbers.append(numbers[-1] + rng.randint(-8, 8))
    return _array_case(numbers)


def _build_run_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _array_case([1, 2, 2, 1, 3]),
        _array_case([5, 4, 3, 2]),
        _array_case([1, 1, 1, 1]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _run_hidden_case)


def _run_hidden_case(rng: random.Random) -> str:
    length = rng.randint(6, 18)
    numbers = [rng.randint(-20, 20)]
    for _ in range(length - 1):
        numbers.append(numbers[-1] + rng.randint(-6, 6))
    return _array_case(numbers)


def _build_two_sum_count_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _target_array_case(5, [1, 2, 3, 4]),
        _target_array_case(2, [1, 1, 1, 1]),
        _target_array_case(0, [-1, 1, 2, -2]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _two_sum_hidden_case)


def _two_sum_hidden_case(rng: random.Random) -> str:
    length = rng.randint(5, 16)
    numbers = [rng.randint(-12, 12) for _ in range(length)]
    target = rng.randint(-10, 10)
    return _target_array_case(target, numbers)


def _build_max_subarray_sum_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _array_case([1, -2, 3, 4, -1]),
        _array_case([-5, -1, -8]),
        _array_case([2, -1, 2, 3, 4, -5]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _max_subarray_hidden_case)


def _max_subarray_hidden_case(rng: random.Random) -> str:
    length = rng.randint(6, 18)
    numbers = [rng.randint(-20, 20) for _ in range(length)]
    return _array_case(numbers)


def _build_group_anagrams_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _word_list_case(["eat", "tea", "tan", "ate", "nat", "bat"]),
        _word_list_case(["abc", "bca", "cab", "foo"]),
        _word_list_case(["a", "b", "ab", "ba"]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _group_anagrams_hidden_case)


def _group_anagrams_hidden_case(rng: random.Random) -> str:
    base_words = ["stone", "tones", "notes", "silent", "listen", "enlist", "rat", "tar", "art"]
    words: list[str] = []
    for _ in range(rng.randint(4, 10)):
        word = rng.choice(base_words)
        if rng.random() < 0.4:
            shuffled = list(word)
            rng.shuffle(shuffled)
            words.append("".join(shuffled))
        else:
            words.append(word)
    return _word_list_case(words)


def _build_balanced_brackets_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        "([]{})\n",
        "([)]\n",
        "{[()]}\n",
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _balanced_brackets_hidden_case)


def _balanced_brackets_hidden_case(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return f"{_make_balanced_brackets(rng, rng.randint(3, 10))}\n"
    return f"{_make_unbalanced_brackets(rng, rng.randint(3, 10))}\n"


def _build_matrix_diagonal_sum_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _matrix_case([[1, 2], [3, 4]]),
        _matrix_case([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        _matrix_case([[2, 0, 2], [1, 5, 1], [2, 0, 2]]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _matrix_diagonal_hidden_case)


def _matrix_diagonal_hidden_case(rng: random.Random) -> str:
    size = rng.randint(3, 6)
    matrix = [[rng.randint(-9, 9) for _ in range(size)] for _ in range(size)]
    return _matrix_case(matrix)


def _build_frequency_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _array_case([1, 2, 2, 3, 3, 3]),
        _array_case([4, 4, 1, 1]),
        _array_case([-1, -1, -2, -2, -2, 3]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _frequency_hidden_case)


def _frequency_hidden_case(rng: random.Random) -> str:
    length = rng.randint(8, 20)
    numbers = [rng.randint(-8, 8) for _ in range(length)]
    numbers.extend([rng.choice(numbers), rng.choice(numbers)])
    return _array_case(numbers)


def _build_reverse_word_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        "hello world here\n",
        "  graph   search tree \n",
        "one\n",
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _reverse_words_hidden_case)


def _reverse_words_hidden_case(rng: random.Random) -> str:
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
    words = [rng.choice(vocabulary) for _ in range(rng.randint(4, 9))]
    spacer = " " * rng.randint(1, 3)
    prefix = " " * rng.randint(0, 2)
    suffix = " " * rng.randint(0, 2)
    return f"{prefix}{spacer.join(words)}{suffix}\n"


def _build_lcs_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _two_line_case("abcde", "ace"),
        _two_line_case("abc", "abc"),
        _two_line_case("abc", "def"),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _lcs_hidden_case)


def _lcs_hidden_case(rng: random.Random) -> str:
    alphabet = "abcdxyz"
    left = "".join(rng.choice(alphabet) for _ in range(rng.randint(6, 14)))
    right = "".join(rng.choice(alphabet) for _ in range(rng.randint(6, 14)))
    return _two_line_case(left, right)


def _build_word_ladder_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _word_ladder_case("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]),
        _word_ladder_case("same", "same", ["same", "lame", "came"]),
        _word_ladder_case("cold", "warm", ["cord", "card", "ward", "sold"]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _word_ladder_hidden_case)


def _word_ladder_hidden_case(rng: random.Random) -> str:
    length = rng.randint(3, 5)
    if rng.random() < 0.7:
        path_length = rng.randint(2, 5)
        path = _build_word_ladder_path(rng, length, path_length)
        extras = _build_word_ladder_extras(rng, length, rng.randint(2, 7), set(path))
        words = path[1:] + extras
        rng.shuffle(words)
        return _word_ladder_case(path[0], path[-1], words)

    start = _random_word(rng, length)
    target = _random_word(rng, length)
    while target == start:
        target = _random_word(rng, length)
    extras = _build_word_ladder_extras(rng, length, rng.randint(4, 10), {start, target})
    return _word_ladder_case(start, target, extras)


def _build_merge_intervals_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _interval_case([(1, 3), (2, 4), (6, 8)]),
        _interval_case([(1, 2), (3, 4), (5, 6)]),
        _interval_case([(0, 5), (2, 3), (4, 10)]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _merge_intervals_hidden_case)


def _merge_intervals_hidden_case(rng: random.Random) -> str:
    intervals = []
    for _ in range(rng.randint(4, 12)):
        start = rng.randint(-10, 20)
        end = start + rng.randint(0, 8)
        intervals.append((start, end))
    return _interval_case(intervals)


def _build_min_coins_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _coin_case([1, 3, 4], 6),
        _coin_case([2, 5], 3),
        _coin_case([2, 5, 7], 14),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _min_coins_hidden_case)


def _min_coins_hidden_case(rng: random.Random) -> str:
    coin_count = rng.randint(2, 6)
    coins = sorted({rng.randint(1, 10) for _ in range(coin_count + 2)})
    coins = coins[:coin_count]
    target = rng.randint(5, 40)
    return _coin_case(coins, target)


def _build_rotate_matrix_cases(rng: random.Random) -> list[str]:
    visible_pool = [
        _matrix_case([[1, 2], [3, 4]]),
        _matrix_case([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        _matrix_case([[5, 1], [0, -1]]),
    ]
    return _cases_from_pool_and_factory(rng, visible_pool, _rotate_matrix_hidden_case)


def _rotate_matrix_hidden_case(rng: random.Random) -> str:
    size = rng.randint(2, 5)
    matrix = [[rng.randint(-9, 9) for _ in range(size)] for _ in range(size)]
    return _matrix_case(matrix)


def _cases_from_pool_and_factory(
    rng: random.Random,
    visible_pool: list[str],
    hidden_factory: Callable[[random.Random], str],
) -> list[str]:
    cases: list[str] = []
    seen: set[str] = set()

    for case_input in rng.sample(visible_pool, k=VISIBLE_TEST_COUNT):
        cases.append(case_input)
        seen.add(case_input)

    attempts = 0
    while len(cases) < TOTAL_TEST_CASES:
        candidate = hidden_factory(rng)
        attempts += 1
        if candidate in seen:
            if attempts > 200:
                raise ValueError("Unable to generate unique test cases.")
            continue
        seen.add(candidate)
        cases.append(candidate)

    return cases


def _array_case(numbers: list[int]) -> str:
    return f"{len(numbers)}\n{' '.join(str(number) for number in numbers)}\n"


def _target_array_case(target: int, numbers: list[int]) -> str:
    return f"{len(numbers)} {target}\n{' '.join(str(number) for number in numbers)}\n"


def _word_list_case(words: list[str]) -> str:
    return f"{len(words)}\n{' '.join(words)}\n"


def _matrix_case(matrix: list[list[int]]) -> str:
    rows = [" ".join(str(value) for value in row) for row in matrix]
    return f"{len(matrix)}\n" + "\n".join(rows) + "\n"


def _two_line_case(first: str, second: str) -> str:
    return f"{first}\n{second}\n"


def _interval_case(intervals: list[tuple[int, int]]) -> str:
    rows = [f"{start} {end}" for start, end in intervals]
    return f"{len(intervals)}\n" + "\n".join(rows) + "\n"


def _coin_case(coins: list[int], target: int) -> str:
    return f"{len(coins)} {target}\n{' '.join(str(coin) for coin in coins)}\n"


def _fizzbuzz_case(n: int, a: int, b: int, label_a: str, label_b: str) -> str:
    return f"{n} {a} {b}\n{label_a} {label_b}\n"


def _word_ladder_case(start: str, target: str, words: list[str]) -> str:
    return f"{start} {target}\n{len(words)}\n{' '.join(words)}\n"


def _solve_sum_even_numbers(stdin: str) -> str:
    _, numbers = _parse_int_array(stdin)
    return str(sum(number for number in numbers if number % 2 == 0))


def _solve_range_span(stdin: str) -> str:
    _, numbers = _parse_int_array(stdin)
    return str(max(numbers) - min(numbers))


def _solve_count_vowels(stdin: str) -> str:
    text = stdin.rstrip("\n")
    return str(sum(1 for char in text.lower() if char in "aeiou"))


def _solve_max_consecutive_ones(stdin: str) -> str:
    binary = stdin.strip()
    best = 0
    current = 0
    for char in binary:
        if char == "1":
            current += 1
            best = max(best, current)
        else:
            current = 0
    return str(best)


def _solve_fizzbuzz_variant(stdin: str) -> str:
    (n, a, b), (label_a, label_b) = _parse_fizzbuzz(stdin)
    output = []
    for value in range(1, n + 1):
        token = ""
        if value % a == 0:
            token += label_a
        if value % b == 0:
            token += label_b
        output.append(token or str(value))
    return " ".join(output)


def _solve_running_total(stdin: str) -> str:
    _, numbers = _parse_int_array(stdin)
    total = 0
    running = []
    for number in numbers:
        total += number
        running.append(str(total))
    return " ".join(running)


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


def _solve_two_sum_count(stdin: str) -> str:
    _, target, numbers = _parse_target_array(stdin)
    counts: Counter[int] = Counter()
    pairs = 0
    for number in numbers:
        pairs += counts[target - number]
        counts[number] += 1
    return str(pairs)


def _solve_max_subarray_sum(stdin: str) -> str:
    _, numbers = _parse_int_array(stdin)
    best = numbers[0]
    current = numbers[0]
    for number in numbers[1:]:
        current = max(number, current + number)
        best = max(best, current)
    return str(best)


def _solve_group_anagrams_count(stdin: str) -> str:
    _, words = _parse_word_list(stdin)
    groups = {"".join(sorted(word)) for word in words}
    return str(len(groups))


def _solve_balanced_brackets(stdin: str) -> str:
    text = stdin.strip()
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return "NO"
    return "YES" if not stack else "NO"


def _solve_matrix_diagonal_sum(stdin: str) -> str:
    _, matrix = _parse_matrix(stdin)
    total = 0
    size = len(matrix)
    for index in range(size):
        total += matrix[index][index]
        mirrored = size - 1 - index
        if mirrored != index:
            total += matrix[index][mirrored]
    return str(total)


def _solve_smallest_most_frequent(stdin: str) -> str:
    _, numbers = _parse_int_array(stdin)
    counts = Counter(numbers)
    best_count = max(counts.values())
    best_value = min(number for number, count in counts.items() if count == best_count)
    return str(best_value)


def _solve_reverse_words(stdin: str) -> str:
    words = stdin.strip().split()
    return " ".join(reversed(words))


def _solve_longest_common_subsequence(stdin: str) -> str:
    left, right = _parse_two_strings(stdin)
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            if left[i - 1] == right[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return str(dp[-1][-1])


def _solve_word_ladder_steps(stdin: str) -> str:
    start, target, words = _parse_word_ladder(stdin)
    if start == target:
        return "0"
    word_set = set(words)
    if target not in word_set:
        return "-1"

    queue: deque[tuple[str, int]] = deque([(start, 0)])
    visited = {start}
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    while queue:
        current, steps = queue.popleft()
        for index in range(len(current)):
            for letter in alphabet:
                if letter == current[index]:
                    continue
                candidate = current[:index] + letter + current[index + 1 :]
                if candidate == target:
                    return str(steps + 1)
                if candidate in word_set and candidate not in visited:
                    visited.add(candidate)
                    queue.append((candidate, steps + 1))
    return "-1"


def _solve_merge_intervals(stdin: str) -> str:
    intervals = _parse_intervals(stdin)
    ordered = sorted(intervals)
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return str(len(merged))


def _solve_min_coins(stdin: str) -> str:
    _, target, coins = _parse_coin_problem(stdin)
    best = [target + 1] * (target + 1)
    best[0] = 0
    for value in range(1, target + 1):
        for coin in coins:
            if coin <= value:
                best[value] = min(best[value], best[value - coin] + 1)
    return str(best[target] if best[target] <= target else -1)


def _solve_rotate_matrix_90(stdin: str) -> str:
    _, matrix = _parse_matrix(stdin)
    size = len(matrix)
    rotated = [[matrix[size - 1 - row][col] for row in range(size)] for col in range(size)]
    flattened = [str(value) for row in rotated for value in row]
    return " ".join(flattened)


def _parse_int_array(stdin: str) -> tuple[int, list[int]]:
    lines = [line.strip() for line in stdin.strip().splitlines() if line.strip()]
    n = int(lines[0])
    numbers = [int(part) for part in lines[1].split()]
    if len(numbers) != n:
        raise ValueError(f"Expected {n} integers, received {len(numbers)}")
    return n, numbers


def _parse_target_array(stdin: str) -> tuple[int, int, list[int]]:
    lines = [line.strip() for line in stdin.strip().splitlines() if line.strip()]
    n, target = map(int, lines[0].split())
    numbers = [int(part) for part in lines[1].split()]
    if len(numbers) != n:
        raise ValueError(f"Expected {n} integers, received {len(numbers)}")
    return n, target, numbers


def _parse_word_list(stdin: str) -> tuple[int, list[str]]:
    lines = [line.strip() for line in stdin.strip().splitlines() if line.strip()]
    n = int(lines[0])
    words = lines[1].split()
    if len(words) != n:
        raise ValueError(f"Expected {n} words, received {len(words)}")
    return n, words


def _parse_matrix(stdin: str) -> tuple[int, list[list[int]]]:
    lines = [line.strip() for line in stdin.strip().splitlines() if line.strip()]
    n = int(lines[0])
    matrix = [[int(part) for part in line.split()] for line in lines[1 : n + 1]]
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("Matrix dimensions do not match n.")
    return n, matrix


def _parse_two_strings(stdin: str) -> tuple[str, str]:
    lines = stdin.strip().splitlines()
    if len(lines) < 2:
        raise ValueError("Expected two lines of text.")
    return lines[0].strip(), lines[1].strip()


def _parse_fizzbuzz(stdin: str) -> tuple[tuple[int, int, int], tuple[str, str]]:
    lines = [line.strip() for line in stdin.strip().splitlines() if line.strip()]
    n, a, b = map(int, lines[0].split())
    label_a, label_b = lines[1].split()
    return (n, a, b), (label_a, label_b)


def _parse_word_ladder(stdin: str) -> tuple[str, str, list[str]]:
    lines = [line.strip() for line in stdin.strip().splitlines() if line.strip()]
    start, target = lines[0].split()
    n = int(lines[1])
    words = lines[2].split()
    if len(words) != n:
        raise ValueError(f"Expected {n} words, received {len(words)}")
    return start, target, words


def _parse_intervals(stdin: str) -> list[tuple[int, int]]:
    lines = [line.strip() for line in stdin.strip().splitlines() if line.strip()]
    n = int(lines[0])
    intervals = [tuple(map(int, line.split())) for line in lines[1 : n + 1]]
    if len(intervals) != n:
        raise ValueError("Interval count does not match n.")
    return [(start, end) for start, end in intervals]


def _parse_coin_problem(stdin: str) -> tuple[int, int, list[int]]:
    lines = [line.strip() for line in stdin.strip().splitlines() if line.strip()]
    n, target = map(int, lines[0].split())
    coins = [int(part) for part in lines[1].split()]
    if len(coins) != n:
        raise ValueError(f"Expected {n} coins, received {len(coins)}")
    return n, target, coins


def _make_balanced_brackets(rng: random.Random, pairs: int) -> str:
    opens = ["(", "[", "{"]
    closing = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    output: list[str] = []
    for _ in range(pairs * 2):
        can_open = len(stack) < pairs and (not stack or rng.random() < 0.6)
        if can_open:
            token = rng.choice(opens)
            stack.append(token)
            output.append(token)
        else:
            output.append(closing[stack.pop()])
    while stack:
        output.append(closing[stack.pop()])
    return "".join(output)


def _make_unbalanced_brackets(rng: random.Random, pairs: int) -> str:
    text = list(_make_balanced_brackets(rng, pairs))
    if not text:
        return "("
    mode = rng.choice(["swap", "drop", "flip"])
    if mode == "swap" and len(text) >= 2:
        index = rng.randrange(len(text) - 1)
        text[index], text[index + 1] = text[index + 1], text[index]
    elif mode == "drop":
        del text[rng.randrange(len(text))]
    else:
        replacements = ["(", ")", "[", "]", "{", "}"]
        text[rng.randrange(len(text))] = rng.choice(replacements)
    return "".join(text)


def _random_word(rng: random.Random, length: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _build_word_ladder_path(rng: random.Random, length: int, steps: int) -> list[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    current = _random_word(rng, length)
    path = [current]
    used = {current}
    while len(path) < steps + 1:
        chars = list(path[-1])
        index = rng.randrange(length)
        replacement = rng.choice(alphabet.replace(chars[index], ""))
        chars[index] = replacement
        candidate = "".join(chars)
        if candidate in used:
            continue
        used.add(candidate)
        path.append(candidate)
    return path


def _build_word_ladder_extras(
    rng: random.Random,
    length: int,
    count: int,
    disallowed: set[str],
) -> list[str]:
    words: list[str] = []
    seen = set(disallowed)
    while len(words) < count:
        candidate = _random_word(rng, length)
        if candidate in seen:
            continue
        seen.add(candidate)
        words.append(candidate)
    return words
