from __future__ import annotations


PROBLEM = {
    "problem": "Given an integer n, print n * 2.",
    "input_format": "A single integer n.",
    "constraints": "-10^9 <= n <= 10^9",
    "examples": [
        {"input": "2\n", "output": "4"},
        {"input": "5\n", "output": "10"},
    ],
}


TEST_CASES = [
    {"input": "2\n", "output": "4"},
    {"input": "5\n", "output": "10"},
    {"input": "0\n", "output": "0"},
    {"input": "1\n", "output": "2"},
    {"input": "-3\n", "output": "-6"},
    {"input": "10\n", "output": "20"},
    {"input": "999999\n", "output": "1999998"},
    {"input": "-1000000000\n", "output": "-2000000000"},
]


VISIBLE_TEST_COUNT = 3


def load_problem() -> dict:
    return dict(PROBLEM)


def load_test_cases() -> list[dict[str, str]]:
    return [dict(test_case) for test_case in TEST_CASES]
