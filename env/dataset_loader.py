from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from env.generator import HIDDEN_TEST_COUNT, TOTAL_TEST_CASES, VISIBLE_TEST_COUNT

MAX_IO_CHARS = 4096
DEFAULT_DATASET_NAME = "deepmind/code_contests"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_PROBLEMS = 5000


def _load_raw_dataset(
    dataset_name: str,
    split: str = DEFAULT_SPLIT,
    max_problems: int = DEFAULT_MAX_PROBLEMS,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    rows: list[dict[str, Any]] = []
    for raw_row in dataset:
        row = dict(raw_row)
        statement = str(row.get("description") or row.get("question") or "").strip()
        if not statement:
            continue
        if not _extract_pairs(row.get("public_tests")):
            continue
        if not (_extract_pairs(row.get("private_tests")) or _extract_pairs(row.get("generated_tests"))):
            continue
        rows.append(row)
        if len(rows) >= int(max_problems):
            break
    return rows


def _normalise_row(raw_row: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
    statement = _extract_problem_statement(raw_row)
    if not statement:
        return None

    public_pairs = _extract_pairs(raw_row.get("public_tests"))
    private_pairs = _extract_pairs(raw_row.get("private_tests"))
    generated_pairs = _extract_pairs(raw_row.get("generated_tests"))
    visible_pairs = public_pairs[:VISIBLE_TEST_COUNT]
    hidden_pairs = (private_pairs + generated_pairs)[:HIDDEN_TEST_COUNT]
    if len(visible_pairs) != VISIBLE_TEST_COUNT or len(hidden_pairs) != HIDDEN_TEST_COUNT:
        return None

    test_cases: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    for index, (raw_input, raw_output) in enumerate(visible_pairs + hidden_pairs):
        if len(str(raw_output or "")) > MAX_IO_CHARS:
            return None
        normalized_input = _normalize_io_text(raw_input, ensure_trailing_newline=True)
        normalized_output = _normalize_io_text(raw_output, ensure_trailing_newline=False)
        if not normalized_input or normalized_input in seen_inputs:
            return None
        seen_inputs.add(normalized_input)
        test_cases.append(
            {
                "input": normalized_input,
                "output": normalized_output,
                "is_visible": index < VISIBLE_TEST_COUNT,
            }
        )

    if len(test_cases) != TOTAL_TEST_CASES:
        return None

    difficulty_label, difficulty_value = _difficulty_fields(raw_row, dataset_name)
    input_format = _extract_section(statement, "input") or "Read from stdin."
    constraints = _extract_constraints(statement)
    problem_type = _infer_problem_type(raw_row, statement)
    problem_id = _problem_id(raw_row, dataset_name)

    visible_examples = [dict(test_case) for test_case in test_cases[:VISIBLE_TEST_COUNT]]
    return {
        "problem_id": problem_id,
        "problem_type": problem_type,
        "difficulty": difficulty_value,
        "difficulty_label": difficulty_label,
        "problem": statement,
        "input_format": input_format,
        "constraints": constraints,
        "test_cases": test_cases,
        "visible_problem": {
            "problem": statement,
            "input_format": input_format,
            "constraints": constraints,
            "examples": visible_examples,
        },
        "generation_mode": "dataset",
        "validity_bonus": 1.0,
    }


class DatasetProblemBank:
    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        split: str = DEFAULT_SPLIT,
        max_problems: int = DEFAULT_MAX_PROBLEMS,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.max_problems = int(max_problems)
        self._by_difficulty: dict[str, list[dict[str, Any]]] = {
            "easy": [],
            "medium": [],
            "hard": [],
        }
        self._by_id: dict[str, dict[str, Any]] = {}

        raw_rows = _load_raw_dataset(dataset_name=dataset_name, split=split, max_problems=max_problems)
        for raw_row in raw_rows:
            normalized = _normalise_row(raw_row, dataset_name)
            if normalized is None:
                continue
            problem_id = str(normalized["problem_id"])
            if problem_id in self._by_id:
                continue
            difficulty = str(normalized.get("difficulty_label", "medium")).lower()
            if difficulty not in self._by_difficulty:
                difficulty = "medium"
                normalized["difficulty_label"] = difficulty
            stored = _copy_problem(normalized)
            self._by_difficulty[difficulty].append(stored)
            self._by_id[problem_id] = stored

        if not self._by_id:
            raise ValueError(
                f"No usable problems were found in dataset `{dataset_name}` split `{split}` with max_problems={max_problems}."
            )

    def sample(self, difficulty: str, rng: random.Random, recent_types: list[str]) -> dict[str, Any] | None:
        requested = str(difficulty).strip().lower()
        candidates = list(self._by_difficulty.get(requested, []))
        if not candidates:
            candidates = [problem for bucket in self._by_difficulty.values() for problem in bucket]
        if not candidates:
            return None

        recent = {problem_type for problem_type in recent_types[-3:] if problem_type}
        diverse = [problem for problem in candidates if str(problem.get("problem_type", "")) not in recent]
        pool = diverse or candidates
        return _copy_problem(rng.choice(pool))

    def all_problem_ids(self) -> list[str]:
        return sorted(self._by_id)

    def get_by_id(self, problem_id: str) -> dict[str, Any]:
        return _copy_problem(self._by_id[str(problem_id)])

    def problem_types_for_difficulty(self, difficulty: str) -> list[str]:
        requested = str(difficulty).strip().lower()
        candidates = self._by_difficulty.get(requested, [])
        return sorted({str(problem.get("problem_type", "")) for problem in candidates if problem.get("problem_type")})


_BANK: DatasetProblemBank | None = None
_BANK_CONFIG: tuple[str, str, int] | None = None


def get_problem_bank(**kwargs: Any) -> DatasetProblemBank:
    global _BANK, _BANK_CONFIG

    dataset_name = str(kwargs.get("dataset_name", DEFAULT_DATASET_NAME))
    split = str(kwargs.get("split", DEFAULT_SPLIT))
    max_problems = int(kwargs.get("max_problems", DEFAULT_MAX_PROBLEMS))
    config = (dataset_name, split, max_problems)
    if _BANK is None or _BANK_CONFIG != config:
        _BANK = DatasetProblemBank(
            dataset_name=dataset_name,
            split=split,
            max_problems=max_problems,
        )
        _BANK_CONFIG = config
    return _BANK


def _extract_problem_statement(raw_row: dict[str, Any]) -> str:
    value = raw_row.get("description") or raw_row.get("question") or raw_row.get("problem")
    return str(value or "").strip()


def _extract_pairs(raw_value: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if raw_value is None:
        return pairs

    if isinstance(raw_value, dict):
        inputs = raw_value.get("input") or raw_value.get("inputs") or raw_value.get("stdin") or []
        outputs = raw_value.get("output") or raw_value.get("outputs") or raw_value.get("stdout") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        for raw_input, raw_output in zip(list(inputs), list(outputs)):
            pairs.append((str(raw_input), str(raw_output)))
        return pairs

    if isinstance(raw_value, list):
        for item in raw_value:
            if isinstance(item, dict):
                raw_input = item.get("input") or item.get("stdin") or item.get("in")
                raw_output = item.get("output") or item.get("stdout") or item.get("out")
                if raw_input is None or raw_output is None:
                    continue
                pairs.append((str(raw_input), str(raw_output)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))
        return pairs

    return pairs


def _normalize_io_text(value: Any, *, ensure_trailing_newline: bool) -> str:
    text = str(value or "")[:MAX_IO_CHARS]
    if ensure_trailing_newline:
        text = text.rstrip("\n")
        return f"{text}\n" if text else "\n"
    return text.strip()


def _difficulty_fields(raw_row: dict[str, Any], dataset_name: str) -> tuple[str, float]:
    dataset_key = dataset_name.lower()
    if "code_contests" in dataset_key or "code-contests" in dataset_key:
        raw_difficulty = raw_row.get("difficulty")
        try:
            rating = float(raw_difficulty)
        except (TypeError, ValueError):
            return "medium", 0.5
        normalized = max(0.0, min((rating - 800.0) / (3500.0 - 800.0), 1.0))
        if rating <= 1200:
            label = "easy"
        elif rating <= 1800:
            label = "medium"
        else:
            label = "hard"
        return label, round(normalized, 4)

    raw_difficulty = str(raw_row.get("difficulty") or "").strip().lower()
    if raw_difficulty in {"easy", "medium", "hard"}:
        return raw_difficulty, {"easy": 0.25, "medium": 0.5, "hard": 0.75}[raw_difficulty]
    return "medium", 0.5


def _problem_id(raw_row: dict[str, Any], dataset_name: str) -> str:
    prefix = "cc" if "code_contests" in dataset_name.lower() else "ds"
    for key in ("problem_id", "id", "name", "source"):
        value = raw_row.get(key)
        if value is not None and str(value).strip():
            candidate = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip()).strip("_")
            if candidate:
                return f"{prefix}_{candidate}"
    digest = hashlib.sha256(repr(sorted(raw_row.items())).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def _extract_section(statement: str, heading: str) -> str:
    pattern = re.compile(
        rf"{heading}\s*:?[\r\n]+(.*?)(?=\n[A-Z][A-Za-z ]{{1,30}}:?[\r\n]|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(statement)
    return match.group(1).strip() if match else ""


def _extract_constraints(statement: str) -> str:
    constraints = _extract_section(statement, "constraints")
    return constraints or "See problem statement."


def _infer_problem_type(raw_row: dict[str, Any], statement: str) -> str:
    parts: list[str] = [statement]
    tags = raw_row.get("tags")
    if isinstance(tags, list):
        parts.extend(str(tag) for tag in tags)
    elif isinstance(tags, str):
        parts.append(tags)
    for key in ("source", "name"):
        value = raw_row.get(key)
        if value:
            parts.append(str(value))
    text = " ".join(parts).lower()

    keyword_map = {
        "graph": "graph",
        "tree": "tree",
        "dynamic programming": "dp",
        " dp ": "dp",
        "string": "string",
        "array": "array",
        "greedy": "greedy",
        "math": "math",
        "sort": "sorting",
        "binary search": "search",
    }
    padded = f" {text} "
    for needle, problem_type in keyword_map.items():
        haystack = padded if needle.startswith(" ") and needle.endswith(" ") else text
        if needle in haystack:
            return problem_type
    return "implementation"


def _copy_problem(problem: dict[str, Any]) -> dict[str, Any]:
    copied = dict(problem)
    copied["test_cases"] = [dict(test_case) for test_case in problem.get("test_cases", [])]
    copied["visible_problem"] = dict(problem.get("visible_problem", {}))
    examples = copied["visible_problem"].get("examples")
    if isinstance(examples, list):
        copied["visible_problem"]["examples"] = [dict(example) for example in examples]
    return copied
