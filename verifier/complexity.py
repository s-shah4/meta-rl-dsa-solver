from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from env.executor import run_code as execute_submission

PROBE_TIMEOUT_SECONDS = 2.0
METRICS_PATTERN = re.compile(r"ADAPT_METRICS:\s*time_ms=([0-9.]+)\s+peak_kb=([0-9.]+)")


@dataclass
class ComplexitySignals:
    nested_loop_depth: int = 0
    list_comprehensions: int = 0
    set_comprehensions: int = 0
    dict_comprehensions: int = 0
    generator_expressions: int = 0
    sorting_calls: int = 0
    materialized_builtin_inputs: int = 0


class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.signals = ComplexitySignals()
        self._loop_depth = 0

    def visit_For(self, node: ast.For) -> Any:
        self._loop_depth += 1
        self.signals.nested_loop_depth = max(self.signals.nested_loop_depth, self._loop_depth)
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> Any:
        self._loop_depth += 1
        self.signals.nested_loop_depth = max(self.signals.nested_loop_depth, self._loop_depth)
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        self.signals.list_comprehensions += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        self.signals.set_comprehensions += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        self.signals.dict_comprehensions += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        self.signals.generator_expressions += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        fn_name = _call_name(node.func)
        if fn_name in {"sorted", "list.sort"}:
            self.signals.sorting_calls += 1
        if fn_name in {"sum", "max", "min", "any", "all", "len"} and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, (ast.ListComp, ast.SetComp, ast.DictComp, ast.List, ast.Set, ast.Dict)):
                self.signals.materialized_builtin_inputs += 1
        self.generic_visit(node)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _probe_timeout() -> float:
    raw_value = os.getenv("ADAPT_PROBE_TIMEOUT", str(PROBE_TIMEOUT_SECONDS))
    try:
        timeout = float(raw_value)
    except ValueError:
        return PROBE_TIMEOUT_SECONDS
    return timeout if timeout > 0 else PROBE_TIMEOUT_SECONDS


def _build_measurement_harness(code: str) -> str:
    return f"""
import sys as _adapt_sys
import time as _adapt_time
import tracemalloc as _adapt_tracemalloc

_adapt_globals = {{"__name__": "__main__"}}
_adapt_source = {code!r}
_adapt_tracemalloc.start()
_adapt_t0 = _adapt_time.perf_counter()
exec(compile(_adapt_source, "<submission>", "exec"), _adapt_globals, _adapt_globals)
_adapt_t1 = _adapt_time.perf_counter()
_adapt_peak_kb = _adapt_tracemalloc.get_traced_memory()[1] / 1024
_adapt_tracemalloc.stop()
print(
    f"ADAPT_METRICS: time_ms={{(_adapt_t1 - _adapt_t0) * 1000:.3f}} peak_kb={{_adapt_peak_kb:.1f}}",
    file=_adapt_sys.stderr,
)
"""


def _parse_harness_output(stderr: str) -> tuple[float, float]:
    match = METRICS_PATTERN.search(str(stderr or ""))
    if match is None:
        return 0.0, 0.0
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError:
        return 0.0, 0.0


def _fit_scaling_exponent(sizes: list[float], values: list[float]) -> float:
    if len(sizes) < 2 or len(values) < 2:
        return 1.0
    log_n = [math.log(max(size, 1.0)) for size in sizes]
    log_v = [math.log(max(value, 1e-6)) for value in values]
    count = len(log_n)
    mean_n = sum(log_n) / count
    mean_v = sum(log_v) / count
    numerator = sum((log_n[index] - mean_n) * (log_v[index] - mean_v) for index in range(count))
    denominator = sum((log_n[index] - mean_n) ** 2 for index in range(count))
    return numerator / denominator if denominator > 1e-9 else 1.0


def _exponent_to_score(alpha: float) -> float:
    if alpha < 0.1:
        return 1.0
    if alpha < 1.2:
        return 0.85
    if alpha < 1.6:
        return 0.75
    if alpha < 2.3:
        return 0.50
    if alpha < 3.2:
        return 0.20
    return 0.0


def _memory_to_score(peak_kb: float) -> float:
    mb = peak_kb / 1024.0
    if mb < 1:
        return 1.0
    if mb < 10:
        return 0.85
    if mb < 50:
        return 0.65
    if mb < 256:
        return 0.40
    return 0.10


def _hints_from_scores(time_score: float, space_score: float, time_alpha: float) -> list[str]:
    hints: list[str] = []
    if time_alpha >= 2.3:
        hints.append("Reduce quadratic-or-worse work; measured runtime growth looks steep across larger inputs.")
    elif time_score < 0.85:
        hints.append("Consider a more scalable algorithm so runtime grows more gently with input size.")
    if space_score < 0.85:
        hints.append("Reduce peak memory usage by avoiding large intermediate containers when possible.")
    return hints


def _merge_hints(*hint_groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in hint_groups:
        for hint in group:
            if hint and hint not in seen:
                seen.add(hint)
                merged.append(hint)
    return merged


def _heuristic_fallback(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "time_complexity_score": 0.0,
            "space_complexity_score": 0.0,
            "efficiency_score": 0.0,
            "optimization_hints": [],
            "complexity_signals": {"measurement_source": "heuristic"},
        }

    visitor = ComplexityVisitor()
    visitor.visit(tree)
    signals = visitor.signals

    time_penalty = 0.0
    space_penalty = 0.0
    hints: list[str] = []

    if signals.nested_loop_depth > 1:
        time_penalty += 0.2 * (signals.nested_loop_depth - 1)
        hints.append("Reduce nested iteration if the problem can be solved in fewer passes.")
    if signals.sorting_calls:
        time_penalty += 0.1 * signals.sorting_calls
        hints.append("Avoid sorting unless it is required by the algorithm; it adds extra time complexity.")

    temporary_materializations = (
        signals.list_comprehensions
        + signals.set_comprehensions
        + signals.dict_comprehensions
        + signals.materialized_builtin_inputs
    )
    if temporary_materializations:
        space_penalty += 0.12 * temporary_materializations
        hints.append("Avoid materializing temporary containers when a streaming pass or generator expression is enough.")
    if signals.materialized_builtin_inputs:
        space_penalty += 0.08 * signals.materialized_builtin_inputs
        hints.append("Use generator expressions inside reducers like sum(...) instead of building an intermediate list.")

    time_score = max(0.0, 1.0 - min(time_penalty, 0.7))
    space_score = max(0.0, 1.0 - min(space_penalty, 0.7))
    efficiency_score = round(0.55 * time_score + 0.45 * space_score, 4)

    return {
        "time_complexity_score": round(time_score, 4),
        "space_complexity_score": round(space_score, 4),
        "efficiency_score": efficiency_score,
        "optimization_hints": hints,
        "complexity_signals": {
            "nested_loop_depth": signals.nested_loop_depth,
            "list_comprehensions": signals.list_comprehensions,
            "set_comprehensions": signals.set_comprehensions,
            "dict_comprehensions": signals.dict_comprehensions,
            "generator_expressions": signals.generator_expressions,
            "sorting_calls": signals.sorting_calls,
            "materialized_builtin_inputs": signals.materialized_builtin_inputs,
            "measurement_source": "heuristic",
        },
    }


def _empirical_complexity(code: str, probe_inputs: list[str]) -> dict[str, Any]:
    heuristic = _heuristic_fallback(code)
    if len(probe_inputs) < 3:
        return heuristic

    harness = _build_measurement_harness(code)
    sizes: list[float] = []
    times: list[float] = []
    mem_peaks: list[float] = []
    for probe_input in probe_inputs:
        result = execute_submission(harness, probe_input, timeout_seconds=_probe_timeout())
        if bool(result.get("timed_out")) or int(result.get("exit_code", 0)) != 0:
            return heuristic
        wall_ms, peak_kb = _parse_harness_output(str(result.get("stderr", "")))
        if wall_ms <= 0.0 and peak_kb <= 0.0:
            return heuristic
        sizes.append(float(len(probe_input)))
        times.append(float(wall_ms))
        mem_peaks.append(float(peak_kb))

    if len(sizes) < 3:
        return heuristic

    empirical_time_score = _exponent_to_score(_fit_scaling_exponent(sizes, times))
    empirical_space_score = _memory_to_score(max(mem_peaks))
    time_alpha = _fit_scaling_exponent(sizes, times)
    space_alpha = _fit_scaling_exponent(sizes, mem_peaks)

    time_score = min(empirical_time_score, float(heuristic["time_complexity_score"]))
    space_score = min(empirical_space_score, float(heuristic["space_complexity_score"]))
    efficiency_score = round(min(0.7 * time_score + 0.3 * space_score, float(heuristic["efficiency_score"])), 4)
    optimization_hints = _merge_hints(
        _hints_from_scores(time_score, space_score, time_alpha),
        list(heuristic.get("optimization_hints", [])),
    )

    complexity_signals = dict(heuristic.get("complexity_signals", {}))
    complexity_signals.update(
        {
            "time_exponent": round(time_alpha, 4),
            "space_exponent": round(space_alpha, 4),
            "peak_memory_kb": round(max(mem_peaks), 1),
            "measurement_source": "empirical",
        }
    )

    return {
        "time_complexity_score": round(time_score, 4),
        "space_complexity_score": round(space_score, 4),
        "efficiency_score": efficiency_score,
        "optimization_hints": optimization_hints,
        "complexity_signals": complexity_signals,
    }


def analyze_code_complexity(code: str, probe_inputs: list[str] | None = None) -> dict[str, Any]:
    if probe_inputs and len(probe_inputs) >= 3:
        try:
            return _empirical_complexity(code, probe_inputs)
        except Exception:
            pass
    return _heuristic_fallback(code)
