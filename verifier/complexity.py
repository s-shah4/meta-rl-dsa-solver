from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


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


def analyze_code_complexity(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "time_complexity_score": 0.0,
            "space_complexity_score": 0.0,
            "efficiency_score": 0.0,
            "optimization_hints": [],
            "complexity_signals": {},
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
        },
    }
