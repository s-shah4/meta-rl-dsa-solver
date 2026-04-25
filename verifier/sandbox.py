from __future__ import annotations

import ast
from typing import Any

from env.executor import run_code as execute_submission

FORBIDDEN_IMPORTS = {
    "ctypes",
    "os",
    "pathlib",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
}
FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "open",
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def validate_code(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "syntax_ok": False,
            "safety_ok": False,
            "execution_status": "syntax_error",
            "error": str(exc),
        }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".", 1)[0]
                if root_name in FORBIDDEN_IMPORTS:
                    return {
                        "syntax_ok": True,
                        "safety_ok": False,
                        "execution_status": "safety_violation",
                        "error": f"Forbidden import: {root_name}",
                    }

        if isinstance(node, ast.ImportFrom):
            root_name = (node.module or "").split(".", 1)[0]
            if root_name in FORBIDDEN_IMPORTS:
                return {
                    "syntax_ok": True,
                    "safety_ok": False,
                    "execution_status": "safety_violation",
                    "error": f"Forbidden import: {root_name}",
                }

        if isinstance(node, ast.Call):
            fn_name = _call_name(node.func)
            if fn_name in FORBIDDEN_CALLS:
                return {
                    "syntax_ok": True,
                    "safety_ok": False,
                    "execution_status": "safety_violation",
                    "error": f"Forbidden call: {fn_name}",
                }

    return {
        "syntax_ok": True,
        "safety_ok": True,
        "execution_status": "ready",
        "error": "",
    }


def run_code(code: str, stdin: str, timeout: int | float = 1) -> dict[str, object]:
    return execute_submission(code, stdin, timeout_seconds=timeout)
