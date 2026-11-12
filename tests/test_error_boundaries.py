"""Repository-wide guards for safe exception rendering at output boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1] / "app"


def test_exception_handlers_do_not_render_raw_exception_text() -> None:
    violations: list[str] = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for handler in (
            node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
        ):
            if handler.name is None:
                continue
            for node in ast.walk(handler):
                if isinstance(node, ast.FormattedValue) and _is_name(
                    node.value, handler.name
                ):
                    violations.append(_location(path, node))
                elif isinstance(node, ast.Call) and _is_unsafe_sink_call(
                    node, handler.name
                ):
                    violations.append(_location(path, node))

    assert violations == [], (
        "异常对象不得直接进入 f-string、str/repr、print 或日志；"
        f"请使用 safe_error_message：{violations}"
    )


def test_legacy_raw_value_repr_pattern_is_absent() -> None:
    violations = [
        path.relative_to(APP_ROOT.parent).as_posix()
        for path in sorted(APP_ROOT.rglob("*.py"))
        if "{value!r}" in path.read_text(encoding="utf-8")
    ]

    assert violations == []


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_unsafe_sink_call(node: ast.Call, exception_name: str) -> bool:
    if not any(_is_name(argument, exception_name) for argument in node.args):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in {"print", "repr", "str"}
    if isinstance(node.func, ast.Attribute):
        return node.func.attr in {
            "critical",
            "debug",
            "error",
            "exception",
            "info",
            "warning",
        }
    return False


def _location(path: Path, node: ast.AST) -> str:
    relative = path.relative_to(APP_ROOT.parent).as_posix()
    return f"{relative}:{node.lineno}"
