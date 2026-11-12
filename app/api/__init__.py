"""Lazy public exports for the optional Day 27 HTTP boundary."""

from __future__ import annotations

from importlib import import_module
from types import MappingProxyType


_PUBLIC_EXPORTS = MappingProxyType(
    {
        "AgentWorkflowAskService": (
            "app.api.adapters",
            "AgentWorkflowAskService",
        ),
        "AskService": ("app.api.services", "AskService"),
        "ConversationSessionReader": (
            "app.api.adapters",
            "ConversationSessionReader",
        ),
        "EvaluationRetrieveService": (
            "app.api.adapters",
            "EvaluationRetrieveService",
        ),
        "DEFAULT_MAX_REQUEST_BODY_BYTES": (
            "app.api.request_boundaries",
            "DEFAULT_MAX_REQUEST_BODY_BYTES",
        ),
        "HttpRequestCompletedEvent": (
            "app.api.request_boundaries",
            "HttpRequestCompletedEvent",
        ),
        "REQUEST_ID_PATTERN": (
            "app.api.request_boundaries",
            "REQUEST_ID_PATTERN",
        ),
        "RuntimeConfig": ("app.api.runtime", "RuntimeConfig"),
        "SERVICE_NAME": ("app.api.lifecycle", "SERVICE_NAME"),
        "SERVICE_VERSION": ("app.api.lifecycle", "SERVICE_VERSION"),
        "ServiceLifecycleEvent": (
            "app.api.lifecycle",
            "ServiceLifecycleEvent",
        ),
        "SessionReader": ("app.api.services", "SessionReader"),
        "RetrieveService": ("app.api.services", "RetrieveService"),
        "create_app": ("app.api.application", "create_app"),
        "load_runtime_config": ("app.api.runtime", "load_runtime_config"),
    }
)

__all__ = sorted(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> object:
    """Resolve optional HTTP exports only when a caller requests one."""
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_PUBLIC_EXPORTS})
