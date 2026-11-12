"""Day 25 local page structure, safety, and responsive-contract tests."""

from __future__ import annotations

import re
import tomllib
from html.parser import HTMLParser
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.api import REQUEST_ID_PATTERN, create_app
from app.api.application import (
    ASK_FAILED_MESSAGE,
    ASK_NOT_FOUND_MESSAGE,
    ASK_UNAVAILABLE_MESSAGE,
    INTERNAL_ERROR_MESSAGE,
    INVALID_REQUEST_MESSAGE,
    REQUEST_BODY_TOO_LARGE_MESSAGE,
)


PROJECT_ROOT = Path(__file__).parents[1]
STATIC_ROOT = PROJECT_ROOT / "app" / "api" / "static"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data: str) -> None:
        self.text_parts.append(data)


def _parse_page(html: str) -> PageParser:
    parser = PageParser()
    parser.feed(html)
    return parser


def test_root_returns_the_minimal_semantic_question_page() -> None:
    with TestClient(create_app(event_sink=lambda _event: None)) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert re.fullmatch(REQUEST_ID_PATTERN, response.headers["x-request-id"])
    parser = _parse_page(response.text)
    visible_text = " ".join(parser.text_parts)
    assert "Podcast AI Assistant" in visible_text
    assert "Ask your podcast." in visible_text
    assert "围绕单期播客提问，获得带有可核验引用的回答。" in visible_text
    assert "正在检查" in visible_text
    assert "仅表示 API 可响应" in visible_text
    assert not any(tag == "img" for tag, _attrs in parser.tags)


def test_form_has_real_labels_expected_fields_and_one_submit_button() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    parser = _parse_page(html)
    tags = parser.tags

    labels = {
        attrs.get("for")
        for tag, attrs in tags
        if tag == "label" and attrs.get("for")
    }
    controls = {
        attrs.get("id")
        for tag, attrs in tags
        if tag in {"input", "textarea"} and attrs.get("id")
    }
    assert controls == {"question", "episode-id", "top-k", "session-id"}
    assert controls <= labels

    submit_buttons = [
        attrs
        for tag, attrs in tags
        if tag == "button" and attrs.get("type") == "submit"
    ]
    assert len(submit_buttons) == 1
    assert submit_buttons[0]["id"] == "submit-button"
    assert 'id="top-k"' in html and 'value="3"' in html
    assert "正在生成回答…" not in html


def test_page_contains_answer_citation_error_and_request_id_regions() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "answer-region",
        "answer-text",
        "answer-request-id",
        "citation-region",
        "citation-list",
        "error-region",
        "error-text",
        "error-request-id",
        "form-status",
        "api-status",
    ):
        assert f'id="{element_id}"' in html
    assert html.count('aria-live="') >= 3
    assert 'aria-busy="false"' in html


def test_page_uses_only_fixed_local_css_and_javascript() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    parser = _parse_page(html)
    scripts = [attrs.get("src") for tag, attrs in parser.tags if tag == "script"]
    styles = [
        attrs.get("href")
        for tag, attrs in parser.tags
        if tag == "link" and attrs.get("rel") == "stylesheet"
    ]
    assert scripts == ["/static/day25.js"]
    assert styles == ["/static/day25.css"]
    assert "http://" not in html
    assert "https://" not in html
    assert "//cdn" not in html.lower()
    assert "@import" not in (STATIC_ROOT / "day25.css").read_text(encoding="utf-8")

    with TestClient(create_app(event_sink=lambda _event: None)) as client:
        css = client.get("/static/day25.css")
        javascript = client.get("/static/day25.js")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert javascript.status_code == 200
    assert javascript.headers["content-type"].startswith("text/javascript")
    assert re.fullmatch(REQUEST_ID_PATTERN, css.headers["x-request-id"])
    assert re.fullmatch(REQUEST_ID_PATTERN, javascript.headers["x-request-id"])


def test_javascript_uses_same_origin_safe_dom_and_fixed_failures() -> None:
    javascript = (STATIC_ROOT / "day25.js").read_text(encoding="utf-8")
    for forbidden in (
        "innerHTML",
        "eval(",
        "localStorage",
        "sessionStorage",
        "document.cookie",
        "console.",
        "http://",
        "https://",
    ):
        assert forbidden not in javascript
    assert 'fetch("/health"' in javascript
    assert 'fetch("/ask"' in javascript
    assert "document.createElement" in javascript
    assert ".textContent" in javascript
    assert 'response.headers.get("X-Request-ID")' in javascript
    assert "REQUEST_ID_PATTERN.test" in javascript
    assert "暂时无法连接服务，请稍后再试。" in javascript
    assert "问答服务暂时不可用。" in javascript
    safe_messages_match = re.search(
        r"const SAFE_API_MESSAGES = new Set\(\[(?P<items>.*?)\]\);",
        javascript,
        re.DOTALL,
    )
    assert safe_messages_match is not None
    safe_messages = set(re.findall(r'"([^"]+)"', safe_messages_match["items"]))
    assert safe_messages == {
        INVALID_REQUEST_MESSAGE,
        REQUEST_BODY_TOO_LARGE_MESSAGE,
        ASK_NOT_FOUND_MESSAGE,
        ASK_UNAVAILABLE_MESSAGE,
        ASK_FAILED_MESSAGE,
        INTERNAL_ERROR_MESSAGE,
    }


def test_loading_state_disables_once_and_is_restored_in_finally() -> None:
    javascript = (STATIC_ROOT / "day25.js").read_text(encoding="utf-8")
    assert "submitButton.disabled = isLoading" in javascript
    assert 'submitButton.setAttribute("aria-busy", String(isLoading))' in javascript
    assert "if (submitButton.disabled" in javascript
    assert re.search(r"finally\s*\{\s*setLoading\(false\);", javascript)


def test_css_has_mobile_single_columns_wrapping_focus_and_reduced_motion() -> None:
    css = (STATIC_ROOT / "day25.css").read_text(encoding="utf-8")
    assert "@media (max-width: 760px)" in css
    assert ".details-grid" in css
    assert ".citation-grid" in css
    assert "grid-template-columns: 1fr" in css
    assert "overflow-wrap: anywhere" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "min-height: 50px" in css


def test_static_resources_are_declared_as_package_data() -> None:
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    assert configuration["tool"]["setuptools"]["package-data"]["app.api"] == [
        "static/*.html",
        "static/*.css",
        "static/*.js",
    ]
