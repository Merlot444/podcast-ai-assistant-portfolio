"""Replaceable SQL candidate generators that never receive a database handle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol

from openai import OpenAI
from pydantic import ValidationError

from app.core.llm import ModelServiceError, create_json_completion
from app.text2sql.catalog import SchemaCatalog
from app.text2sql.schemas import GeneratedQuery, Text2SQLRequest


class QueryGenerationError(RuntimeError):
    """Safe failure raised when a generator cannot produce the strict contract."""


class QueryGenerator(Protocol):
    def generate(
        self,
        request: Text2SQLRequest,
        catalog: SchemaCatalog,
    ) -> GeneratedQuery: ...


class MappingQueryGenerator:
    """Deterministic offline generator for acceptance tests and demos."""

    def __init__(self, queries: Mapping[str, GeneratedQuery]) -> None:
        self._queries = dict(queries)
        self.call_count = 0

    def generate(
        self,
        request: Text2SQLRequest,
        catalog: SchemaCatalog,
    ) -> GeneratedQuery:
        del catalog
        self.call_count += 1
        try:
            return self._queries[request.question]
        except KeyError:
            raise QueryGenerationError("没有可用的离线 SQL 候选。") from None


class OpenAICompatibleQueryGenerator:
    """Optional strict-JSON adapter; its output remains untrusted."""

    def __init__(self, client: OpenAI, *, model: str) -> None:
        self._client = client
        self._model = model

    def generate(
        self,
        request: Text2SQLRequest,
        catalog: SchemaCatalog,
    ) -> GeneratedQuery:
        system = (
            "Generate one SQLite SELECT candidate for the user's question. "
            "Return exactly one JSON object with keys sql and parameters. "
            "All condition values must use named :parameters. Never emit markdown.\n\n"
            + catalog.prompt_text()
        )
        try:
            completion = create_json_completion(
                self._client,
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": request.question},
                ],
                max_tokens=600,
            )
            content = completion.choices[0].message.content
            if not content:
                raise QueryGenerationError("模型没有返回 SQL 候选。")
            return GeneratedQuery.model_validate(json.loads(content))
        except QueryGenerationError:
            raise
        except ModelServiceError:
            raise
        except (IndexError, json.JSONDecodeError, ValidationError, TypeError):
            raise QueryGenerationError("模型返回的 SQL 候选不符合严格契约。") from None
