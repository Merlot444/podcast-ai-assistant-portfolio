"""Thin table-level repositories over the proven Day 18 CRUD functions."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage import crud
from app.storage.models import SessionModel
from app.storage.schemas import (
    DocumentCreate,
    DocumentRead,
    EvalResultCreate,
    EvalResultRead,
    EvalRunCreate,
    EvalRunRead,
    EvalRunStatusUpdate,
    MessageCreate,
    MessageRead,
    SessionCreate,
    SessionRead,
    SessionUpdate,
)


class StorageRepository:
    """Perform table-shaped operations without enforcing business lifecycles."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_document(self, payload: DocumentCreate) -> DocumentRead:
        return crud.create_document(self.session, payload)

    def get_document(self, document_id: str) -> DocumentRead | None:
        return crud.get_document(self.session, document_id)

    def get_document_by_episode(self, episode_id: str) -> DocumentRead | None:
        return crud.get_document_by_episode(self.session, episode_id)

    def delete_document(self, document_id: str) -> bool:
        return crud.delete_document(self.session, document_id)

    def create_session(self, payload: SessionCreate) -> SessionRead:
        return crud.create_session(self.session, payload)

    def get_session(self, session_id: str) -> SessionRead | None:
        return crud.get_session(self.session, session_id)

    def list_sessions(self) -> list[SessionRead]:
        return crud.list_sessions(self.session)

    def update_session(
        self,
        session_id: str,
        payload: SessionUpdate,
    ) -> SessionRead:
        return crud.update_session(self.session, session_id, payload)

    def lock_session_for_append(self, session_id: str) -> SessionRead:
        statement = select(SessionModel).where(SessionModel.session_id == session_id)
        bind = self.session.get_bind()
        if bind.dialect.name == "postgresql":
            statement = statement.with_for_update()
        model = self.session.scalar(statement)
        if model is None:
            raise crud.StorageNotFoundError(f"找不到 session：{session_id}")
        return crud.to_session_read(model)

    def append_message(
        self,
        session_id: str,
        payload: MessageCreate,
    ) -> MessageRead:
        return crud.append_message(self.session, session_id, payload)

    def list_messages(self, session_id: str) -> list[MessageRead]:
        return crud.list_messages(self.session, session_id)

    def delete_session(self, session_id: str) -> bool:
        return crud.delete_session(self.session, session_id)

    def create_eval_run(self, payload: EvalRunCreate) -> EvalRunRead:
        return crud.create_eval_run(self.session, payload)

    def get_eval_run(self, run_id: str) -> EvalRunRead | None:
        return crud.get_eval_run(self.session, run_id)

    def update_eval_run_status(
        self,
        run_id: str,
        payload: EvalRunStatusUpdate,
    ) -> EvalRunRead:
        return crud.update_eval_run_status(self.session, run_id, payload)

    def create_eval_result(
        self,
        run_id: str,
        payload: EvalResultCreate,
    ) -> EvalResultRead:
        return crud.create_eval_result(self.session, run_id, payload)

    def list_eval_results(self, run_id: str) -> list[EvalResultRead]:
        return crud.list_eval_results(self.session, run_id)

    def delete_eval_run(self, run_id: str) -> bool:
        return crud.delete_eval_run(self.session, run_id)
