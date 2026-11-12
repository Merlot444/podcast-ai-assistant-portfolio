"""Business-shaped CRUD functions operating inside caller-owned transactions."""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.storage.models import (
    DocumentModel,
    EvalResultModel,
    EvalRunModel,
    MessageModel,
    SessionModel,
    utc_now,
)
from app.storage.schemas import (
    DocumentCreate,
    DocumentMetadataUpdate,
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


class StorageNotFoundError(LookupError):
    """Raised when a child write or requested update has no parent row."""


def create_document(session: Session, payload: DocumentCreate) -> DocumentRead:
    model = DocumentModel(
        document_id=payload.document_id,
        episode_id=payload.episode_id,
        source_file=payload.source_file,
        source_format=payload.source_format,
        title=payload.title,
        input_summary=payload.input_summary,
        metadata_json=payload.metadata,
    )
    session.add(model)
    session.flush()
    return _document_read(model)


def get_document(session: Session, document_id: str) -> DocumentRead | None:
    model = session.get(DocumentModel, document_id)
    return None if model is None else _document_read(model)


def get_document_by_episode(session: Session, episode_id: str) -> DocumentRead | None:
    model = session.scalar(
        select(DocumentModel).where(DocumentModel.episode_id == episode_id)
    )
    return None if model is None else _document_read(model)


def list_documents(session: Session) -> list[DocumentRead]:
    models = session.scalars(
        select(DocumentModel).order_by(
            DocumentModel.created_at,
            DocumentModel.document_id,
        )
    )
    return [_document_read(model) for model in models]


def update_document_metadata(
    session: Session,
    document_id: str,
    payload: DocumentMetadataUpdate,
) -> DocumentRead:
    model = _require_document(session, document_id)
    fields = payload.model_fields_set
    if "title" in fields:
        model.title = payload.title
    if "input_summary" in fields:
        model.input_summary = payload.input_summary
    if "metadata" in fields:
        model.metadata_json = payload.metadata or {}
    model.updated_at = utc_now()
    session.flush()
    return _document_read(model)


def delete_document(session: Session, document_id: str) -> bool:
    result = session.execute(
        delete(DocumentModel).where(DocumentModel.document_id == document_id)
    )
    return bool(result.rowcount)


def create_session(session: Session, payload: SessionCreate) -> SessionRead:
    model = SessionModel(
        session_id=payload.session_id,
        title=payload.title,
        status=payload.status,
    )
    session.add(model)
    session.flush()
    return to_session_read(model)


def get_session(session: Session, session_id: str) -> SessionRead | None:
    model = session.get(SessionModel, session_id)
    return None if model is None else to_session_read(model)


def list_sessions(session: Session) -> list[SessionRead]:
    models = session.scalars(
        select(SessionModel).order_by(
            SessionModel.created_at,
            SessionModel.session_id,
        )
    )
    return [to_session_read(model) for model in models]


def update_session(
    session: Session,
    session_id: str,
    payload: SessionUpdate,
) -> SessionRead:
    model = _require_session(session, session_id)
    fields = payload.model_fields_set
    if "title" in fields:
        model.title = payload.title
    if "status" in fields and payload.status is not None:
        model.status = payload.status
    model.updated_at = utc_now()
    session.flush()
    return to_session_read(model)


def append_message(
    session: Session,
    session_id: str,
    payload: MessageCreate,
) -> MessageRead:
    parent = _require_session(session, session_id)
    last_sequence = session.scalar(
        select(func.max(MessageModel.sequence)).where(
            MessageModel.session_id == session_id
        )
    )
    sequence = 0 if last_sequence is None else last_sequence + 1
    model = MessageModel(
        message_id=payload.message_id,
        session_id=session_id,
        role=payload.role,
        content=payload.content,
        sequence=sequence,
    )
    session.add(model)
    parent.updated_at = utc_now()
    session.flush()
    return _message_read(model)


def list_messages(session: Session, session_id: str) -> list[MessageRead]:
    _require_session(session, session_id)
    models = session.scalars(
        select(MessageModel)
        .where(MessageModel.session_id == session_id)
        .order_by(MessageModel.sequence, MessageModel.message_id)
    )
    return [_message_read(model) for model in models]


def delete_session(session: Session, session_id: str) -> bool:
    result = session.execute(
        delete(SessionModel).where(SessionModel.session_id == session_id)
    )
    return bool(result.rowcount)


def create_eval_run(session: Session, payload: EvalRunCreate) -> EvalRunRead:
    model = EvalRunModel(
        run_id=payload.run_id,
        evaluation_name=payload.evaluation_name,
        schema_version=payload.schema_version,
        status=payload.status,
        input_artifact_ref=payload.input_artifact_ref,
        input_summary=payload.input_summary,
        config_metadata=payload.config_metadata,
    )
    if payload.status in {"completed", "failed"}:
        model.completed_at = utc_now()
    session.add(model)
    session.flush()
    return _eval_run_read(model)


def get_eval_run(session: Session, run_id: str) -> EvalRunRead | None:
    model = session.get(EvalRunModel, run_id)
    return None if model is None else _eval_run_read(model)


def update_eval_run_status(
    session: Session,
    run_id: str,
    payload: EvalRunStatusUpdate,
) -> EvalRunRead:
    model = _require_eval_run(session, run_id)
    model.status = payload.status
    model.updated_at = utc_now()
    model.completed_at = (
        utc_now() if payload.status in {"completed", "failed"} else None
    )
    session.flush()
    return _eval_run_read(model)


def create_eval_result(
    session: Session,
    run_id: str,
    payload: EvalResultCreate,
) -> EvalResultRead:
    _require_eval_run(session, run_id)
    model = EvalResultModel(
        result_id=payload.result_id,
        eval_run_id=run_id,
        case_id=payload.case_id,
        method_name=payload.method_name,
        metrics_metadata=payload.metrics_metadata,
    )
    session.add(model)
    session.flush()
    return _eval_result_read(model)


def list_eval_results(session: Session, run_id: str) -> list[EvalResultRead]:
    _require_eval_run(session, run_id)
    models = session.scalars(
        select(EvalResultModel)
        .where(EvalResultModel.eval_run_id == run_id)
        .order_by(EvalResultModel.created_at, EvalResultModel.result_id)
    )
    return [_eval_result_read(model) for model in models]


def delete_eval_run(session: Session, run_id: str) -> bool:
    result = session.execute(delete(EvalRunModel).where(EvalRunModel.run_id == run_id))
    return bool(result.rowcount)


def _require_document(session: Session, document_id: str) -> DocumentModel:
    model = session.get(DocumentModel, document_id)
    if model is None:
        raise StorageNotFoundError(f"找不到 document：{document_id}")
    return model


def _require_session(session: Session, session_id: str) -> SessionModel:
    model = session.get(SessionModel, session_id)
    if model is None:
        raise StorageNotFoundError(f"找不到 session：{session_id}")
    return model


def _require_eval_run(session: Session, run_id: str) -> EvalRunModel:
    model = session.get(EvalRunModel, run_id)
    if model is None:
        raise StorageNotFoundError(f"找不到 eval run：{run_id}")
    return model


def _document_read(model: DocumentModel) -> DocumentRead:
    return DocumentRead(
        document_id=model.document_id,
        episode_id=model.episode_id,
        source_file=model.source_file,
        source_format=model.source_format,
        title=model.title,
        input_summary=model.input_summary,
        metadata=model.metadata_json,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def to_session_read(model: SessionModel) -> SessionRead:
    """Project an internal ORM session row to the strict storage contract."""
    return SessionRead(
        session_id=model.session_id,
        title=model.title,
        status=model.status,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _message_read(model: MessageModel) -> MessageRead:
    return MessageRead(
        message_id=model.message_id,
        session_id=model.session_id,
        role=model.role,
        content=model.content,
        sequence=model.sequence,
        created_at=model.created_at,
    )


def _eval_run_read(model: EvalRunModel) -> EvalRunRead:
    return EvalRunRead(
        run_id=model.run_id,
        evaluation_name=model.evaluation_name,
        schema_version=model.schema_version,
        status=model.status,
        input_artifact_ref=model.input_artifact_ref,
        input_summary=model.input_summary,
        config_metadata=model.config_metadata,
        created_at=model.created_at,
        updated_at=model.updated_at,
        completed_at=model.completed_at,
    )


def _eval_result_read(model: EvalResultModel) -> EvalResultRead:
    return EvalResultRead(
        result_id=model.result_id,
        eval_run_id=model.eval_run_id,
        case_id=model.case_id,
        method_name=model.method_name,
        metrics_metadata=model.metrics_metadata,
        created_at=model.created_at,
    )
