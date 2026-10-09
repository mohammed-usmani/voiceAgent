from sqlalchemy.orm import Session

from voiceagent.db.models import Event


def emit(session: Session, entity: str, entity_id, type: str, data: dict | None = None) -> None:
    """Append to the event feed the frontend reads. Committed with the caller's change."""
    session.add(Event(entity=entity, entity_id=str(entity_id), type=type, data=data or {}))
