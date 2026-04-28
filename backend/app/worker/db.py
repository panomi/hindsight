"""Sync DB session for Celery tasks (Celery is sync-first)."""
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

settings = get_settings()

engine_sync = create_engine(settings.database_url_sync, pool_pre_ping=True, future=True)
SessionSync = sessionmaker(engine_sync, expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Session:
    s = SessionSync()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
