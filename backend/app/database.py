from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    # Defaults (5 + 10) get exhausted under thumbnail bursts.  Many video /
    # frame thumbnails on a results page each fire a parallel /file or
    # /image request, plus the agent loop holds connections too.  These
    # numbers are low-cost (each connection is a few KB) and give plenty
    # of headroom for typical use.
    pool_size=10,
    max_overflow=20,
    pool_timeout=15,  # fail faster than 30s so the UI can show an error
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
