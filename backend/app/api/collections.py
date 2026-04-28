from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import CollectionCreate, CollectionOut
from app.database import get_session
from app.models import Collection, Video

router = APIRouter(prefix="/api/collections", tags=["collections"])


@router.post("", response_model=CollectionOut, status_code=status.HTTP_201_CREATED)
async def create_collection(
    payload: CollectionCreate, session: AsyncSession = Depends(get_session)
) -> CollectionOut:
    c = Collection(name=payload.name, description=payload.description)
    session.add(c)
    await session.commit()
    await session.refresh(c)
    return CollectionOut(
        id=c.id, name=c.name, description=c.description,
        created_at=c.created_at, video_count=0,
    )


@router.get("", response_model=list[CollectionOut])
async def list_collections(session: AsyncSession = Depends(get_session)) -> list[CollectionOut]:
    stmt = (
        select(Collection, func.count(Video.id))
        .outerjoin(Video, Video.collection_id == Collection.id)
        .group_by(Collection.id)
        .order_by(Collection.created_at.desc())
    )
    rows = (await session.execute(stmt)).all()
    return [
        CollectionOut(
            id=c.id, name=c.name, description=c.description,
            created_at=c.created_at, video_count=cnt,
        )
        for c, cnt in rows
    ]


@router.get("/{collection_id}", response_model=CollectionOut)
async def get_collection(
    collection_id: UUID, session: AsyncSession = Depends(get_session)
) -> CollectionOut:
    c = await session.get(Collection, collection_id)
    if c is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "collection not found")
    cnt = await session.scalar(
        select(func.count(Video.id)).where(Video.collection_id == collection_id)
    )
    return CollectionOut(
        id=c.id, name=c.name, description=c.description,
        created_at=c.created_at, video_count=cnt or 0,
    )


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: UUID, session: AsyncSession = Depends(get_session)
) -> None:
    c = await session.get(Collection, collection_id)
    if c is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "collection not found")
    await session.delete(c)
    await session.commit()
