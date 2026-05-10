"""Book character roster endpoints.

The app's player-settings screen shows the cumulative character list
(``characters.json`` minus narrator slots) and lets the user tweak the
matcher inputs (gender / age / personality) for any one character. The
matcher then picks a different speaker on the next TTS request.

Concurrency: writes share the per-book lock with chapter analysis, so a
user edit waits behind any in-flight LLM merge but never fails. See
``BookService.update_book_character``.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from ..core.models import Character, CharacterUpdate
from ..core.service import BookService
from .deps import get_book_service


router = APIRouter(prefix="/api/books", tags=["characters"])


@router.get("/{book_id}/characters", response_model=list[Character])
async def list_characters(
    book_id: str,
    service: Annotated[BookService, Depends(get_book_service)],
) -> list[Character]:
    """Return the cumulative book roster, narrator slots filtered out.

    Empty list if the book hasn't had any chapter analysed yet (the
    roster is built lazily during chapter analysis). 404 only when the
    book itself doesn't exist.
    """
    if not service.repo.exists(book_id):
        raise HTTPException(status_code=404, detail="book not found")
    return service.list_book_characters(book_id)


@router.patch("/{book_id}/characters/{character_id}", response_model=Character)
async def update_character(
    book_id: str,
    character_id: int,
    update: CharacterUpdate,
    service: Annotated[BookService, Depends(get_book_service)],
) -> Character:
    """Partially update one character. Body fields are all optional —
    only fields that are present overwrite the stored values.

    The handler runs in a threadpool because the underlying
    ``update_book_character`` may block on the per-book lock while
    chapter analysis holds it (typically <1 s, occasionally a few
    seconds during a merge). Keeping the event loop free preserves
    other endpoints' responsiveness.
    """
    if not service.repo.exists(book_id):
        raise HTTPException(status_code=404, detail="book not found")
    try:
        return await run_in_threadpool(
            service.update_book_character, book_id, character_id, update,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"character {character_id} not in book",
        ) from exc
