"""Book endpoints: upload, list, download, delete."""
from __future__ import annotations

import io
import logging
import time
import zipfile
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse

from ..core.models import BookListResponse, UploadResponse
from ..core.service import BookService
from ..core.storage import BookRepository
from .deps import get_book_service, get_repository


log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/books", tags=["books"])


@router.post("/upload", response_model=UploadResponse)
async def upload_book(
    file: Annotated[UploadFile, File(...)],
    service: Annotated[BookService, Depends(get_book_service)],
    background_tasks: BackgroundTasks,
) -> UploadResponse:
    filename = file.filename or "untitled.txt"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"txt", "epub"}:
        log.warning("upload rejected: unsupported ext .%s (filename=%s)", ext, filename)
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type: .{ext} (only .txt and .epub)",
        )

    raw = await file.read()
    if not raw:
        log.warning("upload rejected: empty file (filename=%s)", filename)
        raise HTTPException(status_code=400, detail="empty file")

    log.info(
        "upload received: filename=%s ext=%s size=%d bytes",
        filename, ext, len(raw),
    )
    t0 = time.monotonic()
    try:
        if ext == "epub":
            meta = service.ingest_epub(raw, source_filename=filename)
        else:
            meta = service.ingest_txt(raw, source_filename=filename)
    except ValueError as exc:
        log.warning("upload rejected: parse failed (filename=%s): %s", filename, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed = time.monotonic() - t0
    log.info(
        "upload complete: book_id=%s title=%r chapters=%d took=%.1fs",
        meta.book_id, meta.title, len(meta.chapters), elapsed,
    )
    # Character extraction + profile analysis runs after the response is sent.
    # Under Plan C this just flips status to ready (no upfront character work).
    background_tasks.add_task(service.analyze_book, meta.book_id)
    return UploadResponse(
        book_id=meta.book_id,
        status=meta.status,
        title=meta.title,
        chapter_count=len(meta.chapters),
    )


@router.get("", response_model=BookListResponse)
async def list_books(
    service: Annotated[BookService, Depends(get_book_service)],
) -> BookListResponse:
    return service.list_books()


@router.delete("/{book_id}", status_code=204)
async def delete_book(
    book_id: str,
    service: Annotated[BookService, Depends(get_book_service)],
) -> Response:
    """Remove a book and all files produced by its import.

    See ``docs/technical-plan.md §2.2.1`` for semantics (what gets cleaned
    up, what doesn't, idempotency).
    """
    try:
        service.delete_book(book_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="book not found") from exc
    return Response(status_code=204)


@router.get("/{book_id}/download")
async def download_book(
    book_id: str,
    repo: Annotated[BookRepository, Depends(get_repository)],
):
    """Bundle meta.json + all chapter texts (no chapter metadata) into a zip stream.

    App-side this is unpacked into ``<Documents>/library/<book_id>/``.
    """
    if not repo.exists(book_id):
        raise HTTPException(status_code=404, detail="book not found")

    meta = repo.load_meta(book_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", meta.model_dump_json(indent=2))
        for chapter in meta.chapters:
            src = repo.book_dir(book_id) / chapter.text_file
            if src.exists():
                zf.write(src, arcname=chapter.text_file)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{book_id}.zip"'},
    )
