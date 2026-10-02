"""FastAPI entrypoint for the Persian administrative AI assistant."""
import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import prompts
from app.config import get_settings
from app.document_loader import SUPPORTED_EXTENSIONS, DocumentReadError, load_document
from app.rag_engine import EmbeddingUnavailableError, LLMUnavailableError, RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("assistant")
settings = get_settings()
MSG = prompts.MSG


class UTF8JSONResponse(JSONResponse):
    """JSON with an explicit charset and raw (non-escaped) Persian characters."""

    media_type = "application/json; charset=utf-8"


# --------------------------------------------------------------------------- schemas
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="پرسش کاربر به فارسی")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="دمای نمونه‌برداری مخصوص همین درخواست")
    top_k: int | None = Field(None, ge=1, le=20, description="تعداد قطعات بازیابی‌شده (اختیاری)")

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(MSG["empty_message"])
        return v


class SourceOut(BaseModel):
    title: str
    doc_id: str | None = None
    page: int | None = None
    url: str | None = None
    score: float | None = None
    snippet: str | None = None


class ChatResponse(BaseModel):
    answer: str
    source: Literal["documents", "web", "none"]
    sources: list[SourceOut]
    temperature: float


class FileResult(BaseModel):
    filename: str
    status: Literal["ingested", "duplicate", "error"]
    message: str
    doc_id: str | None = None
    chunks: int = 0


class UploadResponse(BaseModel):
    results: list[FileResult]


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    uploaded_at: str
    chunks: int


class MessageResponse(BaseModel):
    message: str


# --------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    app.state.engine = await asyncio.to_thread(RAGEngine, settings)
    if not await app.state.engine.llm.is_alive():
        logger.warning("LLM server at %s is not reachable yet", settings.llm_base_url)
    yield


app = FastAPI(
    title="دستیار اداری هوشمند",
    description="API دستیار اداری مبتنی بر RAG برای اسناد فارسی",
    version="1.0.0",
    default_response_class=UTF8JSONResponse,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine(request: Request) -> RAGEngine:
    return request.app.state.engine


_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def require_admin(api_key: str | None = Depends(_admin_key_header)) -> None:
    if not api_key or not secrets.compare_digest(api_key, settings.admin_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=MSG["unauthorized"])


# --------------------------------------------------------------------------- error handlers (Persian)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    # Replace framework-generated English messages ("Not Found", ...) with Persian ones
    if not isinstance(detail, str) or detail.isascii():
        detail = prompts.STATUS_MESSAGES.get(exc.status_code, MSG["internal_error"])
    return UTF8JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = str(err["loc"][-1]) if err.get("loc") else ""
        ctx = err.get("ctx") or {}
        if err["type"] == "value_error" and isinstance(ctx.get("error"), ValueError):
            text = str(ctx["error"])  # our own Persian validator messages
        else:
            template = prompts.VALIDATION_FIELD_MESSAGES.get(err["type"], prompts.VALIDATION_DEFAULT)
            text = template.format(field=field, **{k: v for k, v in ctx.items() if k in ("ge", "le")})
        errors.append({"field": field, "message": text})
    return UTF8JSONResponse(
        status_code=422, content={"detail": MSG["validation_error"], "errors": errors}
    )


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(request: Request, exc: LLMUnavailableError):
    return UTF8JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": MSG["llm_unavailable"]})


@app.exception_handler(EmbeddingUnavailableError)
async def embedding_unavailable_handler(request: Request, exc: EmbeddingUnavailableError):
    return UTF8JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": MSG["embedding_unavailable"]}
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return UTF8JSONResponse(status_code=500, content={"detail": MSG["internal_error"]})


# --------------------------------------------------------------------------- public endpoints
@app.get("/health", tags=["عمومی"])
async def health(engine: RAGEngine = Depends(get_engine)):
    return {
        "status": MSG["status_ok"],
        "llm_available": await engine.llm.is_alive(),
        "embedding": engine.embedding_status,
        "indexed_chunks": await asyncio.to_thread(engine.chunk_count),
    }


@app.post("/chat", response_model=ChatResponse, tags=["کاربران"])
async def chat(body: ChatRequest, engine: RAGEngine = Depends(get_engine)):
    result = await engine.answer(body.message, temperature=body.temperature, top_k=body.top_k)
    return ChatResponse(
        answer=result.answer,
        source=result.source,
        sources=[SourceOut(**vars(s)) for s in result.sources],
        temperature=body.temperature,
    )


# --------------------------------------------------------------------------- admin endpoints
@app.post("/admin/upload", response_model=UploadResponse, tags=["مدیر"], dependencies=[Depends(require_admin)])
async def admin_upload(files: list[UploadFile] = File(...), engine: RAGEngine = Depends(get_engine)):
    if not files:
        raise HTTPException(status_code=400, detail=MSG["no_files"])
    max_bytes = settings.max_upload_mb * 1024 * 1024
    results = []
    for upload in files:
        name = Path(upload.filename or "file").name
        ext = Path(name).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            results.append(FileResult(filename=name, status="error", message=MSG["unsupported_type"].format(name=name)))
            continue
        data = await upload.read(max_bytes + 1)
        if len(data) > max_bytes:
            results.append(FileResult(
                filename=name, status="error", message=MSG["too_large"].format(name=name, max_mb=settings.max_upload_mb)
            ))
            continue
        if not data:
            results.append(FileResult(filename=name, status="error", message=MSG["empty_file"].format(name=name)))
            continue

        doc_id = engine.compute_doc_id(data)
        try:
            pages = await asyncio.to_thread(load_document, data, ext)
        except DocumentReadError:
            results.append(FileResult(filename=name, status="error", message=MSG["read_error"].format(name=name)))
            continue
        if not pages:
            results.append(FileResult(filename=name, status="error", message=MSG["no_text"].format(name=name)))
            continue

        outcome = await asyncio.to_thread(engine.ingest, doc_id, name, pages)
        if outcome.duplicate:
            results.append(FileResult(
                filename=name, status="duplicate", doc_id=doc_id, message=MSG["duplicate"].format(name=name)
            ))
            continue
        if outcome.chunks == 0:
            results.append(FileResult(filename=name, status="error", message=MSG["no_text"].format(name=name)))
            continue
        (settings.upload_dir / f"{doc_id}{ext}").write_bytes(data)  # keep the original for re-indexing
        results.append(FileResult(
            filename=name, status="ingested", doc_id=doc_id, chunks=outcome.chunks,
            message=MSG["ingested"].format(name=name),
        ))
    return UploadResponse(results=results)


@app.get("/admin/documents", response_model=list[DocumentOut], tags=["مدیر"], dependencies=[Depends(require_admin)])
async def admin_list_documents(engine: RAGEngine = Depends(get_engine)):
    return await asyncio.to_thread(engine.list_documents)


@app.delete(
    "/admin/documents/{doc_id}", response_model=MessageResponse, tags=["مدیر"], dependencies=[Depends(require_admin)]
)
async def admin_delete_document(doc_id: str, engine: RAGEngine = Depends(get_engine)):
    if not await asyncio.to_thread(engine.delete_document, doc_id):
        raise HTTPException(status_code=404, detail=MSG["doc_not_found"])
    for path in settings.upload_dir.glob(f"{doc_id}.*"):
        path.unlink(missing_ok=True)
    return MessageResponse(message=MSG["doc_deleted"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port)
