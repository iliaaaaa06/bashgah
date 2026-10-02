"""RAG pipeline: Persian chunking, embeddings, ChromaDB retrieval, LLM generation and web-search fallback."""
import asyncio
import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, OpenAI

from app import prompts
from app.config import Settings
from app.document_loader import Page
from app.persian import CHUNK_SEPARATORS, normalize

logger = logging.getLogger(__name__)

AnswerSource = Literal["documents", "web", "none"]
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMUnavailableError(Exception):
    pass


class EmbeddingUnavailableError(Exception):
    pass


@dataclass
class SourceRef:
    title: str
    doc_id: str | None = None
    page: int | None = None
    url: str | None = None
    score: float | None = None
    snippet: str | None = None


@dataclass
class Answer:
    answer: str
    source: AnswerSource
    sources: list[SourceRef] = field(default_factory=list)


@dataclass
class IngestResult:
    doc_id: str
    chunks: int
    duplicate: bool


# --------------------------------------------------------------------------- embeddings
class Embedder:
    """Runs a multilingual sentence-transformers model (default: BAAI/bge-m3) inside this process.

    EMBEDDING_MODEL may be a local folder, so nothing is downloaded when the model already exists on disk.
    """

    def __init__(self, model_name: str, device: str, batch_size: int):
        from sentence_transformers import SentenceTransformer
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading embedding model %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        if device == "cuda":
            self.model.half()  # halves VRAM, leaves more room for the LLM
        self.batch_size = batch_size
        self.device = device
        # Torch inference on a shared model is not guaranteed thread-safe (esp. on MPS)
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            vectors = self.model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
            )
        return vectors.tolist()


class RemoteEmbedder:
    """Calls an OpenAI-compatible /v1/embeddings endpoint (llama-server --embeddings, vLLM, TEI, Infinity, ...)."""

    def __init__(self, base_url: str, api_key: str, model_name: str, batch_size: int, timeout: float):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        self.model = model_name
        self.batch_size = batch_size
        self.device = f"remote ({base_url})"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            try:
                response = self.client.embeddings.create(model=self.model, input=texts[start:start + self.batch_size])
            except (APIConnectionError, APITimeoutError, APIError) as exc:
                logger.error("Embedding request failed: %s", exc)
                raise EmbeddingUnavailableError from exc
            for item in sorted(response.data, key=lambda d: d.index):
                norm = sum(x * x for x in item.embedding) ** 0.5 or 1.0
                vectors.append([x / norm for x in item.embedding])
        return vectors


def build_embedder(settings: Settings) -> Embedder | RemoteEmbedder:
    if settings.embedding_base_url:
        return RemoteEmbedder(
            settings.embedding_base_url, settings.embedding_api_key, settings.embedding_model,
            settings.embedding_batch_size, settings.llm_timeout,
        )
    return Embedder(settings.embedding_model, settings.embedding_device, settings.embedding_batch_size)


# --------------------------------------------------------------------------- LLM
class LLMClient:
    """Stateless client for an OpenAI-compatible server (llama.cpp llama-server, vLLM, ...).

    Every call sends its own sampling parameters, so one user's temperature never affects another's.
    """

    def __init__(self, settings: Settings):
        self.client = AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=settings.llm_timeout, max_retries=1
        )
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens

    async def chat(self, system: str, user: str, temperature: float, max_tokens: int | None = None) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
        except (APIConnectionError, APITimeoutError, APIError) as exc:
            logger.error("LLM request failed: %s", exc)
            raise LLMUnavailableError from exc
        text = response.choices[0].message.content or ""
        return _THINK_BLOCK.sub("", text).strip()

    async def is_alive(self) -> bool:
        try:
            await self.client.models.list(timeout=3)
            return True
        except Exception:
            return False


# --------------------------------------------------------------------------- web search
class WebSearcher:
    def __init__(self, settings: Settings):
        self.provider = settings.web_search_provider.lower()
        self.max_results = settings.web_search_max_results
        self.region = settings.web_search_region
        self.tavily_key = settings.tavily_api_key

    def search(self, query: str) -> list[SourceRef]:
        try:
            if self.provider == "tavily" and self.tavily_key:
                return self._tavily(query)
            return self._duckduckgo(query)
        except Exception as exc:
            logger.warning("Web search failed (%s): %s", self.provider, exc)
            return []

    def _duckduckgo(self, query: str) -> list[SourceRef]:
        from ddgs import DDGS

        results = DDGS().text(query, region=self.region, max_results=self.max_results) or []
        return [SourceRef(title=r.get("title", ""), url=r.get("href"), snippet=r.get("body", "")) for r in results]

    def _tavily(self, query: str) -> list[SourceRef]:
        from tavily import TavilyClient

        data = TavilyClient(api_key=self.tavily_key).search(query, max_results=self.max_results)
        return [
            SourceRef(title=r.get("title", ""), url=r.get("url"), snippet=r.get("content", ""), score=r.get("score"))
            for r in data.get("results", [])
        ]


# --------------------------------------------------------------------------- engine
class RAGEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._embedder: Embedder | RemoteEmbedder | None = None
        self._embedder_lock = threading.Lock()
        self.chroma = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.chroma.get_or_create_collection(
            name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=CHUNK_SEPARATORS,
            keep_separator="end",
            length_function=len,
        )
        self.llm = LLMClient(settings)
        self.web = WebSearcher(settings)
        self._write_lock = threading.Lock()

    @property
    def embedder(self) -> Embedder | RemoteEmbedder:
        """Built on first use, so the API (and UI) can start before the embedding model is reachable."""
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    try:
                        self._embedder = build_embedder(self.settings)
                    except Exception as exc:
                        logger.error("Embedding model could not be loaded: %s", exc)
                        raise EmbeddingUnavailableError from exc
        return self._embedder

    @property
    def embedding_status(self) -> str:
        if self._embedder is not None:
            return self._embedder.device
        return f"remote ({self.settings.embedding_base_url})" if self.settings.embedding_base_url else "not loaded"

    # ----------------------------------------------------------------- ingestion (admin)
    @staticmethod
    def compute_doc_id(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    def has_document(self, doc_id: str) -> bool:
        return bool(self.collection.get(where={"doc_id": doc_id}, limit=1, include=[])["ids"])

    def ingest(self, doc_id: str, filename: str, pages: list[Page]) -> IngestResult:
        """Chunk, embed and store a document. Blocking — call from a worker thread."""
        with self._write_lock:
            if self.has_document(doc_id):
                return IngestResult(doc_id=doc_id, chunks=0, duplicate=True)

            uploaded_at = datetime.now(timezone.utc).isoformat()
            ids, texts, metadatas = [], [], []
            for page in pages:
                for chunk in self.splitter.split_text(page.text):
                    chunk = chunk.strip()
                    if len(chunk) < 20:  # drop page numbers, headers, noise
                        continue
                    metadata = {
                        "doc_id": doc_id,
                        "filename": filename,
                        "chunk_index": len(ids),
                        "uploaded_at": uploaded_at,
                    }
                    if page.page is not None:
                        metadata["page"] = page.page
                    ids.append(f"{doc_id}:{len(ids)}")
                    texts.append(chunk)
                    metadatas.append(metadata)

            if not texts:
                return IngestResult(doc_id=doc_id, chunks=0, duplicate=False)

            batch = 256
            for start in range(0, len(texts), batch):
                end = start + batch
                self.collection.add(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=self.embedder.embed(texts[start:end]),
                )
            logger.info("Ingested %s (%s) — %d chunks", filename, doc_id, len(texts))
            return IngestResult(doc_id=doc_id, chunks=len(texts), duplicate=False)

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for meta in self.collection.get(include=["metadatas"])["metadatas"] or []:
            entry = docs.setdefault(
                meta["doc_id"],
                {"doc_id": meta["doc_id"], "filename": meta["filename"], "uploaded_at": meta["uploaded_at"], "chunks": 0},
            )
            entry["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["uploaded_at"], reverse=True)

    def delete_document(self, doc_id: str) -> bool:
        with self._write_lock:
            if not self.has_document(doc_id):
                return False
            self.collection.delete(where={"doc_id": doc_id})
            return True

    def chunk_count(self) -> int:
        return self.collection.count()

    # ----------------------------------------------------------------- retrieval
    def retrieve(self, query: str, top_k: int) -> list[SourceRef]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            score = 1.0 - distance  # cosine distance -> similarity
            if score >= self.settings.relevance_threshold:
                hits.append(
                    SourceRef(
                        title=meta["filename"], doc_id=meta["doc_id"], page=meta.get("page"),
                        score=round(score, 4), snippet=text,
                    )
                )
        return hits

    # ----------------------------------------------------------------- answering (users)
    async def answer(self, question: str, temperature: float, top_k: int | None = None) -> Answer:
        question = normalize(question)
        top_k = top_k or self.settings.retrieval_top_k

        # 1) Knowledge base first — always.
        hits = await asyncio.to_thread(self.retrieve, question, top_k)
        if hits:
            text = await self.llm.chat(
                prompts.DOCS_SYSTEM_PROMPT,
                prompts.DOCS_USER_TEMPLATE.format(context=_format_context(hits, prompts.DOC_LABEL), question=question),
                temperature=temperature,
            )
            if prompts.NOT_FOUND_MARKER not in normalize(text):
                return Answer(answer=text, source="documents", sources=_strip_snippets(hits))

        if not self.settings.web_search_enabled:
            return Answer(answer=prompts.NOT_FOUND_ANSWER, source="none")

        # 2) Nothing usable in the files: always search the web; the model decides if the results answer it.
        results = await asyncio.to_thread(self.web.search, question)
        if results:
            text = await self.llm.chat(
                prompts.WEB_SYSTEM_PROMPT,
                prompts.WEB_USER_TEMPLATE.format(context=_format_context(results, prompts.WEB_LABEL), question=question),
                temperature=temperature,
            )
            if prompts.WEB_NOT_FOUND_MARKER not in normalize(text):
                return Answer(answer=text, source="web", sources=_strip_snippets(results))

        # 3) Neither the files nor the web had an answer: refuse, never guess.
        return Answer(answer=prompts.NO_ANSWER, source="none")


def _format_context(refs: list[SourceRef], label: str) -> str:
    blocks = []
    for i, ref in enumerate(refs, start=1):
        header = f"[{label} {_fa_digits(i)}] {ref.title}"
        if ref.page:
            header += f" — صفحه {_fa_digits(ref.page)}"
        if ref.url:
            header += f" — {ref.url}"
        blocks.append(f"{header}\n{ref.snippet or ''}")
    return "\n\n".join(blocks)


def _strip_snippets(refs: list[SourceRef], max_len: int = 300) -> list[SourceRef]:
    for ref in refs:
        if ref.snippet and len(ref.snippet) > max_len:
            ref.snippet = ref.snippet[:max_len] + "…"
    return refs


def _fa_digits(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))
