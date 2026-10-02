"""Local test server: the same API as rag_server.py, but with no language model, embedding model or web search.

Every question gets a fixed test answer, document upload works with a toy embedding, and all data goes to
data_dev/ so it never mixes with the real data/.   Run:  python dev_server.py   (then python frontend_app.py)
"""
import hashlib

import uvicorn

import rag_server as rs

DEV_DATA = rs.BASE_DIR / "data_dev"
rs.settings.chroma_path = DEV_DATA / "chroma"
rs.settings.upload_dir = DEV_DATA / "uploads"
rs.settings.db_path = DEV_DATA / "app.db"
rs.settings.embedding_base_url = ""  # always use the fake embedder below
rs.settings.web_search_enabled = False
rs.settings.relevance_threshold = 0.2  # the toy embedding gives lower similarities than a real model


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words come out similar, which is enough to exercise retrieval."""

    device = "fake (dev_server)"

    def __init__(self, settings):
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * 256
            for word in rs.normalize(text).split():
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class FakeLLM:
    def __init__(self, settings):
        pass

    async def chat(self, system: str, user: str, temperature: float, history: list[dict] = ()) -> str:
        if system == rs.REWRITE_PROMPT:
            return user.rsplit("\n", 1)[-1]  # keep the follow-up question as it is
        if history:
            return f"این یک پاسخ آزمایشی است و {len(history)} پیام قبلی همین گفتگو را به خاطر دارم. (سرور تست — مدل زبانی وصل نیست)"
        if system == rs.DOCS_PROMPT:
            return "این یک پاسخ آزمایشی بر اساس اسناد بارگذاری‌شده است [سند ۱]. (سرور تست — مدل زبانی وصل نیست)"
        return "این یک پاسخ آزمایشی است. سرور تست به هیچ مدل زبانی وصل نیست."

    async def is_alive(self) -> bool:
        return True


rs.LocalEmbedder = FakeEmbedder
rs.LLM = FakeLLM

if __name__ == "__main__":
    rs.log.warning("DEV SERVER — no language model, no embedding model, no web search. Data in %s", DEV_DATA)
    uvicorn.run(rs.app, host=rs.settings.app_host, port=rs.settings.app_port)
