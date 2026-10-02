"""Checks the model server settings in .env before starting the backend.

Run from the project root:  .venv/bin/python scripts/check_connection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from rag_server import settings  # noqa: E402

OK, FAIL = "✅", "❌"


def main() -> int:
    s = settings
    if "SERVER_IP_HERE" in s.llm_base_url:
        print(f"{FAIL} هنوز IP سرور را در .env وارد نکرده‌اید (MODEL_SERVER_IP).")
        return 1
    failed = False

    print(f"\n— مدل زبانی: {s.llm_base_url}")
    llm = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=s.llm_timeout, max_retries=0)
    try:
        models = [m.id for m in llm.with_options(timeout=10).models.list().data]
        print(f"{OK} سرور در دسترس است. مدل‌های موجود: {', '.join(models) or '(خالی)'}")
    except Exception as exc:
        print(f"{FAIL} اتصال برقرار نشد: {exc}")
        print("   IP و پورت، اجرای سرویس روی 0.0.0.0 (در Ollama: OLLAMA_HOST=0.0.0.0) و فایروال سرور را بررسی کنید.")
        return 1

    if models and s.llm_model not in models:
        print(f"{FAIL} مدل «{s.llm_model}» روی سرور نیست. LLM_MODEL را در .env به یکی از نام‌های بالا تغییر دهید.")
        failed = True
    else:
        print("   در حال ارسال یک سؤال آزمایشی (بار اول ممکن است تا لود شدن مدل طول بکشد)…")
        try:
            r = llm.chat.completions.create(
                model=s.llm_model,
                messages=[{"role": "user", "content": "فقط با یک جمله‌ی کوتاه فارسی سلام کن."}],
                temperature=0.2,
                max_tokens=200,
            )
            print(f"{OK} پاسخ مدل: {(r.choices[0].message.content or '').strip()[:200]}")
        except Exception as exc:
            print(f"{FAIL} مدل پاسخ نداد: {exc}")
            failed = True

    if s.embedding_base_url:
        print(f"\n— Embedding: {s.embedding_base_url}  (مدل: {s.embedding_model})")
        if models and s.embedding_model not in models and f"{s.embedding_model}:latest" not in models:
            print(f"{FAIL} مدل «{s.embedding_model}» روی سرور نیست (در Ollama: ollama pull {s.embedding_model}).")
            failed = True
        emb = OpenAI(base_url=s.embedding_base_url, api_key=s.embedding_api_key, timeout=s.llm_timeout, max_retries=0)
        try:
            r = emb.embeddings.create(model=s.embedding_model, input=["آزمایش اتصال"])
            print(f"{OK} بردار با طول {len(r.data[0].embedding)} دریافت شد.")
        except Exception as exc:
            print(f"{FAIL} سرویس embedding پاسخ نداد: {exc}")
            failed = True
    else:
        print(f"\n— Embedding: محلی ({s.embedding_model}) — روی همین سیستم لود می‌شود.")

    print("\nنتیجه:", "مشکل دارد — موارد بالا را اصلاح کنید." if failed else "همه‌چیز آماده است ✅")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
