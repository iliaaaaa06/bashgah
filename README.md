# دستیار اداری هوشمند

پاسخ‌گویی به پرسش‌های فارسی: اول از اسناد سازمان، بعد از دانش خود مدل و در نهایت از جستجوی وب.

## ساختار پروژه

```
bashgah/
├── rag_server.py         # کل بک‌اند: FastAPI، RAG، ChromaDB، مدل زبانی، جستجوی وب، پرامپت‌ها و پیام‌های فارسی
├── frontend_app.py       # کل رابط کاربری Streamlit (گفتگو + مدیریت اسناد)؛ فقط از طریق HTTP به بک‌اند وصل می‌شود
├── .env                  # تنظیمات و آدرس‌ها (در git نیست)
├── .env.example          # الگوی تنظیمات
├── requirements.txt
├── scripts/
│   ├── start_all.sh        # بررسی اتصال + اجرای بک‌اند و رابط کاربری
│   ├── start_frontend.sh   # اجرای رابط کاربری
│   ├── start_llm.sh        # راه‌اندازی llama-server (فقط اگر مدل را روی همین سیستم اجرا کنید)
│   └── check_connection.py # بررسی اتصال به سرور مدل
└── data/                 # ChromaDB و فایل‌های آپلودشده (خودکار ساخته می‌شود)
```

## معماری

```
UI ──HTTP──► FastAPI (:8000) ──► bge-m3 (embedding, داخل همین پروسه)
                  │          ──► ChromaDB (data/chroma, روی دیسک)
                  │          ──► DuckDuckGo / Tavily
                  └──OpenAI API──► llama-server (:8080, روی GPU)
```

- **مدل زبانی پیشنهادی برای RTX 5070 (12GB):** `Qwen2.5-7B-Instruct` با کوانتیزه‌سازی `Q5_K_M` (حدود ۵٫۴ گیگ). با embedding و KV cache تقریباً ۹ گیگ مصرف می‌کند. اگر فارسیِ روان‌تری لازم داشتید `gemma-3-12b-it` با `Q4_K_M` (حدود ۷٫۳ گیگ) هم جا می‌شود، اما باید `LLAMA_CTX_SIZE` را کمتر کنید.
- **Embedding:** `BAAI/bge-m3`. چندزبانه است، روی فارسی عملکرد خوبی دارد، به prefix نیاز ندارد و روی GPU با fp16 حدود ۱٫۲ گیگ جا می‌گیرد.
- **Stateless بودن temperature:** هر درخواست `/chat` دمای خودش را به llama-server می‌فرستد. llama-server با `-np` چند slot مستقل دارد، پس تنظیمات یک کاربر روی کاربر دیگر اثر نمی‌گذارد.

### منطق پاسخ‌دهی
1. **اسناد:** ابتدا در ChromaDB جستجو می‌شود (فقط قطعه‌هایی که شباهتشان بیشتر از `RELEVANCE_THRESHOLD` است). اگر جواب در اسناد بود، همان برگردانده می‌شود و جستجوی وب انجام نمی‌شود.
2. **دانش مدل:** اگر جواب در اسناد نبود، مدل از دانش خودش پاسخ می‌دهد. برای پرسش‌هایی که به اطلاعات به‌روز نیاز دارند (قیمت، اخبار، بخشنامه‌ی جدید و ...) یا وقتی مطمئن نیست، پاسخ نمی‌دهد و کار به مرحله‌ی بعد می‌رسد.
3. **جستجوی وب:** مدل از روی نتایج جستجو پاسخ می‌دهد و در انتهای پاسخ فقط لینک منابع نمایش داده می‌شود.
4. اگر هیچ‌کدام جواب نداشت، جمله‌ی ثابت «متاسفانه نه در فایل‌های من، نه در دانش خودم و نه در جستجوی وب ...» برگردانده می‌شود. با `WEB_SEARCH_ENABLED=false` مرحله‌ی ۳ انجام نمی‌شود.

## اجرا

### پیش‌نیاز
- Python 3.12 (پایتون 3.14 هنوز با torch/chromadb سازگار نیست)
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (`llama-server`)

### روی مک (همین سیستم، برای تست)
```bash
brew install llama.cpp
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env        # ADMIN_API_KEY را عوض کنید

./scripts/start_llm.sh                       # ترمینال ۱ — بار اول مدل را دانلود می‌کند
.venv/bin/python rag_server.py              # ترمینال ۲ — API روی :8000
.venv/bin/python frontend_app.py            # ترمینال ۳ — UI روی :8501
```

### روی سیستم اصلی (RTX 5070، لینوکس یا WSL2)
```bash
# llama.cpp با CUDA (کارت‌های سری 50 / Blackwell به CUDA 12.8 یا بالاتر نیاز دارند)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j 32 --config Release
export PATH="$PWD/build/bin:$PATH"; cd -

python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # پیش‌فرض‌های این فایل برای 12GB تنظیم شده‌اند

./scripts/start_llm.sh
.venv/bin/python rag_server.py
.venv/bin/python frontend_app.py
```
در ویندوز بدون WSL می‌توانید باینری آماده‌ی CUDA را از صفحه‌ی Releases پروژه‌ی llama.cpp دانلود کنید و `llama-server.exe` را با همان آرگومان‌های `scripts/start_llm.sh` اجرا کنید.

اگر به‌جای llama.cpp از vLLM استفاده کنید، فقط `LLM_BASE_URL` و `LLM_MODEL` را در `.env` عوض کنید.

**اتصال به سرورِ آماده:** اگر مدل زبانی و embedding از قبل روی سرور اجرا می‌شوند، `scripts/start_llm.sh` لازم نیست. فقط `LLM_BASE_URL` و `LLM_MODEL` را پر کنید و برای embedding یکی از این دو را تنظیم کنید:
- `EMBEDDING_BASE_URL`: آدرس endpoint سازگار با OpenAI (`/v1/embeddings`)، با `EMBEDDING_MODEL` برابر نام مدل روی همان سرور
- یا `EMBEDDING_BASE_URL` خالی و `EMBEDDING_MODEL` برابر مسیر پوشه‌ی مدل روی دیسک

هر دو فایل `.env` را از کنار خودشان می‌خوانند. اگر سیستم چند IP دارد و Streamlit آدرس اشتباهی به‌عنوان «Network URL» چاپ می‌کند، `FRONTEND_PUBLIC_HOST` را در `.env` تنظیم کنید.

## رابط کاربری و API

رابط کاربری (Streamlit): `http://localhost:8501/`. صفحه‌ی «گفتگو» برای کاربران است و صفحه‌ی «مدیریت اسناد» با کلید مدیر باز می‌شود. اگر فرانت روی سیستم دیگری اجرا می‌شود، `BACKEND_URL` را در `.env` تنظیم کنید.

مستندات تعاملی API: `http://localhost:8000/docs`

| متد | مسیر | دسترسی | توضیح |
|---|---|---|---|
| GET | `/health` | عمومی | وضعیت سرویس، مدل و تعداد قطعه‌های ایندکس‌شده |
| POST | `/chat` | کاربران | `{"message": "...", "temperature": 0.2, "top_k": 5}` |
| POST | `/admin/upload` | مدیر | multipart با فیلد `files` (یک یا چند فایل) |
| GET | `/admin/documents` | مدیر | فهرست اسناد |
| DELETE | `/admin/documents/{doc_id}` | مدیر | حذف یک سند |

برای مسیرهای مدیر، هدر `X-Admin-Key` با مقدار `ADMIN_API_KEY` لازم است.

```bash
curl -X POST localhost:8000/admin/upload -H "X-Admin-Key: $KEY" -F "files=@آیین‌نامه.pdf"
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "مرخصی استحقاقی سالانه چند روز است؟", "temperature": 0.1}'
```

پاسخ `/chat`:
```json
{"answer": "...", "source": "documents | model | web | none", "sources": [{"title": "...", "page": 3, "score": 0.71, "snippet": "..."}], "temperature": 0.1}
```

## نکات
- PDF اسکن‌شده (تصویری) متن ندارد و با پیام خطا رد می‌شود. برای این نوع فایل‌ها باید OCR اضافه شود.
- اگر پاسخ‌ها از اسناد نامرتبط ساخته می‌شوند، `RELEVANCE_THRESHOLD` را بیشتر کنید (مثلاً ۰٫۵۵). اگر اسناد مرتبط پیدا نمی‌شوند، آن را کمتر کنید.
- با تغییر `EMBEDDING_MODEL`، پوشه‌ی `data/chroma` را پاک کنید و فایل‌ها را دوباره آپلود کنید.
