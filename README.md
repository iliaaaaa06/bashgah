# دستیار اداری هوشمند

پاسخ‌گویی به پرسش‌های فارسی: اول از اسناد سازمان، بعد از دانش خود مدل و در نهایت از جستجوی وب.

## ساختار پروژه

```
bashgah/
├── rag_server.py         # کل بک‌اند: FastAPI، RAG، ChromaDB، مدل زبانی، جستجوی وب، پرامپت‌ها و پیام‌های فارسی
├── dev_server.py         # سرور تست برای لپ‌تاپ: همان API، بدون هیچ مدلی (پاسخ آزمایشی)
├── frontend_app.py       # کل رابط کاربری Streamlit (گفتگو + مدیریت اسناد)؛ فقط از طریق HTTP به بک‌اند وصل می‌شود
├── .env                  # تنظیمات و آدرس‌ها (در git نیست)
├── .env.example          # الگوی تنظیمات
├── requirements.txt
├── scripts/
│   ├── start_all.sh        # بررسی اتصال + اجرای بک‌اند و رابط کاربری
│   ├── start_frontend.sh   # اجرای رابط کاربری
│   ├── start_llm.sh        # راه‌اندازی llama-server (فقط اگر مدل را روی همین سیستم اجرا کنید)
│   └── check_connection.py # بررسی اتصال به سرور مدل
└── data/                 # ChromaDB، فایل‌های آپلودشده و app.db (کاربران و گفتگوها)، خودکار ساخته می‌شود
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

## تست روی لپ‌تاپ بدون مدل

`dev_server.py` همان API را بالا می‌آورد، ولی به هیچ مدل زبانی، مدل embedding یا جستجوی وب وصل نمی‌شود. به هر پرسش یک پاسخ آزمایشی ثابت می‌دهد و داده‌هایش را در `data_dev/` نگه می‌دارد تا با داده‌های واقعی قاطی نشود:
```bash
.venv/bin/python dev_server.py      # به‌جای rag_server.py
.venv/bin/python frontend_app.py
```
روی سرور اصلی همان `rag_server.py` را اجرا کنید.

## حساب کاربری و مدیر

- هر کسی از صفحه‌ی ورود می‌تواند «ثبت‌نام» کند. گفتگوهای هر کاربر جدا ذخیره می‌شوند و فقط خودش آن‌ها را می‌بیند.
- با دکمه‌ی «گفتگوی جدید» یک گفتگوی تازه شروع می‌شود و گفتگوهای قبلی در نوار کناری می‌مانند.
- مدل پیام‌های قبلیِ **همان گفتگو** را به خاطر دارد (به تعداد `CHAT_MEMORY_MESSAGES`، پیش‌فرض ۶)، پس پرسش‌های ادامه‌دار مثل «برای قراردادی‌ها چطور؟» هم جواب می‌گیرند. چنین پرسشی پیش از جستجو در اسناد و وب به یک پرسش کامل بازنویسی می‌شود. گفتگوهای دیگر و کاربران دیگر در این حافظه نیستند.
- هر کاربری که یک بار `ADMIN_PASSWORD` را در صفحه‌ی «مدیریت اسناد» وارد کند، برای همیشه مدیر می‌شود و می‌تواند اسناد را بارگذاری و حذف کند.
- کاربران، نشست‌ها و گفتگوها در جدول‌های `users`، `sessions`، `chats` و `messages` ذخیره می‌شوند. اگر `DB_NAME` پر باشد در PostgreSQL، وگرنه در فایل SQLite مسیر `DB_PATH`. جدول‌ها موقع اجرای سرور خودکار ساخته می‌شوند.
- ساخت پایگاه داده‌ی PostgreSQL (یک بار):
  ```bash
  psql -d postgres -c "CREATE ROLE bashgah LOGIN PASSWORD 'رمز'" -c "CREATE DATABASE autenticationbasgha OWNER bashgah"
  # در .env:  DB_NAME=autenticationbasgha  DB_USER=bashgah  DB_PASSWORD=رمز
  ```

## API

مستندات تعاملی: `http://localhost:8000/docs`. همه‌ی مسیرها به‌جز `/health`، `/auth/register` و `/auth/login` هدر `Authorization: Bearer <token>` لازم دارند.

| متد | مسیر | دسترسی | توضیح |
|---|---|---|---|
| GET | `/health` | عمومی | وضعیت سرویس، مدل و تعداد قطعه‌های ایندکس‌شده |
| POST | `/auth/register`، `/auth/login` | عمومی | `{"username": "...", "password": "..."}` ← `{"token": "...", "user": {...}}` |
| POST | `/auth/logout` · GET `/auth/me` | کاربر | خروج · اطلاعات کاربر |
| POST | `/auth/become-admin` | کاربر | `{"password": "<ADMIN_PASSWORD>"}` |
| GET / POST | `/chats` | کاربر | فهرست گفتگوهای من · ساخت گفتگوی جدید |
| GET / DELETE | `/chats/{id}` | کاربر | پیام‌های یک گفتگو · حذف آن |
| POST | `/chats/{id}/messages` | کاربر | `{"message": "...", "temperature": 0.2}`: پرسش و پاسخ هر دو ذخیره می‌شوند |
| POST | `/admin/upload` | مدیر | multipart با فیلد `files` (یک یا چند فایل) |
| GET | `/admin/documents` | مدیر | فهرست اسناد |
| DELETE | `/admin/documents/{doc_id}` | مدیر | حذف یک سند |

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/login -H "Content-Type: application/json" \
        -d '{"username": "ali", "password": "123456"}' | python -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -X POST localhost:8000/chats -H "Authorization: Bearer $TOKEN"
curl -X POST localhost:8000/chats/1/messages -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"message": "مرخصی استحقاقی سالانه چند روز است؟"}'
```
پاسخ: `{"role": "assistant", "content": "...", "source": "documents | model | web | none", "sources": [...], "temperature": 0.2, "title": "..."}`

## نکات
- PDF اسکن‌شده (تصویری) متن ندارد و با پیام خطا رد می‌شود. برای این نوع فایل‌ها باید OCR اضافه شود.
- اگر پاسخ‌ها از اسناد نامرتبط ساخته می‌شوند، `RELEVANCE_THRESHOLD` را بیشتر کنید (مثلاً ۰٫۵۵). اگر اسناد مرتبط پیدا نمی‌شوند، آن را کمتر کنید.
- با تغییر `EMBEDDING_MODEL`، پوشه‌ی `data/chroma` را پاک کنید و فایل‌ها را دوباره آپلود کنید.
