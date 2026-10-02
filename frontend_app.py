"""Persian RAG administrative assistant — Streamlit UI. Talks to rag_server.py over HTTP only.

Settings come from the .env file next to this file.   Run:  python frontend_app.py
"""
import html
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from streamlit import runtime

load_dotenv(Path(__file__).resolve().parent / ".env")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("FRONTEND_REQUEST_TIMEOUT", "300"))
DEFAULT_TEMPERATURE = float(os.getenv("FRONTEND_DEFAULT_TEMPERATURE", "0.2"))
MAX_TEMPERATURE = float(os.getenv("FRONTEND_MAX_TEMPERATURE", "1.5"))

if __name__ == "__main__" and not runtime.exists():
    # `python frontend_app.py` re-runs this file under Streamlit with the address/port from .env
    args = [
        sys.executable, "-m", "streamlit", "run", __file__,
        "--server.port", os.getenv("FRONTEND_PORT", "8501"),
        "--server.address", os.getenv("FRONTEND_HOST", "0.0.0.0"),
        "--server.headless", "true", "--browser.gatherUsageStats", "false", "--theme.primaryColor", "#1f5eb8",
    ]
    # On machines with several network interfaces Streamlit may print the wrong "Network URL"
    if os.getenv("FRONTEND_PUBLIC_HOST"):
        args += ["--browser.serverAddress", os.environ["FRONTEND_PUBLIC_HOST"]]
    os.execv(sys.executable, args)

SOURCE_BADGES = {
    "documents": ":green-badge[پاسخ از اسناد]",
    "model": ":violet-badge[پاسخ از دانش مدل]",
    "web": ":blue-badge[پاسخ از جستجوی وب]",
    "none": ":orange-badge[پاسخی یافت نشد]",
}
SUGGESTIONS = ["مرخصی استحقاقی سالانه چند روز است؟", "روند ثبت درخواست مأموریت چیست؟", "نرخ امروز دلار چقدر است؟"]

CSS = """
@import url("https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&display=swap");
/* Font on containers only, so Streamlit's icon font keeps working */
.stApp, .stMarkdown, .stChatMessage, .stAlert, button, input, textarea, label, h1, h2, h3, h4,
[data-testid="stCaptionContainer"], [data-testid="stSidebarNav"] { font-family: "Vazirmatn", Tahoma, sans-serif !important; }
.stMarkdown, .stChatMessage, .stAlert, label, h1, h2, h3, h4, textarea, input[type="text"],
[data-testid="stCaptionContainer"], [data-testid="stExpander"], [data-testid="stSidebar"],
[data-testid="stFileUploader"], [data-testid="stForm"] { direction: rtl; text-align: right; }
input[type="password"] { direction: ltr; text-align: left; }
.stChatMessage { line-height: 1.9; }
.stChatMessage ul, .stChatMessage ol { padding-right: 1.4rem; padding-left: 0; }
.source-item { background: rgba(127,127,127,.08); border-radius: 8px; padding: .45rem .7rem; margin-bottom: .45rem; font-size: .9rem; }
.source-item .snippet { opacity: .75; font-size: .82rem; margin-top: .15rem; }
.source-item a { direction: ltr; unicode-bidi: embed; word-break: break-all; }
.dot { display: inline-block; width: .6rem; height: .6rem; border-radius: 50%; margin-left: .4rem; }
.dot.ok { background: #1a8a4a; } .dot.warn { background: #d69e2e; } .dot.err { background: #c53030; }
"""


def fa(value) -> str:
    return str(value).translate(str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫"))


def esc(text) -> str:
    return html.escape(str(text or ""))


# ================================================================== backend calls
class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None, errors: list | None = None):
        super().__init__(message)
        self.message, self.status, self.errors = message, status, errors or []


def api(method: str, path: str, admin_key: str | None = None, timeout: float = REQUEST_TIMEOUT, **kwargs):
    headers = {"X-Admin-Key": admin_key} if admin_key else {}
    try:
        r = httpx.request(method, BACKEND_URL + path, headers=headers, timeout=timeout, **kwargs)
    except httpx.TimeoutException as exc:
        raise ApiError("زمان پاسخ‌گویی سرور به پایان رسید. لطفاً دوباره تلاش کنید.") from exc
    except httpx.HTTPError as exc:
        raise ApiError("ارتباط با سرور برقرار نشد. اجرای سرویس بک‌اند و آدرس BACKEND_URL را بررسی کنید.") from exc
    try:
        data = r.json()
    except ValueError:
        data = None
    if r.is_error:
        detail = data.get("detail") if isinstance(data, dict) else None
        errors = data.get("errors") if isinstance(data, dict) else None
        raise ApiError(detail if isinstance(detail, str) else "خطای ناشناخته از سرور دریافت شد.", r.status_code, errors)
    return data


@st.cache_data(ttl=15, show_spinner=False)
def health() -> dict | None:
    try:
        return api("GET", "/health", timeout=5)
    except ApiError:
        return None


# ================================================================== sidebar
def sidebar() -> None:
    with st.sidebar:
        st.subheader("وضعیت سرویس")
        h = health()
        dot, text = ("err", "سرور در دسترس نیست") if h is None else \
            ("warn", "مدل زبانی در دسترس نیست") if not h.get("llm_available") else ("ok", "آماده")
        st.html(f'<div dir="rtl"><span class="dot {dot}"></span>{text}</div>')
        if h is not None:
            st.caption(f"{fa(h.get('indexed_chunks', 0))} قطعه در پایگاه دانش")
        if st.button("بررسی مجدد", icon=":material/refresh:", width="stretch"):
            health.clear()
            st.rerun()
        st.divider()


# ================================================================== chat page
def show_sources(msg: dict) -> None:
    sources = msg.get("sources") or []
    if not sources:
        return
    if msg.get("source") == "web":
        # Web answers: the answer first, then only the reference links
        links = [
            f'<div class="source-item">{fa(i)}. <a href="{esc(s.get("url"))}" target="_blank" rel="noopener noreferrer">'
            f'{esc(s.get("title") or s.get("url"))}</a></div>'
            for i, s in enumerate(sources, start=1)
        ]
        st.html('<div dir="rtl"><b>منابع جستجوی وب:</b>' + "".join(links) + "</div>")
        return
    with st.expander(f"منابع ({fa(len(sources))})"):
        items = []
        for i, s in enumerate(sources, start=1):
            parts = [f"<b>{fa(i)}. {esc(s.get('title') or 'بدون عنوان')}</b>"]
            if s.get("page"):
                parts.append(f"صفحه {fa(s['page'])}")
            if s.get("score") is not None:
                parts.append(f"شباهت {fa(round(s['score'] * 100))}٪")
            snippet = f'<div class="snippet">{esc(s["snippet"])}</div>' if s.get("snippet") else ""
            items.append(f'<div class="source-item" dir="rtl">{" · ".join(parts)}{snippet}</div>')
        st.html("".join(items))


def show_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("error"):
            st.error(msg["content"])
            return
        st.markdown(msg["content"].replace("$", r"\$"))  # "$" in prices must not become LaTeX
        if msg["role"] == "assistant":
            st.caption(f"{SOURCE_BADGES.get(msg.get('source'), '')}  دما {fa(msg.get('temperature', ''))}")
            show_sources(msg)


def ask(question: str) -> dict:
    temperature = st.session_state.temperature
    try:
        r = api("POST", "/chat", json={"message": question, "temperature": temperature})
    except ApiError as exc:
        details = "".join(f"\n- {e.get('message')}" for e in exc.errors)
        return {"role": "assistant", "content": exc.message + details, "error": True}
    return {"role": "assistant", "content": r["answer"], "source": r["source"], "sources": r.get("sources", []),
            "temperature": r.get("temperature", temperature)}


def chat_page() -> None:
    with st.sidebar:
        st.subheader("تنظیمات گفتگو")
        # Lives in st.session_state => independent for every user / browser tab
        st.slider("دما (Temperature)", 0.0, MAX_TEMPERATURE, DEFAULT_TEMPERATURE, step=0.05, key="temperature",
                  help="دمای کمتر = پاسخ دقیق‌تر و ثابت‌تر · دمای بیشتر = پاسخ متنوع‌تر. این تنظیم فقط روی گفتگوی شما اثر دارد.")
        if st.button("پاک کردن گفتگو", icon=":material/delete_sweep:", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    messages = st.session_state.setdefault("messages", [])
    st.title("گفتگو با دستیار")
    if not messages:
        st.info("پرسش خود را درباره‌ی آیین‌نامه‌ها، بخشنامه‌ها و امور اداری بنویسید. "
                "پاسخ ابتدا از اسناد ثبت‌شده، سپس از دانش مدل و در نهایت از جستجوی وب تهیه می‌شود.")
        for col, suggestion in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
            if col.button(suggestion, width="stretch"):
                st.session_state.pending_prompt = suggestion
                st.rerun()

    for msg in messages:
        show_message(msg)

    prompt = st.chat_input("پیام خود را بنویسید…") or st.session_state.pop("pending_prompt", None)
    if prompt:
        messages.append({"role": "user", "content": prompt})
        show_message(messages[-1])
        with st.spinner("در حال جستجو و تهیه‌ی پاسخ…"):
            messages.append(ask(prompt))
        show_message(messages[-1])
    st.caption("پاسخ‌ها را پیش از استفاده‌ی رسمی بررسی کنید.")


# ================================================================== admin page
def logout() -> None:
    for k in ("admin_key", "upload_results", "confirm_delete"):
        st.session_state.pop(k, None)


def admin_call(*args, **kwargs):
    """Backend call with the admin key; a rejected key logs the admin out."""
    try:
        return api(*args, admin_key=st.session_state.admin_key, **kwargs)
    except ApiError as exc:
        if exc.status == 401:
            logout()
            st.rerun()
        st.error(exc.message)
        return None


def admin_page() -> None:
    st.title("مدیریت اسناد")
    if not st.session_state.get("admin_key"):
        st.subheader("ورود مدیر")
        with st.form("admin_login"):
            key = st.text_input("کلید مدیر", type="password",
                                help="برای بارگذاری و حذف اسناد، کلید مدیر (ADMIN_API_KEY) را وارد کنید.").strip()
            submitted = st.form_submit_button("ورود", type="primary")
        if submitted:
            if not key:
                st.error("کلید مدیر را وارد کنید.")
                return
            try:
                api("GET", "/admin/documents", admin_key=key)  # validates the key
            except ApiError as exc:
                st.error(exc.message)
                return
            st.session_state.admin_key = key
            st.rerun()
        return

    with st.sidebar:
        if st.button("خروج", icon=":material/logout:", width="stretch"):
            logout()
            st.rerun()

    # ---------- upload
    st.subheader("بارگذاری اسناد")
    uploader_id = st.session_state.setdefault("uploader_id", 0)  # a new key clears the picked files
    files = st.file_uploader("فایل‌های PDF، DOCX یا TXT را انتخاب کنید", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True, key=f"uploader_{uploader_id}")
    if st.button("بارگذاری و پردازش", type="primary", disabled=not files, icon=":material/upload:"):
        payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
        with st.spinner(f"در حال پردازش {fa(len(files))} فایل… (ممکن است برای فایل‌های بزرگ چند دقیقه طول بکشد)"):
            result = admin_call("POST", "/admin/upload", files=payload)
        if result:
            st.session_state.upload_results = result["results"]
            st.session_state.uploader_id += 1
            health.clear()
            st.rerun()
    show = {"ingested": st.success, "duplicate": st.warning, "error": st.error}
    for r in st.session_state.get("upload_results", []):
        show.get(r["status"], st.info)(r["message"] + (f" ({fa(r['chunks'])} قطعه)" if r.get("chunks") else ""))

    # ---------- document list
    st.divider()
    head, refresh = st.columns([4, 1], vertical_alignment="bottom")
    head.subheader("اسناد ثبت‌شده")
    if refresh.button("بروزرسانی", icon=":material/refresh:", width="stretch"):
        st.rerun()
    docs = admin_call("GET", "/admin/documents")
    if docs is None:
        return
    if not docs:
        st.caption("هنوز سندی بارگذاری نشده است.")
        return
    st.caption(f"{fa(len(docs))} سند")

    widths = [4, 1.3, 2, 1.2]
    for col, title in zip(st.columns(widths), ("نام فایل", "تعداد قطعه", "تاریخ بارگذاری", "")):
        col.markdown(f"**{title}**")
    for doc in docs:
        name, chunks, date, action = st.columns(widths, vertical_alignment="center")
        name.write(doc["filename"])
        chunks.write(fa(doc["chunks"]))
        date.write(fa(datetime.fromisoformat(doc["uploaded_at"]).astimezone().strftime("%Y/%m/%d %H:%M")))
        if action.button("حذف", key=f"del_{doc['doc_id']}", icon=":material/delete:"):
            st.session_state.confirm_delete = doc

    pending = st.session_state.get("confirm_delete")
    if pending:
        st.warning(f"سند «{pending['filename']}» از پایگاه دانش حذف شود؟")
        yes, no, _ = st.columns([1, 1, 3])
        if yes.button("بله، حذف شود", type="primary"):
            admin_call("DELETE", f"/admin/documents/{pending['doc_id']}")
            st.session_state.pop("confirm_delete", None)
            health.clear()
            st.rerun()
        if no.button("انصراف"):
            st.session_state.pop("confirm_delete", None)
            st.rerun()


# ================================================================== app
st.set_page_config(page_title="دستیار اداری هوشمند", page_icon=":material/account_balance:", layout="centered")
st.html(f"<style>{CSS}</style>")
page = st.navigation([
    st.Page(chat_page, title="گفتگو", icon=":material/chat:", url_path="chat", default=True),
    st.Page(admin_page, title="مدیریت اسناد", icon=":material/folder_managed:", url_path="admin"),
])
sidebar()
page.run()
