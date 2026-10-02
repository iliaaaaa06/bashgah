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


def api(method: str, path: str, timeout: float = REQUEST_TIMEOUT, **kwargs):
    """Calls the backend with the logged-in user's token. An expired token sends the user back to the login form."""
    token = st.session_state.get("token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
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
        if r.status_code == 401 and token:
            st.session_state.clear()
            st.rerun()
        detail = data.get("detail") if isinstance(data, dict) else None
        errors = data.get("errors") if isinstance(data, dict) else None
        raise ApiError(detail if isinstance(detail, str) else "خطای ناشناخته از سرور دریافت شد.", r.status_code, errors)
    return data


def error_text(exc: ApiError) -> str:
    return exc.message + "".join(f"\n- {e.get('message')}" for e in exc.errors)


@st.cache_data(ttl=15, show_spinner=False)
def health() -> dict | None:
    try:
        return api("GET", "/health", timeout=5)
    except ApiError:
        return None


# ================================================================== login / sign-up
def login_page() -> None:
    st.title("دستیار اداری هوشمند")
    login_tab, register_tab = st.tabs(["ورود", "ثبت‌نام"])
    for tab, path, button in ((login_tab, "/auth/login", "ورود"), (register_tab, "/auth/register", "ساخت حساب")):
        with tab, st.form(path):
            username = st.text_input("نام کاربری", help="دست‌کم ۳ حرف")
            password = st.text_input("رمز عبور", type="password", help="دست‌کم ۶ حرف")
            if st.form_submit_button(button, type="primary"):
                try:
                    r = api("POST", path, json={"username": username, "password": password})
                except ApiError as exc:
                    st.error(error_text(exc))
                    continue
                st.session_state.token, st.session_state.user = r["token"], r["user"]
                st.rerun()


def logout() -> None:
    try:
        api("POST", "/auth/logout")
    except ApiError:
        pass
    st.session_state.clear()
    st.rerun()


# ================================================================== sidebar
def sidebar() -> None:
    user = st.session_state.user
    with st.sidebar:
        role = " :violet-badge[مدیر]" if user["is_admin"] else ""
        st.markdown(f"**{user['username']}**{role}")
        if st.button("خروج از حساب", icon=":material/logout:", width="stretch"):
            logout()
        h = health()
        dot, text = ("err", "سرور در دسترس نیست") if h is None else \
            ("warn", "مدل زبانی در دسترس نیست") if not h.get("llm_available") else ("ok", "آماده")
        st.html(f'<div dir="rtl"><span class="dot {dot}"></span>{text}</div>')
        if h is not None:
            st.caption(f"{fa(h.get('indexed_chunks', 0))} قطعه در پایگاه دانش")
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


def open_chat(chat_id: int | None) -> None:
    """Loads a saved chat (or starts an empty one when chat_id is None)."""
    st.session_state.chat = {"id": None, "title": "گفتگوی جدید", "messages": []}
    if chat_id is not None:
        try:
            st.session_state.chat = api("GET", f"/chats/{chat_id}")
        except ApiError as exc:
            st.error(exc.message)


def chats_sidebar() -> None:
    chat = st.session_state.chat
    with st.sidebar:
        if st.button("گفتگوی جدید", icon=":material/add_comment:", width="stretch"):
            open_chat(None)  # the chat is saved once its first question is sent
            st.rerun()
        try:
            chats = api("GET", "/chats")
        except ApiError as exc:
            st.error(exc.message)
            chats = []
        if chats:
            st.caption("گفتگوهای من")
        for c in chats:
            current = c["id"] == chat["id"]
            if st.button(c["title"], key=f"chat_{c['id']}", width="stretch", type="primary" if current else "secondary"):
                open_chat(c["id"])
                st.rerun()
        if chat["id"] is not None and st.button("حذف این گفتگو", icon=":material/delete:", width="stretch"):
            try:
                api("DELETE", f"/chats/{chat['id']}")
            except ApiError as exc:
                st.error(exc.message)
            open_chat(None)
            st.rerun()
        st.divider()
        # Lives in st.session_state => independent for every user / browser tab
        st.slider("دما (Temperature)", 0.0, MAX_TEMPERATURE, DEFAULT_TEMPERATURE, step=0.05, key="temperature",
                  help="دمای کمتر = پاسخ دقیق‌تر و ثابت‌تر · دمای بیشتر = پاسخ متنوع‌تر. این تنظیم فقط روی گفتگوی شما اثر دارد.")


def ask(question: str) -> dict:
    chat = st.session_state.chat
    try:
        if chat["id"] is None:
            chat["id"] = api("POST", "/chats")["id"]
        r = api("POST", f"/chats/{chat['id']}/messages", json={"message": question, "temperature": st.session_state.temperature})
    except ApiError as exc:
        return {"role": "assistant", "content": error_text(exc), "error": True}
    chat["title"] = r.pop("title")
    return r


def chat_page() -> None:
    if "chat" not in st.session_state:
        open_chat(None)
    chats_sidebar()
    chat = st.session_state.chat
    st.title(chat["title"] if chat["id"] is not None else "گفتگو با دستیار")
    if not chat["messages"]:
        st.info("پرسش خود را درباره‌ی آیین‌نامه‌ها، بخشنامه‌ها و امور اداری بنویسید. "
                "پاسخ ابتدا از اسناد ثبت‌شده، سپس از دانش مدل و در نهایت از جستجوی وب تهیه می‌شود.")
        for col, suggestion in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
            if col.button(suggestion, width="stretch"):
                st.session_state.pending_prompt = suggestion
                st.rerun()

    for msg in chat["messages"]:
        show_message(msg)

    prompt = st.chat_input("پیام خود را بنویسید…") or st.session_state.pop("pending_prompt", None)
    if prompt:
        is_new = chat["id"] is None
        chat["messages"].append({"role": "user", "content": prompt})
        show_message(chat["messages"][-1])
        with st.spinner("در حال جستجو و تهیه‌ی پاسخ…"):
            chat["messages"].append(ask(prompt))
        show_message(chat["messages"][-1])
        if is_new and chat["id"] is not None:
            st.rerun()  # show the new chat in the sidebar list
    st.caption("پاسخ‌ها را پیش از استفاده‌ی رسمی بررسی کنید.")


# ================================================================== admin page
def become_admin() -> None:
    st.info("برای مدیریت اسناد، رمز مدیر را وارد کنید. پس از آن حساب شما برای همیشه مدیر می‌ماند.")
    with st.form("become_admin"):
        password = st.text_input("رمز مدیر", type="password")
        if st.form_submit_button("تأیید", type="primary"):
            try:
                st.session_state.user = api("POST", "/auth/become-admin", json={"password": password})
            except ApiError as exc:
                st.error(error_text(exc))
                return
            st.rerun()


def admin_page() -> None:
    st.title("مدیریت اسناد")
    if not st.session_state.user["is_admin"]:
        become_admin()
        return

    # ---------- upload
    st.subheader("بارگذاری اسناد")
    uploader_id = st.session_state.setdefault("uploader_id", 0)  # a new key clears the picked files
    files = st.file_uploader("فایل‌های PDF، DOCX یا TXT را انتخاب کنید", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True, key=f"uploader_{uploader_id}")
    if st.button("بارگذاری و پردازش", type="primary", disabled=not files, icon=":material/upload:"):
        payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
        with st.spinner(f"در حال پردازش {fa(len(files))} فایل… (ممکن است برای فایل‌های بزرگ چند دقیقه طول بکشد)"):
            try:
                st.session_state.upload_results = api("POST", "/admin/upload", files=payload)["results"]
                st.session_state.uploader_id += 1
                health.clear()
                st.rerun()
            except ApiError as exc:
                st.error(exc.message)
    show = {"ingested": st.success, "duplicate": st.warning, "error": st.error}
    for r in st.session_state.get("upload_results", []):
        show.get(r["status"], st.info)(r["message"] + (f" ({fa(r['chunks'])} قطعه)" if r.get("chunks") else ""))

    # ---------- document list
    st.divider()
    head, refresh = st.columns([4, 1], vertical_alignment="bottom")
    head.subheader("اسناد ثبت‌شده")
    if refresh.button("بروزرسانی", icon=":material/refresh:", width="stretch"):
        st.rerun()
    try:
        docs = api("GET", "/admin/documents")
    except ApiError as exc:
        st.error(exc.message)
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
            try:
                api("DELETE", f"/admin/documents/{pending['doc_id']}")
            except ApiError as exc:
                st.error(exc.message)
            st.session_state.pop("confirm_delete", None)
            health.clear()
            st.rerun()
        if no.button("انصراف"):
            st.session_state.pop("confirm_delete", None)
            st.rerun()


# ================================================================== app
st.set_page_config(page_title="دستیار اداری هوشمند", page_icon=":material/account_balance:", layout="centered")
st.html(f"<style>{CSS}</style>")
if "token" not in st.session_state:
    login_page()
else:
    page = st.navigation([
        st.Page(chat_page, title="گفتگو", icon=":material/chat:", url_path="chat", default=True),
        st.Page(admin_page, title="مدیریت اسناد", icon=":material/folder_managed:", url_path="admin"),
    ])
    sidebar()
    page.run()
