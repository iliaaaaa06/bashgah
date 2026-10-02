"""Admin page — login with ADMIN_API_KEY, upload, list and delete documents."""
import streamlit as st

import api_client
import texts as T
from ui import sidebar
from ui.formatting import fa, fa_datetime

_STATUS_VIEW = {"ingested": st.success, "duplicate": st.warning, "error": st.error}


def _login() -> None:
    st.subheader(T.ADMIN_LOGIN_TITLE)
    with st.form("admin_login"):
        key = st.text_input(T.ADMIN_KEY_LABEL, type="password", help=T.ADMIN_LOGIN_HELP)
        submitted = st.form_submit_button(T.ADMIN_LOGIN, type="primary")
    if submitted:
        if not key.strip():
            st.error(T.ADMIN_KEY_EMPTY)
            return
        try:
            api_client.list_documents(key.strip())  # validates the key
        except api_client.ApiError as exc:
            st.error(exc.message)
            return
        st.session_state.admin_key = key.strip()
        st.rerun()


def _logout() -> None:
    for k in ("admin_key", "upload_results", "confirm_delete"):
        st.session_state.pop(k, None)


def _handle_auth_error(exc: api_client.ApiError) -> None:
    if exc.status == 401:
        _logout()
        st.rerun()
    st.error(exc.message)


def _upload_section(key: str) -> None:
    st.subheader(T.UPLOAD_TITLE)
    # Changing the widget key after an upload clears the selected files
    uploader_id = st.session_state.setdefault("uploader_id", 0)
    files = st.file_uploader(
        T.UPLOAD_LABEL, type=["pdf", "docx", "txt"], accept_multiple_files=True, key=f"uploader_{uploader_id}"
    )
    if st.button(T.UPLOAD_BUTTON, type="primary", disabled=not files, icon=":material/upload:"):
        payload = [(f.name, f.getvalue(), f.type or "application/octet-stream") for f in files]
        with st.spinner(T.UPLOAD_PROGRESS.format(n=fa(len(payload)))):
            try:
                result = api_client.upload_documents(key, payload)
            except api_client.ApiError as exc:
                _handle_auth_error(exc)
                return
        st.session_state.upload_results = result["results"]
        st.session_state.uploader_id += 1
        sidebar.invalidate_health()
        st.rerun()

    for r in st.session_state.get("upload_results", []):
        suffix = T.UPLOAD_CHUNKS.format(n=fa(r["chunks"])) if r.get("chunks") else ""
        _STATUS_VIEW.get(r["status"], st.info)(r["message"] + suffix)


def _documents_section(key: str) -> None:
    head, refresh = st.columns([4, 1], vertical_alignment="bottom")
    head.subheader(T.DOCS_TITLE)
    if refresh.button(T.DOCS_REFRESH, icon=":material/refresh:", width="stretch"):
        st.rerun()

    try:
        docs = api_client.list_documents(key)
    except api_client.ApiError as exc:
        _handle_auth_error(exc)
        return

    if not docs:
        st.caption(T.DOCS_EMPTY)
        return
    st.caption(T.DOCS_COUNT.format(n=fa(len(docs))))

    widths = [4, 1.3, 2, 1.2]
    header = st.columns(widths)
    for col, title in zip(header, (T.DOCS_COL_NAME, T.DOCS_COL_CHUNKS, T.DOCS_COL_DATE, "")):
        col.markdown(f"**{title}**")

    for doc in docs:
        name, chunks, date, action = st.columns(widths, vertical_alignment="center")
        name.write(doc["filename"])
        chunks.write(fa(doc["chunks"]))
        date.write(fa_datetime(doc["uploaded_at"]))
        if action.button(T.DOCS_DELETE, key=f"del_{doc['doc_id']}", icon=":material/delete:"):
            st.session_state.confirm_delete = doc

    pending = st.session_state.get("confirm_delete")
    if pending:
        st.warning(T.DOCS_DELETE_CONFIRM.format(name=pending["filename"]))
        yes, no, _ = st.columns([1, 1, 3])
        if yes.button(T.DOCS_DELETE_YES, type="primary"):
            try:
                api_client.delete_document(key, pending["doc_id"])
            except api_client.ApiError as exc:
                _handle_auth_error(exc)
            st.session_state.pop("confirm_delete", None)
            sidebar.invalidate_health()
            st.rerun()
        if no.button(T.DOCS_DELETE_NO):
            st.session_state.pop("confirm_delete", None)
            st.rerun()


def render() -> None:
    st.title(T.ADMIN_TITLE)
    key = st.session_state.get("admin_key")
    if not key:
        _login()
        return

    with st.sidebar:
        if st.button(T.ADMIN_LOGOUT, icon=":material/logout:", width="stretch"):
            _logout()
            st.rerun()

    _upload_section(key)
    st.divider()
    _documents_section(key)
