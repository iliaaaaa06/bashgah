"""Sidebar content shared by every page: service status."""
import streamlit as st

import api_client
import texts as T
from config import HEALTH_CACHE_SECONDS
from ui.formatting import fa


@st.cache_data(ttl=HEALTH_CACHE_SECONDS, show_spinner=False)
def _health() -> dict | None:
    try:
        return api_client.health()
    except api_client.ApiError:
        return None


def render() -> None:
    with st.sidebar:
        st.subheader(T.STATUS_TITLE)
        h = _health()
        if h is None:
            dot, text = "err", T.STATUS_BACKEND_DOWN
        elif not h.get("llm_available"):
            dot, text = "warn", T.STATUS_LLM_DOWN
        else:
            dot, text = "ok", T.STATUS_READY
        st.html(f'<div dir="rtl"><span class="status-dot {dot}"></span>{text}</div>')
        if h is not None:
            st.caption(T.STATUS_CHUNKS.format(n=fa(h.get("indexed_chunks", 0))))
        if st.button(T.STATUS_REFRESH, icon=":material/refresh:", width="stretch"):
            _health.clear()
            st.rerun()
        st.divider()


def invalidate_health() -> None:
    _health.clear()
