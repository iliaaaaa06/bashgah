"""Chat page — regular users. History and temperature live in this browser session only."""
import streamlit as st

import api_client
import texts as T
from config import DEFAULT_TEMPERATURE, MAX_TEMPERATURE
from ui.formatting import esc, fa, md_safe


def _sidebar_settings() -> None:
    with st.sidebar:
        st.subheader(T.CHAT_SETTINGS)
        # Stored in st.session_state => independent for every user/browser tab
        st.slider(
            T.CHAT_TEMPERATURE, min_value=0.0, max_value=MAX_TEMPERATURE, step=0.05,
            value=DEFAULT_TEMPERATURE, key="temperature", help=T.CHAT_TEMPERATURE_HELP,
        )
        if st.button(T.CHAT_CLEAR, icon=":material/delete_sweep:", width="stretch"):
            st.session_state.messages = []
            st.rerun()


def _render_sources(sources: list[dict], source: str) -> None:
    if not sources:
        return
    if source == "web":
        # Web answers: the answer comes first, then just the search results as links
        items = []
        for i, s in enumerate(sources, start=1):
            title = esc(s.get("title") or T.SOURCE_UNTITLED)
            url = esc(s.get("url") or "")
            link = f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>' if url else title
            items.append(f'<div class="source-item" dir="rtl">{fa(i)}. {link}</div>')
        st.html(f'<div dir="rtl"><b>{T.WEB_SOURCES_TITLE}</b></div>' + "".join(items))
        return
    with st.expander(T.SOURCES_TITLE.format(n=fa(len(sources)))):
        items = []
        for i, s in enumerate(sources, start=1):
            parts = [f"<b>{fa(i)}. {esc(s.get('title') or T.SOURCE_UNTITLED)}</b>"]
            if s.get("page"):
                parts.append(T.SOURCE_PAGE.format(p=fa(s["page"])))
            if s.get("score") is not None and not s.get("url"):
                parts.append(T.SOURCE_SCORE.format(s=fa(round(s["score"] * 100))))
            if s.get("url"):
                url = esc(s["url"])
                parts.append(f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>')
            snippet = f'<div class="snippet">{esc(s["snippet"])}</div>' if s.get("snippet") else ""
            items.append(f'<div class="source-item" dir="rtl">{" · ".join(parts)}{snippet}</div>')
        st.html("".join(items))


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("error"):
            st.error(msg["content"])
            return
        st.markdown(md_safe(msg["content"]))
        if msg["role"] == "assistant":
            badge = T.SOURCE_BADGES.get(msg.get("source", ""), "")
            st.caption(f"{badge}  {T.TEMPERATURE_CAPTION.format(t=fa(msg.get('temperature', '')))}")
            _render_sources(msg.get("sources", []), msg.get("source", ""))


def _ask(question: str) -> dict:
    temperature = st.session_state.temperature
    try:
        r = api_client.chat(question, temperature)
    except api_client.ApiError as exc:
        details = "\n".join(f"- {e.get('message')}" for e in exc.errors)
        return {"role": "assistant", "content": exc.message + (f"\n{details}" if details else ""), "error": True}
    return {
        "role": "assistant",
        "content": r["answer"],
        "source": r["source"],
        "sources": r.get("sources", []),
        "temperature": r.get("temperature", temperature),
    }


def render() -> None:
    _sidebar_settings()
    messages: list[dict] = st.session_state.setdefault("messages", [])

    st.title(T.CHAT_TITLE)
    if not messages:
        st.info(T.CHAT_INTRO)
        cols = st.columns(len(T.CHAT_SUGGESTIONS))
        for col, suggestion in zip(cols, T.CHAT_SUGGESTIONS):
            if col.button(suggestion, width="stretch"):
                st.session_state.pending_prompt = suggestion
                st.rerun()

    for msg in messages:
        _render_message(msg)

    prompt = st.chat_input(T.CHAT_PLACEHOLDER) or st.session_state.pop("pending_prompt", None)
    if prompt:
        user_msg = {"role": "user", "content": prompt}
        messages.append(user_msg)
        _render_message(user_msg)
        with st.spinner(T.CHAT_THINKING):
            answer = _ask(prompt)
        messages.append(answer)
        _render_message(answer)

    st.caption(T.CHAT_DISCLAIMER)
