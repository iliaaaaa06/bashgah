from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


def inject() -> None:
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
