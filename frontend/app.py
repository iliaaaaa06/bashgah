"""Streamlit frontend entrypoint.  Run:  streamlit run app.py  (from this folder)"""
import streamlit as st

import texts as T

st.set_page_config(page_title=T.APP_TITLE, page_icon=":material/account_balance:", layout="centered")

from ui import admin_page, chat_page, sidebar, style  # noqa: E402  (after set_page_config)

style.inject()
navigation = st.navigation([
    st.Page(chat_page.render, title=T.NAV_CHAT, icon=":material/chat:", url_path="chat", default=True),
    st.Page(admin_page.render, title=T.NAV_ADMIN, icon=":material/folder_managed:", url_path="admin"),
])
sidebar.render()
navigation.run()
