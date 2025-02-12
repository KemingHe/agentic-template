import streamlit as st


def setup_page():
    st.set_page_config(
        page_icon="📝",
        page_title="Agentic Template",
    )
    st.title("Hello World!")
    st.info("Welcome to the Agentic Template.")


def print_hello():
    st.write("Hello, world!")
