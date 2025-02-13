import streamlit as st

def setup_page(
        app_icon: str = "📝",
        app_name: str = "Agentic Template",
        page_title: str = "Simple Chatbot",
        page_description: str = "This is a simple chatbot using LangChain and OpenAI.",
) -> None:
    st.set_page_config(
        page_title=app_name,
        page_icon=app_icon,
    )
    st.title(page_title)
    st.info(page_description)
