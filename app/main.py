import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from components.chat_ui import init_chat_history, is_valid_query
from chains.web_search_chain import (
    get_simple_response_stream,
    get_web_search_response_stream,
)
from components.page_ui import setup_page

setup_page(
    page_title="Chat Comparison",
    page_description="This page demonstrates the difference between a basic chatbot and a web search chatbot.",
)

init_chat_history("simple_chat_history")
simple_chat_history = st.session_state.simple_chat_history

init_chat_history("web_search_chat_history")
web_search_chat_history = st.session_state.web_search_chat_history

demo_container = st.container()

with demo_container:
    user_query = st.chat_input("I have a question about...")
    if is_valid_query(user_query):
        # Add user message to both histories
        simple_chat_history.append(HumanMessage(user_query))
        web_search_chat_history.append(HumanMessage(user_query))

        # Create columns for displaying streaming responses
        response_col1, response_col2 = st.columns(2)

        with response_col1:
            st.header("Basic Chat")
            with st.chat_message("human"):
                st.markdown(user_query)
            with st.chat_message("ai"):
                simple_response = st.write_stream(
                    get_simple_response_stream(user_query, simple_chat_history)
                )
                simple_chat_history.append(AIMessage(simple_response))

        with response_col2:
            st.header("Web Search Chat")
            with st.chat_message("human"):
                st.markdown(user_query)
            with st.chat_message("ai"):
                web_response = st.write_stream(
                    get_web_search_response_stream(user_query, web_search_chat_history)
                )
                web_search_chat_history.append(AIMessage(web_response))
