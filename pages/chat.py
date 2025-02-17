from setup_app import setup_page
from setup_chat import setup_simple_chat
from chat_single_task import get_web_search_response_stream

setup_page(
    page_title="Web Search Chatbot",
    page_description="This chatbot can perform a single-step web search and provide relevant information to the user.",
)

setup_simple_chat(get_web_search_response_stream)
