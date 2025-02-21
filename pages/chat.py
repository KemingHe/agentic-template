from components.page_ui import setup_page
from components.chat_ui import setup_simple_chat

# from llms.groq import groq_llama_lite_model
# from llms.groq import groq_llama_regular_model
from llms.openai import openai_lite_model
from llms.openai import openai_regular_model

# from llms.openai import openai_premium_model
# from llms.anthropic import anthropic_regular_model
# from chains.web_search import get_single_web_search_chain_response_stream
from chains.web_search import get_multi_web_search_chain_response_stream
# from agents.web_search_agent import get_web_search_agent_response_stream

setup_page(
    page_title="(Multi-Task) Web Search Chatbot",
    page_description="This chatbot can perform multi-topic web search and provide relevant information to the user.",
)

setup_simple_chat(
    orchestrator_llm=openai_regular_model,
    summarizer_llm=openai_lite_model,
    get_response_stream=get_multi_web_search_chain_response_stream,
)
