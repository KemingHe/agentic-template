from setup_app import setup_page
from setup_chat import setup_chat
from setup_langchain import get_simple_response_stream

# Simple memory-rich chatbot with web search and OpenAI 4o-mini model.
#
# 1. New user query and existing history passed into the runnable;
# 2. Orchestrator node decides (outputs a json) which downstream nodes to call;
# 3. Downstream nodes are called in parallel and return their results;
# 4. Summary node combines the results and returns the final response.

from pydantic import BaseModel, Field
from typing import Iterator

from setup_duckduckgo import simple_search
from setup_openai import openai_regular_model
from setup_weatherapi import get_weather_data

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser



class SearchTask(BaseModel):
    should_search_web: bool = Field(description="Whether it is necessary to search the web for the query, excluding weather queries.")
    should_search_weather: bool = Field(description="Whether it is necessary to search the weather for the query.")
    web_query: str = Field(description="The reformatted user query to search the web for, optimized for web search, excluding weather queries.")
    web_query_count: int = Field(description="The number of search results to return.")
    weather_query: str = Field(description="The location, preferrably the city (or zipcode if user query is too generalized) to search the weather for.")

orchestrator_parser = PydanticOutputParser(pydantic_object=SearchTask)

orchastrator_template: str = """
You are a helpful assistant that resolves the user's query step by step:

1. Reformats the user query and message history into a context-rich query, always prefer hard facts over general directions
2. Decides whether to search the web or weather for the query, and if so, which tools to use and how many results to return

You are provided with the following information:

- User query: {user_query}
- Chat history: {chat_history}

{format_instructions}

YOUR RESPONSE MUST BE A VALID JSON OBJECT.
"""

orchastrator_prompt = ChatPromptTemplate.from_messages([
    ("system", orchastrator_template)
]).partial(format_instructions=orchestrator_parser.get_format_instructions())

orchestrator_chain = (
    orchastrator_prompt 
    | openai_regular_model 
    | orchestrator_parser
)

summarizer_template: str = """
You are a helpful assistant that can:

- If present, use the search results to provide the user with the most relevant information
- Create reference links to the search results in your response
- And have contextful conversations with users

You are provided with the following information:

- User query: {user_query}
- Chat history: {chat_history}
- Web search results: {web_search_results}
- Weather search results: {weather_search_results}
"""

summarizer_prompt = ChatPromptTemplate.from_template(summarizer_template)

def get_response_stream(
    user_query: str,
    chat_history: str,
) -> Iterator[str]:
    search_task = orchestrator_chain.invoke({
        "user_query": user_query,
        "chat_history": chat_history,
    })

    print(search_task)

    web_search_results: str | None = None
    if search_task.should_search_web:
        web_search_results = simple_search(
            query=search_task.web_query,
            count=search_task.web_query_count,
        )
    
    print(web_search_results)

    weather_search_results: str | None = None
    if search_task.should_search_weather:
        weather_search_results = get_weather_data(search_task.weather_query)

    summarizer_chain = summarizer_prompt | openai_regular_model | StrOutputParser()
    return summarizer_chain.stream({
        "user_query": user_query,
        "chat_history": chat_history,
        "web_search_results": web_search_results,
        "weather_search_results": weather_search_results,
    })


setup_page(
    page_title="Simple Chatbot",
    page_description="This is a simple chatbot with memory using LangChain and OpenAI.",
)

setup_chat(get_response_stream)
