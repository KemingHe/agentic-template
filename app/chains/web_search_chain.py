from pydantic import BaseModel, Field
from typing import Iterator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from tools.web_search import ddg_text_search
from llms.openai import openai_regular_model
from tools.weather import get_weather_data

class SearchTask(BaseModel):
    should_search_web: bool = Field(
        description="Set to true if factual information needs to be searched online"
    )
    should_search_weather: bool = Field(
        description="Set to true if weather information for a specific location is requested"
    )
    web_query: str = Field(
        description="Search query for web results. Must be precise and include key terms. Empty if no search needed"
    )
    web_query_count: int = Field(
        description="Number of web results to fetch (1-5 recommended). Must be 0 if no search needed"
    )
    weather_query: str = Field(
        description="Location name for weather search. Must be specific and unambiguous. Empty if no weather needed"
    )


orchestrator_parser = PydanticOutputParser(pydantic_object=SearchTask)

orchestrator_template: str = """
Analyze the query and determine search requirements. Follow these rules and examples carefully:

Examples:
1. Query: "What's the weather like in New York?"
    Response: {{"should_search_web": false, "should_search_weather": true, "web_query": "", "web_query_count": 0, "weather_query": "New York, NY"}}

2. Query: "Tell me about quantum computing advances"
    Response: {{"should_search_web": true, "should_search_weather": false, "web_query": "latest quantum computing breakthroughs research", "web_query_count": 3, "weather_query": ""}}

3. Query: "How are you doing today?"
    Response: {{"should_search_web": false, "should_search_weather": false, "web_query": "", "web_query_count": 0, "weather_query": ""}}

4. Query: "What's the weather and tech news in San Francisco?"
    Response: {{"should_search_web": true, "should_search_weather": true, "web_query": "latest technology news San Francisco", "web_query_count": 3, "weather_query": "San Francisco, CA"}}

Rules:
1. For weather queries: Enable weather search and provide specific location with state/country when possible
2. For factual queries: Enable web search with precise keywords and 1-5 results, adjustable to user needs
3. For conversation: Disable all searches and return empty strings
4. For multiple topics: Enable relevant searches and create focused queries for each

Input:
- User query: {user_query}
- Chat history: {chat_history}

{format_instructions}

Return strictly valid JSON matching the SearchTask schema.
"""

# Create the system message with format instructions
format_instructions = orchestrator_parser.get_format_instructions()
messages = [("system", orchestrator_template)]

# Create the chat prompt template and add format instructions
orchestrator_prompt = ChatPromptTemplate.from_messages(messages).partial(
    format_instructions=format_instructions
)

orchestrator_chain = orchestrator_prompt | openai_regular_model | orchestrator_parser

# ------------------------------------------------------------------------------
summarizer_template: str = """
You are a knowledgeable and contextually aware assistant. Your role is to provide helpful, accurate, and well-structured responses.

Guidelines for Response:
1. SEARCH RESULTS HANDLING:
    - If web search results exist: Synthesize and cite them with [link](https://www.example.com) inline
    - If weather data exists: Present it clearly and naturally
    - When no search data: Rely on general knowledge and context

2. CONVERSATION CONTEXT:
    - Reference chat history when relevant
    - Maintain consistent context across interactions
    - Acknowledge previous points in ongoing discussions

3. RESPONSE STRUCTURE:
    - Start with direct answers to queries
    - Follow with supporting details or context
    - End with relevant follow-up points if appropriate

4. STYLE:
    - Be concise but thorough, use bullet points if needed
    - Use natural, conversational language
    - Stay factual and objective

Available Information:
- User query: {user_query}
- Chat history: {chat_history}
- Web search results: {web_search_results}
- Weather search results: {weather_search_results}

Respond in a way that naturally integrates all relevant information while maintaining a helpful and engaging conversation.
"""

summarizer_prompt = ChatPromptTemplate.from_template(summarizer_template)
summarizer_chain = summarizer_prompt | openai_regular_model | StrOutputParser()

simple_summarizer_prompt = ChatPromptTemplate.from_template(
    summarizer_template
).partial(web_search_results="", weather_search_results="")
simple_summarizer_chain = (
    simple_summarizer_prompt | openai_regular_model | StrOutputParser()
)

# ------------------------------------------------------------------------------
def get_web_search_response_stream(
    user_query: str,
    chat_history: str,
) -> Iterator[str]:
    search_task: SearchTask = orchestrator_chain.invoke(
        {
            "user_query": user_query,
            "chat_history": chat_history,
        }
    )

    print(search_task)

    web_search_results: str | None = None
    if search_task.should_search_web:
        web_search_results = ddg_text_search(
            query=search_task.web_query,
            count=search_task.web_query_count,
        )

    print(web_search_results)

    weather_search_results: str | None = None
    if search_task.should_search_weather:
        weather_search_results = get_weather_data(search_task.weather_query)

    print(weather_search_results)

    return summarizer_chain.stream(
        {
            "user_query": user_query,
            "chat_history": chat_history,
            "web_search_results": web_search_results,
            "weather_search_results": weather_search_results,
        }
    )

# ------------------------------------------------------------------------------
def get_simple_response_stream(
    user_query: str,
    chat_history: str,
) -> Iterator[str]:
    return simple_summarizer_chain.stream(
        {
            "user_query": user_query,
            "chat_history": chat_history,
        }
    )
