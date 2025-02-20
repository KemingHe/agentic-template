from typing import Iterator
from time import perf_counter
# import logging

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import Runnable

from tools.web_search import ddg_text_search
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
        description="Number of web results to fetch (3-10 recommended). Must be 0 if no search needed"
    )
    weather_query: str = Field(
        description="Location name for weather search. Must be specific and unambiguous. Empty if no weather needed"
    )


orchestrator_parser = PydanticOutputParser(pydantic_object=SearchTask)

orchestrator_template: str = """
You are a JSON output generator. Your task is to analyze the query and output ONLY a valid JSON object matching the SearchTask schema.

RESPONSE FORMAT:
- Must be a single JSON object
- No text before or after the JSON
- No explanations or additional formatting
- Boolean values must be 'true' or 'false' (lowercase)
- String values must be in double quotes
- Numbers must be integers without quotes

Example Responses:
1. {{"should_search_web": false, "should_search_weather": true, "web_query": "", "web_query_count": 0, "weather_query": "New York, NY"}}
2. {{"should_search_web": true, "should_search_weather": false, "web_query": "latest quantum computing breakthroughs", "web_query_count": 3, "weather_query": ""}}
3. {{"should_search_web": false, "should_search_weather": false, "web_query": "", "web_query_count": 0, "weather_query": ""}}

Rules for Field Values:
1. should_search_web: Set true only for factual queries needing online information
2. should_search_weather: Set true only for weather-related queries
3. web_query: Must be empty string if should_search_web is false
4. web_query_count: Must be 0 if should_search_web is false, otherwise 3-10
5. weather_query: Must be empty string if should_search_weather is false

Input to Process:
- User query: {user_query}
- Chat history: {chat_history}

{format_instructions}

IMPORTANT: Return ONLY the JSON object with no additional text or formatting.
"""

# Create the system message with format instructions
format_instructions = orchestrator_parser.get_format_instructions()
messages = [("system", orchestrator_template)]

# Create the chat prompt template and add format instructions
orchestrator_prompt = ChatPromptTemplate.from_messages(messages).partial(
    format_instructions=format_instructions
)

summarizer_template: str = """
You are a friendly and knowledgeable AI assistant. Engage in natural conversation while providing accurate information AND citation from the available sources.

CONTEXT:
User Message: {user_query}
Conversation History: {chat_history}
Web Search Results: {web_search_results}
Weather Data: {weather_search_results}

CONVERSATION GUIDELINES:
1. Style:
   - Be friendly and conversational while maintaining professionalism
   - Use natural language and flowing responses
   - Break information into digestible parts
   - Feel free to use appropriate conversational transitions

2. Information Delivery:
   - Start with a direct, engaging response
   - Present key facts naturally within the conversation
   - Use bullet points only for lists or when it improves clarity
   - Include weather data in a conversational way (e.g., "It's currently 20°C, or 68°F")

3. MUST Include Citations:
   - Blend sources naturally: "According to [Source](URL)..."
   - For weather: "Based on [WeatherAPI.com](https://www.weatherapi.com) data..."
   - Acknowledge general knowledge without citation

4. Quality Standards:
   - Maintain accuracy and factual correctness
   - Stay relevant to the user's query
   - Be honest about limitations
   - End with a natural conversation continuation or question

Response Format:
- First part: Engaging direct response
- Middle: Relevant information and context
- End: Natural conversation continuation

Begin your response now:
"""

summarizer_prompt = ChatPromptTemplate.from_template(summarizer_template)

simple_summarizer_prompt = ChatPromptTemplate.from_template(
    summarizer_template
).partial(web_search_results="", weather_search_results="")


def get_web_search_chain_response_stream(
    llm: Runnable,
    user_query: str,
    chat_history: str,
    track_metrics: bool = True,
) -> Iterator[str]:
    start_time: float | None = perf_counter() if track_metrics else None

    orchestrator_chain = orchestrator_prompt | llm | orchestrator_parser
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
            query_count=search_task.web_query_count,
        )

    print(web_search_results)

    weather_search_results: str | None = None
    if search_task.should_search_weather:
        weather_search_results = get_weather_data(
            location=search_task.weather_query,
        )

    print(weather_search_results)

    summarizer_chain = summarizer_prompt | llm | StrOutputParser()
    result_stream: Iterator[str] = summarizer_chain.stream(
        {
            "user_query": user_query,
            "chat_history": chat_history,
            "web_search_results": web_search_results,
            "weather_search_results": weather_search_results,
        }
    )

    for chunk in result_stream:
        if track_metrics and start_time:
            ttfb: float = perf_counter() - start_time
            # logging.info(f"Time to first byte: {ttfb:.4f}s")
            print(f"Web Search Chain - time to first byte: {ttfb:.4f}s")
            start_time = None  # Only log TTFB once
        yield chunk


def get_simple_chain_response_stream(
    llm: Runnable,
    user_query: str,
    chat_history: str,
    track_metrics: bool = True,
) -> Iterator[str]:
    start_time: float | None = perf_counter() if track_metrics else None

    simple_summarizer_chain = simple_summarizer_prompt | llm | StrOutputParser()
    result_stream: Iterator[str] = simple_summarizer_chain.stream(
        {
            "user_query": user_query,
            "chat_history": chat_history,
        }
    )

    for chunk in result_stream:
        if track_metrics and start_time:
            ttfb: float = perf_counter() - start_time
            # logging.info(f"Time to first byte: {ttfb:.4f}s")
            print(f"Simple Chain - time to first byte: {ttfb:.4f}s")
            start_time = None  # Only log TTFB once
        yield chunk
