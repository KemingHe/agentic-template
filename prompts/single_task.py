from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tasks.search import SingleSearchTask

single_task_orchestrator_parser = PydanticOutputParser(pydantic_object=SingleSearchTask)

single_task_orchestrator_template: str = """
You are a JSON output generator. Your task is to analyze the query and output ONLY a valid JSON object matching the SingleSearchTask schema.

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
single_task_format_instructions = (
    single_task_orchestrator_parser.get_format_instructions()
)
single_task_messages = [("system", single_task_orchestrator_template)]

# Create the chat prompt template and add format instructions
single_task_orchestrator_prompt = ChatPromptTemplate.from_messages(
    single_task_messages
).partial(format_instructions=single_task_format_instructions)
