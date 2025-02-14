from langchain_core.prompts import ChatPromptTemplate

simple_template: str = """
You are a helpful assistant that can:

- Answer the user query
- Provide information to the best of your abilities
- And have contextful conversations with users

You are provided with the following information:

- Chat history: {chat_history}
- User query: {user_query}
"""

simple_prompt = ChatPromptTemplate.from_template(simple_template)
