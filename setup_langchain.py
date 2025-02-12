# Native datetime import.
from datetime import datetime

# Langchain core imports.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser


# Local env imports.
from setup_env import (
    OPENAI_API_KEY,
    OPENAI_LITE_MODEL_ID,
    OPENAI_REGULAR_MODEL_ID,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
