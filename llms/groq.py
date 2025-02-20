from langchain_groq import ChatGroq
from langchain_core.runnables import Runnable

from config.envs import GROQ_API_KEY

groq_llama_lite_model: Runnable = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0,
)

groq_llama_regular_model: Runnable = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0,
)
