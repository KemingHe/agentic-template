from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable

from config.envs import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_LITE_MODEL_ID,
    ANTHROPIC_REGULAR_MODEL_ID,
)

anthropic_lite_model: Runnable = ChatAnthropic(
    api_key=ANTHROPIC_API_KEY,
    model=ANTHROPIC_LITE_MODEL_ID,
    temperature=0,
)
anthropic_regular_model: Runnable = ChatAnthropic(
    api_key=ANTHROPIC_API_KEY,
    model=ANTHROPIC_REGULAR_MODEL_ID,
    temperature=0,
)
