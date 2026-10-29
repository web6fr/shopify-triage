"""OpenRouter chat model configured from .env."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your .env file.")
    return value


def get_llm(**kwargs) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter, using OPENROUTER_API_KEY and OPENROUTER_MODEL from .env."""
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=_require("OPENROUTER_API_KEY"),
        model=_require("OPENROUTER_MODEL"),
        **kwargs,
    )
