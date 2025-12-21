import os
from deepseek_llm import DeepSeekLLM


def get_llm():
    """Returns a configured LLM instance using OpenAI o3 through OpenRouter"""
    return DeepSeekLLM(
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",  # o3 through OpenRouter
        temperature=0
    )

openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
if not openrouter_api_key:
    raise ValueError("No OpenRouter API key found. Make sure OPENROUTER_API_KEY is in your .env file.")
