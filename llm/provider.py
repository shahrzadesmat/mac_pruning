import os
from deepseek_llm import DeepSeekLLM

def get_llm():
    """Returns a configured LLM instance using OpenRouter with GPT-4.5-mini"""
    return DeepSeekLLM(
        provider="openrouter",
        model="openai/gpt-4o-mini",  # Using GPT-4o through OpenRouter
        temperature=0
    )

openrouter_api_key = os.environ.get('OPENROUTER_API_KEY')
if not openrouter_api_key:
    raise ValueError("No OpenRouter API key found. Make sure OPENROUTER_API_KEY is in your .env file.")
