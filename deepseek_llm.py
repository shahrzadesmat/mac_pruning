# import os
# import aiohttp
# import json
# import asyncio
# from typing import Optional, Dict, List
# from dataclasses import dataclass
# from langchain_core.messages import BaseMessage
# from langchain.schema import HumanMessage
# from dotenv import load_dotenv, find_dotenv

# # Import Groq if you want to support that provider.
# from groq import Groq


# from openai import AsyncOpenAI

# load_dotenv()

# @dataclass
# class DeepSeekResponse:
#     """Unified response class"""
#     content: str
    
#     def __str__(self) -> str:
#         return self.content

# class DeepSeekLLM:
#     """
#     Unified LLM class that can use either OpenRouter, Groq, or OpenAI as the backend.
    
#     To switch providers, simply pass provider="openrouter", provider="groq", or provider="openai" when initializing.
#     """
#     def __init__(
#         self,
#         provider: str = "openrouter",  # or "groq" or "openai"
#         model: str = None,
#         temperature: float = 0,
#         api_key: Optional[str] = None,
#     ):
#         self.provider = provider.lower()
#         self.temperature = temperature

#         if self.provider == "openrouter":
#             # Use OpenRouter settings:
#             self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
#             if not self.api_key:
#                 raise ValueError("No OpenRouter API key provided. Set OPENROUTER_API_KEY in your .env file.")
#             # Default OpenRouter model if none provided:
#             self.model = model or "deepseek/deepseek-chat:free"
#             self.api_url = "https://openrouter.ai/api/v1/chat/completions"
#             self.headers = {
#                 "Authorization": f"Bearer {self.api_key}",
#                 "HTTP-Referer": "https://anonymous.submission",
#                 "X-Title": "Hyperparameter Optimization",
#                 "Content-Type": "application/json"
#             }
#             print(f"Initialized OpenRouter with model: {self.model}")
#         elif self.provider == "groq":
#             # Use Groq settings:
#             self.api_key = api_key or os.getenv("GROQ_API_KEY")
#             if not self.api_key:
#                 raise ValueError("No Groq API key provided. Set GROQ_API_KEY in your .env file.")
#             # Default Groq model if none provided:
#             self.model = model or "llama-3.3-70b-versatile"
#             # Initialize the Groq client:
#             self.client = Groq(api_key=self.api_key)
#             print(f"Initialized Groq with model: {self.model}")
#         elif self.provider == "openai":
#             # Check if OpenAI is available
#             # Use OpenAI settings:
#             self.api_key = api_key or os.getenv("OPENAI_API_KEY")
#             if not self.api_key:
#                 raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY in your .env file.")
#             # Default OpenAI model if none provided:
#             self.model = model or "gpt-4-turbo"
#             # Initialize the OpenAI client:
#             self.client = AsyncOpenAI(api_key=self.api_key)
#             print(f"Initialized OpenAI with model: {self.model}")
#         else:
#             raise ValueError("Unsupported provider. Use 'openai', 'openrouter', or 'groq'.")
        
#     def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
#         """Format messages to a list of dictionaries with 'role' and 'content'."""
#         formatted_messages = []
#         for msg in messages:
#             if "human" in msg.__class__.__name__.lower():
#                 role = "user"
#             elif "ai" in msg.__class__.__name__.lower():
#                 role = "assistant"
#             else:
#                 role = "system"
#             formatted_messages.append({
#                 "role": role,
#                 "content": msg.content
#             })
#         return formatted_messages

#     async def ainvoke(self, messages: List[BaseMessage]) -> DeepSeekResponse:
#         """Unified asynchronous method to invoke the backend API."""
#         formatted_messages = self._format_messages(messages)
        
#         if self.provider == "openrouter":
#             # Prepare payload for OpenRouter
#             payload = {
#                 "model": self.model,
#                 "messages": formatted_messages,
#                 "temperature": self.temperature,
#                 "max_tokens": 1000
#             }
#             # Make asynchronous API call via aiohttp
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(
#                     self.api_url, 
#                     headers=self.headers, 
#                     json=payload,
#                     timeout=30  # 30 second timeout
#                 ) as resp:
#                     print(f"\nOpenRouter Response Status: {resp.status}")
#                     response_text = await resp.text()
                    
#                     if resp.status != 200:
#                         try:
#                             error_data = json.loads(response_text)
#                             if 'error' in error_data:
#                                 error_info = error_data['error']
#                                 print(f"Error: {error_info.get('message')}")
#                             raise Exception(
#                                 f"OpenRouter API error:\n"
#                                 f"Status: {resp.status}\n"
#                                 f"Error: {error_data.get('error', {}).get('message', 'Unknown error')}"
#                             )
#                         except json.JSONDecodeError:
#                             raise Exception(f"API returned non-JSON response with status {resp.status}: {response_text}")
                    
#                     try:
#                         response_data = json.loads(response_text)
#                     except json.JSONDecodeError:
#                         raise Exception(f"Failed to parse API response as JSON: {response_text}")
                    
#                     choices = response_data.get("choices", [])
#                     if not choices:
#                         raise Exception("No choices in API response")
#                     message = choices[0].get("message", {})
#                     content = message.get("content")
#                     if not content:
#                         raise Exception("No content in API response message")
                    
#                     return DeepSeekResponse(content=content)
        
#         elif self.provider == "groq":
#             # For Groq, the client library is synchronous.
#             # We wrap the synchronous call using run_in_executor to keep the async interface.
#             loop = asyncio.get_event_loop()
#             response = await loop.run_in_executor(None, self._groq_invoke, formatted_messages)
#             return response
            
#         elif self.provider == "openai":
#             try:
#                 # Using OpenAI's async client
#                 response = await self.client.chat.completions.create(
#                     model=self.model,
#                     messages=formatted_messages,
#                     temperature=self.temperature,
#                     max_tokens=1000
#                 )
#                 content = response.choices[0].message.content
#                 return DeepSeekResponse(content=content)
#             except Exception as e:
#                 raise Exception(f"OpenAI API error: {str(e)}")

#     def _groq_invoke(self, formatted_messages: List[Dict[str, str]]) -> DeepSeekResponse:
#         """Synchronous invocation for Groq using its client."""
#         try:
#             chat_completion = self.client.chat.completions.create(
#                 messages=formatted_messages,
#                 model=self.model,
#             )
#             content = chat_completion.choices[0].message.content
#             return DeepSeekResponse(content=content)
#         except Exception as e:
#             raise Exception(f"Groq API error: {str(e)}")



import os
import aiohttp
import json
import asyncio
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
from langchain_core.messages import BaseMessage
from langchain.schema import HumanMessage
from dotenv import load_dotenv

from groq import Groq
from openai import AsyncOpenAI

load_dotenv()

@dataclass
class DeepSeekResponse:
    content: Union[str, dict]
    
    def __str__(self) -> str:
        return json.dumps(self.content) if isinstance(self.content, dict) else self.content


class DeepSeekLLM:
    def __init__(
        self,
        provider: str = "openrouter",  # or "groq" or "openai"
        model: str = None,
        temperature: float = 0,
        api_key: Optional[str] = None,
        output_format: str = "text"  # "text" or "json"
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.output_format = output_format

        if self.provider == "openrouter":
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("No OpenRouter API key provided. Set OPENROUTER_API_KEY in your .env file.")

            self.model = model or "deepseek/deepseek-chat:free"
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://anonymous.submission",
                "X-Title": "Hyperparameter Optimization",
                "Content-Type": "application/json"
            }
            print(f"✅ Initialized OpenRouter with model: {self.model}")

        elif self.provider == "groq":
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("No Groq API key provided. Set GROQ_API_KEY in your .env file.")
            self.model = model or "llama-3.3-70b-versatile"
            self.client = Groq(api_key=self.api_key)
            print(f"✅ Initialized Groq with model: {self.model}")

        elif self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY in your .env file.")
            self.model = model or "gpt-4-turbo"
            self.client = AsyncOpenAI(api_key=self.api_key)
            print(f"✅ Initialized OpenAI with model: {self.model}")

        else:
            raise ValueError("Unsupported provider. Use 'openai', 'openrouter', or 'groq'.")

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        formatted = []
        for msg in messages:
            if "human" in msg.__class__.__name__.lower():
                role = "user"
            elif "ai" in msg.__class__.__name__.lower():
                role = "assistant"
            else:
                role = "system"
            formatted.append({"role": role, "content": msg.content})
        return formatted

    async def ainvoke(self, messages: List[BaseMessage]) -> DeepSeekResponse:
        # Add system instruction to enforce JSON format
        if self.output_format == "json":
            instruction = HumanMessage(content=(
                "Only return valid JSON. No explanation, no markdown, no formatting. "
                "Just output one JSON object with hyperparameter names and ranges like:\n"
                '{ "learning_rate": {"type": "float", "min": 0.0001, "max": 0.01} }'
            ))
            messages.insert(0, instruction)

        formatted_messages = self._format_messages(messages)

        if self.provider == "openrouter":
            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": self.temperature,
                "max_tokens": 1000
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=90
                ) as resp:
                    print(f"\nOpenRouter Response Status: {resp.status}")
                    response_text = await resp.text()

                    if resp.status != 200:
                        try:
                            error_data = json.loads(response_text)
                            raise Exception(error_data.get("error", {}).get("message", "Unknown error"))
                        except json.JSONDecodeError:
                            raise Exception(f"Non-JSON error response: {response_text}")

                    try:
                        response_data = json.loads(response_text)
                        content = response_data["choices"][0]["message"]["content"]
                    except Exception as e:
                        raise Exception(f"Failed to parse LLM response: {str(e)}")

                    if self.output_format == "json":
                        try:
                            parsed = json.loads(content)
                            return DeepSeekResponse(content=parsed)
                        except json.JSONDecodeError:
                            raise ValueError(f"Expected JSON, but got:\n{content}")

                    return DeepSeekResponse(content=content)

        elif self.provider == "groq":
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._groq_invoke, formatted_messages)
            return response

        elif self.provider == "openai":
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=self.temperature,
                    max_tokens=1000
                )
                content = response.choices[0].message.content
                if self.output_format == "json":
                    try:
                        parsed = json.loads(content)
                        return DeepSeekResponse(content=parsed)
                    except json.JSONDecodeError:
                        raise ValueError(f"Expected JSON, but got:\n{content}")
                return DeepSeekResponse(content=content)
            except Exception as e:
                raise Exception(f"OpenAI API error: {str(e)}")

    def _groq_invoke(self, formatted_messages: List[Dict[str, str]]) -> DeepSeekResponse:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=formatted_messages,
                model=self.model,
            )
            content = chat_completion.choices[0].message.content
            if self.output_format == "json":
                try:
                    parsed = json.loads(content)
                    return DeepSeekResponse(content=parsed)
                except json.JSONDecodeError:
                    raise ValueError(f"Expected JSON, but got:\n{content}")
            return DeepSeekResponse(content=content)
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
