"""LLM Configuration Module.

Provides flexible LLM provider selection based on configuration.
Supports OpenAI, Anthropic, Ollama, and vLLM.
"""

import os
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from src.config import ANAConfig


def get_llm(config: "ANAConfig") -> BaseChatModel:
    """Get LLM instance based on configuration.
    
    Args:
        config: ANA configuration instance
        
    Returns:
        LangChain chat model instance
        
    Raises:
        ValueError: If unknown LLM provider is specified
    """
    provider = config.llm_provider.lower()
    
    if provider == "openai":
        return _get_openai_llm(config)
    elif provider == "anthropic":
        return _get_anthropic_llm(config)
    elif provider == "ollama":
        return _get_ollama_llm(config)
    elif provider == "vllm":
        return _get_vllm_llm(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _get_openai_llm(config: "ANAConfig") -> BaseChatModel:
    """Get OpenAI LLM instance."""
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    
    # Load .env file to ensure OPENAI_API_KEY is available
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    model = config.llm_model
    
    # o1/o3 reasoning models don't support temperature
    reasoning_models = ["o1", "o1-mini", "o1-preview", "o3", "o3-mini"]
    is_reasoning_model = any(model.startswith(m) for m in reasoning_models)
    
    if is_reasoning_model:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
        )
    else:
        return ChatOpenAI(
            model=model,
            temperature=config.llm_temperature,
            api_key=api_key,
        )


def _get_anthropic_llm(config: "ANAConfig") -> BaseChatModel:
    """Get Anthropic LLM instance."""
    from langchain_anthropic import ChatAnthropic
    
    return ChatAnthropic(
        model=config.llm_model,
        temperature=config.llm_temperature,
    )


def _get_ollama_llm(config: "ANAConfig") -> BaseChatModel:
    """Get Ollama LLM instance."""
    from langchain_ollama import ChatOllama
    
    return ChatOllama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=config.llm_temperature,
    )


def _get_vllm_llm(config: "ANAConfig") -> BaseChatModel:
    """Get vLLM LLM instance (OpenAI compatible API)."""
    from langchain_openai import ChatOpenAI
    
    return ChatOpenAI(
        model=config.vllm_model,
        base_url=config.vllm_base_url,
        api_key=os.getenv("VLLM_API_KEY", "not-needed"),
        temperature=config.llm_temperature,
    )
