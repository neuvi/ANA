"""LLM Configuration Module.

Provides flexible LLM provider selection based on configuration.
Supports OpenAI, Anthropic, Ollama, and vLLM.
Includes retry logic for resilient API calls.
"""

import os
import time
from functools import wraps
from typing import TYPE_CHECKING, TypeVar, Callable, Any

from langchain_core.language_models import BaseChatModel

from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.config import ANAConfig

logger = get_logger("llm")

T = TypeVar("T")


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    initial_delay: float = 1.0,
    retryable_exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        initial_delay: Initial delay in seconds
        retryable_exceptions: Tuple of exceptions to retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator


class RetryableLLM:
    """Wrapper around LangChain LLM with built-in retry logic."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        """Initialize retryable LLM wrapper.
        
        Args:
            llm: Underlying LangChain LLM
            max_retries: Maximum retry attempts
            backoff_factor: Exponential backoff multiplier
        """
        self._llm = llm
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def invoke(self, *args, **kwargs) -> Any:
        """Invoke LLM with retry logic."""
        delay = 1.0
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return self._llm.invoke(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Don't retry on authentication errors
                if "api key" in error_str or "authentication" in error_str:
                    logger.error(f"Authentication error, not retrying: {e}")
                    raise
                
                # Don't retry on validation errors
                if "validation" in error_str or "invalid" in error_str:
                    logger.error(f"Validation error, not retrying: {e}")
                    raise
                
                if attempt < self.max_retries:
                    logger.warning(
                        f"LLM attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    logger.error(f"All {self.max_retries + 1} LLM attempts failed")
        
        raise last_exception
    
    def __getattr__(self, name):
        """Proxy other attributes to underlying LLM."""
        return getattr(self._llm, name)


def get_llm(config: "ANAConfig", with_retries: bool = True) -> BaseChatModel:
    """Get LLM instance based on configuration.
    
    Args:
        config: ANA configuration instance
        with_retries: Whether to wrap LLM with retry logic
        
    Returns:
        LangChain chat model instance
        
    Raises:
        ValueError: If unknown LLM provider is specified
    """
    provider = config.llm_provider.lower()
    
    if provider == "openai":
        llm = _get_openai_llm(config)
    elif provider == "anthropic":
        llm = _get_anthropic_llm(config)
    elif provider == "ollama":
        llm = _get_ollama_llm(config)
    elif provider == "vllm":
        llm = _get_vllm_llm(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
    
    if with_retries:
        return RetryableLLM(llm, max_retries=3, backoff_factor=1.5)
    
    return llm


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

