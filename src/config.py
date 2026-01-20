"""ANA Configuration Module.

Manages all configuration settings including Vault path, LLM settings, and agent parameters.
"""

import re
import warnings
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.logging_config import get_logger

logger = get_logger("config")


class ANAConfig(BaseSettings):
    """ANA Agent Configuration.
    
    All settings can be overridden via environment variables with ANA_ prefix.
    Example: ANA_VAULT_PATH, ANA_LLM_PROVIDER, etc.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ANA_",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # =========================================================================
    # Obsidian Vault Settings
    # =========================================================================
    vault_path: Path = Field(
        default=Path("~/vault"),
        description="Path to Obsidian vault"
    )
    
    # =========================================================================
    # LLM Provider Settings
    # =========================================================================
    llm_provider: Literal["openai", "anthropic", "ollama", "vllm"] = Field(
        default="openai",
        description="LLM provider to use"
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Model name for the selected provider"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation"
    )
    
    # =========================================================================
    # Ollama Settings
    # =========================================================================
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama server"
    )
    ollama_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name"
    )
    
    # =========================================================================
    # vLLM Settings
    # =========================================================================
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for vLLM server (OpenAI compatible)"
    )
    vllm_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="vLLM model name"
    )
    
    # =========================================================================
    # Agent Settings
    # =========================================================================
    max_questions: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of questions per round"
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of question rounds"
    )
    
    # =========================================================================
    # Template Settings
    # =========================================================================
    templates_dir: Path = Field(
        default=Path("templates"),
        description="Directory for file-based templates"
    )
    template_db_path: Path = Field(
        default=Path("data/templates.json"),
        description="Path for template database (JSON)"
    )
    
    # =========================================================================
    # Embedding Settings
    # =========================================================================
    embedding_provider: Literal["ollama", "openai", "huggingface"] = Field(
        default="ollama",
        description="Embedding provider to use"
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name"
    )
    embedding_enabled: bool = Field(
        default=True,
        description="Enable embeddings for note linking"
    )
    
    # =========================================================================
    # Note Linking Settings
    # =========================================================================
    max_related_links: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of related note links"
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    rerank_enabled: bool = Field(
        default=True,
        description="Enable reranking for note linking"
    )
    enable_note_linking: bool = Field(
        default=True,
        description="Enable automatic note linking"
    )
    
    # =========================================================================
    # Language Settings
    # =========================================================================
    output_language: str = Field(
        default="ko",
        description="Output language code (ko, en, ja)"
    )
    
    # =========================================================================
    # Embedding Cache Settings
    # =========================================================================
    embedding_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Batch size for embedding processing"
    )
    use_vector_db: bool = Field(
        default=False,
        description="Enable Chroma vector DB backend"
    )
    
    # =========================================================================
    # Custom Prompt Settings
    # =========================================================================
    custom_prompts_dir: Path | None = Field(
        default=None,
        description="Directory for custom prompt files (contains system.txt, analysis.txt, etc.)"
    )
    custom_system_prompt: Path | None = Field(
        default=None,
        description="Path to custom system prompt file"
    )
    custom_analysis_prompt: Path | None = Field(
        default=None,
        description="Path to custom analysis prompt file"
    )
    custom_interrogation_prompt: Path | None = Field(
        default=None,
        description="Path to custom interrogation prompt file"
    )
    custom_synthesis_prompt: Path | None = Field(
        default=None,
        description="Path to custom synthesis prompt file"
    )
    custom_tag_suggestion_prompt: Path | None = Field(
        default=None,
        description="Path to custom tag suggestion prompt file"
    )
    
    # =========================================================================
    # Validators
    # =========================================================================
    
    @field_validator('ollama_base_url', 'vllm_base_url')
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate base URL format."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:[a-zA-Z0-9.-]+|\d{1,3}(?:\.\d{1,3}){3})'  # domain or IP
            r'(?::\d+)?'  # optional port
            r'(?:/.*)?$'  # optional path
        )
        if not url_pattern.match(v):
            raise ValueError(f"Invalid URL format: {v}")
        return v.rstrip('/')  # Remove trailing slash for consistency
    
    @field_validator('output_language')
    @classmethod
    def validate_output_language(cls, v: str) -> str:
        """Validate output language code."""
        valid_languages = {'ko', 'en', 'ja', 'zh'}
        if v.lower() not in valid_languages:
            warnings.warn(f"Unsupported language '{v}', defaulting to 'ko'")
            return 'ko'
        return v.lower()
    
    @field_validator('llm_model')
    @classmethod
    def validate_llm_model(cls, v: str) -> str:
        """Validate LLM model name is not empty."""
        if not v or not v.strip():
            raise ValueError("LLM model name cannot be empty")
        return v.strip()
    
    @model_validator(mode='after')
    def validate_vault_path_warning(self) -> 'ANAConfig':
        """Warn if vault path doesn't exist (don't fail - might be first run)."""
        expanded_path = self.vault_path.expanduser().resolve()
        if not expanded_path.exists():
            logger.warning(f"Vault path does not exist: {expanded_path}")
        elif not (expanded_path / ".obsidian").exists():
            logger.debug(f"No .obsidian folder found in vault: {expanded_path}")
        return self
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def get_vault_path(self) -> Path:
        """Get expanded vault path."""
        return self.vault_path.expanduser().resolve()
    
    def get_templates_dir(self) -> Path:
        """Get expanded templates directory path."""
        return self.templates_dir.expanduser().resolve()
    
    def get_template_db_path(self) -> Path:
        """Get expanded template database path."""
        return self.template_db_path.expanduser().resolve()
    
    def get_effective_model(self) -> str:
        """Get the effective model name based on provider."""
        if self.llm_provider == "ollama":
            return self.ollama_model
        elif self.llm_provider == "vllm":
            return self.vllm_model
        return self.llm_model


def get_config() -> ANAConfig:
    """Get configuration instance."""
    return ANAConfig()

