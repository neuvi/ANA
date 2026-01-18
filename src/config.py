"""ANA Configuration Module.

Manages all configuration settings including Vault path, LLM settings, and agent parameters.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # Note Linking Settings
    # =========================================================================
    max_related_links: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of related note links"
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model for semantic search"
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    enable_note_linking: bool = Field(
        default=True,
        description="Enable automatic note linking"
    )
    
    def get_vault_path(self) -> Path:
        """Get expanded vault path."""
        return self.vault_path.expanduser().resolve()
    
    def get_templates_dir(self) -> Path:
        """Get expanded templates directory path."""
        return self.templates_dir.expanduser().resolve()
    
    def get_template_db_path(self) -> Path:
        """Get expanded template database path."""
        return self.template_db_path.expanduser().resolve()


def get_config() -> ANAConfig:
    """Get configuration instance."""
    return ANAConfig()
