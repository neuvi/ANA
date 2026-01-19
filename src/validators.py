"""Validators Module.

Runtime validation utilities for ANA components.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import requests

from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.config import ANAConfig

logger = get_logger("validators")


class ValidationError(Exception):
    """Custom validation error with detailed message."""
    
    def __init__(self, message: str, field: str | None = None, suggestion: str | None = None):
        self.field = field
        self.suggestion = suggestion
        super().__init__(message)
    
    def __str__(self) -> str:
        msg = super().__str__()
        if self.suggestion:
            msg += f" (Suggestion: {self.suggestion})"
        return msg


class ValidationResult:
    """Result of a validation check."""
    
    def __init__(
        self,
        is_valid: bool,
        message: str = "",
        warnings: list[str] | None = None
    ):
        self.is_valid = is_valid
        self.message = message
        self.warnings = warnings or []
    
    def __bool__(self) -> bool:
        return self.is_valid


def validate_raw_note(content: str, max_length: int = 50000, min_length: int = 10) -> ValidationResult:
    """Validate raw note content.
    
    Args:
        content: Raw note content
        max_length: Maximum allowed length (default: 50000 chars)
        min_length: Minimum length for warning (default: 10 chars)
        
    Returns:
        ValidationResult with status and any warnings
    """
    warnings = []
    
    # Check for empty content
    if not content or not content.strip():
        return ValidationResult(
            is_valid=False,
            message="Note content cannot be empty"
        )
    
    stripped_content = content.strip()
    
    # Check for very short content
    if len(stripped_content) < min_length:
        warnings.append(f"Note is very short ({len(stripped_content)} chars), may not generate meaningful output")
    
    # Check for max length
    if len(content) > max_length:
        return ValidationResult(
            is_valid=False,
            message=f"Note exceeds maximum length of {max_length} characters ({len(content)} chars)"
        )
    
    # Check if content is mostly whitespace
    non_whitespace = len(content.replace(" ", "").replace("\n", "").replace("\t", ""))
    if non_whitespace < len(content) * 0.1:
        warnings.append("Note appears to be mostly whitespace")
    
    return ValidationResult(
        is_valid=True,
        message="Valid",
        warnings=warnings
    )


def validate_vault_connection(config: "ANAConfig") -> ValidationResult:
    """Validate vault path exists and is accessible.
    
    Args:
        config: ANAConfig instance
        
    Returns:
        ValidationResult with status
    """
    warnings = []
    vault_path = config.get_vault_path()
    
    if not vault_path.exists():
        return ValidationResult(
            is_valid=False,
            message=f"Vault path does not exist: {vault_path}"
        )
    
    if not vault_path.is_dir():
        return ValidationResult(
            is_valid=False,
            message=f"Vault path is not a directory: {vault_path}"
        )
    
    # Check for .obsidian directory
    obsidian_dir = vault_path / ".obsidian"
    if not obsidian_dir.exists():
        warnings.append("No .obsidian folder found - this may not be an Obsidian vault")
    
    # Check if readable
    try:
        list(vault_path.iterdir())
    except PermissionError:
        return ValidationResult(
            is_valid=False,
            message=f"Cannot read vault directory: {vault_path}"
        )
    
    return ValidationResult(
        is_valid=True,
        message=f"Vault accessible: {vault_path}",
        warnings=warnings
    )


def validate_ollama_connection(config: "ANAConfig") -> ValidationResult:
    """Validate Ollama server is running and accessible.
    
    Args:
        config: ANAConfig instance
        
    Returns:
        ValidationResult with status
    """
    base_url = config.ollama_base_url
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check if required model is available
            required_model = config.ollama_model if config.llm_provider == "ollama" else config.embedding_model
            
            if any(required_model in name for name in model_names):
                return ValidationResult(
                    is_valid=True,
                    message=f"Ollama running with {len(models)} models, '{required_model}' available"
                )
            else:
                return ValidationResult(
                    is_valid=True,
                    message=f"Ollama running but model '{required_model}' not found",
                    warnings=[f"Run 'ollama pull {required_model}' to install"]
                )
        else:
            return ValidationResult(
                is_valid=False,
                message=f"Ollama returned status {response.status_code}"
            )
    except requests.exceptions.ConnectionError:
        return ValidationResult(
            is_valid=False,
            message=f"Cannot connect to Ollama at {base_url}"
        )
    except requests.exceptions.Timeout:
        return ValidationResult(
            is_valid=False,
            message=f"Connection to Ollama timed out"
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            message=f"Ollama connection error: {e}"
        )


def validate_llm_connection(config: "ANAConfig", timeout: int = 30) -> ValidationResult:
    """Validate LLM connection with a test prompt.
    
    Args:
        config: ANAConfig instance
        timeout: Timeout in seconds
        
    Returns:
        ValidationResult with status
    """
    try:
        from src.llm_config import get_llm
        
        llm = get_llm(config)
        response = llm.invoke("Say 'OK' if you can hear me. Reply with only 'OK'.")
        
        if response and hasattr(response, 'content'):
            content = response.content.strip()[:50]
            return ValidationResult(
                is_valid=True,
                message=f"LLM responded: '{content}'"
            )
        else:
            return ValidationResult(
                is_valid=False,
                message="LLM returned empty response"
            )
    except ImportError as e:
        return ValidationResult(
            is_valid=False,
            message=f"Missing dependency: {e}"
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            message=f"LLM connection failed: {e}"
        )


def validate_note_title(title: str, max_length: int = 200) -> ValidationResult:
    """Validate note title.
    
    Args:
        title: Note title
        max_length: Maximum allowed length
        
    Returns:
        ValidationResult with status
    """
    warnings = []
    
    if not title or not title.strip():
        return ValidationResult(
            is_valid=False,
            message="Title cannot be empty"
        )
    
    title = title.strip()
    
    if len(title) > max_length:
        return ValidationResult(
            is_valid=False,
            message=f"Title exceeds maximum length of {max_length} characters"
        )
    
    # Check for problematic characters in file names
    problematic_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    found_chars = [c for c in problematic_chars if c in title]
    if found_chars:
        warnings.append(f"Title contains characters that may cause issues in filenames: {found_chars}")
    
    # Check for leading/trailing dots or spaces (problematic on some systems)
    if title.startswith('.') or title.endswith('.'):
        warnings.append("Title starts or ends with a dot")
    
    return ValidationResult(
        is_valid=True,
        message="Valid title",
        warnings=warnings
    )


def sanitize_filename(title: str) -> str:
    """Sanitize title for use as filename.
    
    Args:
        title: Raw title
        
    Returns:
        Sanitized filename-safe string
    """
    # Replace problematic characters
    replacements = {
        '/': '-',
        '\\': '-',
        ':': '-',
        '*': '',
        '?': '',
        '"': "'",
        '<': '(',
        '>': ')',
        '|': '-',
    }
    
    result = title
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    
    # Remove leading/trailing dots and spaces
    result = result.strip('. ')
    
    # Collapse multiple spaces/hyphens
    import re
    result = re.sub(r'[-\s]+', ' ', result)
    
    # Limit length
    if len(result) > 200:
        result = result[:197] + "..."
    
    return result
