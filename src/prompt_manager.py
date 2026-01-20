"""Prompt Manager for ANA.

Manages custom and default prompts for the analysis pipeline.
Supports file-based prompt loading with fallback to built-in defaults.
"""

from pathlib import Path
from typing import Literal

from src.config import ANAConfig
from src.logging_config import get_logger
from src.prompts import (
    ANALYSIS_PROMPT,
    INTERROGATION_PROMPT,
    SYNTHESIS_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    TAG_SUGGESTION_PROMPT,
    LANGUAGE_RULES,
)

logger = get_logger("prompt_manager")

# Prompt type literals
PromptType = Literal[
    "system",
    "analysis", 
    "interrogation",
    "synthesis",
    "tag_suggestion"
]

# Default file names for prompts in custom prompts directory
PROMPT_FILE_NAMES: dict[PromptType, str] = {
    "system": "system.txt",
    "analysis": "analysis.txt",
    "interrogation": "interrogation.txt",
    "synthesis": "synthesis.txt",
    "tag_suggestion": "tag_suggestion.txt",
}


class PromptManager:
    """Manages custom and default prompts.
    
    Priority order for loading prompts:
    1. Individual prompt path (e.g., custom_analysis_prompt)
    2. Prompt file in custom_prompts_dir
    3. Built-in default prompt
    
    Attributes:
        config: ANA configuration instance
        language: Output language code
    """
    
    def __init__(self, config: ANAConfig | None = None):
        """Initialize PromptManager.
        
        Args:
            config: Optional configuration. Uses default if not provided.
        """
        if config is None:
            from src.config import get_config
            config = get_config()
        
        self.config = config
        self.language = config.output_language
        self._prompt_cache: dict[str, str] = {}
    
    def get_system_prompt(self) -> str:
        """Get system prompt with language rules applied.
        
        Returns:
            Formatted system prompt
        """
        custom_prompt = self._load_prompt("system")
        
        if custom_prompt:
            # Apply language rules to custom prompt if it has the placeholder
            if "{language_rules}" in custom_prompt:
                lang_rules = LANGUAGE_RULES.get(self.language, LANGUAGE_RULES["en"])
                return custom_prompt.format(language_rules=lang_rules)
            return custom_prompt
        
        # Use default system prompt template
        lang_rules = LANGUAGE_RULES.get(self.language, LANGUAGE_RULES["en"])
        return SYSTEM_PROMPT_TEMPLATE.format(language_rules=lang_rules)
    
    def get_analysis_prompt(self) -> str:
        """Get analysis prompt.
        
        Returns:
            Analysis prompt template
        """
        return self._load_prompt("analysis") or ANALYSIS_PROMPT
    
    def get_interrogation_prompt(self) -> str:
        """Get interrogation (question generation) prompt.
        
        Returns:
            Interrogation prompt template
        """
        return self._load_prompt("interrogation") or INTERROGATION_PROMPT
    
    def get_synthesis_prompt(self) -> str:
        """Get synthesis prompt.
        
        Returns:
            Synthesis prompt template
        """
        return self._load_prompt("synthesis") or SYNTHESIS_PROMPT
    
    def get_tag_suggestion_prompt(self) -> str:
        """Get tag suggestion prompt.
        
        Returns:
            Tag suggestion prompt template
        """
        return self._load_prompt("tag_suggestion") or TAG_SUGGESTION_PROMPT
    
    def _load_prompt(self, prompt_type: PromptType) -> str | None:
        """Load a custom prompt by type.
        
        Priority:
        1. Individual prompt path from config
        2. File in custom_prompts_dir
        
        Args:
            prompt_type: Type of prompt to load
            
        Returns:
            Custom prompt content or None if not found
        """
        # Check cache first
        cache_key = f"{prompt_type}_{self.language}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        
        # Try individual prompt path first
        individual_path = self._get_individual_prompt_path(prompt_type)
        if individual_path:
            content = self._read_prompt_file(individual_path)
            if content:
                self._prompt_cache[cache_key] = content
                logger.info(f"Loaded custom {prompt_type} prompt from: {individual_path}")
                return content
        
        # Try custom prompts directory
        if self.config.custom_prompts_dir:
            dir_path = Path(self.config.custom_prompts_dir).expanduser().resolve()
            if dir_path.exists():
                file_name = PROMPT_FILE_NAMES.get(prompt_type)
                if file_name:
                    file_path = dir_path / file_name
                    content = self._read_prompt_file(file_path)
                    if content:
                        self._prompt_cache[cache_key] = content
                        logger.info(f"Loaded custom {prompt_type} prompt from: {file_path}")
                        return content
        
        return None
    
    def _get_individual_prompt_path(self, prompt_type: PromptType) -> Path | None:
        """Get individual prompt path from config.
        
        Args:
            prompt_type: Type of prompt
            
        Returns:
            Path to prompt file or None
        """
        path_map: dict[PromptType, Path | None] = {
            "system": self.config.custom_system_prompt,
            "analysis": self.config.custom_analysis_prompt,
            "interrogation": self.config.custom_interrogation_prompt,
            "synthesis": self.config.custom_synthesis_prompt,
            "tag_suggestion": self.config.custom_tag_suggestion_prompt,
        }
        
        path = path_map.get(prompt_type)
        if path:
            return Path(path).expanduser().resolve()
        return None
    
    def _read_prompt_file(self, path: Path) -> str | None:
        """Read prompt content from a file.
        
        Args:
            path: Path to prompt file
            
        Returns:
            File content or None if not readable
        """
        try:
            if path.exists() and path.is_file():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    return content
                logger.warning(f"Prompt file is empty: {path}")
        except Exception as e:
            logger.error(f"Failed to read prompt file {path}: {e}")
        
        return None
    
    def validate_prompt(self, prompt_type: PromptType) -> tuple[bool, str]:
        """Validate a custom prompt file.
        
        Checks:
        - File exists and is readable
        - Content is not empty
        - Required placeholders are present (for applicable prompts)
        
        Args:
            prompt_type: Type of prompt to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Get the path
        path = self._get_individual_prompt_path(prompt_type)
        if not path:
            if self.config.custom_prompts_dir:
                dir_path = Path(self.config.custom_prompts_dir).expanduser().resolve()
                file_name = PROMPT_FILE_NAMES.get(prompt_type)
                if file_name:
                    path = dir_path / file_name
        
        if not path:
            return True, f"No custom {prompt_type} prompt configured (using default)"
        
        if not path.exists():
            return False, f"Prompt file not found: {path}"
        
        if not path.is_file():
            return False, f"Path is not a file: {path}"
        
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as e:
            return False, f"Cannot read file: {e}"
        
        if not content:
            return False, f"Prompt file is empty: {path}"
        
        # Check required placeholders
        required_placeholders = self._get_required_placeholders(prompt_type)
        missing = [p for p in required_placeholders if p not in content]
        
        if missing:
            return False, f"Missing required placeholders: {', '.join(missing)}"
        
        return True, f"Valid custom {prompt_type} prompt: {path}"
    
    def _get_required_placeholders(self, prompt_type: PromptType) -> list[str]:
        """Get required placeholders for a prompt type.
        
        Args:
            prompt_type: Type of prompt
            
        Returns:
            List of required placeholder strings
        """
        placeholders: dict[PromptType, list[str]] = {
            "system": [],  # language_rules is optional
            "analysis": ["{existing_metadata}", "{raw_note}"],
            "interrogation": [
                "{detected_concepts}",
                "{missing_context}",
                "{raw_note}",
                "{max_questions}",
            ],
            "synthesis": [
                "{raw_note}",
                "{existing_metadata}",
                "{qa_pairs}",
                "{category}",
                "{template}",
            ],
            "tag_suggestion": [
                "{existing_vault_tags}",
                "{note_content}",
                "{max_tags}",
            ],
        }
        return placeholders.get(prompt_type, [])
    
    def validate_all_prompts(self) -> dict[PromptType, tuple[bool, str]]:
        """Validate all configured custom prompts.
        
        Returns:
            Dictionary mapping prompt type to validation result
        """
        results: dict[PromptType, tuple[bool, str]] = {}
        for prompt_type in PROMPT_FILE_NAMES.keys():
            results[prompt_type] = self.validate_prompt(prompt_type)
        return results
    
    def get_prompt_info(self) -> dict[PromptType, dict[str, str]]:
        """Get information about all prompts.
        
        Returns:
            Dictionary with prompt type -> {source, path} info
        """
        info: dict[PromptType, dict[str, str]] = {}
        
        for prompt_type in PROMPT_FILE_NAMES.keys():
            # Check individual path
            individual_path = self._get_individual_prompt_path(prompt_type)
            if individual_path and individual_path.exists():
                info[prompt_type] = {
                    "source": "custom_file",
                    "path": str(individual_path),
                }
                continue
            
            # Check directory
            if self.config.custom_prompts_dir:
                dir_path = Path(self.config.custom_prompts_dir).expanduser().resolve()
                file_name = PROMPT_FILE_NAMES.get(prompt_type)
                if file_name:
                    file_path = dir_path / file_name
                    if file_path.exists():
                        info[prompt_type] = {
                            "source": "custom_dir",
                            "path": str(file_path),
                        }
                        continue
            
            # Using default
            info[prompt_type] = {
                "source": "default",
                "path": "built-in",
            }
        
        return info
    
    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._prompt_cache.clear()
        logger.debug("Prompt cache cleared")


def get_prompt_manager(config: ANAConfig | None = None) -> PromptManager:
    """Get a PromptManager instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        PromptManager instance
    """
    return PromptManager(config)
