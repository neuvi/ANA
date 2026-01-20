"""Tests for PromptManager module."""

import tempfile
from pathlib import Path

import pytest

from src.config import ANAConfig
from src.prompt_manager import PromptManager, PROMPT_FILE_NAMES
from src.prompts import (
    ANALYSIS_PROMPT,
    INTERROGATION_PROMPT,
    SYNTHESIS_PROMPT,
    SYSTEM_PROMPT_TEMPLATE,
    TAG_SUGGESTION_PROMPT,
)


class TestPromptManagerDefaults:
    """Test default prompt behavior."""
    
    def test_get_system_prompt_default(self):
        """Default system prompt should be returned."""
        pm = PromptManager()
        prompt = pm.get_system_prompt()
        
        assert prompt is not None
        assert len(prompt) > 0
        assert "Knowledge Architect" in prompt
    
    def test_get_analysis_prompt_default(self):
        """Default analysis prompt should be returned."""
        pm = PromptManager()
        prompt = pm.get_analysis_prompt()
        
        assert prompt == ANALYSIS_PROMPT
        assert "{raw_note}" in prompt
        assert "{existing_metadata}" in prompt
    
    def test_get_interrogation_prompt_default(self):
        """Default interrogation prompt should be returned."""
        pm = PromptManager()
        prompt = pm.get_interrogation_prompt()
        
        assert prompt == INTERROGATION_PROMPT
        assert "{max_questions}" in prompt
    
    def test_get_synthesis_prompt_default(self):
        """Default synthesis prompt should be returned."""
        pm = PromptManager()
        prompt = pm.get_synthesis_prompt()
        
        assert prompt == SYNTHESIS_PROMPT
        assert "{qa_pairs}" in prompt
        assert "{template}" in prompt
    
    def test_get_tag_suggestion_prompt_default(self):
        """Default tag suggestion prompt should be returned."""
        pm = PromptManager()
        prompt = pm.get_tag_suggestion_prompt()
        
        assert prompt == TAG_SUGGESTION_PROMPT
        assert "{existing_vault_tags}" in prompt


class TestPromptManagerCustomPrompts:
    """Test custom prompt loading."""
    
    def test_load_custom_prompt_from_individual_file(self):
        """Custom prompt should be loaded from individual file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create custom analysis prompt
            custom_prompt = "Custom analysis: {existing_metadata}\n{raw_note}"
            prompt_file = Path(tmpdir) / "custom_analysis.txt"
            prompt_file.write_text(custom_prompt, encoding="utf-8")
            
            # Create config with custom prompt path
            config = ANAConfig(
                custom_analysis_prompt=prompt_file,
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            prompt = pm.get_analysis_prompt()
            
            assert prompt == custom_prompt
    
    def test_load_custom_prompt_from_directory(self):
        """Custom prompts should be loaded from prompts directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()
            
            # Create custom analysis prompt in directory
            custom_prompt = "Directory analysis: {existing_metadata}\n{raw_note}"
            (prompts_dir / "analysis.txt").write_text(custom_prompt, encoding="utf-8")
            
            # Create config with custom prompts directory
            config = ANAConfig(
                custom_prompts_dir=prompts_dir,
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            prompt = pm.get_analysis_prompt()
            
            assert prompt == custom_prompt
    
    def test_individual_path_overrides_directory(self):
        """Individual prompt path should override directory prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory with prompt
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / "analysis.txt").write_text(
                "Directory prompt: {existing_metadata}\n{raw_note}",
                encoding="utf-8"
            )
            
            # Create individual prompt file
            individual_prompt = "Individual prompt: {existing_metadata}\n{raw_note}"
            individual_file = Path(tmpdir) / "my_analysis.txt"
            individual_file.write_text(individual_prompt, encoding="utf-8")
            
            # Config with both directory and individual path
            config = ANAConfig(
                custom_prompts_dir=prompts_dir,
                custom_analysis_prompt=individual_file,
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            prompt = pm.get_analysis_prompt()
            
            # Individual path should take priority
            assert prompt == individual_prompt
    
    def test_fallback_to_default_when_file_missing(self):
        """Should fallback to default when custom file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ANAConfig(
                custom_analysis_prompt=Path(tmpdir) / "nonexistent.txt",
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            prompt = pm.get_analysis_prompt()
            
            # Should fallback to default
            assert prompt == ANALYSIS_PROMPT
    
    def test_fallback_to_default_when_file_empty(self):
        """Should fallback to default when custom file is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_file = Path(tmpdir) / "empty.txt"
            empty_file.write_text("", encoding="utf-8")
            
            config = ANAConfig(
                custom_analysis_prompt=empty_file,
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            prompt = pm.get_analysis_prompt()
            
            # Should fallback to default
            assert prompt == ANALYSIS_PROMPT


class TestPromptManagerCaching:
    """Test prompt caching behavior."""
    
    def test_prompts_are_cached(self):
        """Prompts should be cached after first load."""
        pm = PromptManager()
        
        # First call
        prompt1 = pm.get_analysis_prompt()
        
        # Second call should return cached value
        prompt2 = pm.get_analysis_prompt()
        
        assert prompt1 == prompt2
    
    def test_cache_cleared_properly(self):
        """Cache should be clearable."""
        pm = PromptManager()
        
        # Load prompt
        pm.get_analysis_prompt()
        assert len(pm._prompt_cache) > 0 or pm._prompt_cache == {}
        
        # Clear cache
        pm.clear_cache()
        
        assert pm._prompt_cache == {}


class TestPromptValidation:
    """Test prompt validation."""
    
    def test_validate_prompt_with_valid_file(self):
        """Valid prompt file should pass validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid analysis prompt
            valid_prompt = "Analyze: {existing_metadata}\n{raw_note}"
            prompt_file = Path(tmpdir) / "analysis.txt"
            prompt_file.write_text(valid_prompt, encoding="utf-8")
            
            config = ANAConfig(
                custom_analysis_prompt=prompt_file,
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            is_valid, message = pm.validate_prompt("analysis")
            
            assert is_valid is True
    
    def test_validate_prompt_missing_placeholders(self):
        """Prompt missing required placeholders should fail validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create prompt missing required placeholders
            invalid_prompt = "Analyze this note without placeholders"
            prompt_file = Path(tmpdir) / "analysis.txt"
            prompt_file.write_text(invalid_prompt, encoding="utf-8")
            
            config = ANAConfig(
                custom_analysis_prompt=prompt_file,
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            is_valid, message = pm.validate_prompt("analysis")
            
            assert is_valid is False
            assert "Missing required placeholders" in message
    
    def test_validate_prompt_file_not_found(self):
        """Non-existent prompt file should fail validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ANAConfig(
                custom_analysis_prompt=Path(tmpdir) / "nonexistent.txt",
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            is_valid, message = pm.validate_prompt("analysis")
            
            assert is_valid is False
            assert "not found" in message
    
    def test_validate_all_prompts(self):
        """validate_all_prompts should check all prompt types."""
        pm = PromptManager()
        results = pm.validate_all_prompts()
        
        # Should have entries for all prompt types
        assert set(results.keys()) == set(PROMPT_FILE_NAMES.keys())
        
        # All defaults should be valid (using defaults)
        for prompt_type, (is_valid, _) in results.items():
            # Default prompts are considered valid (with "using default" message)
            assert is_valid is True


class TestPromptInfo:
    """Test prompt information retrieval."""
    
    def test_get_prompt_info_all_defaults(self):
        """get_prompt_info should show all prompts as default."""
        pm = PromptManager()
        info = pm.get_prompt_info()
        
        assert len(info) == len(PROMPT_FILE_NAMES)
        
        for prompt_type, data in info.items():
            assert data["source"] == "default"
            assert data["path"] == "built-in"
    
    def test_get_prompt_info_with_custom_file(self):
        """get_prompt_info should show custom file source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "analysis.txt"
            prompt_file.write_text("Custom: {existing_metadata}\n{raw_note}", encoding="utf-8")
            
            config = ANAConfig(
                custom_analysis_prompt=prompt_file,
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            info = pm.get_prompt_info()
            
            assert info["analysis"]["source"] == "custom_file"
            assert str(prompt_file) in info["analysis"]["path"]
    
    def test_get_prompt_info_with_custom_directory(self):
        """get_prompt_info should show custom directory source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir) / "prompts"
            prompts_dir.mkdir()
            (prompts_dir / "analysis.txt").write_text(
                "Custom: {existing_metadata}\n{raw_note}",
                encoding="utf-8"
            )
            
            config = ANAConfig(
                custom_prompts_dir=prompts_dir,
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            info = pm.get_prompt_info()
            
            assert info["analysis"]["source"] == "custom_dir"


class TestLanguageSupport:
    """Test language-specific prompt behavior."""
    
    def test_system_prompt_korean_language(self):
        """System prompt should include Korean language rules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ANAConfig(
                output_language="ko",
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            prompt = pm.get_system_prompt()
            
            assert "한국어" in prompt or "Korean" in prompt
    
    def test_system_prompt_english_language(self):
        """System prompt should include English language rules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ANAConfig(
                output_language="en",
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            prompt = pm.get_system_prompt()
            
            assert "English" in prompt
    
    def test_custom_system_prompt_with_language_placeholder(self):
        """Custom system prompt should support language rules placeholder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_prompt = "You are a helper.\n\nLanguage rules:\n{language_rules}"
            prompt_file = Path(tmpdir) / "system.txt"
            prompt_file.write_text(custom_prompt, encoding="utf-8")
            
            config = ANAConfig(
                custom_system_prompt=prompt_file,
                output_language="ko",
                vault_path=Path(tmpdir),
            )
            
            pm = PromptManager(config)
            prompt = pm.get_system_prompt()
            
            assert "You are a helper" in prompt
            assert "한국어" in prompt or "Korean" in prompt
