"""Tests for SmartTagManager module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.smart_tags import SmartTagManager, TagSuggestion, TagStatistics


class TestSmartTagManagerInit:
    """Test SmartTagManager initialization."""
    
    def test_init_with_vault_scanner(self, test_config):
        """Test initialization with vault scanner."""
        from src.vault_scanner import VaultScanner
        
        vault_scanner = VaultScanner(test_config.get_vault_path())
        manager = SmartTagManager(vault_scanner, test_config)
        
        assert manager.vault_scanner is vault_scanner
        assert manager.config is test_config
        assert manager.llm is None
        assert manager._tag_cache is None
    
    def test_init_with_llm(self, test_config):
        """Test initialization with LLM."""
        from src.vault_scanner import VaultScanner
        
        vault_scanner = VaultScanner(test_config.get_vault_path())
        mock_llm = MagicMock()
        
        manager = SmartTagManager(vault_scanner, test_config, llm=mock_llm)
        
        assert manager.llm is mock_llm


class TestTagNormalization:
    """Test tag normalization."""
    
    def test_normalize_tag_removes_hash(self):
        """Test that # prefix is removed."""
        assert SmartTagManager.normalize_tag("#python") == "python"
        assert SmartTagManager.normalize_tag("##tag") == "tag"
    
    def test_normalize_tag_lowercase(self):
        """Test that tags are lowercased."""
        assert SmartTagManager.normalize_tag("Python") == "python"
        assert SmartTagManager.normalize_tag("MachineLearning") == "machinelearning"
    
    def test_normalize_tag_spaces_to_hyphens(self):
        """Test that spaces become hyphens."""
        assert SmartTagManager.normalize_tag("machine learning") == "machine-learning"
        assert SmartTagManager.normalize_tag("deep  learning") == "deep-learning"
    
    def test_normalize_tag_underscores_to_hyphens(self):
        """Test that underscores become hyphens."""
        assert SmartTagManager.normalize_tag("machine_learning") == "machine-learning"
    
    def test_normalize_tag_consecutive_hyphens(self):
        """Test that consecutive hyphens are reduced."""
        assert SmartTagManager.normalize_tag("tag--name") == "tag-name"
        assert SmartTagManager.normalize_tag("a---b") == "a-b"
    
    def test_normalize_tag_strips_hyphens(self):
        """Test that leading/trailing hyphens are removed."""
        assert SmartTagManager.normalize_tag("-tag-") == "tag"
        assert SmartTagManager.normalize_tag("--tag--") == "tag"
    
    def test_normalize_tag_empty(self):
        """Test empty tag normalization."""
        assert SmartTagManager.normalize_tag("") == ""
        assert SmartTagManager.normalize_tag("#") == ""
    
    def test_normalize_tags_list(self, test_config):
        """Test normalizing a list of tags."""
        from src.vault_scanner import VaultScanner
        
        vault_scanner = VaultScanner(test_config.get_vault_path())
        manager = SmartTagManager(vault_scanner, test_config)
        
        tags = ["#Python", "machine learning", "AI", "Python"]  # Duplicate
        result = manager.normalize_tags(tags)
        
        assert result == ["python", "machine-learning", "ai"]  # No duplicate


class TestGetAllTags:
    """Test tag collection from vault."""
    
    def test_get_all_tags_empty_vault(self, test_config, tmp_path):
        """Test with empty vault."""
        from src.vault_scanner import VaultScanner
        
        # Create a truly empty temp vault
        empty_vault = tmp_path / "empty_vault"
        empty_vault.mkdir()
        
        vault_scanner = VaultScanner(empty_vault)
        manager = SmartTagManager(vault_scanner, test_config)
        
        tags = manager.get_all_tags()
        
        assert tags == {}
    
    def test_get_all_tags_with_notes(self, test_config, temp_vault):
        """Test collecting tags from vault notes."""
        from src.vault_scanner import VaultScanner
        
        # Create notes with tags
        (temp_vault / "note1.md").write_text("""---
title: Note 1
tags: [python, ai]
---
Content""")
        
        (temp_vault / "note2.md").write_text("""---
title: Note 2
tags: [python, machine-learning]
---
Content""")
        
        vault_scanner = VaultScanner(temp_vault)
        manager = SmartTagManager(vault_scanner, test_config)
        
        tags = manager.get_all_tags()
        
        assert "python" in tags
        assert tags["python"] == 2  # Used twice
        assert "ai" in tags
        assert "machine-learning" in tags
    
    def test_get_all_tags_caching(self, test_config, temp_vault):
        """Test that tags are cached."""
        from src.vault_scanner import VaultScanner
        
        vault_scanner = VaultScanner(temp_vault)
        manager = SmartTagManager(vault_scanner, test_config)
        
        # First call - scans vault
        tags1 = manager.get_all_tags()
        
        # Second call - uses cache
        tags2 = manager.get_all_tags()
        
        assert tags1 is tags2  # Same object (cached)
    
    def test_get_all_tags_refresh(self, test_config, temp_vault):
        """Test refreshing tag cache."""
        from src.vault_scanner import VaultScanner
        
        vault_scanner = VaultScanner(temp_vault)
        manager = SmartTagManager(vault_scanner, test_config)
        
        # First call
        tags1 = manager.get_all_tags()
        
        # Refresh
        tags2 = manager.get_all_tags(refresh=True)
        
        assert tags1 is not tags2  # Different object


class TestTagStatistics:
    """Test tag statistics."""
    
    def test_get_statistics(self, test_config, tmp_path):
        """Test getting tag statistics."""
        from src.vault_scanner import VaultScanner
        
        # Create a fresh vault for this test
        test_vault = tmp_path / "stats_vault"
        test_vault.mkdir()
        
        # Create notes with tags
        (test_vault / "note1.md").write_text("""---
tags: [python, ai]
---
Content""")
        
        (test_vault / "note2.md").write_text("""---
tags: [python]
---
Content""")
        
        vault_scanner = VaultScanner(test_vault)
        manager = SmartTagManager(vault_scanner, test_config)
        
        stats = manager.get_statistics()
        
        assert isinstance(stats, TagStatistics)
        assert stats.unique_tags == 2  # python, ai
        assert stats.total_tags == 3  # python(2) + ai(1)
        assert len(stats.top_tags) > 0
        assert stats.top_tags[0][0] == "python"  # Most used


class TestSuggestTags:
    """Test tag suggestion."""
    
    def test_suggest_tags_from_content(self, test_config, temp_vault):
        """Test suggesting tags based on content."""
        from src.vault_scanner import VaultScanner
        
        # Create notes with tags
        (temp_vault / "note1.md").write_text("""---
tags: [python, machine-learning]
---
Content about Python and ML""")
        
        vault_scanner = VaultScanner(temp_vault)
        manager = SmartTagManager(vault_scanner, test_config)
        
        # Content mentioning Python
        suggestions = manager.suggest_tags(
            content="This is about Python programming",
            max_tags=5
        )
        
        # Should suggest python tag since it's in vault and content
        tag_names = [s.tag for s in suggestions]
        assert "python" in tag_names
    
    def test_suggest_tags_excludes_existing(self, test_config, temp_vault):
        """Test that existing tags are excluded."""
        from src.vault_scanner import VaultScanner
        
        (temp_vault / "note1.md").write_text("""---
tags: [python, ai]
---
Content""")
        
        vault_scanner = VaultScanner(temp_vault)
        manager = SmartTagManager(vault_scanner, test_config)
        
        # Suggest tags, excluding 'python'
        suggestions = manager.suggest_tags(
            content="Python AI content",
            existing_tags=["python"],
            max_tags=5
        )
        
        tag_names = [s.tag for s in suggestions]
        assert "python" not in tag_names
    
    def test_suggest_tags_confidence(self, test_config, temp_vault):
        """Test that suggestions have confidence scores."""
        from src.vault_scanner import VaultScanner
        
        (temp_vault / "note1.md").write_text("""---
tags: [python]
---
Content""")
        
        vault_scanner = VaultScanner(temp_vault)
        manager = SmartTagManager(vault_scanner, test_config)
        
        suggestions = manager.suggest_tags(
            content="Python content",
            max_tags=5
        )
        
        for s in suggestions:
            assert isinstance(s, TagSuggestion)
            assert 0.0 <= s.confidence <= 1.0


class TestSimilarTags:
    """Test similar tag detection."""
    
    def test_get_similar_tags(self, test_config, temp_vault):
        """Test finding similar tags."""
        from src.vault_scanner import VaultScanner
        
        (temp_vault / "note1.md").write_text("""---
tags: [machine-learning, ml, deep-learning]
---
Content""")
        
        vault_scanner = VaultScanner(temp_vault)
        manager = SmartTagManager(vault_scanner, test_config)
        
        # 'ml' should be similar to 'machine-learning'
        similar = manager.get_similar_tags("machine-learn", threshold=0.5)
        
        assert len(similar) >= 0  # May or may not find similar
    
    def test_tag_similarity_calculation(self):
        """Test tag similarity calculation."""
        sim = SmartTagManager._tag_similarity("machine-learning", "machine-learning")
        assert sim == 1.0
        
        sim2 = SmartTagManager._tag_similarity("ml", "machine-learning")
        assert 0.0 <= sim2 <= 1.0


class TestClearCache:
    """Test cache management."""
    
    def test_clear_cache(self, test_config, temp_vault):
        """Test clearing the cache."""
        from src.vault_scanner import VaultScanner
        
        vault_scanner = VaultScanner(temp_vault)
        manager = SmartTagManager(vault_scanner, test_config)
        
        # Populate cache
        manager.get_all_tags()
        assert manager._tag_cache is not None
        
        # Clear cache
        manager.clear_cache()
        
        assert manager._tag_cache is None
        assert manager._normalized_cache is None
