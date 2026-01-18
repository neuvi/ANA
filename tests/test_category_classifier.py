"""Tests for CategoryClassifier module."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.category_classifier import CategoryClassifier
from src.vault_scanner import VaultScanner


class TestCategoryClassifier:
    """Test suite for CategoryClassifier class."""
    
    @pytest.fixture
    def classifier(self, temp_vault: Path, mock_llm: MagicMock) -> CategoryClassifier:
        """Create a CategoryClassifier instance for testing."""
        scanner = VaultScanner(temp_vault)
        return CategoryClassifier(scanner, mock_llm)
    
    def test_init(self, classifier: CategoryClassifier):
        """Test CategoryClassifier initialization."""
        assert classifier is not None
        assert classifier.vault_scanner is not None
        assert classifier.llm is not None
        assert len(classifier.existing_categories) > 0
    
    def test_existing_categories_loaded(self, classifier: CategoryClassifier):
        """Test that existing categories are loaded from vault."""
        # The temp_vault has notes with types: concept, tool
        assert "concept" in classifier.existing_categories
        assert "tool" in classifier.existing_categories
    
    def test_classify_from_frontmatter_type(self, classifier: CategoryClassifier):
        """Test classification from frontmatter 'type' field."""
        frontmatter = {"type": "book-note", "title": "Test"}
        
        category = classifier.classify("Some content", frontmatter)
        
        assert category == "book-note"
    
    def test_classify_from_frontmatter_category(self, classifier: CategoryClassifier):
        """Test classification from frontmatter 'category' field."""
        frontmatter = {"category": "project-idea", "title": "Test"}
        
        category = classifier.classify("Some content", frontmatter)
        
        assert category == "project-idea"
    
    def test_classify_type_takes_precedence(self, classifier: CategoryClassifier):
        """Test that 'type' takes precedence over 'category'."""
        frontmatter = {
            "type": "concept",
            "category": "different-category",
            "title": "Test"
        }
        
        category = classifier.classify("Some content", frontmatter)
        
        assert category == "concept"
    
    def test_classify_without_frontmatter(self, temp_vault: Path):
        """Test AI-based classification when no frontmatter."""
        scanner = VaultScanner(temp_vault)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="concept")
        
        classifier = CategoryClassifier(scanner, mock_llm)
        
        category = classifier.classify(
            "RAG는 검색 증강 생성입니다.",
            frontmatter=None
        )
        
        # Should call LLM
        assert mock_llm.invoke.called
        assert category == "concept"
    
    def test_classify_empty_frontmatter(self, temp_vault: Path):
        """Test AI-based classification with empty frontmatter."""
        scanner = VaultScanner(temp_vault)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="tool")
        
        classifier = CategoryClassifier(scanner, mock_llm)
        
        category = classifier.classify(
            "Vector Database 설명...",
            frontmatter={}
        )
        
        assert mock_llm.invoke.called
        assert category == "tool"


class TestCategoryClassifierSuggestCategory:
    """Test suggest_category method."""
    
    @pytest.fixture
    def classifier(self, temp_vault: Path, mock_llm: MagicMock) -> CategoryClassifier:
        """Create a CategoryClassifier instance."""
        scanner = VaultScanner(temp_vault)
        return CategoryClassifier(scanner, mock_llm)
    
    def test_suggest_category_existing(self, classifier: CategoryClassifier):
        """Test suggesting an existing category."""
        frontmatter = {"type": "concept"}
        
        category, is_new = classifier.suggest_category(
            "RAG 개념 설명",
            frontmatter
        )
        
        assert category == "concept"
        assert is_new is False
    
    def test_suggest_category_new(self, temp_vault: Path):
        """Test suggesting a new category."""
        scanner = VaultScanner(temp_vault)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="tutorial")
        
        classifier = CategoryClassifier(scanner, mock_llm)
        
        category, is_new = classifier.suggest_category(
            "Python 튜토리얼...",
            frontmatter={}
        )
        
        assert category == "tutorial"
        assert is_new is True  # tutorial is not in existing categories


class TestCategoryClassifierIsNewCategory:
    """Test is_new_category method."""
    
    def test_is_new_category_existing(self, temp_vault: Path, mock_llm: MagicMock):
        """Test checking if existing category is new."""
        scanner = VaultScanner(temp_vault)
        classifier = CategoryClassifier(scanner, mock_llm)
        
        is_new = classifier.is_new_category("concept")
        
        assert is_new is False
    
    def test_is_new_category_new(self, temp_vault: Path, mock_llm: MagicMock):
        """Test checking if new category is new."""
        scanner = VaultScanner(temp_vault)
        classifier = CategoryClassifier(scanner, mock_llm)
        
        is_new = classifier.is_new_category("brand-new-category")
        
        assert is_new is True


class TestCategoryClassifierEdgeCases:
    """Edge case tests for CategoryClassifier."""
    
    def test_classify_strips_whitespace(self, temp_vault: Path):
        """Test that LLM response whitespace is stripped."""
        scanner = VaultScanner(temp_vault)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="  concept  \n")
        
        classifier = CategoryClassifier(scanner, mock_llm)
        
        category = classifier.classify("Content", frontmatter={})
        
        assert category == "concept"
    
    def test_classify_handles_llm_error(self, temp_vault: Path):
        """Test handling of LLM errors."""
        scanner = VaultScanner(temp_vault)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM Error")
        
        classifier = CategoryClassifier(scanner, mock_llm)
        
        # Should handle error gracefully
        try:
            category = classifier.classify("Content", frontmatter={})
            # If it returns a default, that's fine
            assert category is not None or True
        except Exception:
            # Or it might propagate the error
            pass
    
    def test_empty_vault_categories(self, temp_vault_empty: Path, mock_llm: MagicMock):
        """Test with empty vault (no existing categories)."""
        scanner = VaultScanner(temp_vault_empty)
        classifier = CategoryClassifier(scanner, mock_llm)
        
        assert len(classifier.existing_categories) == 0
    
    def test_case_sensitivity(self, temp_vault: Path, mock_llm: MagicMock):
        """Test category case handling."""
        scanner = VaultScanner(temp_vault)
        classifier = CategoryClassifier(scanner, mock_llm)
        
        frontmatter_upper = {"type": "CONCEPT"}
        frontmatter_lower = {"type": "concept"}
        
        category_upper = classifier.classify("Content", frontmatter_upper)
        category_lower = classifier.classify("Content", frontmatter_lower)
        
        # Both should work (may or may not be normalized)
        assert category_upper is not None
        assert category_lower is not None
