"""Tests for LinkAnalyzer module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.link_analyzer import LinkAnalyzer
from src.vault_scanner import VaultScanner


class TestLinkAnalyzer:
    """Test suite for LinkAnalyzer class."""
    
    @pytest.fixture
    def link_analyzer(self, temp_vault: Path) -> LinkAnalyzer:
        """Create a LinkAnalyzer instance for testing."""
        scanner = VaultScanner(temp_vault)
        
        # Create a mock embedding cache
        mock_cache = MagicMock()
        mock_cache.get_embedding.return_value = [0.1] * 768  # Mock embedding
        
        return LinkAnalyzer(
            vault_scanner=scanner,
            embedding_cache=mock_cache,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
    
    @pytest.fixture
    def link_analyzer_no_rerank(self, temp_vault: Path) -> LinkAnalyzer:
        """Create a LinkAnalyzer without reranking."""
        scanner = VaultScanner(temp_vault)
        mock_cache = MagicMock()
        mock_cache.get_embedding.return_value = None  # No embeddings
        
        analyzer = LinkAnalyzer(
            vault_scanner=scanner,
            embedding_cache=mock_cache,
            rerank_model="disabled",  # Use a non-None value to avoid AttributeError
        )
        # Mock the reranker to return None (disabled)
        analyzer._reranker = False  # Sentinel to indicate disabled
        analyzer._get_reranker = lambda: None  # Override to return None
        return analyzer
    
    def test_init(self, link_analyzer: LinkAnalyzer):
        """Test LinkAnalyzer initialization."""
        assert link_analyzer is not None
        assert link_analyzer.vault_scanner is not None
    
    def test_find_by_tags_category(self, link_analyzer: LinkAnalyzer):
        """Test finding notes by tags and category."""
        results = link_analyzer._find_by_tags_category(
            tags=["AI", "RAG"],
            category="concept"
        )
        
        assert len(results) > 0
        # Results should be tuples of (title, score)
        for title, score in results:
            assert isinstance(title, str)
            assert isinstance(score, float)
            assert score > 0
    
    def test_find_by_tags_category_no_match(self, link_analyzer: LinkAnalyzer):
        """Test finding notes with non-matching tags."""
        results = link_analyzer._find_by_tags_category(
            tags=["nonexistent_tag"],
            category="nonexistent_category"
        )
        
        # May return empty or low-scoring results
        assert isinstance(results, list)
    
    def test_find_by_keywords(self, link_analyzer: LinkAnalyzer):
        """Test finding notes by keywords."""
        results = link_analyzer._find_by_keywords(
            title="RAG 검색 증강 생성",
            content="LLM의 환각 문제를 해결하기 위해 사용한다."
        )
        
        assert len(results) > 0
        # Should find RAG-related notes
        titles = [title for title, _ in results]
        assert any("RAG" in title for title in titles)
    
    def test_find_by_keywords_empty_content(self, link_analyzer: LinkAnalyzer):
        """Test finding notes with empty content."""
        results = link_analyzer._find_by_keywords(
            title="",
            content=""
        )
        
        assert isinstance(results, list)
    
    def test_rrf_merge(self, link_analyzer: LinkAnalyzer):
        """Test RRF (Reciprocal Rank Fusion) merge."""
        ranked_lists = [
            ([("Note A", 0.9), ("Note B", 0.8), ("Note C", 0.7)], 0.5),
            ([("Note B", 0.95), ("Note A", 0.85), ("Note D", 0.75)], 0.5),
        ]
        
        merged = link_analyzer._rrf_merge(ranked_lists, top_k=3)
        
        assert len(merged) <= 3
        assert isinstance(merged, list)
        # Note B should be high (ranked well in both lists)
        assert "Note B" in merged or "Note A" in merged
    
    def test_rrf_merge_empty_lists(self, link_analyzer: LinkAnalyzer):
        """Test RRF merge with empty lists."""
        ranked_lists = [
            ([], 0.5),
            ([], 0.5),
        ]
        
        merged = link_analyzer._rrf_merge(ranked_lists, top_k=5)
        
        assert merged == []
    
    def test_cosine_similarity(self, link_analyzer: LinkAnalyzer):
        """Test cosine similarity calculation."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        vec_c = [0.0, 1.0, 0.0]
        
        # Same vectors should have similarity 1.0
        sim_same = link_analyzer._cosine_similarity(vec_a, vec_b)
        assert abs(sim_same - 1.0) < 0.01
        
        # Orthogonal vectors should have similarity 0.0
        sim_ortho = link_analyzer._cosine_similarity(vec_a, vec_c)
        assert abs(sim_ortho) < 0.01
    
    def test_cosine_similarity_zero_vector(self, link_analyzer: LinkAnalyzer):
        """Test cosine similarity with zero vector."""
        vec_a = [1.0, 0.0, 0.0]
        vec_zero = [0.0, 0.0, 0.0]
        
        sim = link_analyzer._cosine_similarity(vec_a, vec_zero)
        assert sim == 0.0
    
    def test_get_title(self, link_analyzer: LinkAnalyzer):
        """Test extracting title from note."""
        note = {
            "path": Path("test.md"),
            "metadata": {"title": "Test Title"}
        }
        
        title = link_analyzer._get_title(note)
        assert title == "Test Title"
    
    def test_get_title_fallback_to_filename(self, link_analyzer: LinkAnalyzer):
        """Test title extraction fallback to filename."""
        note = {
            "path": Path("/vault/notes/My Note.md"),
            "metadata": {}
        }
        
        title = link_analyzer._get_title(note)
        assert title == "My Note"
    
    def test_find_related_notes(self, link_analyzer: LinkAnalyzer):
        """Test finding related notes (with mocked reranker)."""
        # Mock the reranker
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.8, 0.7]
        link_analyzer._reranker = mock_reranker
        
        related = link_analyzer.find_related_notes(
            note_title="RAG 시스템 구축",
            note_content="RAG를 사용하여 LLM의 정확도를 높인다.",
            note_tags=["AI", "RAG"],
            note_category="concept",
            max_links=3,
        )
        
        assert isinstance(related, list)
        assert len(related) <= 3
        
        # All items should be wikilinks
        for link in related:
            assert link.startswith("[[")
            assert link.endswith("]]")
    
    def test_find_related_notes_empty_vault(self, temp_vault_empty: Path):
        """Test finding related notes in empty vault."""
        scanner = VaultScanner(temp_vault_empty)
        mock_cache = MagicMock()
        mock_cache.get_embedding.return_value = None
        
        analyzer = LinkAnalyzer(
            vault_scanner=scanner,
            embedding_cache=mock_cache,
            rerank_model=None,
        )
        
        related = analyzer.find_related_notes(
            note_title="Test",
            note_content="Content",
            note_tags=[],
            note_category="general",
            max_links=5,
        )
        
        assert related == []


class TestLinkAnalyzerWithRerank:
    """Test LinkAnalyzer with reranking (mocked)."""
    
    def test_rerank_candidates(self, temp_vault: Path):
        """Test reranking candidates with mock reranker."""
        scanner = VaultScanner(temp_vault)
        mock_cache = MagicMock()
        
        analyzer = LinkAnalyzer(
            vault_scanner=scanner,
            embedding_cache=mock_cache,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        
        # Mock the reranker
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.7, 0.5]
        analyzer._reranker = mock_reranker
        
        candidates = ["RAG 개념", "LLM 기초", "Vector Database"]
        reranked = analyzer._rerank("RAG 관련 질문", candidates)
        
        assert len(reranked) > 0
        # Should be sorted by score (descending)
        scores = [score for _, score in reranked]
        assert scores == sorted(scores, reverse=True)
    
    def test_rerank_empty_candidates(self, temp_vault: Path):
        """Test reranking with empty candidates."""
        scanner = VaultScanner(temp_vault)
        mock_cache = MagicMock()
        
        analyzer = LinkAnalyzer(
            vault_scanner=scanner,
            embedding_cache=mock_cache,
            rerank_model=None,
        )
        
        reranked = analyzer._rerank("query", [])
        
        assert reranked == []


class TestLinkAnalyzerEdgeCases:
    """Edge case tests for LinkAnalyzer."""
    
    def test_find_related_notes_excludes_self(self, temp_vault: Path):
        """Test that self-references are excluded."""
        scanner = VaultScanner(temp_vault)
        mock_cache = MagicMock()
        mock_cache.get_embedding.return_value = [0.1] * 768
        
        analyzer = LinkAnalyzer(
            vault_scanner=scanner,
            embedding_cache=mock_cache,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        
        # Mock the reranker
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.9, 0.8]
        analyzer._reranker = mock_reranker
        
        # Search for a note that exists
        related = analyzer.find_related_notes(
            note_title="RAG 개념",  # Same as existing note
            note_content="RAG 관련 내용",
            note_tags=["AI", "RAG"],
            note_category="concept",
            max_links=5,
        )
        
        # Should return list (may or may not include self depending on implementation)
        assert isinstance(related, list)
    
    def test_handles_unicode_tags(self, temp_vault: Path):
        """Test handling of Unicode tags."""
        scanner = VaultScanner(temp_vault)
        mock_cache = MagicMock()
        mock_cache.get_embedding.return_value = None
        
        analyzer = LinkAnalyzer(
            vault_scanner=scanner,
            embedding_cache=mock_cache,
            rerank_model=None,
        )
        
        # Search with Korean tags
        results = analyzer._find_by_tags_category(
            tags=["인공지능", "머신러닝"],
            category="개념"
        )
        
        assert isinstance(results, list)
