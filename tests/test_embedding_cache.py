"""Tests for Embedding Cache Module.

Tests for src/embedding_cache.py functionality.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.embedding_cache import EmbeddingCache


class TestEmbeddingCacheInit:
    """Test EmbeddingCache initialization."""
    
    def test_init_creates_cache_dir(self, temp_vault):
        """Test that initialization creates cache directory."""
        cache = EmbeddingCache(vault_path=temp_vault)
        
        cache_dir = temp_vault / ".ana"
        assert cache_dir.exists()
    
    def test_init_with_custom_params(self, temp_vault):
        """Test initialization with custom parameters."""
        cache = EmbeddingCache(
            vault_path=temp_vault,
            ollama_base_url="http://custom:11434",
            embedding_model="custom-model",
            batch_size=20,
        )
        
        assert cache.ollama_base_url == "http://custom:11434"
        assert cache.embedding_model == "custom-model"
        assert cache.batch_size == 20
    
    def test_init_loads_existing_cache(self, temp_vault):
        """Test that existing cache is loaded."""
        # Create cache files
        cache_dir = temp_vault / ".ana"
        cache_dir.mkdir(parents=True)
        
        embeddings = {"test_key": [0.1, 0.2, 0.3]}
        (cache_dir / "embeddings.json").write_text(json.dumps(embeddings))
        
        meta = {"test.md": {"hash": "abc123", "embedding_key": "test_key"}}
        (cache_dir / "embeddings_meta.json").write_text(json.dumps(meta))
        
        cache = EmbeddingCache(vault_path=temp_vault)
        
        assert "test_key" in cache._embeddings
        assert "test.md" in cache._meta


class TestComputeHash:
    """Test hash computation."""
    
    def test_compute_hash_deterministic(self, embedding_cache):
        """Test that hash is deterministic."""
        content = "Test content"
        hash1 = embedding_cache._compute_hash(content)
        hash2 = embedding_cache._compute_hash(content)
        
        assert hash1 == hash2
    
    def test_compute_hash_different_content(self, embedding_cache):
        """Test that different content produces different hash."""
        hash1 = embedding_cache._compute_hash("Content A")
        hash2 = embedding_cache._compute_hash("Content B")
        
        assert hash1 != hash2


class TestNeedsUpdate:
    """Test update detection."""
    
    def test_needs_update_for_new_file(self, embedding_cache, temp_vault):
        """Test that new files need update."""
        new_file = temp_vault / "new_note.md"
        new_file.write_text("New content")
        
        assert embedding_cache.needs_update(new_file) is True
    
    def test_needs_update_for_unchanged_file(self, embedding_cache, temp_vault):
        """Test that unchanged files don't need update."""
        file_path = temp_vault / "existing.md"
        content = "Existing content"
        file_path.write_text(content)
        
        # Add to cache
        rel_path = str(file_path.relative_to(temp_vault))
        file_hash = embedding_cache._compute_hash(content)
        embedding_cache._meta[rel_path] = {"hash": file_hash}
        
        assert embedding_cache.needs_update(file_path, content) is False
    
    def test_needs_update_for_changed_file(self, embedding_cache, temp_vault):
        """Test that changed files need update."""
        file_path = temp_vault / "changed.md"
        file_path.write_text("Original content")
        
        # Add old hash to cache
        rel_path = str(file_path.relative_to(temp_vault))
        embedding_cache._meta[rel_path] = {"hash": "old_hash"}
        
        assert embedding_cache.needs_update(file_path) is True


class TestGetOrCreate:
    """Test get or create embedding."""
    
    def test_get_or_create_cached(self, embedding_cache, temp_vault):
        """Test getting cached embedding."""
        file_path = temp_vault / "cached.md"
        content = "Cached content"
        file_path.write_text(content)
        
        # Pre-populate cache
        rel_path = str(file_path.relative_to(temp_vault))
        file_hash = embedding_cache._compute_hash(content)
        embedding_cache._embeddings["cached_key"] = [0.1, 0.2, 0.3]
        embedding_cache._meta[rel_path] = {
            "hash": file_hash,
            "embedding_key": "cached_key"
        }
        
        result = embedding_cache.get_or_create(file_path, content)
        
        assert result == [0.1, 0.2, 0.3]
    
    @patch("src.embedding_cache.requests.post")
    def test_get_or_create_new(self, mock_post, embedding_cache, temp_vault):
        """Test creating new embedding."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": [0.5, 0.6, 0.7]}
        )
        
        file_path = temp_vault / "new.md"
        content = "New content for embedding"
        file_path.write_text(content)
        
        result = embedding_cache.get_or_create(file_path, content)
        
        assert result == [0.5, 0.6, 0.7]
        mock_post.assert_called_once()


class TestSaveAndLoad:
    """Test saving and loading cache."""
    
    def test_commit_saves_to_disk(self, embedding_cache, temp_vault):
        """Test that commit saves cache to disk."""
        embedding_cache._embeddings["test"] = [0.1, 0.2]
        embedding_cache._meta["test.md"] = {"hash": "abc"}
        embedding_cache._dirty = True
        
        embedding_cache.commit()
        
        embeddings_file = temp_vault / ".ana" / "embeddings.json"
        meta_file = temp_vault / ".ana" / "embeddings_meta.json"
        
        assert embeddings_file.exists()
        assert meta_file.exists()
    
    def test_load_handles_corrupted_file(self, temp_vault):
        """Test that corrupted cache files are handled gracefully."""
        cache_dir = temp_vault / ".ana"
        cache_dir.mkdir(parents=True)
        
        # Write invalid JSON
        (cache_dir / "embeddings.json").write_text("not valid json")
        
        # Should not raise, just use empty cache
        cache = EmbeddingCache(vault_path=temp_vault)
        assert cache._embeddings == {}


class TestGetStats:
    """Test cache statistics."""
    
    def test_get_stats_returns_info(self, embedding_cache):
        """Test that stats returns cache information."""
        embedding_cache._embeddings["key1"] = [0.1, 0.2, 0.3]
        embedding_cache._meta["file1.md"] = {"hash": "abc"}
        
        stats = embedding_cache.get_stats()
        
        assert "total_files" in stats
        assert "total_embeddings" in stats
        assert "embedding_dimension" in stats
        assert stats["total_files"] == 1
        assert stats["total_embeddings"] == 1
        assert stats["embedding_dimension"] == 3


class TestClearCache:
    """Test cache clearing."""
    
    def test_clear_cache_removes_all(self, embedding_cache, temp_vault):
        """Test that clear removes all cached data."""
        embedding_cache._embeddings["key"] = [0.1]
        embedding_cache._meta["file.md"] = {"hash": "abc"}
        
        embedding_cache.clear_cache()
        
        assert embedding_cache._embeddings == {}
        assert embedding_cache._meta == {}


class TestRetryLogic:
    """Test retry logic in embedding creation."""
    
    @patch("src.embedding_cache.requests.post")
    @patch("time.sleep")  # time is imported inside the method
    def test_retry_on_timeout(self, mock_sleep, mock_post, embedding_cache):
        """Test that timeout triggers retry."""
        import requests
        
        # First two calls timeout, third succeeds
        mock_post.side_effect = [
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout(),
            MagicMock(status_code=200, json=lambda: {"embedding": [0.1]})
        ]
        
        result = embedding_cache._create_embedding("test content")
        
        assert result == [0.1]
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
    
    @patch("src.embedding_cache.requests.post")
    @patch("time.sleep")  # time is imported inside the method
    def test_gives_up_after_max_retries(self, mock_sleep, mock_post, embedding_cache):
        """Test that it gives up after max retries."""
        import requests
        
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        result = embedding_cache._create_embedding("test content")
        
        assert result is None
        assert mock_post.call_count == 4  # Initial + 3 retries


# Fixtures
@pytest.fixture
def temp_vault(tmp_path):
    """Create temporary vault directory."""
    vault = tmp_path / "test_vault"
    vault.mkdir()
    return vault


@pytest.fixture
def embedding_cache(temp_vault):
    """Create EmbeddingCache instance."""
    return EmbeddingCache(vault_path=temp_vault)
