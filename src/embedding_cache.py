"""Embedding Cache Module.

Manages embedding vectors with persistent storage in Obsidian Vault.
Supports incremental updates based on file content hash.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


class EmbeddingCache:
    """Embedding cache with vault-based storage and incremental updates.
    
    Stores embeddings in vault/.ana/ directory:
    - embeddings.json: Actual embedding vectors
    - embeddings_meta.json: File hashes for change detection
    """
    
    def __init__(
        self,
        vault_path: Path,
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
    ):
        """Initialize embedding cache.
        
        Args:
            vault_path: Path to Obsidian vault
            ollama_base_url: Ollama API base URL
            embedding_model: Model name for embeddings
        """
        self.vault_path = Path(vault_path)
        self.cache_dir = self.vault_path / ".ana"
        self.embeddings_file = self.cache_dir / "embeddings.json"
        self.meta_file = self.cache_dir / "embeddings_meta.json"
        
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        
        # In-memory cache
        self._embeddings: dict[str, list[float]] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        
        # Load existing cache
        self._load_cache()
    
    def get_or_create(
        self,
        file_path: Path,
        content: str | None = None
    ) -> list[float] | None:
        """Get embedding from cache or create new one.
        
        Args:
            file_path: Path to the note file
            content: Optional content (will read from file if not provided)
            
        Returns:
            Embedding vector or None if failed
        """
        if content is None:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                return None
        
        # Compute content hash
        file_hash = self._compute_hash(content)
        rel_path = self._get_relative_path(file_path)
        
        # Check cache
        if rel_path in self._meta:
            cached = self._meta[rel_path]
            if cached.get("hash") == file_hash:
                # Hash matches, use cached embedding
                embed_key = cached.get("embedding_key")
                if embed_key and embed_key in self._embeddings:
                    return self._embeddings[embed_key]
        
        # Need to create new embedding
        embedding = self._create_embedding(content)
        if embedding:
            self._save_embedding(rel_path, embedding, file_hash)
        
        return embedding
    
    def get_embedding(self, file_path: Path) -> list[float] | None:
        """Get cached embedding without creating new one.
        
        Args:
            file_path: Path to the note file
            
        Returns:
            Cached embedding or None
        """
        rel_path = self._get_relative_path(file_path)
        
        if rel_path in self._meta:
            embed_key = self._meta[rel_path].get("embedding_key")
            if embed_key and embed_key in self._embeddings:
                return self._embeddings[embed_key]
        
        return None
    
    def needs_update(self, file_path: Path, content: str | None = None) -> bool:
        """Check if file needs embedding update.
        
        Args:
            file_path: Path to the note file
            content: Optional content
            
        Returns:
            True if file is new or has changed
        """
        if content is None:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                return True
        
        file_hash = self._compute_hash(content)
        rel_path = self._get_relative_path(file_path)
        
        if rel_path not in self._meta:
            return True
        
        return self._meta[rel_path].get("hash") != file_hash
    
    def sync_vault(
        self,
        vault_scanner,
        progress_callback=None
    ) -> dict[str, int]:
        """Sync embeddings for entire vault.
        
        Only updates embeddings for changed files.
        
        Args:
            vault_scanner: VaultScanner instance
            progress_callback: Optional callback(current, total, file_name)
            
        Returns:
            Stats dict with 'updated', 'cached', 'failed' counts
        """
        stats = {"updated": 0, "cached": 0, "failed": 0}
        
        notes = vault_scanner.scan_all_notes()
        total = len(notes)
        
        for i, note in enumerate(notes):
            file_path = note["path"]
            
            if progress_callback:
                progress_callback(i + 1, total, file_path.name)
            
            try:
                content = file_path.read_text(encoding="utf-8")
                
                if self.needs_update(file_path, content):
                    embedding = self.get_or_create(file_path, content)
                    if embedding:
                        stats["updated"] += 1
                    else:
                        stats["failed"] += 1
                else:
                    stats["cached"] += 1
            except Exception:
                stats["failed"] += 1
        
        return stats
    
    def get_all_embeddings(self) -> dict[str, list[float]]:
        """Get all cached embeddings.
        
        Returns:
            Dict mapping relative paths to embeddings
        """
        result = {}
        for rel_path, meta in self._meta.items():
            embed_key = meta.get("embedding_key")
            if embed_key and embed_key in self._embeddings:
                result[rel_path] = self._embeddings[embed_key]
        return result
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        self._embeddings = {}
        self._meta = {}
        self._save_cache()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        return {
            "total_files": len(self._meta),
            "total_embeddings": len(self._embeddings),
            "cache_dir": str(self.cache_dir),
            "embedding_model": self.embedding_model,
        }
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _compute_hash(self, content: str) -> str:
        """Compute MD5 hash of content."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()
    
    def _get_relative_path(self, file_path: Path) -> str:
        """Get path relative to vault."""
        try:
            return str(file_path.relative_to(self.vault_path))
        except ValueError:
            return str(file_path)
    
    def _create_embedding(self, content: str) -> list[float] | None:
        """Create embedding using Ollama API."""
        try:
            # Truncate content if too long
            text = content[:8000]  # ~2000 tokens
            
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                },
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("embedding")
        except Exception:
            pass
        
        return None
    
    def _save_embedding(
        self,
        rel_path: str,
        embedding: list[float],
        file_hash: str
    ):
        """Save embedding to cache."""
        # Generate unique key
        embed_key = f"embed_{hashlib.md5(rel_path.encode()).hexdigest()[:12]}"
        
        # Update in-memory cache
        self._embeddings[embed_key] = embedding
        self._meta[rel_path] = {
            "hash": file_hash,
            "modified": datetime.now().isoformat(),
            "embedding_key": embed_key,
        }
        
        # Persist to disk
        self._save_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load embeddings
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, "r", encoding="utf-8") as f:
                    self._embeddings = json.load(f)
            except Exception:
                self._embeddings = {}
        
        # Load metadata
        if self.meta_file.exists():
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
            except Exception:
                self._meta = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        try:
            with open(self.embeddings_file, "w", encoding="utf-8") as f:
                json.dump(self._embeddings, f)
        except Exception:
            pass
        
        # Save metadata
        try:
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(self._meta, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
