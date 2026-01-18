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
    
    Features:
    - Hash-based change detection for incremental updates
    - Batch embedding processing for efficiency
    - Deferred save to reduce I/O operations
    - Optional vector DB backend (Chroma)
    """
    
    def __init__(
        self,
        vault_path: Path,
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        batch_size: int = 10,
        use_vector_db: bool = False,
    ):
        """Initialize embedding cache.
        
        Args:
            vault_path: Path to Obsidian vault
            ollama_base_url: Ollama API base URL
            embedding_model: Model name for embeddings
            batch_size: Number of items per batch for batch processing
            use_vector_db: Enable vector DB backend (Chroma)
        """
        self.vault_path = Path(vault_path)
        self.cache_dir = self.vault_path / ".ana"
        self.embeddings_file = self.cache_dir / "embeddings.json"
        self.meta_file = self.cache_dir / "embeddings_meta.json"
        
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.use_vector_db = use_vector_db
        
        # In-memory cache
        self._embeddings: dict[str, list[float]] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        
        # Deferred save tracking
        self._dirty = False
        self._pending_count = 0
        
        # Vector DB backend (optional)
        self._vector_db = None
        if use_vector_db:
            self._init_vector_db()
        
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
            # Add to vector DB if enabled
            if self._vector_db is not None:
                self._add_to_vector_db(rel_path, embedding, content)
        
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
        
        # Commit all pending changes after sync
        self.commit()
        
        return stats
    
    def batch_create_embeddings(
        self,
        items: list[tuple[Path, str]],
        progress_callback=None
    ) -> dict[str, list[float]]:
        """Create embeddings in batches for efficiency.
        
        Args:
            items: List of (file_path, content) tuples
            progress_callback: Optional callback(current, total, file_name)
            
        Returns:
            Dict of path -> embedding
        """
        results = {}
        total = len(items)
        
        for i in range(0, total, self.batch_size):
            batch = items[i:i + self.batch_size]
            
            for j, (file_path, content) in enumerate(batch):
                current = i + j + 1
                if progress_callback:
                    progress_callback(current, total, file_path.name)
                
                file_hash = self._compute_hash(content)
                rel_path = self._get_relative_path(file_path)
                
                # Check if update needed
                if rel_path in self._meta and self._meta[rel_path].get("hash") == file_hash:
                    embed_key = self._meta[rel_path].get("embedding_key")
                    if embed_key and embed_key in self._embeddings:
                        results[rel_path] = self._embeddings[embed_key]
                        continue
                
                # Create new embedding
                embedding = self._create_embedding(content)
                if embedding:
                    self._save_embedding(rel_path, embedding, file_hash)
                    results[rel_path] = embedding
                    
                    # Add to vector DB if enabled
                    if self._vector_db is not None:
                        self._add_to_vector_db(rel_path, embedding, content)
            
            # Commit after each batch
            self.commit()
        
        return results
    
    def commit(self):
        """Persist pending changes to disk.
        
        Call this after batch operations to save changes.
        """
        if self._dirty:
            self._save_cache()
            self._dirty = False
            self._pending_count = 0
    
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
        embedding_dim = 0
        if self._embeddings:
            first_key = next(iter(self._embeddings))
            embedding_dim = len(self._embeddings[first_key])
        
        return {
            "total_files": len(self._meta),
            "total_embeddings": len(self._embeddings),
            "embedding_dimension": embedding_dim,
            "cache_size_bytes": self._get_cache_size(),
            "cache_size_human": self._format_size(self._get_cache_size()),
            "cache_dir": str(self.cache_dir),
            "embedding_model": self.embedding_model,
            "batch_size": self.batch_size,
            "vector_db_enabled": self.use_vector_db,
            "pending_changes": self._pending_count,
        }
    
    def _get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        size = 0
        if self.embeddings_file.exists():
            size += self.embeddings_file.stat().st_size
        if self.meta_file.exists():
            size += self.meta_file.stat().st_size
        return size
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
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
        """Save embedding to cache (deferred)."""
        # Generate unique key
        embed_key = f"embed_{hashlib.md5(rel_path.encode()).hexdigest()[:12]}"
        
        # Update in-memory cache
        self._embeddings[embed_key] = embedding
        self._meta[rel_path] = {
            "hash": file_hash,
            "modified": datetime.now().isoformat(),
            "embedding_key": embed_key,
        }
        
        # Mark as dirty (deferred save)
        self._dirty = True
        self._pending_count += 1
        
        # Auto-commit if pending count exceeds batch size
        if self._pending_count >= self.batch_size:
            self.commit()
    
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
        """Save cache to disk (JSON + optionally vector DB)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Always save to JSON (primary storage)
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
    
    # =========================================================================
    # Vector DB Methods (Optional Chroma Backend)
    # =========================================================================
    
    def _init_vector_db(self):
        """Initialize Chroma vector DB."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            db_path = self.cache_dir / "chroma_db"
            db_path.mkdir(parents=True, exist_ok=True)
            
            self._chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(db_path),
                anonymized_telemetry=False
            ))
            
            self._vector_db = self._chroma_client.get_or_create_collection(
                name="ana_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            
        except ImportError:
            # Chroma not installed, disable vector DB
            self._vector_db = None
            self.use_vector_db = False
        except Exception:
            self._vector_db = None
            self.use_vector_db = False
    
    def _add_to_vector_db(
        self,
        rel_path: str,
        embedding: list[float],
        content: str
    ):
        """Add embedding to vector DB."""
        if self._vector_db is None:
            return
        
        try:
            doc_id = hashlib.md5(rel_path.encode()).hexdigest()
            
            # Upsert to avoid duplicates
            self._vector_db.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content[:1000]],  # Truncate for metadata
                metadatas=[{"path": rel_path}]
            )
        except Exception:
            pass
    
    def search_similar_in_vector_db(
        self,
        query_embedding: list[float],
        n_results: int = 10
    ) -> list[tuple[str, float]]:
        """Search similar documents in vector DB.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            List of (path, distance) tuples
        """
        if self._vector_db is None:
            return []
        
        try:
            results = self._vector_db.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            output = []
            if results and results.get("metadatas"):
                for i, meta in enumerate(results["metadatas"][0]):
                    path = meta.get("path", "")
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    output.append((path, 1 - distance))  # Convert distance to similarity
            
            return output
        except Exception:
            return []
    
    def rebuild_vector_db(self, vault_scanner, progress_callback=None) -> int:
        """Rebuild vector DB from JSON cache.
        
        Args:
            vault_scanner: VaultScanner instance
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Number of embeddings added
        """
        if self._vector_db is None:
            if self.use_vector_db:
                self._init_vector_db()
            if self._vector_db is None:
                return 0
        
        count = 0
        total = len(self._meta)
        
        for i, (rel_path, meta) in enumerate(self._meta.items()):
            if progress_callback:
                progress_callback(i + 1, total, f"Rebuilding: {rel_path}")
            
            embed_key = meta.get("embedding_key")
            if embed_key and embed_key in self._embeddings:
                embedding = self._embeddings[embed_key]
                
                # Get content from vault
                file_path = self.vault_path / rel_path
                content = ""
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                    except Exception:
                        pass
                
                self._add_to_vector_db(rel_path, embedding, content)
                count += 1
        
        return count

