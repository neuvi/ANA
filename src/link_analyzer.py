"""Link Analyzer Module.

2-Stage Retrieval + Rerank for finding related notes.
Stage 1: Tag/Category + Keyword + Embedding similarity (RRF merge)
Stage 2: Cross-Encoder Rerank
"""

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.logging_config import get_logger

logger = get_logger("link_analyzer")

if TYPE_CHECKING:
    from src.embedding_cache import EmbeddingCache
    from src.vault_scanner import VaultScanner


class LinkAnalyzer:
    """Note link analyzer using 2-Stage Retrieval + Rerank.
    
    Finds related notes and generates wikilinks for Frontmatter.
    """
    
    def __init__(
        self,
        vault_scanner: "VaultScanner",
        embedding_cache: "EmbeddingCache",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """Initialize link analyzer.
        
        Args:
            vault_scanner: VaultScanner instance
            embedding_cache: EmbeddingCache instance
            rerank_model: Cross-encoder model name
        """
        self.vault_scanner = vault_scanner
        self.embedding_cache = embedding_cache
        self.rerank_model = rerank_model
        
        # Lazy loading for reranker
        self._reranker = None
    
    def find_related_notes(
        self,
        note_title: str,
        note_content: str,
        note_tags: list[str],
        note_category: str,
        max_links: int = 5,
        exclude_self: bool = True
    ) -> list[str]:
        """Find related notes and return as wikilinks.
        
        Args:
            note_title: Title of the note
            note_content: Content of the note
            note_tags: Tags of the note
            note_category: Category of the note
            max_links: Maximum number of related links
            exclude_self: Whether to exclude the note itself
            
        Returns:
            List of wikilinks like ["[[Note A]]", "[[Note B (new)]]"]
            Notes not in vault are marked with (new)
        """
        query = f"{note_title}\n{note_content[:1500]}"
        
        # Stage 1: Retrieval (parallel methods)
        tag_results = self._find_by_tags_category(note_tags, note_category)
        keyword_results = self._find_by_keywords(note_title, note_content)
        embedding_results = self._find_by_embeddings(query)
        
        # RRF merge
        candidates = self._rrf_merge([
            (tag_results, 0.25),
            (keyword_results, 0.30),
            (embedding_results, 0.45)
        ], top_k=30)
        
        # Exclude self
        if exclude_self:
            candidates = [c for c in candidates if c != note_title]
        
        if not candidates:
            return []
        
        # Stage 2: Rerank
        reranked = self._rerank(query, candidates)
        
        # Top N wikilinks with existence check
        top_links = reranked[:max_links]
        existing_titles = self._get_all_note_titles()
        
        result = []
        for title, _ in top_links:
            if title in existing_titles:
                result.append(f"[[{title}]]")
            else:
                result.append(f"[[{title} (new)]]")
        
        return result
    
    def _get_all_note_titles(self) -> set[str]:
        """Get all existing note titles in vault."""
        titles = set()
        for note in self.vault_scanner.scan_all_notes():
            titles.add(self._get_title(note))
        return titles
    
    # =========================================================================
    # Stage 1: Retrieval Methods
    # =========================================================================
    
    def _find_by_tags_category(
        self,
        tags: list[str],
        category: str
    ) -> list[tuple[str, float]]:
        """Find notes by tag and category matching."""
        results = []
        
        for note in self.vault_scanner.scan_all_notes():
            title = self._get_title(note)
            meta = note.get("metadata", {})
            score = 0.0
            
            # Tag matching
            note_tags = meta.get("tags", [])
            if isinstance(note_tags, list) and tags:
                shared = set(tags) & set(note_tags)
                if shared:
                    score += len(shared) / len(tags) * 0.7
            
            # Category matching
            note_cat = meta.get("type") or meta.get("category")
            if note_cat and note_cat == category:
                score += 0.5
            
            if score > 0:
                results.append((title, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _find_by_keywords(
        self,
        title: str,
        content: str
    ) -> list[tuple[str, float]]:
        """Find notes by keyword similarity (BM25-style)."""
        # Extract query keywords
        query_text = f"{title} {content[:500]}"
        query_words = self._extract_keywords(query_text)
        
        if not query_words:
            return []
        
        results = []
        for note in self.vault_scanner.scan_all_notes():
            note_title = self._get_title(note)
            note_content = self.vault_scanner.get_note_content(note["path"]) or ""
            note_text = f"{note_title} {note_content[:500]}"
            note_words = self._extract_keywords(note_text)
            
            overlap = query_words & note_words
            if overlap:
                score = len(overlap) / len(query_words)
                results.append((note_title, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _find_by_embeddings(self, query: str) -> list[tuple[str, float]]:
        """Find notes by embedding similarity."""
        # Get query embedding
        query_vec = self.embedding_cache._create_embedding(query)
        if query_vec is None:
            return []
        
        results = []
        all_embeddings = self.embedding_cache.get_all_embeddings()
        
        for rel_path, doc_vec in all_embeddings.items():
            # Get title from path
            path = self.embedding_cache.vault_path / rel_path
            note_info = self._find_note_by_path(path)
            if note_info:
                title = self._get_title(note_info)
                sim = self._cosine_similarity(query_vec, doc_vec)
                results.append((title, sim))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    # =========================================================================
    # RRF Merge
    # =========================================================================
    
    def _rrf_merge(
        self,
        ranked_lists: list[tuple[list[tuple[str, float]], float]],
        top_k: int = 30,
        k: int = 60
    ) -> list[str]:
        """Reciprocal Rank Fusion to merge multiple rankings."""
        scores: dict[str, float] = {}
        
        for ranked_list, weight in ranked_lists:
            for rank, (title, _) in enumerate(ranked_list):
                if title not in scores:
                    scores[title] = 0.0
                scores[title] += weight / (k + rank + 1)
        
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [title for title, _ in sorted_items[:top_k]]
    
    # =========================================================================
    # Stage 2: Rerank
    # =========================================================================
    
    def _rerank(
        self,
        query: str,
        candidates: list[str]
    ) -> list[tuple[str, float]]:
        """Rerank candidates using Cross-Encoder."""
        if not candidates:
            return []
        
        reranker = self._get_reranker()
        if reranker is None:
            # Fallback: keep original order
            return [(c, 1.0 - i / len(candidates)) for i, c in enumerate(candidates)]
        
        # Build query-document pairs
        pairs = []
        valid_candidates = []
        
        for title in candidates:
            content = self._get_content_by_title(title)
            if content:
                pairs.append((query, content[:500]))
                valid_candidates.append(title)
        
        if not pairs:
            return []
        
        # Predict scores
        try:
            scores = reranker.predict(pairs)
            results = list(zip(valid_candidates, scores))
            return sorted(results, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.warning(f"Reranking failed, using fallback order: {e}")
            return [(c, 1.0) for c in valid_candidates]
    
    def _get_reranker(self):
        """Lazy load Cross-Encoder reranker.
        
        Checks for local model in data/models/rerank/ first,
        then falls back to downloading from HuggingFace.
        """
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                from pathlib import Path
                
                # Check for local model first
                model_folder = self.rerank_model.replace("/", "_")
                local_path = Path("data/models/rerank") / model_folder
                
                if local_path.exists():
                    logger.debug(f"Loading reranker from local path: {local_path}")
                    self._reranker = CrossEncoder(str(local_path))
                else:
                    # Fall back to HuggingFace (will download to cache)
                    logger.debug(f"Loading reranker from HuggingFace: {self.rerank_model}")
                    self._reranker = CrossEncoder(self.rerank_model)
            except ImportError:
                logger.warning("sentence-transformers not installed, reranking disabled")
                return None
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                return None
        return self._reranker
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _extract_keywords(self, text: str) -> set[str]:
        """Extract keywords from text."""
        words = set(re.findall(r'[가-힣]+|[a-zA-Z]+', text.lower()))
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            'this', 'that', 'these', 'those', 'it', 'its',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where',
            'what', 'which', 'who', 'whom', 'how', 'why',
            'for', 'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about',
            '은', '는', '이', '가', '를', '을', '에', '의', '로', '으로',
            '와', '과', '도', '만', '까지', '부터', '에서', '처럼', '같이',
        }
        return {w for w in words if w not in stopwords and len(w) > 1}
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def _get_title(self, note: dict) -> str:
        """Get title from note metadata."""
        meta = note.get("metadata", {})
        return meta.get("title") or note["path"].stem
    
    def _find_note_by_path(self, path: Path) -> dict | None:
        """Find note info by path."""
        for note in self.vault_scanner.scan_all_notes():
            if note["path"] == path:
                return note
        return None
    
    def _get_content_by_title(self, title: str) -> str | None:
        """Get note content by title."""
        for note in self.vault_scanner.scan_all_notes():
            if self._get_title(note) == title:
                return self.vault_scanner.get_note_content(note["path"])
        return None
