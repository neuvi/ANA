"""Smart Tag Manager Module.

AI-powered tag suggestion and normalization based on Vault's existing tags.
"""

import re
from collections import Counter
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.config import ANAConfig
    from src.vault_scanner import VaultScanner

logger = get_logger("smart_tags")


# =============================================================================
# Schemas
# =============================================================================

class TagSuggestion(BaseModel):
    """AI가 제안하는 태그."""
    
    tag: str = Field(..., description="태그 이름 (정규화된)")
    confidence: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="신뢰도 (0.0~1.0)"
    )
    source: Literal["vault", "ai", "normalized"] = Field(
        default="ai",
        description="태그 출처: vault(기존), ai(새 제안), normalized(정규화)"
    )
    original: str = Field(
        default="", 
        description="정규화 전 원본 태그 (있는 경우)"
    )
    usage_count: int = Field(
        default=0, 
        description="Vault 내 사용 횟수"
    )


class TagStatistics(BaseModel):
    """Vault 태그 통계."""
    
    total_tags: int = Field(default=0, description="총 태그 수")
    unique_tags: int = Field(default=0, description="고유 태그 수")
    top_tags: list[tuple[str, int]] = Field(
        default_factory=list,
        description="가장 많이 사용된 태그 목록 [(tag, count), ...]"
    )


# =============================================================================
# Smart Tag Manager
# =============================================================================

class SmartTagManager:
    """Vault 기반 스마트 태그 관리자.
    
    Features:
    - Vault 전체 태그 수집 및 빈도 분석
    - 태그 정규화 (소문자, 하이픈 통일)
    - AI 기반 태그 제안 (Vault 태그 우선)
    - 유사 태그 감지 (오타/변형)
    """
    
    def __init__(
        self,
        vault_scanner: "VaultScanner",
        config: "ANAConfig",
        llm=None,
    ):
        """Initialize smart tag manager.
        
        Args:
            vault_scanner: VaultScanner instance
            config: ANA configuration
            llm: Optional LLM for AI-powered suggestions
        """
        self.vault_scanner = vault_scanner
        self.config = config
        self.llm = llm
        
        # Cache for vault tags
        self._tag_cache: dict[str, int] | None = None
        self._normalized_cache: dict[str, str] | None = None
    
    # =========================================================================
    # Tag Collection
    # =========================================================================
    
    def get_all_tags(self, refresh: bool = False) -> dict[str, int]:
        """Vault 전체 태그 수집 및 사용 빈도 반환.
        
        Args:
            refresh: 캐시 무시하고 새로 스캔
            
        Returns:
            태그별 사용 횟수 딕셔너리 {tag: count}
        """
        if self._tag_cache is not None and not refresh:
            return self._tag_cache
        
        tag_counter: Counter = Counter()
        
        for note in self.vault_scanner.scan_all_notes():
            meta = note.get("metadata", {})
            tags = meta.get("tags", [])
            
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str):
                        # 정규화해서 카운트
                        normalized = self.normalize_tag(tag)
                        tag_counter[normalized] += 1
            elif isinstance(tags, str):
                # 단일 태그 문자열
                normalized = self.normalize_tag(tags)
                tag_counter[normalized] += 1
        
        self._tag_cache = dict(tag_counter)
        logger.debug(f"Collected {len(self._tag_cache)} unique tags from vault")
        
        return self._tag_cache
    
    def get_statistics(self) -> TagStatistics:
        """Vault 태그 통계 반환."""
        tags = self.get_all_tags()
        
        total_usage = sum(tags.values())
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        
        return TagStatistics(
            total_tags=total_usage,
            unique_tags=len(tags),
            top_tags=sorted_tags[:20]  # Top 20
        )
    
    # =========================================================================
    # Tag Normalization
    # =========================================================================
    
    @staticmethod
    def normalize_tag(tag: str) -> str:
        """단일 태그 정규화.
        
        Rules:
        - '#' 접두사 제거
        - 소문자 변환
        - 공백/언더스코어 → 하이픈
        - 연속 하이픈 제거
        - 앞뒤 하이픈 제거
        
        Args:
            tag: 원본 태그
            
        Returns:
            정규화된 태그
        """
        if not tag:
            return ""
        
        # Remove # prefix
        tag = tag.lstrip("#")
        
        # Lowercase
        tag = tag.lower()
        
        # Replace spaces and underscores with hyphens
        tag = re.sub(r"[\s_]+", "-", tag)
        
        # Remove consecutive hyphens
        tag = re.sub(r"-+", "-", tag)
        
        # Remove leading/trailing hyphens
        tag = tag.strip("-")
        
        return tag
    
    def normalize_tags(self, tags: list[str]) -> list[str]:
        """여러 태그 정규화 및 중복 제거.
        
        Args:
            tags: 원본 태그 리스트
            
        Returns:
            정규화되고 중복 제거된 태그 리스트
        """
        normalized = []
        seen = set()
        
        for tag in tags:
            norm = self.normalize_tag(tag)
            if norm and norm not in seen:
                normalized.append(norm)
                seen.add(norm)
        
        return normalized
    
    def get_normalization_map(self) -> dict[str, str]:
        """Vault 태그의 원본→정규화 매핑 반환.
        
        Returns:
            {original_tag: normalized_tag} 딕셔너리
        """
        if self._normalized_cache is not None:
            return self._normalized_cache
        
        mapping = {}
        
        for note in self.vault_scanner.scan_all_notes():
            meta = note.get("metadata", {})
            tags = meta.get("tags", [])
            
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str):
                        normalized = self.normalize_tag(tag)
                        if tag != normalized:
                            mapping[tag] = normalized
        
        self._normalized_cache = mapping
        return mapping
    
    # =========================================================================
    # Tag Suggestion
    # =========================================================================
    
    def suggest_tags(
        self,
        content: str,
        existing_tags: list[str] | None = None,
        max_tags: int = 5,
    ) -> list[TagSuggestion]:
        """콘텐츠 기반 태그 제안.
        
        우선순위:
        1. Vault에 이미 존재하는 태그 (일관성)
        2. 콘텐츠에서 추출된 키워드 기반
        3. AI 제안 (LLM 사용 가능한 경우)
        
        Args:
            content: 노트 콘텐츠
            existing_tags: 이미 있는 태그 (제외할 태그)
            max_tags: 최대 제안 수
            
        Returns:
            TagSuggestion 리스트
        """
        if existing_tags is None:
            existing_tags = []
        
        existing_normalized = set(self.normalize_tag(t) for t in existing_tags)
        vault_tags = self.get_all_tags()
        suggestions: list[TagSuggestion] = []
        
        # 1. 콘텐츠 키워드 기반으로 Vault 태그 매칭
        content_lower = content.lower()
        
        for tag, count in sorted(vault_tags.items(), key=lambda x: x[1], reverse=True):
            if tag in existing_normalized:
                continue
            
            # 태그가 콘텐츠에 포함되어 있는지 확인
            if self._is_tag_relevant(tag, content_lower):
                suggestions.append(TagSuggestion(
                    tag=tag,
                    confidence=min(0.9, 0.5 + (count / 100) * 0.4),
                    source="vault",
                    usage_count=count
                ))
        
        # 2. AI 기반 제안 (LLM이 있는 경우)
        if self.llm and len(suggestions) < max_tags:
            ai_suggestions = self._get_ai_suggestions(
                content, 
                existing_tags, 
                max_tags - len(suggestions)
            )
            suggestions.extend(ai_suggestions)
        
        # 상위 max_tags개 반환
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions[:max_tags]
    
    def _is_tag_relevant(self, tag: str, content_lower: str) -> bool:
        """태그가 콘텐츠와 관련있는지 확인."""
        # 태그를 구성하는 단어들
        tag_words = tag.replace("-", " ").split()
        
        # 모든 단어가 콘텐츠에 있으면 관련 있음
        for word in tag_words:
            if len(word) > 2 and word in content_lower:
                return True
        
        # 태그 자체가 콘텐츠에 있으면 관련 있음
        if tag in content_lower or tag.replace("-", "") in content_lower:
            return True
        
        return False
    
    def _get_ai_suggestions(
        self,
        content: str,
        existing_tags: list[str],
        max_tags: int,
    ) -> list[TagSuggestion]:
        """LLM을 사용한 AI 태그 제안."""
        if not self.llm:
            return []
        
        try:
            from src.prompts import TAG_SUGGESTION_PROMPT
            
            vault_tags = list(self.get_all_tags().keys())[:50]  # Top 50
            
            prompt = TAG_SUGGESTION_PROMPT.format(
                existing_vault_tags=", ".join(vault_tags) if vault_tags else "None",
                note_content=content[:2000],  # Limit content length
                max_tags=max_tags
            )
            
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            import json
            try:
                data = json.loads(response.content)
                suggested = data.get("suggested_tags", [])
                
                return [
                    TagSuggestion(
                        tag=self.normalize_tag(tag),
                        confidence=0.7,
                        source="ai"
                    )
                    for tag in suggested
                    if self.normalize_tag(tag) not in existing_tags
                ][:max_tags]
            except json.JSONDecodeError:
                logger.warning("Failed to parse AI tag suggestions")
                return []
                
        except Exception as e:
            logger.warning(f"AI tag suggestion failed: {e}")
            return []
    
    # =========================================================================
    # Similar Tags
    # =========================================================================
    
    def get_similar_tags(self, tag: str, threshold: float = 0.6) -> list[str]:
        """유사 태그 찾기 (오타/변형 감지).
        
        Args:
            tag: 검색할 태그
            threshold: 유사도 임계값 (0.0~1.0)
            
        Returns:
            유사한 태그 리스트
        """
        normalized = self.normalize_tag(tag)
        vault_tags = self.get_all_tags()
        similar = []
        
        for existing_tag in vault_tags:
            if existing_tag == normalized:
                continue
            
            similarity = self._tag_similarity(normalized, existing_tag)
            if similarity >= threshold:
                similar.append(existing_tag)
        
        return similar
    
    @staticmethod
    def _tag_similarity(a: str, b: str) -> float:
        """두 태그 간 유사도 계산 (Jaccard + Edit distance).
        
        Returns:
            유사도 (0.0~1.0)
        """
        if a == b:
            return 1.0
        
        # Character-level Jaccard similarity
        set_a = set(a)
        set_b = set(b)
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        
        # Word-level overlap
        words_a = set(a.split("-"))
        words_b = set(b.split("-"))
        
        word_intersection = len(words_a & words_b)
        word_union = len(words_a | words_b)
        
        if word_union > 0:
            word_jaccard = word_intersection / word_union
            return (jaccard + word_jaccard) / 2
        
        return jaccard
    
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def clear_cache(self):
        """캐시 초기화."""
        self._tag_cache = None
        self._normalized_cache = None
        logger.debug("Smart tag cache cleared")
