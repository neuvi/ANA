"""Category Classifier Module.

Classifies notes into categories using frontmatter metadata or AI inference.
"""

from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from src.vault_scanner import VaultScanner


CATEGORY_CLASSIFICATION_PROMPT = """You are a knowledge categorization expert.
Your task is to classify the given note into a category.

Existing categories in the knowledge base:
{existing_categories}

Note content:
---
{note_content}
---

Instructions:
1. If the note fits well into one of the existing categories, return that category name.
2. If no existing category fits, suggest a new category name.
3. Category names should be:
   - Lowercase with hyphens (e.g., "book-note", "project-idea")
   - Descriptive but concise (1-3 words)
   - Generic enough to apply to similar notes

Return ONLY the category name, nothing else."""


class CategoryClassifier:
    """Classifies notes into categories.
    
    Uses a two-step approach:
    1. Check frontmatter for explicit category/type
    2. If not found, use AI to classify based on existing categories
    """
    
    def __init__(self, vault_scanner: "VaultScanner", llm: BaseChatModel):
        """Initialize category classifier.
        
        Args:
            vault_scanner: VaultScanner instance for accessing existing categories
            llm: Language model for AI classification
        """
        self.vault_scanner = vault_scanner
        self.llm = llm
        self._existing_categories: set[str] | None = None
    
    @property
    def existing_categories(self) -> set[str]:
        """Get existing categories (cached)."""
        if self._existing_categories is None:
            self._existing_categories = self.vault_scanner.get_existing_categories()
        return self._existing_categories
    
    def refresh_categories(self):
        """Refresh the cached categories list."""
        self._existing_categories = self.vault_scanner.get_existing_categories(refresh=True)
    
    def classify(
        self,
        raw_note: str,
        frontmatter: dict[str, Any] | None = None
    ) -> str:
        """Classify a note into a category.
        
        Classification priority:
        1. Explicit 'type' field in frontmatter
        2. Explicit 'category' field in frontmatter
        3. AI classification based on content
        
        Args:
            raw_note: Raw note content
            frontmatter: Existing frontmatter metadata
            
        Returns:
            Category name
        """
        # Step 1: Check frontmatter for explicit category
        if frontmatter:
            # Check 'type' field first
            if "type" in frontmatter and isinstance(frontmatter["type"], str):
                category = frontmatter["type"].strip().lower()
                if category:
                    return category
            
            # Check 'category' field
            if "category" in frontmatter and isinstance(frontmatter["category"], str):
                category = frontmatter["category"].strip().lower()
                if category:
                    return category
        
        # Step 2: AI classification
        return self._ai_classify(raw_note)
    
    def _ai_classify(self, raw_note: str) -> str:
        """Use AI to classify the note.
        
        Args:
            raw_note: Raw note content
            
        Returns:
            Category name
        """
        categories_str = ", ".join(sorted(self.existing_categories)) if self.existing_categories else "None yet"
        
        prompt = CATEGORY_CLASSIFICATION_PROMPT.format(
            existing_categories=categories_str,
            note_content=raw_note[:2000]  # Limit content length
        )
        
        messages = [
            SystemMessage(content="You are a precise categorization assistant."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        category = response.content.strip().lower()
        
        # Clean up the category name
        category = category.replace(" ", "-")
        category = "".join(c for c in category if c.isalnum() or c == "-")
        
        # Limit length
        if len(category) > 30:
            category = category[:30]
        
        return category or "general"
    
    def is_new_category(self, category: str) -> bool:
        """Check if a category is new (not in existing categories).
        
        Args:
            category: Category name to check
            
        Returns:
            True if the category is new
        """
        return category.lower() not in {c.lower() for c in self.existing_categories}
    
    def suggest_category(
        self,
        raw_note: str,
        frontmatter: dict[str, Any] | None = None
    ) -> tuple[str, bool]:
        """Suggest a category for the note.
        
        Args:
            raw_note: Raw note content
            frontmatter: Existing frontmatter metadata
            
        Returns:
            Tuple of (category_name, is_new)
        """
        category = self.classify(raw_note, frontmatter)
        is_new = self.is_new_category(category)
        return category, is_new
