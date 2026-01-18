"""Vault Scanner Module.

Scans Obsidian vault to extract frontmatter metadata from existing notes.
Used for category detection and template learning.
"""

import re
from pathlib import Path
from typing import Any

import yaml


class VaultScanner:
    """Scans Obsidian vault for note metadata.
    
    Extracts frontmatter from all markdown files in the vault,
    enabling category detection and pattern learning.
    """
    
    def __init__(self, vault_path: Path):
        """Initialize vault scanner.
        
        Args:
            vault_path: Path to Obsidian vault root
        """
        self.vault_path = Path(vault_path).expanduser().resolve()
        self._cache: dict[str, dict[str, Any]] = {}
        self._categories_cache: set[str] | None = None
    
    def scan_all_notes(self, use_cache: bool = True) -> list[dict[str, Any]]:
        """Scan all notes in the vault.
        
        Args:
            use_cache: Whether to use cached results
            
        Returns:
            List of dicts with 'path' and 'metadata' keys
        """
        if use_cache and self._cache:
            return [
                {"path": Path(p), "metadata": m}
                for p, m in self._cache.items()
            ]
        
        notes = []
        
        if not self.vault_path.exists():
            return notes
        
        for md_file in self.vault_path.rglob("*.md"):
            # Skip hidden files and directories
            if any(part.startswith(".") for part in md_file.parts):
                continue
            
            metadata = self.extract_frontmatter(md_file)
            if metadata is not None:
                self._cache[str(md_file)] = metadata
                notes.append({
                    "path": md_file,
                    "metadata": metadata
                })
        
        return notes
    
    def extract_frontmatter(self, file_path: Path) -> dict[str, Any] | None:
        """Extract frontmatter from a single file.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            Frontmatter as dict, or None if no frontmatter
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except (IOError, UnicodeDecodeError):
            return None
        
        return self.parse_frontmatter(content)
    
    @staticmethod
    def parse_frontmatter(content: str) -> dict[str, Any] | None:
        """Parse frontmatter from markdown content.
        
        Args:
            content: Markdown content string
            
        Returns:
            Frontmatter as dict, or None if no frontmatter
        """
        content = content.strip()
        
        if not content.startswith("---"):
            return None
        
        # Find the closing ---
        end_match = re.search(r"\n---\s*\n", content[3:])
        if not end_match:
            return None
        
        yaml_str = content[3:end_match.start() + 3]
        
        try:
            metadata = yaml.safe_load(yaml_str)
            if isinstance(metadata, dict):
                return metadata
            return None
        except yaml.YAMLError:
            return None
    
    def get_existing_categories(self, refresh: bool = False) -> set[str]:
        """Get all existing categories from vault notes.
        
        Looks for 'type', 'category', and common category-related tags.
        
        Args:
            refresh: Force refresh of cached categories
            
        Returns:
            Set of category names
        """
        if self._categories_cache is not None and not refresh:
            return self._categories_cache
        
        categories: set[str] = set()
        
        for note in self.scan_all_notes():
            meta = note["metadata"]
            
            # Check 'type' field
            if "type" in meta and isinstance(meta["type"], str):
                categories.add(meta["type"])
            
            # Check 'category' field
            if "category" in meta and isinstance(meta["category"], str):
                categories.add(meta["category"])
            
            # Check tags for category-like patterns
            tags = meta.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str):
                        # Remove # prefix if present
                        tag = tag.lstrip("#")
                        # Common category patterns
                        if tag in ["concept", "project", "book-note", "meeting", 
                                   "idea", "reference", "how-to", "tutorial"]:
                            categories.add(tag)
        
        self._categories_cache = categories
        return categories
    
    def find_similar_notes(self, category: str, limit: int = 5) -> list[Path]:
        """Find notes with the same category.
        
        Args:
            category: Category to search for
            limit: Maximum number of notes to return
            
        Returns:
            List of file paths
        """
        similar: list[Path] = []
        
        for note in self.scan_all_notes():
            if len(similar) >= limit:
                break
            
            meta = note["metadata"]
            
            if meta.get("type") == category or meta.get("category") == category:
                similar.append(note["path"])
        
        return similar
    
    def get_note_content(self, file_path: Path) -> str | None:
        """Get full content of a note.
        
        Args:
            file_path: Path to the note
            
        Returns:
            Note content as string, or None if not readable
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except (IOError, UnicodeDecodeError):
            return None
    
    def get_note_body(self, file_path: Path) -> str | None:
        """Get note content without frontmatter.
        
        Args:
            file_path: Path to the note
            
        Returns:
            Note body as string, or None if not readable
        """
        content = self.get_note_content(file_path)
        if content is None:
            return None
        
        content = content.strip()
        
        if content.startswith("---"):
            end_match = re.search(r"\n---\s*\n", content[3:])
            if end_match:
                return content[end_match.end() + 3:].strip()
        
        return content
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._categories_cache = None
