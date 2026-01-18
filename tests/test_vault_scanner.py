"""Tests for VaultScanner module."""

from pathlib import Path

import pytest

from src.vault_scanner import VaultScanner


class TestVaultScanner:
    """Test suite for VaultScanner class."""
    
    def test_init(self, temp_vault: Path):
        """Test VaultScanner initialization."""
        scanner = VaultScanner(temp_vault)
        assert scanner.vault_path == temp_vault
    
    def test_scan_all_notes(self, temp_vault: Path):
        """Test scanning all notes in vault."""
        scanner = VaultScanner(temp_vault)
        notes = scanner.scan_all_notes()
        
        assert len(notes) == 3
        
        # Check note structure
        for note in notes:
            assert "path" in note
            assert "metadata" in note
            assert isinstance(note["path"], Path)
    
    def test_scan_all_notes_empty_vault(self, temp_vault_empty: Path):
        """Test scanning empty vault."""
        scanner = VaultScanner(temp_vault_empty)
        notes = scanner.scan_all_notes()
        
        assert len(notes) == 0
    
    def test_extract_frontmatter(self, temp_vault: Path):
        """Test extracting frontmatter from a file."""
        scanner = VaultScanner(temp_vault)
        note_path = temp_vault / "notes" / "RAG 개념.md"
        
        metadata = scanner.extract_frontmatter(note_path)
        
        assert metadata is not None
        assert metadata["title"] == "RAG 개념"
        assert "AI" in metadata["tags"]
        assert "RAG" in metadata["tags"]
        assert metadata["type"] == "concept"
    
    def test_extract_frontmatter_no_frontmatter(self, temp_vault: Path):
        """Test extracting frontmatter from file without frontmatter."""
        scanner = VaultScanner(temp_vault)
        
        # Create a note without frontmatter
        note_path = temp_vault / "no_frontmatter.md"
        note_path.write_text("Just plain content without frontmatter.", encoding="utf-8")
        
        metadata = scanner.extract_frontmatter(note_path)
        
        assert metadata is None
    
    def test_parse_frontmatter_content(self, sample_raw_note: str):
        """Test parsing frontmatter from content string."""
        scanner = VaultScanner(Path("."))  # Path doesn't matter for this test
        
        metadata = scanner.parse_frontmatter(sample_raw_note)
        
        assert metadata is not None
        assert metadata["title"] == "RAG 개념 정리"
        assert "AI" in metadata["tags"]
    
    def test_parse_frontmatter_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        scanner = VaultScanner(Path("."))
        content = "Just plain content without frontmatter."
        
        metadata = scanner.parse_frontmatter(content)
        
        assert metadata is None
    
    def test_get_existing_categories(self, temp_vault: Path):
        """Test getting existing categories from vault."""
        scanner = VaultScanner(temp_vault)
        categories = scanner.get_existing_categories()
        
        assert "concept" in categories
        assert "tool" in categories
    
    def test_find_similar_notes(self, temp_vault: Path):
        """Test finding notes with similar category."""
        scanner = VaultScanner(temp_vault)
        similar = scanner.find_similar_notes("concept")
        
        # Should find RAG and LLM notes (both are concept type)
        assert len(similar) >= 2
    
    def test_find_similar_notes_no_match(self, temp_vault: Path):
        """Test finding notes with non-existent category."""
        scanner = VaultScanner(temp_vault)
        similar = scanner.find_similar_notes("nonexistent")
        
        assert len(similar) == 0
    
    def test_get_note_body(self, temp_vault: Path):
        """Test getting note body without frontmatter."""
        scanner = VaultScanner(temp_vault)
        note_path = temp_vault / "notes" / "RAG 개념.md"
        
        body = scanner.get_note_body(note_path)
        
        assert body is not None
        assert "RAG는 Retrieval-Augmented Generation입니다" in body
        assert "---" not in body  # Frontmatter should be removed
    
    def test_get_note_content(self, temp_vault: Path):
        """Test getting full note content."""
        scanner = VaultScanner(temp_vault)
        note_path = temp_vault / "notes" / "RAG 개념.md"
        
        content = scanner.get_note_content(note_path)
        
        assert content is not None
        assert "---" in content  # Should include frontmatter
        assert "RAG는 Retrieval-Augmented Generation입니다" in content


class TestVaultScannerEdgeCases:
    """Edge case tests for VaultScanner."""
    
    def test_invalid_yaml_frontmatter(self, temp_vault: Path):
        """Test handling invalid YAML in frontmatter."""
        scanner = VaultScanner(temp_vault)
        
        # Create a note with invalid YAML
        note_path = temp_vault / "invalid_yaml.md"
        note_path.write_text("""---
title: [invalid yaml
tags: unclosed bracket
---

Content here.
""", encoding="utf-8")
        
        metadata = scanner.extract_frontmatter(note_path)
        
        # Should return None or empty dict for invalid YAML
        assert metadata is None or metadata == {}
    
    def test_nested_directories(self, temp_vault: Path):
        """Test scanning notes in nested directories."""
        scanner = VaultScanner(temp_vault)
        
        # Create nested directory with note
        nested_dir = temp_vault / "notes" / "deep" / "nested"
        nested_dir.mkdir(parents=True)
        
        (nested_dir / "Nested Note.md").write_text("""---
title: Nested Note
tags: [test]
type: test
---

This is a nested note.
""", encoding="utf-8")
        
        notes = scanner.scan_all_notes()
        
        # Should find the nested note too
        nested_found = any("Nested Note" in str(note["path"]) for note in notes)
        assert nested_found
    
    def test_unicode_content(self, temp_vault: Path):
        """Test handling Unicode content."""
        scanner = VaultScanner(temp_vault)
        
        # Create a note with various Unicode characters
        note_path = temp_vault / "unicode_test.md"
        note_path.write_text("""---
title: 유니코드 테스트 🚀
tags: [한글, 日本語, émoji]
---

다국어 콘텐츠: 한글, 日本語, العربية, 🎉
""", encoding="utf-8")
        
        metadata = scanner.extract_frontmatter(note_path)
        
        assert metadata is not None
        assert "유니코드 테스트" in metadata["title"]
        assert "한글" in metadata["tags"]
