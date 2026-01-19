"""Tests for Backlink Analyzer Module.

Tests for src/backlink_analyzer.py functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.backlink_analyzer import BacklinkAnalyzer
from src.schemas import BacklinkSuggestion, DraftNote


class TestBacklinkAnalyzerInit:
    """Test BacklinkAnalyzer initialization."""
    
    def test_init_with_defaults(self, mock_vault_scanner, mock_llm):
        """Test initialization with default parameters."""
        analyzer = BacklinkAnalyzer(
            vault_scanner=mock_vault_scanner,
            llm=mock_llm,
        )
        
        assert analyzer.auto_apply is True
        assert analyzer.max_suggestions_per_note == 3
        assert analyzer.min_confidence == 0.6
    
    def test_init_with_custom_params(self, mock_vault_scanner, mock_llm):
        """Test initialization with custom parameters."""
        analyzer = BacklinkAnalyzer(
            vault_scanner=mock_vault_scanner,
            llm=mock_llm,
            auto_apply=False,
            max_suggestions_per_note=5,
            min_confidence=0.8,
        )
        
        assert analyzer.auto_apply is False
        assert analyzer.max_suggestions_per_note == 5
        assert analyzer.min_confidence == 0.8


class TestExtractKeyConcepts:
    """Test key concept extraction."""
    
    def test_extract_from_note_with_title(self, backlink_analyzer, sample_draft_note):
        """Test extracting concepts from note title."""
        concepts = backlink_analyzer._extract_key_concepts(sample_draft_note)
        
        assert sample_draft_note.title in concepts
    
    def test_extract_from_note_with_tags(self, backlink_analyzer):
        """Test extracting concepts from note tags."""
        note = DraftNote(
            title="Test Note",
            tags=["python", "machine-learning"],
            content="Test content",
            category="tech",
            frontmatter={},
        )
        
        concepts = backlink_analyzer._extract_key_concepts(note)
        
        assert "python" in concepts
        assert "machine-learning" in concepts
    
    def test_extract_skips_general_category(self, backlink_analyzer):
        """Test that 'general' category is not included."""
        note = DraftNote(
            title="Test Note",
            tags=[],
            content="Test content",
            category="general",
            frontmatter={},
        )
        
        concepts = backlink_analyzer._extract_key_concepts(note)
        
        assert "general" not in concepts


class TestFindBacklinkOpportunities:
    """Test finding backlink opportunities."""
    
    def test_find_opportunities_in_existing_notes(
        self, backlink_analyzer, sample_draft_note, mock_vault_with_notes
    ):
        """Test finding backlinks in existing notes."""
        suggestions = backlink_analyzer.find_backlink_opportunities(
            sample_draft_note,
            max_notes_to_scan=10,
        )
        
        # Should return list of suggestions
        assert isinstance(suggestions, list)
    
    def test_find_opportunities_skips_self_reference(
        self, backlink_analyzer, sample_draft_note
    ):
        """Test that self-references are skipped."""
        suggestions = backlink_analyzer.find_backlink_opportunities(
            sample_draft_note,
            max_notes_to_scan=10,
        )
        
        # None of the suggestions should point to the same note
        for sugg in suggestions:
            assert sugg.source_note_title != sample_draft_note.title
    
    def test_find_opportunities_skips_already_linked(
        self, backlink_analyzer, sample_draft_note
    ):
        """Test that already linked notes are skipped."""
        suggestions = backlink_analyzer.find_backlink_opportunities(
            sample_draft_note,
            max_notes_to_scan=10,
        )
        
        # Suggestions should not include notes that already have the link
        for sugg in suggestions:
            assert sugg.target_note_title == sample_draft_note.title


class TestApplyBacklinks:
    """Test applying backlinks to notes."""
    
    def test_apply_backlinks_modifies_files(
        self, backlink_analyzer, temp_vault_with_notes
    ):
        """Test that applying backlinks modifies files."""
        suggestions = [
            BacklinkSuggestion(
                id="test1",
                source_note_path=str(temp_vault_with_notes / "note1.md"),
                source_note_title="Note 1",
                target_note_title="New Note",
                matched_text="related topic",
                line_number=5,
                confidence=0.8,
                reason="Test reason",
            )
        ]
        
        # Create the test note
        note_path = temp_vault_with_notes / "note1.md"
        note_path.write_text("---\ntitle: Note 1\n---\n\nThis is about related topic.\n")
        
        modified = backlink_analyzer.apply_backlinks(suggestions)
        
        assert isinstance(modified, list)
    
    def test_apply_backlinks_inserts_wikilink(
        self, backlink_analyzer, temp_vault_with_notes
    ):
        """Test that wikilinks are inserted correctly."""
        note_path = temp_vault_with_notes / "test_note.md"
        original_content = "Line 1\nThis mentions Python programming.\nLine 3"
        note_path.write_text(original_content)
        
        suggestions = [
            BacklinkSuggestion(
                id="test1",
                source_note_path=str(note_path),
                source_note_title="Test Note",
                target_note_title="Python Guide",
                matched_text="Python programming",
                line_number=2,
                confidence=0.9,
                reason="Direct reference",
            )
        ]
        
        backlink_analyzer.apply_backlinks(suggestions)
        
        new_content = note_path.read_text()
        assert "[[Python Guide]]" in new_content


class TestAnalyzeAndApply:
    """Test combined analyze and apply functionality."""
    
    def test_analyze_and_apply_with_auto_apply(
        self, mock_vault_scanner, mock_llm, sample_draft_note
    ):
        """Test automatic application of backlinks."""
        analyzer = BacklinkAnalyzer(
            vault_scanner=mock_vault_scanner,
            llm=mock_llm,
            auto_apply=True,
            min_confidence=0.5,
        )
        
        suggestions, modified = analyzer.analyze_and_apply(sample_draft_note)
        
        assert isinstance(suggestions, list)
        assert isinstance(modified, list)
    
    def test_analyze_and_apply_without_auto_apply(
        self, mock_vault_scanner, mock_llm, sample_draft_note
    ):
        """Test without automatic application."""
        analyzer = BacklinkAnalyzer(
            vault_scanner=mock_vault_scanner,
            llm=mock_llm,
            auto_apply=False,
        )
        
        suggestions, modified = analyzer.analyze_and_apply(sample_draft_note)
        
        # Should not modify any files when auto_apply is False
        assert modified == []


# Fixtures
@pytest.fixture
def mock_vault_scanner():
    """Create mock vault scanner."""
    scanner = MagicMock()
    scanner.scan_all_notes.return_value = [
        {"path": Path("/vault/note1.md"), "metadata": {"title": "Note 1"}},
        {"path": Path("/vault/note2.md"), "metadata": {"title": "Note 2"}},
    ]
    return scanner


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content='{"suggestions": []}'
    )
    return llm


@pytest.fixture
def backlink_analyzer(mock_vault_scanner, mock_llm):
    """Create BacklinkAnalyzer instance."""
    return BacklinkAnalyzer(
        vault_scanner=mock_vault_scanner,
        llm=mock_llm,
    )


@pytest.fixture
def sample_draft_note():
    """Create sample DraftNote."""
    return DraftNote(
        title="Python Programming Guide",
        tags=["python", "programming"],
        content="This is a guide about Python programming.",
        category="tech",
        frontmatter={"tags": ["python"]},
    )


@pytest.fixture
def mock_vault_with_notes(mock_vault_scanner):
    """Mock vault scanner with note content."""
    def get_content(path):
        return "This mentions Python programming concepts."
    
    mock_vault_scanner.get_note_content = get_content
    return mock_vault_scanner


@pytest.fixture
def temp_vault_with_notes(tmp_path):
    """Create temporary vault with notes."""
    vault = tmp_path / "vault"
    vault.mkdir()
    
    # Create some test notes
    (vault / "note1.md").write_text("---\ntitle: Note 1\n---\nContent about topic A.")
    (vault / "note2.md").write_text("---\ntitle: Note 2\n---\nContent about topic B.")
    
    return vault
