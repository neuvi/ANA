"""Backlink Analyzer Module.

Analyzes existing notes for potential backlinks to newly created notes.
Automatically inserts wikilinks to create bidirectional connections.
"""

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.logging_config import get_logger
from src.schemas import BacklinkSuggestion, DraftNote

logger = get_logger("backlink")

if TYPE_CHECKING:
    from src.vault_scanner import VaultScanner


BACKLINK_SYSTEM_PROMPT = """You are analyzing text to find locations where a new note could be linked.

Given:
1. New note title and key concepts
2. Existing note content

Task:
Find sentences or phrases in the existing note that relate to the new note's concepts.
These are potential locations to insert a wikilink [[New Note Title]].

Rules:
- Only suggest links where the connection is meaningful
- The matched text should naturally relate to the new note
- Prefer linking specific mentions over general references
- Maximum 3 suggestions per existing note

Output JSON format:
{
  "suggestions": [
    {
      "matched_text": "exact text from the note that relates",
      "line_number": 5,
      "confidence": 0.85,
      "reason": "why this is a good link location"
    }
  ]
}

If no good matches, return: {"suggestions": []}
"""


class BacklinkAnalyzer:
    """Analyze existing notes for potential backlinks to new notes.
    
    Workflow:
    1. Extract key concepts from new note
    2. Search existing notes for related content
    3. Identify specific locations for backlink insertion
    4. Apply backlinks automatically or with user approval
    """
    
    def __init__(
        self,
        vault_scanner: "VaultScanner",
        llm: BaseChatModel,
        auto_apply: bool = True,
        max_suggestions_per_note: int = 3,
        min_confidence: float = 0.6,
    ):
        """Initialize backlink analyzer.
        
        Args:
            vault_scanner: VaultScanner instance
            llm: Language model for analysis
            auto_apply: Automatically apply backlinks
            max_suggestions_per_note: Max suggestions per existing note
            min_confidence: Minimum confidence threshold
        """
        self.vault_scanner = vault_scanner
        self.llm = llm
        self.auto_apply = auto_apply
        self.max_suggestions_per_note = max_suggestions_per_note
        self.min_confidence = min_confidence
    
    def find_backlink_opportunities(
        self,
        new_note: DraftNote,
        max_notes_to_scan: int = 50,
    ) -> list[BacklinkSuggestion]:
        """Find places in existing notes where new note could be linked.
        
        Args:
            new_note: The newly created note
            max_notes_to_scan: Maximum number of existing notes to scan
            
        Returns:
            List of backlink suggestions
        """
        all_suggestions = []
        
        # Get key concepts from new note
        key_concepts = self._extract_key_concepts(new_note)
        if not key_concepts:
            return []
        
        # Get existing notes
        existing_notes = self.vault_scanner.scan_all_notes()[:max_notes_to_scan]
        
        for note_info in existing_notes:
            note_path = note_info["path"]
            note_title = self._get_note_title(note_info)
            
            # Skip self-reference
            if note_title == new_note.title:
                continue
            
            # Get note content
            try:
                content = note_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.debug(f"Failed to read note {note_path}: {e}")
                continue
            
            # Skip if new note is already linked
            if f"[[{new_note.title}]]" in content:
                continue
            
            # Find link opportunities in this note
            suggestions = self._analyze_note_for_backlinks(
                content=content,
                note_path=str(note_path),
                note_title=note_title,
                new_note=new_note,
                key_concepts=key_concepts,
            )
            
            all_suggestions.extend(suggestions)
        
        # Sort by confidence
        all_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return all_suggestions
    
    def apply_backlinks(
        self,
        suggestions: list[BacklinkSuggestion],
    ) -> list[Path]:
        """Apply backlinks to existing notes.
        
        Args:
            suggestions: List of approved suggestions
            
        Returns:
            List of modified file paths
        """
        modified_files = []
        
        # Group suggestions by source note
        by_note: dict[str, list[BacklinkSuggestion]] = {}
        for sugg in suggestions:
            if sugg.source_note_path not in by_note:
                by_note[sugg.source_note_path] = []
            by_note[sugg.source_note_path].append(sugg)
        
        for note_path_str, note_suggestions in by_note.items():
            note_path = Path(note_path_str)
            
            if not note_path.exists():
                continue
            
            try:
                content = note_path.read_text(encoding="utf-8")
                modified_content = self._insert_backlinks(content, note_suggestions)
                
                if modified_content != content:
                    note_path.write_text(modified_content, encoding="utf-8")
                    modified_files.append(note_path)
            except Exception as e:
                logger.warning(f"Failed to apply backlink to {note_path}: {e}")
                continue
        
        return modified_files
    
    def analyze_and_apply(
        self,
        new_note: DraftNote,
        max_notes_to_scan: int = 50,
    ) -> tuple[list[BacklinkSuggestion], list[Path]]:
        """Find backlink opportunities and apply them automatically.
        
        Args:
            new_note: The newly created note
            max_notes_to_scan: Maximum notes to scan
            
        Returns:
            Tuple of (all suggestions, modified files)
        """
        suggestions = self.find_backlink_opportunities(new_note, max_notes_to_scan)
        
        if self.auto_apply and suggestions:
            # Filter by confidence
            high_confidence = [s for s in suggestions if s.confidence >= self.min_confidence]
            modified = self.apply_backlinks(high_confidence)
            return suggestions, modified
        
        return suggestions, []
    
    def _extract_key_concepts(self, note: DraftNote) -> list[str]:
        """Extract key concepts from a note for matching."""
        concepts = []
        
        # Add title
        concepts.append(note.title)
        
        # Add tags
        concepts.extend(note.tags)
        
        # Extract key phrases from title
        title_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', note.title)
        concepts.extend(title_words)
        
        # Add category if meaningful
        if note.category and note.category != "general":
            concepts.append(note.category)
        
        return list(set(concepts))
    
    def _get_note_title(self, note_info: dict) -> str:
        """Get title from note info."""
        metadata = note_info.get("metadata", {})
        if metadata and metadata.get("title"):
            return metadata["title"]
        return note_info["path"].stem
    
    def _analyze_note_for_backlinks(
        self,
        content: str,
        note_path: str,
        note_title: str,
        new_note: DraftNote,
        key_concepts: list[str],
    ) -> list[BacklinkSuggestion]:
        """Analyze a single note for backlink opportunities."""
        suggestions = []
        
        # Quick check: does content mention any key concepts?
        content_lower = content.lower()
        has_match = any(
            concept.lower() in content_lower 
            for concept in key_concepts
        )
        
        if not has_match:
            return []
        
        # Use LLM for detailed analysis
        try:
            llm_suggestions = self._llm_analyze(content, new_note, key_concepts)
            
            for i, sugg in enumerate(llm_suggestions[:self.max_suggestions_per_note]):
                if sugg.get("confidence", 0) < self.min_confidence:
                    continue
                
                # Generate unique ID
                sugg_id = hashlib.md5(
                    f"{note_path}:{sugg.get('line_number', 0)}:{new_note.title}".encode()
                ).hexdigest()[:8]
                
                # Get context
                lines = content.split("\n")
                line_num = sugg.get("line_number", 1) - 1
                context_before = lines[max(0, line_num - 1)] if line_num > 0 else ""
                context_after = lines[min(len(lines) - 1, line_num + 1)] if line_num < len(lines) - 1 else ""
                
                suggestions.append(BacklinkSuggestion(
                    id=sugg_id,
                    source_note_path=note_path,
                    source_note_title=note_title,
                    target_note_title=new_note.title,
                    context_before=context_before[:100],
                    context_after=context_after[:100],
                    matched_text=sugg.get("matched_text", ""),
                    line_number=sugg.get("line_number", 1),
                    confidence=sugg.get("confidence", 0.5),
                    reason=sugg.get("reason", ""),
                ))
        except Exception as e:
            logger.debug(f"LLM analysis failed, using fallback: {e}")
            # Fallback: simple keyword matching
            suggestions = self._simple_match(
                content, note_path, note_title, new_note, key_concepts
            )
        
        return suggestions
    
    def _llm_analyze(
        self,
        content: str,
        new_note: DraftNote,
        key_concepts: list[str],
    ) -> list[dict]:
        """Use LLM to analyze content for backlink opportunities."""
        import json
        
        prompt = f"""New note to link:
Title: {new_note.title}
Key concepts: {', '.join(key_concepts)}

Existing note content:
{content[:3000]}

Find locations where [[{new_note.title}]] could be linked."""
        
        messages = [
            SystemMessage(content=BACKLINK_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse JSON
        try:
            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            data = json.loads(response_text)
            return data.get("suggestions", [])
        except json.JSONDecodeError:
            return []
    
    def _simple_match(
        self,
        content: str,
        note_path: str,
        note_title: str,
        new_note: DraftNote,
        key_concepts: list[str],
    ) -> list[BacklinkSuggestion]:
        """Simple keyword-based matching fallback."""
        suggestions = []
        lines = content.split("\n")
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            for concept in key_concepts:
                if concept.lower() in line_lower:
                    # Skip if already has a wikilink
                    if "[[" in line:
                        continue
                    
                    # Skip frontmatter
                    if line.strip().startswith("---"):
                        continue
                    
                    sugg_id = hashlib.md5(
                        f"{note_path}:{i}:{new_note.title}".encode()
                    ).hexdigest()[:8]
                    
                    suggestions.append(BacklinkSuggestion(
                        id=sugg_id,
                        source_note_path=note_path,
                        source_note_title=note_title,
                        target_note_title=new_note.title,
                        matched_text=line.strip()[:100],
                        line_number=i + 1,
                        confidence=0.6,
                        reason=f"Contains keyword: {concept}",
                    ))
                    break  # One suggestion per line
        
        return suggestions[:self.max_suggestions_per_note]
    
    def _insert_backlinks(
        self,
        content: str,
        suggestions: list[BacklinkSuggestion],
    ) -> str:
        """Insert backlinks into content."""
        lines = content.split("\n")
        
        # Sort by line number descending to avoid index shifting
        sorted_suggestions = sorted(suggestions, key=lambda x: x.line_number, reverse=True)
        
        for sugg in sorted_suggestions:
            line_idx = sugg.line_number - 1
            
            if 0 <= line_idx < len(lines):
                line = lines[line_idx]
                wikilink = f"[[{sugg.target_note_title}]]"
                
                # Try to insert after matched text
                if sugg.matched_text and sugg.matched_text in line:
                    # Insert wikilink after the matched text
                    lines[line_idx] = line.replace(
                        sugg.matched_text,
                        f"{sugg.matched_text} {wikilink}",
                        1  # Only first occurrence
                    )
                else:
                    # Append to end of line
                    lines[line_idx] = f"{line} {wikilink}"
        
        return "\n".join(lines)
