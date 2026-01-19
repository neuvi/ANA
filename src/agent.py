"""ANA Agent Module.

Main agent class that orchestrates the entire atomic note creation pipeline.
"""

from pathlib import Path
from typing import Any

from src.category_classifier import CategoryClassifier
from src.config import ANAConfig
from src.embedding_cache import EmbeddingCache
from src.graph import build_graph, continue_with_answers, create_initial_state
from src.link_analyzer import LinkAnalyzer
from src.llm_config import get_llm
from src.logging_config import get_logger
from src.schemas import AgentResponse, AgentState, AnalysisResult, BacklinkSuggestion, DraftNote, InteractionPayload
from src.smart_tags import SmartTagManager, TagSuggestion
from src.template_manager import TemplateManager
from src.utils import save_note_to_file
from src.validators import validate_raw_note, ValidationError
from src.vault_scanner import VaultScanner

logger = get_logger("agent")


class AtomicNoteArchitect:
    """ANA (Atomic Note Architect) Agent.
    
    Transforms raw notes into Zettelkasten-style atomic notes through
    a 3-phase pipeline: Analysis → Interrogation → Synthesis.
    
    Features:
    - Frontmatter metadata extraction and preservation
    - Category classification (Frontmatter → AI)
    - Template management (File → DB → AI)
    - Interactive questioning (up to 5 questions per round)
    - Automatic note linking with related notes
    """
    
    def __init__(self, config: ANAConfig | None = None):
        """Initialize the ANA agent.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ANAConfig()
        
        # Initialize LLM
        self.llm = get_llm(self.config)
        
        # Initialize components
        self.vault_scanner = VaultScanner(self.config.get_vault_path())
        self.category_classifier = CategoryClassifier(self.vault_scanner, self.llm)
        self.template_manager = TemplateManager(
            self.config, 
            self.llm, 
            self.vault_scanner
        )
        
        # Initialize note linking components
        self.embedding_cache = EmbeddingCache(
            vault_path=self.config.get_vault_path(),
            ollama_base_url=self.config.ollama_base_url,
            embedding_model=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
            use_vector_db=self.config.use_vector_db,
        )
        self.link_analyzer = LinkAnalyzer(
            vault_scanner=self.vault_scanner,
            embedding_cache=self.embedding_cache,
            rerank_model=self.config.rerank_model,
        )
        
        # Build the graph
        self.graph = build_graph(
            self.llm,
            self.vault_scanner,
            max_questions=self.config.max_questions,
            max_iterations=self.config.max_iterations,
        )
        
        # Current processing state
        self._current_state: AgentState | None = None
        self._current_category: str = "general"
        self._current_template: str = ""
        self._template_source: str = "default"
        
        # Initialize backlink analyzer
        from src.backlink_analyzer import BacklinkAnalyzer
        self.backlink_analyzer = BacklinkAnalyzer(
            vault_scanner=self.vault_scanner,
            llm=self.llm,
            auto_apply=True,
            min_confidence=0.6,
        )
        
        # Initialize smart tag manager
        self.smart_tags = SmartTagManager(
            vault_scanner=self.vault_scanner,
            config=self.config,
            llm=self.llm,
        )
    
    def process(self, raw_note: str, frontmatter: dict[str, Any] | None = None) -> AgentResponse:
        """Process a raw note through the pipeline.
        
        Args:
            raw_note: Raw note content (may include frontmatter)
            frontmatter: Optional pre-extracted frontmatter
            
        Returns:
            AgentResponse with status, analysis, and current draft
            
        Raises:
            ValidationError: If raw note fails validation
        """
        # Validate raw note
        validation = validate_raw_note(raw_note)
        if not validation.is_valid:
            raise ValidationError(validation.message, field="raw_note")
        
        for warning in validation.warnings:
            logger.warning(f"Note validation warning: {warning}")
        
        # Extract frontmatter if not provided
        if frontmatter is None:
            frontmatter = self.vault_scanner.parse_frontmatter(raw_note) or {}
        
        # Classify category
        self._current_category, is_new = self.category_classifier.suggest_category(
            raw_note, frontmatter
        )
        logger.debug(f"Classified category: {self._current_category} (new: {is_new})")
        
        # Get template
        sample_notes = None
        if not is_new:
            sample_notes = self.vault_scanner.find_similar_notes(
                self._current_category, limit=3
            )
        
        self._current_template, self._template_source = self.template_manager.get_template(
            self._current_category, sample_notes
        )
        logger.debug(f"Using template from: {self._template_source}")
        
        # Create initial state
        self._current_state = create_initial_state(
            raw_note=raw_note,
            category=self._current_category,
            template=self._current_template,
            template_source=self._template_source,
        )
        
        # Add frontmatter to state
        self._current_state["input_metadata"] = frontmatter
        
        # Run the graph
        result = self.graph.invoke(self._current_state)
        self._current_state = result
        
        return self._build_response(result)
    
    def answer_questions(self, answers: list[str]) -> AgentResponse:
        """Continue processing with user's answers.
        
        Args:
            answers: List of answers to the questions
            
        Returns:
            AgentResponse with updated status and draft
            
        Raises:
            RuntimeError: If no processing is in progress
        """
        if self._current_state is None:
            raise RuntimeError("No processing in progress. Call process() first.")
        
        # Update state with answers
        self._current_state = continue_with_answers(self._current_state, answers)
        
        # Continue graph execution by re-analyzing with new answers
        # Reset to analyze phase
        result = self.graph.invoke(self._current_state)
        self._current_state = result
        
        return self._build_response(result)
    
    def save_note(
        self,
        note: DraftNote | None = None,
        output_dir: Path | str | None = None,
        overwrite: bool = False
    ) -> Path:
        """Save the final note to a file.
        
        Args:
            note: Note to save. Uses current draft if not provided.
            output_dir: Directory to save to. Uses vault path if not provided.
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to saved file
            
        Raises:
            ValueError: If no note is available to save
        """
        if note is None:
            if self._current_state is None or self._current_state.get("final_note") is None:
                raise ValueError("No note available to save. Complete processing first.")
            note = self._current_state["final_note"]
        
        if output_dir is None:
            output_dir = self.config.get_vault_path()
        
        return save_note_to_file(
            note,
            output_dir,
            overwrite=overwrite
        )
    
    def save_note_with_backlinks(
        self,
        note: DraftNote | None = None,
        output_dir: Path | str | None = None,
        overwrite: bool = False
    ) -> tuple[Path, list[BacklinkSuggestion], list[Path]]:
        """Save the note and automatically add backlinks to existing notes.
        
        Args:
            note: Note to save. Uses current draft if not provided.
            output_dir: Directory to save to. Uses vault path if not provided.
            overwrite: Whether to overwrite existing file
            
        Returns:
            Tuple of (saved_path, backlink_suggestions, modified_files)
        """
        # Save the note first
        saved_path = self.save_note(note, output_dir, overwrite)
        
        # Get the note for backlink analysis
        if note is None:
            note = self._current_state.get("final_note")
        
        if note is None:
            return saved_path, [], []
        
        # Find and apply backlinks
        suggestions, modified_files = self.backlink_analyzer.analyze_and_apply(note)
        
        return saved_path, suggestions, modified_files
    
    def extract_for_split(
        self,
        raw_note: str,
        target_topic: str
    ) -> tuple[str, list[str]]:
        """Extract content for a split topic from the original note.
        
        Args:
            raw_note: Original note content
            target_topic: The topic/title to extract content for
            
        Returns:
            Tuple of (extracted_content, key_points)
        """
        from src.prompts import SPLIT_EXTRACTION_PROMPT
        import json
        
        prompt = SPLIT_EXTRACTION_PROMPT.format(
            raw_note=raw_note,
            target_topic=target_topic
        )
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            # Find JSON in response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)
                
                extracted = data.get("extracted_content", "")
                key_points = data.get("key_points", [])
                
                return extracted, key_points
        except Exception as e:
            logger.warning(f"Failed to extract split content: {e}")
        
        # Fallback: return topic as prompt for user
        return f"# {target_topic}\n\n(Extracted from original note)", []
    
    def reset(self):
        """Reset the agent state for new processing."""
        self._current_state = None
        self._current_category = "general"
        self._current_template = ""
        self._template_source = "default"
    
    def get_current_state(self) -> AgentState | None:
        """Get the current processing state.
        
        Returns:
            Current AgentState or None if not processing
        """
        return self._current_state
    
    def get_category(self) -> str:
        """Get the current category.
        
        Returns:
            Current category name
        """
        return self._current_category
    
    def is_new_category(self) -> bool:
        """Check if the current category is new.
        
        Returns:
            True if category is new
        """
        return self.category_classifier.is_new_category(self._current_category)
    
    def get_smart_tag_suggestions(
        self,
        content: str,
        existing_tags: list[str] | None = None,
        max_tags: int = 5
    ) -> list[TagSuggestion]:
        """Get smart tag suggestions for content.
        
        Uses vault tags and AI to suggest relevant tags.
        
        Args:
            content: Note content to analyze
            existing_tags: Tags to exclude from suggestions
            max_tags: Maximum number of suggestions
            
        Returns:
            List of TagSuggestion objects
        """
        return self.smart_tags.suggest_tags(
            content=content,
            existing_tags=existing_tags or [],
            max_tags=max_tags
        )
    
    def _build_response(self, state: AgentState) -> AgentResponse:
        """Build AgentResponse from current state.
        
        Args:
            state: Current AgentState
            
        Returns:
            AgentResponse
        """
        analysis = state.get("analysis") or AnalysisResult(
            detected_concepts=[],
            missing_context=[],
            is_sufficient=False,
            existing_metadata=state.get("input_metadata", {}),
        )
        
        questions = state.get("questions")
        final_note = state.get("final_note")
        is_complete = state.get("is_complete", False)
        
        # Determine status
        if is_complete and final_note:
            status = "completed"
            
            # Add related notes if enabled
            if self.config.enable_note_linking and final_note:
                try:
                    # 1. Get existing suggestions from LLM
                    llm_suggested = final_note.suggested_links or []
                    processed_links = []
                    
                    # 2. Check existence for LLM suggestions
                    existing_titles = self.link_analyzer._get_all_note_titles()
                    
                    for link in llm_suggested:
                        # Remove [[ ]] if present
                        clean_link = link.replace("[[", "").replace("]]", "")
                        if clean_link in existing_titles:
                            processed_links.append(clean_link)
                        else:
                            processed_links.append(f"{clean_link} (new)")
                            
                    # 3. Get LinkAnalyzer results (guaranteed to be existing)
                    analyzer_related = self.link_analyzer.find_related_notes(
                        note_title=final_note.title,
                        note_content=final_note.content,
                        note_tags=final_note.tags,
                        note_category=final_note.category,
                        max_links=self.config.max_related_links,
                    )
                    
                    # Merge sets (avoid duplicates)
                    # Helper to get raw title from wikilink
                    def get_raw_title(wikilink):
                        return wikilink.replace("[[", "").replace("]]", "").replace(" (new)", "")

                    current_titles = set()
                    final_links_list = []
                    
                    # Add LLM suggestions first (they are more contextually relevant)
                    for link in processed_links:
                        raw = get_raw_title(link)
                        if raw not in current_titles:
                            final_links_list.append(link)
                            current_titles.add(raw)
                    
                    # Add Analyzer results (append existing ones if not present)
                    for link in analyzer_related:
                         # Analyzer returns "[[Title]]" or "[[Title (new)]]" (though currently only existing)
                        raw = get_raw_title(link)
                        if raw not in current_titles:
                            # Analyzer results are already formatted as [[Title]]
                            # We want to store just the content inside [[ ]] for suggested_links compatibility?
                            # The template does: - [[{{ link }}]]
                            # So we should store just "Title" or "Title (new)"
                            content = link.replace("[[", "").replace("]]", "")
                            final_links_list.append(content)
                            current_titles.add(raw)
                    
                    # Update suggested_links (used by Template)
                    final_note.suggested_links = final_links_list
                    
                    # Update related_notes (used by Schema/Frontmatter)
                    final_note.related_notes = [f"[[{l}]]" for l in final_links_list]
                    
                    # Also add to frontmatter
                    final_note.frontmatter["related"] = final_note.related_notes
                    
                except Exception as e:
                    logger.warning(f"Note linking failed: {e}")
        else:
            status = "needs_info"
        
        # Create draft note even if not complete
        if final_note is None:
            final_note = DraftNote(
                title="(Draft)",
                tags=[],
                content=state.get("raw_note", ""),
                category=self._current_category,
                frontmatter=state.get("input_metadata", {}),
                suggested_links=[],
            )
        
        return AgentResponse(
            status=status,
            analysis=analysis,
            interaction=questions if status == "needs_info" else None,
            draft_note=final_note,
            template_used=self._template_source,
        )
    
    def sync_embeddings(self, progress_callback=None) -> dict[str, int]:
        """Sync embeddings for all notes in the vault.
        
        Only updates embeddings for changed files.
        
        Args:
            progress_callback: Optional callback(current, total, file_name)
            
        Returns:
            Stats dict with 'updated', 'cached', 'failed' counts
        """
        return self.embedding_cache.sync_vault(
            self.vault_scanner,
            progress_callback
        )
    
    async def sync_embeddings_async(
        self,
        progress_callback=None,
        max_concurrency: int = 5
    ) -> dict[str, int]:
        """Sync embeddings asynchronously for all notes in the vault.
        
        Uses parallel processing for improved performance.
        
        Args:
            progress_callback: Optional callback(current, total, file_name)
            max_concurrency: Maximum concurrent embedding requests
            
        Returns:
            Stats dict with 'updated', 'cached', 'failed' counts
        """
        return await self.embedding_cache.sync_vault_async(
            self.vault_scanner,
            progress_callback,
            max_concurrency
        )


def create_agent(config: ANAConfig | None = None) -> AtomicNoteArchitect:
    """Factory function to create an ANA agent.
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured AtomicNoteArchitect instance
    """
    return AtomicNoteArchitect(config)
