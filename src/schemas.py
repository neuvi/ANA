"""Data Schemas for ANA.

Defines Pydantic models for input/output validation and LangGraph state management.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# =============================================================================
# Analysis Phase Schemas
# =============================================================================

class AnalysisResult(BaseModel):
    """Phase 1: Analysis result from raw note."""
    
    detected_concepts: list[str] = Field(
        default_factory=list,
        description="List of concepts detected in the note"
    )
    missing_context: list[str] = Field(
        default_factory=list,
        description="List of missing context items that need clarification"
    )
    is_sufficient: bool = Field(
        default=False,
        description="Whether the note has sufficient information"
    )
    detected_category: Optional[str] = Field(
        default=None,
        description="Category detected from content analysis"
    )
    existing_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Existing frontmatter metadata from the note"
    )
    should_split: bool = Field(
        default=False,
        description="Whether the note should be split into multiple notes"
    )
    split_suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested titles for split notes"
    )


# =============================================================================
# Interrogation Phase Schemas
# =============================================================================

class InteractionPayload(BaseModel):
    """Phase 2: Questions to ask the user."""
    
    questions_to_user: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Questions to ask the user (max 5)"
    )
    question_categories: list[str] = Field(
        default_factory=list,
        description="Category of each question: context, relation, or clarification"
    )


# =============================================================================
# Output Schemas
# =============================================================================

class DraftNote(BaseModel):
    """Draft atomic note ready for output."""
    
    title: str = Field(
        ...,
        description="Note title (sentence-style, descriptive)"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for the note"
    )
    content: str = Field(
        ...,
        description="Main content of the note"
    )
    category: str = Field(
        default="general",
        description="Category of the note"
    )
    frontmatter: dict[str, Any] = Field(
        default_factory=dict,
        description="Complete frontmatter (existing + new)"
    )
    suggested_links: list[str] = Field(
        default_factory=list,
        description="Suggested note links"
    )
    related_notes: list[str] = Field(
        default_factory=list,
        description="Related notes as wikilinks (e.g., [[Note Title]])"
    )


class AgentResponse(BaseModel):
    """Complete agent response."""
    
    status: Literal["needs_info", "completed"] = Field(
        ...,
        description="Current status of processing"
    )
    analysis: AnalysisResult = Field(
        ...,
        description="Analysis results"
    )
    interaction: Optional[InteractionPayload] = Field(
        default=None,
        description="Questions for user (if status is needs_info)"
    )
    draft_note: DraftNote = Field(
        ...,
        description="Current draft note"
    )
    template_used: str = Field(
        default="default",
        description="Source of template used: file, db, or ai"
    )


# =============================================================================
# LangGraph State
# =============================================================================

class AgentState(TypedDict, total=False):
    """LangGraph state for the ANA pipeline."""
    
    # Input
    raw_note: str
    input_file_path: Optional[str]
    
    # Metadata
    input_metadata: dict[str, Any]
    
    # Processing state
    user_answers: list[str]
    analysis: Optional[AnalysisResult]
    questions: Optional[InteractionPayload]
    
    # Category & Template
    category: str
    template: str
    template_source: str  # file, db, or ai
    
    # Output
    final_note: Optional[DraftNote]
    
    # Control
    iteration_count: int
    is_complete: bool
    error: Optional[str]


# =============================================================================
# Template Schema
# =============================================================================

class NoteTemplate(BaseModel):
    """Template definition for a category."""
    
    category: str = Field(
        ...,
        description="Category name"
    )
    template_content: str = Field(
        ...,
        description="Jinja2 template content"
    )
    required_fields: list[str] = Field(
        default_factory=list,
        description="Required frontmatter fields"
    )
    suggested_sections: list[str] = Field(
        default_factory=list,
        description="Suggested content sections"
    )
    created_at: str = Field(
        default="",
        description="Creation timestamp"
    )
    source: Literal["file", "db", "ai"] = Field(
        default="ai",
        description="Template source"
    )
