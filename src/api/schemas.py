"""ANA API Schemas.

Pydantic models for API requests and responses.
"""

from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class ProcessRequest(BaseModel):
    """Request to process a note."""
    content: str = Field(..., description="Raw note content")
    frontmatter: dict[str, Any] | None = Field(None, description="Optional frontmatter")
    title: str | None = Field(None, description="Note title for display")


class AnswerRequest(BaseModel):
    """Request to answer questions."""
    session_id: str = Field(..., description="Session ID from process response")
    answers: list[str] = Field(..., description="Answers to the questions")


class SaveRequest(BaseModel):
    """Request to save a note."""
    session_id: str = Field(..., description="Session ID")
    output_path: str | None = Field(None, description="Optional output path")
    overwrite: bool = Field(False, description="Overwrite existing file")


# =============================================================================
# Response Models
# =============================================================================

class AnalysisResult(BaseModel):
    """Analysis result for a note."""
    detected_concepts: list[str] = Field(default_factory=list)
    is_sufficient: bool = False
    should_split: bool = False
    split_suggestions: list[str] = Field(default_factory=list)
    category: str = "general"


class Question(BaseModel):
    """A question with category."""
    text: str
    category: str = "general"


class DraftNote(BaseModel):
    """Draft note content."""
    title: str
    content: str
    frontmatter: dict[str, Any] = Field(default_factory=dict)


class ProcessResponse(BaseModel):
    """Response from processing a note."""
    session_id: str = Field(..., description="Session ID for follow-up requests")
    status: str = Field(..., description="Status: 'needs_info', 'completed', 'error'")
    analysis: AnalysisResult | None = None
    questions: list[Question] = Field(default_factory=list)
    draft_note: DraftNote | None = None
    message: str | None = None


class SaveResponse(BaseModel):
    """Response from saving a note."""
    success: bool
    path: str | None = None
    message: str | None = None


class StatusResponse(BaseModel):
    """Server status response."""
    status: str = "ok"
    version: str = "0.1.0"
    active_sessions: int = 0
