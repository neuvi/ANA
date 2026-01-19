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


# =============================================================================
# Tag Models
# =============================================================================

class TagSuggestRequest(BaseModel):
    """Request to suggest tags for content."""
    content: str = Field(..., description="Note content")
    existing_tags: list[str] = Field(default_factory=list, description="Existing tags to exclude")
    max_tags: int = Field(5, ge=1, le=10, description="Maximum number of tags to suggest")


class TagNormalizeRequest(BaseModel):
    """Request to normalize tags."""
    tags: list[str] = Field(..., description="Tags to normalize")


class TagSuggestion(BaseModel):
    """A suggested tag."""
    tag: str = Field(..., description="Normalized tag name")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    source: str = Field("vault", description="Source: vault, ai, or normalized")
    usage_count: int = Field(0, description="Usage count in vault")


class TagSuggestResponse(BaseModel):
    """Response with tag suggestions."""
    suggestions: list[TagSuggestion] = Field(default_factory=list)


class TagNormalizeResponse(BaseModel):
    """Response with normalized tags."""
    original: list[str] = Field(default_factory=list)
    normalized: list[str] = Field(default_factory=list)


class VaultTagsResponse(BaseModel):
    """Response with all vault tags."""
    tags: dict[str, int] = Field(default_factory=dict, description="Tag name: usage count")
    total_unique: int = Field(0, description="Total unique tags")


# =============================================================================
# Streaming Models
# =============================================================================

class ProgressEvent(BaseModel):
    """SSE progress event."""
    type: str = Field(..., description="Event type: progress, heartbeat, complete, error")
    phase: str | None = Field(None, description="Processing phase")
    progress: float | None = Field(None, ge=0.0, le=1.0, description="Phase progress")
    overall_progress: float | None = Field(None, ge=0.0, le=1.0, description="Overall progress")
    message: str | None = Field(None, description="Status message")
    detail: str | None = Field(None, description="Additional detail")
    result: ProcessResponse | None = Field(None, description="Final result (for complete event)")

