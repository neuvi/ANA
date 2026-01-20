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


# =============================================================================
# Sync Models
# =============================================================================

class SyncRequest(BaseModel):
    """Request to sync embeddings."""
    force: bool = Field(False, description="Force re-sync all embeddings")
    use_async: bool = Field(True, description="Use async parallel processing")


class SyncResponse(BaseModel):
    """Response from sync operation."""
    updated: int = Field(0, description="Number of embeddings updated")
    cached: int = Field(0, description="Number of cached embeddings")
    failed: int = Field(0, description="Number of failed embeddings")
    message: str | None = None


class SyncStatsResponse(BaseModel):
    """Response with embedding cache statistics."""
    total_files: int = 0
    total_embeddings: int = 0
    embedding_dimension: int = 0
    cache_size_human: str = "0 B"
    embedding_model: str = ""
    vector_db_enabled: bool = False


# =============================================================================
# Config Models
# =============================================================================

class ConfigResponse(BaseModel):
    """Response with current configuration."""
    llm_provider: str = ""
    llm_model: str = ""
    vault_path: str = ""
    output_language: str = "ko"
    embedding_model: str = ""
    embedding_enabled: bool = True
    rerank_enabled: bool = False
    ollama_base_url: str = ""


class ConfigSetRequest(BaseModel):
    """Request to set a configuration value."""
    value: str = Field(..., description="New value for the configuration key")


class ConfigSetResponse(BaseModel):
    """Response from setting a configuration value."""
    success: bool
    key: str
    old_value: str | None = None
    new_value: str | None = None
    message: str | None = None


# =============================================================================
# Health/Doctor Models
# =============================================================================

class HealthCheck(BaseModel):
    """A single health check result."""
    name: str = Field(..., description="Check name")
    status: str = Field(..., description="Status: ok, warning, error")
    message: str = Field(..., description="Status message")
    fix_hint: str | None = Field(None, description="Hint for fixing the issue")


class HealthResponse(BaseModel):
    """Response with health check results."""
    status: str = Field("ok", description="Overall status: ok, warning, error")
    version: str = "0.1.0"
    checks: list[HealthCheck] = Field(default_factory=list)
    summary: dict[str, int] = Field(
        default_factory=lambda: {"ok": 0, "warning": 0, "error": 0}
    )


# =============================================================================
# Prompts Models
# =============================================================================

class PromptInfo(BaseModel):
    """Information about a single prompt."""
    prompt_type: str = Field(..., description="Prompt type")
    source: str = Field(..., description="Source: default or custom")
    path: str | None = Field(None, description="Path if custom prompt")
    is_valid: bool = True
    validation_message: str | None = None


class PromptsInfoResponse(BaseModel):
    """Response with prompts configuration info."""
    prompts: list[PromptInfo] = Field(default_factory=list)
    custom_prompts_dir: str | None = None


class PromptsValidateResponse(BaseModel):
    """Response from validating prompts."""
    all_valid: bool = True
    results: list[PromptInfo] = Field(default_factory=list)


# =============================================================================
# Backlink Models
# =============================================================================

class BacklinkSuggestRequest(BaseModel):
    """Request to suggest backlinks for a note."""
    title: str = Field(..., description="Note title")
    content: str = Field(..., description="Note content")
    tags: list[str] = Field(default_factory=list, description="Note tags")
    max_notes_to_scan: int = Field(50, ge=1, le=200, description="Max notes to scan")


class BacklinkSuggestionItem(BaseModel):
    """A single backlink suggestion."""
    source_path: str = Field(..., description="Path to the source note")
    source_title: str = Field(..., description="Title of the source note")
    matched_text: str = Field(..., description="Matched text in source note")
    line_number: int = Field(0, description="Line number of the match")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field("", description="Reason for suggestion")


class BacklinkSuggestResponse(BaseModel):
    """Response with backlink suggestions."""
    suggestions: list[BacklinkSuggestionItem] = Field(default_factory=list)
    notes_scanned: int = 0


class BacklinkApplyRequest(BaseModel):
    """Request to apply backlink suggestions."""
    session_id: str = Field(..., description="Session ID")
    suggestion_indices: list[int] = Field(..., description="Indices of suggestions to apply")

