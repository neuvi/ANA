"""ANA API Server.

FastAPI-based REST API server for Obsidian plugin integration.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.agent import AtomicNoteArchitect
from src.api.schemas import (
    AnswerRequest,
    AnalysisResult,
    DraftNote,
    ProcessRequest,
    ProcessResponse,
    Question,
    SaveRequest,
    SaveResponse,
    StatusResponse,
    TagSuggestRequest,
    TagNormalizeRequest,
    TagSuggestResponse,
    TagNormalizeResponse,
    VaultTagsResponse,
    TagSuggestion,
    # New schemas
    SyncRequest,
    SyncResponse,
    SyncStatsResponse,
    ConfigResponse,
    ConfigSetRequest,
    ConfigSetResponse,
    HealthCheck,
    HealthResponse,
    PromptInfo,
    PromptsInfoResponse,
    PromptsValidateResponse,
    BacklinkSuggestRequest,
    BacklinkSuggestionItem,
    BacklinkSuggestResponse,
    BacklinkApplyRequest,
)
from src.config import ANAConfig
from src.progress import ProgressTracker, ProcessingPhase, SSEEventGenerator
from src.smart_tags import SmartTagManager
from src.vault_scanner import VaultScanner



# Session storage for active processing sessions
_sessions: dict[str, dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    app.state.config = ANAConfig()
    yield
    # Shutdown
    _sessions.clear()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    
    app = FastAPI(
        title="ANA API",
        description="Atomic Note Architect REST API for Obsidian plugin",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Configure CORS for Obsidian plugin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Obsidian uses app:// protocol
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register API routes
    app.include_router(api_router)
    
    # Serve WebUI static files
    webui_path = Path(__file__).parent.parent / "webui" / "static"
    if webui_path.exists():
        app.mount("/webui/static", StaticFiles(directory=str(webui_path)), name="webui-static")
        
        @app.get("/webui")
        @app.get("/webui/")
        async def serve_webui():
            """Serve the WebUI main page."""
            index_path = webui_path / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return {"error": "WebUI not found"}
    
    return app


# =============================================================================
# API Routes
# =============================================================================

from fastapi import APIRouter

api_router = APIRouter(prefix="/api", tags=["ana"])


@api_router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get server status."""
    return StatusResponse(
        status="ok",
        version="0.1.0",
        active_sessions=len(_sessions)
    )


@api_router.post("/process", response_model=ProcessResponse)
async def process_note(request: ProcessRequest):
    """Process a raw note through ANA pipeline."""
    try:
        # Create new session
        session_id = str(uuid.uuid4())
        
        # Initialize agent
        config = ANAConfig()
        agent = AtomicNoteArchitect(config)
        
        # Process note
        response = agent.process(request.content, request.frontmatter)
        
        # Store session
        _sessions[session_id] = {
            "agent": agent,
            "response": response,
        }
        
        # Build API response
        analysis = None
        if response.analysis:
            analysis = AnalysisResult(
                detected_concepts=response.analysis.detected_concepts,
                is_sufficient=response.analysis.is_sufficient,
                should_split=response.analysis.should_split,
                split_suggestions=response.analysis.split_suggestions or [],
                category=agent.get_category(),
            )
        
        questions = []
        if response.interaction and response.interaction.questions_to_user:
            categories = response.interaction.question_categories or []
            for i, q in enumerate(response.interaction.questions_to_user):
                cat = categories[i] if i < len(categories) else "general"
                questions.append(Question(text=q, category=cat))
        
        draft = None
        if response.draft_note:
            draft = DraftNote(
                title=response.draft_note.title,
                content=response.draft_note.content,
                frontmatter=response.draft_note.frontmatter or {},
            )
        
        return ProcessResponse(
            session_id=session_id,
            status=response.status,
            analysis=analysis,
            questions=questions,
            draft_note=draft,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/answer", response_model=ProcessResponse)
async def answer_questions(request: AnswerRequest):
    """Answer questions and continue processing."""
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        agent: AtomicNoteArchitect = session["agent"]
        
        # Continue with answers
        response = agent.answer_questions(request.answers)
        
        # Update session
        session["response"] = response
        
        # Build API response
        questions = []
        if response.interaction and response.interaction.questions_to_user:
            categories = response.interaction.question_categories or []
            for i, q in enumerate(response.interaction.questions_to_user):
                cat = categories[i] if i < len(categories) else "general"
                questions.append(Question(text=q, category=cat))
        
        draft = None
        if response.draft_note:
            draft = DraftNote(
                title=response.draft_note.title,
                content=response.draft_note.content,
                frontmatter=response.draft_note.frontmatter or {},
            )
        
        return ProcessResponse(
            session_id=request.session_id,
            status=response.status,
            analysis=None,
            questions=questions,
            draft_note=draft,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/save", response_model=SaveResponse)
async def save_note(request: SaveRequest):
    """Save the completed note."""
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        agent: AtomicNoteArchitect = session["agent"]
        
        # Save note
        saved_path = agent.save_note(
            output_dir=request.output_path,
            overwrite=request.overwrite
        )
        
        # Clean up session
        del _sessions[request.session_id]
        
        return SaveResponse(
            success=True,
            path=str(saved_path),
            message="Note saved successfully"
        )
        
    except FileExistsError:
        return SaveResponse(
            success=False,
            message="File already exists. Set overwrite=true to replace."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session without saving."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "ok", "message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


# =============================================================================
# Tag Routes
# =============================================================================

@api_router.get("/tags", response_model=VaultTagsResponse)
async def get_vault_tags():
    """Get all tags from the vault with usage counts."""
    try:
        config = ANAConfig()
        vault_scanner = VaultScanner(config.get_vault_path())
        smart_tags = SmartTagManager(vault_scanner, config)
        
        tags = smart_tags.get_all_tags()
        
        return VaultTagsResponse(
            tags=tags,
            total_unique=len(tags)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/tags/suggest", response_model=TagSuggestResponse)
async def suggest_tags(request: TagSuggestRequest):
    """Suggest tags for content based on vault tags and AI."""
    try:
        config = ANAConfig()
        vault_scanner = VaultScanner(config.get_vault_path())
        smart_tags = SmartTagManager(vault_scanner, config)
        
        suggestions = smart_tags.suggest_tags(
            content=request.content,
            existing_tags=request.existing_tags,
            max_tags=request.max_tags
        )
        
        return TagSuggestResponse(
            suggestions=[
                TagSuggestion(
                    tag=s.tag,
                    confidence=s.confidence,
                    source=s.source,
                    usage_count=s.usage_count
                )
                for s in suggestions
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/tags/normalize", response_model=TagNormalizeResponse)
async def normalize_tags(request: TagNormalizeRequest):
    """Normalize a list of tags."""
    try:
        config = ANAConfig()
        vault_scanner = VaultScanner(config.get_vault_path())
        smart_tags = SmartTagManager(vault_scanner, config)
        
        normalized = smart_tags.normalize_tags(request.tags)
        
        return TagNormalizeResponse(
            original=request.tags,
            normalized=normalized
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Streaming Routes
# =============================================================================

@api_router.post("/process/stream")
async def process_note_stream(request: ProcessRequest):
    """Process a note with SSE streaming for real-time progress updates.
    
    Returns Server-Sent Events (SSE) stream with progress updates.
    Event types:
    - progress: Processing progress update
    - heartbeat: Keep-alive signal
    - complete: Processing completed with result
    - error: Processing error
    """
    import asyncio
    import json
    
    async def event_generator():
        sse = SSEEventGenerator(heartbeat_interval=5.0)
        
        try:
            # Create new session
            session_id = str(uuid.uuid4())
            config = ANAConfig()
            
            # Progress callback that sends SSE events
            async def send_progress(update):
                await sse.send_update(update)
            
            # Create tracker with async callback  
            tracker = ProgressTracker(async_callback=send_progress)
            
            # Send initial event
            tracker.update(ProcessingPhase.INITIALIZING, 0.0, "처리 시작...")
            await asyncio.sleep(0.1)  # Allow event to be sent
            
            # Initialize agent
            tracker.update(ProcessingPhase.INITIALIZING, 0.5, "에이전트 초기화 중...")
            agent = AtomicNoteArchitect(config)
            await asyncio.sleep(0.1)
            
            # Process note (run in thread to not block)
            tracker.update(ProcessingPhase.ANALYZING, 0.0, "노트 분석 중...")
            
            # Run blocking process in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: agent.process(request.content, request.frontmatter)
            )
            
            tracker.update(ProcessingPhase.ANALYZING, 1.0, "분석 완료")
            
            # Store session
            _sessions[session_id] = {
                "agent": agent,
                "response": response,
            }
            
            # Build API response
            analysis = None
            if response.analysis:
                analysis = AnalysisResult(
                    detected_concepts=response.analysis.detected_concepts,
                    is_sufficient=response.analysis.is_sufficient,
                    should_split=response.analysis.should_split,
                    split_suggestions=response.analysis.split_suggestions or [],
                    category=agent.get_category(),
                )
            
            questions = []
            if response.interaction and response.interaction.questions_to_user:
                tracker.update(ProcessingPhase.GENERATING_QUESTIONS, 1.0, "질문 생성 완료")
                categories = response.interaction.question_categories or []
                for i, q in enumerate(response.interaction.questions_to_user):
                    cat = categories[i] if i < len(categories) else "general"
                    questions.append(Question(text=q, category=cat))
            
            draft = None
            if response.draft_note:
                tracker.update(ProcessingPhase.SYNTHESIZING, 1.0, "노트 합성 완료")
                draft = DraftNote(
                    title=response.draft_note.title,
                    content=response.draft_note.content,
                    frontmatter=response.draft_note.frontmatter or {},
                )
            
            # Send completion
            tracker.complete("처리 완료!")
            
            api_response = ProcessResponse(
                session_id=session_id,
                status=response.status,
                analysis=analysis,
                questions=questions,
                draft_note=draft,
            )
            
            await sse.send_complete(api_response.model_dump())
            
        except Exception as e:
            await sse.send_error(str(e))
        finally:
            sse.close()
        
        # Yield all queued events
        async for event in sse.generate():
            yield event
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
# Sync Routes
# =============================================================================

@api_router.post("/sync", response_model=SyncResponse)
async def sync_embeddings(request: SyncRequest = SyncRequest()):
    """Sync embeddings for all vault notes."""
    try:
        config = ANAConfig()
        vault_path = config.get_vault_path()
        
        from src.embedding_cache import EmbeddingCache
        
        cache = EmbeddingCache(
            vault_path=vault_path,
            ollama_base_url=config.get_ollama_base_url(),
            embedding_model=config.get_embedding_model(),
        )
        vault_scanner = VaultScanner(vault_path)
        
        # Clear cache if force is specified
        if request.force:
            cache.clear_cache()
        
        # Use async or sync method based on request
        if request.use_async:
            stats = await cache.sync_vault_async(vault_scanner)
        else:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None,
                lambda: cache.sync_vault(vault_scanner)
            )
        
        return SyncResponse(
            updated=stats.get("updated", 0),
            cached=stats.get("cached", 0),
            failed=stats.get("failed", 0),
            message="Sync completed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/sync/stats", response_model=SyncStatsResponse)
async def get_sync_stats():
    """Get embedding cache statistics."""
    try:
        config = ANAConfig()
        vault_path = config.get_vault_path()
        
        from src.embedding_cache import EmbeddingCache
        
        cache = EmbeddingCache(
            vault_path=vault_path,
            ollama_base_url=config.get_ollama_base_url(),
            embedding_model=config.get_embedding_model(),
        )
        
        stats = cache.get_stats()
        
        return SyncStatsResponse(
            total_files=stats.get("total_files", 0),
            total_embeddings=stats.get("total_embeddings", 0),
            embedding_dimension=stats.get("embedding_dimension", 0),
            cache_size_human=stats.get("cache_size_human", "0 B"),
            embedding_model=stats.get("embedding_model", ""),
            vector_db_enabled=stats.get("vector_db_enabled", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Config Routes
# =============================================================================

@api_router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration."""
    try:
        config = ANAConfig()
        
        return ConfigResponse(
            llm_provider=config.get_llm_provider(),
            llm_model=config.get_llm_model(),
            vault_path=str(config.get_vault_path()),
            output_language=config.get_output_language(),
            embedding_model=config.get_embedding_model(),
            embedding_enabled=config.get_embedding_enabled(),
            rerank_enabled=config.get_rerank_enabled(),
            ollama_base_url=config.get_ollama_base_url(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.put("/config/{key}", response_model=ConfigSetResponse)
async def set_config(key: str, request: ConfigSetRequest):
    """Set a configuration value."""
    try:
        from src.cli.config_wizard import set_config_value
        
        # Get old value for response
        config = ANAConfig()
        old_value = None
        try:
            getter = getattr(config, f"get_{key}", None)
            if getter:
                old_value = str(getter())
        except Exception:
            pass
        
        # Set new value
        set_config_value(key, request.value)
        
        return ConfigSetResponse(
            success=True,
            key=key,
            old_value=old_value,
            new_value=request.value,
            message=f"Configuration '{key}' updated successfully"
        )
    except Exception as e:
        return ConfigSetResponse(
            success=False,
            key=key,
            message=str(e)
        )


# =============================================================================
# Health/Doctor Routes
# =============================================================================

@api_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check."""
    try:
        from src.cli.doctor import (
            check_python_version,
            check_env_file,
            check_vault_path,
            check_llm_provider,
            check_ollama,
            check_embedding_model,
        )
        
        checks = []
        summary = {"ok": 0, "warning": 0, "error": 0}
        
        # Run all diagnostic checks
        check_functions = [
            check_python_version,
            check_env_file,
            check_vault_path,
            check_llm_provider,
            check_ollama,
            check_embedding_model,
        ]
        
        for check_fn in check_functions:
            try:
                result = check_fn()
                checks.append(HealthCheck(
                    name=result.name,
                    status=result.status,
                    message=result.message,
                    fix_hint=result.fix_hint,
                ))
                summary[result.status] = summary.get(result.status, 0) + 1
            except Exception as e:
                checks.append(HealthCheck(
                    name=check_fn.__name__,
                    status="error",
                    message=str(e),
                ))
                summary["error"] += 1
        
        # Determine overall status
        overall_status = "ok"
        if summary["error"] > 0:
            overall_status = "error"
        elif summary["warning"] > 0:
            overall_status = "warning"
        
        return HealthResponse(
            status=overall_status,
            version="0.1.0",
            checks=checks,
            summary=summary,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Prompts Routes
# =============================================================================

@api_router.get("/prompts", response_model=PromptsInfoResponse)
async def get_prompts_info():
    """Get prompt configuration info."""
    try:
        from src.prompt_manager import PromptManager
        
        config = ANAConfig()
        pm = PromptManager(config)
        
        info = pm.get_prompt_info()
        prompts = []
        
        for prompt_type, details in info.items():
            prompts.append(PromptInfo(
                prompt_type=prompt_type,
                source=details.get("source", "default"),
                path=details.get("path"),
                is_valid=True,
            ))
        
        custom_dir = config.get_custom_prompts_dir()
        
        return PromptsInfoResponse(
            prompts=prompts,
            custom_prompts_dir=str(custom_dir) if custom_dir else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/prompts/validate", response_model=PromptsValidateResponse)
async def validate_prompts():
    """Validate custom prompt files."""
    try:
        from src.prompt_manager import PromptManager
        
        config = ANAConfig()
        pm = PromptManager(config)
        
        validation_results = pm.validate_all_prompts()
        results = []
        all_valid = True
        
        for prompt_type, (is_valid, message) in validation_results.items():
            info = pm.get_prompt_info().get(prompt_type, {})
            results.append(PromptInfo(
                prompt_type=prompt_type,
                source=info.get("source", "default"),
                path=info.get("path"),
                is_valid=is_valid,
                validation_message=message if not is_valid else None,
            ))
            if not is_valid:
                all_valid = False
        
        return PromptsValidateResponse(
            all_valid=all_valid,
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Backlink Routes
# =============================================================================

# Store backlink suggestions per session
_backlink_sessions: dict[str, list] = {}


@api_router.post("/backlinks/suggest", response_model=BacklinkSuggestResponse)
async def suggest_backlinks(request: BacklinkSuggestRequest):
    """Suggest backlinks for a note."""
    try:
        from src.backlink_analyzer import BacklinkAnalyzer
        from src.schemas import DraftNote as SchemaDraftNote
        from src.llm_config import get_llm
        
        config = ANAConfig()
        vault_path = config.get_vault_path()
        vault_scanner = VaultScanner(vault_path)
        llm = get_llm(config)
        
        analyzer = BacklinkAnalyzer(
            vault_scanner=vault_scanner,
            llm=llm,
            auto_apply=False,
        )
        
        # Create a DraftNote from the request
        draft_note = SchemaDraftNote(
            title=request.title,
            content=request.content,
            frontmatter={"tags": request.tags} if request.tags else {},
        )
        
        # Find backlink opportunities
        suggestions = analyzer.find_backlink_opportunities(
            new_note=draft_note,
            max_notes_to_scan=request.max_notes_to_scan,
        )
        
        # Store suggestions in session for later application
        session_id = str(uuid.uuid4())
        _backlink_sessions[session_id] = {
            "suggestions": suggestions,
            "analyzer": analyzer,
        }
        
        # Convert to response format
        response_suggestions = []
        for s in suggestions:
            response_suggestions.append(BacklinkSuggestionItem(
                source_path=s.source_path,
                source_title=getattr(s, "source_title", Path(s.source_path).stem),
                matched_text=s.matched_text,
                line_number=s.line_number,
                confidence=s.confidence,
                reason=s.reason,
            ))
        
        return BacklinkSuggestResponse(
            suggestions=response_suggestions,
            notes_scanned=request.max_notes_to_scan,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/backlinks/apply")
async def apply_backlinks(request: BacklinkApplyRequest):
    """Apply backlink suggestions."""
    session = _backlink_sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Backlink session not found")
    
    try:
        analyzer = session["analyzer"]
        all_suggestions = session["suggestions"]
        
        # Filter to selected suggestions
        selected = [
            all_suggestions[i] 
            for i in request.suggestion_indices 
            if i < len(all_suggestions)
        ]
        
        # Apply backlinks
        modified_files = analyzer.apply_backlinks(selected)
        
        # Clean up session
        del _backlink_sessions[request.session_id]
        
        return {
            "success": True,
            "applied_count": len(selected),
            "modified_files": [str(f) for f in modified_files],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Server Runner
# =============================================================================

def run_server(host: str = "127.0.0.1", port: int = 8765):
    """Run the API server."""
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
