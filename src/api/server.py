"""ANA API Server.

FastAPI-based REST API server for Obsidian plugin integration.
"""

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
)
from src.config import ANAConfig


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
    
    # Register routes
    app.include_router(api_router)
    
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
# Server Runner
# =============================================================================

def run_server(host: str = "127.0.0.1", port: int = 8765):
    """Run the API server."""
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
