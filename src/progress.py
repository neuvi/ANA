"""Progress Tracking Module.

Provides real-time progress tracking for long note processing operations.
Supports SSE (Server-Sent Events) streaming for progress updates.
"""

import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Callable, Optional, Any
import json

from src.logging_config import get_logger

logger = get_logger("progress")


# =============================================================================
# Processing Phases
# =============================================================================

class ProcessingPhase(str, Enum):
    """노트 처리 단계."""
    
    INITIALIZING = "initializing"
    EXTRACTING_METADATA = "extracting_metadata"
    ANALYZING = "analyzing"
    CLASSIFYING = "classifying"
    GENERATING_QUESTIONS = "generating_questions"
    AWAITING_ANSWERS = "awaiting_answers"
    SYNTHESIZING = "synthesizing"
    SUGGESTING_TAGS = "suggesting_tags"
    FINDING_LINKS = "finding_links"
    SAVING = "saving"
    COMPLETED = "completed"
    ERROR = "error"


# Phase weights for overall progress calculation
PHASE_WEIGHTS = {
    ProcessingPhase.INITIALIZING: 0.02,
    ProcessingPhase.EXTRACTING_METADATA: 0.05,
    ProcessingPhase.ANALYZING: 0.15,
    ProcessingPhase.CLASSIFYING: 0.08,
    ProcessingPhase.GENERATING_QUESTIONS: 0.15,
    ProcessingPhase.AWAITING_ANSWERS: 0.05,  # User interaction
    ProcessingPhase.SYNTHESIZING: 0.25,
    ProcessingPhase.SUGGESTING_TAGS: 0.10,
    ProcessingPhase.FINDING_LINKS: 0.10,
    ProcessingPhase.SAVING: 0.05,
    ProcessingPhase.COMPLETED: 0.0,
    ProcessingPhase.ERROR: 0.0,
}

# Phase display messages (Korean)
PHASE_MESSAGES = {
    ProcessingPhase.INITIALIZING: "초기화 중...",
    ProcessingPhase.EXTRACTING_METADATA: "메타데이터 추출 중...",
    ProcessingPhase.ANALYZING: "노트 분석 중...",
    ProcessingPhase.CLASSIFYING: "카테고리 분류 중...",
    ProcessingPhase.GENERATING_QUESTIONS: "질문 생성 중...",
    ProcessingPhase.AWAITING_ANSWERS: "답변 대기 중...",
    ProcessingPhase.SYNTHESIZING: "노트 합성 중...",
    ProcessingPhase.SUGGESTING_TAGS: "태그 제안 중...",
    ProcessingPhase.FINDING_LINKS: "관련 노트 검색 중...",
    ProcessingPhase.SAVING: "저장 중...",
    ProcessingPhase.COMPLETED: "처리 완료!",
    ProcessingPhase.ERROR: "오류 발생",
}


# =============================================================================
# Progress Update Data
# =============================================================================

@dataclass
class ProgressUpdate:
    """진행률 업데이트 데이터."""
    
    phase: ProcessingPhase
    progress: float  # 0.0 ~ 1.0 (phase-local progress)
    overall_progress: float  # 0.0 ~ 1.0 (overall progress)
    message: str
    detail: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": "progress",
            "phase": self.phase.value,
            "progress": round(self.progress, 2),
            "overall_progress": round(self.overall_progress, 2),
            "message": self.message,
            "detail": self.detail,
        }
    
    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


# =============================================================================
# Progress Tracker
# =============================================================================

class ProgressTracker:
    """노트 처리 진행률 추적기.
    
    Features:
    - 단계별 진행률 추적
    - 전체 진행률 계산
    - 동기/비동기 콜백 지원
    - SSE 이벤트 생성
    """
    
    def __init__(
        self,
        callback: Optional[Callable[[ProgressUpdate], None]] = None,
        async_callback: Optional[Callable[[ProgressUpdate], Any]] = None,
    ):
        """Initialize progress tracker.
        
        Args:
            callback: Synchronous callback for progress updates
            async_callback: Asynchronous callback for progress updates
        """
        self.callback = callback
        self.async_callback = async_callback
        
        self.current_phase = ProcessingPhase.INITIALIZING
        self.phase_progress = 0.0
        self.completed_phases: list[ProcessingPhase] = []
        
        # For error tracking
        self.error_message: Optional[str] = None
    
    def update(
        self,
        phase: ProcessingPhase,
        progress: float = 0.0,
        message: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> ProgressUpdate:
        """진행률 업데이트.
        
        Args:
            phase: 현재 처리 단계
            progress: 단계 내 진행률 (0.0~1.0)
            message: 표시할 메시지 (None이면 기본 메시지)
            detail: 상세 정보
            
        Returns:
            생성된 ProgressUpdate
        """
        # Update current state
        if self.current_phase != phase:
            if self.current_phase not in self.completed_phases:
                self.completed_phases.append(self.current_phase)
            self.current_phase = phase
        
        self.phase_progress = max(0.0, min(1.0, progress))
        
        # Calculate overall progress
        overall = self._calculate_overall_progress()
        
        # Get message
        if message is None:
            message = PHASE_MESSAGES.get(phase, str(phase))
        
        # Create update
        update = ProgressUpdate(
            phase=phase,
            progress=self.phase_progress,
            overall_progress=overall,
            message=message,
            detail=detail,
        )
        
        # Invoke callbacks
        if self.callback:
            try:
                self.callback(update)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        
        if self.async_callback:
            try:
                # Check if we're in an async context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._invoke_async_callback(update))
            except RuntimeError:
                pass  # No event loop available
        
        logger.debug(f"Progress: {phase.value} - {progress:.0%} (overall: {overall:.0%})")
        
        return update
    
    async def _invoke_async_callback(self, update: ProgressUpdate):
        """Invoke async callback."""
        if self.async_callback:
            try:
                result = self.async_callback(update)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Async progress callback failed: {e}")
    
    def _calculate_overall_progress(self) -> float:
        """전체 진행률 계산."""
        total = 0.0
        
        # Add completed phases
        for phase in self.completed_phases:
            total += PHASE_WEIGHTS.get(phase, 0.0)
        
        # Add current phase progress
        current_weight = PHASE_WEIGHTS.get(self.current_phase, 0.0)
        total += current_weight * self.phase_progress
        
        return min(1.0, total)
    
    def complete(self, message: Optional[str] = None) -> ProgressUpdate:
        """처리 완료."""
        return self.update(
            ProcessingPhase.COMPLETED,
            progress=1.0,
            message=message or PHASE_MESSAGES[ProcessingPhase.COMPLETED]
        )
    
    def error(self, error_message: str) -> ProgressUpdate:
        """오류 발생."""
        self.error_message = error_message
        return self.update(
            ProcessingPhase.ERROR,
            progress=0.0,
            message=PHASE_MESSAGES[ProcessingPhase.ERROR],
            detail=error_message
        )
    
    def get_current_state(self) -> dict[str, Any]:
        """현재 상태 반환."""
        return {
            "phase": self.current_phase.value,
            "phase_progress": self.phase_progress,
            "overall_progress": self._calculate_overall_progress(),
            "completed_phases": [p.value for p in self.completed_phases],
            "error": self.error_message,
        }


# =============================================================================
# SSE Event Generator
# =============================================================================

class SSEEventGenerator:
    """SSE 이벤트 생성기.
    
    FastAPI StreamingResponse와 함께 사용하여
    실시간 진행률을 클라이언트에 전송.
    """
    
    def __init__(self, heartbeat_interval: float = 5.0):
        """Initialize SSE generator.
        
        Args:
            heartbeat_interval: Heartbeat 전송 간격 (초)
        """
        self.heartbeat_interval = heartbeat_interval
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._closed = False
    
    async def send_update(self, update: ProgressUpdate):
        """진행률 업데이트 전송."""
        if not self._closed:
            await self.queue.put(update.to_sse())
    
    async def send_heartbeat(self):
        """Heartbeat 전송."""
        if not self._closed:
            await self.queue.put('data: {"type": "heartbeat"}\n\n')
    
    async def send_complete(self, result: Any):
        """완료 이벤트 전송."""
        if not self._closed:
            event = {
                "type": "complete",
                "result": result if isinstance(result, dict) else str(result)
            }
            await self.queue.put(f"data: {json.dumps(event)}\n\n")
    
    async def send_error(self, error: str):
        """오류 이벤트 전송."""
        if not self._closed:
            event = {"type": "error", "message": error}
            await self.queue.put(f"data: {json.dumps(event)}\n\n")
    
    def close(self):
        """생성기 종료."""
        self._closed = True
    
    async def generate(self):
        """SSE 이벤트 스트림 생성.
        
        Yields:
            SSE 형식의 이벤트 문자열
        """
        while not self._closed:
            try:
                # Wait for next event with timeout (for heartbeat)
                event = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=self.heartbeat_interval
                )
                yield event
            except asyncio.TimeoutError:
                # Send heartbeat on timeout
                yield 'data: {"type": "heartbeat"}\n\n'
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SSE generator error: {e}")
                yield f'data: {{"type": "error", "message": "{str(e)}"}}\n\n'
                break


# =============================================================================
# Helper Functions
# =============================================================================

def create_progress_callback(queue: asyncio.Queue) -> Callable[[ProgressUpdate], None]:
    """Queue 기반 진행률 콜백 생성.
    
    Args:
        queue: 업데이트를 넣을 asyncio.Queue
        
    Returns:
        콜백 함수
    """
    def callback(update: ProgressUpdate):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(queue.put(update))
        except RuntimeError:
            pass
    
    return callback
