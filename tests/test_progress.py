"""Tests for Progress tracking module."""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from src.progress import (
    ProcessingPhase,
    ProgressUpdate,
    ProgressTracker,
    SSEEventGenerator,
    PHASE_WEIGHTS,
    PHASE_MESSAGES,
)


class TestProcessingPhase:
    """Test ProcessingPhase enum."""
    
    def test_all_phases_have_weights(self):
        """Test that all phases have weight definitions."""
        for phase in ProcessingPhase:
            assert phase in PHASE_WEIGHTS
    
    def test_all_phases_have_messages(self):
        """Test that all phases have message definitions."""
        for phase in ProcessingPhase:
            assert phase in PHASE_MESSAGES
    
    def test_phase_string_values(self):
        """Test phase string values."""
        assert ProcessingPhase.INITIALIZING.value == "initializing"
        assert ProcessingPhase.ANALYZING.value == "analyzing"
        assert ProcessingPhase.COMPLETED.value == "completed"


class TestProgressUpdate:
    """Test ProgressUpdate dataclass."""
    
    def test_create_progress_update(self):
        """Test creating a progress update."""
        update = ProgressUpdate(
            phase=ProcessingPhase.ANALYZING,
            progress=0.5,
            overall_progress=0.25,
            message="분석 중...",
            detail="Step 1 of 2"
        )
        
        assert update.phase == ProcessingPhase.ANALYZING
        assert update.progress == 0.5
        assert update.overall_progress == 0.25
        assert update.message == "분석 중..."
        assert update.detail == "Step 1 of 2"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        update = ProgressUpdate(
            phase=ProcessingPhase.SYNTHESIZING,
            progress=0.7,
            overall_progress=0.8,
            message="합성 중..."
        )
        
        d = update.to_dict()
        
        assert d["type"] == "progress"
        assert d["phase"] == "synthesizing"
        assert d["progress"] == 0.7
        assert d["overall_progress"] == 0.8
        assert d["message"] == "합성 중..."
    
    def test_to_sse(self):
        """Test converting to SSE format."""
        update = ProgressUpdate(
            phase=ProcessingPhase.COMPLETED,
            progress=1.0,
            overall_progress=1.0,
            message="완료!"
        )
        
        sse = update.to_sse()
        
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        assert '"type": "progress"' in sse


class TestProgressTracker:
    """Test ProgressTracker class."""
    
    def test_init(self):
        """Test tracker initialization."""
        tracker = ProgressTracker()
        
        assert tracker.current_phase == ProcessingPhase.INITIALIZING
        assert tracker.phase_progress == 0.0
        assert tracker.completed_phases == []
        assert tracker.callback is None
        assert tracker.async_callback is None
    
    def test_init_with_callback(self):
        """Test tracker with callback."""
        callback = MagicMock()
        tracker = ProgressTracker(callback=callback)
        
        assert tracker.callback is callback
    
    def test_update_returns_progress_update(self):
        """Test that update returns ProgressUpdate."""
        tracker = ProgressTracker()
        
        update = tracker.update(
            ProcessingPhase.ANALYZING,
            progress=0.5,
            message="Test"
        )
        
        assert isinstance(update, ProgressUpdate)
        assert update.phase == ProcessingPhase.ANALYZING
        assert update.progress == 0.5
    
    def test_update_invokes_callback(self):
        """Test that update invokes callback."""
        callback = MagicMock()
        tracker = ProgressTracker(callback=callback)
        
        tracker.update(ProcessingPhase.ANALYZING, 0.5, "Test")
        
        callback.assert_called_once()
        update = callback.call_args[0][0]
        assert isinstance(update, ProgressUpdate)
    
    def test_phase_transitions(self):
        """Test phase transition tracking."""
        tracker = ProgressTracker()
        
        tracker.update(ProcessingPhase.ANALYZING, 1.0)
        assert ProcessingPhase.INITIALIZING in tracker.completed_phases
        assert tracker.current_phase == ProcessingPhase.ANALYZING
        
        tracker.update(ProcessingPhase.SYNTHESIZING, 0.0)
        assert ProcessingPhase.ANALYZING in tracker.completed_phases
        assert tracker.current_phase == ProcessingPhase.SYNTHESIZING
    
    def test_overall_progress_calculation(self):
        """Test overall progress calculation."""
        tracker = ProgressTracker()
        
        # Start at initializing
        update = tracker.update(ProcessingPhase.INITIALIZING, 1.0)
        assert update.overall_progress >= 0.0
        
        # Move to analyzing
        update = tracker.update(ProcessingPhase.ANALYZING, 0.5)
        assert update.overall_progress > tracker.phase_progress * PHASE_WEIGHTS[ProcessingPhase.ANALYZING]
    
    def test_complete(self):
        """Test complete method."""
        tracker = ProgressTracker()
        
        update = tracker.complete()
        
        assert update.phase == ProcessingPhase.COMPLETED
        assert update.progress == 1.0
    
    def test_error(self):
        """Test error method."""
        tracker = ProgressTracker()
        
        update = tracker.error("Something went wrong")
        
        assert update.phase == ProcessingPhase.ERROR
        assert tracker.error_message == "Something went wrong"
        assert update.detail == "Something went wrong"
    
    def test_get_current_state(self):
        """Test getting current state."""
        tracker = ProgressTracker()
        tracker.update(ProcessingPhase.ANALYZING, 0.5)
        
        state = tracker.get_current_state()
        
        assert state["phase"] == "analyzing"
        assert state["phase_progress"] == 0.5
        assert "overall_progress" in state
        assert "completed_phases" in state
    
    def test_default_messages(self):
        """Test using default phase messages."""
        tracker = ProgressTracker()
        
        update = tracker.update(ProcessingPhase.ANALYZING, 0.5)
        
        assert update.message == PHASE_MESSAGES[ProcessingPhase.ANALYZING]


class TestSSEEventGenerator:
    """Test SSEEventGenerator class."""
    
    @pytest.mark.asyncio
    async def test_send_update(self):
        """Test sending progress updates."""
        sse = SSEEventGenerator()
        
        update = ProgressUpdate(
            phase=ProcessingPhase.ANALYZING,
            progress=0.5,
            overall_progress=0.3,
            message="Test"
        )
        
        await sse.send_update(update)
        
        # Check queue has the event
        event = await sse.queue.get()
        assert "data:" in event
        assert "analyzing" in event
    
    @pytest.mark.asyncio
    async def test_send_complete(self):
        """Test sending complete event."""
        sse = SSEEventGenerator()
        
        await sse.send_complete({"session_id": "123", "status": "completed"})
        
        event = await sse.queue.get()
        assert "complete" in event
        assert "123" in event
    
    @pytest.mark.asyncio
    async def test_send_error(self):
        """Test sending error event."""
        sse = SSEEventGenerator()
        
        await sse.send_error("Something failed")
        
        event = await sse.queue.get()
        assert "error" in event
        assert "Something failed" in event
    
    @pytest.mark.asyncio
    async def test_send_heartbeat(self):
        """Test sending heartbeat."""
        sse = SSEEventGenerator()
        
        await sse.send_heartbeat()
        
        event = await sse.queue.get()
        assert "heartbeat" in event
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the generator."""
        sse = SSEEventGenerator()
        
        sse.close()
        
        assert sse._closed is True
        
        # Sending after close should not add to queue
        await sse.send_heartbeat()
        assert sse.queue.empty()
    
    @pytest.mark.asyncio
    async def test_generate_yields_events(self):
        """Test event generation."""
        sse = SSEEventGenerator(heartbeat_interval=0.1)
        
        # Add some events first
        await sse.send_heartbeat()
        await sse.send_heartbeat()
        
        events = []
        
        # Get events with a short timeout
        try:
            event1 = await asyncio.wait_for(sse.queue.get(), timeout=0.5)
            events.append(event1)
            event2 = await asyncio.wait_for(sse.queue.get(), timeout=0.5)
            events.append(event2)
        except asyncio.TimeoutError:
            pass
        
        assert len(events) == 2
        assert all("heartbeat" in e for e in events)


class TestProgressIntegration:
    """Integration tests for progress tracking."""
    
    def test_full_processing_flow(self):
        """Test a full processing flow with progress tracking."""
        updates = []
        
        def capture_update(update):
            updates.append(update)
        
        tracker = ProgressTracker(callback=capture_update)
        
        # Simulate processing phases
        tracker.update(ProcessingPhase.INITIALIZING, 1.0, "초기화 완료")
        tracker.update(ProcessingPhase.ANALYZING, 0.0, "분석 시작")
        tracker.update(ProcessingPhase.ANALYZING, 0.5, "분석 중...")
        tracker.update(ProcessingPhase.ANALYZING, 1.0, "분석 완료")
        tracker.update(ProcessingPhase.SYNTHESIZING, 0.5, "합성 중...")
        tracker.complete()
        
        assert len(updates) == 6
        assert updates[0].phase == ProcessingPhase.INITIALIZING
        assert updates[-1].phase == ProcessingPhase.COMPLETED
        
        # Verify progress is increasing
        overall_progress = [u.overall_progress for u in updates]
        for i in range(1, len(overall_progress)):
            assert overall_progress[i] >= overall_progress[i - 1]
