"""Tests for API Server Module.

Tests for FastAPI endpoints in src/api/server.py.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_endpoint_returns_ok(self, test_client):
        """Test that /api/health returns OK status."""
        response = test_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
    
    def test_health_endpoint_includes_config(self, test_client):
        """Test that health endpoint includes config info."""
        response = test_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "config" in data


class TestProcessEndpoint:
    """Test note processing endpoint."""
    
    def test_process_endpoint_with_valid_note(self, test_client, mock_agent):
        """Test processing a valid note."""
        response = test_client.post(
            "/api/process",
            json={"raw_note": "This is a test note about Python programming."}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_process_endpoint_with_empty_note(self, test_client):
        """Test processing an empty note returns error."""
        response = test_client.post(
            "/api/process",
            json={"raw_note": ""}
        )
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    def test_process_endpoint_with_frontmatter(self, test_client, mock_agent):
        """Test processing note with frontmatter."""
        response = test_client.post(
            "/api/process",
            json={
                "raw_note": "Test content",
                "frontmatter": {"tags": ["test"], "category": "note"}
            }
        )
        
        assert response.status_code == 200


class TestAnswerEndpoint:
    """Test question answering endpoint."""
    
    def test_answer_endpoint_with_answers(self, test_client, mock_agent_with_questions):
        """Test answering questions."""
        # First process to get questions
        test_client.post(
            "/api/process",
            json={"raw_note": "Test note"}
        )
        
        # Then answer questions
        response = test_client.post(
            "/api/answer",
            json={"answers": ["Answer 1", "Answer 2"]}
        )
        
        assert response.status_code == 200
    
    def test_answer_endpoint_without_session(self, test_client):
        """Test answering without active session."""
        response = test_client.post(
            "/api/answer",
            json={"answers": ["Answer 1"]}
        )
        
        # Should return error when no session exists
        assert response.status_code in [400, 404]


class TestConfigEndpoint:
    """Test configuration endpoints."""
    
    def test_get_config_endpoint(self, test_client):
        """Test getting current configuration."""
        response = test_client.get("/api/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "llm_provider" in data
        assert "vault_path" in data


class TestSaveEndpoint:
    """Test note saving endpoint."""
    
    def test_save_endpoint_after_process(self, test_client, mock_agent):
        """Test saving note after processing."""
        # First process
        test_client.post(
            "/api/process",
            json={"raw_note": "Test note for saving"}
        )
        
        # Then save
        response = test_client.post("/api/save")
        
        assert response.status_code == 200
        data = response.json()
        assert "path" in data or "error" in data


# Fixtures for API tests
@pytest.fixture
def test_client():
    """Create test client for API."""
    try:
        from fastapi.testclient import TestClient
        from src.api.server import app
        
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI test client not available")


@pytest.fixture
def mock_agent():
    """Mock agent for API tests."""
    with patch("src.api.server.agent") as mock:
        mock.process.return_value = MagicMock(
            status="completed",
            draft_note=MagicMock(
                title="Test Note",
                content="Test content",
                tags=["test"],
                category="general"
            )
        )
        yield mock


@pytest.fixture
def mock_agent_with_questions():
    """Mock agent that returns questions."""
    with patch("src.api.server.agent") as mock:
        mock.process.return_value = MagicMock(
            status="needs_info",
            interaction=MagicMock(
                questions=["Question 1?", "Question 2?"]
            )
        )
        mock.answer_questions.return_value = MagicMock(
            status="completed",
            draft_note=MagicMock(title="Test", content="Content")
        )
        yield mock
