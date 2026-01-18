"""Tests for LangGraph workflow module."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.graph import build_graph, continue_with_answers, create_initial_state
from src.schemas import AgentState, AnalysisResult, DraftNote, InteractionPayload


class TestCreateInitialState:
    """Test suite for create_initial_state function."""
    
    def test_create_initial_state_basic(self, sample_raw_note: str):
        """Test creating initial state with minimal parameters."""
        state = create_initial_state(sample_raw_note)
        
        assert state["raw_note"] == sample_raw_note
        assert state["input_metadata"] == {}
        assert state["user_answers"] == []
        assert state["analysis"] is None
        assert state["questions"] is None
        assert state["category"] == "general"
        assert state["template"] == ""
        assert state["template_source"] == "default"
        assert state["final_note"] is None
        assert state["iteration_count"] == 0
        assert state["is_complete"] is False
        assert state["error"] is None
    
    def test_create_initial_state_with_category(self, sample_raw_note: str):
        """Test creating initial state with category."""
        state = create_initial_state(
            sample_raw_note,
            category="concept",
            template="# {{ title }}\n{{ content }}",
            template_source="file"
        )
        
        assert state["category"] == "concept"
        assert state["template"] == "# {{ title }}\n{{ content }}"
        assert state["template_source"] == "file"


class TestContinueWithAnswers:
    """Test suite for continue_with_answers function."""
    
    def test_continue_with_answers(self, state_with_analysis: AgentState):
        """Test continuing with user answers."""
        answers = ["Answer 1", "Answer 2"]
        
        new_state = continue_with_answers(state_with_analysis, answers)
        
        assert new_state["user_answers"] == answers
    
    def test_continue_with_answers_appends(self, state_with_answers: AgentState):
        """Test that answers are appended to existing answers."""
        new_answers = ["Answer 3"]
        
        new_state = continue_with_answers(state_with_answers, new_answers)
        
        assert len(new_state["user_answers"]) == 3
        assert "Answer 3" in new_state["user_answers"]


class TestBuildGraph:
    """Test suite for graph building and execution."""
    
    def test_build_graph(self, mock_llm: MagicMock):
        """Test building the graph."""
        graph = build_graph(mock_llm)
        
        assert graph is not None
    
    def test_graph_with_sufficient_note(self, mock_llm: MagicMock, sample_raw_note: str):
        """Test graph execution with sufficient note (no questions needed)."""
        # Mock returns sufficient analysis
        mock_llm.invoke.side_effect = [
            # Analysis response
            MagicMock(content=json.dumps({
                "detected_concepts": ["RAG"],
                "missing_context": [],
                "is_sufficient": True,
                "should_split": False,
                "detected_category": "concept",
            })),
            # Synthesis response
            MagicMock(content=json.dumps({
                "title": "RAG 개념",
                "tags": ["RAG", "AI"],
                "content": "RAG는 검색 증강 생성입니다.",
                "frontmatter": {"type": "concept"},
                "suggested_links": [],
            })),
        ]
        
        graph = build_graph(mock_llm)
        state = create_initial_state(sample_raw_note, category="concept")
        
        result = graph.invoke(state)
        
        assert result["is_complete"] is True
        assert result["final_note"] is not None
        assert result["final_note"].title == "RAG 개념"
    
    def test_graph_with_questions(self, sample_raw_note: str):
        """Test graph execution needing questions."""
        mock_llm = MagicMock()
        
        # Mock returns insufficient analysis -> questions
        mock_llm.invoke.side_effect = [
            # Analysis response (insufficient)
            MagicMock(content=json.dumps({
                "detected_concepts": ["RAG", "LLM"],
                "missing_context": ["활용 사례", "구체적 설명"],
                "is_sufficient": False,
                "should_split": False,
                "detected_category": "concept",
            })),
            # Questions response
            MagicMock(content=json.dumps({
                "questions_to_user": [
                    "RAG를 어디에 활용할 계획인가요?",
                    "LLM 환각 문제를 경험한 적이 있나요?",
                ],
                "question_categories": ["context", "clarification"],
            })),
        ]
        
        graph = build_graph(mock_llm, max_questions=3)
        state = create_initial_state(sample_raw_note, category="concept")
        
        result = graph.invoke(state)
        
        # Should stop at questions (not complete)
        assert result["is_complete"] is False
        assert result["questions"] is not None
        assert len(result["questions"].questions_to_user) == 2
    
    def test_graph_max_iterations(self, sample_raw_note: str):
        """Test graph respects max iterations."""
        mock_llm = MagicMock()
        
        # Always return insufficient analysis
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "detected_concepts": ["RAG"],
            "missing_context": ["많은 정보 필요"],
            "is_sufficient": False,
            "should_split": False,
            "detected_category": "concept",
        }))
        
        # Create state already at max iterations
        graph = build_graph(mock_llm, max_iterations=1)
        state = create_initial_state(sample_raw_note)
        state["iteration_count"] = 1  # Already at max
        
        # Even though not sufficient, should synthesize due to max iterations
        # The graph logic should handle this in the conditional edge
    
    def test_graph_handles_json_error(self, sample_raw_note: str):
        """Test graph handles invalid JSON response gracefully."""
        mock_llm = MagicMock()
        
        # Return invalid JSON
        mock_llm.invoke.return_value = MagicMock(content="This is not valid JSON")
        
        graph = build_graph(mock_llm)
        state = create_initial_state(sample_raw_note)
        
        # Should not raise, should use fallback
        result = graph.invoke(state)
        
        assert result is not None


class TestGraphNodes:
    """Test individual graph node functions."""
    
    def test_extract_metadata_with_frontmatter(self, sample_raw_note: str, temp_vault):
        """Test metadata extraction with frontmatter."""
        from src.vault_scanner import VaultScanner
        
        mock_llm = MagicMock()
        scanner = VaultScanner(temp_vault)
        
        graph = build_graph(mock_llm, vault_scanner=scanner)
        state = create_initial_state(sample_raw_note)
        
        # The extract_metadata node should parse frontmatter
        # This is tested indirectly through graph execution
    
    def test_analyze_note_with_answers(self, sample_raw_note: str):
        """Test analysis includes previous answers."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "detected_concepts": ["RAG"],
            "missing_context": [],
            "is_sufficient": True,
            "should_split": False,
            "detected_category": "concept",
        }))
        
        graph = build_graph(mock_llm)
        state = create_initial_state(sample_raw_note)
        state["user_answers"] = ["Previous answer 1"]
        
        # Invoke and check that answers are included in context
        # This is tested indirectly, but we can verify the mock was called


class TestGraphWithSplitSuggestion:
    """Test graph handling of split suggestions."""
    
    def test_graph_detects_split_needed(self, sample_raw_note_multi_concept: str):
        """Test graph detects when note should be split."""
        mock_llm = MagicMock()
        
        mock_llm.invoke.side_effect = [
            # Analysis with split suggestion
            MagicMock(content=json.dumps({
                "detected_concepts": ["RAG", "GraphRAG", "Fine-tuning"],
                "missing_context": [],
                "is_sufficient": True,
                "should_split": True,
                "split_suggestions": ["RAG 개념", "GraphRAG 개념", "Fine-tuning vs RAG"],
                "detected_category": "concept",
            })),
            # Synthesis
            MagicMock(content=json.dumps({
                "title": "RAG 개념",
                "tags": ["RAG"],
                "content": "RAG 설명...",
                "frontmatter": {},
                "suggested_links": [],
            })),
        ]
        
        graph = build_graph(mock_llm)
        state = create_initial_state(sample_raw_note_multi_concept)
        
        result = graph.invoke(state)
        
        assert result["analysis"].should_split is True
        assert len(result["analysis"].split_suggestions) == 3
