"""Pytest fixtures for ANA tests."""

import json
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from src.config import ANAConfig
from src.schemas import AgentState, AnalysisResult, DraftNote, InteractionPayload


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_raw_note() -> str:
    """Sample raw note for testing."""
    return """---
title: RAG 개념 정리
tags: [AI, RAG]
---

RAG는 검색 증강 생성(Retrieval-Augmented Generation)의 약자다.
LLM의 환각 문제를 해결하기 위해 사용한다.
외부 지식베이스에서 관련 정보를 검색하여 LLM에 제공한다.
"""


@pytest.fixture
def sample_raw_note_multi_concept() -> str:
    """Sample raw note with multiple concepts."""
    return """RAG는 검색 증강 생성이다. LLM 환각 해결용.
GraphRAG도 있더라. 그래프 기반 검색이라고 한다.
Fine-tuning과 RAG의 차이점도 있다."""


@pytest.fixture
def sample_frontmatter() -> dict[str, Any]:
    """Sample frontmatter metadata."""
    return {
        "title": "RAG 개념 정리",
        "tags": ["AI", "RAG"],
        "type": "concept",
        "created": "2026-01-18",
    }


@pytest.fixture
def sample_analysis_result() -> AnalysisResult:
    """Sample analysis result."""
    return AnalysisResult(
        detected_concepts=["RAG", "LLM 환각"],
        missing_context=["구체적인 활용 사례", "다른 기법과의 비교"],
        is_sufficient=False,
        should_split=False,
        split_suggestions=[],
        detected_category="concept",
        existing_metadata={"title": "RAG 개념 정리"},
    )


@pytest.fixture
def sample_analysis_result_sufficient() -> AnalysisResult:
    """Sample analysis result that is sufficient."""
    return AnalysisResult(
        detected_concepts=["RAG"],
        missing_context=[],
        is_sufficient=True,
        should_split=False,
        split_suggestions=[],
        detected_category="concept",
        existing_metadata={},
    )


@pytest.fixture
def sample_questions() -> InteractionPayload:
    """Sample questions payload."""
    return InteractionPayload(
        questions_to_user=[
            "RAG를 실제로 어떤 프로젝트에 적용할 계획인가요?",
            "LLM 환각 문제를 경험한 구체적인 사례가 있나요?",
        ],
        question_categories=["context", "clarification"],
    )


@pytest.fixture
def sample_draft_note() -> DraftNote:
    """Sample draft note."""
    return DraftNote(
        title="RAG(검색 증강 생성)의 개념과 활용",
        tags=["RAG", "LLM", "AI"],
        content="""## 개요

RAG(Retrieval-Augmented Generation)는 검색 증강 생성 기법입니다.

## 핵심 개념

- LLM의 환각(Hallucination) 문제 해결
- 외부 지식베이스 활용
- 실시간 정보 반영 가능
""",
        category="concept",
        frontmatter={
            "title": "RAG(검색 증강 생성)의 개념과 활용",
            "tags": ["RAG", "LLM", "AI"],
            "type": "concept",
            "created": "2026-01-18",
        },
        suggested_links=["LLM 환각 문제", "Vector Database"],
    )


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM for testing."""
    mock = MagicMock()
    
    # Default response
    mock.invoke.return_value = MagicMock(
        content=json.dumps({
            "detected_concepts": ["RAG"],
            "missing_context": [],
            "is_sufficient": True,
            "should_split": False,
            "split_suggestions": [],
            "detected_category": "concept",
        })
    )
    
    return mock


@pytest.fixture
def mock_llm_with_questions() -> MagicMock:
    """Mock LLM that returns questions."""
    mock = MagicMock()
    
    # First call returns analysis needing more info
    # Second call returns questions
    # Third call returns synthesis
    responses = [
        MagicMock(content=json.dumps({
            "detected_concepts": ["RAG", "LLM"],
            "missing_context": ["활용 사례"],
            "is_sufficient": False,
            "should_split": False,
            "detected_category": "concept",
        })),
        MagicMock(content=json.dumps({
            "questions_to_user": ["RAG를 어디에 활용할 계획인가요?"],
            "question_categories": ["context"],
        })),
        MagicMock(content=json.dumps({
            "title": "RAG 개념과 활용",
            "tags": ["RAG", "AI"],
            "content": "RAG는 검색 증강 생성입니다.",
            "frontmatter": {"type": "concept"},
            "suggested_links": [],
        })),
    ]
    mock.invoke.side_effect = responses
    
    return mock


# =============================================================================
# Temporary Vault Fixtures
# =============================================================================

@pytest.fixture
def temp_vault() -> Generator[Path, None, None]:
    """Create a temporary vault directory with sample notes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir)
        
        # Create sample notes
        notes_dir = vault_path / "notes"
        notes_dir.mkdir()
        
        # Note 1: RAG concept
        (notes_dir / "RAG 개념.md").write_text("""---
title: RAG 개념
tags: [AI, RAG]
type: concept
---

RAG는 Retrieval-Augmented Generation입니다.
""", encoding="utf-8")
        
        # Note 2: LLM basics
        (notes_dir / "LLM 기초.md").write_text("""---
title: LLM 기초
tags: [AI, LLM]
type: concept
---

LLM은 Large Language Model입니다.
""", encoding="utf-8")
        
        # Note 3: Vector DB
        (notes_dir / "Vector Database.md").write_text("""---
title: Vector Database
tags: [Database, AI]
type: tool
---

벡터 데이터베이스는 임베딩을 저장합니다.
""", encoding="utf-8")
        
        yield vault_path


@pytest.fixture
def temp_vault_empty() -> Generator[Path, None, None]:
    """Create an empty temporary vault directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir)
        yield vault_path


# =============================================================================
# Config Fixtures
# =============================================================================

@pytest.fixture
def test_config(temp_vault: Path) -> ANAConfig:
    """Test configuration with temporary vault."""
    return ANAConfig(
        vault_path=temp_vault,
        llm_provider="openai",
        llm_model="gpt-4o",
        max_questions=3,
        max_iterations=2,
        enable_note_linking=False,  # Disable for faster tests
    )


@pytest.fixture
def test_config_with_linking(temp_vault: Path) -> ANAConfig:
    """Test configuration with note linking enabled."""
    return ANAConfig(
        vault_path=temp_vault,
        llm_provider="openai",
        llm_model="gpt-4o",
        max_questions=3,
        max_iterations=2,
        enable_note_linking=True,
        embedding_enabled=False,  # Disable external calls
        rerank_enabled=False,
    )


# =============================================================================
# Agent State Fixtures
# =============================================================================

@pytest.fixture
def initial_state(sample_raw_note: str) -> AgentState:
    """Initial agent state for testing."""
    return AgentState(
        raw_note=sample_raw_note,
        input_metadata={},
        user_answers=[],
        analysis=None,
        questions=None,
        category="general",
        template="",
        template_source="default",
        final_note=None,
        iteration_count=0,
        is_complete=False,
        error=None,
    )


@pytest.fixture
def state_with_analysis(
    sample_raw_note: str,
    sample_analysis_result: AnalysisResult
) -> AgentState:
    """Agent state after analysis."""
    return AgentState(
        raw_note=sample_raw_note,
        input_metadata={"title": "RAG 개념 정리"},
        user_answers=[],
        analysis=sample_analysis_result,
        questions=None,
        category="concept",
        template="",
        template_source="default",
        final_note=None,
        iteration_count=0,
        is_complete=False,
        error=None,
    )


@pytest.fixture
def state_with_answers(
    sample_raw_note: str,
    sample_analysis_result: AnalysisResult,
    sample_questions: InteractionPayload
) -> AgentState:
    """Agent state with user answers."""
    return AgentState(
        raw_note=sample_raw_note,
        input_metadata={"title": "RAG 개념 정리"},
        user_answers=["실제 프로젝트에서 RAG를 적용 중입니다.", "정확도가 크게 향상되었습니다."],
        analysis=sample_analysis_result,
        questions=sample_questions,
        category="concept",
        template="",
        template_source="default",
        final_note=None,
        iteration_count=1,
        is_complete=False,
        error=None,
    )
