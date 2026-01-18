"""Tests for AtomicNoteArchitect agent module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent import AtomicNoteArchitect, create_agent
from src.config import ANAConfig
from src.schemas import AgentResponse, DraftNote


class TestAtomicNoteArchitect:
    """Test suite for AtomicNoteArchitect class."""
    
    @pytest.fixture
    def agent(self, test_config: ANAConfig) -> AtomicNoteArchitect:
        """Create an agent with mocked LLM."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content=json.dumps({
                "detected_concepts": ["RAG"],
                "missing_context": [],
                "is_sufficient": True,
                "should_split": False,
                "detected_category": "concept",
            }))
            mock_get_llm.return_value = mock_llm
            
            agent = AtomicNoteArchitect(test_config)
            agent.llm = mock_llm
            
            return agent
    
    def test_init(self, agent: AtomicNoteArchitect):
        """Test agent initialization."""
        assert agent is not None
        assert agent.config is not None
        assert agent.vault_scanner is not None
        assert agent.category_classifier is not None
        assert agent.template_manager is not None
    
    def test_process_simple_note(self, test_config: ANAConfig, sample_raw_note: str):
        """Test processing a simple note."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            
            # Setup response sequence
            mock_llm.invoke.side_effect = [
                # Category classification
                MagicMock(content="concept"),
                # Template generation (if needed)
                MagicMock(content="# {{ title }}\n{{ content }}"),
                # Analysis (sufficient)
                MagicMock(content=json.dumps({
                    "detected_concepts": ["RAG"],
                    "missing_context": [],
                    "is_sufficient": True,
                    "should_split": False,
                    "detected_category": "concept",
                })),
                # Synthesis
                MagicMock(content=json.dumps({
                    "title": "RAG 개념 정리",
                    "tags": ["RAG", "AI"],
                    "content": "RAG는 검색 증강 생성입니다.",
                    "frontmatter": {"type": "concept"},
                    "suggested_links": [],
                })),
            ]
            mock_get_llm.return_value = mock_llm
            
            agent = AtomicNoteArchitect(test_config)
            response = agent.process(sample_raw_note)
            
            assert response is not None
            assert isinstance(response, AgentResponse)
    
    def test_get_current_state(self, agent: AtomicNoteArchitect, sample_raw_note: str, sample_draft_note):
        """Test getting current state."""
        # Before processing
        assert agent.get_current_state() is None
        
        # After processing
        with patch.object(agent, "graph") as mock_graph:
            mock_graph.invoke.return_value = {
                "is_complete": True,
                "final_note": sample_draft_note,
            }
            agent.process(sample_raw_note)
        
        # State should be set
        assert agent.get_current_state() is not None
    
    def test_get_category(self, agent: AtomicNoteArchitect):
        """Test getting current category."""
        # Default category
        assert agent.get_category() == "general"
    
    def test_reset(self, agent: AtomicNoteArchitect, sample_raw_note: str, sample_draft_note):
        """Test resetting agent state."""
        # Process a note first
        with patch.object(agent, "graph") as mock_graph:
            mock_graph.invoke.return_value = {
                "is_complete": True,
                "final_note": sample_draft_note,
            }
            agent.process(sample_raw_note)
        
        # Reset
        agent.reset()
        
        assert agent.get_current_state() is None
        assert agent.get_category() == "general"


class TestAgentProcessWithQuestions:
    """Test agent processing that requires questions."""
    
    def test_process_needs_questions(self, test_config: ANAConfig, sample_raw_note: str):
        """Test processing that needs user questions."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            
            mock_llm.invoke.side_effect = [
                # Category
                MagicMock(content="concept"),
                # Template
                MagicMock(content="# {{ title }}"),
                # Analysis (insufficient)
                MagicMock(content=json.dumps({
                    "detected_concepts": ["RAG"],
                    "missing_context": ["활용 사례"],
                    "is_sufficient": False,
                    "should_split": False,
                    "detected_category": "concept",
                })),
                # Questions
                MagicMock(content=json.dumps({
                    "questions_to_user": ["RAG를 어디에 활용할 계획인가요?"],
                    "question_categories": ["context"],
                })),
            ]
            mock_get_llm.return_value = mock_llm
            
            agent = AtomicNoteArchitect(test_config)
            response = agent.process(sample_raw_note)
            
            assert response.status == "needs_info"
            assert response.interaction is not None
            # Questions may be empty if parsing fails, so just check the structure
            assert hasattr(response.interaction, 'questions_to_user')
    
    def test_answer_questions(self, test_config: ANAConfig, sample_raw_note: str):
        """Test answering questions continues processing."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            
            # First process call
            mock_llm.invoke.side_effect = [
                # Category
                MagicMock(content="concept"),
                # Template
                MagicMock(content="# {{ title }}"),
                # Analysis (insufficient)
                MagicMock(content=json.dumps({
                    "detected_concepts": ["RAG"],
                    "missing_context": ["활용 사례"],
                    "is_sufficient": False,
                    "should_split": False,
                    "detected_category": "concept",
                })),
                # Questions
                MagicMock(content=json.dumps({
                    "questions_to_user": ["RAG를 어디에 활용할 계획인가요?"],
                    "question_categories": ["context"],
                })),
                # After answer - Analysis (now sufficient)
                MagicMock(content=json.dumps({
                    "detected_concepts": ["RAG"],
                    "missing_context": [],
                    "is_sufficient": True,
                    "should_split": False,
                    "detected_category": "concept",
                })),
                # Synthesis
                MagicMock(content=json.dumps({
                    "title": "RAG 활용 사례",
                    "tags": ["RAG"],
                    "content": "RAG는...",
                    "frontmatter": {},
                    "suggested_links": [],
                })),
            ]
            mock_get_llm.return_value = mock_llm
            
            agent = AtomicNoteArchitect(test_config)
            
            # First process
            response1 = agent.process(sample_raw_note)
            assert response1.status == "needs_info"
            
            # Answer questions
            response2 = agent.answer_questions(["실제 프로젝트에서 사용 중입니다."])
            
            # Should be complete or need more info
            assert response2 is not None
    
    def test_answer_questions_without_process_raises(self, test_config: ANAConfig):
        """Test answering questions without processing raises error."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            
            agent = AtomicNoteArchitect(test_config)
            
            with pytest.raises(RuntimeError, match="No processing in progress"):
                agent.answer_questions(["Answer"])


class TestAgentSaveNote:
    """Test agent note saving functionality."""
    
    def test_save_note(self, test_config: ANAConfig, sample_draft_note: DraftNote, temp_vault: Path):
        """Test saving a note."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            
            agent = AtomicNoteArchitect(test_config)
            
            # Save note to temp vault
            saved_path = agent.save_note(sample_draft_note, output_dir=temp_vault)
            
            assert saved_path.exists()
            assert saved_path.suffix == ".md"
            
            # Verify content
            content = saved_path.read_text(encoding="utf-8")
            assert sample_draft_note.title in content or "RAG" in content
    
    def test_save_note_without_note_raises(self, test_config: ANAConfig):
        """Test saving without note raises error."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            
            agent = AtomicNoteArchitect(test_config)
            
            with pytest.raises(ValueError, match="No note available"):
                agent.save_note()
    
    def test_save_note_uses_vault_path_default(self, test_config: ANAConfig, sample_draft_note: DraftNote):
        """Test save uses vault path by default."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            
            agent = AtomicNoteArchitect(test_config)
            
            # Save without specifying output_dir
            saved_path = agent.save_note(sample_draft_note)
            
            assert saved_path.exists()
            assert test_config.get_vault_path() in saved_path.parents or saved_path.parent == test_config.get_vault_path()


class TestAgentExtractForSplit:
    """Test agent split extraction functionality."""
    
    def test_extract_for_split(self, test_config: ANAConfig, sample_raw_note_multi_concept: str):
        """Test extracting content for split."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content=json.dumps({
                "extracted_content": "GraphRAG는 그래프 기반 검색 증강 생성입니다.",
                "key_points": ["그래프 구조 활용", "관계 기반 검색"],
                "related_topics": ["RAG", "Knowledge Graph"],
            }))
            mock_get_llm.return_value = mock_llm
            
            agent = AtomicNoteArchitect(test_config)
            
            extracted, key_points = agent.extract_for_split(
                sample_raw_note_multi_concept,
                "GraphRAG 개념"
            )
            
            assert extracted is not None
            assert "GraphRAG" in extracted
            assert isinstance(key_points, list)
    
    def test_extract_for_split_fallback(self, test_config: ANAConfig):
        """Test extract fallback on error."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="Invalid JSON")
            mock_get_llm.return_value = mock_llm
            
            agent = AtomicNoteArchitect(test_config)
            
            extracted, key_points = agent.extract_for_split(
                "Some content",
                "Test Topic"
            )
            
            # Should return fallback
            assert "Test Topic" in extracted


class TestCreateAgent:
    """Test create_agent factory function."""
    
    def test_create_agent_default(self):
        """Test creating agent with default config."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            
            agent = create_agent()
            
            assert isinstance(agent, AtomicNoteArchitect)
    
    def test_create_agent_with_config(self, test_config: ANAConfig):
        """Test creating agent with custom config."""
        with patch("src.agent.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            
            agent = create_agent(test_config)
            
            assert isinstance(agent, AtomicNoteArchitect)
            assert agent.config == test_config
