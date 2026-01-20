"""LangGraph Workflow for ANA.

Implements the 3-phase pipeline using LangGraph:
1. Analysis (extract_metadata → classify_category → analyze_note)
2. Interrogation (generate_questions → await_user_input)
3. Synthesis (synthesize_note)
"""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.prompts import (
    format_metadata,
    format_qa_pairs,
)
from src.prompt_manager import PromptManager
from src.schemas import (
    AgentState,
    AnalysisResult,
    DraftNote,
    InteractionPayload,
)
from src.vault_scanner import VaultScanner


def build_graph(
    llm: BaseChatModel,
    vault_scanner: VaultScanner | None = None,
    max_questions: int = 5,
    max_iterations: int = 3,
    prompt_manager: PromptManager | None = None,
) -> StateGraph:
    """Build the LangGraph workflow.
    
    Args:
        llm: Language model to use
        vault_scanner: Optional vault scanner for metadata extraction
        max_questions: Maximum questions per round
        max_iterations: Maximum question rounds
        prompt_manager: Optional prompt manager for custom prompts
        
    Returns:
        Compiled StateGraph
    """
    # Initialize prompt manager if not provided
    if prompt_manager is None:
        prompt_manager = PromptManager()
    
    # =========================================================================
    # Node Functions
    # =========================================================================
    
    def extract_metadata(state: AgentState) -> dict[str, Any]:
        """Extract frontmatter metadata from raw note."""
        raw_note = state.get("raw_note", "")
        
        # Parse frontmatter if present
        metadata = {}
        if vault_scanner:
            metadata = vault_scanner.parse_frontmatter(raw_note) or {}
        
        return {
            "input_metadata": metadata,
        }
    
    def analyze_note(state: AgentState) -> dict[str, Any]:
        """Analyze the raw note for concepts, gaps, and sufficiency."""
        raw_note = state.get("raw_note", "")
        metadata = state.get("input_metadata", {})
        user_answers = state.get("user_answers", [])
        
        # Include previous answers in context if any
        context = raw_note
        if user_answers:
            context += "\n\n[Previous Answers]\n" + "\n".join(user_answers)
        
        prompt = prompt_manager.get_analysis_prompt().format(
            existing_metadata=format_metadata(metadata),
            raw_note=context
        )
        
        messages = [
            SystemMessage(content=prompt_manager.get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Parse JSON response
        try:
            # Clean up response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = json.loads(content)
            
            analysis = AnalysisResult(
                detected_concepts=data.get("detected_concepts", []),
                missing_context=data.get("missing_context", []),
                is_sufficient=data.get("is_sufficient", False),
                should_split=data.get("should_split", False),
                split_suggestions=data.get("split_suggestions", []),
                detected_category=data.get("detected_category"),
                existing_metadata=metadata,
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback to incomplete analysis
            analysis = AnalysisResult(
                detected_concepts=[],
                missing_context=["Unable to parse analysis"],
                is_sufficient=False,
                existing_metadata=metadata,
            )
        
        return {
            "analysis": analysis,
        }
    
    def generate_questions(state: AgentState) -> dict[str, Any]:
        """Generate questions to fill information gaps."""
        analysis = state.get("analysis")
        raw_note = state.get("raw_note", "")
        metadata = state.get("input_metadata", {})
        
        if not analysis:
            return {
                "questions": InteractionPayload(
                    questions_to_user=["Can you provide more context about this note?"],
                    question_categories=["context"]
                )
            }
        
        prompt = prompt_manager.get_interrogation_prompt().format(
            detected_concepts=", ".join(analysis.detected_concepts),
            missing_context=", ".join(analysis.missing_context),
            detected_category=analysis.detected_category or "unknown",
            existing_metadata=format_metadata(metadata),
            raw_note=raw_note,
            max_questions=max_questions
        )
        
        messages = [
            SystemMessage(content=prompt_manager.get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Parse JSON response
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = json.loads(content)
            
            questions_list = data.get("questions_to_user", [])[:max_questions]
            categories = data.get("question_categories", [])
            
            # Ensure categories match questions length
            while len(categories) < len(questions_list):
                categories.append("context")
            
            questions = InteractionPayload(
                questions_to_user=questions_list,
                question_categories=categories[:len(questions_list)]
            )
        except (json.JSONDecodeError, KeyError):
            questions = InteractionPayload(
                questions_to_user=["Could you provide more details about this topic?"],
                question_categories=["context"]
            )
        
        return {
            "questions": questions,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    
    def synthesize_note(state: AgentState) -> dict[str, Any]:
        """Synthesize the final atomic note."""
        raw_note = state.get("raw_note", "")
        metadata = state.get("input_metadata", {})
        user_answers = state.get("user_answers", [])
        questions = state.get("questions")
        category = state.get("category", "general")
        template = state.get("template", "")
        
        # Format Q&A pairs
        qa_pairs = ""
        if questions and user_answers:
            qa_pairs = format_qa_pairs(
                questions.questions_to_user,
                user_answers
            )
        
        prompt = prompt_manager.get_synthesis_prompt().format(
            raw_note=raw_note,
            existing_metadata=format_metadata(metadata),
            qa_pairs=qa_pairs or "No additional questions were asked.",
            category=category,
            template=template or "Use default Obsidian format."
        )
        
        messages = [
            SystemMessage(content=prompt_manager.get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        
        # Parse JSON response
        try:
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = json.loads(content)
            
            final_note = DraftNote(
                title=data.get("title", "Untitled Note"),
                tags=data.get("tags", []),
                content=data.get("content", raw_note),
                category=category,
                frontmatter=data.get("frontmatter", {}),
                suggested_links=data.get("suggested_links", [])
            )
        except (json.JSONDecodeError, KeyError):
            # Fallback to basic note
            final_note = DraftNote(
                title="Untitled Note",
                tags=[],
                content=raw_note,
                category=category,
                frontmatter=metadata,
                suggested_links=[]
            )
        
        return {
            "final_note": final_note,
            "is_complete": True,
        }
    
    # =========================================================================
    # Conditional Edge Functions
    # =========================================================================
    
    def should_ask_questions(state: AgentState) -> str:
        """Determine if we should ask questions or synthesize."""
        analysis = state.get("analysis")
        iteration_count = state.get("iteration_count", 0)
        
        # If no analysis, try to ask questions
        if not analysis:
            if iteration_count >= max_iterations:
                return "synthesize"
            return "ask_questions"
        
        # If sufficient or max iterations reached, synthesize
        if analysis.is_sufficient:
            return "synthesize"
        
        if iteration_count >= max_iterations:
            return "synthesize"
        
        return "ask_questions"
    
    # =========================================================================
    # Build Graph
    # =========================================================================
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_metadata", extract_metadata)
    workflow.add_node("analyze_note", analyze_note)
    workflow.add_node("generate_questions", generate_questions)
    workflow.add_node("synthesize_note", synthesize_note)
    
    # Add edges
    workflow.set_entry_point("extract_metadata")
    workflow.add_edge("extract_metadata", "analyze_note")
    
    # Conditional edge after analysis
    workflow.add_conditional_edges(
        "analyze_note",
        should_ask_questions,
        {
            "ask_questions": "generate_questions",
            "synthesize": "synthesize_note"
        }
    )
    
    # Questions lead to interrupt (will be handled by agent)
    workflow.add_edge("generate_questions", END)
    
    # Synthesis is the final step
    workflow.add_edge("synthesize_note", END)
    
    return workflow.compile()


def create_initial_state(
    raw_note: str,
    category: str = "general",
    template: str = "",
    template_source: str = "default"
) -> AgentState:
    """Create initial state for the graph.
    
    Args:
        raw_note: Raw note content
        category: Note category
        template: Template to use
        template_source: Source of the template
        
    Returns:
        Initial AgentState
    """
    return AgentState(
        raw_note=raw_note,
        input_metadata={},
        user_answers=[],
        analysis=None,
        questions=None,
        category=category,
        template=template,
        template_source=template_source,
        final_note=None,
        iteration_count=0,
        is_complete=False,
        error=None,
    )


def continue_with_answers(
    state: AgentState,
    answers: list[str]
) -> AgentState:
    """Create state for continuing after user answers.
    
    Args:
        state: Current state
        answers: User's answers
        
    Returns:
        Updated AgentState
    """
    new_state = dict(state)
    
    # Append new answers to existing ones
    existing_answers = new_state.get("user_answers", [])
    new_state["user_answers"] = existing_answers + answers
    
    return AgentState(**new_state)
