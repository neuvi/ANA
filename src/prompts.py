"""Prompts for ANA Agent.

Contains all system prompts and prompt templates for the 3-phase pipeline.
"""

# =============================================================================
# Language Rules
# =============================================================================

LANGUAGE_RULES = {
    "ko": (
        "- **모든 출력은 한국어(Korean)로 작성하세요**\n"
        "- 기술 용어는 영어 원문을 유지하세요 (예: RAG, GNN, Fine-tuning, API, LLM)\n"
        "- 예시: \"RAG(Retrieval-Augmented Generation)는 검색 기반의 생성 기법입니다.\""
    ),
    "en": (
        "- **Write all outputs in English**\n"
        "- Use standard technical terminology\n"
        "- Example: \"RAG (Retrieval-Augmented Generation) is a retrieval-based generation technique.\""
    ),
    "ja": (
        "- **すべての出力は日本語で書いてください**\n"
        "- 技術用語は英語のまま維持してください（例：RAG, GNN, Fine-tuning, API, LLM）\n"
        "- 例：「RAG（Retrieval-Augmented Generation）は検索ベースの生成技術です。」"
    ),
    "zh": (
        "- **所有输出请用中文撰写**\n"
        "- 技术术语保持英文原文（例如：RAG, GNN, Fine-tuning, API, LLM）\n"
        "- 示例：\"RAG（Retrieval-Augmented Generation）是一种基于检索的生成技术。\""
    ),
}

# =============================================================================
# System Prompt Template
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a strict and insightful 'Knowledge Architect'.
Your mission is to transform the user's rough memos into 'Atomic Notes' worthy of permanent preservation.

Core Principles:
1. **Atomicity**: One concept per note. If multiple topics are mixed, suggest splitting into n separate notes.
2. **Proactive Inquiry**: Extract tacit knowledge from the user's mind through targeted questions.
3. **Rewrite, Don't Append**: Integrate user answers naturally into the text, not as Q&A format.
4. **Format Compliance**: Follow Obsidian-compatible Markdown with YAML frontmatter.
5. **Metadata Preservation**: Utilize and preserve existing frontmatter information.

Language Rules:
{language_rules}
"""


def get_system_prompt(language: str = "ko") -> str:
    """Get system prompt with appropriate language rules.
    
    Args:
        language: Language code (ko, en, ja, zh)
        
    Returns:
        Formatted system prompt
    """
    lang_rules = LANGUAGE_RULES.get(language, LANGUAGE_RULES["en"])
    return SYSTEM_PROMPT_TEMPLATE.format(language_rules=lang_rules)


# Default system prompt (for backward compatibility)
SYSTEM_PROMPT = get_system_prompt("ko")


# =============================================================================
# Phase 1: Analysis Prompt
# =============================================================================

ANALYSIS_PROMPT = """Analyze the following raw note and determine:
1. What concepts are present
2. Whether the information is sufficient for an atomic note
3. What context is missing
4. Whether the note should be split

Existing metadata from frontmatter:
{existing_metadata}

Raw note content:
---
{raw_note}
---

Respond in JSON format with this structure:
{{
    "detected_concepts": ["concept1", "concept2"],
    "missing_context": ["what's missing 1", "what's missing 2"],
    "is_sufficient": true/false,
    "should_split": true/false,
    "split_suggestions": ["title for note 1", "title for note 2"],
    "detected_category": "category-name or null"
}}"""


# =============================================================================
# Split Extraction Prompt
# =============================================================================

SPLIT_EXTRACTION_PROMPT = """Extract content for a specific topic from the original note.

Original note:
---
{raw_note}
---

Target topic to extract: {target_topic}

Instructions:
1. Extract all relevant information from the original note that belongs to this topic
2. Include definitions, explanations, examples, and any related details
3. Write in the same language as the original note
4. Make the extracted content self-contained and coherent
5. Do NOT include information that belongs to other topics

Respond in JSON format:
{{
    "extracted_content": "The relevant content extracted for this topic",
    "key_points": ["main point 1", "main point 2"],
    "related_topics": ["other topic from original note that connects to this"]
}}"""


# =============================================================================
# Phase 2: Interrogation Prompt
# =============================================================================

INTERROGATION_PROMPT = """Based on the analysis, generate questions to fill the information gaps.

Analysis result:
- Detected concepts: {detected_concepts}
- Missing context: {missing_context}
- Detected category: {detected_category}

Existing metadata:
{existing_metadata}

Original note:
---
{raw_note}
---

Generate up to {max_questions} essential questions to extract tacit knowledge.

Question categories to cover:
1. **Context (맥락)**: Why is this important? What problem does this solve?
2. **Relation (관계)**: How does this connect to existing knowledge or projects?
3. **Clarification (구체화)**: What do specific terms or metrics mean?

Respond in JSON format:
{{
    "questions_to_user": [
        "Question 1",
        "Question 2"
    ],
    "question_categories": ["context", "relation"]
}}

Guidelines:
- Maximum {max_questions} questions
- Make questions specific and actionable
- Avoid yes/no questions
- Focus on what would make the note self-contained"""


# =============================================================================
# Phase 3: Synthesis Prompt
# =============================================================================

SYNTHESIS_PROMPT = """Create a final Atomic Note by synthesizing all information.

Original note:
---
{raw_note}
---

Existing metadata to preserve:
{existing_metadata}

Questions asked and user's answers:
{qa_pairs}

Category: {category}

Template to follow:
---
{template}
---

Instructions:
1. Create a clear, descriptive title (sentence-style)
2. Preserve existing metadata and add new relevant fields
3. Integrate user answers naturally into the content (DO NOT use Q&A format)
4. Write in a self-contained manner (the note should be understandable on its own)
5. Suggest relevant tags based on content
6. Suggest related notes/topics for linking
7. Follow the template structure

Respond in JSON format:
{{
    "title": "A descriptive sentence-style title",
    "tags": ["tag1", "tag2"],
    "content": "The complete note content in Markdown",
    "frontmatter": {{
        "title": "...",
        "tags": [...],
        "type": "{category}",
        "created": "...",
        ...preserved and new metadata...
    }},
    "suggested_links": ["Related Topic 1", "Related Topic 2"]
}}"""


# =============================================================================
# Category Classification Prompt
# =============================================================================

CATEGORY_PROMPT = """Classify this note into a category.

Existing categories in the knowledge base:
{existing_categories}

Note content:
---
{note_content}
---

Instructions:
1. If the note fits an existing category, return that category name
2. If no existing category fits, suggest a new one
3. Category names should be lowercase with hyphens (e.g., "book-note", "project-idea")

Return ONLY the category name, nothing else."""


# =============================================================================
# Template Generation Prompt
# =============================================================================

TEMPLATE_GENERATION_PROMPT = """Create a Markdown template for the category: "{category}"

{samples_section}

Requirements:
1. YAML frontmatter with: title, tags, type, created, source
2. Use Jinja2 syntax: {{{{ variable }}}}
3. Appropriate sections for this category
4. "## Related Links" section at the end

Return ONLY the Markdown template."""


# =============================================================================
# Tag Suggestion Prompt
# =============================================================================

TAG_SUGGESTION_PROMPT = """Based on the note content, suggest relevant tags.

Existing tags in the vault (prioritize these for consistency):
{existing_vault_tags}

Note content:
---
{note_content}
---

Instructions:
1. PRIORITIZE existing vault tags for consistency across the knowledge base
2. Suggest new tags only if no existing tag fits well
3. Use lowercase with hyphens (e.g., "machine-learning", "deep-learning")
4. Tags should be specific enough to be useful but general enough to apply to multiple notes
5. Maximum {max_tags} tags

Respond in JSON format:
{{
    "suggested_tags": ["tag1", "tag2", "tag3"],
    "reasoning": "Brief explanation of why these tags were chosen"
}}"""


# =============================================================================
# Helper Functions
# =============================================================================

def format_qa_pairs(questions: list[str], answers: list[str]) -> str:
    """Format question-answer pairs for the synthesis prompt.
    
    Args:
        questions: List of questions asked
        answers: List of user answers
        
    Returns:
        Formatted string of Q&A pairs
    """
    if not questions or not answers:
        return "No questions were asked."
    
    pairs = []
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        pairs.append(f"Q{i}: {q}\nA{i}: {a}")
    
    return "\n\n".join(pairs)


def format_metadata(metadata: dict) -> str:
    """Format metadata dictionary for prompt insertion.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        Formatted YAML-like string
    """
    if not metadata:
        return "None"
    
    lines = []
    for key, value in metadata.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- {key}: {value}")
    
    return "\n".join(lines)
