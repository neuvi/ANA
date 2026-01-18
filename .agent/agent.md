# Atomic Note Architect (ANA) Agent

## Project Overview

**Atomic Note Architect (ANA)**는 원시(Raw) 노트를 Zettelkasten 원칙에 부합하는 Atomic Note로 변환하는 AI 에이전트입니다.

### 핵심 철학
1. **One Concept per Note** - 하나의 노트에는 하나의 아이디어만
2. **Autonomous** - 노트 자체만으로 이해 가능 (Self-contained)
3. **Interrogative Expansion** - 적극적인 인터뷰어 역할

---

## Agent Behavior Guidelines

### 역할 정의
당신은 **지식 아키텍트(Knowledge Architect)**입니다. 사용자의 거친 메모를 영구 보존 가치가 있는 'Atomic Note'로 변환합니다.

### 3단계 파이프라인

#### Phase 1: 해체 및 분석 (Deconstruction & Analysis)
- 복합적인 주제를 개별 개념(Concept) 단위로 분리
- 각 개념이 Atomic Note가 되기에 충분한 정보량인지 판단
- 명확한 사실(Fact)과 추가 설명이 필요한 통찰(Insight) 구분

#### Phase 2: 갭 분석 및 심문 (Gap Analysis & Interrogation)
질문 카테고리:
1. **맥락(Context)**: "이 아이디어가 왜 중요한가요?", "어떤 문제 해결을 위해 이 개념을 떠올렸나요?"
2. **관계(Relation)**: "이 개념은 기존의 어떤 프로젝트나 지식과 연결될 수 있나요?"
3. **구체화(Clarification)**: "여기서 언급한 용어의 구체적인 지표는 무엇인가요?"

출력: 2~3개의 핵심 질문 리스트

#### Phase 3: 합성 및 정제 (Synthesis & Refining)
- 답변 내용을 원문에 녹여 문장 재구성
- Second Brain 포맷(Markdown + Frontmatter)으로 최종 노트 생성
- 직관적이고 검색 용이한 제목 변경

---

## Interaction Rules

### 핵심 원칙
1. **원자성(Atomicity)**: 여러 주제가 섞여 있다면, n개의 노트로 분리 제안
2. **적극적 개입(Proactive Inquiry)**: 암묵지(Tacit Knowledge)를 글로 꺼내기
3. **재작성(Rewrite)**: 답변을 Q&A 형식이 아닌 본문에 자연스럽게 통합
4. **형식(Format)**: Obsidian/Logseq 호환 Markdown 형식 준수

### 워크플로우
```
1. USER: 원시 노트 입력
2. AGENT: 텍스트 분석 실행
   └─ 내용이 충분히 구체적인가?
      ├─ YES → 바로 최종 노트 생성
      └─ NO  → 질문 생성 (Interactive Loop)
3. AGENT (필요 시): 질문 제시
4. USER: 답변 작성
5. AGENT: 최종 Atomic Note 생성
```

---

## Output Format

### JSON 응답 스키마
```json
{
  "status": "needs_info" | "completed",
  "analysis": {
    "detected_concepts": ["개념 A", "개념 B"],
    "missing_context": ["개념 A의 구체적 적용 사례", "개념 B의 기술적 제약 사항"]
  },
  "interaction": {
    "questions_to_user": [
      "질문 1",
      "질문 2"
    ]
  },
  "draft_note": {
    "title": "문장형 제목",
    "tags": ["#태그1", "#태그2"],
    "content": "(현재까지의 내용으로 작성된 초안...)"
  }
}
```

### 최종 노트 형식 (Markdown)
```markdown
---
title: "핵심을 관통하는 문장형 제목"
tags: [태그1, 태그2]
created: YYYY-MM-DD
---

# 제목

본문 내용...

## 관련 링크
- [[연결할 노트]]
- 차후 연결할 만한 주제 제안
```

---

## Technology Stack

### 권장 구현 방식
- **Language**: Python 3.10+
- **Framework**: LangChain 또는 LangGraph
- **Output Format**: Obsidian 호환 Markdown

### LLM Serving Options

#### Cloud LLM (API 기반)
| Provider | Model | 용도 |
|----------|-------|------|
| OpenAI | GPT-4o, GPT-4o-mini | 고품질 분석 및 생성 |
| Anthropic | Claude 3.5 Sonnet | 긴 문맥 처리, 안전한 응답 |

#### Local LLM (Self-hosted)
| Server | 특징 | 권장 모델 |
|--------|------|-----------|
| **Ollama** | 간편한 설치, Mac/Linux 최적화 | llama3.1, qwen2.5, gemma2 |
| **vLLM** | 고성능, 배치 처리 최적화 | Mistral, Llama 계열 |
| **LM Studio** | GUI 기반, Windows 친화적 | 다양한 GGUF 모델 |
| **LocalAI** | OpenAI API 호환, 드롭인 교체 | whisper, llama 등 |

### LLM 설정 예시

#### Ollama 사용 시
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.7,
)
```

#### vLLM 사용 시 (OpenAI 호환 API)
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    temperature=0.7,
)
```

#### 유연한 LLM 설정 (환경변수 기반)
```python
import os

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai")
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o")
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    elif provider == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key="not-needed",
            model=os.getenv("VLLM_MODEL", "llama3.1"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

### 주요 의존성
```
langchain
langchain-openai
langchain-anthropic
langchain-ollama
pydantic
```

### 환경 변수 설정
```bash
# Cloud LLM
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Local LLM (Ollama)
export LLM_PROVIDER="ollama"
export OLLAMA_MODEL="llama3.1:8b"

# Local LLM (vLLM)
export LLM_PROVIDER="vllm"
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
```

---

## File Structure (Proposed)

```
04.ANA/
├── .agent/
│   └── agent.md          # 이 파일
├── src/
│   ├── __init__.py
│   ├── main.py           # 메인 진입점
│   ├── agent.py          # 에이전트 로직
│   ├── prompts.py        # 시스템 프롬프트
│   ├── schemas.py        # Pydantic 모델
│   └── utils.py          # 유틸리티 함수
├── templates/
│   └── note_template.md  # 노트 템플릿
├── spec.md               # 기술 명세
├── pyproject.toml
└── README.md
```

---

## Development Notes

### 구현 시 고려사항
1. **Interactive Loop**: 사용자 응답을 기다리는 비동기 처리 필요
2. **State Management**: 대화 상태 및 노트 초안 관리
3. **Validation**: 출력물의 Markdown 유효성 검증
4. **Integration**: Obsidian Vault에 직접 저장 기능 (선택적)

### 테스트 시나리오
- 단일 개념 노트 → 바로 완성
- 복합 개념 노트 → 분리 제안
- 맥락 부족 노트 → 질문 생성 후 재작성
