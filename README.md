# Atomic Note Architect (ANA)

원시(Raw) 노트를 **Zettelkasten 원칙**에 부합하는 **Atomic Note**로 변환하는 AI 에이전트입니다.

## 핵심 철학

1. **One Concept per Note** - 하나의 노트에는 하나의 아이디어만
2. **Autonomous** - 노트 자체만으로 이해 가능 (Self-contained)
3. **Interrogative Expansion** - 적극적인 인터뷰어 역할로 맥락 완성

## 주요 기능

- 🔍 **3단계 파이프라인**: 분석(Analysis) → 심문(Interrogation) → 합성(Synthesis)
- 💬 **Interactive Loop**: 최대 5개의 심층 질문을 통해 암묵지(Tacit Knowledge) 추출
- 🔀 **Smart Note Splitting**: 다중 개념이 혼재된 노트를 감지하여 AI가 자동으로 분리 및 내용 추출
- � **Auto-Linking (Hybrid)**:
  - 2-Stage Retrieval (Tag/Keyword + Embedding) + Rerank 아키텍처
  - Vault 내 존재하는 노트는 `[[Title]]`, 없으면 `[[Title (new)]]`로 자동 연결
- 🇰🇷 **Smart Localization**: 
  - 한글 중심 작성 (설명/문장)
  - 전문 기술 용어는 영어 원문 유지 (e.g., RAG, LLM)
- 📁 **Metadata Preservation**: 기존 Frontmatter 보존 및 자동 확장
- 📝 **Adaptive Template**: 카테고리별 맞춤형 템플릿 자동 적용

## 설치

```bash
# uv 사용 (권장)
uv sync

# pip 사용
pip install -e .
```

## 설정

1. `.env.example`을 `.env`로 복사:
```bash
cp .env.example .env
```

2. `.env` 파일 수정:
```bash
# LLM Provider 선택 (openai, anthropic, ollama, vllm)
ANA_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key

# Obsidian Vault 경로
ANA_VAULT_PATH=~/vault
```

## 사용법

### CLI 실행

```bash
# 대화형 모드
python -m src.main

# 파일에서 입력
python -m src.main --input raw_note.txt

# 출력 디렉토리 지정
python -m src.main --output ~/vault/notes/
```

### Python 코드에서 사용

```python
from src.agent import AtomicNoteArchitect
from src.config import ANAConfig

# 에이전트 초기화
config = ANAConfig()
agent = AtomicNoteArchitect(config)

# 노트 처리
raw_note = """
RAG는 검색 증강 생성의 약자다.
LLM의 환각 문제를 해결하기 위해 사용한다.
"""

response = agent.process(raw_note)

# 질문이 있으면 답변
if response.status == "needs_info":
    print("Questions:", response.interaction.questions_to_user)
    answers = ["...", "..."]  # 사용자 답변
    response = agent.answer_questions(answers)

# 최종 노트 저장
agent.save_note(response.draft_note)
```

## 프로젝트 구조

```
04.ANA/
├── src/
│   ├── __init__.py
│   ├── config.py           # 설정 관리
│   ├── llm_config.py       # LLM provider 설정
│   ├── schemas.py          # 데이터 모델
│   ├── vault_scanner.py    # Vault 메타데이터 스캔
│   ├── category_classifier.py  # 카테고리 분류
│   ├── template_manager.py # 템플릿 관리
│   ├── prompts.py          # 시스템 프롬프트
│   ├── graph.py            # LangGraph 워크플로우
│   ├── agent.py            # 에이전트 클래스
│   ├── utils.py            # 유틸리티 함수
│   └── main.py             # CLI 진입점
├── templates/              # 노트 템플릿
├── data/                   # 템플릿 DB
├── tests/                  # 테스트
├── pyproject.toml
└── README.md
```

## LLM Provider 설정

### OpenAI
```bash
ANA_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...
ANA_LLM_MODEL=o3
```

### Anthropic
```bash
ANA_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANA_LLM_MODEL=claude-3-5-sonnet-20241022
```

### Ollama (로컬)
```bash
ANA_LLM_PROVIDER=ollama
ANA_OLLAMA_BASE_URL=http://localhost:11434
ANA_OLLAMA_MODEL=llama3.1:8b
```

### vLLM (로컬)
```bash
ANA_LLM_PROVIDER=vllm
ANA_VLLM_BASE_URL=http://localhost:8000/v1
ANA_VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

## 라이선스

MIT License
