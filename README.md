# Atomic Note Architect (ANA)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)

원시(Raw) 노트를 **Zettelkasten 원칙**에 부합하는 **Atomic Note**로 변환하는 AI 에이전트입니다.

> 이 에이전트의 핵심은 단순히 텍스트를 요약하거나 나누는 것이 아니라, 사용자의 **'생각을 확장(Expansion)'** 시키고, **'맥락(Context)'** 을 완성하여, 향후 Second Brain에서 다른 지식과 유기적으로 연결될 수 있는 **'완결성 있는 지식 블록'** 을 만드는 것입니다.

## 빠른 시작 (Quick Start)

```bash
# 1. 클론 및 설치
git clone https://github.com/your-repo/ana.git && cd ana
uv sync

# 2. 환경 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY와 ANA_VAULT_PATH 설정

# 3. 실행
ana  # 또는 python -m src.main
```

## 요구사항

| 항목 | 요구사항 |
|------|---------|
| **Python** | 3.10 이상 |
| **패키지 매니저** | uv (권장) 또는 pip |
| **LLM API** | OpenAI, Anthropic, 또는 로컬 LLM (Ollama/vLLM) |
| **Obsidian Vault** | 기존 Vault 경로 (연결 기능 사용 시) |

## 핵심 철학

1. **One Concept per Note** - 하나의 노트에는 하나의 아이디어만
2. **Autonomous** - 노트 자체만으로 이해 가능 (Self-contained)
3. **Interrogative Expansion** - 적극적인 인터뷰어 역할로 맥락 완성

## 주요 기능

- 🔍 **3단계 파이프라인**: 분석(Analysis) → 심문(Interrogation) → 합성(Synthesis)
- 💬 **Interactive Loop**: 최대 5개의 심층 질문을 통해 암묵지(Tacit Knowledge) 추출
- 🔀 **Smart Note Splitting**: 다중 개념이 혼재된 노트를 감지하여 AI가 자동으로 분리 및 내용 추출
- 🔗 **Auto-Linking (Hybrid)**:
  - 2-Stage Retrieval (Tag/Keyword + Embedding) + Rerank 아키텍처
  - Vault 내 존재하는 노트는 `[[Title]]`, 없으면 `[[Title (new)]]`로 자동 연결
- 🇰🇷 **Smart Localization**: 
  - 한글 중심 작성 (설명/문장)
  - 전문 기술 용어는 영어 원문 유지 (e.g., RAG, LLM)
- 📁 **Metadata Preservation**: 기존 Frontmatter 보존 및 자동 확장
- 📝 **Adaptive Template**: 카테고리별 맞춤형 템플릿 자동 적용

## 동작 예시

**입력:**
```
RAG는 검색 증강 생성이다. LLM 환각 해결용. GraphRAG도 있더라.
```

**처리:**
1. 다중 개념 감지 → 2개 노트로 분리 제안
2. 질문 생성: "RAG의 구체적인 활용 사례는?", "GraphRAG와 기존 RAG의 차이점은?"
3. 답변 반영 후 Atomic Note 생성

**출력:** `RAG-검색-증강-생성.md`, `GraphRAG-개요.md`

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

### CLI 명령어

```bash
# 도움말 보기
ana --help

# 대화형 모드로 새 노트 생성
ana new

# 파일 처리
ana process raw_note.txt

# 출력 디렉토리 지정
ana new --output ~/vault/notes/

# 비대화형 모드
ana new --no-interactive

# 임베딩 동기화
ana sync
```

### 설정 관리

```bash
# 대화형 설정 마법사 (처음 사용시 권장)
ana config init

# 현재 설정 확인
ana config show

# 개별 설정 변경
ana config set llm_provider ollama
ana config set vault_path ~/Documents/Obsidian
```

### 환경 진단

```bash
# 환경 및 설정 진단
ana doctor
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
│   ├── cli/                   # CLI 모듈
│   │   ├── main.py            # Click 기반 CLI 진입점
│   │   ├── commands.py        # 서브커맨드 구현
│   │   ├── config_wizard.py   # 설정 마법사
│   │   └── doctor.py          # 환경 진단
│   ├── api/                   # API 서버 (Obsidian 플러그인용)
│   │   ├── server.py          # FastAPI 서버
│   │   └── schemas.py         # API 스키마
│   ├── config.py              # 설정 관리
│   ├── errors.py              # 사용자 친화적 에러
│   ├── agent.py               # 에이전트 클래스
│   └── ...                    # 기타 모듈
├── obsidian-ana-plugin/       # Obsidian 플러그인
│   ├── main.ts                # 플러그인 진입점
│   ├── api.ts                 # API 클라이언트
│   ├── settings.ts            # 설정 UI
│   ├── modal.ts               # 모달 다이얼로그
│   └── styles.css             # 스타일
├── templates/                 # 노트 템플릿
├── Makefile                   # 빌드/설치 스크립트
└── pyproject.toml
```

## Obsidian 플러그인

Obsidian 내에서 직접 ANA를 사용할 수 있는 플러그인입니다.

### 설치

```bash
# 1. API 서버 시작 (터미널)
ana serve

# 2. 플러그인 빌드 (처음 한 번만)
cd obsidian-ana-plugin
npm install && npm run build

# 3. 플러그인을 Vault에 복사
mkdir -p ~/Obsidian/.obsidian/plugins/ana-atomic-note-architect
cp main.js manifest.json styles.css ~/Obsidian/.obsidian/plugins/ana-atomic-note-architect/

# 4. Obsidian 재시작 후 플러그인 활성화
```

### 사용법

1. **서버 시작**: 터미널에서 `ana serve`
2. **Obsidian 열기**: 노트를 열고 `Ctrl/Cmd + P`
3. **명령 실행**: "ANA: Process Current Note" 선택
4. **질문 답변**: 모달에서 질문에 답변
5. **저장**: 미리보기 확인 후 저장

### 사용 가능한 명령어

| 명령어 | 설명 |
|--------|------|
| `ANA: Process Current Note` | 현재 노트 전체 처리 |
| `ANA: Process Selected Text` | 선택한 텍스트만 처리 |
| `ANA: Check Server Connection` | 서버 연결 확인 |



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

## FAQ

### Q: Obsidian Vault 경로는 어떻게 찾나요?
Obsidian 앱에서 **설정 > 파일 및 링크 > Vault 위치**에서 확인할 수 있습니다. 또는 `.obsidian` 폴더가 있는 디렉토리가 Vault 경로입니다.

### Q: API 키 없이 사용할 수 있나요?
네, [Ollama](https://ollama.ai/)를 설치하면 로컬에서 무료로 사용 가능합니다:
```bash
# Ollama 설치 후
ollama pull llama3.1:8b
# .env 설정
ANA_LLM_PROVIDER=ollama
```

### Q: 예제 노트로 테스트하려면?
```bash
python -m src.main --input examples/sample_note.txt
```

### Q: 질문 없이 바로 노트를 생성하려면?
현재 버전에서는 맥락 완성을 위해 질문이 필수입니다. 질문을 건너뛰려면 빈 답변(`Enter`)을 입력하세요.

## 기여하기

버그 리포트, 기능 제안, PR을 환영합니다!

```bash
# 개발 환경 설치
uv sync --group dev

# 테스트 실행
pytest
```

## 라이선스

MIT License
