**Atomic Note Architect Agent** 구축을 위한 기술 명세(Specification)를 제안합니다.

이 에이전트의 핵심은 단순히 텍스트를 요약하거나 나누는 것이 아니라, 사용자의 **'생각을 확장(Expansion)'** 시키고, **'맥락(Context)'** 을 완성하여, 향후 Second Brain에서 다른 지식과 유기적으로 연결될 수 있는 **'완결성 있는 지식 블록'** 을 만드는 것입니다.

---

## 1. 프로젝트 개요 (Overview)

* **에이전트 명:** Atomic Note Architect (가칭: ANA)
* **목적:** 원시(Raw) 노트 또는 문서를 입력받아, [Zettelkasten](https://zettelkasten.de/introduction/) 원칙에 부합하는 Atomic Note로 변환하고, 부족한 맥락을 질문을 통해 보완하여 지식베이스(Second Brain)에 등재 가능한 형태로 출력함.
* **핵심 철학:**
1. **One Concept per Note:** 하나의 노트에는 하나의 아이디어만 담는다.
2. **Autonomous:** 노트 자체만으로도 이해가 가능해야 한다 (Self-contained).
3. **Interrogative Expansion:** 에이전트는 수동적인 편집자가 아니라, 적극적인 인터뷰어가 되어야 한다.



---

## 2. 기능 명세 (Functional Specification)

이 에이전트는 **3단계 파이프라인(분석 - 심문 - 합성)** 을 따릅니다.

### Phase 1: 해체 및 분석 (Deconstruction & Analysis)

* **입력:** 사용자의 원시 노트 (회의록, 아이디어 스케치, 논문 요약 등).
* **기능:**
* 복합적인 주제를 개별 개념(Concept) 단위로 분리.
* 각 개념이 Atomic Note가 되기에 충분한 정보량을 가졌는지 판단.
* 이미 명확한 사실(Fact)과 추가 설명이 필요한 통찰(Insight)을 구분.



### Phase 2: 갭 분석 및 심문 (Gap Analysis & Interrogation) - *핵심 기능*

* **기능:** Second Brain 구축을 위해 필요한 '연결 고리'가 빠져 있는지 확인하고 질문을 생성합니다.
* **질문 카테고리:**
1. **맥락(Context):** "이 아이디어가 왜 중요한가요?", "어떤 문제 해결을 위해 이 개념을 떠올렸나요?"
2. **관계(Relation):** "이 개념은 기존의 어떤 프로젝트나 지식과 연결될 수 있나요?" (유사/반대 개념)
3. **구체화(Clarification):** "여기서 언급한 '효율적'이라는 단어의 구체적인 지표는 무엇인가요?"


* **출력:** 사용자에게 던질 2~3개의 핵심 질문 리스트.

### Phase 3: 합성 및 정제 (Synthesis & Refining)

* **입력:** 원시 노트 + (Phase 2에 대한) 사용자 답변.
* **기능:**
* 답변 내용을 원문에 녹여내어 문장을 재구성.
* Second Brain 포맷(Markdown + Frontmatter)에 맞춰 최종 노트 생성.
* 제목(Title)을 직관적이고 검색 용이하게 변경.



---

## 3. 워크플로우 로직 (Workflow Logic)

에이전트의 동작 흐름도(Flow)입니다.

1. **USER:** 원시 노트 입력
2. **AGENT:** 텍스트 분석 실행
* *판단:* 내용이 충분히 구체적인가? -> (YES: 바로 4번으로 이동 / NO: 3번으로 이동)


3. **AGENT (Interactive Loop):**
* "이 내용을 Atomic Note로 만들기 위해 몇 가지 정보가 더 필요합니다."
* **Q1:** [질문 내용]
* **Q2:** [질문 내용]
* **USER:** 답변 작성


4. **AGENT:** 최종 Atomic Note 생성
* **Title:** [핵심을 관통하는 문장형 제목]
* **Tags:** #키워드 #카테고리
* **Body:** (사용자 답변이 통합된) 완성된 본문
* **Insight/Link:** 차후 연결할 만한 주제 제안



---

## 4. 시스템 프롬프트 설계 (System Prompt Strategy)

LLM에게 부여할 페르소나와 지침입니다.

> **Role:** 당신은 엄격하고 통찰력 있는 '지식 아키텍트(Knowledge Architect)'입니다.
> **Task:** 사용자의 거친 메모를 영구 보존 가치가 있는 'Atomic Note'로 변환하십시오.
> **Rules:**
> 1. **원자성(Atomicity):** 입력된 텍스트에 여러 주제가 섞여 있다면, 이를 n개의 노트로 분리할 것을 제안하십시오.
> 2. **적극적 개입(Proactive Inquiry):** 사용자가 "A는 좋다"라고만 적었다면, 반드시 "어떤 맥락에서 좋은가?", "비교 대상은 무엇인가?"를 물어보십시오. 사용자의 머릿속에 있는 '암묵지(Tacit Knowledge)'를 글로 꺼내는 것이 당신의 목표입니다.
> 3. **재작성(Rewrite):** 사용자의 답변을 받으면, 그것을 문답 형식(Q&A)으로 밑에 붙이지 말고, 본문 텍스트 안에 자연스럽게 통합하여 하나의 완결된 글로 재작성(Rewrite)하십시오.
> 4. **형식(Format):** Obsidian/Logseq 등의 툴에서 사용 가능한 Markdown 형식을 준수하십시오.
> 
> 

---

## 5. 데이터 스키마 (Data Schema)

출력물은 프로그램적으로 처리 가능하도록 구조화되어야 합니다 (예: JSON 모드 활용 후 Markdown 렌더링).

```json
{
  "status": "needs_info" | "completed",
  "analysis": {
    "detected_concepts": ["개념 A", "개념 B"],
    "missing_context": ["개념 A의 구체적 적용 사례", "개념 B의 기술적 제약 사항"]
  },
  "interaction": {
    "questions_to_user": [
      "이 기술을 현재 진행 중인 [프로젝트 명]에 적용할 계획이신가요? 그렇다면 예상되는 장애물은 무엇인가요?",
      "'효율성'을 판단하는 기준(Metric)을 정의해 주실 수 있나요?"
    ]
  },
  "draft_note": {
    "title": "예: RAG 시스템에서 Hybrid Search가 필요한 이유",
    "tags": ["#RAG", "#Search", "#AI"],
    "content": "(현재까지의 내용으로 작성된 초안...)"
  }
}

```

---

## 6. 구현을 위한 제안 (Next Step)

이 스펙을 바탕으로 실제 에이전트를 구축할 때, **어떤 형태**로 시작하고 싶으신가요?

1. **프롬프트 엔지니어링:** ChatGPT나 Claude의 Custom Instructions(또는 System Prompt)에 바로 복사해서 쓸 수 있는 **'최적화된 프롬프트 템플릿'** 을 작성해 드릴까요?
2. **Python 프로토타입:** LangChain 등을 이용하여 입력 -> 질문 생성 -> 답변 결합 -> 파일 저장을 수행하는 **'Python 코드 스켈레톤'** 을 작성해 드릴까요?

원하시는 방향을 말씀해 주시면 바로 구체적인 결과물을 드리겠습니다.