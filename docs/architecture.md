# ANA Architecture

> Atomic Note Architect - AI-powered Zettelkasten note transformation system

## System Overview

```mermaid
graph TB
    subgraph Input
        A[Raw Note] --> B[CLI / API / Plugin]
    end
    
    subgraph Core["ANA Core"]
        B --> C[AtomicNoteArchitect]
        C --> D[VaultScanner]
        C --> E[CategoryClassifier]
        C --> F[TemplateManager]
        C --> G[LangGraph Workflow]
        
        subgraph Pipeline["3-Phase Pipeline"]
            G --> H[Phase 1: Analysis]
            H --> I[Phase 2: Interrogation]
            I --> J[Phase 3: Synthesis]
        end
    end
    
    subgraph Linking["Note Linking"]
        C --> K[LinkAnalyzer]
        K --> L[2-Stage Retrieval]
        L --> M[EmbeddingCache]
        L --> N[Reranker]
        
        C --> O[BacklinkAnalyzer]
    end
    
    subgraph Output
        J --> P[DraftNote]
        P --> Q[Obsidian Vault]
    end
```

## Module Dependencies

```mermaid
graph LR
    agent[agent.py] --> graph[graph.py]
    agent --> config[config.py]
    agent --> llm_config[llm_config.py]
    agent --> vault_scanner[vault_scanner.py]
    agent --> category_classifier[category_classifier.py]
    agent --> template_manager[template_manager.py]
    agent --> link_analyzer[link_analyzer.py]
    agent --> backlink_analyzer[backlink_analyzer.py]
    agent --> embedding_cache[embedding_cache.py]
    
    graph --> schemas[schemas.py]
    graph --> prompts[prompts.py]
    
    link_analyzer --> embedding_cache
    link_analyzer --> vault_scanner
    
    backlink_analyzer --> vault_scanner
    backlink_analyzer --> schemas
```

## Data Flow

### Note Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI/API
    participant Agent
    participant Graph
    participant LLM
    
    User->>CLI/API: Raw note
    CLI/API->>Agent: process(raw_note)
    Agent->>Agent: Extract frontmatter
    Agent->>Agent: Classify category
    Agent->>Agent: Get template
    Agent->>Graph: invoke(initial_state)
    
    Graph->>LLM: Analyze note
    LLM-->>Graph: AnalysisResult
    
    alt Needs clarification
        Graph->>LLM: Generate questions
        LLM-->>Graph: Questions
        Graph-->>Agent: needs_info status
        Agent-->>CLI/API: InteractionPayload
        CLI/API-->>User: Display questions
        User->>CLI/API: Answers
        CLI/API->>Agent: answer_questions(answers)
        Agent->>Graph: continue with answers
    end
    
    Graph->>LLM: Synthesize note
    LLM-->>Graph: DraftNote
    Graph-->>Agent: completed status
    
    Agent->>Agent: Find related notes
    Agent->>Agent: Apply backlinks
    Agent-->>CLI/API: AgentResponse
    CLI/API-->>User: Final note
```

### Note Linking Flow

```mermaid
sequenceDiagram
    participant Agent
    participant LinkAnalyzer
    participant VaultScanner
    participant EmbeddingCache
    participant Reranker
    
    Agent->>LinkAnalyzer: find_related_notes(note)
    
    par Stage 1: Retrieval
        LinkAnalyzer->>VaultScanner: get all notes
        LinkAnalyzer->>LinkAnalyzer: Tag/Category matching
        LinkAnalyzer->>LinkAnalyzer: Keyword similarity (BM25)
        LinkAnalyzer->>EmbeddingCache: get embeddings
        EmbeddingCache-->>LinkAnalyzer: vectors
    end
    
    LinkAnalyzer->>LinkAnalyzer: RRF merge (30 candidates)
    
    LinkAnalyzer->>Reranker: rerank(query, candidates)
    Reranker-->>LinkAnalyzer: scored results
    
    LinkAnalyzer-->>Agent: top 5 wikilinks
```

## Key Components

### Core Modules

| Module | Purpose |
|--------|---------|
| [agent.py](file:///home/shkim/workspace/04.ANA/src/agent.py) | Main orchestrator, manages pipeline |
| [graph.py](file:///home/shkim/workspace/04.ANA/src/graph.py) | LangGraph workflow definition |
| [schemas.py](file:///home/shkim/workspace/04.ANA/src/schemas.py) | Pydantic data models |
| [config.py](file:///home/shkim/workspace/04.ANA/src/config.py) | Configuration management |
| [llm_config.py](file:///home/shkim/workspace/04.ANA/src/llm_config.py) | LLM provider abstraction |

### Note Linking

| Module | Purpose |
|--------|---------|
| [link_analyzer.py](file:///home/shkim/workspace/04.ANA/src/link_analyzer.py) | 2-Stage Retrieval + Rerank |
| [backlink_analyzer.py](file:///home/shkim/workspace/04.ANA/src/backlink_analyzer.py) | Bidirectional link discovery |
| [embedding_cache.py](file:///home/shkim/workspace/04.ANA/src/embedding_cache.py) | Incremental embedding storage |

### Utilities

| Module | Purpose |
|--------|---------|
| [vault_scanner.py](file:///home/shkim/workspace/04.ANA/src/vault_scanner.py) | Obsidian vault traversal |
| [category_classifier.py](file:///home/shkim/workspace/04.ANA/src/category_classifier.py) | Note categorization |
| [template_manager.py](file:///home/shkim/workspace/04.ANA/src/template_manager.py) | Template resolution (File → DB → AI) |
| [validators.py](file:///home/shkim/workspace/04.ANA/src/validators.py) | Runtime validation utilities |

## Technology Stack

- **Workflow**: LangGraph (stateful agent orchestration)
- **LLM Integration**: LangChain (multi-provider support)
- **Data Validation**: Pydantic
- **CLI**: Click + Rich
- **API Server**: FastAPI + Uvicorn
- **Embeddings**: Ollama / OpenAI
- **Reranking**: sentence-transformers (CrossEncoder)
- **Vector DB**: Optional Chroma integration
