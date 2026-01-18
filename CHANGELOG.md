# Changelog

All notable changes to the ANA (Atomic Note Architect) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Test Suite**: Comprehensive pytest test suite with 6 test modules covering all major components
  - `tests/test_agent.py` - AtomicNoteArchitect agent tests
  - `tests/test_graph.py` - LangGraph workflow tests
  - `tests/test_link_analyzer.py` - Note linking analyzer tests
  - `tests/test_vault_scanner.py` - Vault scanner tests
  - `tests/test_category_classifier.py` - Category classifier tests
  - `tests/conftest.py` - Shared pytest fixtures
- **Logging Module**: `src/logging_config.py` with Rich formatting support
- **Multi-language Support**: Configurable output language (ko, en, ja, zh) via `ANA_OUTPUT_LANGUAGE`
- **New Error Classes**: `LLMParseError` and `RerankerError` for better error handling
- **Enhanced Embedding Cache**:
  - Batch embedding processing with configurable batch size (`ANA_EMBEDDING_BATCH_SIZE`)
  - Deferred save to reduce I/O operations
  - Optional Chroma vector DB backend (`ANA_USE_VECTOR_DB`)
  - Extended statistics (cache size, dimension, pending changes)
- **Enhanced Doctor Diagnostics**:
  - Configuration value validation
  - Reranker model availability check
  - LLM connection test (`ana doctor --debug`)
- **Backlink Analysis** (`src/backlink_analyzer.py`):
  - Automatic detection of backlink opportunities in existing notes
  - AI-powered and keyword-based matching
  - Auto-apply backlinks when saving notes
  - `save_note_with_backlinks()` method in agent

### Changed
- **Error Handling**: Silent exceptions now log warnings instead of passing silently
- **prompts.py**: System prompt now uses dynamic language rules instead of hardcoded Korean
- **embedding_cache.py**: Now supports batch processing and vector DB backend

### Fixed
- Improved error messages for debugging

## [0.1.0] - 2026-01-18

### Added
- Initial release of ANA (Atomic Note Architect)
- **3-Phase Pipeline**: Analysis → Interrogation → Synthesis
- **LangGraph Integration**: Stateful workflow management
- **Multi-LLM Support**: OpenAI, Anthropic, Ollama, vLLM
- **CLI Interface**: `ana new`, `ana process`, `ana config`, `ana doctor`
- **Note Linking**: 2-Stage Retrieval + Rerank architecture
  - Tag/Category matching
  - Keyword BM25 similarity
  - Embedding similarity (Ollama/OpenAI)
  - Cross-Encoder reranking
- **Smart Note Splitting**: Automatic detection and suggestion for multi-concept notes
- **Template Management**: File → DB → AI priority system
- **Category Classification**: Frontmatter-based or AI-powered
- **Obsidian Plugin**: TypeScript plugin for in-editor processing
- **FastAPI Server**: API server for plugin communication
- **Embedding Cache**: Incremental embedding updates with hash-based caching
