# ANA Development Guide

> Guide for contributing to and developing ANA

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Ollama (for local LLM) or OpenAI API key

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/ana.git
cd ana

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your settings
```

### Environment Variables

```bash
# .env file
ANA_VAULT_PATH=~/your-vault        # Obsidian vault path
ANA_LLM_PROVIDER=ollama             # openai, anthropic, ollama, vllm
ANA_OLLAMA_MODEL=llama3.1:8b        # Model for Ollama
ANA_OLLAMA_BASE_URL=http://localhost:11434

# Optional
OPENAI_API_KEY=sk-...               # If using OpenAI
ANTHROPIC_API_KEY=sk-ant-...        # If using Anthropic
```

---

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_agent.py

# Run specific test
uv run pytest tests/test_agent.py::TestAtomicNoteArchitect::test_init

# Run with verbose output
uv run pytest -v
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_agent.py            # Agent tests
├── test_graph.py            # LangGraph workflow tests
├── test_link_analyzer.py    # Note linking tests
├── test_backlink_analyzer.py # Backlink tests
├── test_embedding_cache.py  # Embedding cache tests
├── test_vault_scanner.py    # Vault scanner tests
├── test_category_classifier.py # Category tests
└── test_api_server.py       # API endpoint tests
```

---

## Code Style

### Formatting

We use **Ruff** for linting and formatting:

```bash
# Check formatting
uv run ruff check src/

# Auto-fix issues
uv run ruff check --fix src/

# Format code
uv run ruff format src/
```

### Type Hints

All functions should have type hints:

```python
def process_note(
    raw_note: str,
    frontmatter: dict[str, Any] | None = None
) -> AgentResponse:
    """Process a raw note through the pipeline."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def find_related_notes(
    self,
    note_title: str,
    note_content: str,
    max_links: int = 5
) -> list[str]:
    """Find related notes and return as wikilinks.
    
    Args:
        note_title: Title of the note
        note_content: Content of the note
        max_links: Maximum number of related links
        
    Returns:
        List of wikilinks like ["[[Note A]]", "[[Note B]]"]
        
    Raises:
        ValueError: If note_title is empty
    """
```

---

## Project Structure

```
ana/
├── src/
│   ├── agent.py           # Main AtomicNoteArchitect class
│   ├── graph.py           # LangGraph workflow
│   ├── schemas.py         # Pydantic models
│   ├── config.py          # Configuration (pydantic-settings)
│   ├── llm_config.py      # LLM provider abstraction
│   ├── prompts.py         # System prompts
│   ├── link_analyzer.py   # 2-Stage Retrieval + Rerank
│   ├── backlink_analyzer.py # Bidirectional links
│   ├── embedding_cache.py # Embedding storage
│   ├── vault_scanner.py   # Obsidian vault access
│   ├── category_classifier.py # Note categorization
│   ├── template_manager.py # Template resolution
│   ├── validators.py      # Runtime validation
│   ├── logging_config.py  # Logging setup
│   ├── errors.py          # Custom exceptions
│   ├── utils.py           # Utilities
│   ├── cli/               # CLI commands
│   │   ├── main.py
│   │   ├── commands.py
│   │   ├── config_wizard.py
│   │   └── doctor.py
│   └── api/               # FastAPI server
│       └── server.py
├── tests/                 # Test files
├── templates/             # Note templates
├── data/                  # Data files
│   └── models/            # Local models
├── docs/                  # Documentation
└── obsidian-ana-plugin/   # Obsidian plugin
```

---

## Adding New Features

### 1. Create a New Module

```python
# src/my_feature.py
"""My Feature Module.

Description of what this module does.
"""

from src.logging_config import get_logger

logger = get_logger("my_feature")


class MyFeature:
    """Description of the class."""
    
    def __init__(self, config):
        self.config = config
        logger.debug("MyFeature initialized")
    
    def do_something(self) -> str:
        """Do something useful."""
        logger.info("Doing something")
        return "result"
```

### 2. Add Tests

```python
# tests/test_my_feature.py
"""Tests for My Feature Module."""

import pytest
from src.my_feature import MyFeature


class TestMyFeature:
    def test_do_something(self):
        feature = MyFeature(config=None)
        result = feature.do_something()
        assert result == "result"
```

### 3. Update Agent if Needed

```python
# In src/agent.py
from src.my_feature import MyFeature

class AtomicNoteArchitect:
    def __init__(self, config):
        ...
        self.my_feature = MyFeature(config)
```

---

## CLI Commands

### Development Commands

```bash
# Run doctor to check setup
ana doctor --debug

# Show configuration
ana config show

# Initialize configuration
ana config init

# Start API server
ana serve --reload

# Process a note
ana new

# Sync embeddings
ana sync
```

---

## Debugging

### Enable Debug Logging

```python
import logging
from src.logging_config import setup_logging

setup_logging(level="DEBUG")
```

### Common Issues

1. **Ollama connection error**: Ensure Ollama is running (`ollama serve`)
2. **Embedding failures**: Check model is pulled (`ollama pull nomic-embed-text`)
3. **API key missing**: Ensure `.env` file has required keys

---

## Pull Request Guidelines

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass (`uv run pytest`)
4. Update documentation if needed
5. Submit PR with clear description
