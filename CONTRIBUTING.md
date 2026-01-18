# Contributing to ANA

Thank you for your interest in contributing to ANA (Atomic Note Architect)! 🎉

## 📋 Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## 🛠️ Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/ana.git
cd ana

# Install dependencies with dev extras
uv sync --group dev

# Or with pip
pip install -e ".[dev]"
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
nano .env
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_agent.py

# Run with coverage
pytest --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_agent.py            # Agent tests
├── test_graph.py            # LangGraph workflow tests
├── test_link_analyzer.py    # Note linking tests
├── test_vault_scanner.py    # Vault scanner tests
└── test_category_classifier.py  # Category tests
```

## 📝 Code Style

We follow Python best practices:

- **Type Hints**: Use type annotations for function parameters and return values
- **Docstrings**: Use Google-style docstrings
- **Line Length**: 100 characters max
- **Formatting**: Follow PEP 8

### Example

```python
def process_note(
    raw_note: str,
    category: str | None = None
) -> DraftNote:
    """Process a raw note into an atomic note.
    
    Args:
        raw_note: Raw note content as string
        category: Optional category override
        
    Returns:
        Processed DraftNote object
        
    Raises:
        ValueError: If raw_note is empty
    """
    if not raw_note:
        raise ValueError("raw_note cannot be empty")
    ...
```

### Linting (Optional)

```bash
# Install ruff
pip install ruff

# Check code
ruff check src/

# Auto-fix issues
ruff check src/ --fix
```

## 🚀 Submitting Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run tests
pytest

# Test CLI locally
ana doctor
ana new --help
```

### 4. Commit

Write clear commit messages:

```bash
git add .
git commit -m "feat: add multi-language support for prompts"
# or
git commit -m "fix: handle empty vault in link analyzer"
```

Commit message prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Test changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

## 🐛 Reporting Issues

### Bug Reports

Please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Step-by-step instructions
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - OS (Windows/macOS/Linux)
   - Python version
   - ANA version (`ana --version`)
   - LLM provider (OpenAI/Ollama/etc.)

### Feature Requests

Describe:
1. **Problem**: What problem does this solve?
2. **Solution**: Your proposed solution
3. **Alternatives**: Other approaches you've considered

## 📚 Project Structure

```
ana/
├── src/
│   ├── agent.py          # Main agent class
│   ├── graph.py          # LangGraph workflow
│   ├── config.py         # Configuration
│   ├── prompts.py        # LLM prompts
│   ├── schemas.py        # Data models
│   ├── link_analyzer.py  # Note linking
│   ├── cli/              # CLI commands
│   └── api/              # FastAPI server
├── tests/                # Test suite
├── obsidian-ana-plugin/  # Obsidian plugin
└── templates/            # Note templates
```

## 🙏 Thank You!

Every contribution makes ANA better. We appreciate your time and effort!

---

Questions? Open an issue or start a discussion. Happy coding! 🚀
