# ANA - Atomic Note Architect Makefile
# ==================================

.PHONY: help install dev test doctor clean sync

# Default target
help:
	@echo "🏛️ ANA - Atomic Note Architect"
	@echo ""
	@echo "사용 가능한 명령어:"
	@echo "  make install   - 의존성 설치"
	@echo "  make dev       - 개발 환경 설치"
	@echo "  make test      - 테스트 실행"
	@echo "  make doctor    - 환경 진단"
	@echo "  make sync      - 임베딩 동기화"
	@echo "  make clean     - 캐시 정리"
	@echo ""
	@echo "빠른 시작:"
	@echo "  1. make install"
	@echo "  2. cp .env.example .env"
	@echo "  3. ana config init"
	@echo "  4. ana doctor"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@if command -v uv > /dev/null 2>&1; then \
		uv sync; \
	else \
		pip install -e .; \
	fi
	@echo "✅ Installation complete!"
	@echo ""
	@echo "다음 단계:"
	@echo "  1. cp .env.example .env"
	@echo "  2. ana config init"

# Install with dev dependencies
dev:
	@echo "📦 Installing with dev dependencies..."
	@if command -v uv > /dev/null 2>&1; then \
		uv sync --group dev; \
	else \
		pip install -e ".[dev]"; \
	fi
	@echo "✅ Dev installation complete!"

# Run tests
test:
	@echo "🧪 Running tests..."
	@pytest -v

# Run doctor
doctor:
	@echo "🩺 Running diagnostics..."
	@ana doctor

# Sync embeddings
sync:
	@echo "🔄 Syncing embeddings..."
	@ana sync

# Clean cache and build files
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf __pycache__ .pytest_cache .mypy_cache
	@rm -rf src/__pycache__ src/*/__pycache__
	@rm -rf dist build *.egg-info
	@rm -rf data/embeddings_cache.json
	@echo "✅ Cleanup complete!"

# Setup for first-time users
setup: install
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "📝 .env file created from template"; \
	fi
	@echo ""
	@echo "🧙 설정을 시작하려면 다음 명령어를 실행하세요:"
	@echo "   ana config init"
