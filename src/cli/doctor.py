"""ANA Doctor - Environment Diagnostics.

Diagnose and verify ANA installation and configuration.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class DiagnosticResult:
    """Result of a diagnostic check."""
    
    def __init__(self, name: str, status: str, message: str, fix_hint: str | None = None):
        self.name = name
        self.status = status  # "ok", "warning", "error"
        self.message = message
        self.fix_hint = fix_hint
    
    @property
    def icon(self) -> str:
        icons = {"ok": "✅", "warning": "⚠️", "error": "❌"}
        return icons.get(self.status, "❓")
    
    @property
    def color(self) -> str:
        colors = {"ok": "green", "warning": "yellow", "error": "red"}
        return colors.get(self.status, "white")


def run_doctor(fix: bool = False, debug: bool = False) -> None:
    """Run all diagnostic checks."""
    console.print(Panel.fit(
        "[bold blue]🩺 ANA Doctor - 환경 진단[/bold blue]\n"
        "[dim]시스템 설정을 확인하고 문제를 진단합니다[/dim]",
        border_style="blue"
    ))
    console.print()
    
    results = []
    
    # Run all checks
    results.append(check_python_version())
    results.append(check_dependencies())
    results.append(check_env_file())
    results.append(check_vault_path())
    results.append(check_llm_provider())
    results.append(check_api_keys())
    results.append(check_ollama())
    results.append(check_embedding_model())
    
    # New checks
    results.append(check_config_values())
    results.append(check_reranker())
    
    # Debug-only checks (slow)
    if debug:
        console.print("[dim]🔍 Debug 모드: LLM 연결 테스트 실행 중...[/dim]")
        results.append(check_llm_connection())
    
    # Display results
    table = Table(show_header=True, header_style="bold")
    table.add_column("상태", width=4)
    table.add_column("검사 항목", width=25)
    table.add_column("결과", width=45)
    
    for result in results:
        table.add_row(
            result.icon,
            f"[{result.color}]{result.name}[/{result.color}]",
            result.message
        )
    
    console.print(table)
    console.print()
    
    # Show fixes if there are errors
    errors = [r for r in results if r.status == "error"]
    warnings = [r for r in results if r.status == "warning"]
    
    if errors:
        console.print("[bold red]❌ 오류가 발견되었습니다:[/bold red]")
        console.print()
        for error in errors:
            if error.fix_hint:
                console.print(f"  • {error.name}: {error.fix_hint}")
        console.print()
    
    if warnings:
        console.print("[bold yellow]⚠️  경고:[/bold yellow]")
        console.print()
        for warning in warnings:
            if warning.fix_hint:
                console.print(f"  • {warning.name}: {warning.fix_hint}")
        console.print()
    
    if not errors and not warnings:
        console.print(Panel(
            "[bold green]✅ 모든 검사를 통과했습니다![/bold green]\n\n"
            "ANA를 사용할 준비가 되었습니다.\n"
            "[bold]ana new[/bold] 명령으로 시작하세요.",
            border_style="green"
        ))
    elif errors:
        console.print("[dim]문제를 해결한 후 다시 ana doctor를 실행하세요.[/dim]")
    
    if not debug:
        console.print("[dim]LLM 연결 테스트: ana doctor --debug[/dim]")


def check_python_version() -> DiagnosticResult:
    """Check Python version."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 10:
        return DiagnosticResult(
            "Python 버전",
            "ok",
            f"Python {version_str}"
        )
    else:
        return DiagnosticResult(
            "Python 버전",
            "error",
            f"Python {version_str} (3.10+ 필요)",
            fix_hint="Python 3.10 이상을 설치하세요"
        )


def check_dependencies() -> DiagnosticResult:
    """Check if required packages are installed."""
    required = ["langchain", "langgraph", "rich", "click", "pydantic"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if not missing:
        return DiagnosticResult(
            "의존성 패키지",
            "ok",
            "모든 필수 패키지 설치됨"
        )
    else:
        return DiagnosticResult(
            "의존성 패키지",
            "error",
            f"누락: {', '.join(missing)}",
            fix_hint="uv sync 또는 pip install -e . 실행"
        )


def check_env_file() -> DiagnosticResult:
    """Check if .env file exists."""
    env_path = Path(".env")
    example_path = Path(".env.example")
    
    if env_path.exists():
        return DiagnosticResult(
            ".env 파일",
            "ok",
            "설정 파일 존재"
        )
    elif example_path.exists():
        return DiagnosticResult(
            ".env 파일",
            "warning",
            ".env 파일 없음 (.env.example 존재)",
            fix_hint="cp .env.example .env 실행 후 설정"
        )
    else:
        return DiagnosticResult(
            ".env 파일",
            "error",
            ".env 파일 없음",
            fix_hint="ana config init 으로 생성"
        )


def check_vault_path() -> DiagnosticResult:
    """Check if vault path is valid."""
    try:
        from src.config import ANAConfig
        config = ANAConfig()
        vault_path = config.get_vault_path()
        
        if vault_path.exists() and vault_path.is_dir():
            # Check if it looks like an Obsidian vault
            obsidian_dir = vault_path / ".obsidian"
            if obsidian_dir.exists():
                return DiagnosticResult(
                    "Vault 경로",
                    "ok",
                    f"{vault_path} (Obsidian vault 확인됨)"
                )
            else:
                return DiagnosticResult(
                    "Vault 경로",
                    "warning",
                    f"{vault_path} (.obsidian 폴더 없음)",
                    fix_hint="Obsidian vault 경로가 맞는지 확인"
                )
        else:
            return DiagnosticResult(
                "Vault 경로",
                "error",
                f"경로를 찾을 수 없음: {vault_path}",
                fix_hint="ana config set vault_path /your/path"
            )
    except Exception as e:
        return DiagnosticResult(
            "Vault 경로",
            "error",
            f"설정 오류: {e}",
            fix_hint="ana config init 실행"
        )


def check_llm_provider() -> DiagnosticResult:
    """Check LLM provider configuration."""
    try:
        from src.config import ANAConfig
        config = ANAConfig()
        
        valid_providers = ["openai", "anthropic", "ollama", "vllm"]
        if config.llm_provider in valid_providers:
            # Get the correct model name based on provider
            if config.llm_provider == "ollama":
                model_name = config.ollama_model
            elif config.llm_provider == "vllm":
                model_name = config.vllm_model
            else:
                model_name = config.llm_model
            
            return DiagnosticResult(
                "LLM Provider",
                "ok",
                f"{config.llm_provider} (모델: {model_name})"
            )
        else:
            return DiagnosticResult(
                "LLM Provider",
                "error",
                f"잘못된 provider: {config.llm_provider}",
                fix_hint=f"유효한 값: {', '.join(valid_providers)}"
            )
    except Exception as e:
        return DiagnosticResult(
            "LLM Provider",
            "error",
            f"설정 오류: {e}",
            fix_hint="ana config init 실행"
        )


def check_api_keys() -> DiagnosticResult:
    """Check if required API keys are set."""
    import os
    from dotenv import load_dotenv
    
    # Load .env file to ensure environment variables are set
    load_dotenv()
    
    try:
        from src.config import ANAConfig
        config = ANAConfig()
        
        if config.llm_provider == "openai":
            key = os.environ.get("OPENAI_API_KEY", "")
            if key and key.startswith("sk-"):
                return DiagnosticResult(
                    "API Key",
                    "ok",
                    f"OpenAI: {key[:8]}...{key[-4:]}"
                )
            else:
                return DiagnosticResult(
                    "API Key",
                    "error",
                    "OPENAI_API_KEY 미설정",
                    fix_hint="https://platform.openai.com/api-keys"
                )
                
        elif config.llm_provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            if key:
                return DiagnosticResult(
                    "API Key",
                    "ok",
                    f"Anthropic: {key[:8]}..."
                )
            else:
                return DiagnosticResult(
                    "API Key",
                    "error",
                    "ANTHROPIC_API_KEY 미설정",
                    fix_hint="https://console.anthropic.com/settings/keys"
                )
                
        else:
            return DiagnosticResult(
                "API Key",
                "ok",
                "로컬 LLM 사용 (API 키 불필요)"
            )
            
    except Exception as e:
        return DiagnosticResult(
            "API Key",
            "warning",
            f"확인 불가: {e}"
        )


def check_ollama() -> DiagnosticResult:
    """Check Ollama installation and status.
    
    Supports both local installation and Docker-based Ollama.
    """
    try:
        from src.config import ANAConfig
        config = ANAConfig()
        
        # Check if ollama command exists locally
        ollama_path = shutil.which("ollama")
        
        # Try API connection first (supports both local and Docker)
        try:
            import requests
            base_url = config.ollama_base_url
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check if required model is available
                required_model = config.ollama_model if config.llm_provider == "ollama" else config.embedding_model
                if any(required_model in name for name in model_names):
                    source = "로컬" if ollama_path else "Docker/원격"
                    return DiagnosticResult(
                        "Ollama",
                        "ok",
                        f"실행 중 - {source} (모델 {len(models)}개)"
                    )
                else:
                    return DiagnosticResult(
                        "Ollama",
                        "warning",
                        f"모델 '{required_model}' 없음",
                        fix_hint=f"ollama pull {required_model}"
                    )
            else:
                return DiagnosticResult(
                    "Ollama",
                    "error",
                    "서버 응답 오류",
                    fix_hint="ollama serve 실행"
                )
        except Exception:
            # API connection failed, check local installation
            if not ollama_path:
                if config.llm_provider == "ollama":
                    return DiagnosticResult(
                        "Ollama",
                        "error",
                        f"연결 불가 ({config.ollama_base_url})",
                        fix_hint="로컬: ollama serve 실행 / Docker: 포트 매핑 확인 (-p 11434:11434)"
                    )
                else:
                    return DiagnosticResult(
                        "Ollama",
                        "warning",
                        "Ollama 미설치 (임베딩에 필요할 수 있음)",
                        fix_hint="https://ollama.ai 에서 설치"
                    )
            else:
                return DiagnosticResult(
                    "Ollama",
                    "error",
                    "서버에 연결할 수 없음",
                    fix_hint="ollama serve 실행"
                )
            
    except Exception as e:
        return DiagnosticResult(
            "Ollama",
            "warning",
            f"확인 불가: {e}"
        )


def check_embedding_model() -> DiagnosticResult:
    """Check embedding model availability."""
    try:
        from src.config import ANAConfig
        config = ANAConfig()
        
        if not config.enable_note_linking:
            return DiagnosticResult(
                "임베딩 모델",
                "ok",
                "노트 링킹 비활성화됨"
            )
        
        model = config.embedding_model
        
        # Check if ollama is available for embedding
        try:
            import requests
            response = requests.get(f"{config.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                if any(model in name for name in model_names):
                    return DiagnosticResult(
                        "임베딩 모델",
                        "ok",
                        f"{model} 사용 가능"
                    )
                else:
                    return DiagnosticResult(
                        "임베딩 모델",
                        "warning",
                        f"{model} 미설치",
                        fix_hint=f"ollama pull {model}"
                    )
        except Exception:
            return DiagnosticResult(
                "임베딩 모델",
                "warning",
                "Ollama 연결 필요",
                fix_hint="ollama serve 실행 후 ollama pull nomic-embed-text"
            )
            
    except Exception as e:
        return DiagnosticResult(
            "임베딩 모델",
            "warning",
            f"확인 불가: {e}"
        )


def check_config_values() -> DiagnosticResult:
    """Validate configuration values."""
    try:
        from src.config import ANAConfig
        config = ANAConfig()
        
        issues = []
        
        # Check numeric ranges
        if config.max_questions < 1 or config.max_questions > 10:
            issues.append(f"max_questions={config.max_questions} (1-10 권장)")
        
        if config.max_iterations < 1 or config.max_iterations > 5:
            issues.append(f"max_iterations={config.max_iterations} (1-5 권장)")
        
        if config.max_related_links < 1 or config.max_related_links > 10:
            issues.append(f"max_related_links={config.max_related_links} (1-10 권장)")
        
        # Check temperature
        if config.llm_temperature < 0 or config.llm_temperature > 2:
            issues.append(f"llm_temperature={config.llm_temperature} (0-2 범위)")
        
        # Check language
        valid_langs = ["ko", "en", "ja", "zh"]
        if config.output_language not in valid_langs:
            issues.append(f"output_language={config.output_language}")
        
        # Check batch size
        if config.embedding_batch_size < 1 or config.embedding_batch_size > 100:
            issues.append(f"embedding_batch_size={config.embedding_batch_size} (1-100 범위)")
        
        if not issues:
            return DiagnosticResult(
                "설정 값 유효성",
                "ok",
                "모든 설정 값 정상"
            )
        else:
            return DiagnosticResult(
                "설정 값 유효성",
                "warning",
                f"{len(issues)}개 주의 필요",
                fix_hint="; ".join(issues[:3])  # Limit hint length
            )
    except Exception as e:
        return DiagnosticResult(
            "설정 값 유효성",
            "warning",
            f"확인 불가: {e}"
        )


def check_reranker() -> DiagnosticResult:
    """Check reranker model availability."""
    try:
        from src.config import ANAConfig
        config = ANAConfig()
        
        if not config.rerank_enabled:
            return DiagnosticResult(
                "Reranker 모델",
                "ok",
                "비활성화됨"
            )
        
        model_name = config.rerank_model
        
        try:
            from sentence_transformers import CrossEncoder
            
            # Check if model is already downloaded (don't download here)
            import os
            from pathlib import Path
            
            # Check common cache locations
            cache_dirs = [
                Path.home() / ".cache" / "huggingface" / "hub",
                Path.home() / ".cache" / "torch" / "sentence_transformers",
            ]
            
            model_folder = model_name.replace("/", "_")
            model_found = False
            
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    for item in cache_dir.iterdir():
                        if model_folder in str(item) or model_name.split("/")[-1] in str(item):
                            model_found = True
                            break
            
            if model_found:
                return DiagnosticResult(
                    "Reranker 모델",
                    "ok",
                    f"{model_name.split('/')[-1]} 설치됨"
                )
            else:
                return DiagnosticResult(
                    "Reranker 모델",
                    "warning",
                    f"{model_name.split('/')[-1]} 미설치",
                    fix_hint="첫 실행 시 자동 다운로드됨"
                )
                
        except ImportError:
            return DiagnosticResult(
                "Reranker 모델",
                "warning",
                "sentence-transformers 미설치",
                fix_hint="pip install sentence-transformers"
            )
            
    except Exception as e:
        return DiagnosticResult(
            "Reranker 모델",
            "warning",
            f"확인 불가: {e}"
        )


def check_llm_connection() -> DiagnosticResult:
    """Test actual LLM connection with a simple prompt."""
    try:
        from src.config import ANAConfig
        from src.llm_config import get_llm
        
        config = ANAConfig()
        llm = get_llm(config)
        
        # Simple test prompt
        response = llm.invoke("Say 'OK' if you can hear me. Reply with only 'OK'.")
        
        if response and hasattr(response, 'content'):
            content = response.content.strip()[:20]
            return DiagnosticResult(
                "LLM 연결 테스트",
                "ok",
                f"{config.llm_provider} 응답: '{content}'"
            )
        else:
            return DiagnosticResult(
                "LLM 연결 테스트",
                "error",
                "응답 없음",
                fix_hint="API 키와 설정 확인"
            )
    except Exception as e:
        error_msg = str(e)[:50]
        return DiagnosticResult(
            "LLM 연결 테스트",
            "error",
            f"연결 실패: {error_msg}",
            fix_hint="API 키, 네트워크, 서버 상태 확인"
        )

