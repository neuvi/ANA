"""ANA Custom Error Module.

Provides user-friendly error messages with solution suggestions.
"""

from rich.console import Console
from rich.panel import Panel

console = Console()


class ANAError(Exception):
    """Base error class for ANA with user-friendly messages."""
    
    def __init__(self, message: str, solution: str | None = None, details: str | None = None):
        self.message = message
        self.solution = solution
        self.details = details
        super().__init__(message)
    
    def display(self) -> None:
        """Display error with rich formatting."""
        content = f"[bold red]❌ {self.message}[/bold red]"
        
        if self.details:
            content += f"\n\n[dim]{self.details}[/dim]"
        
        if self.solution:
            content += f"\n\n[bold green]💡 해결 방법:[/bold green]\n{self.solution}"
        
        console.print(Panel(content, title="오류 발생", border_style="red"))


class ConfigurationError(ANAError):
    """Configuration related errors."""
    pass


class APIKeyError(ConfigurationError):
    """API key related errors."""
    
    def __init__(self, provider: str):
        providers_info = {
            "openai": (
                "OPENAI_API_KEY",
                "https://platform.openai.com/api-keys 에서 API 키를 발급받으세요."
            ),
            "anthropic": (
                "ANTHROPIC_API_KEY", 
                "https://console.anthropic.com/settings/keys 에서 API 키를 발급받으세요."
            ),
            "ollama": (
                None,
                "Ollama는 API 키가 필요 없습니다. 서버가 실행 중인지 확인하세요:\n   ollama serve"
            ),
            "vllm": (
                None,
                "vLLM 서버가 실행 중인지 확인하세요."
            ),
        }
        
        env_var, guide = providers_info.get(provider, ("UNKNOWN", "문서를 참조하세요."))
        
        if env_var:
            message = f"{provider.upper()} API 키가 설정되지 않았습니다."
            solution = (
                f"1. .env 파일에 {env_var}를 설정하세요\n"
                f"2. {guide}\n"
                f"3. 또는 무료로 Ollama를 사용하세요: ana config set llm_provider ollama"
            )
        else:
            message = f"{provider.upper()} 연결에 실패했습니다."
            solution = guide
            
        super().__init__(message, solution)
        self.provider = provider


class VaultPathError(ConfigurationError):
    """Vault path related errors."""
    
    def __init__(self, path: str, reason: str = "not_found"):
        reasons = {
            "not_found": f"Vault 경로를 찾을 수 없습니다: {path}",
            "not_directory": f"Vault 경로가 디렉토리가 아닙니다: {path}",
            "no_permission": f"Vault 경로에 접근 권한이 없습니다: {path}",
        }
        
        message = reasons.get(reason, f"Vault 경로 오류: {path}")
        solution = (
            "1. Obsidian에서 Vault 위치를 확인하세요 (설정 > 파일 및 링크 > Vault 위치)\n"
            "2. .env 파일에서 ANA_VAULT_PATH를 올바른 경로로 수정하세요\n"
            "3. 또는 다음 명령어로 설정: ana config set vault_path /your/vault/path"
        )
        
        super().__init__(message, solution)
        self.path = path
        self.reason = reason


class LLMConnectionError(ANAError):
    """LLM connection related errors."""
    
    def __init__(self, provider: str, base_url: str | None = None):
        message = f"{provider.upper()} 서버에 연결할 수 없습니다."
        
        if provider == "ollama":
            solution = (
                "1. Ollama가 설치되어 있는지 확인: ollama --version\n"
                "2. Ollama 서버 시작: ollama serve\n"
                "3. 모델 다운로드: ollama pull llama3.1:8b"
            )
        elif provider == "vllm":
            solution = (
                f"1. vLLM 서버가 실행 중인지 확인 ({base_url or 'http://localhost:8000'})\n"
                "2. 서버 로그에서 오류를 확인하세요"
            )
        else:
            solution = (
                "1. 인터넷 연결을 확인하세요\n"
                "2. API 키가 올바른지 확인하세요\n"
                "3. 서비스 상태 페이지를 확인하세요"
            )
        
        if base_url:
            details = f"연결 시도: {base_url}"
        else:
            details = None
            
        super().__init__(message, solution, details)
        self.provider = provider
        self.base_url = base_url


class TemplateError(ANAError):
    """Template related errors."""
    
    def __init__(self, template_name: str, reason: str = "not_found"):
        message = f"템플릿 오류: {template_name}"
        solution = (
            "1. templates/ 디렉토리에 템플릿이 있는지 확인하세요\n"
            "2. data/templates.json 파일이 올바른지 확인하세요"
        )
        super().__init__(message, solution)


class EmbeddingError(ANAError):
    """Embedding related errors."""
    
    def __init__(self, reason: str = "model_not_found"):
        if reason == "model_not_found":
            message = "임베딩 모델을 찾을 수 없습니다."
            solution = (
                "1. Ollama에서 임베딩 모델 다운로드:\n"
                "   ollama pull nomic-embed-text\n"
                "2. 또는 다른 모델 사용:\n"
                "   ana config set embedding_model mxbai-embed-large"
            )
        else:
            message = "임베딩 처리 중 오류가 발생했습니다."
            solution = "자세한 오류 내용은 --debug 플래그로 확인하세요."
        super().__init__(message, solution)


class LLMParseError(ANAError):
    """LLM response parsing errors."""
    
    def __init__(self, raw_response: str | None = None, expected_format: str = "JSON"):
        message = f"LLM 응답을 {expected_format} 형식으로 파싱할 수 없습니다."
        solution = (
            "1. LLM 모델이 올바른 형식으로 응답하는지 확인하세요\n"
            "2. 프롬프트에 형식 지시가 명확한지 확인하세요\n"
            "3. 다른 LLM 모델을 사용해 보세요"
        )
        
        details = None
        if raw_response:
            # Truncate long responses
            truncated = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
            details = f"받은 응답: {truncated}"
        
        super().__init__(message, solution, details)
        self.raw_response = raw_response
        self.expected_format = expected_format


class RerankerError(ANAError):
    """Reranker model related errors."""
    
    def __init__(self, model_name: str, reason: str = "model_not_found"):
        if reason == "model_not_found":
            message = f"Reranker 모델을 찾을 수 없습니다: {model_name}"
            solution = (
                "1. sentence-transformers가 설치되어 있는지 확인:\\n"
                "   pip install sentence-transformers\\n"
                "2. 모델 이름이 올바른지 확인:\\n"
                "   cross-encoder/ms-marco-MiniLM-L-6-v2\\n"
                "3. 인터넷 연결을 확인하세요 (첫 실행 시 모델 다운로드 필요)"
            )
        elif reason == "prediction_failed":
            message = f"Reranker 예측 실패: {model_name}"
            solution = "입력 데이터가 올바른지 확인하세요."
        else:
            message = f"Reranker 오류: {model_name}"
            solution = "자세한 오류 내용은 --debug 플래그로 확인하세요."
        
        super().__init__(message, solution)
        self.model_name = model_name
        self.reason = reason


def handle_error(error: Exception) -> None:
    """Generic error handler that displays user-friendly messages."""
    if isinstance(error, ANAError):
        error.display()
    else:
        # Wrap unknown errors
        ana_error = ANAError(
            message=str(error),
            solution="--debug 플래그로 다시 실행하여 자세한 오류를 확인하세요.",
            details=error.__class__.__name__
        )
        ana_error.display()
