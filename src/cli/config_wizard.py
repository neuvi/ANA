"""ANA Configuration Wizard.

Interactive configuration setup for first-time users.
"""

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()


def run_config_wizard() -> None:
    """Run the interactive configuration wizard."""
    console.print(Panel.fit(
        "[bold blue]🧙 ANA 설정 마법사[/bold blue]\n"
        "[dim]처음 사용하시는 분을 위한 대화형 설정 도우미입니다[/dim]",
        border_style="blue"
    ))
    console.print()
    
    config_values = {}
    
    # Step 1: LLM Provider
    console.print("[bold]1️⃣  LLM 제공자 선택[/bold]")
    console.print()
    
    provider_table = Table(show_header=True, header_style="bold cyan")
    provider_table.add_column("옵션", width=10)
    provider_table.add_column("제공자", width=15)
    provider_table.add_column("설명", width=40)
    provider_table.add_row("1", "OpenAI", "GPT-4o, o3 등 (API 키 필요)")
    provider_table.add_row("2", "Anthropic", "Claude 3.5 Sonnet (API 키 필요)")
    provider_table.add_row("3", "Ollama", "로컬 LLM (무료, 설치 필요)")
    provider_table.add_row("4", "vLLM", "로컬 vLLM 서버")
    console.print(provider_table)
    console.print()
    
    provider_choice = Prompt.ask(
        "선택하세요",
        choices=["1", "2", "3", "4"],
        default="1"
    )
    
    provider_map = {"1": "openai", "2": "anthropic", "3": "ollama", "4": "vllm"}
    config_values["ANA_LLM_PROVIDER"] = provider_map[provider_choice]
    
    console.print()
    
    # Step 2: API Key or Local Settings
    provider = config_values["ANA_LLM_PROVIDER"]
    
    if provider == "openai":
        console.print("[bold]2️⃣  OpenAI API 키 설정[/bold]")
        console.print("[dim]https://platform.openai.com/api-keys 에서 발급받으세요[/dim]")
        console.print()
        
        api_key = Prompt.ask("OpenAI API Key", password=True)
        config_values["OPENAI_API_KEY"] = api_key
        
        model = Prompt.ask(
            "모델 선택",
            default="gpt-4o",
            show_default=True
        )
        config_values["ANA_LLM_MODEL"] = model
        
    elif provider == "anthropic":
        console.print("[bold]2️⃣  Anthropic API 키 설정[/bold]")
        console.print("[dim]https://console.anthropic.com/settings/keys 에서 발급받으세요[/dim]")
        console.print()
        
        api_key = Prompt.ask("Anthropic API Key", password=True)
        config_values["ANTHROPIC_API_KEY"] = api_key
        
        model = Prompt.ask(
            "모델 선택",
            default="claude-3-5-sonnet-20241022",
            show_default=True
        )
        config_values["ANA_LLM_MODEL"] = model
        
    elif provider == "ollama":
        console.print("[bold]2️⃣  Ollama 설정[/bold]")
        console.print("[dim]Ollama가 설치되어 있어야 합니다: https://ollama.ai[/dim]")
        console.print()
        
        base_url = Prompt.ask(
            "Ollama 서버 URL",
            default="http://localhost:11434",
            show_default=True
        )
        config_values["ANA_OLLAMA_BASE_URL"] = base_url
        
        model = Prompt.ask(
            "모델 선택",
            default="llama3.1:8b",
            show_default=True
        )
        config_values["ANA_OLLAMA_MODEL"] = model
        
    elif provider == "vllm":
        console.print("[bold]2️⃣  vLLM 설정[/bold]")
        console.print()
        
        base_url = Prompt.ask(
            "vLLM 서버 URL",
            default="http://localhost:8000/v1",
            show_default=True
        )
        config_values["ANA_VLLM_BASE_URL"] = base_url
        
        model = Prompt.ask(
            "모델 이름",
            default="meta-llama/Llama-3.1-8B-Instruct",
            show_default=True
        )
        config_values["ANA_VLLM_MODEL"] = model
    
    console.print()
    
    # Step 3: Vault Path
    console.print("[bold]3️⃣  Obsidian Vault 경로[/bold]")
    console.print("[dim]Obsidian 설정 > 파일 및 링크 > Vault 위치에서 확인 가능[/dim]")
    console.print()
    
    default_vault = Path.home() / "Obsidian"
    vault_path = Prompt.ask(
        "Vault 경로",
        default=str(default_vault) if default_vault.exists() else "~/vault"
    )
    config_values["ANA_VAULT_PATH"] = vault_path
    
    console.print()
    
    # Step 4: Additional Settings
    console.print("[bold]4️⃣  추가 설정 (선택사항)[/bold]")
    console.print()
    
    if Confirm.ask("고급 설정을 구성하시겠습니까?", default=False):
        max_questions = Prompt.ask("라운드당 최대 질문 수", default="5")
        config_values["ANA_MAX_QUESTIONS"] = max_questions
        
        max_iterations = Prompt.ask("최대 질문 라운드 수", default="3")
        config_values["ANA_MAX_ITERATIONS"] = max_iterations
        
        temperature = Prompt.ask("LLM Temperature (0.0-2.0)", default="0.7")
        config_values["ANA_LLM_TEMPERATURE"] = temperature
    
    console.print()
    
    # Summary and confirmation
    console.print("[bold]📋 설정 요약[/bold]")
    console.print()
    
    summary_table = Table(show_header=True, header_style="bold")
    summary_table.add_column("설정", width=25)
    summary_table.add_column("값", width=40)
    
    # Show non-sensitive values
    for key, value in config_values.items():
        display_value = value
        if "KEY" in key and value:
            display_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "****"
        summary_table.add_row(key, display_value)
    
    console.print(summary_table)
    console.print()
    
    if Confirm.ask("이 설정으로 .env 파일을 생성하시겠습니까?", default=True):
        _write_env_file(config_values)
        console.print()
        console.print(Panel(
            "[bold green]✅ 설정 완료![/bold green]\n\n"
            ".env 파일이 생성되었습니다.\n"
            "이제 [bold]ana new[/bold] 명령으로 노트를 생성할 수 있습니다.",
            border_style="green"
        ))
    else:
        console.print("[yellow]설정이 취소되었습니다.[/yellow]")


def _write_env_file(config_values: dict[str, str]) -> None:
    """Write configuration to .env file."""
    env_path = Path(".env")
    
    # Read existing content if file exists
    existing_lines = []
    existing_keys = set()
    
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.split("=", 1)[0]
                    if key not in config_values:
                        existing_lines.append(line)
                    existing_keys.add(key)
                elif stripped.startswith("#") or not stripped:
                    existing_lines.append(line)
    
    # Write new file
    with open(env_path, "w", encoding="utf-8") as f:
        # Write header if new file
        if not existing_lines:
            f.write("# ANA Configuration\n")
            f.write("# Generated by ana config init\n")
            f.write("\n")
        else:
            for line in existing_lines:
                f.write(line)
            f.write("\n")
        
        # Write new values
        for key, value in config_values.items():
            f.write(f"{key}={value}\n")


def show_current_config() -> None:
    """Display current configuration."""
    console.print(Panel.fit(
        "[bold blue]⚙️  현재 ANA 설정[/bold blue]",
        border_style="blue"
    ))
    console.print()
    
    try:
        from src.config import ANAConfig
        config = ANAConfig()
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("설정", width=25)
        table.add_column("값", width=45)
        
        table.add_row("LLM Provider", config.llm_provider)
        table.add_row("LLM Model", config.llm_model)
        table.add_row("Temperature", str(config.llm_temperature))
        table.add_row("Vault Path", str(config.vault_path))
        table.add_row("Max Questions", str(config.max_questions))
        table.add_row("Max Iterations", str(config.max_iterations))
        table.add_row("Note Linking", "✅ Enabled" if config.enable_note_linking else "❌ Disabled")
        table.add_row("Embedding Model", config.embedding_model)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]설정을 불러올 수 없습니다: {e}[/red]")
        console.print("[dim]ana config init 명령으로 설정을 생성하세요.[/dim]")


def set_config_value(key: str, value: str) -> None:
    """Set a configuration value in .env file."""
    env_path = Path(".env")
    
    # Map friendly names to env var names
    key_map = {
        "llm_provider": "ANA_LLM_PROVIDER",
        "llm_model": "ANA_LLM_MODEL",
        "vault_path": "ANA_VAULT_PATH",
        "temperature": "ANA_LLM_TEMPERATURE",
        "max_questions": "ANA_MAX_QUESTIONS",
        "max_iterations": "ANA_MAX_ITERATIONS",
        "embedding_model": "ANA_EMBEDDING_MODEL",
    }
    
    env_key = key_map.get(key.lower(), key.upper())
    if not env_key.startswith("ANA_") and env_key not in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        env_key = f"ANA_{env_key}"
    
    # Read and update
    lines = []
    key_found = False
    
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    line_key = stripped.split("=", 1)[0]
                    if line_key == env_key:
                        lines.append(f"{env_key}={value}\n")
                        key_found = True
                        continue
                lines.append(line)
    
    if not key_found:
        lines.append(f"{env_key}={value}\n")
    
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    console.print(f"[green]✅ {env_key}={value}[/green]")
