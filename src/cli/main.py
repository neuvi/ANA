"""ANA CLI Main Entry Point.

Click-based CLI with subcommands for better user experience.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from src.errors import handle_error

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="ANA")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """🏛️ ANA - Atomic Note Architect
    
    Transform raw notes into Zettelkasten-style atomic notes.
    
    Examples:
    
    \b
      ana new                 # Interactive mode
      ana process note.txt    # Process a file
      ana config init         # Setup wizard
      ana doctor              # Check environment
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@cli.command()
@click.argument("input_file", type=click.Path(exists=True), required=False)
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--no-interactive", is_flag=True, help="Non-interactive mode")
@click.pass_context
def new(ctx: click.Context, input_file: str | None, output: str | None, no_interactive: bool) -> None:
    """Create a new atomic note (interactive mode).
    
    If INPUT_FILE is provided, process that file.
    Otherwise, enter interactive mode for note input.
    """
    from src.cli.commands import run_new_command
    
    try:
        run_new_command(
            input_file=input_file,
            output_dir=output,
            no_interactive=no_interactive,
            debug=ctx.obj.get("debug", False)
        )
    except Exception as e:
        if ctx.obj.get("debug"):
            raise
        handle_error(e)
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--no-interactive", is_flag=True, help="Non-interactive mode")
@click.pass_context
def process(ctx: click.Context, input_file: str, output: str | None, no_interactive: bool) -> None:
    """Process a note file into atomic notes.
    
    INPUT_FILE: Path to the raw note file to process.
    """
    from src.cli.commands import run_new_command
    
    try:
        run_new_command(
            input_file=input_file,
            output_dir=output,
            no_interactive=no_interactive,
            debug=ctx.obj.get("debug", False)
        )
    except Exception as e:
        if ctx.obj.get("debug"):
            raise
        handle_error(e)
        sys.exit(1)


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Force full resync")
@click.pass_context
def sync(ctx: click.Context, force: bool) -> None:
    """Sync embeddings for vault notes.
    
    Updates embeddings only for changed files unless --force is used.
    """
    from src.cli.commands import run_sync_command
    
    try:
        run_sync_command(force=force, debug=ctx.obj.get("debug", False))
    except Exception as e:
        if ctx.obj.get("debug"):
            raise
        handle_error(e)
        sys.exit(1)


@cli.group()
def config() -> None:
    """Manage ANA configuration.
    
    \b
    Subcommands:
      init    Interactive setup wizard
      show    Display current configuration
      set     Set a configuration value
    """
    pass


@config.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Interactive configuration wizard.
    
    Guides you through setting up ANA for the first time.
    """
    from src.cli.config_wizard import run_config_wizard
    
    try:
        run_config_wizard()
    except Exception as e:
        if ctx.obj.get("debug"):
            raise
        handle_error(e)
        sys.exit(1)


@config.command()
def show() -> None:
    """Display current configuration."""
    from src.cli.config_wizard import show_current_config
    show_current_config()


@config.command()
@click.argument("key")
@click.argument("value")
def set(key: str, value: str) -> None:
    """Set a configuration value.
    
    \b
    Examples:
      ana config set llm_provider ollama
      ana config set vault_path ~/Documents/Obsidian
    """
    from src.cli.config_wizard import set_config_value
    set_config_value(key, value)


@cli.command()
@click.option("--fix", is_flag=True, help="Attempt to fix issues automatically")
@click.pass_context
def doctor(ctx: click.Context, fix: bool) -> None:
    """Diagnose environment and configuration.
    
    Checks:
    - Python version
    - Dependencies
    - API keys
    - Vault path
    - LLM connection
    """
    from src.cli.doctor import run_doctor
    
    try:
        run_doctor(fix=fix, debug=ctx.obj.get("debug", False))
    except Exception as e:
        if ctx.obj.get("debug"):
            raise
        handle_error(e)
        sys.exit(1)


@cli.command()
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8765, help="Port to bind to")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the API server for Obsidian plugin.
    
    The server provides REST API endpoints for the ANA Obsidian plugin
    to process notes and handle the question-answer workflow.
    
    \b
    Examples:
      ana serve              # Start on localhost:8765
      ana serve -p 9000      # Use custom port
    """
    console.print(Panel.fit(
        "[bold blue]🌐 ANA API Server[/bold blue]\n"
        f"[dim]Starting server at http://{host}:{port}[/dim]",
        border_style="blue"
    ))
    console.print()
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    console.print()
    
    try:
        from src.api.server import run_server
        run_server(host=host, port=port)
    except Exception as e:
        if ctx.obj.get("debug"):
            raise
        handle_error(e)
        sys.exit(1)


# =============================================================================
# Prompts Commands
# =============================================================================

@cli.group()
def prompts() -> None:
    """Manage custom prompts.
    
    \\b
    Subcommands:
      show      Display current prompt configuration
      init      Create default prompt template files
      validate  Validate custom prompt files
    """
    pass


@prompts.command()
def show() -> None:
    """Display current prompt configuration.
    
    Shows which prompts are custom vs default.
    """
    from rich.table import Table
    
    from src.config import get_config
    from src.prompt_manager import PromptManager
    
    config = get_config()
    pm = PromptManager(config)
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]📝 Prompt Configuration[/bold blue]",
        border_style="blue"
    ))
    console.print()
    
    # Show prompt info
    info = pm.get_prompt_info()
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Prompt Type", style="white")
    table.add_column("Source", style="white")
    table.add_column("Path", style="dim")
    
    for prompt_type, data in info.items():
        source = data["source"]
        path = data["path"]
        
        if source == "default":
            source_display = "[green]default (built-in)[/green]"
        elif source == "custom_file":
            source_display = "[yellow]custom (file)[/yellow]"
        else:
            source_display = "[yellow]custom (directory)[/yellow]"
        
        table.add_row(prompt_type, source_display, path)
    
    console.print(table)
    console.print()
    
    # Show directory setting if configured
    if config.custom_prompts_dir:
        console.print(f"[dim]Custom prompts directory: {config.custom_prompts_dir}[/dim]")
    else:
        console.print("[dim]No custom prompts directory configured.[/dim]")
    
    console.print()


@prompts.command()
@click.option("--output-dir", "-o", default="prompts", help="Output directory for prompt files")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def init(output_dir: str, force: bool) -> None:
    """Create default prompt template files.
    
    Creates editable prompt files that you can customize.
    """
    from src.prompts import (
        ANALYSIS_PROMPT,
        INTERROGATION_PROMPT,
        SYNTHESIS_PROMPT,
        SYSTEM_PROMPT_TEMPLATE,
        TAG_SUGGESTION_PROMPT,
    )
    
    prompts_to_create = {
        "system.txt": SYSTEM_PROMPT_TEMPLATE,
        "analysis.txt": ANALYSIS_PROMPT,
        "interrogation.txt": INTERROGATION_PROMPT,
        "synthesis.txt": SYNTHESIS_PROMPT,
        "tag_suggestion.txt": TAG_SUGGESTION_PROMPT,
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]📝 Creating Prompt Templates[/bold blue]",
        border_style="blue"
    ))
    console.print()
    
    created = 0
    skipped = 0
    
    for filename, content in prompts_to_create.items():
        file_path = output_path / filename
        
        if file_path.exists() and not force:
            console.print(f"  [yellow]⏭️  Skipped[/yellow] {filename} (already exists)")
            skipped += 1
        else:
            file_path.write_text(content, encoding="utf-8")
            console.print(f"  [green]✅ Created[/green] {filename}")
            created += 1
    
    # Create README
    readme_path = output_path / "README.md"
    if not readme_path.exists() or force:
        readme_content = """# Custom Prompts

이 디렉토리에는 ANA의 커스텀 프롬프트 파일이 있습니다.

## 사용법

1. `.env` 파일에 프롬프트 디렉토리 설정:
   ```
   ANA_CUSTOM_PROMPTS_DIR=prompts
   ```

2. 또는 개별 프롬프트 파일 경로 설정:
   ```
   ANA_CUSTOM_ANALYSIS_PROMPT=prompts/analysis.txt
   ```

## 프롬프트 파일

| 파일 | 설명 |
|------|------|
| `system.txt` | 시스템 프롬프트 (AI 역할 정의) |
| `analysis.txt` | 노트 분석 프롬프트 |
| `interrogation.txt` | 질문 생성 프롬프트 |
| `synthesis.txt` | 최종 노트 합성 프롬프트 |
| `tag_suggestion.txt` | 태그 제안 프롬프트 |

## 주의사항

- 프롬프트 내 `{variable}` 형식의 플레이스홀더를 유지해야 합니다
- `ana prompts validate` 명령으로 유효성 검사 가능
"""
        readme_path.write_text(readme_content, encoding="utf-8")
        console.print(f"  [green]✅ Created[/green] README.md")
        created += 1
    
    console.print()
    console.print(f"[green]Created {created} files[/green]", end="")
    if skipped > 0:
        console.print(f", [yellow]skipped {skipped} files[/yellow]")
    else:
        console.print()
    
    console.print()
    console.print("[dim]To use custom prompts, add to your .env file:[/dim]")
    console.print(f"[cyan]ANA_CUSTOM_PROMPTS_DIR={output_dir}[/cyan]")
    console.print()


@prompts.command()
def validate() -> None:
    """Validate custom prompt files.
    
    Checks that all configured custom prompts are valid.
    """
    from src.config import get_config
    from src.prompt_manager import PromptManager
    
    config = get_config()
    pm = PromptManager(config)
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]🔍 Validating Prompts[/bold blue]",
        border_style="blue"
    ))
    console.print()
    
    results = pm.validate_all_prompts()
    
    all_valid = True
    for prompt_type, (is_valid, message) in results.items():
        if is_valid:
            console.print(f"  [green]✅[/green] {prompt_type}: {message}")
        else:
            console.print(f"  [red]❌[/red] {prompt_type}: {message}")
            all_valid = False
    
    console.print()
    if all_valid:
        console.print("[green]All prompts are valid![/green]")
    else:
        console.print("[red]Some prompts have issues. Please fix them.[/red]")
    console.print()


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
