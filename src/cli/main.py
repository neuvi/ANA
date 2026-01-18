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


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
