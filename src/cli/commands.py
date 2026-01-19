"""ANA CLI Commands.

Implementation of main CLI commands: new, process, sync.
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.agent import AtomicNoteArchitect
from src.config import ANAConfig
from src.utils import (
    load_note_from_file,
    print_error,
    print_info,
    print_questions,
    print_success,
    render_note_preview,
)

console = Console()


def run_new_command(
    input_file: str | None = None,
    output_dir: str | None = None,
    no_interactive: bool = False,
    debug: bool = False
) -> None:
    """Run the new/process command."""
    
    # Load configuration
    config = ANAConfig()
    
    if output_dir:
        config.vault_path = Path(output_dir)
    
    # Print header
    console.print(Panel.fit(
        "[bold blue]🏛️ ANA - Atomic Note Architect[/bold blue]\n"
        "[dim]Transform raw notes into Zettelkasten-style atomic notes[/dim]",
        border_style="blue"
    ))
    console.print()
    
    # Initialize agent
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Initializing agent...", total=None)
        agent = AtomicNoteArchitect(config)
    
    print_success("Agent initialized")
    
    # Get input
    if input_file:
        raw_note, frontmatter = load_note_from_file(input_file)
        print_success(f"Loaded note from {input_file}")
    else:
        raw_note, frontmatter = _get_interactive_input()
        if not raw_note:
            print_error("No input provided")
            sys.exit(1)
    
    console.print()
    
    # Process note
    _process_note(agent, raw_note, frontmatter, no_interactive)


def _get_interactive_input() -> tuple[str, dict | None]:
    """Get note input interactively."""
    console.print("[bold]Enter your raw note:[/bold]")
    console.print("[dim](Press Enter twice to finish, or Ctrl+D)[/dim]")
    console.print()
    
    lines = []
    empty_count = 0
    
    try:
        while True:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 2:
                    break
                lines.append(line)
            else:
                empty_count = 0
                lines.append(line)
    except EOFError:
        pass
    
    return "\n".join(lines).strip(), None


def _process_note(
    agent: AtomicNoteArchitect,
    raw_note: str,
    frontmatter: dict | None,
    no_interactive: bool
) -> None:
    """Process a note through the pipeline."""
    
    print_info("Analyzing note...")
    response = agent.process(raw_note, frontmatter)
    
    # Check if note should be split
    if response.analysis.should_split and response.analysis.split_suggestions:
        console.print(Panel(
            "[bold yellow]⚠️ This note contains multiple concepts![/bold yellow]\n\n"
            f"Detected concepts: {', '.join(response.analysis.detected_concepts)}",
            title="🔀 Split Suggestion",
            border_style="yellow"
        ))
        
        # Show split suggestions table
        table = Table(title="Suggested Split Notes", show_header=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Suggested Title", style="white")
        
        for i, title in enumerate(response.analysis.split_suggestions, 1):
            table.add_row(str(i), title)
        
        console.print(table)
        console.print()
        
        if not no_interactive:
            split_choice = Prompt.ask(
                "[bold]Choose an option[/bold]",
                choices=["split", "continue", "cancel"],
                default="split"
            )
            
            if split_choice == "split":
                _process_split_notes(agent, raw_note, response.analysis.split_suggestions, no_interactive)
                return
            elif split_choice == "cancel":
                console.print("[yellow]Cancelled[/yellow]")
                return
    
    # Display analysis
    console.print(Panel(
        f"[bold]Detected Concepts:[/bold] {', '.join(response.analysis.detected_concepts) or 'None'}\n"
        f"[bold]Category:[/bold] {agent.get_category()}\n"
        f"[bold]Is Sufficient:[/bold] {'✅ Yes' if response.analysis.is_sufficient else '❌ No'}",
        title="📊 Analysis Result",
        border_style="cyan"
    ))
    console.print()
    
    # Interactive question loop
    while response.status == "needs_info" and not no_interactive:
        if response.interaction and response.interaction.questions_to_user:
            print_questions(
                response.interaction.questions_to_user,
                response.interaction.question_categories
            )
            
            # Get answers
            answers = []
            for i, question in enumerate(response.interaction.questions_to_user, 1):
                answer = Prompt.ask(f"  [bold cyan]A{i}[/bold cyan]")
                answers.append(answer)
            
            console.print()
            print_info("Processing answers...")
            response = agent.answer_questions(answers)
        else:
            break
    
    # Show final note preview
    console.print()
    render_note_preview(response.draft_note)
    console.print()
    
    # Ask to save
    if response.status == "completed":
        if no_interactive or Confirm.ask("Save this note?", default=True):
            try:
                saved_path, backlink_suggestions, modified_files = agent.save_note_with_backlinks()
                print_success(f"Note saved to: {saved_path}")
                
                # Show backlink results
                if modified_files:
                    console.print()
                    console.print(Panel(
                        f"[bold green]🔗 Backlinks Added[/bold green]\n\n"
                        f"Found {len(backlink_suggestions)} potential backlinks\n"
                        f"Updated {len(modified_files)} existing notes:",
                        border_style="green"
                    ))
                    for f in modified_files[:5]:
                        console.print(f"  • {f.name}")
                    if len(modified_files) > 5:
                        console.print(f"  ... and {len(modified_files) - 5} more")
                elif backlink_suggestions:
                    console.print(f"[dim]Found {len(backlink_suggestions)} potential backlinks (below confidence threshold)[/dim]")
                    
            except FileExistsError:
                if Confirm.ask("File already exists. Overwrite?", default=False):
                    saved_path, _, modified_files = agent.save_note_with_backlinks(overwrite=True)
                    print_success(f"Note saved to: {saved_path}")
                    if modified_files:
                        console.print(f"[green]Updated {len(modified_files)} existing notes with backlinks[/green]")
                else:
                    print_info("Note not saved")
    else:
        print_info("Note is incomplete. More information may be needed.")


def _process_split_notes(
    agent: AtomicNoteArchitect,
    raw_note: str,
    split_suggestions: list[str],
    no_interactive: bool
) -> None:
    """Process split notes."""
    console.print()
    console.print("[bold green]Extracting content for each split note...[/bold green]")
    console.print()
    
    for i, suggested_title in enumerate(split_suggestions, 1):
        console.print(f"\n{'='*60}")
        console.print(f"[bold cyan]📝 Split Note {i}/{len(split_suggestions)}: {suggested_title}[/bold cyan]")
        console.print(f"{'='*60}\n")
        
        # Auto-extract content from original note
        print_info(f"Extracting content for: {suggested_title}")
        extracted_content, key_points = agent.extract_for_split(raw_note, suggested_title)
        
        if extracted_content:
            # Show extracted content preview
            console.print(Panel(
                extracted_content[:500] + ("..." if len(extracted_content) > 500 else ""),
                title="📋 Extracted Content Preview",
                border_style="dim"
            ))
            
            if key_points:
                console.print(f"[dim]Key points: {', '.join(key_points[:3])}[/dim]")
            console.print()
            
            # Ask to proceed or modify
            proceed = no_interactive or Confirm.ask("Process this split note?", default=True)
            
            if proceed:
                _process_single_split_note(agent, extracted_content, suggested_title, no_interactive)
            else:
                print_info(f"Skipped: {suggested_title}")
        else:
            print_info(f"Skipped: {suggested_title} (no content extracted)")
    
    console.print("\n[bold green]✓ All split notes processed![/bold green]")


def _process_single_split_note(
    agent: AtomicNoteArchitect,
    content: str,
    title: str,
    no_interactive: bool
) -> None:
    """Process a single split note."""
    print_info(f"Processing: {title}...")
    response = agent.process(content, None)
    
    # Display analysis
    console.print(Panel(
        f"[bold]Detected Concepts:[/bold] {', '.join(response.analysis.detected_concepts) or 'None'}\n"
        f"[bold]Category:[/bold] {agent.get_category()}\n"
        f"[bold]Is Sufficient:[/bold] {'✅ Yes' if response.analysis.is_sufficient else '❌ No'}",
        title=f"📊 Analysis: {title}",
        border_style="cyan"
    ))
    console.print()
    
    # Interactive question loop
    while response.status == "needs_info" and not no_interactive:
        if response.interaction and response.interaction.questions_to_user:
            print_questions(
                response.interaction.questions_to_user,
                response.interaction.question_categories
            )
            
            answers = []
            for i, question in enumerate(response.interaction.questions_to_user, 1):
                answer = Prompt.ask(f"  [bold cyan]A{i}[/bold cyan]")
                answers.append(answer)
            
            console.print()
            print_info("Processing answers...")
            response = agent.answer_questions(answers)
        else:
            break
    
    # Show final note preview
    console.print()
    render_note_preview(response.draft_note)
    console.print()
    
    # Ask to save
    if response.status == "completed":
        if no_interactive or Confirm.ask("Save this note?", default=True):
            try:
                saved_path, _, modified_files = agent.save_note_with_backlinks()
                print_success(f"Note saved to: {saved_path}")
                if modified_files:
                    console.print(f"[green]Updated {len(modified_files)} notes with backlinks[/green]")
            except FileExistsError:
                if Confirm.ask("File already exists. Overwrite?", default=False):
                    saved_path, _, _ = agent.save_note_with_backlinks(overwrite=True)
                    print_success(f"Note saved to: {saved_path}")
                else:
                    print_info("Note not saved")
    
    # Reset agent for next note
    agent.reset()


def run_sync_command(force: bool = False, debug: bool = False, use_async: bool = True) -> None:
    """Run the sync command.
    
    Args:
        force: Force re-sync all embeddings
        debug: Enable debug output
        use_async: Use async parallel processing (default: True)
    """
    import asyncio
    
    console.print(Panel.fit(
        "[bold blue]🔄 ANA - Embedding Sync[/bold blue]\n"
        "[dim]Synchronize embeddings for vault notes[/dim]",
        border_style="blue"
    ))
    console.print()
    
    config = ANAConfig()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Initializing agent...", total=None)
        agent = AtomicNoteArchitect(config)
    
    if use_async:
        print_info("Scanning vault and updating embeddings (async mode)...")
    else:
        print_info("Scanning vault and updating embeddings...")
    
    def progress_callback(current: int, total: int, file_name: str) -> None:
        console.print(f"  [{current}/{total}] {file_name}", highlight=False)
    
    try:
        if use_async:
            # Use async version for parallel processing
            stats = asyncio.run(
                agent.sync_embeddings_async(progress_callback=progress_callback)
            )
        else:
            # Fallback to synchronous version
            stats = agent.sync_embeddings(progress_callback=progress_callback)
        
        console.print()
        console.print(Panel(
            f"[bold green]✅ Sync Complete[/bold green]\n\n"
            f"Updated: {stats.get('updated', 0)}\n"
            f"Cached: {stats.get('cached', 0)}\n"
            f"Failed: {stats.get('failed', 0)}",
            border_style="green"
        ))
    except Exception as e:
        print_error(f"Sync failed: {e}")
        if debug:
            raise
