"""ANA CLI Main Entry Point.

Provides interactive command-line interface for the Atomic Note Architect.
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
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


def process_single_note(agent: AtomicNoteArchitect, raw_note: str, frontmatter: dict | None, args, note_title: str = "Note"):
    """Process a single note through the pipeline.
    
    Args:
        agent: ANA agent instance
        raw_note: Raw note content
        frontmatter: Optional frontmatter dict
        args: CLI arguments
        note_title: Title for display purposes
    """
    print_info(f"Processing: {note_title}...")
    response = agent.process(raw_note, frontmatter)
    
    # Display analysis
    console.print(Panel(
        f"[bold]Detected Concepts:[/bold] {', '.join(response.analysis.detected_concepts) or 'None'}\n"
        f"[bold]Category:[/bold] {agent.get_category()}\n"
        f"[bold]Is Sufficient:[/bold] {'✅ Yes' if response.analysis.is_sufficient else '❌ No'}",
        title=f"📊 Analysis: {note_title}",
        border_style="cyan"
    ))
    console.print()
    
    # Interactive question loop
    while response.status == "needs_info" and not args.no_interactive:
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
        if args.no_interactive or Confirm.ask("Save this note?", default=True):
            try:
                saved_path = agent.save_note()
                print_success(f"Note saved to: {saved_path}")
            except FileExistsError:
                if Confirm.ask("File already exists. Overwrite?", default=False):
                    saved_path = agent.save_note(overwrite=True)
                    print_success(f"Note saved to: {saved_path}")
                else:
                    print_info("Note not saved")
            except Exception as e:
                print_error(f"Failed to save note: {e}")
    else:
        print_info("Note is incomplete. More information may be needed.")
    
    # Reset agent for next note
    agent.reset()


def main():
    """Main entry point for ANA CLI."""
    parser = argparse.ArgumentParser(
        description="ANA - Atomic Note Architect: Transform raw notes into atomic notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                     # Interactive mode
  python -m src.main --input note.txt    # Process a file
  python -m src.main --output ~/vault    # Specify output directory
"""
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input file path (raw note to process)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for saved notes"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to .env configuration file"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Non-interactive mode (don't ask questions)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        import os
        os.environ["ANA_ENV_FILE"] = args.config
    
    try:
        config = ANAConfig()
    except Exception as e:
        print_error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Override output directory if specified
    if args.output:
        config.vault_path = Path(args.output)
    
    # Print header
    console.print(Panel.fit(
        "[bold blue]🏛️ ANA - Atomic Note Architect[/bold blue]\n"
        "[dim]Transform raw notes into Zettelkasten-style atomic notes[/dim]",
        border_style="blue"
    ))
    console.print()
    
    # Initialize agent
    try:
        print_info("Initializing agent...")
        agent = AtomicNoteArchitect(config)
        print_success("Agent initialized")
    except Exception as e:
        print_error(f"Failed to initialize agent: {e}")
        sys.exit(1)
    
    # Get input
    if args.input:
        # Load from file
        try:
            raw_note, frontmatter = load_note_from_file(args.input)
            print_success(f"Loaded note from {args.input}")
        except FileNotFoundError:
            print_error(f"File not found: {args.input}")
            sys.exit(1)
        except Exception as e:
            print_error(f"Failed to load file: {e}")
            sys.exit(1)
    else:
        # Interactive input
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
        
        raw_note = "\n".join(lines).strip()
        frontmatter = None
        
        if not raw_note:
            print_error("No input provided")
            sys.exit(1)
    
    console.print()
    
    # Process note
    try:
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
            
            if not args.no_interactive:
                split_choice = Prompt.ask(
                    "[bold]Choose an option[/bold]",
                    choices=["split", "continue", "cancel"],
                    default="split"
                )
                
                if split_choice == "split":
                    # Process each split note with auto-extracted content
                    console.print()
                    console.print("[bold green]Extracting content for each split note...[/bold green]")
                    console.print()
                    
                    for i, suggested_title in enumerate(response.analysis.split_suggestions, 1):
                        console.print(f"\n{'='*60}")
                        console.print(f"[bold cyan]📝 Split Note {i}/{len(response.analysis.split_suggestions)}: {suggested_title}[/bold cyan]")
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
                            proceed = Confirm.ask("Process this split note?", default=True)
                            
                            if proceed:
                                process_single_note(agent, extracted_content, None, args, suggested_title)
                            else:
                                print_info(f"Skipped: {suggested_title}")
                        else:
                            print_info(f"Skipped: {suggested_title} (no content extracted)")
                    
                    console.print("\n[bold green]✓ All split notes processed![/bold green]")
                    return
                
                elif split_choice == "cancel":
                    console.print("[yellow]Cancelled[/yellow]")
                    return
                # else: continue with original note
        
        # Display analysis for non-split note
        console.print(Panel(
            f"[bold]Detected Concepts:[/bold] {', '.join(response.analysis.detected_concepts) or 'None'}\n"
            f"[bold]Category:[/bold] {agent.get_category()}\n"
            f"[bold]Is Sufficient:[/bold] {'✅ Yes' if response.analysis.is_sufficient else '❌ No'}",
            title="📊 Analysis Result",
            border_style="cyan"
        ))
        console.print()
        
        # Interactive question loop
        while response.status == "needs_info" and not args.no_interactive:
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
            if args.no_interactive or Confirm.ask("Save this note?", default=True):
                try:
                    saved_path = agent.save_note()
                    print_success(f"Note saved to: {saved_path}")
                except FileExistsError:
                    if Confirm.ask("File already exists. Overwrite?", default=False):
                        saved_path = agent.save_note(overwrite=True)
                        print_success(f"Note saved to: {saved_path}")
                    else:
                        print_info("Note not saved")
                except Exception as e:
                    print_error(f"Failed to save note: {e}")
        else:
            print_info("Note is incomplete. More information may be needed.")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        print_error(f"Error processing note: {e}")
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()

