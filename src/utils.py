"""Utility Functions for ANA.

Provides helper functions for file operations, note rendering, and frontmatter handling.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.schemas import DraftNote

console = Console()


# =============================================================================
# File Operations
# =============================================================================

def load_note_from_file(file_path: Path | str) -> tuple[str, dict[str, Any]]:
    """Load a note from a file, separating content and frontmatter.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Tuple of (content_without_frontmatter, frontmatter_dict)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file can't be read
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Note file not found: {file_path}")
    
    content = file_path.read_text(encoding="utf-8")
    
    # Parse frontmatter
    frontmatter = {}
    body = content
    
    if content.strip().startswith("---"):
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError:
                frontmatter = {}
            body = match.group(2)
    
    return body.strip(), frontmatter


def save_note_to_file(
    note: DraftNote,
    output_path: Path | str,
    overwrite: bool = False
) -> Path:
    """Save a draft note to a markdown file.
    
    Args:
        note: DraftNote to save
        output_path: Directory or file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        Path to saved file
        
    Raises:
        FileExistsError: If file exists and overwrite is False
    """
    output_path = Path(output_path)
    
    # If output_path is a directory, create filename from title
    if output_path.is_dir() or not output_path.suffix:
        output_path.mkdir(parents=True, exist_ok=True)
        filename = sanitize_filename(note.title) + ".md"
        output_path = output_path / filename
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")
    
    # Generate content
    content = render_note(note)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_path.write_text(content, encoding="utf-8")
    
    return output_path


def sanitize_filename(title: str) -> str:
    """Convert title to a safe filename.
    
    Args:
        title: Note title
        
    Returns:
        Sanitized filename (without extension)
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', "", title)
    filename = re.sub(r"\s+", " ", filename).strip()
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename or "untitled"


# =============================================================================
# Frontmatter Handling
# =============================================================================

def generate_frontmatter(
    title: str,
    tags: list[str],
    category: str,
    extra_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Generate complete frontmatter for a note.
    
    Args:
        title: Note title
        tags: List of tags
        category: Note category
        extra_metadata: Additional metadata to merge
        
    Returns:
        Complete frontmatter dictionary
    """
    frontmatter = {
        "title": title,
        "tags": tags,
        "type": category,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "source": "ANA-generated",
    }
    
    # Merge extra metadata
    if extra_metadata:
        for key, value in extra_metadata.items():
            if key not in frontmatter:
                frontmatter[key] = value
    
    return frontmatter


def merge_metadata(
    existing: dict[str, Any],
    new: dict[str, Any]
) -> dict[str, Any]:
    """Merge existing metadata with new metadata.
    
    New values override existing, except for special cases like tags.
    
    Args:
        existing: Existing metadata
        new: New metadata to merge
        
    Returns:
        Merged metadata dictionary
    """
    merged = existing.copy()
    
    for key, value in new.items():
        if key == "tags" and key in merged:
            # Merge tags without duplicates
            existing_tags = set(merged[key]) if isinstance(merged[key], list) else {merged[key]}
            new_tags = set(value) if isinstance(value, list) else {value}
            merged[key] = sorted(existing_tags | new_tags)
        else:
            merged[key] = value
    
    return merged


def frontmatter_to_yaml(frontmatter: dict[str, Any]) -> str:
    """Convert frontmatter dict to YAML string.
    
    Args:
        frontmatter: Frontmatter dictionary
        
    Returns:
        YAML-formatted string
    """
    return yaml.dump(
        frontmatter,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False
    ).strip()


# =============================================================================
# Note Rendering
# =============================================================================

def render_note(note: DraftNote, template: str | None = None) -> str:
    """Render a DraftNote to markdown string.
    
    Args:
        note: DraftNote to render
        template: Optional Jinja2 template string
        
    Returns:
        Complete markdown string with frontmatter
    """
    if template:
        try:
            jinja_template = Template(template)
            return jinja_template.render(
                title=note.title,
                tags=note.tags,
                category=note.category,
                content=note.content,
                created=datetime.now().strftime("%Y-%m-%d"),
                suggested_links=note.suggested_links,
                extra_metadata={
                    k: v for k, v in note.frontmatter.items()
                    if k not in ["title", "tags", "type", "created", "source"]
                },
                **note.frontmatter
            )
        except Exception:
            pass  # Fall back to default rendering
    
    # Default rendering
    frontmatter = note.frontmatter or generate_frontmatter(
        note.title, note.tags, note.category
    )
    
    yaml_str = frontmatter_to_yaml(frontmatter)
    
    # Build content
    content_parts = [
        "---",
        yaml_str,
        "---",
        "",
        f"# {note.title}",
        "",
        note.content,
    ]
    
    # Add related links if present
    if note.suggested_links:
        content_parts.extend([
            "",
            "---",
            "",
            "## Related Links",
        ])
        for link in note.suggested_links:
            content_parts.append(f"- [[{link}]]")
    
    return "\n".join(content_parts)


# =============================================================================
# Console Output
# =============================================================================

def render_note_preview(note: DraftNote):
    """Render a rich preview of the note in the console.
    
    Args:
        note: DraftNote to preview
    """
    content = render_note(note)
    
    console.print(Panel(
        Markdown(content),
        title=f"📝 {note.title}",
        subtitle=f"Category: {note.category}",
        border_style="green"
    ))


def print_questions(questions: list[str], categories: list[str] | None = None):
    """Print questions to the user in a formatted way.
    
    Args:
        questions: List of questions to display
        categories: Optional list of question categories
    """
    console.print("\n[bold yellow]📋 Please answer the following questions:[/bold yellow]\n")
    
    for i, question in enumerate(questions, 1):
        category = categories[i-1] if categories and i <= len(categories) else ""
        category_label = f"[dim]({category})[/dim] " if category else ""
        console.print(f"  {i}. {category_label}{question}")
    
    console.print()


def print_success(message: str):
    """Print a success message.
    
    Args:
        message: Message to display
    """
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str):
    """Print an error message.
    
    Args:
        message: Message to display
    """
    console.print(f"[bold red]✗[/bold red] {message}")


def print_info(message: str):
    """Print an info message.
    
    Args:
        message: Message to display
    """
    console.print(f"[bold blue]ℹ[/bold blue] {message}")
