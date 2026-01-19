"""Template Manager Module.

Manages note templates with priority: File (A) -> DB (B) -> AI Generation (C).
Always saves to DB (B) regardless of source.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.logging_config import get_logger

logger = get_logger("template")

if TYPE_CHECKING:
    from src.config import ANAConfig
    from src.vault_scanner import VaultScanner


TEMPLATE_GENERATION_PROMPT = """You are a template design expert for Obsidian notes.
Create a Markdown template for the category: "{category}"

{samples_section}

Template Requirements:
1. Include YAML frontmatter with these fields:
   - title: "{{{{ title }}}}"
   - tags: {{{{ tags }}}}
   - type: {category}
   - created: {{{{ created }}}}
   - source: "ANA-generated"
2. Use Jinja2 template syntax: {{{{ variable }}}}
3. Design section structure appropriate for this category
4. Include a "## Related Links" section at the end with:
   {{% for link in suggested_links %}}
   - [[{{{{ link }}}}]]
   {{% endfor %}}
5. Keep the template clean and well-organized

Return ONLY the Markdown template, no explanations."""


class TemplateManager:
    """Manages note templates for different categories.
    
    Template Resolution Priority:
    A. File-based (templates/{category}_template.md)
    B. DB-based (data/templates.json)
    C. AI-generated dynamically
    
    Important: Always saves to DB (B) regardless of which method was used.
    """
    
    def __init__(
        self,
        config: "ANAConfig",
        llm: BaseChatModel,
        vault_scanner: "VaultScanner | None" = None
    ):
        """Initialize template manager.
        
        Args:
            config: ANA configuration
            llm: Language model for template generation
            vault_scanner: Optional vault scanner for sample notes
        """
        self.templates_dir = config.get_templates_dir()
        self.db_path = config.get_template_db_path()
        self.llm = llm
        self.vault_scanner = vault_scanner
        self._db: dict[str, str] = self._load_db()
    
    def get_template(
        self,
        category: str,
        sample_notes: list[Path] | None = None
    ) -> tuple[str, str]:
        """Get template for a category.
        
        Tries in order: File (A) -> DB (B) -> AI (C)
        Only ONE method is used, but B always saves.
        
        Args:
            category: Category name
            sample_notes: Optional list of sample note paths for AI generation
            
        Returns:
            Tuple of (template_content, source) where source is 'file', 'db', or 'ai'
        """
        template = None
        source = None
        
        # A: Try file-based template first
        file_template = self._get_file_template(category)
        if file_template:
            template = file_template
            source = "file"
        
        # B: Try DB-based template (only if A not found)
        if template is None:
            db_template = self._get_db_template(category)
            if db_template:
                template = db_template
                source = "db"
        
        # C: Generate with AI (only if A and B not found)
        if template is None:
            template = self._generate_template_with_ai(category, sample_notes)
            source = "ai"
        
        # Always save to DB (B) regardless of source
        self._save_to_db(category, template)
        
        return template, source
    
    def _get_file_template(self, category: str) -> str | None:
        """A: Get file-based template.
        
        Args:
            category: Category name
            
        Returns:
            Template content or None
        """
        # Try exact match first
        template_file = self.templates_dir / f"{category}_template.md"
        if template_file.exists():
            try:
                return template_file.read_text(encoding="utf-8")
            except IOError as e:
                logger.warning(f"Failed to read template file {template_file}: {e}")
        
        # Try without underscores/hyphens normalization
        normalized = category.replace("-", "_")
        template_file = self.templates_dir / f"{normalized}_template.md"
        if template_file.exists():
            try:
                return template_file.read_text(encoding="utf-8")
            except IOError as e:
                logger.warning(f"Failed to read template file {template_file}: {e}")
        
        # Try default template
        default_file = self.templates_dir / "default_template.md"
        if default_file.exists():
            try:
                return default_file.read_text(encoding="utf-8")
            except IOError as e:
                logger.warning(f"Failed to read default template: {e}")
        
        return None
    
    def _get_db_template(self, category: str) -> str | None:
        """B: Get DB-based template.
        
        Args:
            category: Category name
            
        Returns:
            Template content or None
        """
        return self._db.get(category)
    
    def _generate_template_with_ai(
        self,
        category: str,
        sample_notes: list[Path] | None = None
    ) -> str:
        """C: Generate template using AI.
        
        Args:
            category: Category name
            sample_notes: Optional sample notes for context
            
        Returns:
            Generated template content
        """
        samples_section = ""
        
        if sample_notes and self.vault_scanner:
            samples = []
            for note_path in sample_notes[:3]:  # Max 3 samples
                content = self.vault_scanner.get_note_content(note_path)
                if content:
                    # Truncate long samples
                    if len(content) > 1000:
                        content = content[:1000] + "\n... (truncated)"
                    samples.append(f"--- Sample Note ---\n{content}")
            
            if samples:
                samples_section = "Reference these existing notes for style:\n\n" + "\n\n".join(samples)
        
        if not samples_section:
            samples_section = "No sample notes available. Create a general template for this category."
        
        prompt = TEMPLATE_GENERATION_PROMPT.format(
            category=category,
            samples_section=samples_section
        )
        
        messages = [
            SystemMessage(content="You are an expert template designer for knowledge management."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        template = response.content.strip()
        
        # Clean up markdown code blocks if present
        if template.startswith("```markdown"):
            template = template[11:]
        elif template.startswith("```md"):
            template = template[5:]
        elif template.startswith("```"):
            template = template[3:]
        
        if template.endswith("```"):
            template = template[:-3]
        
        return template.strip()
    
    def _save_to_db(self, category: str, template: str):
        """Save template to DB (always executed).
        
        Args:
            category: Category name
            template: Template content
        """
        self._db[category] = template
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self._db, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.warning(f"Could not save template DB to {self.db_path}: {e}")
    
    def _load_db(self) -> dict[str, str]:
        """Load template DB from file.
        
        Returns:
            Dictionary of category -> template
        """
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load template DB from {self.db_path}: {e}")
        return {}
    
    def list_available_templates(self) -> dict[str, str]:
        """List all available templates with their sources.
        
        Returns:
            Dictionary of category -> source (file/db)
        """
        available = {}
        
        # Check file-based templates
        if self.templates_dir.exists():
            for template_file in self.templates_dir.glob("*_template.md"):
                category = template_file.stem.replace("_template", "")
                available[category] = "file"
        
        # Add DB-based templates (don't override file-based)
        for category in self._db:
            if category not in available:
                available[category] = "db"
        
        return available
    
    def save_template_to_file(self, category: str, template: str) -> Path:
        """Save template as a file (for user customization).
        
        Args:
            category: Category name
            template: Template content
            
        Returns:
            Path to saved file
        """
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        template_file = self.templates_dir / f"{category}_template.md"
        template_file.write_text(template, encoding="utf-8")
        
        return template_file
    
    def get_default_template(self) -> str:
        """Get a basic default template.
        
        Returns:
            Default template content
        """
        return '''---
title: "{{ title }}"
tags: {{ tags }}
type: {{ category }}
created: {{ created }}
source: "ANA-generated"
{% for key, value in extra_metadata.items() %}
{{ key }}: {{ value }}
{% endfor %}
---

# {{ title }}

{{ content }}

---

## Related Links
{% for link in suggested_links %}
- [[{{ link }}]]
{% endfor %}
'''
    
    def propose_new_template(self, category: str) -> str:
        """Generate a proposal message for a new category template.
        
        Args:
            category: Category name
            
        Returns:
            Formatted proposal message
        """
        template, source = self.get_template(category)
        
        return f"""
## New Category Template Proposal: '{category}'

The following template was generated for this category:

```markdown
{template}
```

Template source: {source}

Do you want to:
- (y) Use this template
- (e) Edit the template
- (s) Save as file for future customization
- (c) Cancel and use default template
"""
