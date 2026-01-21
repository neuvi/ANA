/**
 * ANA Note - Markdown Parser Module
 * Handles markdown parsing and syntax highlighting
 */

class MarkdownParser {
    constructor() {
        this.initMarked();
    }

    /**
     * Initialize marked.js with custom configuration
     */
    initMarked() {
        // Configure marked options
        marked.setOptions({
            breaks: true,       // Convert line breaks to <br>
            gfm: true,          // GitHub Flavored Markdown
            headerIds: true,    // Add IDs to headers
            mangle: false,      // Don't escape autolinks
            smartypants: true,  // Smart punctuation
            highlight: (code, lang) => {
                // Use highlight.js for syntax highlighting
                if (lang && hljs.getLanguage(lang)) {
                    try {
                        return hljs.highlight(code, { language: lang }).value;
                    } catch (err) {
                        console.warn('Highlight error:', err);
                    }
                }
                // Auto-detect language if not specified
                try {
                    return hljs.highlightAuto(code).value;
                } catch (err) {
                    return code;
                }
            }
        });

        // Custom renderer for enhanced features
        const renderer = new marked.Renderer();

        // Enhanced link rendering (open external links in new tab)
        renderer.link = (href, title, text) => {
            const isExternal = href && (href.startsWith('http://') || href.startsWith('https://'));
            const target = isExternal ? ' target="_blank" rel="noopener noreferrer"' : '';
            const titleAttr = title ? ` title="${title}"` : '';
            return `<a href="${href}"${titleAttr}${target}>${text}</a>`;
        };

        // Enhanced image rendering with lazy loading
        renderer.image = (href, title, text) => {
            const titleAttr = title ? ` title="${title}"` : '';
            const alt = text ? ` alt="${text}"` : '';
            return `<img src="${href}"${alt}${titleAttr} loading="lazy">`;
        };

        // Task list support
        renderer.listitem = (text, task, checked) => {
            if (task) {
                const checkedAttr = checked ? ' checked disabled' : ' disabled';
                text = text.replace(/^\[[ xX]\]\s*/, '');
                return `<li class="task-list-item"><input type="checkbox"${checkedAttr}> ${text}</li>`;
            }
            return `<li>${text}</li>`;
        };

        // Enhanced code block with copy button placeholder
        renderer.code = (code, language, isEscaped) => {
            const lang = language || 'plaintext';
            const highlighted = this.highlightCode(code, lang);
            return `<pre><code class="hljs language-${lang}">${highlighted}</code></pre>`;
        };

        marked.use({ renderer });
    }

    /**
     * Highlight code using highlight.js
     */
    highlightCode(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return hljs.highlight(code, { language: lang }).value;
            } catch (err) {
                console.warn('Highlight error:', err);
            }
        }
        return hljs.highlightAuto(code).value;
    }

    /**
     * Parse markdown to HTML
     * @param {string} markdown - Markdown text to parse
     * @returns {string} HTML output
     */
    parse(markdown) {
        if (!markdown || !markdown.trim()) {
            return '';
        }

        try {
            // Pre-process for Obsidian-style features
            let processed = this.preProcess(markdown);

            // Parse with marked
            let html = marked.parse(processed);

            // Post-process for additional features
            html = this.postProcess(html);

            return html;
        } catch (error) {
            console.error('Markdown parsing error:', error);
            return `<p class="error">마크다운 파싱 오류: ${error.message}</p>`;
        }
    }

    /**
     * Pre-process markdown for custom features
     */
    preProcess(markdown) {
        // Convert Obsidian-style wiki links [[link]] to standard markdown
        markdown = markdown.replace(/\[\[([^\]|]+)\|?([^\]]*)\]\]/g, (match, link, display) => {
            const text = display || link;
            return `[${text}](${link})`;
        });

        // Convert Obsidian-style highlights ==text== to <mark>
        markdown = markdown.replace(/==([^=]+)==/g, '<mark>$1</mark>');

        // Handle callouts/admonitions
        markdown = markdown.replace(/^> \[!(\w+)\]([^\n]*)\n((?:>[^\n]*\n?)*)/gm, (match, type, title, content) => {
            const cleanContent = content.replace(/^> ?/gm, '');
            return `<div class="callout callout-${type.toLowerCase()}">
        <div class="callout-title">${type}${title}</div>
        <div class="callout-content">${cleanContent}</div>
      </div>\n`;
        });

        return markdown;
    }

    /**
     * Post-process HTML for additional features
     */
    postProcess(html) {
        // Add class to tables for styling
        html = html.replace(/<table>/g, '<table class="markdown-table">');

        return html;
    }

    /**
     * Count words in text
     */
    countWords(text) {
        if (!text || !text.trim()) return 0;

        // Remove markdown syntax for accurate count
        const cleaned = text
            .replace(/```[\s\S]*?```/g, '')  // Remove code blocks
            .replace(/`[^`]+`/g, '')          // Remove inline code
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')  // Replace links with text
            .replace(/[#*_~`]/g, '')          // Remove formatting chars
            .replace(/\s+/g, ' ')             // Normalize whitespace
            .trim();

        if (!cleaned) return 0;

        // Count Korean and English words differently
        const koreanWords = (cleaned.match(/[\uac00-\ud7a3]+/g) || []).length;
        const englishWords = cleaned
            .replace(/[\uac00-\ud7a3]+/g, ' ')
            .trim()
            .split(/\s+/)
            .filter(word => word.length > 0).length;

        return koreanWords + englishWords;
    }

    /**
     * Count characters in text
     */
    countChars(text) {
        if (!text) return 0;
        return text.length;
    }
}

// Export for use in app.js
window.MarkdownParser = MarkdownParser;
