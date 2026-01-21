/**
 * ANA Note - Main Application
 * Note-taking WebUI with real-time markdown preview and ANA AI integration
 */

class ANANote {
    constructor() {
        // DOM Elements
        this.editor = document.getElementById('editor');
        this.previewContent = document.getElementById('preview-content');
        this.noteTitle = document.getElementById('note-title');
        this.saveBtn = document.getElementById('btn-save');
        this.newBtn = document.getElementById('btn-new');
        this.themeToggle = document.getElementById('theme-toggle');
        this.saveIndicator = document.getElementById('save-indicator');
        this.saveStatus = document.getElementById('save-status');
        this.wordCount = document.getElementById('word-count');
        this.charCount = document.getElementById('char-count');
        this.lineInfo = document.getElementById('line-info');
        this.toastContainer = document.getElementById('toast-container');

        // ANA Panel Elements
        this.anaPanel = document.getElementById('ana-panel');
        this.analyzeBtn = document.getElementById('btn-analyze');
        this.toggleAnaBtn = document.getElementById('btn-toggle-ana');
        this.closeAnaBtn = document.getElementById('btn-close-ana');
        this.anaIndicator = document.getElementById('ana-indicator');
        this.anaStatusText = document.getElementById('ana-status-text');
        this.anaAnalysis = document.getElementById('ana-analysis');
        this.anaTags = document.getElementById('ana-tags');
        this.anaBacklinks = document.getElementById('ana-backlinks');
        this.anaQuestions = document.getElementById('ana-questions');
        this.questionsSection = document.getElementById('questions-section');
        this.refreshTagsBtn = document.getElementById('btn-refresh-tags');
        this.refreshBacklinksBtn = document.getElementById('btn-refresh-backlinks');
        this.connectionStatus = document.getElementById('ana-connection-status');

        // State
        this.currentNoteId = null;
        this.isDirty = false;
        this.saveTimeout = null;
        this.renderTimeout = null;
        this.anaSessionId = null;
        this.selectedTags = new Set();
        this.isAnalyzing = false;

        // Initialize
        this.parser = new MarkdownParser();
        this.api = new ANAApiClient();
        this.init();
    }

    /**
     * Initialize the application
     */
    async init() {
        this.loadTheme();
        this.loadFromLocalStorage();
        this.bindEvents();
        this.render();
        this.updateStats();
        await this.checkANAConnection();
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Editor input for live preview
        this.editor.addEventListener('input', () => this.handleEditorInput());

        // Track cursor position
        this.editor.addEventListener('click', () => this.updateLineInfo());
        this.editor.addEventListener('keyup', () => this.updateLineInfo());

        // Title change
        this.noteTitle.addEventListener('input', () => this.markDirty());

        // Save button
        this.saveBtn.addEventListener('click', () => this.save());

        // New note button
        this.newBtn.addEventListener('click', () => this.newNote());

        // Theme toggle
        this.themeToggle.addEventListener('click', () => this.toggleTheme());

        // ANA Panel controls
        if (this.analyzeBtn) {
            this.analyzeBtn.addEventListener('click', () => this.analyzeNote());
        }
        if (this.toggleAnaBtn) {
            this.toggleAnaBtn.addEventListener('click', () => this.toggleAnaPanel());
        }
        if (this.closeAnaBtn) {
            this.closeAnaBtn.addEventListener('click', () => this.toggleAnaPanel());
        }
        if (this.refreshTagsBtn) {
            this.refreshTagsBtn.addEventListener('click', () => this.refreshTagSuggestions());
        }
        if (this.refreshBacklinksBtn) {
            this.refreshBacklinksBtn.addEventListener('click', () => this.refreshBacklinkSuggestions());
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));

        // Auto-save on page unload
        window.addEventListener('beforeunload', (e) => {
            if (this.isDirty) {
                this.saveToLocalStorage();
                e.returnValue = '저장되지 않은 변경사항이 있습니다.';
            }
        });

        // Tab key handling for editor
        this.editor.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                e.preventDefault();
                this.insertAtCursor('  ');
            }
        });
    }

    /**
     * Check ANA server connection
     */
    async checkANAConnection() {
        this.setANAStatus('loading', '연결 확인 중...');

        try {
            const health = await this.api.checkHealth();
            if (health.isHealthy) {
                this.setANAStatus('connected', '연결됨');
                this.connectionStatus.textContent = 'ANA: 연결됨 ✓';
                this.connectionStatus.style.color = 'var(--accent-success)';
            } else {
                this.setANAStatus('disconnected', '연결 실패');
                this.connectionStatus.textContent = 'ANA: 연결 실패';
                this.connectionStatus.style.color = 'var(--accent-danger)';
            }
        } catch (error) {
            console.error('ANA connection check failed:', error);
            this.setANAStatus('disconnected', '연결 불가');
            this.connectionStatus.textContent = 'ANA: 오프라인';
            this.connectionStatus.style.color = 'var(--accent-danger)';
        }
    }

    /**
     * Set ANA status indicator
     */
    setANAStatus(status, text) {
        this.anaIndicator.className = `status-indicator ${status}`;
        this.anaStatusText.textContent = text;
    }

    /**
     * Toggle ANA panel visibility
     */
    toggleAnaPanel() {
        this.anaPanel.classList.toggle('hidden');
    }

    /**
     * Analyze note with ANA
     */
    async analyzeNote() {
        const content = this.editor.value.trim();
        if (!content) {
            this.showToast('분석할 내용을 입력해주세요.', 'warning');
            return;
        }

        if (this.isAnalyzing) {
            this.showToast('분석이 진행 중입니다.', 'warning');
            return;
        }

        this.isAnalyzing = true;
        this.analyzeBtn.disabled = true;
        this.analyzeBtn.innerHTML = '<span class="spinner"></span><span>분석 중...</span>';

        // Show ANA panel if hidden
        this.anaPanel.classList.remove('hidden');

        this.anaAnalysis.innerHTML = '<div class="loading-content"><div class="spinner"></div><span>AI 분석 중...</span></div>';

        try {
            const result = await this.api.processNote(content, {});
            this.anaSessionId = result.session_id;

            this.renderAnalysisResult(result);

            // Also refresh tag suggestions
            await this.refreshTagSuggestions();

            // And backlink suggestions
            await this.refreshBacklinkSuggestions();

            this.showToast('분석이 완료되었습니다.', 'success');
        } catch (error) {
            console.error('Analysis failed:', error);
            this.anaAnalysis.innerHTML = `<div class="empty-state-mini">분석 실패: ${error.message}</div>`;
            this.showToast('분석에 실패했습니다.', 'error');
        } finally {
            this.isAnalyzing = false;
            this.analyzeBtn.disabled = false;
            this.analyzeBtn.innerHTML = '<span>🤖</span><span>분석</span>';
        }
    }

    /**
     * Render analysis result in ANA panel
     */
    renderAnalysisResult(result) {
        let html = '';

        if (result.analysis) {
            const analysis = result.analysis;

            // Category badge
            if (analysis.category) {
                html += `<div class="ana-analysis-category">📁 ${analysis.category}</div>`;
            }

            // Detected concepts
            if (analysis.detected_concepts && analysis.detected_concepts.length > 0) {
                html += '<div class="ana-analysis-concepts">';
                html += '<div class="ana-analysis-concepts-title">감지된 개념:</div>';
                analysis.detected_concepts.forEach(concept => {
                    html += `<div class="ana-analysis-item">• ${concept}</div>`;
                });
                html += '</div>';
            }

            // Split suggestions
            if (analysis.should_split && analysis.split_suggestions) {
                html += '<div class="ana-analysis-concepts">';
                html += '<div class="ana-analysis-concepts-title">⚠️ 분할 제안:</div>';
                analysis.split_suggestions.forEach(suggestion => {
                    html += `<div class="ana-analysis-item">• ${suggestion}</div>`;
                });
                html += '</div>';
            }
        }

        // Show questions if any
        if (result.questions && result.questions.length > 0) {
            this.renderQuestions(result.questions);
        } else {
            this.questionsSection.classList.add('hidden');
        }

        if (!html) {
            html = '<div class="empty-state-mini">분석 결과가 없습니다</div>';
        }

        this.anaAnalysis.innerHTML = html;
    }

    /**
     * Render questions from ANA
     */
    renderQuestions(questions) {
        this.questionsSection.classList.remove('hidden');

        let html = '';
        questions.forEach((q, index) => {
            html += `
        <div class="question-item">
          <div class="question-text">${index + 1}. ${q.text}</div>
          <input type="text" class="question-input" data-index="${index}" placeholder="답변을 입력하세요...">
        </div>
      `;
        });

        html += `
      <div class="question-actions">
        <button class="btn btn-primary btn-sm" id="btn-submit-answers">답변 제출</button>
        <button class="btn btn-ghost btn-sm" id="btn-skip-questions">건너뛰기</button>
      </div>
    `;

        this.anaQuestions.innerHTML = html;

        // Bind submit button
        document.getElementById('btn-submit-answers')?.addEventListener('click', () => this.submitAnswers());
        document.getElementById('btn-skip-questions')?.addEventListener('click', () => {
            this.questionsSection.classList.add('hidden');
        });
    }

    /**
     * Submit answers to ANA questions
     */
    async submitAnswers() {
        const inputs = this.anaQuestions.querySelectorAll('.question-input');
        const answers = Array.from(inputs).map(input => input.value || '');

        if (!this.anaSessionId) {
            this.showToast('세션이 만료되었습니다. 다시 분석해주세요.', 'error');
            return;
        }

        try {
            const result = await this.api.answerQuestions(this.anaSessionId, answers);
            this.renderAnalysisResult(result);
            this.showToast('답변이 제출되었습니다.', 'success');
        } catch (error) {
            console.error('Failed to submit answers:', error);
            this.showToast('답변 제출에 실패했습니다.', 'error');
        }
    }

    /**
     * Refresh tag suggestions
     */
    async refreshTagSuggestions() {
        const content = this.editor.value.trim();
        if (!content) {
            this.anaTags.innerHTML = '<div class="empty-state-mini">태그 추천을 받으려면<br>노트를 작성하세요</div>';
            return;
        }

        this.anaTags.innerHTML = '<div class="loading-content"><div class="spinner"></div></div>';

        try {
            const result = await this.api.suggestTags(content, Array.from(this.selectedTags));
            this.renderTagSuggestions(result.suggestions || []);
        } catch (error) {
            console.error('Failed to get tag suggestions:', error);
            this.anaTags.innerHTML = '<div class="empty-state-mini">태그 추천 실패</div>';
        }
    }

    /**
     * Render tag suggestions
     */
    renderTagSuggestions(suggestions) {
        if (!suggestions || suggestions.length === 0) {
            this.anaTags.innerHTML = '<div class="empty-state-mini">추천할 태그가 없습니다</div>';
            return;
        }

        let html = '';
        suggestions.forEach(tag => {
            const isSelected = this.selectedTags.has(tag.tag);
            const confidence = Math.round(tag.confidence * 100);
            html += `
        <span class="tag-chip ${isSelected ? 'selected' : ''}" data-tag="${tag.tag}">
          #${tag.tag}
          <span class="tag-chip-confidence">${confidence}%</span>
        </span>
      `;
        });

        this.anaTags.innerHTML = html;

        // Bind click events
        this.anaTags.querySelectorAll('.tag-chip').forEach(chip => {
            chip.addEventListener('click', () => this.toggleTag(chip));
        });
    }

    /**
     * Toggle tag selection
     */
    toggleTag(chipElement) {
        const tag = chipElement.dataset.tag;

        if (this.selectedTags.has(tag)) {
            this.selectedTags.delete(tag);
            chipElement.classList.remove('selected');
        } else {
            this.selectedTags.add(tag);
            chipElement.classList.add('selected');

            // Insert tag into editor frontmatter or content
            this.insertTag(tag);
        }
    }

    /**
     * Insert tag into editor
     */
    insertTag(tag) {
        const currentContent = this.editor.value;
        const tagText = `#${tag}`;

        // Check if already has the tag
        if (currentContent.includes(tagText)) {
            this.showToast(`태그 "${tag}"가 이미 있습니다.`, 'warning');
            return;
        }

        // Insert at the end with a space
        this.editor.value = currentContent + (currentContent.endsWith('\n') ? '' : '\n') + tagText + ' ';
        this.handleEditorInput();
        this.showToast(`태그 "${tag}" 추가됨`, 'success');
    }

    /**
     * Refresh backlink suggestions
     */
    async refreshBacklinkSuggestions() {
        const content = this.editor.value.trim();
        const title = this.noteTitle.value.trim() || '제목 없음';

        if (!content) {
            this.anaBacklinks.innerHTML = '<div class="empty-state-mini">연결할 수 있는<br>관련 노트가 없습니다</div>';
            return;
        }

        this.anaBacklinks.innerHTML = '<div class="loading-content"><div class="spinner"></div></div>';

        try {
            const result = await this.api.suggestBacklinks(title, content, Array.from(this.selectedTags));
            this.renderBacklinkSuggestions(result.suggestions || []);
        } catch (error) {
            console.error('Failed to get backlink suggestions:', error);
            this.anaBacklinks.innerHTML = '<div class="empty-state-mini">백링크 조회 실패</div>';
        }
    }

    /**
     * Render backlink suggestions
     */
    renderBacklinkSuggestions(suggestions) {
        if (!suggestions || suggestions.length === 0) {
            this.anaBacklinks.innerHTML = '<div class="empty-state-mini">연결 가능한<br>관련 노트가 없습니다</div>';
            return;
        }

        let html = '';
        suggestions.slice(0, 5).forEach(link => {
            const confidence = Math.round(link.confidence * 100);
            html += `
        <div class="backlink-card" data-path="${link.source_path}">
          <div class="backlink-title">
            <span class="backlink-title-icon">📄</span>
            ${link.source_title || 'Untitled'}
          </div>
          <div class="backlink-match">"${link.matched_text}"</div>
          <div class="backlink-confidence">
            <span>✓</span>
            ${confidence}% 일치
          </div>
        </div>
      `;
        });

        this.anaBacklinks.innerHTML = html;

        // Bind click events
        this.anaBacklinks.querySelectorAll('.backlink-card').forEach(card => {
            card.addEventListener('click', () => {
                const path = card.dataset.path;
                this.showToast(`노트 링크: ${path}`, 'success');
                // In a real implementation, this would insert a wiki link
                this.insertWikiLink(path);
            });
        });
    }

    /**
     * Insert wiki link into editor
     */
    insertWikiLink(path) {
        // Extract filename without extension
        const filename = path.split('/').pop().replace('.md', '');
        const wikiLink = `[[${filename}]]`;

        // Insert at cursor position
        this.insertAtCursor(wikiLink + ' ');
        this.showToast(`링크 "${filename}" 삽입됨`, 'success');
    }

    /**
     * Handle editor input
     */
    handleEditorInput() {
        this.markDirty();

        // Debounced rendering
        clearTimeout(this.renderTimeout);
        this.renderTimeout = setTimeout(() => {
            this.render();
            this.updateStats();
        }, 100);

        // Auto-save debounce
        clearTimeout(this.saveTimeout);
        this.saveTimeout = setTimeout(() => {
            this.saveToLocalStorage();
        }, 2000);
    }

    /**
     * Render markdown to preview
     */
    render() {
        const markdown = this.editor.value;
        const html = this.parser.parse(markdown);

        if (html) {
            this.previewContent.innerHTML = html;
        } else {
            this.previewContent.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">📄</div>
          <div class="empty-state-title">미리보기</div>
          <div class="empty-state-description">
            왼쪽 에디터에 마크다운을 입력하면 여기에 실시간으로 렌더링됩니다.
          </div>
        </div>
      `;
        }
    }

    /**
     * Update word and character counts
     */
    updateStats() {
        const text = this.editor.value;
        const words = this.parser.countWords(text);
        const chars = this.parser.countChars(text);

        this.wordCount.textContent = `${words.toLocaleString()} 단어`;
        this.charCount.textContent = `${chars.toLocaleString()} 글자`;
    }

    /**
     * Update line and column info
     */
    updateLineInfo() {
        const text = this.editor.value;
        const cursorPos = this.editor.selectionStart;

        const textBeforeCursor = text.substring(0, cursorPos);
        const lines = textBeforeCursor.split('\n');
        const lineNum = lines.length;
        const colNum = lines[lines.length - 1].length + 1;

        this.lineInfo.textContent = `Line ${lineNum}, Col ${colNum}`;
    }

    /**
     * Mark document as dirty (unsaved changes)
     */
    markDirty() {
        if (!this.isDirty) {
            this.isDirty = true;
            this.saveIndicator.classList.add('saving');
            this.saveStatus.textContent = '수정됨';
        }
    }

    /**
     * Mark document as clean (saved)
     */
    markClean() {
        this.isDirty = false;
        this.saveIndicator.classList.remove('saving');
        this.saveStatus.textContent = '저장됨';
    }

    /**
     * Save note to localStorage
     */
    saveToLocalStorage() {
        const noteData = {
            id: this.currentNoteId || this.generateId(),
            title: this.noteTitle.value || '제목 없는 노트',
            content: this.editor.value,
            tags: Array.from(this.selectedTags),
            updatedAt: new Date().toISOString()
        };

        this.currentNoteId = noteData.id;
        localStorage.setItem('ana-note-current', JSON.stringify(noteData));
        this.markClean();
    }

    /**
     * Load note from localStorage
     */
    loadFromLocalStorage() {
        try {
            const saved = localStorage.getItem('ana-note-current');
            if (saved) {
                const noteData = JSON.parse(saved);
                this.currentNoteId = noteData.id;
                this.noteTitle.value = noteData.title || '';
                this.editor.value = noteData.content || '';
                if (noteData.tags) {
                    this.selectedTags = new Set(noteData.tags);
                }
            }
        } catch (error) {
            console.error('Failed to load from localStorage:', error);
        }
    }

    /**
     * Save note (manual save)
     */
    async save() {
        this.saveIndicator.classList.add('saving');
        this.saveStatus.textContent = '저장 중...';

        try {
            this.saveToLocalStorage();
            this.showToast('노트가 저장되었습니다.', 'success');
        } catch (error) {
            console.error('Save failed:', error);
            this.saveIndicator.classList.add('error');
            this.saveStatus.textContent = '저장 실패';
            this.showToast('저장에 실패했습니다.', 'error');
        }
    }

    /**
     * Create new note
     */
    newNote() {
        if (this.isDirty) {
            if (!confirm('저장되지 않은 변경사항이 있습니다. 계속하시겠습니까?')) {
                return;
            }
        }

        this.currentNoteId = null;
        this.noteTitle.value = '';
        this.editor.value = '';
        this.selectedTags.clear();
        this.anaSessionId = null;
        this.render();
        this.updateStats();
        this.markClean();
        this.editor.focus();

        // Reset ANA panel
        this.anaAnalysis.innerHTML = '<div class="empty-state-mini">"분석" 버튼을 클릭하여<br>AI 분석을 시작하세요</div>';
        this.anaTags.innerHTML = '<div class="empty-state-mini">태그 추천을 받으려면<br>노트를 작성하세요</div>';
        this.anaBacklinks.innerHTML = '<div class="empty-state-mini">연결할 수 있는<br>관련 노트가 없습니다</div>';
        this.questionsSection.classList.add('hidden');

        this.showToast('새 노트가 생성되었습니다.', 'success');
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyboard(e) {
        // Ctrl/Cmd + S: Save
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            this.save();
        }

        // Ctrl/Cmd + N: New note
        if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
            e.preventDefault();
            this.newNote();
        }

        // Ctrl/Cmd + B: Bold
        if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
            e.preventDefault();
            this.wrapSelection('**', '**');
        }

        // Ctrl/Cmd + I: Italic
        if ((e.ctrlKey || e.metaKey) && e.key === 'i') {
            e.preventDefault();
            this.wrapSelection('*', '*');
        }

        // Ctrl/Cmd + K: Link
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            this.wrapSelection('[', '](url)');
        }

        // Ctrl/Cmd + `: Code
        if ((e.ctrlKey || e.metaKey) && e.key === '`') {
            e.preventDefault();
            this.wrapSelection('`', '`');
        }

        // Ctrl/Cmd + Shift + A: Analyze
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'A') {
            e.preventDefault();
            this.analyzeNote();
        }
    }

    /**
     * Insert text at cursor position
     */
    insertAtCursor(text) {
        const start = this.editor.selectionStart;
        const end = this.editor.selectionEnd;
        const before = this.editor.value.substring(0, start);
        const after = this.editor.value.substring(end);

        this.editor.value = before + text + after;
        this.editor.selectionStart = this.editor.selectionEnd = start + text.length;
        this.editor.focus();
        this.handleEditorInput();
    }

    /**
     * Wrap selected text with prefix and suffix
     */
    wrapSelection(prefix, suffix) {
        const start = this.editor.selectionStart;
        const end = this.editor.selectionEnd;
        const selected = this.editor.value.substring(start, end);
        const before = this.editor.value.substring(0, start);
        const after = this.editor.value.substring(end);

        if (selected) {
            this.editor.value = before + prefix + selected + suffix + after;
            this.editor.selectionStart = start + prefix.length;
            this.editor.selectionEnd = end + prefix.length;
        } else {
            this.editor.value = before + prefix + suffix + after;
            this.editor.selectionStart = this.editor.selectionEnd = start + prefix.length;
        }

        this.editor.focus();
        this.handleEditorInput();
    }

    /**
     * Toggle theme
     */
    toggleTheme() {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('ana-note-theme', newTheme);

        // Update theme toggle icon
        const slider = this.themeToggle.querySelector('.theme-toggle-slider');
        slider.textContent = newTheme === 'dark' ? '🌙' : '☀️';

        // Update highlight.js theme
        this.updateHighlightTheme(newTheme);
    }

    /**
     * Update highlight.js theme
     */
    updateHighlightTheme(theme) {
        const lightLink = document.getElementById('hljs-light');
        if (lightLink) {
            lightLink.media = theme === 'light' ? 'all' : '(prefers-color-scheme: light)';
        }
    }

    /**
     * Load theme from localStorage
     */
    loadTheme() {
        const savedTheme = localStorage.getItem('ana-note-theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);

        const slider = this.themeToggle.querySelector('.theme-toggle-slider');
        slider.textContent = savedTheme === 'dark' ? '🌙' : '☀️';

        this.updateHighlightTheme(savedTheme);
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        const icons = {
            success: '✅',
            warning: '⚠️',
            error: '❌'
        };

        toast.innerHTML = `
      <span class="toast-icon">${icons[type] || '💬'}</span>
      <span class="toast-message">${message}</span>
    `;

        this.toastContainer.appendChild(toast);

        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.animation = 'slide-out 0.3s ease forwards';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    /**
     * Generate unique ID
     */
    generateId() {
        return `note-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.anaNote = new ANANote();
});
