/**
 * ANA Sidebar View
 * 
 * Terminal-like sidebar panel for ANA processing workflow.
 */

import { ItemView, WorkspaceLeaf, MarkdownRenderer, Component } from 'obsidian';
import type ANAPlugin from './main';
import type { ProcessResponse, Question, DraftNote } from './api';

export const VIEW_TYPE_ANA = 'ana-sidebar-view';

interface LogEntry {
    type: 'info' | 'success' | 'error' | 'question' | 'answer' | 'preview';
    content: string;
    timestamp: Date;
}

export class ANASidebarView extends ItemView {
    plugin: ANAPlugin;
    private logContainer: HTMLElement;
    private inputContainer: HTMLElement;
    private currentQuestions: Question[] = [];
    private currentAnswers: string[] = [];
    private currentQuestionIndex: number = 0;
    private onAnswersComplete: ((answers: string[]) => void) | null = null;

    constructor(leaf: WorkspaceLeaf, plugin: ANAPlugin) {
        super(leaf);
        this.plugin = plugin;
    }

    getViewType(): string {
        return VIEW_TYPE_ANA;
    }

    getDisplayText(): string {
        return 'ANA - Atomic Note Architect';
    }

    getIcon(): string {
        return 'brain';
    }

    async onOpen(): Promise<void> {
        const container = this.containerEl.children[1];
        container.empty();
        container.addClass('ana-sidebar');

        // Header
        const header = container.createEl('div', { cls: 'ana-sidebar-header' });
        header.createEl('h4', { text: '🏛️ ANA' });

        const headerButtons = header.createEl('div', { cls: 'ana-header-buttons' });

        const processBtn = headerButtons.createEl('button', {
            text: '▶ Process',
            cls: 'ana-btn ana-btn-primary ana-btn-sm'
        });
        processBtn.addEventListener('click', () => this.plugin.processCurrentNote());

        const clearBtn = headerButtons.createEl('button', {
            text: 'Clear',
            cls: 'ana-btn ana-btn-sm'
        });
        clearBtn.addEventListener('click', () => this.clear());

        // Log container (terminal-like output)
        this.logContainer = container.createEl('div', { cls: 'ana-log-container' });

        // Initial message
        this.log('info', 'ANA 준비 완료. "Process" 버튼을 클릭하거나 Ctrl+P → "ANA: Process Current Note"를 실행하세요.');

        // Input container (for questions)
        this.inputContainer = container.createEl('div', { cls: 'ana-input-container' });
        this.inputContainer.style.display = 'none';
    }

    async onClose(): Promise<void> {
        // Cleanup
    }

    /**
     * Clear the log
     */
    clear(): void {
        this.logContainer.empty();
        this.inputContainer.style.display = 'none';
        this.log('info', 'Log cleared.');
    }

    /**
     * Add a log entry
     */
    log(type: LogEntry['type'], content: string): void {
        const entry = this.logContainer.createEl('div', {
            cls: `ana-log-entry ana-log-${type}`
        });

        const time = new Date().toLocaleTimeString('ko-KR', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });

        const icons: Record<string, string> = {
            info: 'ℹ️',
            success: '✅',
            error: '❌',
            question: '❓',
            answer: '💬',
            preview: '📝'
        };

        entry.createEl('span', { text: `[${time}] ${icons[type]} `, cls: 'ana-log-time' });
        entry.createEl('span', { text: content });

        // Auto-scroll to bottom
        this.logContainer.scrollTop = this.logContainer.scrollHeight;
    }

    /**
     * Show analysis results and handle split suggestions
     * Returns selected topics to process (empty means continue with full note)
     */
    async showAnalysis(response: ProcessResponse): Promise<string[]> {
        return new Promise((resolve) => {
            this.log('info', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
            this.log('info', '📊 분석 결과');

            if (response.analysis) {
                this.log('info', `카테고리: ${response.analysis.category}`);
                this.log('info', `감지된 개념: ${response.analysis.detected_concepts.join(', ') || 'None'}`);
                this.log('info', `충분한 정보: ${response.analysis.is_sufficient ? '예' : '아니오'}`);

                if (response.analysis.should_split && response.analysis.split_suggestions.length > 0) {
                    const topics = response.analysis.split_suggestions;
                    this.log('info', `⚠️ ${topics.length}개의 개념이 감지되었습니다!`);
                    this.log('info', `분리 제안: ${topics.join(', ')}`);
                    this.log('info', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

                    // Show split choice buttons
                    this.inputContainer.empty();
                    this.inputContainer.style.display = 'flex';

                    const continueBtn = this.inputContainer.createEl('button', {
                        text: '▶ 전체 노트로 계속',
                        cls: 'ana-btn'
                    });
                    continueBtn.addEventListener('click', () => {
                        this.inputContainer.style.display = 'none';
                        this.log('info', '전체 노트로 계속 진행...');
                        resolve([]);
                    });

                    const allBtn = this.inputContainer.createEl('button', {
                        text: `📝 모두 분리 (${topics.length}개)`,
                        cls: 'ana-btn ana-btn-primary'
                    });
                    allBtn.addEventListener('click', () => {
                        this.inputContainer.style.display = 'none';
                        this.log('info', `${topics.length}개 주제를 순차적으로 처리합니다...`);
                        resolve([...topics]);
                    });

                    // Add buttons for each individual topic
                    for (const topic of topics) {
                        const splitBtn = this.inputContainer.createEl('button', {
                            text: `📝 ${topic}`,
                            cls: 'ana-btn ana-btn-sm'
                        });
                        splitBtn.addEventListener('click', () => {
                            this.inputContainer.style.display = 'none';
                            this.log('info', `"${topic}" 주제만 처리...`);
                            resolve([topic]);
                        });
                    }

                    this.logContainer.scrollTop = this.logContainer.scrollHeight;
                    return; // Wait for user choice
                }
            }

            this.log('info', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
            resolve([]);
        });
    }

    /**
     * Ask user to continue with next topic
     */
    async askContinueWithNextTopic(nextTopic: string, remaining: number): Promise<boolean> {
        return new Promise((resolve) => {
            this.log('info', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
            this.log('info', `✅ 현재 주제 완료!`);
            this.log('info', `다음 주제: "${nextTopic}" (남은 ${remaining}개)`);

            this.inputContainer.empty();
            this.inputContainer.style.display = 'flex';

            const continueBtn = this.inputContainer.createEl('button', {
                text: `▶ 다음: ${nextTopic}`,
                cls: 'ana-btn ana-btn-primary'
            });
            continueBtn.addEventListener('click', () => {
                this.inputContainer.style.display = 'none';
                this.log('info', `"${nextTopic}" 처리 시작...`);
                resolve(true);
            });

            const stopBtn = this.inputContainer.createEl('button', {
                text: '⏹ 여기서 중단',
                cls: 'ana-btn'
            });
            stopBtn.addEventListener('click', () => {
                this.inputContainer.style.display = 'none';
                this.log('info', '분리 처리 중단됨');
                resolve(false);
            });

            this.logContainer.scrollTop = this.logContainer.scrollHeight;
        });
    }

    /**
     * Show questions and get answers
     */
    async askQuestions(questions: Question[]): Promise<string[]> {
        return new Promise((resolve) => {
            this.currentQuestions = questions;
            this.currentAnswers = new Array(questions.length).fill('');
            this.currentQuestionIndex = 0;
            this.onAnswersComplete = resolve;

            this.log('info', `\n🤔 ${questions.length}개의 질문에 답해주세요:`);

            this.showNextQuestion();
        });
    }

    private showNextQuestion(): void {
        if (this.currentQuestionIndex >= this.currentQuestions.length) {
            // All questions answered
            this.inputContainer.style.display = 'none';
            this.log('success', '모든 질문에 답변 완료!');

            if (this.onAnswersComplete) {
                this.onAnswersComplete(this.currentAnswers);
                this.onAnswersComplete = null;
            }
            return;
        }

        const question = this.currentQuestions[this.currentQuestionIndex];
        const qNum = this.currentQuestionIndex + 1;
        const total = this.currentQuestions.length;

        this.log('question', `Q${qNum}/${total}: ${question.text}`);

        // Show input
        this.inputContainer.empty();
        this.inputContainer.style.display = 'flex';

        const inputWrapper = this.inputContainer.createEl('div', { cls: 'ana-input-wrapper' });

        inputWrapper.createEl('span', {
            text: `A${qNum}: `,
            cls: 'ana-input-label'
        });

        const input = inputWrapper.createEl('input', {
            type: 'text',
            cls: 'ana-input',
            attr: { placeholder: '답변 입력 (Enter로 제출, 빈 값으로 스킵)' }
        });

        input.focus();

        const submitBtn = this.inputContainer.createEl('button', {
            text: '→',
            cls: 'ana-btn ana-btn-primary ana-btn-sm'
        });

        const submitAnswer = () => {
            const answer = input.value.trim();
            this.currentAnswers[this.currentQuestionIndex] = answer;

            if (answer) {
                this.log('answer', `A${qNum}: ${answer}`);
            } else {
                this.log('answer', `A${qNum}: (스킵됨)`);
            }

            this.currentQuestionIndex++;
            this.showNextQuestion();
        };

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                submitAnswer();
            }
        });

        submitBtn.addEventListener('click', submitAnswer);

        // Skip all button
        const skipBtn = this.inputContainer.createEl('button', {
            text: 'Skip All',
            cls: 'ana-btn ana-btn-sm'
        });
        skipBtn.addEventListener('click', () => {
            this.currentQuestionIndex = this.currentQuestions.length;
            this.log('info', '나머지 질문 스킵됨');
            this.showNextQuestion();
        });
    }

    /**
     * Show draft note preview
     */
    async showPreview(draft: DraftNote): Promise<'save' | 'edit' | 'cancel'> {
        return new Promise((resolve) => {
            this.log('info', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
            this.log('preview', `📝 생성된 노트: ${draft.title}`);
            this.log('info', '━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

            // Show preview content
            const previewDiv = this.logContainer.createEl('div', { cls: 'ana-preview-inline' });

            // Frontmatter
            if (Object.keys(draft.frontmatter).length > 0) {
                const fmDiv = previewDiv.createEl('div', { cls: 'ana-fm-preview' });
                fmDiv.createEl('code', { text: '---\n' + JSON.stringify(draft.frontmatter, null, 2) + '\n---' });
            }

            // Content (truncated)
            const contentPreview = draft.content.length > 500
                ? draft.content.substring(0, 500) + '...'
                : draft.content;
            previewDiv.createEl('pre', { text: contentPreview, cls: 'ana-content-preview-text' });

            // Action buttons
            this.inputContainer.empty();
            this.inputContainer.style.display = 'flex';

            const saveBtn = this.inputContainer.createEl('button', {
                text: '💾 Save',
                cls: 'ana-btn ana-btn-primary'
            });
            saveBtn.addEventListener('click', () => {
                this.inputContainer.style.display = 'none';
                resolve('save');
            });

            const editBtn = this.inputContainer.createEl('button', {
                text: '✏️ Edit in Obsidian',
                cls: 'ana-btn'
            });
            editBtn.addEventListener('click', () => {
                this.inputContainer.style.display = 'none';
                resolve('edit');
            });

            const cancelBtn = this.inputContainer.createEl('button', {
                text: 'Cancel',
                cls: 'ana-btn'
            });
            cancelBtn.addEventListener('click', () => {
                this.inputContainer.style.display = 'none';
                resolve('cancel');
            });

            this.logContainer.scrollTop = this.logContainer.scrollHeight;
        });
    }

    /**
     * Show processing status
     */
    showProcessing(message: string): void {
        this.log('info', `⏳ ${message}...`);
    }

    /**
     * Show success message
     */
    showSuccess(message: string): void {
        this.log('success', message);
    }

    /**
     * Show error message
     */
    showError(message: string): void {
        this.log('error', message);
    }
}
