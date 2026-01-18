/**
 * ANA Modals
 * 
 * Modal dialogs for questions and preview.
 */

import { App, Modal, Setting, MarkdownRenderer, Component } from 'obsidian';
import type { Question, DraftNote } from './api';

/**
 * Modal for answering questions from ANA
 */
export class QuestionModal extends Modal {
    private questions: Question[];
    private answers: string[] = [];
    private onSubmit: (answers: string[]) => void;
    private onCancel: () => void;

    constructor(
        app: App,
        questions: Question[],
        onSubmit: (answers: string[]) => void,
        onCancel: () => void
    ) {
        super(app);
        this.questions = questions;
        this.answers = new Array(questions.length).fill('');
        this.onSubmit = onSubmit;
        this.onCancel = onCancel;
    }

    onOpen() {
        const { contentEl } = this;

        contentEl.createEl('h2', { text: '🤔 추가 정보가 필요합니다' });

        contentEl.createEl('p', {
            text: '더 좋은 Atomic Note를 생성하기 위해 질문에 답해주세요.',
            cls: 'ana-modal-description'
        });

        const form = contentEl.createEl('div', { cls: 'ana-question-form' });

        this.questions.forEach((question, index) => {
            const questionDiv = form.createEl('div', { cls: 'ana-question-item' });

            // Category badge
            if (question.category && question.category !== 'general') {
                questionDiv.createEl('span', {
                    text: question.category,
                    cls: 'ana-category-badge'
                });
            }

            // Question text
            questionDiv.createEl('label', {
                text: `Q${index + 1}. ${question.text}`,
                cls: 'ana-question-label'
            });

            // Answer input
            const textarea = questionDiv.createEl('textarea', {
                cls: 'ana-answer-input',
                attr: {
                    placeholder: '답변을 입력하세요... (선택사항)',
                    rows: '3'
                }
            });

            textarea.addEventListener('input', (e) => {
                this.answers[index] = (e.target as HTMLTextAreaElement).value;
            });
        });

        // Buttons
        const buttonDiv = contentEl.createEl('div', { cls: 'ana-modal-buttons' });

        const cancelBtn = buttonDiv.createEl('button', {
            text: 'Cancel',
            cls: 'ana-btn ana-btn-secondary'
        });
        cancelBtn.addEventListener('click', () => {
            this.close();
            this.onCancel();
        });

        const submitBtn = buttonDiv.createEl('button', {
            text: 'Submit Answers',
            cls: 'ana-btn ana-btn-primary'
        });
        submitBtn.addEventListener('click', () => {
            this.close();
            this.onSubmit(this.answers);
        });

        // Skip button
        const skipBtn = buttonDiv.createEl('button', {
            text: 'Skip Questions',
            cls: 'ana-btn'
        });
        skipBtn.addEventListener('click', () => {
            this.close();
            this.onSubmit(this.answers.map(() => ''));
        });
    }

    onClose() {
        const { contentEl } = this;
        contentEl.empty();
    }
}

/**
 * Modal for previewing the generated note
 */
export class PreviewModal extends Modal {
    private draftNote: DraftNote;
    private onSave: () => void;
    private onEdit: () => void;
    private onCancel: () => void;

    constructor(
        app: App,
        draftNote: DraftNote,
        onSave: () => void,
        onEdit: () => void,
        onCancel: () => void
    ) {
        super(app);
        this.draftNote = draftNote;
        this.onSave = onSave;
        this.onEdit = onEdit;
        this.onCancel = onCancel;
    }

    async onOpen() {
        const { contentEl } = this;

        contentEl.createEl('h2', { text: '📝 생성된 Atomic Note' });

        // Title
        contentEl.createEl('h3', {
            text: this.draftNote.title,
            cls: 'ana-preview-title'
        });

        // Frontmatter preview
        if (Object.keys(this.draftNote.frontmatter).length > 0) {
            const fmDiv = contentEl.createEl('div', { cls: 'ana-frontmatter-preview' });
            fmDiv.createEl('strong', { text: 'Frontmatter:' });
            const fmPre = fmDiv.createEl('pre');
            fmPre.setText(JSON.stringify(this.draftNote.frontmatter, null, 2));
        }

        // Content preview
        const previewDiv = contentEl.createEl('div', { cls: 'ana-content-preview' });

        // Use Obsidian's markdown renderer
        await MarkdownRenderer.render(
            this.app,
            this.draftNote.content,
            previewDiv,
            '',
            new Component()
        );

        // Buttons
        const buttonDiv = contentEl.createEl('div', { cls: 'ana-modal-buttons' });

        const cancelBtn = buttonDiv.createEl('button', {
            text: 'Cancel',
            cls: 'ana-btn ana-btn-secondary'
        });
        cancelBtn.addEventListener('click', () => {
            this.close();
            this.onCancel();
        });

        const editBtn = buttonDiv.createEl('button', {
            text: 'Edit in New Note',
            cls: 'ana-btn'
        });
        editBtn.addEventListener('click', () => {
            this.close();
            this.onEdit();
        });

        const saveBtn = buttonDiv.createEl('button', {
            text: 'Save Note',
            cls: 'ana-btn ana-btn-primary'
        });
        saveBtn.addEventListener('click', () => {
            this.close();
            this.onSave();
        });
    }

    onClose() {
        const { contentEl } = this;
        contentEl.empty();
    }
}

/**
 * Modal for showing analysis results with split options
 */
export class AnalysisModal extends Modal {
    private concepts: string[];
    private category: string;
    private splitSuggestions: string[];
    private onContinue: () => void;
    private onSplit: (topic: string) => void;

    constructor(
        app: App,
        concepts: string[],
        category: string,
        splitSuggestions: string[],
        onContinue: () => void,
        onSplit: (topic: string) => void
    ) {
        super(app);
        this.concepts = concepts;
        this.category = category;
        this.splitSuggestions = splitSuggestions;
        this.onContinue = onContinue;
        this.onSplit = onSplit;
    }

    onOpen() {
        const { contentEl } = this;

        contentEl.createEl('h2', { text: '📊 분석 결과' });

        // Detected concepts
        const conceptsDiv = contentEl.createEl('div', { cls: 'ana-analysis-section' });
        conceptsDiv.createEl('strong', { text: '감지된 개념:' });
        const conceptsList = conceptsDiv.createEl('ul');
        this.concepts.forEach(concept => {
            conceptsList.createEl('li', { text: concept });
        });

        // Category
        contentEl.createEl('p', { text: `카테고리: ${this.category}` });

        // Split suggestions
        if (this.splitSuggestions.length > 0) {
            contentEl.createEl('h3', { text: '⚠️ 여러 개념이 감지되었습니다' });
            contentEl.createEl('p', { text: '노트를 분리하시겠습니까?' });

            const splitDiv = contentEl.createEl('div', { cls: 'ana-split-options' });
            this.splitSuggestions.forEach(suggestion => {
                const btn = splitDiv.createEl('button', {
                    text: suggestion,
                    cls: 'ana-btn ana-split-btn'
                });
                btn.addEventListener('click', () => {
                    this.close();
                    this.onSplit(suggestion);
                });
            });
        }

        // Buttons
        const buttonDiv = contentEl.createEl('div', { cls: 'ana-modal-buttons' });

        const continueBtn = buttonDiv.createEl('button', {
            text: 'Continue with Full Note',
            cls: 'ana-btn ana-btn-primary'
        });
        continueBtn.addEventListener('click', () => {
            this.close();
            this.onContinue();
        });
    }

    onClose() {
        const { contentEl } = this;
        contentEl.empty();
    }
}
