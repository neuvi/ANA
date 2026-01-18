/**
 * ANA - Atomic Note Architect Obsidian Plugin
 * 
 * Main plugin entry point.
 */

import { App, Editor, MarkdownView, Notice, Plugin, TFile } from 'obsidian';
import { ANAApiClient, ProcessResponse, DraftNote } from './api';
import { ANASettings, ANASettingTab, DEFAULT_SETTINGS } from './settings';
import { QuestionModal, PreviewModal, AnalysisModal } from './modal';

export default class ANAPlugin extends Plugin {
    settings: ANASettings;
    apiClient: ANAApiClient;
    private currentSessionId: string | null = null;

    async onload() {
        await this.loadSettings();

        // Initialize API client
        this.apiClient = new ANAApiClient(this.settings.serverUrl);

        // Add ribbon icon
        this.addRibbonIcon('brain', 'ANA: Process Note', async () => {
            await this.processCurrentNote();
        });

        // Add commands
        this.addCommand({
            id: 'process-current-note',
            name: 'Process Current Note',
            editorCallback: async (editor: Editor, view: MarkdownView) => {
                await this.processCurrentNote();
            }
        });

        this.addCommand({
            id: 'process-selection',
            name: 'Process Selected Text',
            editorCallback: async (editor: Editor, view: MarkdownView) => {
                const selection = editor.getSelection();
                if (selection) {
                    await this.processContent(selection, 'Selection');
                } else {
                    new Notice('No text selected');
                }
            }
        });

        this.addCommand({
            id: 'check-server',
            name: 'Check Server Connection',
            callback: async () => {
                await this.checkServerConnection();
            }
        });

        // Register settings tab
        this.addSettingTab(new ANASettingTab(this.app, this));
    }

    onunload() {
        // Cleanup any active sessions
        if (this.currentSessionId) {
            this.apiClient.deleteSession(this.currentSessionId);
        }
    }

    async loadSettings() {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    }

    async saveSettings() {
        await this.saveData(this.settings);
        // Update API client with new URL
        this.apiClient = new ANAApiClient(this.settings.serverUrl);
    }

    /**
     * Check server connection and show result
     */
    async checkServerConnection() {
        new Notice('Checking ANA server connection...');
        const isConnected = await this.apiClient.checkStatus();

        if (isConnected) {
            new Notice('✅ ANA server is running');
        } else {
            new Notice('❌ Cannot connect to ANA server. Make sure to run "ana serve"');
        }
    }

    /**
     * Process the current note
     */
    async processCurrentNote() {
        const activeView = this.app.workspace.getActiveViewOfType(MarkdownView);

        if (!activeView) {
            new Notice('No active markdown note');
            return;
        }

        const content = activeView.editor.getValue();
        const file = activeView.file;
        const title = file?.basename || 'Untitled';

        await this.processContent(content, title);
    }

    /**
     * Process content through ANA pipeline
     */
    async processContent(content: string, title: string) {
        // Check server connection first
        const isConnected = await this.apiClient.checkStatus();
        if (!isConnected) {
            new Notice('❌ ANA server not running. Run "ana serve" in terminal.');
            return;
        }

        new Notice('🔄 Processing note...');

        try {
            // Step 1: Process the note
            const response = await this.apiClient.processNote(content, undefined, title);
            this.currentSessionId = response.session_id;

            // Step 2: Handle analysis if there are split suggestions
            if (response.analysis?.should_split && response.analysis.split_suggestions.length > 0) {
                await this.handleAnalysis(response, content);
                return;
            }

            // Step 3: Handle questions
            await this.handleResponse(response);

        } catch (error) {
            new Notice(`❌ Error: ${error.message}`);
            this.currentSessionId = null;
        }
    }

    /**
     * Handle analysis with split options
     */
    private async handleAnalysis(response: ProcessResponse, originalContent: string) {
        if (!response.analysis) return;

        const modal = new AnalysisModal(
            this.app,
            response.analysis.detected_concepts,
            response.analysis.category,
            response.analysis.split_suggestions,
            // Continue with full note
            async () => {
                await this.handleResponse(response);
            },
            // Split by topic
            async (topic: string) => {
                new Notice(`Splitting for: ${topic}`);
                // For now, continue with the original content
                // In future, could extract specific content
                await this.handleResponse(response);
            }
        );
        modal.open();
    }

    /**
     * Handle response (questions or completion)
     */
    private async handleResponse(response: ProcessResponse) {
        if (response.status === 'needs_info' && response.questions.length > 0) {
            // Show question modal
            const modal = new QuestionModal(
                this.app,
                response.questions,
                // On submit
                async (answers: string[]) => {
                    await this.submitAnswers(answers);
                },
                // On cancel
                () => {
                    this.cleanupSession();
                    new Notice('Processing cancelled');
                }
            );
            modal.open();
        } else if (response.status === 'completed' && response.draft_note) {
            // Show preview or save directly
            if (this.settings.showPreview) {
                await this.showPreview(response.draft_note);
            } else if (this.settings.autoSave) {
                await this.saveNote();
            } else {
                await this.createNoteInObsidian(response.draft_note);
            }
        }
    }

    /**
     * Submit answers and continue processing
     */
    private async submitAnswers(answers: string[]) {
        if (!this.currentSessionId) return;

        new Notice('🔄 Processing answers...');

        try {
            const response = await this.apiClient.answerQuestions(this.currentSessionId, answers);
            await this.handleResponse(response);
        } catch (error) {
            new Notice(`❌ Error: ${error.message}`);
            this.cleanupSession();
        }
    }

    /**
     * Show preview modal
     */
    private async showPreview(draftNote: DraftNote) {
        const modal = new PreviewModal(
            this.app,
            draftNote,
            // On save
            async () => {
                await this.saveNote();
            },
            // On edit (create in Obsidian)
            async () => {
                await this.createNoteInObsidian(draftNote);
            },
            // On cancel
            () => {
                this.cleanupSession();
            }
        );
        modal.open();
    }

    /**
     * Save note via API
     */
    private async saveNote() {
        if (!this.currentSessionId) return;

        try {
            const result = await this.apiClient.saveNote(this.currentSessionId);
            if (result.success) {
                new Notice(`✅ Note saved: ${result.path}`);
            } else {
                new Notice(`❌ Save failed: ${result.message}`);
            }
        } catch (error) {
            new Notice(`❌ Error saving: ${error.message}`);
        } finally {
            this.cleanupSession();
        }
    }

    /**
     * Create note directly in Obsidian
     */
    private async createNoteInObsidian(draftNote: DraftNote) {
        try {
            // Build content with frontmatter
            let content = '';
            if (Object.keys(draftNote.frontmatter).length > 0) {
                content += '---\n';
                for (const [key, value] of Object.entries(draftNote.frontmatter)) {
                    if (Array.isArray(value)) {
                        content += `${key}:\n`;
                        value.forEach(v => content += `  - ${v}\n`);
                    } else {
                        content += `${key}: ${value}\n`;
                    }
                }
                content += '---\n\n';
            }
            content += draftNote.content;

            // Create file
            const fileName = `${draftNote.title}.md`;
            const file = await this.app.vault.create(fileName, content);

            // Open the new file
            await this.app.workspace.getLeaf().openFile(file);

            new Notice(`✅ Created: ${fileName}`);
        } catch (error) {
            if (error.message.includes('already exists')) {
                new Notice(`❌ File already exists: ${draftNote.title}.md`);
            } else {
                new Notice(`❌ Error creating file: ${error.message}`);
            }
        } finally {
            this.cleanupSession();
        }
    }

    /**
     * Cleanup session
     */
    private cleanupSession() {
        if (this.currentSessionId) {
            this.apiClient.deleteSession(this.currentSessionId);
            this.currentSessionId = null;
        }
    }
}
