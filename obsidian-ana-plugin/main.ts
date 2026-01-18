/**
 * ANA - Atomic Note Architect Obsidian Plugin
 * 
 * Main plugin entry point with sidebar view.
 */

import { App, Editor, MarkdownView, Notice, Plugin, TFile, WorkspaceLeaf } from 'obsidian';
import { ANAApiClient, ProcessResponse, DraftNote } from './api';
import { ANASettings, ANASettingTab, DEFAULT_SETTINGS } from './settings';
import { ANASidebarView, VIEW_TYPE_ANA } from './sidebar';

export default class ANAPlugin extends Plugin {
    settings: ANASettings;
    apiClient: ANAApiClient;
    private currentSessionId: string | null = null;
    private sidebarView: ANASidebarView | null = null;

    async onload() {
        await this.loadSettings();

        // Initialize API client
        this.apiClient = new ANAApiClient(this.settings.serverUrl);

        // Register sidebar view
        this.registerView(
            VIEW_TYPE_ANA,
            (leaf) => {
                this.sidebarView = new ANASidebarView(leaf, this);
                return this.sidebarView;
            }
        );

        // Add ribbon icon to open sidebar
        this.addRibbonIcon('brain', 'Open ANA Panel', async () => {
            await this.activateSidebar();
        });

        // Add commands
        this.addCommand({
            id: 'open-sidebar',
            name: 'Open ANA Panel',
            callback: async () => {
                await this.activateSidebar();
            }
        });

        this.addCommand({
            id: 'process-current-note',
            name: 'Process Current Note',
            editorCallback: async (editor: Editor, view: MarkdownView) => {
                await this.activateSidebar();
                await this.processCurrentNote();
            }
        });

        this.addCommand({
            id: 'process-selection',
            name: 'Process Selected Text',
            editorCallback: async (editor: Editor, view: MarkdownView) => {
                const selection = editor.getSelection();
                if (selection) {
                    await this.activateSidebar();
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

        // Activate sidebar on startup if it was open
        this.app.workspace.onLayoutReady(() => {
            this.initLeaf();
        });
    }

    onunload() {
        // Cleanup
        if (this.currentSessionId) {
            this.apiClient.deleteSession(this.currentSessionId);
        }
        this.app.workspace.detachLeavesOfType(VIEW_TYPE_ANA);
    }

    private initLeaf(): void {
        if (this.app.workspace.getLeavesOfType(VIEW_TYPE_ANA).length === 0) {
            // Don't auto-open, user will open via ribbon or command
        }
    }

    async activateSidebar(): Promise<void> {
        const leaves = this.app.workspace.getLeavesOfType(VIEW_TYPE_ANA);

        if (leaves.length === 0) {
            // Create new leaf in right sidebar
            const leaf = this.app.workspace.getRightLeaf(false);
            if (leaf) {
                await leaf.setViewState({
                    type: VIEW_TYPE_ANA,
                    active: true,
                });
            }
        }

        // Focus the sidebar
        const activeLeaf = this.app.workspace.getLeavesOfType(VIEW_TYPE_ANA)[0];
        if (activeLeaf) {
            this.app.workspace.revealLeaf(activeLeaf);
            this.sidebarView = activeLeaf.view as ANASidebarView;
        }
    }

    async loadSettings() {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    }

    async saveSettings() {
        await this.saveData(this.settings);
        this.apiClient = new ANAApiClient(this.settings.serverUrl);
    }

    async checkServerConnection() {
        const isConnected = await this.apiClient.checkStatus();

        if (this.sidebarView) {
            if (isConnected) {
                this.sidebarView.showSuccess('ANA 서버 연결됨');
            } else {
                this.sidebarView.showError('서버에 연결할 수 없습니다. "ana serve" 실행 필요');
            }
        } else {
            new Notice(isConnected ? '✅ ANA server is running' : '❌ Cannot connect to ANA server');
        }
    }

    async processCurrentNote() {
        // Try to get active markdown view first
        let activeView = this.app.workspace.getActiveViewOfType(MarkdownView);

        // If not found (sidebar might be focused), search all leaves
        if (!activeView) {
            const leaves = this.app.workspace.getLeavesOfType('markdown');
            for (const leaf of leaves) {
                if (leaf.view instanceof MarkdownView) {
                    activeView = leaf.view;
                    break;
                }
            }
        }

        if (!activeView) {
            if (this.sidebarView) {
                this.sidebarView.showError('열린 마크다운 노트가 없습니다. Obsidian에서 노트를 먼저 열어주세요.');
            } else {
                new Notice('No active markdown note');
            }
            return;
        }

        const content = activeView.editor.getValue();
        const file = activeView.file;
        const title = file?.basename || 'Untitled';

        await this.processContent(content, title);
    }

    async processContent(content: string, title: string) {
        // Ensure sidebar is active
        if (!this.sidebarView) {
            await this.activateSidebar();
        }

        const view = this.sidebarView;
        if (!view) {
            new Notice('Failed to open ANA panel');
            return;
        }

        // Check server connection
        const isConnected = await this.apiClient.checkStatus();
        if (!isConnected) {
            view.showError('ANA 서버가 실행 중이지 않습니다. 터미널에서 "ana serve" 실행');
            return;
        }

        view.showProcessing(`"${title}" 처리 중`);

        try {
            // Step 1: Process the note
            const response = await this.apiClient.processNote(content, undefined, title);
            this.currentSessionId = response.session_id;

            // Show analysis and get topics to process
            const topics = await view.showAnalysis(response);

            if (topics.length === 0) {
                // Continue with full note
                await this.handleResponse(response, view);
            } else {
                // Process topics sequentially
                let currentResponse = response;

                for (let i = 0; i < topics.length; i++) {
                    const topic = topics[i];

                    view.log('info', `\n📝 주제 ${i + 1}/${topics.length}: ${topic}`);

                    // Process this topic
                    await this.handleResponse(currentResponse, view);

                    // If there are more topics, ask to continue
                    if (i < topics.length - 1) {
                        const nextTopic = topics[i + 1];
                        const remaining = topics.length - i - 1;
                        const shouldContinue = await view.askContinueWithNextTopic(nextTopic, remaining);

                        if (!shouldContinue) {
                            view.log('info', '분리 처리가 중단되었습니다.');
                            break;
                        }

                        // Process next topic with new session
                        view.showProcessing(`"${nextTopic}" 처리 중`);
                        currentResponse = await this.apiClient.processNote(content, undefined, nextTopic);
                        this.currentSessionId = currentResponse.session_id;
                    }
                }

                view.showSuccess('🎉 분리 처리 완료!');
            }

        } catch (error) {
            view.showError(`오류: ${error.message}`);
            this.currentSessionId = null;
        }
    }

    private async handleResponse(response: ProcessResponse, view: ANASidebarView) {
        if (response.status === 'needs_info' && response.questions.length > 0) {
            // Get answers via sidebar
            const answers = await view.askQuestions(response.questions);

            view.showProcessing('답변 처리 중');

            try {
                const newResponse = await this.apiClient.answerQuestions(
                    this.currentSessionId!,
                    answers
                );

                // Recursive call for more questions or completion
                await this.handleResponse(newResponse, view);
            } catch (error) {
                view.showError(`오류: ${error.message}`);
                this.cleanupSession();
            }
        } else if (response.status === 'completed' && response.draft_note) {
            // Show preview and get action
            const action = await view.showPreview(response.draft_note);

            if (action === 'save') {
                await this.saveNoteViaAPI(view);
            } else if (action === 'edit') {
                await this.createNoteInObsidian(response.draft_note, view);
            } else {
                view.log('info', '취소됨');
                this.cleanupSession();
            }
        }
    }

    private async saveNoteViaAPI(view: ANASidebarView) {
        if (!this.currentSessionId) return;

        try {
            const result = await this.apiClient.saveNote(this.currentSessionId);
            if (result.success) {
                view.showSuccess(`노트 저장됨: ${result.path}`);
            } else {
                view.showError(`저장 실패: ${result.message}`);
            }
        } catch (error) {
            view.showError(`저장 오류: ${error.message}`);
        } finally {
            this.cleanupSession();
        }
    }

    private async createNoteInObsidian(draftNote: DraftNote, view: ANASidebarView) {
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

            view.showSuccess(`Obsidian에 생성됨: ${fileName}`);
        } catch (error) {
            if (error.message.includes('already exists')) {
                view.showError(`파일이 이미 존재합니다: ${draftNote.title}.md`);
            } else {
                view.showError(`파일 생성 오류: ${error.message}`);
            }
        } finally {
            this.cleanupSession();
        }
    }

    private cleanupSession() {
        if (this.currentSessionId) {
            this.apiClient.deleteSession(this.currentSessionId);
            this.currentSessionId = null;
        }
    }
}
