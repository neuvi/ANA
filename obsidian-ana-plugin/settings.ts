/**
 * ANA Settings
 * 
 * Plugin settings and settings tab UI.
 */

import { App, PluginSettingTab, Setting } from 'obsidian';
import type ANAPlugin from './main';

export interface ANASettings {
    serverUrl: string;
    autoSave: boolean;
    showPreview: boolean;
    maxQuestions: number;
}

export const DEFAULT_SETTINGS: ANASettings = {
    serverUrl: 'http://127.0.0.1:8765',
    autoSave: false,
    showPreview: true,
    maxQuestions: 5
};

export class ANASettingTab extends PluginSettingTab {
    plugin: ANAPlugin;

    constructor(app: App, plugin: ANAPlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        const { containerEl } = this;

        containerEl.empty();

        containerEl.createEl('h2', { text: 'ANA - Atomic Note Architect' });

        containerEl.createEl('p', {
            text: 'AI를 사용하여 원시 노트를 Zettelkasten 스타일 Atomic Note로 변환합니다.',
            cls: 'setting-item-description'
        });

        new Setting(containerEl)
            .setName('Server URL')
            .setDesc('ANA API 서버 주소 (터미널에서 "ana serve" 실행 필요)')
            .addText(text => text
                .setPlaceholder('http://127.0.0.1:8765')
                .setValue(this.plugin.settings.serverUrl)
                .onChange(async (value) => {
                    this.plugin.settings.serverUrl = value;
                    await this.plugin.saveSettings();
                }));

        new Setting(containerEl)
            .setName('Auto Save')
            .setDesc('처리 완료 후 자동으로 노트 저장')
            .addToggle(toggle => toggle
                .setValue(this.plugin.settings.autoSave)
                .onChange(async (value) => {
                    this.plugin.settings.autoSave = value;
                    await this.plugin.saveSettings();
                }));

        new Setting(containerEl)
            .setName('Show Preview')
            .setDesc('저장 전 생성된 노트 미리보기 표시')
            .addToggle(toggle => toggle
                .setValue(this.plugin.settings.showPreview)
                .onChange(async (value) => {
                    this.plugin.settings.showPreview = value;
                    await this.plugin.saveSettings();
                }));

        // Connection test button
        new Setting(containerEl)
            .setName('Connection Test')
            .setDesc('ANA 서버 연결 테스트')
            .addButton(button => button
                .setButtonText('Test Connection')
                .onClick(async () => {
                    button.setButtonText('Testing...');
                    const isConnected = await this.plugin.apiClient.checkStatus();
                    if (isConnected) {
                        button.setButtonText('✅ Connected');
                    } else {
                        button.setButtonText('❌ Failed');
                    }
                    setTimeout(() => button.setButtonText('Test Connection'), 2000);
                }));

        // Help section
        containerEl.createEl('h3', { text: '사용 방법' });

        const helpList = containerEl.createEl('ol');
        helpList.createEl('li', { text: '터미널에서 "ana serve" 실행' });
        helpList.createEl('li', { text: '노트를 열고 Command Palette (Ctrl/Cmd + P) 실행' });
        helpList.createEl('li', { text: '"ANA: Process Current Note" 선택' });
        helpList.createEl('li', { text: '질문에 답변하여 Atomic Note 생성' });
    }
}
