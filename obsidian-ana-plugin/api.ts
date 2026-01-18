/**
 * ANA API Client
 * 
 * Handles communication with the ANA API server.
 */

export interface AnalysisResult {
    detected_concepts: string[];
    is_sufficient: boolean;
    should_split: boolean;
    split_suggestions: string[];
    category: string;
}

export interface Question {
    text: string;
    category: string;
}

export interface DraftNote {
    title: string;
    content: string;
    frontmatter: Record<string, any>;
}

export interface ProcessResponse {
    session_id: string;
    status: string;
    analysis: AnalysisResult | null;
    questions: Question[];
    draft_note: DraftNote | null;
    message?: string;
}

export interface SaveResponse {
    success: boolean;
    path?: string;
    message?: string;
}

export class ANAApiClient {
    private baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
    }

    async checkStatus(): Promise<boolean> {
        try {
            const response = await fetch(`${this.baseUrl}/api/status`);
            return response.ok;
        } catch {
            return false;
        }
    }

    async processNote(content: string, frontmatter?: Record<string, any>, title?: string): Promise<ProcessResponse> {
        const response = await fetch(`${this.baseUrl}/api/process`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, frontmatter, title })
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`API error: ${error}`);
        }

        return await response.json();
    }

    async answerQuestions(sessionId: string, answers: string[]): Promise<ProcessResponse> {
        const response = await fetch(`${this.baseUrl}/api/answer`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, answers })
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`API error: ${error}`);
        }

        return await response.json();
    }

    async saveNote(sessionId: string, outputPath?: string, overwrite: boolean = false): Promise<SaveResponse> {
        const response = await fetch(`${this.baseUrl}/api/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                output_path: outputPath,
                overwrite
            })
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`API error: ${error}`);
        }

        return await response.json();
    }

    async deleteSession(sessionId: string): Promise<void> {
        await fetch(`${this.baseUrl}/api/session/${sessionId}`, {
            method: 'DELETE'
        });
    }
}
