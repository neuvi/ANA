/**
 * ANA API Client Module
 * Handles all communication with ANA backend API
 */

class ANAApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || window.location.origin;
        this.currentSessionId = null;
    }

    /**
     * Make API request with error handling
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}/api${endpoint}`;

        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const response = await fetch(url, { ...defaultOptions, ...options });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `API Error: ${response.status}`);
        }

        return response.json();
    }

    /**
     * Check server health
     */
    async checkHealth() {
        try {
            const result = await this.request('/health');
            return {
                isHealthy: result.status === 'ok',
                status: result.status,
                checks: result.checks || [],
            };
        } catch (error) {
            return {
                isHealthy: false,
                status: 'error',
                error: error.message,
            };
        }
    }

    /**
     * Get server status
     */
    async getStatus() {
        return this.request('/status');
    }

    /**
     * Process note through ANA pipeline
     * @param {string} content - Note content
     * @param {object} frontmatter - Optional frontmatter
     */
    async processNote(content, frontmatter = {}) {
        const result = await this.request('/process', {
            method: 'POST',
            body: JSON.stringify({ content, frontmatter }),
        });

        this.currentSessionId = result.session_id;
        return result;
    }

    /**
     * Answer questions from ANA
     * @param {string} sessionId - Session ID
     * @param {string[]} answers - Array of answers
     */
    async answerQuestions(sessionId, answers) {
        const result = await this.request('/answer', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                answers: answers,
            }),
        });
        return result;
    }

    /**
     * Get tag suggestions for content
     * @param {string} content - Note content
     * @param {string[]} existingTags - Already applied tags
     * @param {number} maxTags - Maximum tags to suggest
     */
    async suggestTags(content, existingTags = [], maxTags = 5) {
        return this.request('/tags/suggest', {
            method: 'POST',
            body: JSON.stringify({
                content,
                existing_tags: existingTags,
                max_tags: maxTags,
            }),
        });
    }

    /**
     * Get all vault tags
     */
    async getVaultTags() {
        return this.request('/tags');
    }

    /**
     * Normalize tags
     * @param {string[]} tags - Tags to normalize
     */
    async normalizeTags(tags) {
        return this.request('/tags/normalize', {
            method: 'POST',
            body: JSON.stringify({ tags }),
        });
    }

    /**
     * Get backlink suggestions
     * @param {string} title - Note title
     * @param {string} content - Note content
     * @param {string[]} tags - Note tags
     * @param {number} maxNotes - Max notes to scan
     */
    async suggestBacklinks(title, content, tags = [], maxNotes = 50) {
        return this.request('/backlinks/suggest', {
            method: 'POST',
            body: JSON.stringify({
                title,
                content,
                tags,
                max_notes_to_scan: maxNotes,
            }),
        });
    }

    /**
     * Get current configuration
     */
    async getConfig() {
        return this.request('/config');
    }

    /**
     * Sync embeddings
     * @param {boolean} force - Force re-sync all
     * @param {boolean} useAsync - Use async method
     */
    async syncEmbeddings(force = false, useAsync = true) {
        return this.request('/sync', {
            method: 'POST',
            body: JSON.stringify({ force, use_async: useAsync }),
        });
    }

    /**
     * Get sync statistics
     */
    async getSyncStats() {
        return this.request('/sync/stats');
    }

    /**
     * Delete session
     * @param {string} sessionId - Session to delete
     */
    async deleteSession(sessionId) {
        return this.request(`/session/${sessionId}`, {
            method: 'DELETE',
        });
    }
}

// Export for use in app.js
window.ANAApiClient = ANAApiClient;
