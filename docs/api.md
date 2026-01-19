# ANA API Reference

> REST API for Atomic Note Architect

## Base URL

```
http://localhost:8000/api
```

Start the API server with:
```bash
ana serve
# or
ana serve --port 8080
```

---

## Endpoints

### Health Check

Check if the server is running.

```http
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "config": {
    "llm_provider": "ollama",
    "vault_path": "/path/to/vault"
  }
}
```

---

### Process Note

Process a raw note through the ANA pipeline.

```http
POST /api/process
Content-Type: application/json
```

**Request Body:**
```json
{
  "raw_note": "Your raw note content here...",
  "frontmatter": {
    "tags": ["optional", "tags"],
    "category": "optional-category"
  }
}
```

**Response (completed):**
```json
{
  "status": "completed",
  "draft_note": {
    "title": "Generated Title",
    "content": "Processed content...",
    "tags": ["tag1", "tag2"],
    "category": "determined-category",
    "suggested_links": ["[[Related Note 1]]", "[[Related Note 2]]"]
  },
  "analysis": {
    "detected_concepts": ["concept1", "concept2"],
    "is_sufficient": true
  }
}
```

**Response (needs_info):**
```json
{
  "status": "needs_info",
  "interaction": {
    "questions": [
      "Question 1?",
      "Question 2?"
    ],
    "context": "Why these questions are being asked..."
  },
  "draft_note": null
}
```

---

### Answer Questions

Provide answers to follow-up questions.

```http
POST /api/answer
Content-Type: application/json
```

**Request Body:**
```json
{
  "answers": [
    "Answer to question 1",
    "Answer to question 2"
  ]
}
```

**Response:**
Same format as `/api/process` response.

---

### Save Note

Save the processed note to the vault.

```http
POST /api/save
Content-Type: application/json
```

**Request Body (optional):**
```json
{
  "output_dir": "/custom/path",
  "overwrite": false,
  "apply_backlinks": true
}
```

**Response:**
```json
{
  "success": true,
  "path": "/path/to/saved/note.md",
  "backlinks_applied": 3
}
```

---

### Get Configuration

Get current ANA configuration.

```http
GET /api/config
```

**Response:**
```json
{
  "llm_provider": "ollama",
  "llm_model": "llama3.1:8b",
  "vault_path": "/home/user/vault",
  "max_questions": 5,
  "enable_note_linking": true
}
```

---

### Sync Embeddings

Sync embeddings for all notes in vault.

```http
POST /api/sync
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "updated": 15,
    "cached": 85,
    "failed": 0
  }
}
```

---

## Error Responses

All endpoints may return error responses:

```json
{
  "error": true,
  "message": "Description of the error",
  "code": "ERROR_CODE"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Invalid input data |
| `NO_SESSION` | No active processing session |
| `LLM_ERROR` | LLM provider error |
| `VAULT_ERROR` | Vault access error |
| `SAVE_ERROR` | Failed to save note |

---

## Usage Examples

### cURL

```bash
# Process a note
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"raw_note": "My note about Python..."}'

# Answer questions
curl -X POST http://localhost:8000/api/answer \
  -H "Content-Type: application/json" \
  -d '{"answers": ["Answer 1", "Answer 2"]}'

# Save the note
curl -X POST http://localhost:8000/api/save
```

### Python

```python
import requests

BASE_URL = "http://localhost:8000/api"

# Process note
response = requests.post(
    f"{BASE_URL}/process",
    json={"raw_note": "My raw note content..."}
)
result = response.json()

if result["status"] == "needs_info":
    # Answer questions
    answers = ["Answer 1", "Answer 2"]
    response = requests.post(
        f"{BASE_URL}/answer",
        json={"answers": answers}
    )
    result = response.json()

# Save when complete
if result["status"] == "completed":
    response = requests.post(f"{BASE_URL}/save")
    print(f"Saved to: {response.json()['path']}")
```

### JavaScript

```javascript
const BASE_URL = 'http://localhost:8000/api';

async function processNote(rawNote) {
  const response = await fetch(`${BASE_URL}/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ raw_note: rawNote })
  });
  return response.json();
}

async function answerQuestions(answers) {
  const response = await fetch(`${BASE_URL}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answers })
  });
  return response.json();
}
```
