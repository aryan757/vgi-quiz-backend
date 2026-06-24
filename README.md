# VGI MCQ Generator

A FastAPI backend that generates job-readiness MCQs for VGI Skill Lab's AI/ML curriculum and writes them into MongoDB. The frontend reads questions directly from the `question_bank` collection — this API never returns question content inline.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Variables](#environment-variables)
3. [How It Works — Full Flow](#how-it-works--full-flow)
4. [MongoDB Collections](#mongodb-collections)
5. [API Reference](#api-reference)
6. [Postman Testing Guide](#postman-testing-guide)
7. [Knowledge Base Seeding](#knowledge-base-seeding)
8. [Project Structure](#project-structure)
9. [Running Tests](#running-tests)
10. [TODOs Before Production](#todos-before-production)

---

## Quick Start

```bash
# 1. Create and activate the virtualenv
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — fill in OPENAI_API_KEY (always required) and DATABASE_URL

# 4. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | Yes | — | Full MongoDB URI with credentials |
| `DATABASE_NAME` | No | `vgi_skill_lab` | MongoDB database name |
| `KNOWLEDGE_BASE_COLLECTION` | No | `knowledge_base` | Internal seed content collection |
| `QUESTION_BANK_COLLECTION` | No | `question_bank` | Frontend-facing output collection |
| `LLM_PROVIDER` | No | `openai` | `openai` or `anthropic` — for question generation only |
| `LLM_MODEL_NAME` | No | `gpt-4o-mini` | Model name matching the selected provider |
| `OPENAI_API_KEY` | **Always** | — | Required even when `LLM_PROVIDER=anthropic` — embeddings always use OpenAI |
| `ANTHROPIC_API_KEY` | Only if `LLM_PROVIDER=anthropic` | — | |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |
| `TOPIC_MATCH_THRESHOLD` | No | `0.80` | Cosine similarity threshold for KB topic matching |
| `SEED_QUESTIONS_PER_TOPIC_PER_DIFFICULTY` | No | `50` | Seeding target per topic/difficulty slot |

---

## How It Works — Full Flow

When `POST /generate-questions` is called, the service runs a **5-step decision flow** to decide where questions come from. Understanding this flow is key to understanding the whole system.

```
POST /generate-questions
        │
        ▼
┌───────────────────────────────────────────────┐
│  STEP 1: Is description or job_description    │
│          present in the request?              │
└───────────────────────────────────────────────┘
        │                       │
       YES                      NO
        │                       │
        ▼                       ▼
  ┌──────────┐        ┌─────────────────────────────────────┐
  │ BRANCH C │        │  STEP 2: Topic Match Check          │
  │ Full LLM │        │  Does this topic exist in the       │
  │          │        │  knowledge_base?                    │
  └──────────┘        └─────────────────────────────────────┘
        │                       │                  │
        │               Confident Match        No Match
        │                       │                  │
        │                       ▼                  ▼
        │         ┌─────────────────────────┐  ┌──────────┐
        │         │  STEP 3: Check          │  │ BRANCH C │
        │         │  question_bank          │  │ Full LLM │
        │         │  inventory              │  └──────────┘
        │         └─────────────────────────┘
        │                       │
        │          ┌────────────┼────────────┐
        │       Has enough   Shortfall    Empty
        │          │            │            │
        │          ▼            ▼            ▼
        │      ┌────────┐ ┌─────────┐ ┌──────────┐
        │      │BRANCH A│ │BRANCH A │ │ BRANCH B │
        │      │  Full  │ │ Partial │ │ Full KB  │
        │      │ Reuse  │ │  Top-up │ │   Seed   │
        │      └────────┘ └─────────┘ └──────────┘
        │          │            │            │
        └──────────┴────────────┴────────────┘
                                │
                                ▼
                   Write new docs to question_bank
                                │
                                ▼
              Return { success, message, count }
```

### Branch Details

#### Branch A — Full Reuse (fastest, no LLM call, no write)
- **When**: topic matches KB, `question_bank` already has ≥ `question_count` docs for this topic/difficulty
- **What happens**: nothing is generated or written. Returns `count = question_count` immediately.
- **Cost**: zero — no LLM call, no embedding call, no DB write.

#### Branch A — Partial Top-up (KB fill + optional LLM)
- **When**: topic matches KB, `question_bank` has *some* but not enough docs
- **What happens**: pulls the shortfall from `knowledge_base` (ranked by embedding similarity), maps them into `question_bank` format, inserts them.
- **If KB itself doesn't have enough**: tops up the remainder via LLM (same as Branch C logic).

#### Branch B — Full KB Seed
- **When**: topic matches KB, `question_bank` is empty for this topic/difficulty
- **What happens**: pulls all `question_count` docs from `knowledge_base`, inserts into `question_bank`.
- **If KB doesn't have enough**: LLM fills the gap.

#### Branch C — Full LLM Generation
- **When**: `description` or `job_description` is present in the request, **OR** the topic has no confident match in `knowledge_base`
- **What happens**: builds a LangChain structured-output prompt (with grounding examples from KB if available), generates questions via OpenAI/Anthropic, inserts into `question_bank`.
- **Retry**: if the LLM returns invalid structured output, retries once with a correction prompt before failing.

---

### Topic Matching (Step 2 detail)

The topic matcher decides whether a request topic maps to something in the knowledge base. It runs three checks in order, stopping as soon as one resolves:

```
1. topic == "auto"
   → No match needed. Generates broadly across the taxonomy
     (domain inferred from description/job_description if available).

2. Fast path (no API call)
   → Case-insensitive exact match or string containment against
     all known canonical topic names.
   → e.g. "transformers" → matches "Transformers"
   → e.g. "the Transformers guide" → matches "Transformers"
   → Resolves ~90% of real requests without any API call.

3. Embedding fallback (live OpenAI API call)
   → Embeds the incoming topic string via text-embedding-3-small.
   → Compares against the distinct topic_embedding values stored
     in knowledge_base (one per unique topic, cached in-process).
   → Cosine similarity ≥ 0.80 → confident match.
   → Below threshold → no confident match → Branch C.
```

> **Note**: OpenAI embeddings are always used (even when `LLM_PROVIDER=anthropic`). The fast path avoids this on most requests — make sure it stays working as topic volume grows.

---

### count semantics (important)

The `count` field in the response means **total questions now satisfying the request in `question_bank`** — that is, existing reused + newly inserted. It is NOT strictly "questions written this call." See `services/quiz_generator.py::compute_response_count` if you need to change this.

---

## MongoDB Collections

Two collections. **Do not conflate them.**

### `knowledge_base` (internal seed content — never read by frontend)
- Pre-seeded reference content. The "source of truth" the API draws from.
- Written by the seeding script (`scripts/seed_knowledge_base.py`), never by the live API.
- Contains richer fields: `embedding`, `topic_embedding`, `job_relevance`, `quality_reviewed`.
- Difficulty stored as **lowercase**: `beginner / intermediate / advanced`

### `question_bank` (frontend-facing output log)
- Every question this API has ever served, in the exact schema the frontend expects.
- Written by the live `/generate-questions` endpoint on every call.
- Difficulty stored as **UPPERCASE**: `BEGINNER / INTERMEDIATE / ADVANCED`
- Uses `correctAnswer` (camelCase) — do not rename this field.

```json
{
  "_id": "685a3d89c87f21a3b7d8d123",
  "type": "CUSTOM",
  "topic": "Transformers",
  "difficulty": "INTERMEDIATE",
  "question": "What is the primary advantage of the Transformer over RNNs?",
  "options": [
    {"key": "A", "value": "Lower memory usage", "position": 1},
    {"key": "B", "value": "Parallel processing of sequences", "position": 2},
    {"key": "C", "value": "Smaller model size", "position": 3},
    {"key": "D", "value": "No need for training data", "position": 4}
  ],
  "correctAnswer": ["B"],
  "explanation": "Transformers process tokens in parallel using self-attention.",
  "createdAt": "2026-06-24T10:00:00Z",
  "updatedAt": "2026-06-24T10:00:00Z"
}
```

---

## API Reference

### `GET /health`
Liveness + Mongo connectivity check.

**Response**
```json
{
  "status": "ok",
  "mongo": "up",
  "llm_provider": "openai",
  "embedding_model": "text-embedding-3-small"
}
```

---

### `POST /generate-questions`

Generates MCQs and writes them to `question_bank`. Does **not** return question content.

**Request body**

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `type` | `CUSTOM` \| `DAILY` \| `SESSION` | Yes | — | Only `CUSTOM` implemented. `DAILY`/`SESSION` return 501. |
| `topic` | string | Yes | — | Any topic name, or `"auto"` to spread across the taxonomy |
| `difficulty` | `BEGINNER` \| `INTERMEDIATE` \| `ADVANCED` \| `MIXED` | No | `MIXED` | `MIXED` = no difficulty constraint |
| `description` | string | No | — | Extra context to steer LLM output. If present, skips KB lookup entirely. |
| `job_description` | string | No | — | Job description to calibrate question relevance. If present, skips KB lookup. |
| `question_count` | integer 1–100 | Yes | — | How many questions to satisfy |

**Response (always this shape)**
```json
{
  "success": true,
  "message": "Questions generated successfully",
  "count": 10
}
```

**Error responses**

| Scenario | HTTP Status | `success` |
|---|---|---|
| Invalid enum value / out-of-range `question_count` | `422` | — (FastAPI validation error) |
| `type` is `DAILY` or `SESSION` | `501` | `false` |
| LLM call failed after retry | `502` | `false` |
| MongoDB write failed | `503` | `false` |

---

## Postman Testing Guide

Base URL: `http://localhost:8000`

---

### 1. Health Check
```
Method : GET
URL    : http://localhost:8000/health
```
Expected: `200 OK`
```json
{ "status": "ok", "mongo": "up", "llm_provider": "openai", "embedding_model": "text-embedding-3-small" }
```

---

### 2. Basic — topic only, no description
*Triggers Branch A/B (KB lookup) or Branch C (LLM) depending on KB state.*
```
Method : POST
URL    : http://localhost:8000/generate-questions
Headers: Content-Type: application/json
Body:
```
```json
{
  "type": "CUSTOM",
  "topic": "Transformers",
  "difficulty": "INTERMEDIATE",
  "question_count": 10
}
```
Expected: `200 OK` → `{ "success": true, "message": "Questions generated successfully", "count": 10 }`

---

### 3. With description — steers LLM output
*Always Branch C — KB lookup is skipped entirely.*
```json
{
  "type": "CUSTOM",
  "topic": "RAG",
  "difficulty": "ADVANCED",
  "description": "Focus on hybrid search and reranking strategies",
  "question_count": 10
}
```
Expected: `200 OK` → `{ "success": true, "count": 10 }`

---

### 4. With job description
*Always Branch C — job description personalises question relevance.*
```json
{
  "type": "CUSTOM",
  "topic": "LSTM",
  "difficulty": "BEGINNER",
  "job_description": "NLP Engineer at a fintech startup",
  "question_count": 5
}
```
Expected: `200 OK` → `{ "success": true, "count": 5 }`

---

### 5. Both description + job description
```json
{
  "type": "CUSTOM",
  "topic": "LangChain",
  "difficulty": "INTERMEDIATE",
  "description": "Focus on LCEL and custom tool creation",
  "job_description": "Senior AI Engineer building internal LLM tooling",
  "question_count": 8
}
```
Expected: `200 OK` → `{ "success": true, "count": 8 }`

---

### 6. MIXED difficulty (omit difficulty field)
*Difficulty defaults to MIXED — no difficulty filter applied.*
```json
{
  "type": "CUSTOM",
  "topic": "Random Forest",
  "question_count": 10
}
```
Expected: `200 OK` → `{ "success": true, "count": 10 }`

---

### 7. Auto topic — no specific topic
*Spreads questions across multiple topics in the taxonomy.*
```json
{
  "type": "CUSTOM",
  "topic": "auto",
  "difficulty": "INTERMEDIATE",
  "question_count": 10
}
```
Expected: `200 OK` → `{ "success": true, "count": 10 }`

---

### 8. Auto topic with domain hint via description
```json
{
  "type": "CUSTOM",
  "topic": "auto",
  "difficulty": "BEGINNER",
  "description": "Focus on computer vision topics",
  "question_count": 10
}
```
Expected: `200 OK` → `{ "success": true, "count": 10 }`

---

### 9. DAILY → 501 Not Implemented
```json
{
  "type": "DAILY",
  "topic": "Transformers",
  "question_count": 5
}
```
Expected: `501` → `{ "success": false, "message": "type='DAILY' is not implemented yet. Only CUSTOM is supported.", "count": 0 }`

---

### 10. SESSION → 501 Not Implemented
```json
{
  "type": "SESSION",
  "topic": "Transformers",
  "question_count": 5
}
```
Expected: `501` → `{ "success": false, "count": 0 }`

---

### 11. question_count too high → 422
```json
{
  "type": "CUSTOM",
  "topic": "Transformers",
  "question_count": 200
}
```
Expected: `422` — `"Input should be less than or equal to 100"`

---

### 12. question_count = 0 → 422
```json
{
  "type": "CUSTOM",
  "topic": "Transformers",
  "question_count": 0
}
```
Expected: `422` — `"Input should be greater than or equal to 1"`

---

### 13. Invalid difficulty → 422
```json
{
  "type": "CUSTOM",
  "topic": "Transformers",
  "difficulty": "EXPERT",
  "question_count": 5
}
```
Expected: `422` — `"Input should be 'BEGINNER', 'INTERMEDIATE', 'ADVANCED' or 'MIXED'"`

---

### 14. Missing required field (topic) → 422
```json
{
  "type": "CUSTOM",
  "question_count": 5
}
```
Expected: `422` — `"Field required"` on `topic`

---

### 15. Missing required field (question_count) → 422
```json
{
  "type": "CUSTOM",
  "topic": "Transformers"
}
```
Expected: `422` — `"Field required"` on `question_count`

---

## Knowledge Base Seeding

The `knowledge_base` collection is pre-seeded offline — the live API never writes to it. Seeding gives the API a pool of vetted questions to reuse, making repeat requests faster and free (no LLM call).

> **The API works fine with an empty KB.** The LLM always acts as fallback. The KB is an optimisation, not a requirement.

### Seeding via the script (automated)

```bash
# Test run first — verify quality before scaling
python scripts/seed_knowledge_base.py --topic "Transformers" --difficulty intermediate

# Then seed domain by domain
python scripts/seed_knowledge_base.py --domain genai
python scripts/seed_knowledge_base.py --domain machine_learning
python scripts/seed_knowledge_base.py --domain deep_learning
python scripts/seed_knowledge_base.py --domain computer_vision

# Dry run — see the plan without writing anything
python scripts/seed_knowledge_base.py --domain genai --dry-run
```

The script is **resumable** — progress is saved in `scripts/.seed_progress.json`. Restart any interrupted run with the same command and it skips completed slots automatically.

### Seeding via another LLM (manual import)

If you want to generate questions with another LLM (ChatGPT, Gemini, etc.) and import them:

```bash
# Generate the prompts (4 domain files, one per domain)
python scripts/generate_prompts.py

# This creates:
#   prompts/computer_vision.txt   — paste into ChatGPT
#   prompts/machine_learning.txt
#   prompts/deep_learning.txt
#   prompts/genai.txt
```

1. Open each `.txt` file, paste into your LLM of choice
2. Save the JSON response as the matching `.json` file (e.g. `prompts/computer_vision.json`)
3. Import it:

```bash
# Import one domain
python scripts/import_questions.py prompts/computer_vision.json

# Import all at once (after all JSONs are filled)
bash prompts/IMPORT_ALL.sh
```

---

## Project Structure

```
app/
  config.py                       # All env vars in one place (pydantic-settings)
  db.py                           # Motor async client + index setup
  main.py                         # FastAPI app, /health endpoint, lifespan
  kb_taxonomy.py                  # domain → topic → subtopics (edit here to add topics)
  errors.py                       # Domain exceptions (LLMError, DBError, etc.)
  models/
    enums.py                      # QuizType, Difficulty enums
    request.py                    # GenerateQuestionsRequest
    response.py                   # GenerateQuestionsResponse
    question_bank_document.py     # Frontend-facing document schema
    knowledge_base_document.py    # Internal seed document schema
  routes/
    generate_questions.py         # Thin route — validates, delegates, maps errors to HTTP codes
  services/
    llm_provider.py               # Anthropic / OpenAI switch via env var
    embeddings.py                 # OpenAI embeddings wrapper (async + sync for seeding)
    topic_matcher.py              # Fast-path string match → embedding fallback
    kb_retriever.py               # Filter + rank knowledge_base docs by similarity
    llm_question_generator.py     # Structured output chain + retry on parse failure
    quiz_generator.py             # 5-step orchestration — the core logic lives here
  utils/
    transforms.py                 # ONLY place that maps correct_answer ↔ correctAnswer
    similarity.py                 # Cosine similarity helpers
scripts/
  seed_knowledge_base.py          # Offline batch seeder (resumable, dedup)
  generate_prompts.py             # Generates domain prompt files for manual LLM seeding
  import_questions.py             # Imports LLM-generated JSON into knowledge_base
prompts/
  computer_vision.txt             # Ready-to-paste domain prompts for external LLMs
  machine_learning.txt
  deep_learning.txt
  genai.txt
  IMPORT_ALL.sh                   # Run after all domain JSONs are filled
tests/
  test_request_validation.py      # Enum, range, required-field validation
  test_topic_matcher.py           # Fast-path, embedding fallback, domain inference
  test_kb_retriever.py            # Filter/rank + transforms field mapping
  test_quiz_generator_flow.py     # All 5 branch paths (A full, A shortfall, B, C)
```

---

## Running Tests

All 40 tests run fully offline — no live Mongo or OpenAI calls needed.

```bash
pytest tests/ -v
```

---

## TODOs Before Production

- **Authentication / rate limiting** — none currently. Add API key auth or JWT before any public exposure.
- **`count` semantics** — `compute_response_count` in `services/quiz_generator.py` uses "total satisfying the request" (existing + newly inserted), not "newly inserted only." Confirm the intended meaning with the frontend team before relying on this in production logic.
- **DAILY / SESSION types** — accepted by the enum but return `501`. Implement when specs are confirmed.
- **Spot-check seeded content** — all seeded docs have `quality_reviewed: false` by default. Review a sample per domain before using for production quizzes.
- **Embedding cost at scale** — every request that doesn't resolve via the fast-path string match makes a live OpenAI embeddings API call. Monitor this as request volume grows.
