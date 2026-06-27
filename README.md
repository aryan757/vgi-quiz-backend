# VGI MCQ Generator

A FastAPI backend that generates job-readiness MCQs for VGI Skill Lab's AI/ML curriculum and writes them into MongoDB. The frontend reads questions directly from the `question_bank` collection — this API never returns question content inline.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Variables](#environment-variables)
3. [How It Works — Full Flow](#how-it-works--full-flow)
4. [MongoDB Collections](#mongodb-collections)
5. [Supported Domains](#supported-domains)
6. [API Reference](#api-reference)
7. [Postman Testing Guide](#postman-testing-guide)
8. [Knowledge Base Seeding](#knowledge-base-seeding)
9. [Project Structure](#project-structure)
10. [Running Tests](#running-tests)
11. [TODOs Before Production](#todos-before-production)

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
# Edit .env — fill in OPENAI_API_KEY and DATABASE_URL

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
| `TOPIC_MATCH_THRESHOLD` | No | `0.80` | Cosine similarity threshold for domain matching |
| `SEED_QUESTIONS_PER_TOPIC_PER_DIFFICULTY` | No | `50` | Seeding target per domain/difficulty slot |

---

## How It Works — Full Flow

When `POST /generate-questions` is called, the service runs a **5-step decision flow**.

```
POST /generate-questions
        │
        ▼
┌─────────────────────────────────────────┐
│  STEP 1: description or job_description │
│          present in the request?        │
└─────────────────────────────────────────┘
        │                     │
       YES                    NO
        │                     │
        ▼                     ▼
  ┌──────────┐     ┌──────────────────────────────────┐
  │ BRANCH C │     │  STEP 2: Domain Match Check      │
  │ Full LLM │     │  Does the topic map to one of    │
  │          │     │  our 5 known domains?            │
  └──────────┘     └──────────────────────────────────┘
        │                     │                │
        │             Confident Match       No Match
        │                     │                │
        │                     ▼                ▼
        │        ┌────────────────────┐   ┌──────────┐
        │        │  STEP 3: Check     │   │ BRANCH C │
        │        │  question_bank     │   │ Full LLM │
        │        │  inventory         │   └──────────┘
        │        └────────────────────┘
        │                     │
        │         ┌───────────┼───────────┐
        │      Has enough  Shortfall   Empty
        │          │           │           │
        │          ▼           ▼           ▼
        │     ┌────────┐ ┌─────────┐ ┌─────────┐
        │     │BRANCH A│ │BRANCH A │ │BRANCH B │
        │     │ Reuse  │ │ Top-up  │ │ KB Seed │
        │     └────────┘ └─────────┘ └─────────┘
        │          │           │           │
        └──────────┴───────────┴───────────┘
                               │
                               ▼
              Write new docs to question_bank
                               │
                               ▼
             Return { success, message, count }
```

### Branch Details

| Branch | When | What happens | LLM call? |
|---|---|---|---|
| **A — Full Reuse** | Domain matches, `question_bank` already has ≥ requested | Nothing generated or written | No |
| **A — Shortfall** | Domain matches, `question_bank` has some but not enough | Pulls remainder from `knowledge_base`, inserts | Only if KB also short |
| **B — KB Seed** | Domain matches, `question_bank` empty for this domain/difficulty | Pulls all from `knowledge_base`, inserts | Only if KB short |
| **C — Full LLM** | `description`/`job_description` present, OR no domain match | Generates via LLM with KB grounding examples, inserts | Yes |

### Domain Matching (Step 2)

The `topic` field in the request is matched against the 5 known domains. Three checks run in order:

```
1. topic == "auto"   → no match, generates broadly across domains

2. Fast path         → checks the topic against canonical domain names
                       AND a set of aliases (no API call)
                       e.g. "cv"             → computer_vision
                            "llm"            → genai
                            "neural network" → deep_learning
                            "sklearn"        → machine_learning
                            "ai"             → ai_fundamentals

3. Embedding fallback → embeds the topic via OpenAI, compares against
                        domain_embedding values stored in knowledge_base
                        cosine ≥ 0.80 → confident match
                        cosine < 0.80 → no match → Branch C
```

---

## MongoDB Collections

### `knowledge_base` (internal — never read by frontend)

Stores pre-seeded questions organised by **domain + difficulty only**. No topic or subtopic fields.

```json
{
  "_id": "ObjectId",
  "domain": "genai",
  "difficulty": "intermediate",
  "question": "What is the primary purpose of RAG?",
  "options": [
    {"key": "A", "value": "...", "position": 1},
    {"key": "B", "value": "...", "position": 2},
    {"key": "C", "value": "...", "position": 3},
    {"key": "D", "value": "...", "position": 4}
  ],
  "correct_answer": ["B"],
  "explanation": "...",
  "job_relevance": "...",
  "embedding": [...],         // 1536-dim content embedding for retrieval ranking
  "domain_embedding": [...],  // 1536-dim domain embedding for domain matching
  "quality_reviewed": false,
  "created_at": "ISODate"
}
```

**Current KB inventory: 844 questions**

| Domain | beginner | intermediate | advanced | mixed | Total |
|---|---|---|---|---|---|
| `ai_fundamentals` | 55 | 37 | 31 | 30 | **153** |
| `computer_vision` | 78 | 73 | 43 | 49 | **243** |
| `deep_learning` | 47 | 42 | 32 | 26 | **147** |
| `genai` | 57 | 31 | 15 | 17 | **120** |
| `machine_learning` | 65 | 45 | 34 | 37 | **181** |

### `question_bank` (frontend-facing output log)

Every question this API has ever served. Written by the live endpoint only.

```json
{
  "_id": "685a3d89c87f21a3b7d8d123",
  "type": "CUSTOM",
  "topic": "genai",
  "difficulty": "INTERMEDIATE",
  "question": "What is the primary purpose of RAG?",
  "options": [
    {"key": "A", "value": "...", "position": 1},
    {"key": "B", "value": "...", "position": 2},
    {"key": "C", "value": "...", "position": 3},
    {"key": "D", "value": "...", "position": 4}
  ],
  "correctAnswer": ["B"],
  "explanation": "...",
  "createdAt": "2026-06-25T10:00:00Z",
  "updatedAt": "2026-06-25T10:00:00Z"
}
```

> `correctAnswer` is camelCase — this is intentional per the frontend contract. Do not rename it.

### `random-question-collection` (output log for `GET /random-questions`)

One document per `GET /random-questions` call — a saved batch of the AI-generated set.

```json
{
  "_id": "ObjectId",
  "batch_id": "e7e55dee-047d-4014-b29c-fa2706578991",
  "count": 15,
  "questions": [
    {
      "domain": "computer_vision",
      "difficulty": "beginner",
      "question": "...",
      "options": [ {"key": "A", "value": "...", "position": 1}, ... ],
      "correct_answer": ["B"],
      "explanation": "...",
      "job_relevance": "..."
    }
  ],
  "created_at": "ISODate"
}
```

---

## Supported Domains

The `topic` field accepts either an exact domain name or any recognised alias:

| Domain | Accepted values (examples) |
|---|---|
| `computer_vision` | `computer_vision`, `cv`, `vision`, `yolo`, `cnn`, `object detection` |
| `machine_learning` | `machine_learning`, `ml`, `sklearn`, `regression`, `random forest` |
| `deep_learning` | `deep_learning`, `dl`, `neural network`, `lstm`, `transformer` |
| `genai` | `genai`, `llm`, `rag`, `langchain`, `agents`, `fine-tuning`, `gpt` |
| `ai_fundamentals` | `ai_fundamentals`, `ai`, `ml basics`, `fundamentals`, `mlops` |

Any value not matching the above goes to the embedding matcher, and if still unmatched, the LLM generates freely on that topic.

---

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "mongo": "up",
  "llm_provider": "openai",
  "embedding_model": "text-embedding-3-small"
}
```

### `POST /generate-questions`

**Request body**

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `type` | `CUSTOM` \| `DAILY` \| `SESSION` | Yes | — | Only `CUSTOM` is implemented |
| `topic` | string | Yes | — | Domain name, alias, or `"auto"` |
| `difficulty` | `BEGINNER` \| `INTERMEDIATE` \| `ADVANCED` \| `MIXED` | No | `MIXED` | |
| `description` | string | No | — | Extra LLM steering context. If present, skips KB entirely. |
| `job_description` | string | No | — | Calibrates question relevance. If present, skips KB entirely. |
| `question_count` | integer 1–100 | Yes | — | |

**Response**
```json
{
  "success": true,
  "message": "Questions generated successfully",
  "count": 10
}
```

**Error responses**

| Scenario | HTTP |
|---|---|
| Invalid enum / out-of-range `question_count` | `422` |
| `type` is `DAILY` or `SESSION` | `501` |
| LLM call failed after retry | `502` |
| MongoDB write failed | `503` |

### `GET /random-questions`

**AI-generates** a fresh set of questions on every call (it does **not** read from the
`knowledge_base`). The set is spread **across all domains** at an **easy–medium** level
(`beginner`/`intermediate` only). Each generated batch is saved to the
`random-question-collection` Mongo collection.

- 15 questions → ≈3 from each of the 5 domains (`computer_vision`, `machine_learning`,
  `deep_learning`, `genai`, `ai_fundamentals`).
- Uses the LLM via LangChain structured output (same provider/model as `/generate-questions`).

**Query params**

| Param | Type | Required | Default | Notes |
|---|---|---|---|---|
| `count` | integer 1–50 | No | `15` | Number of questions to generate, spread across domains |

**Response**
```json
{
  "success": true,
  "message": "Generated 15 questions across all domains.",
  "count": 15,
  "questions": [
    {
      "domain": "computer_vision",
      "difficulty": "beginner",
      "question": "What is the primary purpose of image classification?",
      "options": [ { "key": "A", "value": "...", "position": 1 }, ... ],
      "correct_answer": ["B"],
      "explanation": "...",
      "job_relevance": "..."
    }
  ]
}
```

**Saved batch** (in `random-question-collection`):
```json
{ "batch_id": "<uuid>", "count": 15, "questions": [ ... ], "created_at": "<utc>" }
```

| Scenario | HTTP |
|---|---|
| `count` out of range (not 1–50) | `422` |
| LLM returned no valid questions / generation failed | `502` |

> Mongo save failures are logged but do **not** fail the response — you still get the
> generated questions back.

---

## Postman Testing Guide

**Base URL:** `http://localhost:8000`

Start the server first:
```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

### Happy path — KB reuse (no LLM cost)

**1. genai — INTERMEDIATE**
```json
{"type":"CUSTOM","topic":"genai","difficulty":"INTERMEDIATE","question_count":10}
```
Expected: `200` → `{ "success": true, "count": 10 }`

**2. computer_vision — BEGINNER**
```json
{"type":"CUSTOM","topic":"computer_vision","difficulty":"BEGINNER","question_count":10}
```

**3. machine_learning — ADVANCED**
```json
{"type":"CUSTOM","topic":"machine_learning","difficulty":"ADVANCED","question_count":10}
```

**4. deep_learning — MIXED (omit difficulty)**
```json
{"type":"CUSTOM","topic":"deep_learning","question_count":10}
```

**5. ai_fundamentals — BEGINNER**
```json
{"type":"CUSTOM","topic":"ai_fundamentals","difficulty":"BEGINNER","question_count":8}
```

---

### Alias matching

**6. `cv` → computer_vision**
```json
{"type":"CUSTOM","topic":"cv","difficulty":"INTERMEDIATE","question_count":5}
```

**7. `llm` → genai**
```json
{"type":"CUSTOM","topic":"llm","difficulty":"BEGINNER","question_count":5}
```

**8. `neural network` → deep_learning**
```json
{"type":"CUSTOM","topic":"neural network","difficulty":"ADVANCED","question_count":5}
```

**9. `sklearn` → machine_learning**
```json
{"type":"CUSTOM","topic":"sklearn","difficulty":"BEGINNER","question_count":5}
```

**10. `ai` → ai_fundamentals**
```json
{"type":"CUSTOM","topic":"ai","difficulty":"MIXED","question_count":5}
```

---

### Branch C — LLM generation (with context)

**11. With description**
```json
{
  "type": "CUSTOM",
  "topic": "genai",
  "difficulty": "ADVANCED",
  "description": "Focus on LangGraph multi-agent orchestration",
  "question_count": 5
}
```

**12. With job_description**
```json
{
  "type": "CUSTOM",
  "topic": "machine_learning",
  "difficulty": "INTERMEDIATE",
  "job_description": "Data Scientist at a healthcare startup",
  "question_count": 5
}
```

**13. topic=auto (spreads across domains)**
```json
{"type":"CUSTOM","topic":"auto","difficulty":"BEGINNER","question_count":6}
```

**14. Unknown domain (LLM fallback)**
```json
{"type":"CUSTOM","topic":"quantum computing","difficulty":"BEGINNER","question_count":3}
```

---

### Error cases

**15. DAILY → 501**
```json
{"type":"DAILY","topic":"genai","question_count":5}
```
Expected: `501` → `{ "success": false, "message": "type='DAILY' is not implemented...", "count": 0 }`

**16. SESSION → 501**
```json
{"type":"SESSION","topic":"genai","question_count":5}
```

**17. question_count=0 → 422**
```json
{"type":"CUSTOM","topic":"genai","question_count":0}
```

**18. question_count=200 → 422**
```json
{"type":"CUSTOM","topic":"genai","question_count":200}
```

**19. Invalid difficulty → 422**
```json
{"type":"CUSTOM","topic":"genai","difficulty":"EXPERT","question_count":5}
```

**20. Missing topic → 422**
```json
{"type":"CUSTOM","question_count":5}
```

**21. Missing question_count → 422**
```json
{"type":"CUSTOM","topic":"genai"}
```

---

### Random questions (GET — AI-generated, all domains)

**22. Default 15 questions**
- Method: `GET`
- URL: `http://localhost:8000/random-questions`

Expected: `200` → 15 AI-generated questions spread across all 5 domains (easy–medium),
also saved to `random-question-collection`.

**23. Custom count**
- Method: `GET`
- URL: `http://localhost:8000/random-questions?count=10`

**24. Out-of-range count → 422**
- URL: `http://localhost:8000/random-questions?count=0` (or `count=99`)

---

## Knowledge Base Seeding

The `knowledge_base` is pre-seeded with **844 questions** across 5 domains (beginner / intermediate / advanced / mixed difficulties). The live API never writes to it.

> **The API works fine with an empty KB.** The LLM always acts as fallback. The KB is an optimisation — repeat requests for the same domain/difficulty are served from KB with no LLM call.

### Adding more questions via another LLM

Prompt files for each domain are in `prompts/domains/`:

```
prompts/domains/
  computer_vision.txt     # paste into ChatGPT/Claude
  machine_learning.txt
  deep_learning.txt
  genai.txt
  ai_fundamentals.txt
```

1. Paste the content of a `.txt` file into any LLM
2. Save the JSON response as the matching `.json` file (e.g. `prompts/domains/genai.json`)
3. Import it:

```bash
python scripts/import_questions.py prompts/domains/genai.json
```

Expected JSON format from the LLM:
```json
{
  "domain": "genai",
  "questions": [
    {
      "difficulty": "beginner",
      "question": "...",
      "options": [
        {"key": "A", "value": "...", "position": 1},
        {"key": "B", "value": "...", "position": 2},
        {"key": "C", "value": "...", "position": 3},
        {"key": "D", "value": "...", "position": 4}
      ],
      "correct_answer": ["B"],
      "explanation": "...",
      "job_relevance": "..."
    }
  ]
}
```

---

## Scheduling `GET /random-questions` with n8n

A scheduled n8n workflow can hit `GET /random-questions` automatically (e.g. to generate
and store a fresh batch every day).

**Workflow:** `Daily Random Questions (midnight IST)` — runs daily at **00:00 Asia/Kolkata**.
It has two nodes: a **Schedule Trigger** → an **HTTP Request** (`GET` the endpoint).

You can create it through the n8n public API (replace the placeholders — never commit the key):

```bash
export N8N_KEY="<your-n8n-public-api-key>"     # keep this out of git
export N8N_BASE="https://<your-n8n-host>"
export API_URL="https://<your-public-api-host>/random-questions"

curl -X POST "$N8N_BASE/api/v1/workflows" \
  -H "X-N8N-API-KEY: $N8N_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Daily Random Questions (midnight IST)",
    "nodes": [
      {
        "parameters": { "rule": { "interval": [
          { "field": "days", "daysInterval": 1, "triggerAtHour": 0, "triggerAtMinute": 0 }
        ] } },
        "id": "node-schedule", "name": "Daily at Midnight IST",
        "type": "n8n-nodes-base.scheduleTrigger", "typeVersion": 1.2, "position": [400, 300]
      },
      {
        "parameters": {
          "method": "GET", "url": "'"$API_URL"'",
          "sendQuery": true,
          "queryParameters": { "parameters": [ { "name": "count", "value": "15" } ] },
          "options": { "timeout": 60000 }
        },
        "id": "node-http", "name": "GET /random-questions",
        "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.2, "position": [640, 300]
      }
    ],
    "connections": {
      "Daily at Midnight IST": { "main": [ [ { "node": "GET /random-questions", "type": "main", "index": 0 } ] ] }
    },
    "settings": { "executionOrder": "v1", "timezone": "Asia/Kolkata" }
  }'

# then activate it (use the id returned above):
curl -X POST "$N8N_BASE/api/v1/workflows/<workflow-id>/activate" \
  -H "X-N8N-API-KEY: $N8N_KEY"
```

Notes:
- `settings.timezone: "Asia/Kolkata"` makes `triggerAtHour: 0` mean midnight **IST**.
- The API must be reachable from n8n's server (a public URL or tunnel — not `localhost`).
- Add auth headers to the HTTP node if/when the endpoint is no longer public.

---

## Project Structure

```
app/
  config.py                      # All env vars (pydantic-settings)
  db.py                          # Motor async client + index setup
  main.py                        # FastAPI app, /health, lifespan
  kb_taxonomy.py                 # 5 domains + aliases (edit here to add domains)
  errors.py                      # Domain exceptions (LLMError, DBError, etc.)
  models/
    enums.py                     # QuizType, Difficulty
    request.py                   # GenerateQuestionsRequest
    response.py                  # GenerateQuestionsResponse
    question_bank_document.py    # Frontend-facing schema
    knowledge_base_document.py   # Internal seed schema (domain-level, no topics)
  routes/
    generate_questions.py        # Thin route — validates, delegates, maps errors
    random_questions.py          # GET /random-questions — AI-generate + save batch
  services/
    llm_provider.py              # Anthropic/OpenAI switch via env var
    embeddings.py                # OpenAI embeddings wrapper (async + sync)
    topic_matcher.py             # Domain matcher — alias → embedding fallback
    kb_retriever.py              # Filter + rank KB docs by domain + similarity
    llm_question_generator.py    # Structured output chain + retry
    random_question_generator.py # All-domains easy-medium generator (/random-questions)
    quiz_generator.py            # 5-step orchestration — all branching logic lives here
  utils/
    transforms.py                # ONLY place mapping correct_answer ↔ correctAnswer
    similarity.py                # Cosine similarity helpers
scripts/
  seed_knowledge_base.py         # Automated offline seeder (resumable, dedup)
  generate_prompts.py            # Generates domain prompt .txt files
  import_questions.py            # Imports LLM-generated JSON into knowledge_base
prompts/
  domains/
    computer_vision.txt          # Ready-to-paste prompts for external LLMs
    machine_learning.txt
    deep_learning.txt
    genai.txt
    ai_fundamentals.txt
tests/
  test_request_validation.py
  test_topic_matcher.py
  test_kb_retriever.py
  test_quiz_generator_flow.py    # Covers all 5 branch paths
```

---

## Running Tests

All 41 tests run fully offline — no live Mongo or OpenAI calls.

```bash
pytest tests/ -v
```

---

## TODOs Before Production

- **Authentication / rate limiting** — none currently. Add API key auth or JWT before any public exposure.
- **`count` semantics** — `compute_response_count` in `services/quiz_generator.py` returns "total satisfying the request" (existing + newly inserted), not "newly inserted only." Confirm with the frontend team before relying on this in production.
- **DAILY / SESSION types** — accepted by the enum but return `501`. Implement when specs are confirmed.
- **Expand KB** — current seed is 844 questions. Use the prompts in `prompts/domains/` to generate more and import via `scripts/import_questions.py`.
- **Embedding cost at scale** — requests that don't resolve via alias fast-path make a live OpenAI embeddings call. Monitor as traffic grows.
