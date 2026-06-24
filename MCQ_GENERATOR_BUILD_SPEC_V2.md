# VGI MCQ Generator — Build Specification (v2)

> This document is written to be fed directly into **Claude Code CLI** as the working spec for building this service. It supersedes the earlier draft — the API contract, response shape, and data model below are the confirmed, final versions. Build phase by phase, in order, as laid out in Section 11.

---

## 1. What we're building

A **FastAPI backend** with one primary endpoint — `POST /generate-questions` — that produces job-readiness MCQ (multiple-choice question) content for VGI Skill Lab's curriculum (Computer Vision, Machine Learning, Deep Learning, Generative AI / LLM Systems).

This endpoint does **not** return questions inline. It generates/fetches them, **writes them into MongoDB**, and returns only a confirmation (`success`, `message`, `count`). The frontend team reads the actual question content separately, directly from the `question_bank` collection.

There are **two MongoDB collections with distinct roles** — do not conflate them:

| Collection | Role | Written by |
|---|---|---|
| `knowledge_base` | Rich, pre-seeded reference content per topic/difficulty. The "source of truth" the API draws from when it can. Never read directly by the frontend. | A one-time/periodic **seeding script** (Section 7), run from the terminal — not the live API. |
| `question_bank` | Lean output log of every question this API has ever served, in the exact schema the frontend expects to read. | The **live `/generate-questions` endpoint**, on every call. |

---

## 2. Tech stack

- **Python 3.11+**
- **FastAPI** — API layer
- **LangChain** — LLM orchestration (prompt templates, structured output parsing)
- **MongoDB** (self-hosted, replica set) — both collections above
- **Motor** (async MongoDB driver) for the live request path; **PyMongo** (sync) is fine for the one-off seeding script
- **Pydantic v2** — request/response/document schemas
- **OpenAI Embeddings API** (`text-embedding-3-small`) — used for topic matching, KB retrieval ranking, and seeding-time duplicate detection. This is a **live API dependency on every request that touches the knowledge base** (Section 8/9), not just during seeding — see Section 3 and the cost/latency note in Section 8.1.
- **python-dotenv** / **pydantic-settings** — config

LLM provider must be **configurable** (Anthropic or OpenAI), selected via env var — do not hardcode to one provider. The embeddings provider is fixed to OpenAI regardless of which LLM provider is selected for generation (these are independent choices — you can generate questions with Anthropic's Claude while still using OpenAI purely for embeddings).

---

## 3. Environment configuration

`.env` (and a checked-in `.env.example` with blanks):

```env
# Mongo
DATABASE_URL=mongodb://<user>:<password>@<host>:27017/vgi_skill_lab?authSource=admin&replicaSet=rs0&directConnection=true
DATABASE_NAME=vgi_skill_lab
KNOWLEDGE_BASE_COLLECTION=knowledge_base
QUESTION_BANK_COLLECTION=question_bank

# LLM provider: "anthropic" or "openai" — controls question generation only
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=
LLM_MODEL_NAME=claude-sonnet-4-6

# OpenAI API key — ALWAYS required regardless of LLM_PROVIDER above, because
# embeddings (topic matching + KB retrieval ranking) always use OpenAI (see below).
OPENAI_API_KEY=

# Embeddings — OpenAI, used live on every request that touches the knowledge base
# (topic matching + KB retrieval ranking), and during seeding for duplicate detection.
# Independent of LLM_PROVIDER above — always OpenAI regardless of which provider generates questions.
EMBEDDING_MODEL=text-embedding-3-small

# Seeding targets
SEED_QUESTIONS_PER_TOPIC_PER_DIFFICULTY=100

# App
APP_ENV=development
LOG_LEVEL=INFO
```

**Add `.env` to `.gitignore` in Phase 0, before any other code is written.** The Mongo URI contains live credentials.

---

## 4. MongoDB data model

Database: `vgi_skill_lab`.

### 4.1 `knowledge_base` collection (seed content — internal only)

One document per seeded question. Richer schema than `question_bank`, since this is never exposed directly — it exists purely to ground both Branch A (KB fetch) and Branch B/C (LLM generation, for grounding context) with vetted material.

```json
{
  "_id": "ObjectId",
  "domain": "genai",
  "topic": "Transformers",
  "subtopics": ["self-attention", "encoder-decoder"],
  "difficulty": "intermediate",
  "question": "What is the primary advantage of the Transformer architecture over RNNs?",
  "options": [
    {"key": "A", "value": "Lower memory usage", "position": 1},
    {"key": "B", "value": "Parallel processing of sequences", "position": 2},
    {"key": "C", "value": "Smaller model size", "position": 3},
    {"key": "D", "value": "No need for training data", "position": 4}
  ],
  "correct_answer": ["B"],
  "explanation": "Transformers process tokens in parallel using self-attention instead of sequential processing like RNNs.",
  "job_relevance": "frequently asked in ML/NLP engineer interviews",
  "embedding": [0.0123, -0.0456, "... float vector"],
  "topic_embedding": [0.0123, -0.0456, "... float vector for topic-matching, see Section 8.1"],
  "quality_reviewed": false,
  "created_at": "ISODate"
}
```

Note: the **option/answer shape here matches `question_bank` exactly** (`options` as `{key, value, position}` objects, `correct_answer` as an array of keys) so that copying a `knowledge_base` document into `question_bank` is a near-direct field mapping, not a transform.

Indexes (Phase 1):
- Compound index on `(domain, topic, difficulty)`
- Text index on `topic` and `subtopics` (fallback keyword matching if embedding similarity is inconclusive)

### 4.2 `question_bank` collection (output log — frontend-facing)

Exact schema as specified by your senior. **Do not add extra fields beyond what's shown here** — this is the contract the frontend depends on.

```json
{
  "_id": "685a3d89c87f21a3b7d8d123",
  "type": "CUSTOM",
  "topic": "Transformers",
  "difficulty": "INTERMEDIATE",
  "question": "What is the primary advantage of the Transformer architecture over RNNs?",
  "options": [
    {"key": "A", "value": "Lower memory usage", "position": 1},
    {"key": "B", "value": "Parallel processing of sequences", "position": 2},
    {"key": "C", "value": "Smaller model size", "position": 3},
    {"key": "D", "value": "No need for training data", "position": 4}
  ],
  "correctAnswer": ["B"],
  "explanation": "Transformers process tokens in parallel using self-attention instead of sequential processing like RNNs.",
  "createdAt": "2026-06-24T10:00:00Z",
  "updatedAt": "2026-06-24T10:00:00Z"
}
```

Field naming note (intentional, do not "fix"): `question_bank` uses `correctAnswer` (camelCase) per the confirmed schema, while `knowledge_base` internally uses `correct_answer` (snake_case). Keep this distinction exact — map explicitly between them in code, don't assume they're interchangeable.

Indexes (Phase 1):
- Compound index on `(topic, difficulty)` — this is the index Step 3 of the generation flow (Section 9) queries against to check existing inventory before deciding what to generate/insert.

---

## 5. Knowledge base taxonomy (seed content plan)

Same taxonomy as before — domains: `computer_vision`, `machine_learning`, `deep_learning`, `genai` — calibrated to job-interview level, not research/proof-heavy math. See the detailed topic breakdown below; this is what the seeding script (Section 7) iterates over.

### Domain: `computer_vision`
CNN fundamentals (kernels, filters, padding, stride, pooling), activation functions, batch normalization, dropout, regularization, hidden layer/depth tradeoffs, classic architectures (LeNet, VGG, ResNet, Inception — conceptual), object detection fundamentals (IoU, anchor boxes, NMS), YOLO family (v5/v8/v10/v11, YOLO-World — architecture intuition, training/export/inference workflow), image segmentation (semantic vs instance), practical pipeline (OpenCV preprocessing, annotation workflows, export formats — ONNX/TensorRT/CoreML/OpenVINO, monitoring with MLflow/W&B).

### Domain: `machine_learning`
Core algorithms (Linear/Logistic Regression, Decision Trees, Random Forest, Bagging/Boosting, KNN, SVM, K-Means, DBSCAN, Agglomerative Clustering, Naive Bayes), classification vs regression, loss functions (conceptual), evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix), supervised vs unsupervised, applied interview-level statistics (bias-variance, overfitting/underfitting, EDA-level distribution checks), scikit-learn practical usage (pipelines, train/test split, cross-validation, hyperparameter tuning at API level).

### Domain: `deep_learning`
NN fundamentals (forward pass, backprop intuition — conceptual, no calculus derivations), RNN (sequence modeling, vanishing gradient), LSTM (gates, why it solves RNN's long-term dependency problem), Transformer architecture (self-attention, multi-head attention, positional encoding — at interview depth).

### Domain: `genai`
LangChain (prompt templates, chains, tools, agents, memory/`InMemorySaver`, LCEL), RAG (embeddings, chunking, vector search algorithms like HNSW, reranking, hybrid search), AI agents (ReAct pattern, tool calling, agent loops), MCP (what it is, why it exists, how it differs from a plain tool call), Google ADK (concepts, when used vs LangChain), fine-tuning (full fine-tuning vs LoRA/QLoRA vs prompt-tuning vs instruction-tuning), Hugging Face (Transformers library, model hub, pipelines).

**Claude Code may add a small number of additional well-known, frequently-interviewed topics** within these domains (vector databases, inference optimization/quantization, LLM evaluation) — these appeared in earlier sample payloads and should be supported.

### Seeding volume target
**100 questions per topic, per difficulty level** (beginner / intermediate / advanced) → **300 questions per topic**. Across ~30-40 topics, expect a total seed of roughly **9,000–12,000 questions** in `knowledge_base`. This is a large, deliberately thorough seed — the script (Section 7) must support running in batches (per-domain, per-topic) rather than one single blocking run, given this volume.

---

## 6. API contract

### 6.1 Endpoint

```
POST /generate-questions
```

### 6.2 Request body

```json
{
  "type": "CUSTOM",
  "topic": "Transformers",
  "difficulty": "INTERMEDIATE",
  "description": "Focus on self-attention and encoder-decoder architecture",
  "job_description": "Machine Learning Engineer with NLP experience",
  "question_count": 20
}
```

**Field rules:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | Enum: `CUSTOM`, `DAILY`, `SESSION` | Yes | Single selection only. Only `CUSTOM` behavior is specified (Section 9). `DAILY`/`SESSION` are scaffolded but not implemented — see Section 6.4. |
| `topic` | String | Yes | A topic name, or the literal string `"auto"`. |
| `difficulty` | Enum: `BEGINNER`, `INTERMEDIATE`, `ADVANCED`, `MIXED` | No | Defaults to `MIXED` if omitted — document this default clearly in code. |
| `description` | String | No | Additional generation context for the LLM. |
| `job_description` | String | No | Job description to calibrate question relevance toward. |
| `question_count` | Number | Yes | No fixed enum this time — accept any reasonable positive integer (e.g. validate `1 <= question_count <= 100`, reject outside that range with `422`). |

**Special case — `topic: "auto"`**: treat as "no specific topic constraint" — select topics broadly across the matched domain (if inferable from `description`/`job_description`) or across the full taxonomy if no other signal exists. Implement this as a documented fallback in the topic-matching step (Section 8.1), not a special-cased branch scattered through the route logic.

### 6.3 Response body

**Always this exact shape, regardless of which internal branch was taken:**

```json
{
  "success": true,
  "message": "Questions generated successfully",
  "count": 20
}
```

- `success`: `false` on any failure (see Section 6.4 for error cases) — when `false`, still return this same shape with `message` describing the failure, plus the appropriate non-200 HTTP status.
- `count`: the **total number of questions now satisfying this request in `question_bank`** — i.e. existing matched documents reused (not duplicated) **plus** newly inserted documents this call. This is a deliberate definition choice (see note below) — not strictly "documents inserted this call."

> **Note on `count` semantics — flag this to your senior before relying on it in production**: the source material shows `count: 20` always matching `question_count: 20` exactly, which is consistent whether `count` means "total satisfying" or "newly inserted." This spec defines it as "total satisfying," because that's what Section 9's reuse-existing-matches logic naturally produces. If the intended meaning is strictly "rows newly written to `question_bank` this call," that's a one-line change in `services/quiz_generator.py` — isolate this calculation behind a single clearly-named variable (`total_count` vs `newly_inserted_count`) so it's trivial to swap.

### 6.4 Error / edge cases

- Invalid `type` / `difficulty` enum values → `422`
- `question_count` outside `1..100` → `422`
- `type: "DAILY"` or `type: "SESSION"` requested → `501 Not Implemented`, with a clear message that only `CUSTOM` is currently supported. Do not silently treat them as `CUSTOM`.
- LLM call failure (timeout, API error) after retry → `502`, `success: false`
- Mongo connection/write failure → `503`, `success: false`
- Partial LLM output (couldn't generate the full shortfall, e.g. LLM returned fewer valid questions than needed after retry) → still return `success: true` but with `count` reflecting the actual total achieved, and log a warning server-side. Don't hard-fail a partial result.

---

## 7. Knowledge base seeding (separate batch script — run by you, not the live API)

`scripts/seed_knowledge_base.py` — populates `knowledge_base` only. This is **never called by the live API**; you run it yourself from the terminal, ahead of time, to build up reference content.

- Taxonomy defined in `app/kb_taxonomy.py` (structured dict: domain → topic → subtopics), separate from script logic so it's editable without touching code.
- For each `(domain, topic, difficulty)` combination, calls the LLM with a tightly-scoped prompt: *"Generate {N} job-interview-calibrated MCQs on {topic} within {domain} at {difficulty} level. No research-level math. Include a clear explanation per question. Calibrate for someone preparing for an ML/AI engineering job interview."*
- Target: **100 questions per topic per difficulty** (`SEED_QUESTIONS_PER_TOPIC_PER_DIFFICULTY` env var) — given the resulting volume (~9,000-12,000 total), the script must support:
  - `--domain` filter (run one domain at a time)
  - `--topic` filter (run/top-up a single topic)
  - `--difficulty` filter
  - Resumability: track progress so a long run can be restarted without re-generating from scratch (e.g. a local progress log file, or checking existing counts per `(topic, difficulty)` in Mongo before generating more)
- Computes two embeddings per question: a content embedding (`question + explanation`, for grounding-context retrieval in Section 9.2) and a `topic_embedding` (topic + subtopics text only, for the topic-matching step in Section 8.1).
- Before inserting, checks for near-duplicates (cosine similarity > 0.95 against existing questions in the same topic/difficulty) to avoid bloating the collection across repeated runs.
- Inserts with `quality_reviewed: false` by default — given the volume here, **spot-check a sample per domain rather than reviewing all ~9-12k individually**, but don't skip review entirely.
- Prints a per-domain/per-topic summary on completion (generated count, skipped-as-duplicate count, failures).

Example usage:
```bash
python scripts/seed_knowledge_base.py --domain computer_vision
python scripts/seed_knowledge_base.py --topic "Transformers" --difficulty intermediate
```

Given the scale (100/topic/difficulty), **run this domain-by-domain**, not as one all-at-once invocation — budget real time for this and spot-check output before moving to the next domain.

---

## 8. Topic matching (deciding when the KB applies)

This is the logic that answers: *"does this request's `topic` correspond to something we have good coverage for in `knowledge_base`?"* — used in Step 2 of the generation flow (Section 9).

### 8.1 Matching approach

```python
class TopicMatcher:
    async def match(self, topic: str, description: str | None, job_description: str | None) -> MatchResult:
        ...
```

**Cost/latency note**: every call into this method that reaches the embedding step makes a live call to the OpenAI Embeddings API (`text-embedding-3-small`). This is a deliberate, confirmed choice (simpler dependency footprint over the cost/latency of a local model) — but it does mean topic matching is no longer a "free" in-process step. Mitigate this with the fast-path below: most real requests should resolve via cheap string matching and never reach the embedding call at all.

- If `topic == "auto"`: no strict match needed — return a `MatchResult` indicating "broad/unconstrained," letting downstream logic pull from across the taxonomy (or a domain inferred from `description`/`job_description` if either is present, even though presence of those fields will route to LLM-only generation per Section 9 — inference can still help pick *which* topics to feature in the LLM prompt).
- **Fast path first**: run a cheap exact/fuzzy string check (case-insensitive exact match, or simple containment) against the set of known canonical topic names before calling the embeddings API at all — most real requests will exact-match a known topic name, no need to spend an API call on the common case.
- **Fallback to embeddings**: only when the fast path doesn't resolve, embed the incoming `topic` string via the OpenAI Embeddings API, and compare against the `topic_embedding` field on `knowledge_base` documents (grouped/deduplicated by distinct topic values first — don't compare against every document, compare against the set of unique topics; this also means embedding each unique topic name only once and caching it in-process, not recomputing on every request).
- Define a similarity threshold (start at cosine ≥ 0.80, make it configurable) — above it, treat as a **confident match** to that existing topic (use the matched topic's canonical name internally from this point on, even if the input had different casing/wording — e.g. `"transformer models"` → matches `"Transformers"`). Below it, treat as **no confident match** → routes to full LLM generation (Branch C, Section 9).

---

## 9. The full generation flow (the core logic — `POST /generate-questions`)

Implement this as a single orchestrating service, `services/quiz_generator.py`, called by the route. Do not inline this logic into the route handler.

```python
async def generate_questions(request: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
    ...
```

### Step 1 — Branch on `description` / `job_description`
- Both empty/omitted → go to Step 2.
- Either present → skip to Step 4 (full LLM generation), using `description`/`job_description` as primary steering context alongside `topic`/`difficulty`.

### Step 2 — Topic match check (only reached when no description/job_description)
Call `TopicMatcher.match()` (Section 8.1).
- No confident match → go to Step 4 (full LLM generation), context = `topic`/`difficulty` only.
- Confident match → go to Step 3.

### Step 3 — Check existing inventory in `question_bank`
Query `question_bank` for documents matching `(topic, difficulty)` (using the matched canonical topic name from Step 2).

- `existing_count = len(matches)`
- If `existing_count >= question_count`: **no new generation needed.** `count = question_count` (capped at what was asked for — don't report more than requested even if more exist). No new inserts.
- If `0 < existing_count < question_count`: shortfall = `question_count - existing_count`. Pull `shortfall` questions from `knowledge_base` (matching topic/difficulty, ranked by relevance — reuse the `KnowledgeBaseRetriever` pattern: filter then rank by content embedding similarity against a synthetic query built from topic + any available context), transform each into the `question_bank` schema (map `correct_answer` → `correctAnswer`, generate fresh `_id`/`createdAt`/`updatedAt`), and insert them. `count = existing_count + shortfall`.
- If `existing_count == 0`: pull all `question_count` from `knowledge_base` the same way, insert all as new `question_bank` documents. `count = question_count` (or fewer, if `knowledge_base` itself doesn't have enough for that topic/difficulty — in which case, top up the remainder via LLM, same as Step 4's generation logic, rather than silently under-delivering).

### Step 4 — Full LLM generation
Reached when: description/job_description present, OR no confident topic match, OR knowledge_base itself can't fully satisfy Step 3.

- Build a LangChain prompt including: `topic`, `difficulty`, `description` (if present), `job_description` (if present), and — **even in this branch** — a handful of representative grounding examples pulled from `knowledge_base` for the closest-matching topic if one exists (keeps LLM output stylistically consistent and accurate even when not doing a strict KB-fetch).
- Use a LangChain structured output chain (Pydantic output parser or `.with_structured_output()`) constrained to the `question_bank` document shape directly — don't regex-parse free text.
- Validate parsed output; on validation failure, retry once with an error-correction prompt (`OutputFixingParser` pattern), then fail clearly (502) if still invalid.
- Insert all newly generated questions into `question_bank` with fresh `_id`/timestamps. `count` = number successfully generated and inserted (may be less than `question_count` on partial LLM failure — see Section 6.4).

### Step 5 — Respond
Return `{success: true, message: "Questions generated successfully", count: <as computed above>}`.

Isolate the `count` computation behind one clearly named function so the semantics (Section 6.3's note) are easy to audit/change later:
```python
def compute_response_count(existing_reused: int, newly_inserted: int) -> int:
    # PLACEHOLDER SEMANTICS: total satisfying the request, not strictly "newly written."
    # Change here only if "newly inserted only" is confirmed as the intended meaning.
    return existing_reused + newly_inserted
```

---

## 10. Project structure

```
vgi-mcq-generator/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── app/
│   ├── main.py
│   ├── config.py                      # pydantic-settings
│   ├── db.py                          # motor client + db handle, exposes both collections
│   ├── kb_taxonomy.py                 # domain → topic → subtopics (Section 5)
│   ├── models/
│   │   ├── request.py                 # GenerateQuestionsRequest (Section 6.2)
│   │   ├── response.py                # GenerateQuestionsResponse (Section 6.3)
│   │   ├── question_bank_document.py  # Section 4.2 schema
│   │   └── knowledge_base_document.py # Section 4.1 schema
│   ├── routes/
│   │   └── generate_questions.py      # POST /generate-questions — thin, delegates to services
│   ├── services/
│   │   ├── llm_provider.py            # provider-agnostic Anthropic/OpenAI interface
│   │   ├── embeddings.py              # OpenAI Embeddings API wrapper (text-embedding-3-small)
│   │   ├── topic_matcher.py           # Section 8
│   │   ├── kb_retriever.py            # filter + rank against knowledge_base (used in Step 3 & grounding)
│   │   ├── llm_question_generator.py  # Step 4 — structured generation chain
│   │   └── quiz_generator.py          # Section 9 orchestration — the main entry point
│   └── utils/
│       └── transforms.py              # knowledge_base doc → question_bank doc field mapping
├── scripts/
│   └── seed_knowledge_base.py         # Section 7
└── tests/
    ├── test_request_validation.py
    ├── test_topic_matcher.py
    ├── test_kb_retriever.py
    └── test_quiz_generator_flow.py    # cover all branches: A (full reuse), A (shortfall), B, C
```

---

## 11. Build plan (in order)

**Phase 0 — Scaffold**
Folder structure, `requirements.txt`, `.env.example`, `.gitignore` (commit this before anything else — the Mongo URI has live credentials). Basic FastAPI app, `GET /health`. Confirm Mongo connectivity to both intended collections.

**Phase 1 — Data models & DB layer**
Pydantic models for both collections (Sections 4.1/4.2) and the request/response schemas (Section 6). Motor setup in `db.py`. Create indexes listed in Section 4.

**Phase 2 — LLM provider abstraction**
`services/llm_provider.py` — single interface, Anthropic/OpenAI switch via env var. Used by both the seeding script and the runtime generator.

**Phase 3 — Topic matcher & embeddings**
`services/embeddings.py` (thin wrapper around the OpenAI Embeddings API) and `services/topic_matcher.py` (Section 8). Implement the in-process topic-name embedding cache here (Section 8.1) to avoid redundant API calls. Can be unit tested with mocked `knowledge_base` data and a mocked embeddings client before real seed content or API credits are needed.

**Phase 4 — Knowledge base seeding**
Build `kb_taxonomy.py` and `scripts/seed_knowledge_base.py` (Section 7). **Run domain-by-domain.** Given the 100-per-topic-per-difficulty target, do a small test run first (e.g. `--topic "Transformers" --difficulty intermediate` only) and manually review quality before scaling to full domains.

**Phase 5 — KB retriever & transforms**
`services/kb_retriever.py` and `utils/transforms.py` — the filter/rank logic against `knowledge_base`, and the field-mapping helper that converts a `knowledge_base` document into a valid `question_bank` document.

**Phase 6 — LLM question generator (Step 4)**
`services/llm_question_generator.py` — structured output chain with grounding context injection and retry-on-validation-failure.

**Phase 7 — Orchestration**
`services/quiz_generator.py` (Section 9) — wire up the full 5-step branch logic, `question_bank` inventory checks, insert logic, and `count` computation. This is the most important phase to test thoroughly — cover all branch paths explicitly.

**Phase 8 — Route**
`routes/generate_questions.py` — thin route, request validation, calls `quiz_generator`, returns the fixed response shape. Implement the `DAILY`/`SESSION` → `501` stub here.

**Phase 9 — Error handling**
Implement all cases from Section 6.4.

**Phase 10 — Tests & polish**
Cover request validation, topic matching, all five branch paths in the orchestration flow, and error cases. `README.md` with run instructions, seeding script usage, and an example `curl` request/response pair.

---

## 12. Open implementation notes for Claude Code

- Keep `routes/generate_questions.py` thin — validation + delegation only. All branching logic lives in `services/quiz_generator.py`, fully unit-testable without spinning up FastAPI.
- `utils/transforms.py` is small but important — it's the only place that should know about the field-name differences between `knowledge_base` (`correct_answer`, snake_case) and `question_bank` (`correctAnswer`, camelCase). Don't duplicate this mapping elsewhere.
- The `count` semantics note in Section 6.3 is a real open question, not a settled detail — make sure whoever reviews this code sees that comment before this ships anywhere beyond a demo.
- `DAILY` and `SESSION` are accepted by the enum but explicitly unimplemented (`501`) — do not infer or invent behavior for them.
- **Embeddings are OpenAI-only, by deliberate choice, even when `LLM_PROVIDER=anthropic`.** This means `OPENAI_API_KEY` is always required, and every request that reaches the embedding fallback in topic matching (Section 8.1) or does KB retrieval ranking makes a live OpenAI API call. This was chosen for dependency simplicity over the cost/latency of a local model — make sure the fast-path string matching in Section 8.1 is implemented and working before assuming this is cheap at scale; if request volume grows significantly, revisit this tradeoff.
- No authentication/rate-limiting in this pass — note as a clear TODO in the README.
- Given the ~9,000–12,000 question seeding volume, log progress verbosely in the seeding script (per-topic counters, running totals) so a long-running batch job's progress is visible and resumable, not a silent black box.
