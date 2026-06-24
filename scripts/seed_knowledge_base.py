#!/usr/bin/env python3
"""Seed the knowledge_base collection with job-interview-calibrated MCQs.

Section 7 of the spec — run this from the terminal, NOT the live API.

Usage examples:
    python scripts/seed_knowledge_base.py --domain computer_vision
    python scripts/seed_knowledge_base.py --topic "Transformers" --difficulty intermediate
    python scripts/seed_knowledge_base.py --domain genai --difficulty advanced

Resumable: tracks per-(topic, difficulty) progress in .seed_progress.json so a
long-running job can be restarted without regenerating from scratch.

Dedup: checks cosine similarity of new questions' content embeddings against existing
KB docs for the same (topic, difficulty) before inserting. Skips near-duplicates
(threshold: SEED_DUPLICATE_THRESHOLD env var, default 0.95).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make sure `app/` is importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.config import get_settings
from app.kb_taxonomy import TAXONOMY
from app.models.knowledge_base_document import KnowledgeBaseDocument
from app.models.question_bank_document import Option
from app.services.embeddings import embed_sync
from app.utils.similarity import cosine_similarity

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_kb")

PROGRESS_FILE = Path(__file__).parent / ".seed_progress.json"
DIFFICULTIES = ["beginner", "intermediate", "advanced"]
TARGET_PER_SLOT = settings.seed_questions_per_topic_per_difficulty
DEDUP_THRESHOLD = settings.seed_duplicate_threshold
BATCH_SIZE = 10  # questions per LLM call — keeps individual calls manageable


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def progress_key(domain: str, topic: str, difficulty: str) -> str:
    return f"{domain}|{topic}|{difficulty}"


# ---------------------------------------------------------------------------
# Mongo helpers (sync via PyMongo)
# ---------------------------------------------------------------------------

def get_pymongo_collection():
    from pymongo import MongoClient

    client = MongoClient(settings.database_url, serverSelectionTimeoutMS=5000)
    db = client[settings.database_name]
    return db[settings.knowledge_base_collection]


# ---------------------------------------------------------------------------
# LLM call (sync wrapper for the seeding context)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert technical interviewer creating multiple-choice questions (MCQs) to "
    "assess job-readiness for AI/ML engineering roles. Questions must be calibrated to real "
    "job interviews — NOT research-level math or proofs. Every question must have exactly "
    "4 options (A, B, C, D) with one correct answer and a concise explanation."
)


def generate_questions_sync(
    domain: str, topic: str, subtopics: list[str], difficulty: str, n: int
) -> list[dict]:
    """Call the LLM synchronously and return a list of raw question dicts."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    from app.services.llm_provider import get_chat_model

    class _Option(BaseModel):
        key: str
        value: str
        position: int

    class _Question(BaseModel):
        question: str
        options: list[_Option]
        correct_answer: list[str] = Field(description="List of correct option keys e.g. ['B']")
        explanation: str
        job_relevance: str = Field(default="", description="Why this is asked in ML interviews")

    class _QuestionSet(BaseModel):
        questions: list[_Question]

    subtopics_str = ", ".join(subtopics) if subtopics else topic
    prompt = (
        f"Generate {n} job-interview-calibrated MCQs on the topic: \"{topic}\" "
        f"(domain: {domain}, subtopics: {subtopics_str}) at {difficulty.upper()} level.\n"
        "Rules:\n"
        "- No research-level math or proofs\n"
        "- Each question must have exactly 4 options (keys A, B, C, D; positions 1-4)\n"
        "- Exactly one correct_answer per question (a list with one key)\n"
        "- Include a clear explanation and a brief job_relevance note\n"
        "- Calibrate for someone preparing for an ML/AI engineering interview"
    )

    try:
        chat_model = get_chat_model(settings, temperature=0.6)
        structured = chat_model.with_structured_output(_QuestionSet)
        result = structured.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        if not result or not result.questions:
            return []
        return [q.model_dump() for q in result.questions]
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM call failed for %s/%s/%s: %s", domain, topic, difficulty, exc)
        return []


# ---------------------------------------------------------------------------
# Embedding & dedup helpers
# ---------------------------------------------------------------------------

def content_text(q: dict) -> str:
    return f"{q.get('question', '')} {q.get('explanation', '')}".strip()


def topic_text(topic: str, subtopics: list[str]) -> str:
    return f"{topic} {' '.join(subtopics)}".strip()


def load_existing_embeddings(col, topic: str, difficulty: str) -> list[list[float]]:
    """Load content embeddings of existing KB docs for dedup comparison."""
    docs = list(col.find(
        {"topic": topic, "difficulty": difficulty, "embedding": {"$exists": True, "$ne": []}},
        {"embedding": 1},
    ))
    return [d["embedding"] for d in docs]


def is_duplicate(new_emb: list[float], existing_embs: list[list[float]], threshold: float) -> bool:
    return any(cosine_similarity(new_emb, e) >= threshold for e in existing_embs)


# ---------------------------------------------------------------------------
# Core seeding loop
# ---------------------------------------------------------------------------

def seed_slot(
    col,
    domain: str,
    topic: str,
    subtopics: list[str],
    difficulty: str,
    progress: dict,
) -> tuple[int, int, int]:
    """Seed one (topic, difficulty) slot. Returns (generated, inserted, skipped_as_dup)."""
    key = progress_key(domain, topic, difficulty)
    existing_count = col.count_documents({"topic": topic, "difficulty": difficulty})
    already_done = progress.get(key, 0)

    if existing_count >= TARGET_PER_SLOT:
        logger.info("  [SKIP] %s/%s already has %d >= target %d", topic, difficulty, existing_count, TARGET_PER_SLOT)
        progress[key] = existing_count
        save_progress(progress)
        return 0, 0, 0

    needed = TARGET_PER_SLOT - existing_count
    logger.info(
        "  [SEED] %s / %s / %s — need %d more (have %d, target %d)",
        topic, difficulty, domain, needed, existing_count, TARGET_PER_SLOT,
    )

    existing_embs = load_existing_embeddings(col, topic, difficulty)
    total_generated = 0
    total_inserted = 0
    total_skipped = 0

    while total_inserted < needed:
        batch_want = min(BATCH_SIZE, needed - total_inserted)
        raw_qs = generate_questions_sync(domain, topic, subtopics, difficulty, batch_want)
        if not raw_qs:
            logger.warning("    LLM returned no questions; stopping this slot.")
            break

        total_generated += len(raw_qs)

        # Embed content + topic text for the whole batch at once (one API call).
        content_texts = [content_text(q) for q in raw_qs]
        topic_t = topic_text(topic, subtopics)
        try:
            content_embs = embed_sync(content_texts, settings)
            topic_embs = embed_sync([topic_t] * len(raw_qs), settings)
        except Exception as exc:  # noqa: BLE001
            logger.error("    Embedding failed: %s. Inserting without embeddings.", exc)
            content_embs = [[] for _ in raw_qs]
            topic_embs = [[] for _ in raw_qs]

        docs_to_insert = []
        for q, c_emb, t_emb in zip(raw_qs, content_embs, topic_embs):
            if c_emb and is_duplicate(c_emb, existing_embs, DEDUP_THRESHOLD):
                total_skipped += 1
                continue
            options = [Option(key=o["key"], value=o["value"], position=o["position"]) for o in q.get("options", [])]
            doc = KnowledgeBaseDocument(
                domain=domain,
                topic=topic,
                subtopics=subtopics,
                difficulty=difficulty,
                question=q.get("question", ""),
                options=options,
                correct_answer=list(q.get("correct_answer", [])),
                explanation=q.get("explanation", ""),
                job_relevance=q.get("job_relevance", ""),
                embedding=c_emb,
                topic_embedding=t_emb,
                quality_reviewed=False,
            )
            docs_to_insert.append(doc.to_mongo())
            if c_emb:
                existing_embs.append(c_emb)  # add to local dedup set immediately

        if docs_to_insert:
            col.insert_many(docs_to_insert)
            total_inserted += len(docs_to_insert)
            logger.info(
                "    Inserted %d  (skipped %d as dup, total_inserted=%d/%d)",
                len(docs_to_insert), total_skipped, total_inserted, needed,
            )

        # Avoid hammering the LLM/embeddings API back-to-back.
        time.sleep(0.5)

    progress[key] = existing_count + total_inserted
    save_progress(progress)
    return total_generated, total_inserted, total_skipped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed knowledge_base with MCQs.")
    parser.add_argument("--domain", help="Only seed this domain (e.g. computer_vision)")
    parser.add_argument("--topic", help="Only seed this topic (e.g. 'Transformers')")
    parser.add_argument(
        "--difficulty",
        choices=DIFFICULTIES,
        help="Only seed this difficulty level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without calling LLM or writing to Mongo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    col = get_pymongo_collection()
    progress = load_progress()

    # Build the work plan from taxonomy, filtered by CLI args.
    plan: list[tuple[str, str, list[str], str]] = []
    for domain, topics in TAXONOMY.items():
        if args.domain and domain != args.domain:
            continue
        for topic, subtopics in topics.items():
            if args.topic and topic.lower() != args.topic.lower():
                continue
            for diff in DIFFICULTIES:
                if args.difficulty and diff != args.difficulty:
                    continue
                plan.append((domain, topic, subtopics, diff))

    if not plan:
        logger.warning("No matching slots found. Check --domain / --topic / --difficulty values.")
        return

    logger.info("Seeding plan: %d slot(s), target %d questions each.", len(plan), TARGET_PER_SLOT)

    if args.dry_run:
        for domain, topic, _, diff in plan:
            existing = col.count_documents({"topic": topic, "difficulty": diff})
            print(f"  {domain}/{topic}/{diff}: have {existing}, need {TARGET_PER_SLOT - existing} more")
        return

    totals: dict[str, Any] = {"generated": 0, "inserted": 0, "skipped": 0}
    domain_stats: dict[str, dict] = {}

    for domain, topic, subtopics, diff in plan:
        if domain not in domain_stats:
            domain_stats[domain] = {"generated": 0, "inserted": 0, "skipped": 0}
        gen, ins, dup = seed_slot(col, domain, topic, subtopics, diff, progress)
        domain_stats[domain]["generated"] += gen
        domain_stats[domain]["inserted"] += ins
        domain_stats[domain]["skipped"] += dup
        totals["generated"] += gen
        totals["inserted"] += ins
        totals["skipped"] += dup

    # Summary
    print("\n" + "=" * 60)
    print("SEEDING COMPLETE")
    print("=" * 60)
    for domain, stats in domain_stats.items():
        print(f"  {domain}: generated={stats['generated']}, inserted={stats['inserted']}, dup_skipped={stats['skipped']}")
    print(f"\n  TOTAL: generated={totals['generated']}, inserted={totals['inserted']}, dup_skipped={totals['skipped']}")


if __name__ == "__main__":
    main()
