#!/usr/bin/env python3
"""Import questions from a JSON file into knowledge_base.

Supports three JSON formats (all produced by the domain prompts):
  - domain-level : { "domain":..., "topics": [{ "topic":..., "subtopics":..., "batches":[...] }] }
  - topic-level  : { "domain":..., "topic":..., "subtopics":..., "batches":[...] }
  - single       : { "domain":..., "topic":..., "difficulty":..., "questions":[...] }

Usage:
    python scripts/import_questions.py prompts/computer_vision.json
    python scripts/import_questions.py prompts/genai.json --skip-embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.config import get_settings
from app.models.knowledge_base_document import KnowledgeBaseDocument
from app.models.question_bank_document import Option
from app.utils.similarity import cosine_similarity

settings = get_settings()


def embed_texts(texts: list[str]) -> list[list[float]]:
    from app.services.embeddings import embed_sync
    return embed_sync(texts, settings)


def load_existing_embeddings(col, topic: str, difficulty: str) -> list[list[float]]:
    docs = list(col.find(
        {"topic": topic, "difficulty": difficulty, "embedding": {"$exists": True, "$ne": []}},
        {"embedding": 1},
    ))
    return [d["embedding"] for d in docs]


def import_batch(col, domain, topic, subtopics, difficulty, questions, skip_embeddings):
    """Import one (topic, difficulty) batch. Returns (inserted, dup_skipped, invalid_skipped)."""
    existing_count = col.count_documents({"topic": topic, "difficulty": difficulty})
    print(f"    [{difficulty}] in KB: {existing_count}, incoming: {len(questions)}")

    existing_embs = load_existing_embeddings(col, topic, difficulty) if not skip_embeddings else []
    topic_text = f"{topic} {' '.join(subtopics)}".strip()
    docs_to_insert = []
    inserted = dup_skipped = invalid_skipped = 0

    for i, q in enumerate(questions):
        if not q.get("question") or not q.get("options") or not q.get("correct_answer"):
            print(f"      [SKIP] Q{i+1}: missing required fields")
            invalid_skipped += 1
            continue
        if len(q["options"]) != 4:
            print(f"      [SKIP] Q{i+1}: must have 4 options, got {len(q['options'])}")
            invalid_skipped += 1
            continue

        content_text = f"{q['question']} {q.get('explanation', '')}".strip()
        c_emb, t_emb = [], []
        if not skip_embeddings:
            try:
                [c_emb, t_emb] = embed_texts([content_text, topic_text])
                if c_emb and any(cosine_similarity(c_emb, e) >= 0.95 for e in existing_embs):
                    dup_skipped += 1
                    continue
                if c_emb:
                    existing_embs.append(c_emb)
            except Exception as exc:
                print(f"      [WARN] Embedding failed for Q{i+1}: {exc}. Inserting without.")

        options = [Option(key=o["key"], value=o["value"], position=o["position"]) for o in q["options"]]
        docs_to_insert.append(KnowledgeBaseDocument(
            domain=domain, topic=topic, subtopics=subtopics, difficulty=difficulty,
            question=q["question"], options=options,
            correct_answer=list(q["correct_answer"]),
            explanation=q.get("explanation", ""),
            job_relevance=q.get("job_relevance", ""),
            embedding=c_emb, topic_embedding=t_emb, quality_reviewed=False,
        ).to_mongo())

    if docs_to_insert:
        col.insert_many(docs_to_insert)
        inserted = len(docs_to_insert)

    print(f"      → inserted={inserted}, dup_skipped={dup_skipped}, invalid_skipped={invalid_skipped}")
    print(f"      → total in KB for {topic}/{difficulty}: {existing_count + inserted}")
    return inserted, dup_skipped, invalid_skipped


def main():
    parser = argparse.ArgumentParser(description="Import questions JSON into knowledge_base.")
    parser.add_argument("file", help="Path to the questions JSON file")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Insert without computing embeddings (faster, but disables KB ranking)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    data = json.loads(path.read_text())
    domain = data["domain"]

    # Normalise all three formats into a flat list of (topic, subtopics, batches)
    if "topics" in data:
        entries = [(t["topic"], t.get("subtopics", []), t["batches"]) for t in data["topics"]]
    elif "batches" in data:
        entries = [(data["topic"], data.get("subtopics", []), data["batches"])]
    else:
        entries = [(
            data["topic"], data.get("subtopics", []),
            [{"difficulty": data["difficulty"], "questions": data["questions"]}],
        )]

    from pymongo import MongoClient
    col = MongoClient(settings.database_url, serverSelectionTimeoutMS=5000)[
        settings.database_name][settings.knowledge_base_collection]

    print(f"\nImporting domain: {domain} — {len(entries)} topic(s)\n{'='*55}")
    grand_inserted = grand_dup = grand_invalid = 0

    for topic, subtopics, batches in entries:
        print(f"\n  {topic}")
        for batch in batches:
            ins, dup, inv = import_batch(
                col, domain, topic, subtopics,
                batch["difficulty"].lower(), batch["questions"],
                args.skip_embeddings,
            )
            grand_inserted += ins
            grand_dup += dup
            grand_invalid += inv

    print(f"\n{'='*55}")
    print(f"GRAND TOTAL — inserted={grand_inserted}, dup_skipped={grand_dup}, invalid_skipped={grand_invalid}")


if __name__ == "__main__":
    main()
