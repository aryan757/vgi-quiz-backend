#!/usr/bin/env python3
"""Import domain questions JSON into knowledge_base.

Expected JSON format (output from the domain prompts):
{
  "domain": "computer_vision",
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

Usage:
    python scripts/import_questions.py prompts/domains/computer_vision.json
    python scripts/import_questions.py prompts/domains/genai.json --skip-embeddings
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
from app.kb_taxonomy import VALID_DOMAINS
from app.models.knowledge_base_document import KnowledgeBaseDocument
from app.models.question_bank_document import Option
from app.utils.similarity import cosine_similarity

settings = get_settings()


def embed_texts(texts: list[str]) -> list[list[float]]:
    from app.services.embeddings import embed_sync
    return embed_sync(texts, settings)


def load_existing_embeddings(col, domain: str, difficulty: str) -> list[list[float]]:
    docs = list(col.find(
        {"domain": domain, "difficulty": difficulty,
         "embedding": {"$exists": True, "$ne": []}},
        {"embedding": 1},
    ))
    return [d["embedding"] for d in docs]


def main():
    parser = argparse.ArgumentParser(description="Import domain questions into knowledge_base.")
    parser.add_argument("file", help="Path to the questions JSON file")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Insert without computing embeddings (faster, disables KB ranking)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    data = json.loads(path.read_text())
    domain = data["domain"]
    questions = data["questions"]

    print(f"\nImporting domain: {domain}")
    print(f"Total questions in file: {len(questions)}")

    # Count per difficulty
    from collections import Counter
    diff_counts = Counter(q.get("difficulty", "unknown") for q in questions)
    for diff, cnt in sorted(diff_counts.items()):
        print(f"  {diff}: {cnt}")

    if domain not in VALID_DOMAINS:
        print(f"WARNING: '{domain}' is not in VALID_DOMAINS {VALID_DOMAINS}")
        print("Proceeding anyway...")

    from pymongo import MongoClient
    col = MongoClient(settings.database_url, serverSelectionTimeoutMS=5000)[
        settings.database_name][settings.knowledge_base_collection]

    existing_total = col.count_documents({"domain": domain})
    print(f"Already in KB for domain '{domain}': {existing_total}")

    # Pre-compute domain embedding once for all docs
    domain_emb: list[float] = []
    if not args.skip_embeddings:
        try:
            [domain_emb] = embed_texts([domain])
            print(f"Domain embedding computed ({len(domain_emb)} dims)")
        except Exception as exc:
            print(f"WARNING: Domain embedding failed: {exc}. Inserting without.")

    # Group by difficulty so we can load existing embeddings per group
    from collections import defaultdict
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_difficulty[q.get("difficulty", "beginner").lower()].append(q)

    grand_inserted = grand_dup = grand_invalid = 0

    for difficulty, qs in by_difficulty.items():
        print(f"\n  [{difficulty}] incoming: {len(qs)}, "
              f"in KB: {col.count_documents({'domain': domain, 'difficulty': difficulty})}")

        existing_embs = load_existing_embeddings(col, domain, difficulty) \
            if not args.skip_embeddings else []
        docs_to_insert = []
        inserted = dup_skipped = invalid_skipped = 0

        for i, q in enumerate(qs):
            if not q.get("question") or not q.get("options") or not q.get("correct_answer"):
                print(f"    [SKIP] Q{i+1}: missing required fields")
                invalid_skipped += 1
                continue
            if len(q["options"]) != 4:
                print(f"    [SKIP] Q{i+1}: must have 4 options, got {len(q['options'])}")
                invalid_skipped += 1
                continue

            c_emb: list[float] = []
            if not args.skip_embeddings:
                content_text = f"{q['question']} {q.get('explanation', '')}".strip()
                try:
                    [c_emb] = embed_texts([content_text])
                    if c_emb and any(cosine_similarity(c_emb, e) >= 0.95 for e in existing_embs):
                        dup_skipped += 1
                        continue
                    if c_emb:
                        existing_embs.append(c_emb)
                except Exception as exc:
                    print(f"    [WARN] Embedding Q{i+1} failed: {exc}. Inserting without.")

            options = [Option(key=o["key"], value=o["value"], position=o["position"])
                       for o in q["options"]]
            docs_to_insert.append(KnowledgeBaseDocument(
                domain=domain,
                difficulty=difficulty,
                question=q["question"],
                options=options,
                correct_answer=list(q["correct_answer"]),
                explanation=q.get("explanation", ""),
                job_relevance=q.get("job_relevance", ""),
                embedding=c_emb,
                domain_embedding=domain_emb,
                quality_reviewed=False,
            ).to_mongo())

        if docs_to_insert:
            col.insert_many(docs_to_insert)
            inserted = len(docs_to_insert)

        print(f"    → inserted={inserted}, dup_skipped={dup_skipped}, invalid_skipped={invalid_skipped}")
        grand_inserted += inserted
        grand_dup += dup_skipped
        grand_invalid += invalid_skipped

    print(f"\n{'='*55}")
    print(f"DONE — inserted={grand_inserted}, dup_skipped={grand_dup}, invalid_skipped={grand_invalid}")
    print(f"Total in KB for '{domain}': {col.count_documents({'domain': domain})}")


if __name__ == "__main__":
    main()
