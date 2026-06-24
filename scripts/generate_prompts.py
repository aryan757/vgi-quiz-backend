#!/usr/bin/env python3
"""Generate one prompt file per domain (all topics + all difficulties in one shot).

Output:
  prompts/computer_vision.txt
  prompts/machine_learning.txt
  prompts/deep_learning.txt
  prompts/genai.txt
  prompts/IMPORT_ALL.sh
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.kb_taxonomy import TAXONOMY

ALREADY_DONE = {
    ("LangChain", "beginner"), ("LangChain", "intermediate"), ("LangChain", "advanced"),
    ("RAG", "beginner"),
    ("Transformers", "intermediate"),
}

DIFFICULTIES = ["beginner", "intermediate", "advanced"]

DOMAIN_CONTEXT = {
    "genai": "Generative AI / LLM Systems",
    "machine_learning": "Classical Machine Learning",
    "deep_learning": "Deep Learning",
    "computer_vision": "Computer Vision",
}


def needed_difficulties(topic: str) -> list[str]:
    return [d for d in DIFFICULTIES if (topic, d) not in ALREADY_DONE]


def build_domain_prompt(domain: str, topics: dict) -> str:
    # Build the topic list section
    topic_lines = []
    total_questions = 0
    for topic, subtopics in topics.items():
        diffs = needed_difficulties(topic)
        if not diffs:
            continue
        diffs_str = " + ".join(d.upper() for d in diffs)
        subs = ", ".join(subtopics[:5]) if subtopics else topic
        topic_lines.append(f"  - {topic} [{diffs_str}] — subtopics: {subs}")
        total_questions += len(diffs) * 50

    topics_block = "\n".join(topic_lines)

    # Build the expected JSON structure skeleton
    json_skeleton_parts = []
    for topic, subtopics in topics.items():
        diffs = needed_difficulties(topic)
        if not diffs:
            continue
        batch_lines = ",\n        ".join(
            f'{{"difficulty": "{d}", "questions": [ ...50 questions... ]}}'
            for d in diffs
        )
        json_skeleton_parts.append(
            f'    {{\n'
            f'      "topic": "{topic}",\n'
            f'      "subtopics": {subtopics[:4]},\n'
            f'      "batches": [\n        {batch_lines}\n      ]\n    }}'
        )

    json_skeleton = ",\n".join(json_skeleton_parts)

    return f"""You are generating MCQ questions for a job-readiness quiz platform for AI/ML engineers.

Domain: {DOMAIN_CONTEXT[domain]}

Generate questions for ALL of the following topics and difficulties ({total_questions} questions total):

{topics_block}

---

Difficulty calibration:
- BEGINNER     → fundamental concepts, definitions, basic usage
- INTERMEDIATE → practical application, common patterns, tradeoffs
- ADVANCED     → deep internals, edge cases, system design, optimization

Rules (follow strictly):
1. Exactly 50 questions per topic per difficulty level listed above
2. Every question has exactly 4 options — keys A, B, C, D — positions 1, 2, 3, 4
3. correct_answer is a list with exactly ONE key e.g. ["B"]
4. No LaTeX, no math derivations, no proofs — real job interview level only
5. Spread questions across the subtopics listed for each topic
6. Return ONLY the JSON object — no markdown fences, no explanation text

Each question must follow this structure:
{{
  "question": "...",
  "options": [
    {{"key": "A", "value": "...", "position": 1}},
    {{"key": "B", "value": "...", "position": 2}},
    {{"key": "C", "value": "...", "position": 3}},
    {{"key": "D", "value": "...", "position": 4}}
  ],
  "correct_answer": ["B"],
  "explanation": "1-2 sentence explanation of why the answer is correct.",
  "job_relevance": "Why this is asked in ML/AI engineering interviews."
}}

Return this exact JSON structure:

{{
  "domain": "{domain}",
  "topics": [
{json_skeleton}
  ]
}}

Save the output as: prompts/{domain}.json
Then run: python scripts/import_questions.py prompts/{domain}.json""".strip()


def main():
    prompts_dir = Path("prompts")
    prompts_dir.mkdir(exist_ok=True)

    import_commands = ["#!/bin/bash", "# Run after all domain JSONs are filled\n"]

    for domain, topics in TAXONOMY.items():
        # Skip domain if everything already done
        needed = {t: s for t, s in topics.items() if needed_difficulties(t)}
        if not needed:
            print(f"[SKIP] {domain} — all slots already seeded")
            continue

        prompt = build_domain_prompt(domain, needed)
        txt_path = prompts_dir / f"{domain}.txt"
        txt_path.write_text(prompt)

        slots = sum(len(needed_difficulties(t)) for t in needed)
        questions = slots * 50
        print(f"  {domain}.txt — {len(needed)} topics, {slots} difficulty slots, {questions} questions")

        import_commands.append(f"python scripts/import_questions.py prompts/{domain}.json")

    Path("prompts/IMPORT_ALL.sh").write_text("\n".join(import_commands))
    Path("prompts/IMPORT_ALL.sh").chmod(0o755)

    print(f"\nFiles written to prompts/")
    print(f"Import all with: bash prompts/IMPORT_ALL.sh")


if __name__ == "__main__":
    main()
