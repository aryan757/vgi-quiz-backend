"""Domain registry — the 5 known domains and their aliases/keywords.

No topics or subtopics here. The knowledge base stores questions at the domain level only.
This file is the single source of truth for:
  - Valid domain names
  - Aliases used in fast-path domain matching (e.g. "vision" → "computer_vision")
  - Scope descriptions used in LLM generation prompts
"""

from __future__ import annotations

DOMAINS: dict[str, dict] = {
    "computer_vision": {
        "aliases": ["cv", "vision", "computer vision", "image processing", "object detection",
                    "yolo", "cnn", "convolutional"],
        "description": "Computer Vision — CNNs, object detection, YOLO, segmentation, OpenCV pipelines",
    },
    "machine_learning": {
        "aliases": ["ml", "classical ml", "sklearn", "scikit", "random forest", "svm",
                    "regression", "classification", "clustering"],
        "description": "Machine Learning — regression, trees, ensembles, clustering, evaluation metrics",
    },
    "deep_learning": {
        "aliases": ["dl", "neural network", "neural networks", "lstm", "rnn", "transformer",
                    "backprop", "deep neural"],
        "description": "Deep Learning — neural networks, RNN, LSTM, Transformers",
    },
    "genai": {
        "aliases": ["gen ai", "generative ai", "llm", "large language model", "langchain",
                    "rag", "agents", "fine tuning", "fine-tuning", "gpt", "claude"],
        "description": "Generative AI — LangChain, RAG, agents, fine-tuning, vector DBs, LLM evaluation",
    },
    "ai_fundamentals": {
        "aliases": ["ai", "artificial intelligence", "ml basics", "ai basics", "fundamentals",
                    "mlops", "responsible ai", "ml lifecycle", "model deployment"],
        "description": "AI Fundamentals — ML lifecycle, MLOps, responsible AI, production ML",
    },
}

VALID_DOMAINS = list(DOMAINS.keys())


def all_domain_aliases() -> dict[str, str]:
    """Flat map of every alias → canonical domain name."""
    result = {}
    for domain, meta in DOMAINS.items():
        result[domain] = domain  # canonical name maps to itself
        for alias in meta["aliases"]:
            result[alias.lower()] = domain
    return result


def domain_description(domain: str) -> str:
    return DOMAINS.get(domain, {}).get("description", domain)
