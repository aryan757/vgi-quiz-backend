"""Async MongoDB access (Motor) for the live request path.

Exposes a single shared client plus accessors for the two collections described in
Section 4 of the spec. The seeding script uses PyMongo (sync) separately — see
scripts/seed_knowledge_base.py.
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(settings.database_url, serverSelectionTimeoutMS=5000)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    return get_client()[get_settings().database_name]


def get_knowledge_base() -> AsyncIOMotorCollection:
    return get_database()[get_settings().knowledge_base_collection]


def get_question_bank() -> AsyncIOMotorCollection:
    return get_database()[get_settings().question_bank_collection]


async def ping() -> bool:
    """Return True if the server responds to a ping. Used by /health and startup."""
    await get_client().admin.command("ping")
    return True


async def ensure_indexes() -> None:
    """Create the indexes from Section 4. Idempotent — safe to call on every startup."""
    kb = get_knowledge_base()
    qb = get_question_bank()

    # knowledge_base: compound (domain, topic, difficulty) + text index for keyword fallback
    await kb.create_index([("domain", 1), ("topic", 1), ("difficulty", 1)], name="kb_dom_top_diff")
    await kb.create_index([("topic", "text"), ("subtopics", "text")], name="kb_text")

    # question_bank: compound (topic, difficulty) — queried in Step 3 inventory check
    await qb.create_index([("topic", 1), ("difficulty", 1)], name="qb_topic_diff")
    logger.info("MongoDB indexes ensured.")


async def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
