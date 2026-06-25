"""Async MongoDB access (Motor) for the live request path."""

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
    await get_client().admin.command("ping")
    return True


async def ensure_indexes() -> None:
    """Create indexes. Idempotent — safe to call on every startup."""
    kb = get_knowledge_base()
    qb = get_question_bank()

    # knowledge_base: domain + difficulty is the primary filter
    await kb.create_index([("domain", 1), ("difficulty", 1)], name="kb_domain_diff")

    # question_bank: domain + difficulty — inventory check in Step 3
    await qb.create_index([("domain", 1), ("difficulty", 1)], name="qb_domain_diff")
    # keep topic index for backward compat with any existing question_bank docs
    await qb.create_index([("topic", 1), ("difficulty", 1)], name="qb_topic_diff")

    logger.info("MongoDB indexes ensured.")


async def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
