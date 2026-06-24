"""Tests for request schema validation (Section 6.2 field rules)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models.request import GenerateQuestionsRequest

client = TestClient(app, raise_server_exceptions=False)


# --- Unit tests on the Pydantic model ---

def test_valid_custom_request():
    r = GenerateQuestionsRequest(type="CUSTOM", topic="Transformers", question_count=10)
    assert r.type.value == "CUSTOM"
    assert r.difficulty.value == "MIXED"  # default


def test_difficulty_defaults_to_mixed():
    r = GenerateQuestionsRequest(type="CUSTOM", topic="RAG", question_count=5)
    assert r.difficulty.value == "MIXED"


def test_auto_topic():
    r = GenerateQuestionsRequest(type="CUSTOM", topic="auto", question_count=5)
    assert r.is_auto_topic is True


def test_topic_case_insensitive_auto():
    r = GenerateQuestionsRequest(type="CUSTOM", topic="  AUTO  ", question_count=5)
    assert r.is_auto_topic is True


def test_question_count_boundaries():
    GenerateQuestionsRequest(type="CUSTOM", topic="t", question_count=1)
    GenerateQuestionsRequest(type="CUSTOM", topic="t", question_count=100)
    with pytest.raises(ValidationError):
        GenerateQuestionsRequest(type="CUSTOM", topic="t", question_count=0)
    with pytest.raises(ValidationError):
        GenerateQuestionsRequest(type="CUSTOM", topic="t", question_count=101)


def test_invalid_difficulty():
    with pytest.raises(ValidationError):
        GenerateQuestionsRequest(type="CUSTOM", topic="t", difficulty="EXPERT", question_count=5)


def test_invalid_type():
    with pytest.raises(ValidationError):
        GenerateQuestionsRequest(type="WEEKLY", topic="t", question_count=5)


def test_blank_topic_rejected():
    with pytest.raises(ValidationError):
        GenerateQuestionsRequest(type="CUSTOM", topic="   ", question_count=5)


# --- HTTP-level validation (FastAPI 422) ---

def test_http_422_on_out_of_range_count():
    resp = client.post("/generate-questions", json={
        "type": "CUSTOM", "topic": "X", "question_count": 200
    })
    assert resp.status_code == 422


def test_http_422_on_missing_topic():
    resp = client.post("/generate-questions", json={
        "type": "CUSTOM", "question_count": 5
    })
    assert resp.status_code == 422


def test_http_501_on_daily():
    resp = client.post("/generate-questions", json={
        "type": "DAILY", "topic": "Transformers", "question_count": 5
    })
    assert resp.status_code == 501
    body = resp.json()
    assert body["success"] is False
    assert body["count"] == 0


def test_http_501_on_session():
    resp = client.post("/generate-questions", json={
        "type": "SESSION", "topic": "Transformers", "question_count": 5
    })
    assert resp.status_code == 501
