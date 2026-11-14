from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from app import classifier, generator
from app.main import app

class DummyPipeline:
    def predict_proba(self, text):
        return np.array([[0.9, 0.8, 0.1, 0.1, 0.1]])

DUMMY = {"pipeline": DummyPipeline(), "ids": (102, 103, 104, 105, 106), "thresholds": [0.5] * 5}

def test_classify():
    assert classifier.classify(DUMMY, "штукатурим и укладываем плитку", 101) == [102, 103]
    assert classifier.classify(DUMMY, "штукатурим и укладываем плитку", 102) == [103]

def test_validation():
    src = "Выполняем укладку плитки"
    assert generator.validate_draft("Укладка плитки", "Выполняем укладку плитки.", "укладку плитки", src)
    assert not generator.validate_draft("Гарантия 5 лет", "Укладываем плитку с гарантией.", "укладку плитки", src)
    assert not generator.validate_draft("Плитка", "Укладываем плитку", "неверная цитата", src)

def test_api_classification_only():
    with patch("app.main.load_model", return_value=DUMMY):
        with TestClient(app) as client:
            r = client.post("/generate?generate_text=false", json={"description": "штукатурим и укладываем плитку", "mcId": 101})
            assert r.status_code == 200
            assert r.json() == {"drafts": [
                {"mcId": 102, "title": None, "text": None},
                {"mcId": 103, "title": None, "text": None},
            ]}
            assert client.get("/health").json() == {"status": "ok"}

def test_bad_input():
    with patch("app.main.load_model", return_value=DUMMY):
        with TestClient(app) as client:
            assert client.post("/generate", json={"description": "текст", "mcId": 999}).status_code == 422


def test_fallback_from_exact_quote():
    src = "Укладываем плитку и керамогранит, устанавливаем раковины и унитазы."
    assert generator.fallback_from_evidence(src, "Укладываем плитку и керамогранит", 103) == (
        "Плиточные работы", "Укладываем плитку и керамогранит")
    assert generator.fallback_from_evidence(src, "устанавливаем раковины и унитазы", 104) == (
        "Сантехнические работы", "Устанавливаем раковины и унитазы")

def test_fallback_rejects_unverified_and_denied():
    src = "Электрику не делаем, плитку кладём."
    assert generator.fallback_from_evidence(src, "электрические работы", 105) is None
    assert generator.fallback_from_evidence(src, "Электрику не делаем", 105) is None
    assert generator.fallback_from_evidence(src, "плитку кладём", 104) is None

