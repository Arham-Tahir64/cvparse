"""Tests for OCR backend discovery and failover."""

from vision.cv import ocr_engines


def test_initialization_failure_falls_back(monkeypatch):
    fallback = object()

    def unreadable_model_cache():
        raise PermissionError("model cache is unreadable")

    monkeypatch.setattr(ocr_engines, "_load_paddle", unreadable_model_cache)
    monkeypatch.setattr(ocr_engines, "_load_tesseract", lambda: fallback)

    assert ocr_engines._load_engine("paddle") is fallback


def test_all_initialization_failures_return_none(monkeypatch):
    def broken():
        raise RuntimeError("backend failed")

    monkeypatch.setattr(ocr_engines, "_load_paddle", broken)
    monkeypatch.setattr(ocr_engines, "_load_tesseract", broken)

    assert ocr_engines._load_engine("paddle") is None
