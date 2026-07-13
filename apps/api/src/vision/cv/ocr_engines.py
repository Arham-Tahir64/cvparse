"""OCR engine loading and text extraction shared by modules 03 and 10.

The engine instance is cached at module level. `get_engine` returns None when
no OCR backend is importable; callers decide whether that is fatal (module 10)
or degradable (module 03 first pass, which only locates text).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .models import TextElement

logger = logging.getLogger("flowbuildr.cv.ocr")

_ENGINE_CACHE: dict[str, Any] = {}
_UNAVAILABLE = object()


def get_engine(preferred: str = "paddle") -> Optional[Any]:
    """Return a cached OCR engine wrapper, or None if none is available."""
    cached = _ENGINE_CACHE.get(preferred)
    if cached is _UNAVAILABLE:
        return None
    if cached is not None:
        return cached

    engine = _load_engine(preferred)
    _ENGINE_CACHE[preferred] = engine if engine is not None else _UNAVAILABLE
    return engine


def _load_engine(preferred: str) -> Optional[Any]:
    loaders = [_load_paddle, _load_tesseract]
    if preferred == "tesseract":
        loaders.reverse()
    for loader in loaders:
        try:
            engine = loader()
        except Exception as exc:
            # Importability does not guarantee usability: model caches can be
            # missing/corrupt/unreadable and native runtimes can fail during
            # construction. Treat one backend's initialization failure the
            # same as unavailability so the next configured backend can run.
            logger.warning("OCR backend %s failed to initialize: %s",
                           loader.__name__, exc)
            continue
        if engine is not None:
            return engine
    return None


def _load_paddle():
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return None
    try:  # PaddleOCR 3.x
        ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            # paddlepaddle 3.x oneDNN backend crashes on some Windows CPUs
            enable_mkldnn=False,
        )
    except (ValueError, TypeError):  # 2.x signature
        ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    logger.info("using PaddleOCR engine")
    return _PaddleEngine(ocr)


def _load_tesseract():
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:
        return None
    logger.info("using Tesseract engine")
    return _TesseractEngine(pytesseract)


def read_tiled(
    engine, image: np.ndarray, confidence_threshold: float,
    tile_px: int = 2400, overlap_px: int = 120,
) -> list[TextElement]:
    """Run OCR over overlapping tiles at native resolution.

    Large sheets get internally downscaled by the engines (PaddleOCR caps the
    long side at 4000 px), which destroys small dimension text. Tiling keeps
    the text at full resolution. Duplicates in overlap zones are dropped by
    center proximity.
    """
    h, w = image.shape[:2]
    if max(h, w) <= tile_px:
        return engine.read(image, confidence_threshold)

    elements: list[TextElement] = []
    step = tile_px - overlap_px
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(y0 + tile_px, h), min(x0 + tile_px, w)
            tile = image[y0:y1, x0:x1]
            if tile.size == 0 or tile.min() == tile.max():
                continue  # blank tile
            for t in engine.read(tile, confidence_threshold):
                elements.append(TextElement(
                    text=t.text,
                    bbox=(t.bbox[0] + x0, t.bbox[1] + y0,
                          t.bbox[2] + x0, t.bbox[3] + y0),
                    confidence=t.confidence,
                ))
            if x1 == w:
                break
        if y0 + tile_px >= h:
            break

    # dedupe overlap-zone duplicates: same text with nearby centers
    deduped: list[TextElement] = []
    for t in sorted(elements, key=lambda e: -e.confidence):
        c = t.center
        if any(
            d.text == t.text and abs(d.center.x - c.x) < overlap_px
            and abs(d.center.y - c.y) < overlap_px
            for d in deduped
        ):
            continue
        deduped.append(t)
    return deduped


class _PaddleEngine:
    name = "paddle"

    def __init__(self, ocr):
        self._ocr = ocr

    def read(self, image: np.ndarray, confidence_threshold: float) -> list[TextElement]:
        if image.ndim == 2:  # paddle expects 3-channel input
            image = np.repeat(image[:, :, None], 3, axis=2)
        if hasattr(self._ocr, "predict"):  # 3.x
            return self._read_v3(image, confidence_threshold)
        return self._read_v2(image, confidence_threshold)

    def _read_v3(self, image, confidence_threshold) -> list[TextElement]:
        elements: list[TextElement] = []
        for res in self._ocr.predict(image) or []:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("rec_polys", res.get("dt_polys", []))
            for text, conf, poly in zip(texts, scores, polys):
                if conf < confidence_threshold or not text.strip():
                    continue
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                elements.append(TextElement(
                    text=text, bbox=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=float(conf),
                ))
        return elements

    def _read_v2(self, image, confidence_threshold) -> list[TextElement]:
        results = self._ocr.ocr(image, cls=False)
        elements: list[TextElement] = []
        for page in results or []:
            for box, (text, conf) in page or []:
                if conf < confidence_threshold or not text.strip():
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                elements.append(TextElement(
                    text=text, bbox=(min(xs), min(ys), max(xs), max(ys)),
                    confidence=float(conf),
                ))
        return elements


class _TesseractEngine:
    name = "tesseract"

    def __init__(self, pytesseract):
        self._pt = pytesseract

    def read(self, image: np.ndarray, confidence_threshold: float) -> list[TextElement]:
        data = self._pt.image_to_data(image, output_type=self._pt.Output.DICT)
        elements: list[TextElement] = []
        for i, text in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if not text.strip() or conf < 0:
                continue
            conf /= 100.0
            if conf < confidence_threshold:
                continue
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            elements.append(TextElement(
                text=text, bbox=(float(x), float(y), float(x + w), float(y + h)),
                confidence=conf,
            ))
        return elements
