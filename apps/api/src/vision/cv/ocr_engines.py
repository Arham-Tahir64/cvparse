"""OCR engine loading and text extraction shared by modules 03 and 10.

The engine instance is cached at module level. `get_engine` returns None when
no OCR backend is importable; callers decide whether that is fatal (module 10)
or degradable (module 03 first pass, which only locates text).
"""
from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, Optional

import numpy as np

from .models import TextElement

logger = logging.getLogger("flowbuildr.cv.ocr")

_ENGINE_CACHE: dict[str, Any] = {}
_UNAVAILABLE = object()

_EXECUTOR: Optional[ProcessPoolExecutor] = None
_EXECUTOR_KEY: Optional[tuple[str, int]] = None
_WORKER_PREFERRED = "paddle"


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


def supports_worker_pool(engine) -> bool:
    """True when `engine` can be recreated identically in a worker process.

    Injected/custom engines (tests, adapters) must keep running in-process;
    a worker would silently resolve a different backend for them.
    """
    return isinstance(engine, (_PaddleEngine, _TesseractEngine))


def _worker_init(preferred: str) -> None:
    global _WORKER_PREFERRED
    _WORKER_PREFERRED = preferred
    get_engine(preferred)  # load the model eagerly so startup overlaps


def _worker_read(image: np.ndarray) -> list[TextElement]:
    """Unfiltered read in an OCR worker process; callers apply thresholds."""
    engine = get_engine(_WORKER_PREFERRED)
    if engine is None:
        raise RuntimeError("no OCR engine available in worker process")
    return engine.read(image, 0.0)


def get_executor(preferred: str, workers: int) -> Optional[ProcessPoolExecutor]:
    """Shared OCR worker pool, or None when parallel OCR is disabled/broken.

    Workers each hold their own engine instance, so per-image results are
    identical to the in-process engine; only scheduling differs.
    """
    global _EXECUTOR, _EXECUTOR_KEY
    if workers <= 1:
        return None
    key = (preferred, workers)
    if _EXECUTOR is not None and _EXECUTOR_KEY == key:
        return _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False)
        _EXECUTOR = None
    try:
        _EXECUTOR = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(preferred,),
        )
        _EXECUTOR_KEY = key
    except Exception as exc:
        logger.warning("OCR worker pool unavailable, falling back to sequential: %s", exc)
        _EXECUTOR = None
        _EXECUTOR_KEY = None
    return _EXECUTOR


def submit_read(
    preferred: str, workers: int, image: np.ndarray
) -> Optional[Future]:
    """Submit an unfiltered engine.read to the worker pool, if enabled."""
    executor = get_executor(preferred, workers)
    if executor is None:
        return None
    try:
        return executor.submit(_worker_read, image)
    except Exception as exc:
        logger.warning("OCR submit failed, falling back to sequential: %s", exc)
        return None


def read_tiled(
    engine, image: np.ndarray, confidence_threshold: float,
    tile_px: int = 2400, overlap_px: int = 120,
    executor: Optional[ProcessPoolExecutor] = None,
) -> list[TextElement]:
    """Run OCR over overlapping tiles at native resolution.

    Large sheets get internally downscaled by the engines (PaddleOCR caps the
    long side at 4000 px), which destroys small dimension text. Tiling keeps
    the text at full resolution. Duplicates in overlap zones are dropped by
    center proximity. With an executor, tiles are read by worker processes in
    parallel; results are collected in the same tile order as the sequential
    path, so output is identical.
    """
    h, w = image.shape[:2]
    if max(h, w) <= tile_px:
        return engine.read(image, confidence_threshold)

    tiles: list[tuple[int, int, np.ndarray]] = []
    step = tile_px - overlap_px
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(y0 + tile_px, h), min(x0 + tile_px, w)
            tile = image[y0:y1, x0:x1]
            if tile.size > 0 and tile.min() != tile.max():  # skip blank tiles
                tiles.append((x0, y0, tile))
            if x1 == w:
                break
        if y0 + tile_px >= h:
            break

    per_tile: Optional[list[list[TextElement]]] = None
    if executor is not None:
        try:
            futures = [executor.submit(_worker_read, tile) for _, _, tile in tiles]
            per_tile = [
                [t for t in f.result() if t.confidence >= confidence_threshold]
                for f in futures
            ]
        except Exception as exc:
            logger.warning("parallel tile OCR failed, retrying sequentially: %s", exc)
            per_tile = None
    if per_tile is None:
        per_tile = [engine.read(tile, confidence_threshold) for _, _, tile in tiles]

    elements: list[TextElement] = []
    for (x0, y0, _), texts in zip(tiles, per_tile):
        for t in texts:
            elements.append(TextElement(
                text=t.text,
                bbox=(t.bbox[0] + x0, t.bbox[1] + y0,
                      t.bbox[2] + x0, t.bbox[3] + y0),
                confidence=t.confidence,
            ))

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
