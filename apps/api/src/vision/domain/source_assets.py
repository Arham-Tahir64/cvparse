"""Content-addressed persistence for original plan uploads."""
from __future__ import annotations

import hashlib
import re
import threading
import uuid
from pathlib import Path
from typing import Protocol


class SourceAssetError(Exception):
    pass


class SourceAssetNotFoundError(SourceAssetError):
    pass


class SourceAssetIntegrityError(SourceAssetError):
    pass


class SourceAssetRepository(Protocol):
    def get(self, fingerprint: str) -> bytes: ...

    def save(self, fingerprint: str, content: bytes) -> None: ...


_FINGERPRINT = re.compile(r"^[a-f0-9]{64}$")


def _validate(fingerprint: str, content: bytes) -> str:
    normalized = fingerprint.lower()
    if not _FINGERPRINT.fullmatch(normalized):
        raise SourceAssetIntegrityError("source fingerprint must be a SHA-256 hex digest")
    if not content:
        raise SourceAssetIntegrityError("source asset cannot be empty")
    actual = hashlib.sha256(content).hexdigest()
    if actual != normalized:
        raise SourceAssetIntegrityError(
            f"source content SHA-256 {actual} does not match {normalized}"
        )
    return normalized


class InMemorySourceAssetRepository:
    def __init__(self):
        self._assets: dict[str, bytes] = {}
        self._lock = threading.RLock()

    def get(self, fingerprint: str) -> bytes:
        normalized = fingerprint.lower()
        with self._lock:
            try:
                return bytes(self._assets[normalized])
            except KeyError as exc:
                raise SourceAssetNotFoundError(
                    f"source asset {normalized} was not found"
                ) from exc

    def save(self, fingerprint: str, content: bytes) -> None:
        normalized = _validate(fingerprint, content)
        with self._lock:
            existing = self._assets.get(normalized)
            if existing is not None and existing != content:
                raise SourceAssetIntegrityError(
                    "different content cannot replace a fingerprinted source asset"
                )
            self._assets[normalized] = bytes(content)


class FileSourceAssetRepository:
    """Atomic, idempotent storage keyed only by verified upload SHA-256."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()

    def _path(self, fingerprint: str) -> Path:
        normalized = fingerprint.lower()
        if not _FINGERPRINT.fullmatch(normalized):
            raise SourceAssetIntegrityError(
                "source fingerprint must be a SHA-256 hex digest"
            )
        return self.root / f"{normalized}.bin"

    def get(self, fingerprint: str) -> bytes:
        path = self._path(fingerprint)
        with self._lock:
            if not path.exists():
                raise SourceAssetNotFoundError(
                    f"source asset {fingerprint.lower()} was not found"
                )
            content = path.read_bytes()
            _validate(fingerprint, content)
            return content

    def save(self, fingerprint: str, content: bytes) -> None:
        normalized = _validate(fingerprint, content)
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self._path(normalized)
            if path.exists():
                existing = path.read_bytes()
                _validate(normalized, existing)
                if existing != content:
                    raise SourceAssetIntegrityError(
                        "different content cannot replace a fingerprinted source asset"
                    )
                return
            temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_bytes(content)
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
