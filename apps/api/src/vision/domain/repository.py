"""Persistence repositories with optimistic revision protection."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Protocol

from .models import TakeoffModel
from .serialize import from_json_dict, to_json_dict


class ModelRepositoryError(Exception):
    """Base repository error."""


class ModelNotFoundError(ModelRepositoryError):
    pass


class RevisionConflictError(ModelRepositoryError):
    pass


class ModelRepository(Protocol):
    def get(self, model_id: str) -> TakeoffModel: ...

    def save(
        self, model: TakeoffModel, *, expected_revision: int | None = None,
    ) -> None: ...


def _copy(model: TakeoffModel) -> TakeoffModel:
    return from_json_dict(to_json_dict(model))


def _check_revision(
    current: TakeoffModel | None,
    incoming: TakeoffModel,
    expected_revision: int | None,
) -> None:
    if current is None:
        if expected_revision is not None:
            raise RevisionConflictError(
                f"model {incoming.id} does not exist at revision {expected_revision}"
            )
        return
    if expected_revision is None:
        raise RevisionConflictError(
            f"model {incoming.id} already exists at revision {current.revision}"
        )
    if current.revision != expected_revision:
        raise RevisionConflictError(
            f"model {incoming.id} is revision {current.revision}, "
            f"not expected revision {expected_revision}"
        )
    if incoming.revision != expected_revision + 1:
        raise RevisionConflictError(
            f"updated model revision must be {expected_revision + 1}, "
            f"got {incoming.revision}"
        )


class InMemoryModelRepository:
    """Isolated repository for tests and single-process embedding."""

    def __init__(self):
        self._models: dict[str, TakeoffModel] = {}
        self._lock = threading.RLock()

    def get(self, model_id: str) -> TakeoffModel:
        with self._lock:
            try:
                return _copy(self._models[model_id])
            except KeyError as exc:
                raise ModelNotFoundError(f"model {model_id} was not found") from exc

    def save(
        self, model: TakeoffModel, *, expected_revision: int | None = None,
    ) -> None:
        with self._lock:
            current = self._models.get(model.id)
            _check_revision(current, model, expected_revision)
            self._models[model.id] = _copy(model)


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class JsonFileModelRepository:
    """Atomic JSON-file persistence suitable for the current single API service.

    A process-local lock protects read/check/write. A database-backed repository
    should replace this when multiple API worker processes are introduced.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()

    def _path(self, model_id: str) -> Path:
        if not _SAFE_ID.fullmatch(model_id):
            raise ModelRepositoryError("model id contains unsafe path characters")
        return self.root / f"{model_id}.json"

    def _read_unlocked(self, model_id: str) -> TakeoffModel | None:
        path = self._path(model_id)
        if not path.exists():
            return None
        return from_json_dict(json.loads(path.read_text(encoding="utf-8")))

    def get(self, model_id: str) -> TakeoffModel:
        with self._lock:
            model = self._read_unlocked(model_id)
            if model is None:
                raise ModelNotFoundError(f"model {model_id} was not found")
            return model

    def save(
        self, model: TakeoffModel, *, expected_revision: int | None = None,
    ) -> None:
        with self._lock:
            current = self._read_unlocked(model.id)
            _check_revision(current, model, expected_revision)
            self.root.mkdir(parents=True, exist_ok=True)
            path = self._path(model.id)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(to_json_dict(model), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
