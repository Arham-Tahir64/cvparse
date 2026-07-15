"""Shared model repository dependency."""
from __future__ import annotations

import os
from pathlib import Path

from vision.domain.repository import JsonFileModelRepository, ModelRepository


_repository = JsonFileModelRepository(
    Path(os.environ.get("FLOWBUILDR_MODEL_DIR", "data/models"))
)


def get_model_repository() -> ModelRepository:
    return _repository
