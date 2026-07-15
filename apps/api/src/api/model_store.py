"""Shared model repository dependency."""
from __future__ import annotations

import os
from pathlib import Path

from vision.domain.repository import JsonFileModelRepository, ModelRepository
from vision.domain.source_assets import (
    FileSourceAssetRepository,
    SourceAssetRepository,
)


_repository = JsonFileModelRepository(
    Path(os.environ.get("FLOWBUILDR_MODEL_DIR", "data/models"))
)
_source_repository = FileSourceAssetRepository(
    Path(os.environ.get("FLOWBUILDR_SOURCE_DIR", "data/sources"))
)


def get_model_repository() -> ModelRepository:
    return _repository


def get_source_asset_repository() -> SourceAssetRepository:
    return _source_repository
