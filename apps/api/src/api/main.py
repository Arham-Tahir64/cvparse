"""FlowBuildr API entry point."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from .routes.cv_takeoff import router as cv_takeoff_router
from .routes.takeoff_models import router as takeoff_models_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="FlowBuildr API")
app.include_router(cv_takeoff_router)
app.include_router(takeoff_models_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
