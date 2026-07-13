"""FlowBuildr API entry point."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from .routes.cv_takeoff import router as cv_takeoff_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="FlowBuildr API")
app.include_router(cv_takeoff_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
