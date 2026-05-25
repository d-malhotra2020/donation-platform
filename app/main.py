"""FastAPI live demo for the donation-platform recommender Slice 1.

Wraps the trained two-tower from bench/eval/run.py + a slim demo bundle into
an operator-console UI. The model itself is loaded once at boot; per-request
recommendation is a single user-tower forward pass + FAISS top-K lookup.

Endpoints:
  GET  /                      → operator console (single HTML page)
  GET  /api/v1/health         → liveness probe
  GET  /api/v1/dataset        → dataset metadata
  GET  /api/v1/comparison     → 6-model comparison table
  GET  /api/v1/invariants     → synthetic-user invariant statuses
  GET  /api/v1/users          → list of demo users for the UI picker
  GET  /api/v1/users/{uid}    → user detail + train history + top-10 recs
"""
from __future__ import annotations

import os

# torch + faiss both link OpenMP on macOS dev; harmless on Linux containers
# but the macOS dev loop crashes without this. Safe to leave on.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.recommender_service import RecommenderService

logger = logging.getLogger(__name__)

APP_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_ROOT / "artifacts"
TEMPLATES_DIR = APP_ROOT / "templates"
STATIC_DIR = APP_ROOT / "static"

service: RecommenderService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    logger.info("loading recommender artifacts from %s", ARTIFACTS_DIR)
    if not ARTIFACTS_DIR.exists():
        raise RuntimeError(
            f"missing {ARTIFACTS_DIR}. Run `make bench` first to produce artifacts."
        )
    service = RecommenderService(ARTIFACTS_DIR)
    logger.info(
        "service ready · %d orgs, %d demo users, %d donations",
        service.orgs.shape[0],
        service.users.shape[0],
        service.donations.shape[0],
    )
    yield


app = FastAPI(
    title="donation-platform · recommender operator console",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _require_service() -> RecommenderService:
    if service is None:
        raise HTTPException(status_code=503, detail="service not ready")
    return service


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    s = _require_service()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "dataset": s.dataset_summary(),
        },
    )


@app.get("/api/v1/health")
async def health() -> JSONResponse:
    ok = service is not None
    return JSONResponse({"status": "ok" if ok else "starting", "ready": ok})


@app.get("/api/v1/dataset")
async def dataset() -> JSONResponse:
    return JSONResponse(_require_service().dataset_summary())


@app.get("/api/v1/comparison")
async def comparison() -> JSONResponse:
    return JSONResponse({"models": _require_service().comparison_table()})


@app.get("/api/v1/invariants")
async def invariants() -> JSONResponse:
    return JSONResponse({"invariants": _require_service().invariants()})


@app.get("/api/v1/users")
async def list_users(limit: int = 200) -> JSONResponse:
    return JSONResponse({"users": _require_service().list_users(limit=limit)})


@app.get("/api/v1/users/{user_id}")
async def user_detail(user_id: str, k: int = 10) -> JSONResponse:
    detail = _require_service().user_detail(user_id, k=k)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"user {user_id} not in demo bundle")
    return JSONResponse(detail)


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
