"""
app/main.py

FastAPI application.
Exposes:
  GET  /health   → {"status": "ok"}
  POST /chat     → ChatResponse
"""
from __future__ import annotations
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from groq import APIError, APIConnectionError

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from app.agent import Agent
from app.catalog import Catalog
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retrieval import RetrievalEngine

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Module-level singletons (initialised in lifespan)
_catalog: Catalog | None = None
_retrieval: RetrievalEngine | None = None
_agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load catalog + build retrieval index. Shutdown: nothing."""
    global _catalog, _retrieval, _agent
    logger.info("Starting up — loading catalog ...")
    try:
        _catalog = Catalog.load()
        logger.info("Catalog loaded: %d entries", len(_catalog.entries))
        _retrieval = RetrievalEngine(_catalog)
        logger.info("Warming up retrieval engine...")
        _retrieval.warmup()
        logger.info("Retrieval engine initialised")
        _agent = Agent(catalog=_catalog, retrieval=_retrieval)
        logger.info("Agent ready")
    except Exception as exc:
        logger.error("Startup failed: %s", exc)
        raise
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    description="Conversational SHL assessment selection agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)



# Middleware: request timing


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response



# Exception handlers


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )



# Routes

@app.get("/")
async def root():
    return {
        "message": "SHL Recommender API is running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Readiness check. Returns 200 with status=ok when the service is ready."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint.

    Accepts a stateless conversation history and returns:
    - reply: the agent's natural-language response
    - recommendations: 0–10 SHL assessments (empty when still clarifying)
    - end_of_conversation: true when the task is complete
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Turn cap enforcement at API layer (belt-and-suspenders)
    total_turns = len(request.messages)
    if total_turns > 8:
        raise HTTPException(
            status_code=400,
            detail="Conversation exceeds maximum of 8 turns.",
        )

    try:
        response: ChatResponse = _agent.process(request)
        return response
    except APIConnectionError as exc:
        logger.error("Groq connection error: %s", exc)

        raise HTTPException(
            status_code=504,
            detail="Could not reach the language model. Please retry.",
        )

    except APIError as exc:
        logger.error("Groq API error: %s", exc)

        status_code = getattr(exc, "status_code", 502)

        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM error ({status_code})",
        )
    except Exception as exc:
        logger.error("Chat processing error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request.",
        )



# Dev server entrypoint


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("DEV", "false").lower() == "true",
        log_level="info",
    )
