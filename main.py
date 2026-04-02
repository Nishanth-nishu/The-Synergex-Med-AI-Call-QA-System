"""
FastAPI application entry point for the Synergex Med Call QA System.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

load_dotenv()

from app.api.routes import router
from app.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Synergex Med Call QA API",
        description=(
            "AI-powered phone call quality analysis system. "
            "Analyzes call transcripts for HIPAA compliance, agent performance, "
            "and escalation decisions using Gemini 2.5 Flash."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — permissive for local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Include routers ──
    app.include_router(router)

    # ── Startup: initialize LLM provider and inject into app state ──
    @app.on_event("startup")
    async def startup_event() -> None:
        provider_name = os.environ.get("LLM_PROVIDER", "gemini").lower()

        if provider_name == "gemini":
            from app.providers.gemini import GeminiProvider
            provider = GeminiProvider()
        elif provider_name == "openai":
            from app.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider()
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER='{provider_name}'. Must be 'gemini' or 'openai'."
            )

        from app.services.qa_analyzer import QAAnalyzer
        app.state.analyzer = QAAnalyzer(provider=provider)

        logger.info(
            "app_startup",
            provider=provider.provider_name,
            model=provider.model_name,
        )

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        logger.info("app_shutdown")

    # ── Root Landing Page ──
    @app.get("/", tags=["System"], include_in_schema=False)
    async def root() -> HTMLResponse:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Synergex Med Call QA API</title>
                <style>
                    body {
                        font-family: 'Inter', -apple-system, blinkmacsystemfont, 'Segoe UI', roboto, oxygen, ubuntu, cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                        background-color: #0f172a;
                        color: #f8fafc;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        text-align: center;
                    }
                    .container {
                        max-width: 600px;
                        padding: 2rem;
                        background: rgba(30, 41, 59, 0.7);
                        backdrop-filter: blur(10px);
                        border-radius: 1rem;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                    }
                    h1 { color: #38bdf8; margin-bottom: 0.5rem; }
                    p { color: #94a3b8; line-height: 1.6; margin-bottom: 2rem; }
                    .links { display: flex; gap: 1rem; justify-content: center; }
                    a {
                        padding: 0.75rem 1.5rem;
                        border-radius: 0.5rem;
                        text-decoration: none;
                        font-weight: 600;
                        transition: all 0.2s;
                    }
                    .primary { background-color: #38bdf8; color: #0f172a; }
                    .primary:hover { background-color: #7dd3fc; transform: translateY(-2px); }
                    .secondary { border: 1px solid #334155; color: #f8fafc; }
                    .secondary:hover { background-color: #1e293b; transform: translateY(-2px); }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Synergex Med Call QA API</h1>
                    <p>AI-powered phone call quality analysis system. Automatically monitoring 100% of calls for HIPAA compliance, professionalism, and accuracy using Gemini 2.5 Flash.</p>
                    <div class="links">
                        <a href="/docs" class="primary">Interactive Documentation</a>
                        <a href="/health" class="secondary">Health Status</a>
                    </div>
                </div>
            </body>
        </html>
        """)

    # ── Health check ──
    @app.get("/health", tags=["System"], summary="Health check")
    async def health() -> dict:
        return {"status": "ok", "service": "synergex-qa"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
