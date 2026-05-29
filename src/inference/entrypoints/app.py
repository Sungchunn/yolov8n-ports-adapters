from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from inference.bootstrap import bootstrap
from inference.config import Settings, get_settings
from inference.domain.exceptions import (
    DomainError,
    InferenceError,
    InvalidImageError,
    InvalidVideoError,
    UnsupportedMediaTypeError,
)
from inference.entrypoints.routes import create_router
from inference.service_layer.ports import AbstractInferenceEngine, AbstractMediaProcessor


STATIC_DIR = Path(__file__).with_name("static")
ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"


def create_app(
    *,
    engine: AbstractInferenceEngine | None = None,
    media_processor: AbstractMediaProcessor | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if engine is None or media_processor is None:
            dependencies = bootstrap(resolved_settings)
            app.state.inference_engine = engine or dependencies.inference_engine
            app.state.media_processor = media_processor or dependencies.media_processor
        yield

    app = FastAPI(title="Vision Inference API", lifespan=lifespan)
    if engine is not None:
        app.state.inference_engine = engine
    if media_processor is not None:
        app.state.media_processor = media_processor

    app.include_router(create_router(resolved_settings))
    _register_frontend(app)
    _register_exception_handlers(app)
    return app


def _register_frontend(app: FastAPI) -> None:
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def frontend_index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidImageError)
    async def invalid_image_handler(
        _request: Request, exc: InvalidImageError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(InvalidVideoError)
    async def invalid_video_handler(
        _request: Request, exc: InvalidVideoError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(UnsupportedMediaTypeError)
    async def unsupported_media_type_handler(
        _request: Request, exc: UnsupportedMediaTypeError
    ) -> JSONResponse:
        return JSONResponse(status_code=415, content={"detail": str(exc)})

    @app.exception_handler(InferenceError)
    async def inference_error_handler(
        _request: Request, exc: InferenceError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(DomainError)
    async def domain_error_handler(_request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


app = create_app()
