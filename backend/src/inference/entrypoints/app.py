from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.bootstrap import create_inference_engine, create_media_processor
from inference.config import Settings, get_settings
from inference.domain.exceptions import DomainError
from inference.service_layer.errors import (
    ApplicationError,
    InferenceError,
    InvalidImageError,
    InvalidVideoError,
    UnsupportedMediaTypeError,
)
from inference.entrypoints.routes import create_router
from inference.service_layer.ports import AbstractInferenceEngine, AbstractMediaProcessor


def create_app(
    *,
    engine: AbstractInferenceEngine | None = None,
    media_processor: AbstractMediaProcessor | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if engine is None:
            app.state.inference_engine = create_inference_engine(resolved_settings)
        if media_processor is None:
            app.state.media_processor = create_media_processor(resolved_settings)
        yield

    app = FastAPI(title="Vision Inference API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    if engine is not None:
        app.state.inference_engine = engine
    if media_processor is not None:
        app.state.media_processor = media_processor

    app.include_router(create_router(resolved_settings))
    _register_health_check(app)
    _register_exception_handlers(app)
    return app


async def health_check() -> dict[str, str]:
    return {"status": "ok"}


async def invalid_image_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def invalid_video_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def unsupported_media_type_handler(
    _request: Request, exc: Exception
) -> JSONResponse:
    return JSONResponse(status_code=415, content={"detail": str(exc)})


async def inference_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


async def domain_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def application_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _register_health_check(app: FastAPI) -> None:
    app.add_api_route(
        "/health",
        health_check,
        methods=["GET"],
        include_in_schema=False,
    )


def _register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(InvalidImageError, invalid_image_handler)
    app.add_exception_handler(InvalidVideoError, invalid_video_handler)
    app.add_exception_handler(
        UnsupportedMediaTypeError,
        unsupported_media_type_handler,
    )
    app.add_exception_handler(InferenceError, inference_error_handler)
    app.add_exception_handler(DomainError, domain_error_handler)
    app.add_exception_handler(ApplicationError, application_error_handler)


app = create_app()
