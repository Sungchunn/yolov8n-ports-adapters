from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from inference.config import Settings
from inference.entrypoints.schemas import UploadResponse, media_detection_to_response
from inference.service_layer.ports import AbstractInferenceEngine, AbstractMediaProcessor
from inference.service_layer.services import detect_media


def create_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/v1/detect")

    @router.post("/upload", response_model=UploadResponse)
    async def detect_upload(
        request: Request, file: Annotated[UploadFile, File(...)]
    ) -> UploadResponse:
        media_bytes = await _read_upload(
            file=file,
            max_bytes=settings.max_upload_bytes,
        )
        result = detect_media(
            media_bytes=media_bytes,
            content_type=file.content_type or "",
            inference_engine=_get_inference_engine(request),
            media_processor=_get_media_processor(request),
        )
        return media_detection_to_response(result)

    return router


async def _read_upload(
    file: UploadFile,
    max_bytes: int,
) -> bytes:
    payload = await file.read(max_bytes + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="upload payload must not be empty")
    if len(payload) > max_bytes:
        raise HTTPException(status_code=413, detail="upload payload is too large")
    return payload


def _get_inference_engine(request: Request) -> AbstractInferenceEngine:
    return cast(AbstractInferenceEngine, request.app.state.inference_engine)


def _get_media_processor(request: Request) -> AbstractMediaProcessor:
    return cast(AbstractMediaProcessor, request.app.state.media_processor)
