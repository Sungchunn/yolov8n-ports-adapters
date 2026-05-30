from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from inference.config import Settings
from inference.entrypoints.schemas import UploadResponse, media_detection_to_response
from inference.service_layer.errors import UnsupportedMediaTypeError
from inference.service_layer.media import MediaFormat
from inference.service_layer.ports import AbstractInferenceEngine, AbstractMediaProcessor
from inference.service_layer.services import detect_media

CONTENT_TYPE_TO_MEDIA_FORMAT = {
    "image/jpeg": MediaFormat.JPEG,
    "image/jpg": MediaFormat.JPEG,
    "image/png": MediaFormat.PNG,
    "video/mp4": MediaFormat.MP4,
    "video/quicktime": MediaFormat.QUICKTIME,
    "video/x-msvideo": MediaFormat.AVI,
    "video/webm": MediaFormat.WEBM,
}


def create_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/v1/detect")

    async def detect_upload(
        request: Request, file: Annotated[UploadFile, File(...)]
    ) -> UploadResponse:
        media_bytes = await _read_upload(
            file=file,
            max_bytes=settings.max_upload_bytes,
        )
        result = detect_media(
            media_bytes=media_bytes,
            media_format=_media_format_from_content_type(file.content_type),
            inference_engine=_get_inference_engine(request),
            media_processor=_get_media_processor(request),
        )
        return media_detection_to_response(result)

    router.add_api_route(
        "/upload",
        detect_upload,
        methods=["POST"],
        response_model=UploadResponse,
    )
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


def _media_format_from_content_type(content_type: str | None) -> MediaFormat:
    normalized = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
    try:
        return CONTENT_TYPE_TO_MEDIA_FORMAT[normalized]
    except KeyError as exc:
        raise UnsupportedMediaTypeError(
            f"unsupported media type: {content_type or 'unknown'}"
        ) from exc


def _get_inference_engine(request: Request) -> AbstractInferenceEngine:
    return cast(AbstractInferenceEngine, request.app.state.inference_engine)


def _get_media_processor(request: Request) -> AbstractMediaProcessor:
    return cast(AbstractMediaProcessor, request.app.state.media_processor)
