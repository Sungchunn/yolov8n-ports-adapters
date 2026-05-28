from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Annotated, cast

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, WebSocket
from starlette.websockets import WebSocketDisconnect

from inference.adapters import media
from inference.config import Settings
from inference.domain.exceptions import DomainError
from inference.entrypoints.schemas import (
    InferenceResponse,
    VideoFrameResponse,
    VideoResponse,
    inference_to_response,
)
from inference.service_layer.ports import AbstractInferenceEngine
from inference.service_layer.services import detect_objects

JPEG_CONTENT_TYPES = {"image/jpeg", "image/jpg"}
VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
}


def create_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/v1/detect")

    @router.post("/image", response_model=InferenceResponse)
    async def detect_image(
        request: Request, file: Annotated[UploadFile, File(...)]
    ) -> InferenceResponse:
        image_bytes = await _read_upload(
            file=file,
            allowed_content_types=JPEG_CONTENT_TYPES,
            max_bytes=settings.max_upload_bytes,
            expected_kind="JPEG image",
        )
        _validate_jpeg_magic(image_bytes)
        result = detect_objects(image_bytes, _get_engine(request))
        return inference_to_response(result)

    @router.post("", response_model=InferenceResponse, include_in_schema=False)
    async def detect_image_legacy(
        request: Request, file: Annotated[UploadFile, File(...)]
    ) -> InferenceResponse:
        return await detect_image(request=request, file=file)

    @router.post("/video", response_model=VideoResponse)
    async def detect_video(
        request: Request, file: Annotated[UploadFile, File(...)]
    ) -> VideoResponse:
        video_bytes = await _read_upload(
            file=file,
            allowed_content_types=VIDEO_CONTENT_TYPES,
            max_bytes=settings.max_upload_bytes,
            expected_kind="video",
        )
        engine = _get_engine(request)
        frames = media.sample_video_bytes(
            video_bytes,
            sample_interval_seconds=settings.video_sample_interval_seconds,
            max_frames=settings.max_video_frames,
        )
        frame_responses: list[VideoFrameResponse] = []
        for frame in frames:
            response = inference_to_response(detect_objects(frame.image_bytes, engine))
            frame_responses.append(
                VideoFrameResponse(
                    **response.model_dump(),
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                )
            )
        return VideoResponse(
            sample_interval_seconds=settings.video_sample_interval_seconds,
            frames=frame_responses,
        )

    @router.websocket("/webcam")
    async def detect_webcam(websocket: WebSocket) -> None:
        await websocket.accept()
        engine = websocket.app.state.engine
        try:
            capture = media.open_capture(settings.webcam_index)
        except DomainError as exc:
            await websocket.send_json({"error": "webcam_unavailable", "message": str(exc)})
            await websocket.close(code=1011)
            return
        if not capture.isOpened():
            await websocket.send_json(
                {
                    "error": "webcam_unavailable",
                    "message": f"camera index {settings.webcam_index} is unavailable",
                }
            )
            capture.release()
            await websocket.close(code=1011)
            return

        frame_index = 0
        delay_seconds = (
            1.0 / settings.webcam_fps_limit if settings.webcam_fps_limit > 0 else 0.0
        )
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    await websocket.send_json(
                        {
                            "error": "frame_read_failed",
                            "message": "failed to read frame from webcam",
                        }
                    )
                    await websocket.close(code=1011)
                    return
                payload = inference_to_response(
                    detect_objects(media.encode_frame_to_jpeg(frame), engine)
                ).model_dump(mode="json", by_alias=True)
                payload["frame_index"] = frame_index
                payload["timestamp"] = datetime.now(timezone.utc).isoformat()
                await websocket.send_json(payload)
                frame_index += 1
                if delay_seconds:
                    await asyncio.sleep(delay_seconds)
        except WebSocketDisconnect:
            return
        except DomainError as exc:
            await websocket.send_json({"error": "inference_failed", "message": str(exc)})
            await websocket.close(code=1011)
        finally:
            capture.release()

    return router


async def _read_upload(
    file: UploadFile,
    allowed_content_types: set[str],
    max_bytes: int,
    expected_kind: str,
) -> bytes:
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported media type for {expected_kind}",
        )

    payload = await file.read(max_bytes + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="upload payload must not be empty")
    if len(payload) > max_bytes:
        raise HTTPException(status_code=413, detail="upload payload is too large")
    return payload


def _validate_jpeg_magic(payload: bytes) -> None:
    if not payload.startswith(b"\xff\xd8\xff"):
        raise HTTPException(status_code=415, detail="upload payload is not a JPEG")


def _get_engine(request: Request) -> AbstractInferenceEngine:
    return cast(AbstractInferenceEngine, request.app.state.engine)
