from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from inference.domain.model import Detection, InferenceResult


class BoxResponse(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    area: float
    center: tuple[float, float]


class LabelResponse(BaseModel):
    class_id: int
    name: str


class DetectionResponse(BaseModel):
    label: LabelResponse
    confidence: float
    box: BoxResponse


class ImageResponse(BaseModel):
    width: int
    height: int


class InferenceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(serialization_alias="model")
    image: ImageResponse
    inference_ms: float
    detections: list[DetectionResponse]
    created_at: datetime


class VideoFrameResponse(InferenceResponse):
    frame_index: int
    timestamp_seconds: float


class VideoResponse(BaseModel):
    sample_interval_seconds: float
    frames: list[VideoFrameResponse]


def detection_to_response(detection: Detection) -> DetectionResponse:
    return DetectionResponse(
        label=LabelResponse(
            class_id=detection.label.class_id,
            name=detection.label.name,
        ),
        confidence=detection.confidence,
        box=BoxResponse(
            x1=detection.box.x1,
            y1=detection.box.y1,
            x2=detection.box.x2,
            y2=detection.box.y2,
            width=detection.box.width,
            height=detection.box.height,
            area=detection.box.area,
            center=detection.box.center,
        ),
    )


def inference_to_response(result: InferenceResult) -> InferenceResponse:
    return InferenceResponse(
        model_name=result.model_name,
        image=ImageResponse(width=result.image_width, height=result.image_height),
        inference_ms=result.inference_ms,
        detections=[detection_to_response(item) for item in result.detections],
        created_at=result.created_at,
    )
