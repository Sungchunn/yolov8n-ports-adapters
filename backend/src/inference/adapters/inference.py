from __future__ import annotations

from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

from inference.domain.model import BoundingBox, Detection, DetectionLabel, InferenceResult
from inference.service_layer.errors import InferenceError, InvalidImageError
from inference.service_layer.ports import AbstractInferenceEngine


class YoloInferenceEngine(AbstractInferenceEngine):
    def __init__(self, model_path: str, confidence_threshold: float = 0.25) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model_name = Path(model_path).stem
        try:
            yolo_constructor = getattr(import_module("ultralytics"), "YOLO")
        except Exception as exc:  # pragma: no cover - depends on optional runtime extra
            raise InferenceError(
                "ultralytics is required for YoloInferenceEngine; "
                "install with `uv sync --extra vision`"
            ) from exc

        try:
            self._model = yolo_constructor(model_path)
        except Exception as exc:  # pragma: no cover - depends on model/runtime
            raise InferenceError(f"failed to load YOLO model at {model_path}") from exc

    def predict(self, image_bytes: bytes) -> InferenceResult:
        image = self._decode_image(image_bytes)
        image_height, image_width = image.shape[:2]

        started = perf_counter()
        try:
            raw_results = self._model(
                image, conf=self.confidence_threshold, verbose=False
            )
        except Exception as exc:  # pragma: no cover - depends on model/runtime
            raise InferenceError("YOLO inference failed") from exc
        inference_ms = (perf_counter() - started) * 1000

        detections = self._map_result(raw_results[0])
        return InferenceResult(
            detections=detections,
            image_width=int(image_width),
            image_height=int(image_height),
            inference_ms=inference_ms,
            model_name=self.model_name,
        )

    def _decode_image(self, image_bytes: bytes) -> Any:
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover - depends on optional runtime extra
            raise InferenceError(
                "opencv-python and numpy are required for image decoding; "
                "install with `uv sync --extra vision`"
            ) from exc

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("image payload is not decodable")
        return image

    def _map_result(self, result: Any) -> tuple[Detection, ...]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return ()

        names = getattr(result, "names", None) or getattr(self._model, "names", {})
        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()

        detections: list[Detection] = []
        for box, confidence, class_id in zip(xyxy, confidences, classes, strict=True):
            class_int = int(class_id)
            detections.append(
                Detection(
                    box=BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    ),
                    label=DetectionLabel(
                        class_id=class_int,
                        name=str(names.get(class_int, class_int)),
                    ),
                    confidence=float(confidence),
                )
            )
        return tuple(detections)
