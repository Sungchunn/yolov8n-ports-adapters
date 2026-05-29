# Vision Inference API

FastAPI service for object detection with a clean Ports and Adapters / Onion Architecture. The application core runs upload detection through abstract ports; HTTP upload handling, image/video processing, OpenCV, and YOLO stay at the outer adapter edge.

## Project Structure

```text
.
├── pyproject.toml
├── .importlinter
├── src/inference/
│   ├── config.py
│   ├── bootstrap.py
│   ├── domain/
│   │   ├── exceptions.py
│   │   └── model.py
│   ├── service_layer/
│   │   ├── ports.py
│   │   └── services.py
│   ├── adapters/
│   │   ├── inference.py
│   │   └── media.py
│   └── entrypoints/
│       ├── app.py
│       ├── routes.py
│       └── schemas.py
└── tests/
    ├── unit/
    └── integration/
```

## Architecture

- `domain/` contains immutable value objects such as `InferenceResult`, `Detection`, `DetectionLabel`, `BoundingBox`, `ProcessedMedia`, and `MediaFrame`. It imports only standard-library code and owns validation invariants.
- `service_layer/` owns the use cases and the core-owned ports: `AbstractInferenceEngine` and `AbstractMediaProcessor`. It depends on the domain and never imports FastAPI, OpenCV, Ultralytics, or concrete adapters.
- `adapters/` contains concrete infrastructure: `YoloInferenceEngine` and `OpenCvMediaProcessor` for uploaded images and videos.
- `entrypoints/` contains FastAPI routes, request validation, DTO serialization, and exception-to-HTTP mapping.

The core ports are:

```python
AbstractInferenceEngine.predict(image_bytes: bytes) -> InferenceResult
AbstractMediaProcessor.process(media_bytes: bytes, content_type: str) -> ProcessedMedia
```

Image and video uploads are converted into inference-ready image frames by the media processor adapter, then sent through the same one-frame inference port.

## Setup

Install `uv`, then create the environment:

```bash
uv sync --extra dev --extra vision
```

For fast tests without the real model stack, the vision extra is optional:

```bash
uv sync --extra dev
```

Download or place the YOLO weights at the configured path:

```bash
mkdir -p models
curl -L -o models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```

Model weights are intentionally gitignored. The default runtime settings are environment-driven:

```text
MODEL_PATH=models/yolov8n.pt
CONFIDENCE_THRESHOLD=0.25
MAX_UPLOAD_BYTES=20971520
VIDEO_SAMPLE_INTERVAL_SECONDS=1.0
MAX_VIDEO_FRAMES=60
```

## Run

```bash
uv run --extra vision uvicorn inference.entrypoints.app:app --reload
```

### Upload Image Or Video

```bash
curl -X POST http://127.0.0.1:8000/v1/detect/upload \
  -F "file=@sample.jpg;type=image/jpeg"
```

```bash
curl -X POST http://127.0.0.1:8000/v1/detect/upload \
  -F "file=@sample.avi;type=video/x-msvideo"
```

Supported image types are JPEG and PNG. Supported video types are MP4, MOV/QuickTime, AVI, and WebM. Videos are sampled at `VIDEO_SAMPLE_INTERVAL_SECONDS` and capped by `MAX_VIDEO_FRAMES` to keep local CPU usage bounded.

Image responses are wrapped as:

```json
{
  "kind": "image",
  "result": {
    "model": "yolov8n",
    "image": {"width": 640, "height": 480},
    "inference_ms": 12.5,
    "detections": [],
    "created_at": "2026-05-28T00:00:00Z"
  }
}
```

Video responses contain sampled frame results:

```json
{
  "kind": "video",
  "sample_interval_seconds": 1.0,
  "frames": [
    {
      "model": "yolov8n",
      "image": {"width": 640, "height": 480},
      "inference_ms": 12.5,
      "detections": [],
      "created_at": "2026-05-28T00:00:00Z",
      "frame_index": 0,
      "timestamp_seconds": 0.0
    }
  ]
}
```

## Tests And Checks

```bash
uv run --extra dev pytest
uv run --extra dev lint-imports
uv run --extra dev mypy src
```

The normal test suite uses fake `AbstractInferenceEngine` and `AbstractMediaProcessor` implementations, so it is fast and deterministic. Real YOLO smoke tests can be added under `@pytest.mark.slow` without making the default suite depend on model weights.

## AI Orchestration And Guardrails

The implementation was structured from an explicit architecture plan: domain value objects first, then service-layer ports, then concrete adapters, then FastAPI entrypoints. Prompts and review passes were scoped around keeping YOLO/OpenCV outside the core and making every upload mode reuse one-frame inference.

Guardrails used in the codebase:

- `AbstractInferenceEngine` and `AbstractMediaProcessor` live in `service_layer/ports.py`; the service depends on abstractions, not `YoloInferenceEngine` or `OpenCvMediaProcessor`.
- `.importlinter` forbids domain and service-layer imports from FastAPI, Pydantic, OpenCV, Ultralytics, Torch, and adapters.
- Tests inject `FakeInferenceEngine` and `FakeMediaProcessor` through `create_app(...)`, proving the API and service can run without loading model weights or OpenCV.
- Image/video upload handling is an adapter concern. Raw OpenCV frames and YOLO tensors never enter the domain or service layer.
