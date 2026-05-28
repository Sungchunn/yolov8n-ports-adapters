# Vision Inference API

FastAPI service for object detection with a clean Ports and Adapters / Onion Architecture. The application core runs one-frame inference through an abstract port; HTTP upload handling, video sampling, webcam capture, OpenCV, and YOLO stay at the outer adapter edge.

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

- `domain/` contains immutable value objects such as `InferenceResult`, `Detection`, `DetectionLabel`, and `BoundingBox`. It imports only standard-library code and owns validation invariants.
- `service_layer/` owns the use case and the `AbstractInferenceEngine` port. It depends on the domain and never imports FastAPI, OpenCV, Ultralytics, or concrete adapters.
- `adapters/` contains concrete infrastructure: `YoloInferenceEngine` and OpenCV media helpers for uploaded videos and local webcam frames.
- `entrypoints/` contains FastAPI routes, request validation, DTO serialization, exception-to-HTTP mapping, and WebSocket streaming.

The central use case is frame-based:

```python
AbstractInferenceEngine.predict(image_bytes: bytes) -> InferenceResult
```

Image uploads, sampled video frames, and webcam frames are all converted to JPEG bytes at the edge and then sent through the same service-layer use case.

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
WEBCAM_INDEX=0
WEBCAM_FPS_LIMIT=2
```

## Run

```bash
uv run --extra vision uvicorn inference.entrypoints.app:app --reload
```

### JPEG Image

```bash
curl -X POST http://127.0.0.1:8000/v1/detect/image \
  -F "file=@sample.jpg;type=image/jpeg"
```

### Uploaded Video

```bash
curl -X POST http://127.0.0.1:8000/v1/detect/video \
  -F "file=@sample.mp4;type=video/mp4"
```

Videos are sampled at `VIDEO_SAMPLE_INTERVAL_SECONDS` and capped by `MAX_VIDEO_FRAMES` to keep local CPU usage bounded.

### Local Webcam

```text
ws://127.0.0.1:8000/v1/detect/webcam
```

The webcam endpoint uses `cv2.VideoCapture(WEBCAM_INDEX)` on the machine running FastAPI. A hosted backend cannot directly access a user laptop camera without a browser/client frontend that captures and sends frames.

Each WebSocket message contains one processed frame:

```json
{
  "model": "yolov8n",
  "image": {"width": 640, "height": 480},
  "inference_ms": 12.5,
  "detections": [],
  "created_at": "2026-05-28T00:00:00Z",
  "frame_index": 0,
  "timestamp": "2026-05-28T00:00:00Z"
}
```

Public livestream URLs are out of scope for v1 because they are fragile and often have licensing or embedding restrictions.

## Tests And Checks

```bash
uv run --extra dev pytest
uv run --extra dev lint-imports
uv run --extra dev mypy src
```

The normal test suite uses a fake `AbstractInferenceEngine`, so it is fast and deterministic. Real YOLO smoke tests can be added under `@pytest.mark.slow` without making the default suite depend on model weights.

## AI Orchestration And Guardrails

The implementation was structured from an explicit architecture plan: domain value objects first, then the service-layer port, then concrete adapters, then FastAPI entrypoints. Prompts and review passes were scoped around keeping YOLO/OpenCV outside the core and making every input mode reuse one-frame inference.

Guardrails used in the codebase:

- `AbstractInferenceEngine` lives in `service_layer/ports.py`; the service depends on the abstraction, not `YoloInferenceEngine`.
- `.importlinter` forbids domain and service-layer imports from FastAPI, Pydantic, OpenCV, Ultralytics, Torch, and adapters.
- Tests inject `FakeInferenceEngine` through `create_app(engine=...)`, proving the API and service can run without loading model weights.
- Video and webcam handling are adapter/entrypoint concerns. Raw OpenCV frames and YOLO tensors never enter the domain or service layer.
