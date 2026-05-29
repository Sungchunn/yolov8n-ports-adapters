# Upload-Only Detection Preview With Media Processing Port

## Summary

Refactor the repository to expose one upload-only preview endpoint, `POST /v1/detect/upload`, while preserving strict Ports and
Adapters / Hexagonal Architecture. The application core will own the contracts it needs: one port for inference and one new port
for media processing. Concrete image/video parsing, `OpenCV` usage, and upload MIME handling stay outside the core.

## Architecture

- Keep `AbstractInferenceEngine` in `service_layer/ports.py`:
  `predict(image_bytes: bytes) -> InferenceResult`

- Add `AbstractMediaProcessor` in `service_layer/ports.py`:
  `process(media_bytes: bytes, content_type: str) -> ProcessedMedia`

- Add framework-free media value objects, preferably in `domain/model.py` or a small domain media module:
  - `MediaKind`: enum with `image` and `video`
  - `MediaFrame`: immutable frame metadata plus `image_bytes`
  - `ProcessedMedia`: immutable collection of frames and media kind

- The service layer coordinates both ports:
  - Media bytes enter the service.
  - The service asks `AbstractMediaProcessor` for inference-ready frames.
  - The service runs each frame through `AbstractInferenceEngine`.
  - The service returns an image or video detection result for serialization.

- Implement the concrete media processor in `adapters/media.py`, using existing `OpenCV`/video sampling helpers.

## Public API

- Remove public webcam support entirely.
- Replace image/video-specific routes with one official endpoint:
  `POST /v1/detect/upload`

- Accepted still image content types:
  - `image/jpeg`
  - `image/jpg`
  - `image/png`

- Accepted video content types:
  - `video/mp4`
  - `video/quicktime`
  - `video/x-msvideo`
  - `video/webm`

- Response shape:
  - Image upload:
    `{"kind": "image", "result": InferenceResponse}`

  - Video upload:
    `{"kind": "video", "sample_interval_seconds": float, "frames": list[VideoFrameResponse]}`

- Remove `/v1/detect/image`, `/v1/detect/video`, legacy hidden `/v1/detect`, and `/v1/detect/webcam`.

## Implementation Changes

- Update `config.py`:
  - Keep `MAX_UPLOAD_BYTES`, `VIDEO_SAMPLE_INTERVAL_SECONDS`, and `MAX_VIDEO_FRAMES`.
  - Remove `WEBCAM_INDEX` and `WEBCAM_FPS_LIMIT`.

- Update `bootstrap.py`:
  - Instantiate both `YoloInferenceEngine` and `OpenCvMediaProcessor`.
  - Return or expose both dependencies for app startup injection.

- Update `entrypoints/app.py`:
  - Store both ports on `app.state`, or use a small dependency container object.
  - Preserve test injection for fake inference and fake media processor.

- Update `entrypoints/routes.py`:
  - Keep route code thin: read upload, enforce size limit, call the service, serialize the response.
  - Do not import `OpenCV`-specific logic into the route.

- Update `adapters/media.py`:
  - Implement `OpenCvMediaProcessor`.
  - Validate `JPEG` and `PNG` magic bytes.
  - Treat `AVI` as `video/x-msvideo`.
  - Convert image/video inputs into `MediaFrame` objects whose bytes are suitable for `AbstractInferenceEngine.predict`.

- Update `entrypoints/schemas.py`:
  - Add upload response DTOs for the `kind: image | video` union.
  - Reuse existing `InferenceResponse`, `VideoFrameResponse`, and mapping helpers where possible.

- Update docs:
  - `README` should describe upload-only preview support.
  - Architecture docs should show `AbstractMediaProcessor` as a core-owned driven port.
  - Remove webcam setup, settings, `WebSocket` payloads, and hosted-backend camera notes.

- Unsupported media type maps to `415`.
- Empty upload maps to `400`.
- Oversized upload maps to `413`.
- Corrupt or undecodable image/video maps to `422`.
- Inference/runtime failures map to `503`.
- HTTP status decisions remain in entrypoints; domain/service/adapters raise framework-free exceptions.

## Test Plan

- Service unit tests:
  - Fake media processor returns one image frame and causes one inference call.
  - Fake media processor returns multiple video frames and causes one inference call per frame.
  - Unsupported/invalid media errors propagate without `FastAPI` imports.

- Adapter unit tests:
  - `JPEG` validation succeeds.
  - `PNG` validation succeeds.
  - Bad image magic bytes fail.
  - `AVI` MIME type dispatches to video sampling.
  - Invalid sample interval and max frame settings still fail cleanly.

- Integration tests:
  - `POST /v1/detect/upload` accepts `JPEG`.
  - `POST /v1/detect/upload` accepts `PNG`.
  - `POST /v1/detect/upload` accepts `video/x-msvideo`.
  - Unsupported MIME type returns `415`.
  - Empty upload returns `400`.
  - Oversized upload returns `413`.

- Remove webcam `WebSocket` tests.
- Run:
  - `uv run --extra dev pytest`
  - `uv run --extra dev lint-imports`
  - `uv run --extra dev mypy src`

## Assumptions

- “AVI image” means an uploaded `AVI` media file, handled as `video/x-msvideo`.
- `AbstractMediaProcessor` is the chosen port name because it fits the existing `AbstractInferenceEngine` naming style.
- The core owns the media processing interface, but concrete image/video decoding remains an adapter concern.
- The single upload endpoint is the only supported preview API after this refactor.
