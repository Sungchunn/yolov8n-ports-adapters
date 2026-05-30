# Vision Inference Backend

FastAPI service for image and video object detection. The application core uses
ports owned by the service layer, while FastAPI, OpenCV, and YOLO stay at the
outer adapter edge.

Run from the repository root:

```bash
make backend
```

Run checks:

```bash
make test
make lint-imports
make typecheck
```
