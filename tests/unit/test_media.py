from __future__ import annotations

from typing import Any

from inference.adapters.media import sample_video_capture


class FakeCapture:
    def __init__(self, frames: list[str]) -> None:
        self.frames = frames
        self.read_count = 0

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, Any]:
        if self.read_count >= len(self.frames):
            return False, None
        frame = self.frames[self.read_count]
        self.read_count += 1
        return True, frame

    def get(self, _prop_id: int) -> float:
        return 0

    def release(self) -> None:
        return None


def test_sample_video_capture_uses_interval_and_max_frames() -> None:
    capture = FakeCapture(["a", "b", "c", "d", "e", "f"])

    frames = list(
        sample_video_capture(
            capture,
            sample_interval_seconds=0.5,
            max_frames=2,
            encode_frame=lambda frame: f"jpeg-{frame}".encode(),
            fps=4,
        )
    )

    assert [frame.frame_index for frame in frames] == [0, 2]
    assert [frame.timestamp_seconds for frame in frames] == [0, 0.5]
    assert [frame.image_bytes for frame in frames] == [b"jpeg-a", b"jpeg-c"]
