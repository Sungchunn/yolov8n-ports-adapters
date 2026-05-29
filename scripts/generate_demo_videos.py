from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "demo" / "videos"
WIDTH = 640
HEIGHT = 360
FPS = 24
DURATION_SECONDS = 6
FRAME_COUNT = FPS * DURATION_SECONDS


PALETTE = {
    "grass": (122, 172, 109),
    "road": (79, 86, 88),
    "lane": (230, 233, 225),
    "shadow": (50, 54, 56),
    "blue": (37, 99, 235),
    "teal": (20, 184, 166),
    "red": (239, 68, 68),
    "yellow": (250, 204, 21),
    "white": (245, 245, 245),
    "orange": (249, 115, 22),
    "purple": (168, 85, 247),
    "green": (34, 197, 94),
    "pink": (236, 72, 153),
}


def fill_rect(
    frame: bytearray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(WIDTH, x + width)
    y1 = min(HEIGHT, y + height)
    if x0 >= x1 or y0 >= y1:
        return

    row = bytes(color) * (x1 - x0)
    for yy in range(y0, y1):
        start = (yy * WIDTH + x0) * 3
        frame[start : start + len(row)] = row


def draw_background(frame: bytearray) -> None:
    fill_rect(frame, 0, 0, WIDTH, HEIGHT, PALETTE["grass"])
    fill_rect(frame, 0, 90, WIDTH, 180, PALETTE["road"])
    fill_rect(frame, 0, 148, WIDTH, 4, PALETTE["lane"])
    fill_rect(frame, 0, 208, WIDTH, 4, PALETTE["lane"])

    for x in range(0, WIDTH, 80):
        fill_rect(frame, x + 18, 178, 42, 4, PALETTE["lane"])


def draw_vehicle(
    frame: bytearray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    fill_rect(frame, x + 4, y + height - 4, width, 5, PALETTE["shadow"])
    fill_rect(frame, x, y + 6, width, height - 8, color)
    fill_rect(frame, x + 12, y, width - 24, 10, color)
    fill_rect(frame, x + 10, y + height - 3, 12, 5, PALETTE["shadow"])
    fill_rect(frame, x + width - 22, y + height - 3, 12, 5, PALETTE["shadow"])


def vehicle_x(frame_index: int, start: float, speed: float, left_to_right: bool) -> int:
    progress = frame_index / (FRAME_COUNT - 1)
    x = start + speed * progress
    if not left_to_right:
        x = WIDTH - x
    return int(round(x))


def render_clip(
    output_path: Path,
    vehicles: list[tuple[float, float, bool, int, tuple[int, int, int]]],
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for frame_index in range(FRAME_COUNT):
            frame = bytearray(PALETTE["grass"] * WIDTH * HEIGHT)
            draw_background(frame)

            for start, speed, left_to_right, lane_y, color in vehicles:
                x = vehicle_x(frame_index, start, speed, left_to_right)
                bob = int(math.sin(frame_index / FPS * math.tau) * 1.5)
                draw_vehicle(frame, x, lane_y + bob, 74, 30, color)

            frame_path = temp_dir / f"frame-{frame_index:04d}.ppm"
            frame_path.write_bytes(
                f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii") + bytes(frame)
            )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(FPS),
                "-i",
                str(temp_dir / "frame-%04d.ppm"),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = {
        "synthetic-light.mp4": [
            (-90, 760, True, 108, PALETTE["blue"]),
            (-160, 720, True, 228, PALETTE["red"]),
            (-80, 680, False, 168, PALETTE["yellow"]),
        ],
        "synthetic-medium.mp4": [
            (-90, 760, True, 108, PALETTE["blue"]),
            (-260, 740, True, 108, PALETTE["teal"]),
            (-120, 720, False, 168, PALETTE["yellow"]),
            (-320, 700, False, 168, PALETTE["white"]),
            (-150, 700, True, 228, PALETTE["red"]),
            (-360, 690, True, 228, PALETTE["purple"]),
        ],
        "synthetic-heavy.mp4": [
            (-80, 460, True, 108, PALETTE["blue"]),
            (70, 430, True, 108, PALETTE["teal"]),
            (220, 400, True, 108, PALETTE["red"]),
            (-70, 450, False, 168, PALETTE["yellow"]),
            (80, 420, False, 168, PALETTE["white"]),
            (230, 390, False, 168, PALETTE["orange"]),
            (-90, 430, True, 228, PALETTE["purple"]),
            (60, 400, True, 228, PALETTE["green"]),
            (210, 370, True, 228, PALETTE["pink"]),
        ],
    }

    for file_name, vehicles in clips.items():
        render_clip(OUT_DIR / file_name, vehicles)


if __name__ == "__main__":
    main()
