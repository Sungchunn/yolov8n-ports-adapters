from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "assets" / "video"
DEFAULT_OUT_DIR = ROOT / "frontend" / "public" / "assets" / "demo" / "videos"
DEFAULT_THUMBNAIL_DIR = (
    ROOT / "frontend" / "public" / "assets" / "demo" / "thumbnails"
)
DEFAULT_MANIFEST = ROOT / "frontend" / "public" / "assets" / "demo" / "videos.json"
DEFAULT_COUNT = 10
DEFAULT_SEED = 20260530


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample TrafficDB AVI clips, convert them to browser-friendly "
            "MP4 files, and update the frontend demo manifest."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory containing source AVI files. Default: {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for generated MP4 files. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Manifest JSON path to write. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--thumbnail-dir",
        type=Path,
        default=DEFAULT_THUMBNAIL_DIR,
        help=f"Directory for generated poster JPEG files. Default: {DEFAULT_THUMBNAIL_DIR}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of AVI clips to sample. Default: {DEFAULT_COUNT}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Random seed used for reproducible sampling. Change this to rotate "
            f"clips. Default: {DEFAULT_SEED}"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="trafficdb-demo",
        help="Output filename and manifest id prefix. Default: trafficdb-demo",
    )
    return parser.parse_args()


def sample_avi_files(source_dir: Path, count: int, seed: int) -> list[Path]:
    avi_files = sorted(source_dir.glob("*.avi"))
    if not avi_files:
        raise FileNotFoundError(f"No AVI files found in {source_dir}")
    if count <= 0:
        raise ValueError("--count must be positive")
    if count > len(avi_files):
        raise ValueError(
            f"Requested {count} clips, but only found {len(avi_files)} AVI files"
        )

    return random.Random(seed).sample(avi_files, count)


def convert_to_mp4(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "fatal",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


def write_thumbnail(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "fatal",
            "-ss",
            "0.75",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "scale=160:120",
            "-q:v",
            "4",
            "-update",
            "1",
            str(destination),
        ],
        check=True,
    )


def build_manifest_entry(prefix: str, index: int, source: Path) -> dict[str, str]:
    demo_id = f"{prefix}-{index:02d}"
    return {
        "id": demo_id,
        "label": f"TrafficDB sample {index:02d} ({source.stem})",
        "src": f"/assets/demo/videos/{demo_id}.mp4",
        "thumbnailSrc": f"/assets/demo/thumbnails/{demo_id}.jpg",
        "contentType": "video/mp4",
    }


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert AVI files to MP4")

    args = parse_args()
    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    thumbnail_dir = args.thumbnail_dir.resolve()
    manifest_path = args.manifest.resolve()

    selected = sample_avi_files(source_dir, args.count, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    for index, source in enumerate(selected, start=1):
        destination = out_dir / f"{args.prefix}-{index:02d}.mp4"
        thumbnail = thumbnail_dir / f"{args.prefix}-{index:02d}.jpg"
        convert_to_mp4(source, destination)
        write_thumbnail(destination, thumbnail)
        manifest.append(build_manifest_entry(args.prefix, index, source))
        print(
            f"{source.name} -> {display_path(destination)}, {display_path(thumbnail)}"
        )

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {display_path(manifest_path)} with {len(manifest)} demos")


if __name__ == "__main__":
    main()
