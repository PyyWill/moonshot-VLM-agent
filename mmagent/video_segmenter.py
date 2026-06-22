#!/usr/bin/env python3
"""Split one input video into fixed-duration clips for memory construction."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True, help="Input video path.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "clips",
        help="Directory to write segmented clips and manifest.json.",
    )
    p.add_argument("--clip-seconds", type=float, default=10.0)
    p.add_argument(
        "--prefix-seconds",
        type=float,
        default=None,
        help="Optionally segment only the first N seconds.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def segment_video(
    video_path: Path,
    out_dir: Path,
    *,
    clip_seconds: float = 10.0,
    prefix_seconds: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if clip_seconds <= 0:
        raise ValueError("--clip-seconds must be positive")
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"invalid video metadata for {video_path}")

    frames_per_clip = max(1, int(round(clip_seconds * fps)))
    max_frames = total_frames
    if prefix_seconds is not None and prefix_seconds > 0:
        max_frames = min(total_frames, int(round(prefix_seconds * fps)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None
    writer_path: Path | None = None
    clip_index = -1
    frame_index = 0
    frames_in_clip = 0
    clips: list[dict[str, Any]] = []

    def close_writer() -> None:
        nonlocal writer, writer_path, frames_in_clip, clip_index
        if writer is None or writer_path is None or frames_in_clip == 0:
            return
        writer.release()
        start_sec = (clip_index * frames_per_clip) / fps
        duration_sec = frames_in_clip / fps
        clips.append(
            {
                "clip_id": clip_index,
                "path": str(writer_path),
                "start_sec": start_sec,
                "end_sec": start_sec + duration_sec,
                "duration_sec": duration_sec,
                "frames": frames_in_clip,
            }
        )
        writer = None
        writer_path = None
        frames_in_clip = 0

    while frame_index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        next_clip_index = frame_index // frames_per_clip
        if writer is None or next_clip_index != clip_index:
            close_writer()
            clip_index = next_clip_index
            writer_path = out_dir / f"clip_{clip_index:03d}.mp4"
            writer = cv2.VideoWriter(str(writer_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise IOError(f"cannot write {writer_path}")
        writer.write(frame)
        frames_in_clip += 1
        frame_index += 1

    close_writer()
    cap.release()

    if not clips:
        raise RuntimeError(f"no clips written from {video_path}")

    manifest = {
        "source_video": str(video_path),
        "out_dir": str(out_dir),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames_read": frame_index,
        "clip_seconds": clip_seconds,
        "prefix_seconds": prefix_seconds,
        "clips": clips,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = segment_video(
        args.video,
        args.out_dir,
        clip_seconds=args.clip_seconds,
        prefix_seconds=args.prefix_seconds,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
