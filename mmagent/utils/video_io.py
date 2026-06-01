"""Minimal video IO helpers (no audio). Uses decord for fast random-access."""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def get_clip_duration(clip_path: str) -> float:
    """Return clip duration in seconds using OpenCV (no moviepy/ffmpeg spawn)."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {clip_path}")
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"invalid fps for {clip_path}")
    return float(frame_count) / float(fps)


def extract_frames_at_timestamps(clip_path: str, timestamps_sec: list[float]) -> list[np.ndarray]:
    """Return a list of RGB uint8 ndarrays sampled at the given timestamps (seconds).

    Clamps requested times to [0, duration - 1/fps]; raises if a frame fails to decode.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {clip_path}")
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0.0
    cap.release()

    # Re-open with decord for reliable seeking (OpenCV seek on compressed streams
    # can snap to the nearest keyframe and miss requested frames).
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(clip_path)
    fps = float(vr.get_avg_fps())
    duration = len(vr) / fps

    out = []
    for t in timestamps_sec:
        t_clamped = max(0.0, min(t, max(duration - 1.0 / fps, 0.0)))
        frame_idx = int(round(t_clamped * fps))
        frame_idx = min(frame_idx, len(vr) - 1)
        frame = vr[frame_idx].asnumpy()  # HxWx3 uint8 RGB
        out.append(frame)
    return out


def sample_uniform_frames(clip_path: str, fps: float = 2.0, max_frames: int = 64) -> list[np.ndarray]:
    """Uniformly sample RGB frames at `fps` from clip. Caps at `max_frames`."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(clip_path)
    src_fps = float(vr.get_avg_fps())
    duration = len(vr) / src_fps if src_fps > 0 else 0.0
    n = max(1, min(int(round(duration * fps)), max_frames))
    if n == 1:
        idxs = [len(vr) // 2]
    else:
        idxs = [int(round(i * (len(vr) - 1) / (n - 1))) for i in range(n)]
    return [vr[i].asnumpy() for i in idxs]


def encode_rgb_to_base64_png(frame_rgb: np.ndarray) -> str:
    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def read_video_base64(clip_path: str) -> str:
    with open(clip_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def save_rgb_png(frame_rgb: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_rgb).save(path, format="PNG")
