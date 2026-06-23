#!/usr/bin/env python3
"""Build graph-based long-term memory from segmented drone-part clips.

Pipeline stages:
1. Segment foreground part candidates from each clip keyframe.
2. Use Gemini to convert object features into `[f]` fact and `[r]` reasoning memory.
3. Merge clip memories into a long-term VideoGraph.
4. Render the default two-tab HTML memory view.
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]

from mmagent.prompt import PromptObjectFeature, build_memory_prompt
from mmagent.safety import generate_safety_warnings_with_gemini
from mmagent.utils.general import save_video_graph
from mmagent.videograph import VideoGraph, _extract_refs
from mmagent.visualization import render_memory_html


VIEW_SUFFIX_RE = re.compile(r"-(outside|top|bottom|side)$", re.IGNORECASE)
SHORT_LABEL_PREFIXES = (
    "Lumenier_QAV-S_2_Joshua_Bardwell_Aluminum_",
    "Lumenier_QAV-S_2_Joshua_Bardwell_SE_",
    "Lumenier_QAV-S_2_Joshua_Bardwell_",
)
UNSUPPORTED_VISIBILITY_CHANGE_RE = re.compile(
    r"\b(disappearance|disappears?|no longer visible|missing from|prior locations?|"
    r"selected for|picked up|removed)\b",
    re.IGNORECASE,
)
CURRENT_ASSEMBLY_LABEL = "Current_Assembly"
REJECT_LABEL = "Reject"
DEFAULT_KEYFRAME_RATIOS = ",".join(f"{(i + 0.5) / 15:.6f}" for i in range(15))


@dataclass
class Candidate:
    candidate_id: int
    clip_id: int
    frame_index: int
    timestamp_sec: float
    bbox_xywh: tuple[int, int, int, int]
    area: float
    crop_rgb: np.ndarray
    crop_mask: np.ndarray
    stats: dict[str, float]
    label: str | None = None
    confidence: float = 0.0
    rationale: str = ""

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox_xywh
        return x + w * 0.5, y + h * 0.5

    @property
    def size(self) -> float:
        _, _, w, h = self.bbox_xywh
        return float(max(w, h))


@dataclass
class CandidateCluster:
    members: list[Candidate]

    @property
    def representative(self) -> Candidate:
        return max(self.members, key=lambda c: (c.area, -abs(c.frame_index - 7)))

    @property
    def seen_count(self) -> int:
        return len({c.frame_index for c in self.members})

    @property
    def max_area(self) -> float:
        return max(c.area for c in self.members)

    @property
    def mean_center(self) -> tuple[float, float]:
        centers = np.array([c.center for c in self.members], dtype=np.float32)
        return float(centers[:, 0].mean()), float(centers[:, 1].mean())


@dataclass
class ObjectFeature:
    object_id: int
    token: str
    label: str
    image: Image.Image


@dataclass
class KeyframeFeature:
    frame_index: int
    timestamp_sec: float
    image: Image.Image


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", action="append", type=Path, default=None,
                   help="Clip path. Repeat for multiple clips.")
    p.add_argument("--videos-dir", type=Path, default=None,
                   help="Directory of .mp4 clips; sorted by numeric stem when possible.")
    p.add_argument("--manual", type=Path, default=REPO_ROOT / "docs" / "Manual" / "manual.txt")
    p.add_argument("--parts-dir", type=Path, default=REPO_ROOT / "docs" / "Part Images")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "memory",
    )
    p.add_argument("--keyframe-ratios", default=DEFAULT_KEYFRAME_RATIOS)
    p.add_argument("--max-frame-width", type=int, default=1280)
    p.add_argument(
        "--frame-workers",
        type=int,
        default=int(os.getenv("FRAME_WORKERS", "0")),
        help="Parallel keyframe workers per clip. Use 0 for auto, 1 for serial.",
    )
    p.add_argument("--min-component-area", type=float, default=80.0)
    p.add_argument("--crop-padding", type=int, default=15)
    p.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    p.add_argument(
        "--object-gemini-model",
        default=os.getenv("OBJECT_GEMINI_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model used for VLM-enhanced crop/object recognition.",
    )
    p.add_argument(
        "--safety-gemini-model",
        default=os.getenv("SAFETY_GEMINI_MODEL", "gemini-2.5-flash-lite"),
        help="Faster Gemini model used only for safety-warning generation.",
    )
    p.add_argument(
        "--gemini-embedding-model",
        default=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
    )
    p.add_argument("--reasoning-merge-threshold", type=float, default=0.84)
    return p.parse_args()


def _sort_video_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def _resolve_videos(args: argparse.Namespace) -> list[Path]:
    videos: list[Path] = []
    if args.video:
        videos.extend(args.video)
    if args.videos_dir:
        videos.extend(sorted(args.videos_dir.glob("*.mp4"), key=_sort_video_key))
    if not videos:
        videos = [REPO_ROOT / "0.mp4"]
    missing = [str(p) for p in videos if not p.exists()]
    if missing:
        raise SystemExit(f"missing video(s): {missing}")
    return videos


def _read_manual_labels(manual_path: Path) -> list[str]:
    text = manual_path.read_text()
    seen: set[str] = set()
    labels: list[str] = []
    for label in re.findall(r"\[([^\]]+)\]", text):
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _load_reference_paths(parts_dir: Path, labels: list[str]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {label: [] for label in labels}
    for path in sorted(parts_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label = VIEW_SUFFIX_RE.sub("", path.stem)
        if label in grouped:
            grouped[label].append(path)
    return grouped


def _video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"invalid fps for {video_path}")
    return float(frames) / float(fps)


def _keyframe_timestamps(duration: float, ratios: list[float]) -> list[float]:
    if duration <= 0:
        return []
    return [max(0.05, min(duration - 0.05, duration * r)) for r in ratios]


def _extract_frame(video_path: Path, ts: float, max_width: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(round(ts * fps)) if fps > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise IOError(f"could not decode frame at {ts:.2f}s from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    if w > max_width:
        new_h = int(round(h * max_width / w))
        frame_rgb = cv2.resize(frame_rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(frame_rgb)


def _short_label(label: str | None) -> str | None:
    if label is None:
        return None
    for prefix in SHORT_LABEL_PREFIXES:
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pil_to_jpeg_bytes(img: Image.Image, *, max_width: int = 960) -> bytes:
    rgb = img.convert("RGB")
    if rgb.width > max_width:
        new_h = int(round(rgb.height * max_width / rgb.width))
        rgb = rgb.resize((max_width, new_h), Image.Resampling.LANCZOS)
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=86, optimize=True)
    return buf.getvalue()


def _is_fine_detail_component(area: float, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> bool:
    if area < 70 or area > 2200:
        return False
    if w < 6 or h < 6 or max(w, h) > 95 or w * h > 5200:
        return False
    if x <= 5 or y <= 5 or x + w >= frame_w - 5 or y + h >= frame_h - 5:
        return False
    aspect = max(w / max(h, 1), h / max(w, 1))
    return aspect <= 9.0


def _fine_detail_score(rgb: np.ndarray, mask: np.ndarray, area: float, w: int, h: int) -> float:
    stats = _compute_stats(rgb, mask, area)
    aspect = max(w / max(h, 1), h / max(w, 1))
    dark_bonus = 140.0 if stats["val_mean"] < 130 else 0.0
    purple_bonus = 120.0 if 95 <= stats["hue_mean"] <= 170 and stats["sat_median"] >= 35 else 0.0
    elongated_bonus = 90.0 if aspect >= 1.7 else 0.0
    tiny_bonus = max(0.0, 95.0 - max(w, h))
    return float(area + dark_bonus + purple_bonus + elongated_bonus + tiny_bonus)


def _crop_context(rgb: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = bbox_xywh
    frame_h, frame_w = rgb.shape[:2]
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = int(max(72, min(170, max(w, h) * 3.1)))
    x0 = max(0, int(round(cx - side / 2)))
    y0 = max(0, int(round(cy - side / 2)))
    x1 = min(frame_w, x0 + side)
    y1 = min(frame_h, y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    return Image.fromarray(rgb[y0:y1, x0:x1])


def _build_fine_detail_sheet(
    keyframes: list[KeyframeFeature],
    *,
    max_per_frame: int = 10,
    tile_size: int = 132,
    columns: int = 8,
) -> Image.Image | None:
    tiles: list[tuple[int, int, int, str, Image.Image, float]] = []
    for keyframe in keyframes:
        rgb = np.asarray(keyframe.image.convert("RGB"), dtype=np.uint8)
        mask = _foreground_mask(rgb)
        frame_h, frame_w = rgb.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        selected: list[tuple[float, int, int, str, Image.Image]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            if not _is_fine_detail_component(area, x, y, w, h, frame_w, frame_h):
                continue
            crop_mask = mask[y : y + h, x : x + w]
            crop_rgb = rgb[y : y + h, x : x + w]
            score = _fine_detail_score(crop_rgb, crop_mask, area, w, h)
            label = f"K{keyframe.frame_index} {keyframe.timestamp_sec:.1f}s"
            selected.append((score, y, x, label, _crop_context(rgb, (x, y, w, h))))

        selected.sort(key=lambda item: item[0], reverse=True)
        for score, y, x, label, crop in selected[:max_per_frame]:
            tiles.append((keyframe.frame_index, y, x, label, crop, score))

    if not tiles:
        return None

    tiles.sort(key=lambda item: (item[0], item[1], item[2], -item[5]))
    rows = int(np.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), (250, 248, 252))
    draw = ImageDraw.Draw(sheet)
    for idx, (_, _, _, label, crop, _) in enumerate(tiles):
        col = idx % columns
        row = idx // columns
        x0 = col * tile_size
        y0 = row * tile_size
        thumb = crop.convert("RGB")
        thumb.thumbnail((tile_size - 12, tile_size - 24), Image.Resampling.LANCZOS)
        px = x0 + (tile_size - thumb.width) // 2
        py = y0 + 18 + (tile_size - 24 - thumb.height) // 2
        sheet.paste(thumb, (px, py))
        draw.rectangle((x0, y0, x0 + tile_size - 1, y0 + tile_size - 1), outline=(228, 220, 234))
        draw.text((x0 + 6, y0 + 3), label, fill=(82, 67, 91))
    return sheet


def _build_keyframe_sheet(
    keyframes: list[KeyframeFeature],
    *,
    tile_width: int = 320,
    tile_height: int = 180,
    columns: int = 4,
) -> Image.Image | None:
    if not keyframes:
        return None
    rows = int(np.ceil(len(keyframes) / columns))
    header = 22
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + header)), (250, 248, 252))
    draw = ImageDraw.Draw(sheet)
    for idx, keyframe in enumerate(keyframes):
        col = idx % columns
        row = idx // columns
        x0 = col * tile_width
        y0 = row * (tile_height + header)
        draw.rectangle(
            (x0, y0, x0 + tile_width - 1, y0 + tile_height + header - 1),
            outline=(228, 220, 234),
        )
        draw.text((x0 + 6, y0 + 4), f"K{keyframe.frame_index} {keyframe.timestamp_sec:.1f}s", fill=(82, 67, 91))
        thumb = keyframe.image.convert("RGB")
        thumb.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        px = x0 + (tile_width - thumb.width) // 2
        py = y0 + header + (tile_height - thumb.height) // 2
        sheet.paste(thumb, (px, py))
    return sheet


def _transparent_crop(candidate: Candidate, *, max_side: int = 360) -> Image.Image:
    rgb = np.asarray(candidate.crop_rgb, dtype=np.uint8)
    mask = (np.asarray(candidate.crop_mask) > 0).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    if len(xs) and len(ys):
        object_w = int(xs.max() - xs.min() + 1)
        object_h = int(ys.max() - ys.min() + 1)
        object_max = max(object_w, object_h)
        is_small = object_max < 90 or candidate.area < 1600
        is_medium = object_max < 150 or candidate.area < 4200
        pad = 2
        if is_small:
            pad = max(12, min(30, int(round(object_max * 0.55))))
        elif is_medium:
            pad = max(8, min(22, int(round(object_max * 0.25))))
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
        rgb = rgb[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
        if is_small:
            # Tiny wedge/standoff masks are often partial; showing the local
            # crop context is more useful than a perfectly cut but incomplete
            # silhouette.
            alpha = np.full(mask.shape, 255, dtype=np.uint8)
        else:
            if is_medium:
                k = max(3, min(9, int(round(object_max * 0.08)) | 1))
                kernel = np.ones((k, k), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            alpha = mask.astype(np.uint8) * 255
    else:
        alpha = np.full(mask.shape, 255, dtype=np.uint8)

    rgba = np.dstack([rgb, alpha])
    img = Image.fromarray(rgba, mode="RGBA")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def _representative_candidate(candidates: list[Candidate]) -> Candidate:
    return max(candidates, key=lambda c: (c.confidence, c.area))


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Segment dark carbon, purple aluminum, and black screw-like distractors."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 2] < 165) | ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 240))).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _compute_stats(rgb: np.ndarray, mask: np.ndarray, area: float) -> dict[str, float]:
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    fg = mask > 0
    vals = hsv[fg] if np.any(fg) else hsv.reshape(-1, 3)
    return {
        "w": float(w),
        "h": float(h),
        "aspect_hw": float(h / max(w, 1)),
        "area": float(area),
        "fill_frac": float(np.count_nonzero(fg) / max(w * h, 1)),
        "hue_mean": float(vals[:, 0].mean()),
        "sat_mean": float(vals[:, 1].mean()),
        "sat_median": float(np.median(vals[:, 1])),
        "val_mean": float(vals[:, 2].mean()),
        "val_median": float(np.median(vals[:, 2])),
    }


def _extract_candidates(
    frame_rgb: np.ndarray,
    *,
    clip_id: int,
    frame_index: int,
    timestamp_sec: float,
    min_area: float,
    pad: int,
) -> list[Candidate]:
    mask = _foreground_mask(frame_rgb)
    h_img, w_img = frame_rgb.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        if area < min_area or w < 8 or h < 8:
            continue
        if w * h > 0.20 * w_img * h_img:
            continue
        # Border fragments from the tabletop/video edge are not part nodes.
        if x <= 5 or y <= 5:
            continue
        raw.append((x, y, w, h, area))
    raw.sort(key=lambda b: (b[1], b[0]))

    candidates: list[Candidate] = []
    for idx, (x, y, w, h, area) in enumerate(raw):
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
        crop_rgb = frame_rgb[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        stats = _compute_stats(frame_rgb[y : y + h, x : x + w], mask[y : y + h, x : x + w], area)
        stats["frame_w"] = float(w_img)
        stats["frame_h"] = float(h_img)
        candidates.append(
            Candidate(
                candidate_id=idx,
                clip_id=clip_id,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                bbox_xywh=(x, y, w, h),
                area=area,
                crop_rgb=crop_rgb,
                crop_mask=crop_mask,
                stats=stats,
            )
        )
    return candidates


def _resolve_frame_workers(requested: int, task_count: int) -> int:
    if task_count <= 1:
        return 1
    if requested > 0:
        return max(1, min(requested, task_count))
    cpu_count = os.cpu_count() or 1
    return max(1, min(task_count, cpu_count, 8))


def _process_keyframe_detection(
    video_path: Path,
    clip_id: int,
    frame_index: int,
    timestamp_sec: float,
    *,
    max_frame_width: int,
    min_area: float,
    pad: int,
) -> tuple[KeyframeFeature, list[Candidate]]:
    pil = _extract_frame(video_path, timestamp_sec, max_frame_width)
    keyframe = KeyframeFeature(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        image=pil.convert("RGB"),
    )
    frame_rgb = np.array(keyframe.image)
    candidates = _extract_candidates(
        frame_rgb,
        clip_id=clip_id,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        min_area=min_area,
        pad=pad,
    )
    return keyframe, candidates


def _process_clip_keyframes(
    video_path: Path,
    clip_id: int,
    timestamps: list[float],
    *,
    max_frame_width: int,
    min_area: float,
    pad: int,
    frame_workers: int,
) -> list[tuple[KeyframeFeature, list[Candidate]]]:
    tasks = list(enumerate(timestamps))
    workers = _resolve_frame_workers(frame_workers, len(tasks))
    if workers == 1:
        return [
            _process_keyframe_detection(
                video_path,
                clip_id,
                frame_index,
                timestamp_sec,
                max_frame_width=max_frame_width,
                min_area=min_area,
                pad=pad,
            )
            for frame_index, timestamp_sec in tasks
        ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_keyframe_detection,
                video_path,
                clip_id,
                frame_index,
                timestamp_sec,
                max_frame_width=max_frame_width,
                min_area=min_area,
                pad=pad,
            )
            for frame_index, timestamp_sec in tasks
        ]
        return [future.result() for future in futures]


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _cluster_candidates(candidates: list[Candidate]) -> list[CandidateCluster]:
    clusters: list[CandidateCluster] = []
    for candidate in sorted(candidates, key=lambda c: (c.frame_index, -c.area)):
        cx, cy = candidate.center
        best: CandidateCluster | None = None
        best_dist = float("inf")
        for cluster in clusters:
            rep = cluster.representative
            rx, ry = cluster.mean_center
            dist = float(np.hypot(cx - rx, cy - ry))
            size_gate = max(14.0, 0.35 * min(candidate.size, rep.size))
            area_ratio = candidate.area / max(rep.area, 1.0)
            overlaps = _bbox_iou(candidate.bbox_xywh, rep.bbox_xywh) >= 0.12
            nearly_same_center = dist <= size_gate
            if (overlaps or nearly_same_center) and 0.25 <= area_ratio <= 4.0 and dist < best_dist:
                best = cluster
                best_dist = dist
        if best is None:
            best = CandidateCluster(members=[])
            clusters.append(best)
        best.members.append(candidate)

    kept: list[CandidateCluster] = []
    for cluster in clusters:
        rep = cluster.representative
        _, _, w, h = rep.bbox_xywh
        persistent = cluster.seen_count >= 2
        large = cluster.max_area >= 700
        distinctive_small = cluster.max_area >= 110 and max(w, h) <= 70 and cluster.seen_count >= 2
        if persistent or large or distinctive_small:
            kept.append(cluster)

    kept.sort(key=lambda c: (c.representative.frame_index, c.representative.bbox_xywh[1], c.representative.bbox_xywh[0]))
    return kept[:60]


def _candidate_color_fractions(candidate: Candidate) -> tuple[float, float]:
    hsv = cv2.cvtColor(candidate.crop_rgb, cv2.COLOR_RGB2HSV)
    fg = candidate.crop_mask > 0
    vals = hsv[fg] if np.any(fg) else hsv.reshape(-1, 3)
    purple = float(np.mean((vals[:, 0] >= 105) & (vals[:, 0] <= 165) & (vals[:, 1] >= 35) & (vals[:, 2] >= 60)))
    dark = float(np.mean(vals[:, 2] < 125))
    return purple, dark


def _fit_thumb(image: Image.Image, size: tuple[int, int], bg=(255, 255, 255)) -> Image.Image:
    thumb = image.copy().convert("RGB")
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
    return canvas


def _candidate_as_namespace(candidate: Candidate) -> Any:
    return type(
        "CandidateView",
        (),
        {
            "crop_rgb": candidate.crop_rgb,
            "crop_mask": candidate.crop_mask,
            "area": candidate.area,
        },
    )()


def _build_candidate_sheet(clusters: list[CandidateCluster]) -> tuple[Image.Image, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    crops: list[Image.Image] = []
    for idx, cluster in enumerate(clusters):
        candidate = cluster.representative
        purple, dark = _candidate_color_fractions(candidate)
        crop = _transparent_crop(_candidate_as_namespace(candidate), max_side=360)
        crops.append(crop)
        records.append(
            {
                "candidate": f"<candidate_{idx:02d}>",
                "cluster": cluster,
                "representative": candidate,
                "frame_index": candidate.frame_index,
                "timestamp_sec": round(candidate.timestamp_sec, 3),
                "bbox_xywh": [int(v) for v in candidate.bbox_xywh],
                "area": round(candidate.area, 2),
                "cluster_seen_count": cluster.seen_count,
                "cluster_member_count": len(cluster.members),
                "purple_frac": round(purple, 4),
                "dark_frac": round(dark, 4),
            }
        )

    cols, tile_w, tile_h = 8, 164, 178
    rows = int(np.ceil(len(records) / cols)) or 1
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h + 42), (250, 248, 252))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 13), "candidate_crop_sheet: unlabeled candidate crops", fill=(17, 24, 39))
    for idx, (record, crop) in enumerate(zip(records, crops)):
        col, row = idx % cols, idx // cols
        x0, y0 = col * tile_w, 42 + row * tile_h
        draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + tile_h - 1), outline=(228, 220, 234), fill=(255, 255, 255))
        draw.text((x0 + 7, y0 + 7), record["candidate"], fill=(124, 83, 166))
        draw.text((x0 + 7, y0 + 27), f"K{record['frame_index']:02d}  t={record['timestamp_sec']:.1f}s", fill=(113, 108, 118))
        img = crop.copy().convert("RGBA")
        img.thumbnail((tile_w - 20, tile_h - 62), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (tile_w - 14, tile_h - 58), (255, 255, 255, 255))
        canvas.alpha_composite(img, ((canvas.width - img.width) // 2, (canvas.height - img.height) // 2))
        sheet.paste(canvas.convert("RGB"), (x0 + 7, y0 + 51))
    return sheet, records


def _build_reference_sheet(labels: list[str], refs_paths: dict[str, list[Path]]) -> Image.Image:
    items: list[tuple[str, Path]] = []
    for full_label in labels:
        short = _short_label(full_label) or full_label
        for path in sorted(refs_paths.get(full_label, [])):
            items.append((short, path))

    tile_w, thumb_h, header, cols = 220, 150, 42, 4
    rows = int(np.ceil(len(items) / cols)) or 1
    sheet = Image.new("RGB", (cols * tile_w, rows * (thumb_h + header)), (250, 248, 252))
    draw = ImageDraw.Draw(sheet)
    for idx, (short, path) in enumerate(items):
        col, row = idx % cols, idx // cols
        x0, y0 = col * tile_w, row * (thumb_h + header)
        draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + thumb_h + header - 1), outline=(228, 220, 234), fill=(255, 255, 255))
        draw.text((x0 + 7, y0 + 5), short[:28], fill=(34, 36, 40))
        draw.text((x0 + 7, y0 + 23), path.stem[-28:], fill=(113, 108, 118))
        sheet.paste(_fit_thumb(Image.open(path).convert("RGB"), (tile_w - 14, thumb_h - 8)), (x0 + 7, y0 + header))
    return sheet


def _plate_like_tokens(records: list[dict[str, Any]]) -> list[str]:
    return [
        record["candidate"]
        for record in records
        if record["area"] >= 4500 and record["purple_frac"] < 0.03 and record["dark_frac"] >= 0.75
    ]


def _build_plate_candidate_sheet(records: list[dict[str, Any]], tokens: list[str]) -> Image.Image:
    by_token = {record["candidate"]: record for record in records}
    cell_w, cell_h = 260, 260
    sheet = Image.new("RGB", (max(1, len(tokens)) * cell_w, cell_h), (250, 248, 252))
    draw = ImageDraw.Draw(sheet)
    for idx, token in enumerate(tokens):
        record = by_token[token]
        candidate = record["representative"]
        crop = _transparent_crop(_candidate_as_namespace(candidate), max_side=360)
        x0 = idx * cell_w
        draw.rectangle((x0, 0, x0 + cell_w - 1, cell_h - 1), outline=(228, 220, 234), fill=(255, 255, 255))
        draw.text((x0 + 8, 8), token, fill=(124, 83, 166))
        draw.text((x0 + 8, 27), f"K{record['frame_index']:02d}  t={record['timestamp_sec']:.1f}s", fill=(113, 108, 118))
        crop.thumbnail((cell_w - 32, cell_h - 62), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (cell_w - 16, cell_h - 56), (255, 255, 255, 255))
        canvas.alpha_composite(crop.convert("RGBA"), ((canvas.width - crop.width) // 2, (canvas.height - crop.height) // 2))
        sheet.paste(canvas.convert("RGB"), (x0 + 8, 48))
    return sheet


def _object_label_prompt(records: list[dict[str, Any]], closed_labels: list[str]) -> str:
    tokens = [record["candidate"] for record in records]
    plate_like = _plate_like_tokens(records)
    small_components = [record["candidate"] for record in records if record["area"] <= 500]
    possible_assemblies = [
        record["candidate"]
        for record in records
        if record["area"] >= 1200 and record["purple_frac"] >= 0.08 and record["dark_frac"] >= 0.18
    ]
    return f"""You are verifying object labels for a drone assembly video clip.

You receive exactly two images:
1. candidate_crop_sheet: unlabeled candidate crops, each identified as <candidate_XX>.
2. reference_part_sheet: the 8 valid standalone part classes with reference images and labels.

Assign every candidate token below to exactly one label from the closed set.

Candidates:
{json.dumps(tokens, ensure_ascii=False)}

Closed label set:
{json.dumps(closed_labels, ensure_ascii=False)}

Candidate visual hints computed from crop geometry/color; these are not labels:
- Large black/carbon plate-like candidates: {json.dumps(plate_like, ensure_ascii=False)}. First choose among Top_Plate, Split_Front_Plate, and Split_Rear_Plate by silhouette and hole layout.
- Small loose-component candidates: {json.dumps(small_components, ensure_ascii=False)}. Prefer Arm_Wedge_5mm, Knurled_Standoff, or Reject unless an inserted/fastened connection is clearly visible.
- Purple-plus-dark large candidates that may be assemblies: {json.dumps(possible_assemblies, ensure_ascii=False)}.

Visual criteria:
- Top_Plate: long, narrow black/carbon plate with a slim body, row-like central openings, and a fork/U-shaped end.
- Split_Front_Plate: tapered black/carbon split plate with two lower rounded feet/lobes and fewer paired circular holes.
- Split_Rear_Plate: broader, more symmetric black/carbon split plate with a wide U/notch and many paired circular screw holes.
- 5_inch_Arm: long single black arm, usually with a large round/hex motor hole near one end.
- X-Lock: purple cross/X shaped aluminum plate.
- FPV_Camera_Mounts: small purple camera bracket pieces with circular/rounded central cutout.
- Arm_Wedge_5mm: tiny purple/black aluminum wedge; in top view it can look like a purple crescent.
- Knurled_Standoff: tiny black/dark cylindrical round post or peg, often visible from top as a small black circle.
- Current_Assembly: multiple parts clearly physically connected, inserted, fastened, or overlapping as an assembled subassembly.

Important rules:
- Do not use Split_Rear_Plate as a generic label for all black plates.
- Use Current_Assembly only when there is a clear physical connection; loose nearby parts on the table are not Current_Assembly.
- Use Reject only for shadows, hands/tools, background, unknown objects, duplicated slivers, or isolated fragments with no visible part-part connection.
- Return strict JSON only. Include every candidate exactly once.

Output schema:
{{"assignments": [{{"candidate": "<candidate_00>", "label": "one of the closed-set labels"}}]}}
"""


def _parse_label_response(raw: str, expected_tokens: set[str], closed_labels: set[str]) -> dict[str, str]:
    text = _strip_code_fence(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("assignments"), list):
        raise ValueError("Gemini label response must contain an assignments list.")
    out: dict[str, str] = {}
    for item in parsed["assignments"]:
        if not isinstance(item, dict):
            raise ValueError("assignment items must be objects.")
        token = str(item.get("candidate", "")).strip()
        label = str(item.get("label", "")).strip()
        if token not in expected_tokens or label not in closed_labels:
            raise ValueError(f"invalid label assignment: {item!r}")
        out[token] = label
    if set(out) != expected_tokens:
        raise ValueError("label response did not include every candidate exactly once.")
    return out


def _call_label_gemini(
    *,
    prompt: str,
    images: list[Image.Image],
    model: str,
    expected_tokens: set[str],
    closed_labels: set[str],
    max_retries: int = 2,
) -> dict[str, str]:
    client, types = _get_gemini_client()
    contents: list[Any] = [prompt]
    for index, image in enumerate(images):
        contents.append(f"image_{index}")
        contents.append(types.Part.from_bytes(data=_pil_to_jpeg_bytes(image, max_width=1280), mime_type="image/jpeg"))

    last_raw = ""
    for attempt in range(max_retries):
        config_kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "top_p": 0.1,
            "seed": 1 + attempt,
            "response_mime_type": "application/json",
        }
        if hasattr(types, "ThinkingConfig"):
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        last_raw = getattr(response, "text", "") or ""
        try:
            return _parse_label_response(last_raw, expected_tokens, closed_labels)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            continue
    raise RuntimeError(f"Gemini did not return valid object labels. Last response: {last_raw[:500]!r}")


def _refine_plate_labels(
    records: list[dict[str, Any]],
    reference_sheet: Image.Image,
    *,
    model: str,
) -> dict[str, str]:
    tokens = _plate_like_tokens(records)
    if not tokens:
        return {}
    prompt = f"""Classify only these large black/carbon plate-like candidates.

You receive two images:
1. plate_candidate_sheet: enlarged crops for the candidate tokens below.
2. reference_part_sheet: reference images for all valid parts.

Candidate tokens:
{json.dumps(tokens, ensure_ascii=False)}

Allowed labels:
["Top_Plate", "Split_Front_Plate", "Split_Rear_Plate", "Reject"]

Compare silhouette and hole layout carefully. Mentally rotate candidates if needed.
- Top_Plate: long and narrow, slim body, row of central openings, fork/U-shaped end.
- Split_Front_Plate: tapered split plate with two lower rounded feet/lobes and fewer paired circular holes.
- Split_Rear_Plate: broader and more symmetric split plate with a wide U/notch and many paired circular screw holes.

Return strict JSON only:
{{"assignments": [{{"candidate": "<candidate_00>", "label": "one of the allowed labels"}}]}}
"""
    return _call_label_gemini(
        prompt=prompt,
        images=[_build_plate_candidate_sheet(records, tokens), reference_sheet],
        model=model,
        expected_tokens=set(tokens),
        closed_labels={"Top_Plate", "Split_Front_Plate", "Split_Rear_Plate", REJECT_LABEL},
    )


def _apply_label_guards(assignments: dict[str, str], records: list[dict[str, Any]]) -> dict[str, str]:
    corrected = dict(assignments)
    by_token = {record["candidate"]: record for record in records}
    for token, label in list(corrected.items()):
        record = by_token[token]
        _, _, w, h = record["bbox_xywh"]
        if (
            label in {REJECT_LABEL, "Knurled_Standoff", "Arm_Wedge_5mm"}
            and 90 <= record["area"] <= 500
            and w / max(h, 1) >= 1.2
            and record["purple_frac"] >= 0.18
            and record["dark_frac"] >= 0.45
        ):
            corrected[token] = CURRENT_ASSEMBLY_LABEL
            continue
        if label == "Arm_Wedge_5mm" and record["purple_frac"] < 0.08 and record["dark_frac"] > 0.12:
            corrected[token] = "Knurled_Standoff"
    return corrected


def _assign_clip_object_labels_with_gemini(
    candidates: list[Candidate],
    labels: list[str],
    refs_paths: dict[str, list[Path]],
    *,
    model: str,
) -> None:
    clusters = _cluster_candidates(candidates)
    if not clusters:
        return
    candidate_sheet, records = _build_candidate_sheet(clusters)
    reference_sheet = _build_reference_sheet(labels, refs_paths)
    short_to_full = {_short_label(label) or label: label for label in labels}
    closed_labels = list(short_to_full) + [CURRENT_ASSEMBLY_LABEL, REJECT_LABEL]

    assignments = _call_label_gemini(
        prompt=_object_label_prompt(records, closed_labels),
        images=[candidate_sheet, reference_sheet],
        model=model,
        expected_tokens={record["candidate"] for record in records},
        closed_labels=set(closed_labels),
    )
    assignments.update(_refine_plate_labels(records, reference_sheet, model=model))
    assignments = _apply_label_guards(assignments, records)

    for record in records:
        label = assignments[record["candidate"]]
        if label == REJECT_LABEL:
            full_label = None
            confidence = 0.0
        elif label == CURRENT_ASSEMBLY_LABEL:
            full_label = CURRENT_ASSEMBLY_LABEL
            confidence = 0.96
        else:
            full_label = short_to_full.get(label)
            confidence = 0.95 if full_label else 0.0
        for candidate in record["cluster"].members:
            candidate.label = full_label
            candidate.confidence = confidence
            candidate.rationale = f"Gemini crop label: {label}"


def _build_graph(
    accepted: list[Candidate],
    labels: list[str],
) -> tuple[VideoGraph, dict[str, Any], dict[str, int]]:
    graph = VideoGraph(max_object_embeddings=1, max_object_crops=8, object_matching_threshold=1.0)
    zero = np.zeros((1,), dtype=np.float32)

    by_label: dict[str, list[Candidate]] = {}
    for c in accepted:
        assert c.label is not None
        by_label.setdefault(c.label, []).append(c)

    label_to_oid: dict[str, int] = {}
    for full_label in labels:
        group = by_label.get(full_label, [])
        if not group:
            continue
        label = _short_label(full_label) or full_label
        crop = _transparent_crop(_representative_candidate(group))
        contents_by_clip: dict[str, list[str]] = {}
        for clip_id in sorted({c.clip_id for c in group}):
            clip_group = [c for c in group if c.clip_id == clip_id]
            if clip_group:
                clip_crop = _transparent_crop(_representative_candidate(clip_group))
                contents_by_clip[str(clip_id)] = [_pil_to_b64_png(clip_crop)]
        first_clip = min(c.clip_id for c in group)
        last_clip = max(c.clip_id for c in group)
        oid = graph.add_object_node(
            {
                "embeddings": [zero],
                "contents": [_pil_to_b64_png(crop)],
                "name": label,
                "first_clip": first_clip,
                "last_clip": last_clip,
                "seen_count": len(group),
            }
        )
        graph.nodes[oid].metadata.update(
            {
                "label": label,
                "full_label": full_label,
                "source": "vlm_enhanced_crop_part_images",
                "contents_by_clip": contents_by_clip,
            }
        )
        for clip_id in sorted({c.clip_id for c in group}):
            if oid not in graph.object_nodes_by_clip[clip_id]:
                graph.object_nodes_by_clip[clip_id].append(oid)
        label_to_oid[full_label] = oid

    stats = {
        "labels_observed": sorted(_short_label(label) for label in label_to_oid.keys()),
        "labels_missing": [_short_label(label) for label in labels if label not in label_to_oid],
        "accepted_count": len(accepted),
        "object_nodes": len(graph.object_nodes),
    }
    return graph, stats, label_to_oid


def _add_current_assembly_node(
    graph: VideoGraph,
    keyframes_by_clip: dict[int, list[KeyframeFeature]],
    accepted: list[Candidate],
) -> int | None:
    clip_ids = sorted(keyframes_by_clip)
    if not clip_ids:
        return None
    zero = np.zeros((1,), dtype=np.float32)
    assembly_candidates = [c for c in accepted if c.label == CURRENT_ASSEMBLY_LABEL]
    contents_by_clip: dict[str, list[str]] = {}
    for clip_id in clip_ids:
        clip_assembly = [c for c in assembly_candidates if c.clip_id == clip_id]
        if clip_assembly:
            image = _transparent_crop(_representative_candidate(clip_assembly))
            contents_by_clip[str(clip_id)] = [_pil_to_b64_png(image)]
    representative = None
    if contents_by_clip:
        representative_b64 = contents_by_clip[str(max(int(cid) for cid in contents_by_clip))][-1]
        representative = Image.open(BytesIO(base64.b64decode(representative_b64))).convert("RGBA")
    elif assembly_candidates:
        representative = _transparent_crop(_representative_candidate(assembly_candidates))
    if representative is None:
        representative = _build_keyframe_sheet(keyframes_by_clip[clip_ids[-1]])
    if representative is None:
        representative = Image.new("RGB", (320, 180), (250, 248, 252))
    oid = graph.add_object_node(
        {
            "embeddings": [zero],
            "contents": [_pil_to_b64_png(representative)],
            "name": CURRENT_ASSEMBLY_LABEL,
            "first_clip": clip_ids[0],
            "last_clip": clip_ids[-1],
            "seen_count": len(clip_ids),
        }
    )
    graph.nodes[oid].metadata.update(
        {
            "label": CURRENT_ASSEMBLY_LABEL,
            "full_label": CURRENT_ASSEMBLY_LABEL,
            "source": "vlm_enhanced_current_assembly_crop_or_keyframe_state",
            "contents_by_clip": contents_by_clip,
        }
    )
    for clip_id in clip_ids:
        if oid not in graph.object_nodes_by_clip[clip_id]:
            graph.object_nodes_by_clip[clip_id].append(oid)
    return oid


def _collect_clip_object_features(
    graph: VideoGraph,
    accepted: list[Candidate],
    label_to_oid: dict[str, int],
    clip_id: int,
    *,
    current_assembly_oid: int | None = None,
    keyframes: list[KeyframeFeature] | None = None,
) -> list[ObjectFeature]:
    by_oid: dict[int, list[Candidate]] = {}
    for candidate in accepted:
        if candidate.clip_id != clip_id or candidate.label is None:
            continue
        oid = label_to_oid.get(candidate.label)
        if oid is None:
            continue
        by_oid.setdefault(oid, []).append(candidate)

    features: list[ObjectFeature] = []
    for oid in sorted(by_oid):
        node = graph.nodes[oid]
        label = str(node.metadata.get("label") or node.metadata.get("name") or f"object_{oid}")
        crop = _transparent_crop(_representative_candidate(by_oid[oid]))
        features.append(
            ObjectFeature(
                object_id=oid,
                token=f"<object_{oid}>",
                label=label,
                image=crop,
            )
        )
    if current_assembly_oid is not None:
        sheet = _build_keyframe_sheet(keyframes or [])
        if sheet is not None:
            features.append(
                ObjectFeature(
                    object_id=current_assembly_oid,
                    token=f"<object_{current_assembly_oid}>",
                    label=CURRENT_ASSEMBLY_LABEL,
                    image=sheet,
                )
            )
    return features


def _get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY to generate prompt-based memory.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install google-genai to use Gemini memory generation.") from exc
    return genai.Client(api_key=api_key), types


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_memory_response(raw: str, allowed_tokens: set[str]) -> list[str]:
    text = _strip_code_fence(raw)
    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        raise ValueError("Gemini memory response must be a dict.")
    raw_lines = parsed.get("memory")
    if not isinstance(raw_lines, list):
        raise ValueError("Gemini memory response must contain a list key named 'memory'.")

    lines: list[str] = []
    for item in raw_lines:
        line = str(item).strip()
        if not line.startswith(("[f]", "[r]")):
            continue
        if UNSUPPORTED_VISIBILITY_CHANGE_RE.search(line):
            continue
        refs = set(_extract_refs(line))
        if refs - allowed_tokens:
            continue
        lines.append(line)
    return lines


def _generate_memory_lines_with_gemini(
    keyframes: list[KeyframeFeature],
    object_features: list[ObjectFeature],
    *,
    model: str,
    max_retries: int = 3,
) -> list[str]:
    if not object_features:
        return []

    client, types = _get_gemini_client()
    prompt_features = [
        PromptObjectFeature(token=f.token, label=f.label)
        for f in object_features
    ]
    prompt = build_memory_prompt(prompt_features)
    allowed_tokens = {f"object_{f.object_id}" for f in object_features}

    contents: list[Any] = [
        prompt,
    ]
    for keyframe in keyframes:
        contents.append(f"Keyframe {keyframe.frame_index} at {keyframe.timestamp_sec:.2f}s")
        contents.append(
            types.Part.from_bytes(
                data=_pil_to_jpeg_bytes(keyframe.image),
                mime_type="image/jpeg",
            )
        )
    detail_sheet = _build_fine_detail_sheet(keyframes)
    if detail_sheet is not None:
        contents.append(
            "Fine-detail evidence sheet: chronological high-resolution crops of small foreground "
            "components and local connection regions from the same keyframes. These crops are visual "
            "evidence only, not additional object IDs."
        )
        contents.append(
            types.Part.from_bytes(
                data=_pil_to_jpeg_bytes(detail_sheet, max_width=1280),
                mime_type="image/jpeg",
            )
        )
    for feature in object_features:
        contents.append(f"Object feature: {feature.token}")
        contents.append(types.Part.from_bytes(data=_pil_to_png_bytes(feature.image), mime_type="image/png"))

    last_raw = ""
    for _ in range(max_retries):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.1,
                seed=1,
                response_mime_type="application/json",
            ),
        )
        last_raw = getattr(response, "text", "") or ""
        try:
            lines = _parse_memory_response(last_raw, allowed_tokens)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            continue
        if lines:
            return lines
    raise RuntimeError(f"Gemini did not return valid memory lines. Last response: {last_raw[:500]!r}")


def _fallback_text_embeddings(texts: list[str], dims: int = 256) -> list[np.ndarray]:
    embeddings: list[np.ndarray] = []
    for text in texts:
        vec = np.zeros((dims,), dtype=np.float32)
        for token in re.findall(r"[a-zA-Z0-9_<>-]+", text.lower()):
            vec[hash(token) % dims] += 1.0
        norm = float(np.linalg.norm(vec))
        embeddings.append(vec / norm if norm else vec)
    return embeddings


def _embed_memory_lines_with_gemini(texts: list[str], *, model: str) -> list[np.ndarray]:
    if not texts:
        return []
    try:
        client, _ = _get_gemini_client()
        result = client.models.embed_content(model=model, contents=texts)
        embeddings = getattr(result, "embeddings", None) or []
        out = [
            np.asarray(getattr(embedding, "values"), dtype=np.float32)
            for embedding in embeddings
        ]
        if len(out) == len(texts):
            return out
    except Exception:
        pass
    return _fallback_text_embeddings(texts)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _insert_memory_node(
    graph: VideoGraph,
    clip_id: int,
    text: str,
    embedding: np.ndarray,
    *,
    time_range: dict[str, float] | None = None,
    timestamp: dict[str, float | str] | None = None,
) -> int:
    tid = graph.add_text_node("memory", clip_id, text, embedding)
    graph.nodes[tid].metadata["task_status"] = None
    if time_range is not None:
        graph.nodes[tid].metadata["time_range"] = dict(time_range)
        graph.nodes[tid].metadata["time_ranges"] = [dict(time_range)]
    if timestamp is not None:
        graph.nodes[tid].metadata["timestamp"] = dict(timestamp)
    for ref in _extract_refs(text):
        try:
            oid = int(ref.split("_")[1])
        except (IndexError, ValueError):
            continue
        node = graph.nodes.get(oid)
        if node is not None and node.type == "object":
            graph.add_edge(tid, oid, relation="mention", weight=1)
    return tid


def _memory_timestamp_for_line(
    line_index: int,
    line_count: int,
    time_range: dict[str, float] | None,
) -> dict[str, float | str] | None:
    if not time_range or line_count <= 0:
        return None
    start = float(time_range.get("clip_start_sec", 0.0))
    end = float(time_range.get("clip_end_sec", start))
    duration = max(0.0, end - start)
    clip_ts = ((line_index + 0.5) / line_count) * duration if duration else 0.0
    global_ts = start + clip_ts
    return {
        "clip_time_sec": round(clip_ts, 3),
        "global_time_sec": round(global_ts, 3),
        "label": f"{global_ts:.1f}s",
    }


def _find_matching_reasoning_node(
    graph: VideoGraph,
    text: str,
    embedding: np.ndarray,
    threshold: float,
) -> tuple[int | None, float]:
    refs = set(_extract_refs(text))
    if not refs:
        return None, 0.0

    best_id: int | None = None
    best_sim = -1.0
    for tid in graph.text_nodes:
        node = graph.nodes[tid]
        if node.type != "memory":
            continue
        existing_text = node.metadata["contents"][-1]
        if not existing_text.startswith("[r]"):
            continue
        if set(_extract_refs(existing_text)) != refs:
            continue
        existing_embedding = node.metadata.get("embedding")
        if existing_embedding is None:
            continue
        sim = _cosine(embedding, np.asarray(existing_embedding, dtype=np.float32))
        if sim > best_sim:
            best_id = tid
            best_sim = sim

    if best_id is not None and best_sim >= threshold:
        return best_id, best_sim
    return None, best_sim


def _write_memory_lines_to_graph(
    graph: VideoGraph,
    clip_id: int,
    lines: list[str],
    embeddings: list[np.ndarray],
    *,
    reasoning_merge_threshold: float,
    time_range: dict[str, float] | None = None,
) -> dict[str, int]:
    counts = {"facts": 0, "reasoning": 0, "reinforced_reasoning": 0}
    line_count = min(len(lines), len(embeddings))
    for line_index, (text, embedding) in enumerate(zip(lines, embeddings)):
        timestamp = _memory_timestamp_for_line(line_index, line_count, time_range)
        if text.startswith("[r]"):
            match_id, _ = _find_matching_reasoning_node(
                graph,
                text,
                embedding,
                reasoning_merge_threshold,
            )
            if match_id is not None:
                graph.reinforce_text_node(match_id, new_content=text)
                if time_range is not None:
                    md = graph.nodes[match_id].metadata
                    md.setdefault("time_range", dict(time_range))
                    md.setdefault("time_ranges", []).append(dict(time_range))
                if timestamp is not None:
                    md = graph.nodes[match_id].metadata
                    md.setdefault("timestamp", dict(timestamp))
                    md.setdefault("timestamps", []).append(dict(timestamp))
                counts["reinforced_reasoning"] += 1
            else:
                _insert_memory_node(graph, clip_id, text, embedding, time_range=time_range, timestamp=timestamp)
                counts["reasoning"] += 1
        else:
            _insert_memory_node(graph, clip_id, text, embedding, time_range=time_range, timestamp=timestamp)
            counts["facts"] += 1
    return counts


def main() -> None:
    total_start = time.perf_counter()
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, Any] = {"clips": {}}

    setup_start = time.perf_counter()
    labels = _read_manual_labels(args.manual)
    refs_paths = _load_reference_paths(args.parts_dir, labels)
    missing_refs = [label for label, paths in refs_paths.items() if not paths]
    if missing_refs:
        raise SystemExit(f"missing part reference images for labels: {missing_refs}")

    video_paths = _resolve_videos(args)
    ratios = [float(x) for x in args.keyframe_ratios.split(",") if x.strip()]
    timings["setup_sec"] = round(time.perf_counter() - setup_start, 3)

    all_candidates: list[Candidate] = []
    keyframes_by_clip: dict[int, list[KeyframeFeature]] = {}
    clip_time_ranges: dict[int, dict[str, float]] = {}
    elapsed_sec = 0.0
    for clip_id, video_path in enumerate(video_paths):
        clip_timing: dict[str, float] = {}
        clip_start = time.perf_counter()
        duration = _video_duration(video_path)
        clip_time_ranges[clip_id] = {
            "clip_start_sec": round(elapsed_sec, 3),
            "clip_end_sec": round(elapsed_sec + duration, 3),
        }
        elapsed_sec += duration
        timestamps = _keyframe_timestamps(duration, ratios)
        detect_start = time.perf_counter()
        results = _process_clip_keyframes(
            video_path,
            clip_id,
            timestamps,
            max_frame_width=args.max_frame_width,
            min_area=args.min_component_area,
            pad=args.crop_padding,
            frame_workers=args.frame_workers,
        )
        clip_timing["keyframe_crop_sec"] = round(time.perf_counter() - detect_start, 3)
        clip_candidates: list[Candidate] = []
        for keyframe, candidates in results:
            keyframes_by_clip.setdefault(clip_id, []).append(keyframe)
            clip_candidates.extend(candidates)
            all_candidates.extend(candidates)
        object_start = time.perf_counter()
        _assign_clip_object_labels_with_gemini(
            clip_candidates,
            labels,
            refs_paths,
            model=args.object_gemini_model,
        )
        clip_timing["object_recognition_sec"] = round(time.perf_counter() - object_start, 3)
        clip_timing["clip_total_sec"] = round(time.perf_counter() - clip_start, 3)
        clip_timing["raw_candidates"] = float(len(clip_candidates))
        keyframes_by_clip[clip_id].sort(key=lambda keyframe: keyframe.frame_index)
        timings["clips"][str(clip_id)] = clip_timing

    graph_start = time.perf_counter()
    accepted = [c for c in all_candidates if c.label is not None]
    graph, stats, label_to_oid = _build_graph(accepted, labels)
    current_assembly_oid = _add_current_assembly_node(graph, keyframes_by_clip, accepted)
    timings["graph_build_sec"] = round(time.perf_counter() - graph_start, 3)

    memory_counts = {"facts": 0, "reasoning": 0, "reinforced_reasoning": 0}
    safety_warning_count = 0
    for clip_id, video_path in enumerate(video_paths):
        clip_timing = timings["clips"].setdefault(str(clip_id), {})
        clip_keyframes = keyframes_by_clip.get(clip_id, [])
        feature_start = time.perf_counter()
        object_features = _collect_clip_object_features(
            graph,
            accepted,
            label_to_oid,
            clip_id,
            current_assembly_oid=current_assembly_oid,
            keyframes=clip_keyframes,
        )
        clip_timing["object_feature_sec"] = round(time.perf_counter() - feature_start, 3)
        memory_start = time.perf_counter()
        lines = _generate_memory_lines_with_gemini(
            clip_keyframes,
            object_features,
            model=args.gemini_model,
        )
        clip_timing["memory_generation_sec"] = round(time.perf_counter() - memory_start, 3)
        embed_start = time.perf_counter()
        embeddings = _embed_memory_lines_with_gemini(lines, model=args.gemini_embedding_model)
        clip_timing["embedding_sec"] = round(time.perf_counter() - embed_start, 3)
        write_start = time.perf_counter()
        counts = _write_memory_lines_to_graph(
            graph,
            clip_id,
            lines,
            embeddings,
            reasoning_merge_threshold=args.reasoning_merge_threshold,
            time_range=clip_time_ranges.get(clip_id),
        )
        clip_timing["memory_graph_write_sec"] = round(time.perf_counter() - write_start, 3)
        for key, value in counts.items():
            memory_counts[key] += value
        safety_start = time.perf_counter()
        safety_warnings = generate_safety_warnings_with_gemini(
            clip_keyframes,
            model=args.safety_gemini_model,
        )
        clip_timing["safety_generation_sec"] = round(time.perf_counter() - safety_start, 3)
        graph.safety_warnings_by_clip[clip_id] = safety_warnings
        safety_warning_count += len(safety_warnings)

    save_start = time.perf_counter()
    graph_path = args.out_dir / "graph.pkl"
    save_video_graph(graph, str(graph_path))
    timings["save_graph_sec"] = round(time.perf_counter() - save_start, 3)

    html_start = time.perf_counter()
    html_path = args.out_dir / "grounded_memory.html"
    render_memory_html(graph, html_path)
    timings["render_html_sec"] = round(time.perf_counter() - html_start, 3)
    timings["total_sec"] = round(time.perf_counter() - total_start, 3)

    report = {
        "out_dir": str(args.out_dir),
        "html": str(html_path),
        "graph": str(graph_path),
        "labels_observed": stats["labels_observed"],
        "labels_missing": stats["labels_missing"],
        "accepted_count": stats["accepted_count"],
        "object_nodes": len(graph.object_nodes),
        "current_assembly_node": current_assembly_oid,
        "memory_nodes": len(graph.text_nodes),
        "facts": memory_counts["facts"],
        "reasoning": memory_counts["reasoning"],
        "reinforced_reasoning": memory_counts["reinforced_reasoning"],
        "safety_warnings": safety_warning_count,
        "timings": timings,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
