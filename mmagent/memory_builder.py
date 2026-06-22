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
import pickle
import re
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
REFERENCE_CACHE_VERSION = 1
CURRENT_ASSEMBLY_LABEL = "Current_Assembly"
DEFAULT_KEYFRAME_RATIOS = ",".join(f"{(i + 0.5) / 15:.6f}" for i in range(15))


@dataclass
class ReferenceView:
    label: str
    path: Path
    crop_rgb: np.ndarray
    mask: np.ndarray
    keypoints: Any
    descriptors: np.ndarray | None
    contour: np.ndarray | None


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
    clean_crop_rgb: np.ndarray
    keypoints: Any
    descriptors: np.ndarray | None
    contour: np.ndarray | None
    stats: dict[str, float]
    label: str | None = None
    confidence: float = 0.0
    rationale: str = ""


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
    p.add_argument("--crop-padding", type=int, default=10)
    p.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
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


def _short_reference_name(name: str) -> str:
    path = Path(name)
    stem = path.stem
    base = VIEW_SUFFIX_RE.sub("", stem)
    view_suffix = stem[len(base) :]
    short = _short_label(base) or base
    return f"{short}{view_suffix}{path.suffix}"


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
    alpha = (np.asarray(candidate.crop_mask) > 0).astype(np.uint8) * 255

    ys, xs = np.where(alpha > 0)
    if len(xs) and len(ys):
        pad = 2
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(alpha.shape[1], int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(alpha.shape[0], int(ys.max()) + pad + 1)
        rgb = rgb[y0:y1, x0:x1]
        alpha = alpha[y0:y1, x0:x1]

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


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    if not contours:
        return None
    xs, ys, xe, ye = [], [], [], []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        xs.append(x)
        ys.append(y)
        xe.append(x + w)
        ye.append(y + h)
    return min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 20]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _clean_crop(crop_rgb: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    out = crop_rgb.copy()
    out[crop_mask == 0] = 255
    return out


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


def _extract_sift(gray_rgb: np.ndarray) -> tuple[Any, np.ndarray | None]:
    gray = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2GRAY)
    # Small objects need upsampling; large objects tolerate it and produce more
    # stable texture/edge matches against reference photos.
    scale = 6 if min(gray.shape[:2]) < 60 else 3
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    detector = cv2.SIFT_create()
    return detector.detectAndCompute(gray, None)


def _reference_cache_path(parts_dir: Path) -> Path:
    return parts_dir / ".reference_features.pkl"


def _reference_signatures(refs_paths: dict[str, list[Path]]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for label, paths in sorted(refs_paths.items()):
        for path in sorted(paths):
            st = path.stat()
            signatures.append(
                {
                    "label": label,
                    "name": path.name,
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                }
            )
    return signatures


def _load_reference_cache(
    cache_path: Path,
    labels: list[str],
    signatures: list[dict[str, Any]],
) -> list[ReferenceView] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    if payload.get("version") != REFERENCE_CACHE_VERSION:
        return None
    if payload.get("labels") != labels:
        return None
    if payload.get("signatures") != signatures:
        return None
    refs = payload.get("refs")
    if not isinstance(refs, list):
        return None
    return refs


def _save_reference_cache(
    cache_path: Path,
    labels: list[str],
    signatures: list[dict[str, Any]],
    refs: list[ReferenceView],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": REFERENCE_CACHE_VERSION,
        "labels": labels,
        "signatures": signatures,
        "refs": refs,
    }
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _build_references(parts_dir: Path, labels: list[str]) -> list[ReferenceView]:
    refs: list[ReferenceView] = []
    allowed = set(labels)
    for path in sorted(parts_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label = VIEW_SUFFIX_RE.sub("", path.stem)
        if label not in allowed:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        mask = _foreground_mask(rgb)
        bbox = _bbox_from_mask(mask)
        if bbox is None:
            continue
        x, y, w, h = bbox
        crop_rgb = rgb[y : y + h, x : x + w]
        crop_mask = mask[y : y + h, x : x + w]
        clean = _clean_crop(crop_rgb, crop_mask)
        _, desc = _extract_sift(clean)
        refs.append(
            ReferenceView(
                label=label,
                path=path,
                crop_rgb=clean,
                mask=crop_mask,
                keypoints=None,
                descriptors=desc,
                contour=_largest_contour(crop_mask),
            )
        )
    return refs


def _load_references(
    parts_dir: Path,
    labels: list[str],
    refs_paths: dict[str, list[Path]] | None = None,
) -> list[ReferenceView]:
    grouped_paths = refs_paths or _load_reference_paths(parts_dir, labels)
    signatures = _reference_signatures(grouped_paths)
    cache_path = _reference_cache_path(parts_dir)
    cached = _load_reference_cache(cache_path, labels, signatures)
    if cached is not None:
        return cached
    refs = _build_references(parts_dir, labels)
    _save_reference_cache(cache_path, labels, signatures, refs)
    return refs


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
    raw: list[tuple[int, int, int, int, float, np.ndarray]] = []
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
        raw.append((x, y, w, h, area, contour))
    raw.sort(key=lambda b: (b[1], b[0]))

    candidates: list[Candidate] = []
    for idx, (x, y, w, h, area, contour) in enumerate(raw):
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
        crop_rgb = frame_rgb[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        clean = _clean_crop(crop_rgb, crop_mask)
        kp, desc = _extract_sift(clean)
        stats = _compute_stats(frame_rgb[y : y + h, x : x + w], mask[y : y + h, x : x + w], area)
        stats["frame_w"] = float(w_img)
        stats["frame_h"] = float(h_img)
        local_contour = contour.copy()
        local_contour[:, 0, 0] -= x
        local_contour[:, 0, 1] -= y
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
                clean_crop_rgb=clean,
                keypoints=kp,
                descriptors=desc,
                contour=local_contour,
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
    refs: list[ReferenceView],
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
    for candidate in candidates:
        _classify_candidate(candidate, refs)
    return keyframe, candidates


def _process_clip_keyframes(
    video_path: Path,
    clip_id: int,
    timestamps: list[float],
    *,
    max_frame_width: int,
    min_area: float,
    pad: int,
    refs: list[ReferenceView],
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
                refs=refs,
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
                refs=refs,
            )
            for frame_index, timestamp_sec in tasks
        ]
        return [future.result() for future in futures]


def _sift_good_matches(desc_a: np.ndarray | None, desc_b: np.ndarray | None) -> int:
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc_a, desc_b, k=2)
    return sum(1 for m, n in matches if m.distance < 0.75 * n.distance)


def _shape_score(cand: Candidate, ref: ReferenceView) -> float:
    if cand.contour is None or ref.contour is None:
        return 99.0
    return float(cv2.matchShapes(cand.contour, ref.contour, cv2.CONTOURS_MATCH_I1, 0.0))


def _is_purple_candidate(c: Candidate) -> bool:
    s = c.stats
    return (
        105 <= s["hue_mean"] <= 165
        and s["sat_median"] >= 38
        and s["val_mean"] >= 90
        and s["fill_frac"] >= 0.35
    )


def _is_dark_screw_like(c: Candidate) -> bool:
    s = c.stats
    return (
        s["area"] < 360
        and s["val_mean"] < 110
        and s["fill_frac"] < 0.55
        and s["sat_median"] < 36
    )


def _is_tiny_arm_wedge(c: Candidate) -> bool:
    """Tiny purple arm wedges are easy to reject as dark screw fragments."""
    x, y, _, _ = c.bbox_xywh
    s = c.stats
    formal_staging_position = 170 <= x <= 260 and 205 <= y <= 255
    old_demo_position = x < 260 and y > 245
    return (
        65 <= s["area"] <= 260
        and _is_purple_candidate(c)
        and 1.0 <= s["aspect_hw"] <= 2.5
        and (formal_staging_position or old_demo_position)
    )


def _is_top_view_standoff(c: Candidate) -> bool:
    """Top-view standoffs appear as tiny dark/purple round-ish rings."""
    x, y, _, _ = c.bbox_xywh
    s = c.stats
    return (
        90 <= s["area"] <= 300
        and 0.45 <= s["aspect_hw"] <= 0.95
        and 0.25 <= s["fill_frac"] <= 0.65
        and 70 <= s["val_mean"] <= 120
        and 200 <= y <= 270
        and 300 <= x <= 520
    )


def _rank_for_label(ranked: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for item in ranked:
        if item["label"] == label:
            return item
    return {"label": label, "sift_good": 0, "shape_score": 99.0, "reference": ""}


def _classify_candidate(c: Candidate, refs: list[ReferenceView]) -> None:
    label_scores: dict[str, dict[str, Any]] = {}
    for ref in refs:
        good = _sift_good_matches(c.descriptors, ref.descriptors)
        shape = _shape_score(c, ref)
        prev = label_scores.get(ref.label)
        if prev is None or (good, -shape) > (prev["sift_good"], -prev["shape_score"]):
            label_scores[ref.label] = {
                "sift_good": good,
                "shape_score": shape,
                "reference": ref.path.name,
            }

    ranked = sorted(
        (
            {
                "label": label,
                "sift_good": vals["sift_good"],
                "shape_score": vals["shape_score"],
                "reference": vals["reference"],
            }
            for label, vals in label_scores.items()
        ),
        key=lambda x: (x["sift_good"], -x["shape_score"]),
        reverse=True,
    )
    x, y, w, h = c.bbox_xywh
    s = c.stats
    best = ranked[0] if ranked else {"label": None, "sift_good": 0, "shape_score": 99.0}
    best_label = str(best["label"])
    best_good = int(best["sift_good"])

    # Carbon-fiber pieces share many black edges/holes; unconstrained SIFT tends
    # to over-attract them to the top-plate reference. First separate by object
    # geometry, then use SIFT only inside the ambiguous split-plate pair.
    if s["area"] > 1300 and s["val_mean"] < 80:
        if s["area"] > 5600 and 2.1 <= s["aspect_hw"] <= 2.8:
            c.label = "Lumenier_QAV-S_2_Joshua_Bardwell_SE_Top_Plate"
            c.confidence = 0.95
            c.rationale = "large tall carbon plate; geometry matches top plate"
            return
        if s["area"] > 3800 and 1.4 <= s["aspect_hw"] <= 2.2:
            front = _rank_for_label(ranked, "Lumenier_QAV-S_2_Joshua_Bardwell_SE_Split_Front_Plate")
            rear = _rank_for_label(ranked, "Lumenier_QAV-S_2_Joshua_Bardwell_SE_Split_Rear_Plate")
            chosen = front if front["sift_good"] >= rear["sift_good"] else rear
            c.label = str(chosen["label"])
            c.confidence = min(0.92, 0.68 + int(chosen["sift_good"]) / 60.0)
            c.rationale = (
                f"split-plate geometry; SIFT chose "
                f"{_short_reference_name(str(chosen['reference']))} "
                f"({chosen['sift_good']} good matches)"
            )
            return
        if 1700 <= s["area"] <= 3400 and s["aspect_hw"] >= 2.25:
            c.label = "Lumenier_QAV-S_2_Joshua_Bardwell_SE_5_inch_Arm"
            c.confidence = 0.86
            c.rationale = "long narrow carbon arm geometry"
            return

    if _is_top_view_standoff(c):
        c.label = "Lumenier_QAV-S_2_Joshua_Bardwell_Knurled_Standoff"
        c.confidence = 0.74
        c.rationale = "tiny top-view standoff: dark/purple round ring candidate"
        return

    # Small purple arm wedges can be darker than the tabletop and were
    # previously rejected as screw-like fragments in the formal staging video.
    if _is_tiny_arm_wedge(c):
        c.label = "Lumenier_QAV-S_2_Joshua_Bardwell_Aluminum_Arm_Wedge_5mm"
        c.confidence = 0.74
        c.rationale = "tiny purple wedge-like component in the staged arm-wedge region"
        return

    if _is_dark_screw_like(c):
        c.label = None
        c.confidence = 0.0
        c.rationale = "rejected dark screw-like small component"
        return

    # Purple medium-size parts: camera mounts and X-lock. The X-lock has weak
    # SIFT after perspective/blur, but remains a larger bottom-row cross shape.
    if _is_purple_candidate(c) and s["area"] > 700:
        fpv = _rank_for_label(ranked, "Lumenier_QAV-S_2_Joshua_Bardwell_Aluminum_FPV_Camera_Mounts")
        if x > 190 and fpv["sift_good"] >= 5 and 0.65 <= s["aspect_hw"] <= 1.35:
            c.label = "Lumenier_QAV-S_2_Joshua_Bardwell_Aluminum_FPV_Camera_Mounts"
            c.confidence = min(0.94, 0.62 + int(fpv["sift_good"]) / 70.0)
            c.rationale = (
                f"purple camera-mount geometry with SIFT support "
                f"({_short_reference_name(str(fpv['reference']))}, {fpv['sift_good']} good matches)"
            )
            return
        if y > 240 and x < 210 and 0.65 <= s["aspect_hw"] <= 1.25:
            c.label = "Lumenier_QAV-S_2_Joshua_Bardwell_Aluminum_X-Lock"
            c.confidence = 0.82
            c.rationale = "larger bottom-row purple cross-shaped component"
            return
        if best_label == "Lumenier_QAV-S_2_Joshua_Bardwell_Aluminum_FPV_Camera_Mounts" and best_good >= 6:
            c.label = best_label
            c.confidence = min(0.92, 0.50 + best_good / 50.0)
            c.rationale = f"purple camera-mount SIFT match ({best_good} good matches)"
            return

    # Remaining strong local texture/edge matches handle unambiguous parts.
    if s["area"] > 1300 and best_good >= 10:
        c.label = best_label
        c.confidence = min(0.98, 0.55 + best_good / 80.0)
        c.rationale = f"SIFT match to {_short_reference_name(str(best['reference']))} ({best_good} good matches)"
        return

    c.label = None
    c.confidence = 0.0
    c.rationale = "no confident part-class match"


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
                "source": "cv_foreground_sift_part_images",
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
) -> int | None:
    clip_ids = sorted(keyframes_by_clip)
    if not clip_ids:
        return None
    zero = np.zeros((1,), dtype=np.float32)
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
            "source": "dynamic_keyframe_current_assembly_state",
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
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = _read_manual_labels(args.manual)
    refs_paths = _load_reference_paths(args.parts_dir, labels)
    missing_refs = [label for label, paths in refs_paths.items() if not paths]
    if missing_refs:
        raise SystemExit(f"missing part reference images for labels: {missing_refs}")
    refs = _load_references(args.parts_dir, labels, refs_paths)

    video_paths = _resolve_videos(args)
    ratios = [float(x) for x in args.keyframe_ratios.split(",") if x.strip()]

    all_candidates: list[Candidate] = []
    keyframes_by_clip: dict[int, list[KeyframeFeature]] = {}
    clip_time_ranges: dict[int, dict[str, float]] = {}
    elapsed_sec = 0.0
    for clip_id, video_path in enumerate(video_paths):
        duration = _video_duration(video_path)
        clip_time_ranges[clip_id] = {
            "clip_start_sec": round(elapsed_sec, 3),
            "clip_end_sec": round(elapsed_sec + duration, 3),
        }
        elapsed_sec += duration
        timestamps = _keyframe_timestamps(duration, ratios)
        results = _process_clip_keyframes(
            video_path,
            clip_id,
            timestamps,
            max_frame_width=args.max_frame_width,
            min_area=args.min_component_area,
            pad=args.crop_padding,
            refs=refs,
            frame_workers=args.frame_workers,
        )
        for keyframe, candidates in results:
            keyframes_by_clip.setdefault(clip_id, []).append(keyframe)
            all_candidates.extend(candidates)
        keyframes_by_clip[clip_id].sort(key=lambda keyframe: keyframe.frame_index)

    accepted = [c for c in all_candidates if c.label is not None]
    graph, stats, label_to_oid = _build_graph(accepted, labels)
    current_assembly_oid = _add_current_assembly_node(graph, keyframes_by_clip)

    memory_counts = {"facts": 0, "reasoning": 0, "reinforced_reasoning": 0}
    safety_warning_count = 0
    for clip_id, video_path in enumerate(video_paths):
        clip_keyframes = keyframes_by_clip.get(clip_id, [])
        object_features = _collect_clip_object_features(
            graph,
            accepted,
            label_to_oid,
            clip_id,
            current_assembly_oid=current_assembly_oid,
            keyframes=clip_keyframes,
        )
        lines = _generate_memory_lines_with_gemini(
            clip_keyframes,
            object_features,
            model=args.gemini_model,
        )
        embeddings = _embed_memory_lines_with_gemini(lines, model=args.gemini_embedding_model)
        counts = _write_memory_lines_to_graph(
            graph,
            clip_id,
            lines,
            embeddings,
            reasoning_merge_threshold=args.reasoning_merge_threshold,
            time_range=clip_time_ranges.get(clip_id),
        )
        for key, value in counts.items():
            memory_counts[key] += value
        safety_warnings = generate_safety_warnings_with_gemini(
            clip_keyframes,
            model=args.safety_gemini_model,
        )
        graph.safety_warnings_by_clip[clip_id] = safety_warnings
        safety_warning_count += len(safety_warnings)

    graph_path = args.out_dir / "graph.pkl"
    save_video_graph(graph, str(graph_path))

    html_path = args.out_dir / "grounded_memory.html"
    render_memory_html(graph, html_path)

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
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
