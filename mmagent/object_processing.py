"""
Object detection and cross-clip ReID for Moonshot VLM Agent.

Per clip:
  1. Sample K keyframes.
  2. SAM2.1 AutomaticMaskGenerator on each keyframe.
  3. Filter by area / stability.
  4. Extract masked crops, white-background (avoids VLM confusing black
     fill with black carbon-fiber drone arms).
  5. DINOv2 embed each crop.
  6. Intra-clip dedupe via cosine-sim clustering.
  7. For each cluster representative: cross-clip match against existing
     object nodes in the VideoGraph (cosine-sim), update or add.

Outputs:
  - Mutates the VideoGraph (object nodes).
  - Returns a per-clip debug record whose clusters include the local cluster id
    and matched global object id.
  - Optionally dumps a JSON record with crops/bboxes for debugging.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from .utils import dinov2_wrapper, sam2_wrapper, video_io
from .videograph import VideoGraph

logger = logging.getLogger(__name__)


def _filter_masks(
    masks: list[dict[str, Any]],
    frame_hw: tuple[int, int],
    *,
    min_area: int = 500,
    max_area_ratio: float = 0.70,
    min_stability: float = 0.92,
) -> list[dict[str, Any]]:
    H, W = frame_hw
    max_area = int(max_area_ratio * H * W)
    kept = []
    for m in masks:
        if m["area"] < min_area:
            continue
        if m["area"] > max_area:
            continue
        if m.get("stability_score", 1.0) < min_stability:
            continue
        kept.append(m)
    return kept


def _masked_crop(frame_rgb: np.ndarray, mask: np.ndarray, bbox: tuple, pad: int = 8) -> np.ndarray:
    """Return a tight crop of frame_rgb where mask==False is filled white.

    White fill (not black) because black fill gets confused with black
    carbon-fiber drone arms by the downstream VLM.
    """
    x, y, w, h = [int(v) for v in bbox]
    H, W = frame_rgb.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    crop = frame_rgb[y0:y1, x0:x1].copy()
    cmask = mask[y0:y1, x0:x1]
    crop[~cmask] = 255
    return crop


def _bbox_iou(b1: tuple, b2: tuple) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _intra_clip_dedupe(
    embeddings: np.ndarray,
    bboxes: list[tuple] | None = None,
    *,
    sim_threshold: float = 0.93,
    spatial_sim_floor: float = 0.85,
    iou_threshold: float = 0.30,
) -> np.ndarray:
    """Greedy single-link clustering with optional spatial prior.

    Merge candidate j into seed i's cluster if either:
      - dino_sim(i, j) > sim_threshold  (strict appearance match), OR
      - dino_sim(i, j) > spatial_sim_floor AND bbox_iou(i, j) > iou_threshold
        (spatial-assisted match: same physical object sitting at near-stable
        position across keyframes, even with pose/crop variation).

    The spatial floor prevents pure-geometric false merges (two unrelated
    objects briefly co-located); the IoU gate prevents pure-appearance false
    merges (two visually similar but spatially distinct parts, e.g. mirror
    carbon arms).
    """
    n = len(embeddings)
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    cluster_ids = -np.ones(n, dtype=np.int32)
    next_cid = 0
    for i in range(n):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_cid
        sims = embeddings[i + 1 :] @ embeddings[i]
        for j_offset, s in enumerate(sims):
            j = i + 1 + j_offset
            if cluster_ids[j] != -1:
                continue
            s = float(s)
            if s > sim_threshold:
                cluster_ids[j] = next_cid
                continue
            if bboxes is not None and s > spatial_sim_floor:
                if _bbox_iou(bboxes[i], bboxes[j]) > iou_threshold:
                    cluster_ids[j] = next_cid
        next_cid += 1
    return cluster_ids


def process_objects_for_clip(
    graph: VideoGraph,
    clip_path: str,
    clip_id: int,
    *,
    keyframe_timestamps_sec: list[float],
    memory_config: dict,
    debug_dir: str | None = None,
) -> dict:
    """Run full object pipeline on one clip, mutate `graph`, return a debug record.

    debug record structure:
        {
          "clip_id": int,
          "clip_path": str,
          "duration": float,
          "n_masks_raw": int,
          "n_masks_kept": int,
          "n_clusters": int,
          "clusters": [
            {
              "local_id": int, "global_object_id": int, "is_new": bool,
              "rep_keyframe_idx": int, "bbox": [x,y,w,h], "area": int,
              "stability_score": float, "matched_sim": float | None,
              "crop_path": str | None,  # when debug_dir is set
            },
            ...
          ]
        }
    """
    duration = video_io.get_clip_duration(clip_path)

    # 1) keyframes
    ts = [min(t, max(duration - 0.1, 0.0)) for t in keyframe_timestamps_sec]
    frames = video_io.extract_frames_at_timestamps(clip_path, ts)
    if not frames:
        return {
            "clip_id": clip_id, "clip_path": clip_path, "duration": duration,
            "n_masks_raw": 0, "n_masks_kept": 0, "n_clusters": 0, "clusters": [],
        }
    H, W = frames[0].shape[:2]

    # 2+3) SAM2 mask gen on each keyframe + filter
    all_masks: list[tuple[int, dict]] = []   # (frame_idx, mask_record)
    n_raw = 0
    for fi, frame in enumerate(frames):
        masks = sam2_wrapper.segment_everything(frame)
        n_raw += len(masks)
        kept = _filter_masks(
            masks, (H, W),
            min_area=memory_config["min_mask_area_px"],
            max_area_ratio=memory_config["max_mask_area_ratio"],
            min_stability=memory_config["sam2_stability_score_thresh"],
        )
        for m in kept:
            all_masks.append((fi, m))

    n_kept = len(all_masks)
    if n_kept == 0:
        return {
            "clip_id": clip_id, "clip_path": clip_path, "duration": duration,
            "n_masks_raw": n_raw, "n_masks_kept": 0, "n_clusters": 0, "clusters": [],
        }

    # 4) crops
    crops = []
    for fi, m in all_masks:
        crop = _masked_crop(
            frames[fi], m["segmentation"].astype(bool), m["bbox"],
            pad=memory_config["crop_padding_px"],
        )
        crops.append(crop)

    # 5) DINOv2 embeddings
    embs = dinov2_wrapper.embed(crops, batch_size=16)

    # 6) intra-clip dedupe (appearance + spatial prior)
    bboxes_for_dedupe = [tuple(int(v) for v in m["bbox"]) for _, m in all_masks]
    cluster_ids = _intra_clip_dedupe(
        embs,
        bboxes=bboxes_for_dedupe,
        sim_threshold=memory_config["object_dedupe_threshold"],
        spatial_sim_floor=memory_config["object_dedupe_spatial_sim_floor"],
        iou_threshold=memory_config["object_dedupe_iou_threshold"],
    )
    n_clusters = int(cluster_ids.max()) + 1 if len(cluster_ids) else 0

    # Representative = largest area within each cluster
    cluster_members: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        cluster_members.setdefault(int(cid), []).append(idx)
    cluster_rep_idx: dict[int, int] = {
        cid: max(members, key=lambda i: all_masks[i][1]["area"])
        for cid, members in cluster_members.items()
    }

    # 7) cross-clip match + graph mutate
    debug_clusters = []
    for cid in sorted(cluster_members.keys()):
        rep = cluster_rep_idx[cid]
        emb = embs[rep]
        crop = crops[rep]
        fi, m = all_masks[rep]

        candidates = graph.search_object_nodes(
            emb,
            threshold=memory_config["object_matching_threshold"],
            exclude_clip_id=clip_id,
        )
        is_new = False
        matched_sim = None
        if candidates:
            oid, matched_sim = candidates[0]
            graph.update_object_node(
                oid, embedding=emb,
                crop_b64=video_io.encode_rgb_to_base64_png(crop),
                clip_id=clip_id,
            )
        else:
            is_new = True
            oid = graph.add_object_node({
                "embeddings": [emb],
                "contents": [video_io.encode_rgb_to_base64_png(crop)],
                "name": None,
                "first_clip": clip_id,
                "last_clip": clip_id,
                "seen_count": 1,
            })

        rec = {
            "local_id": cid, "global_object_id": oid, "is_new": is_new,
            "rep_keyframe_idx": fi, "bbox": [int(v) for v in m["bbox"]],
            "area": int(m["area"]),
            "stability_score": float(m["stability_score"]),
            "matched_sim": None if matched_sim is None else float(matched_sim),
        }

        if debug_dir is not None:
            out_png = Path(debug_dir) / f"clip{clip_id:03d}_obj{oid:04d}_c{cid:02d}.png"
            video_io.save_rgb_png(crop, str(out_png))
            rec["crop_path"] = str(out_png)
        debug_clusters.append(rec)

    return {
        "clip_id": clip_id, "clip_path": clip_path, "duration": duration,
        "n_masks_raw": n_raw, "n_masks_kept": n_kept,
        "n_clusters": n_clusters, "clusters": debug_clusters,
    }


def save_debug_record(record: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(record, f, indent=2)
