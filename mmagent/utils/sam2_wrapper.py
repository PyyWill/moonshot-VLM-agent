"""Singleton loader + inference helpers for SAM2.1 class-agnostic segmentation."""
from __future__ import annotations

import os
import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MASK_GENERATOR = None
_DEVICE = None


def build_mask_generator(
    cfg_rel_path: str,
    ckpt_abs_path: str,
    device: str = "cuda:0",
    *,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.70,
    stability_score_thresh: float = 0.92,
    box_nms_thresh: float = 0.75,
    min_mask_region_area: int = 500,
):
    """Create and cache a SAM2AutomaticMaskGenerator.

    Args:
        cfg_rel_path: Hydra config path relative to the installed `sam2` package
                      (e.g. "configs/sam2.1/sam2.1_hiera_l.yaml").
        ckpt_abs_path: Absolute path to the `.pt` checkpoint file.
    """
    global _MASK_GENERATOR, _DEVICE
    if _MASK_GENERATOR is not None:
        return _MASK_GENERATOR

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    if not os.path.isabs(ckpt_abs_path):
        raise ValueError(f"ckpt_abs_path must be absolute, got {ckpt_abs_path!r}")

    logger.info(f"Loading SAM2 {cfg_rel_path} + {ckpt_abs_path} on {device}")
    sam2_model = build_sam2(cfg_rel_path, ckpt_abs_path, device=device)
    _MASK_GENERATOR = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    _DEVICE = device
    return _MASK_GENERATOR


def segment_everything(frame_rgb: np.ndarray) -> list[dict[str, Any]]:
    """Run automatic mask generation on a single HxWx3 uint8 RGB frame.

    Returns list of masks, each dict with keys:
      segmentation (HxW bool), area (int), bbox (x,y,w,h int),
      predicted_iou (float), stability_score (float), crop_box, point_coords.
    """
    if _MASK_GENERATOR is None:
        raise RuntimeError("call build_mask_generator(...) first")
    assert frame_rgb.dtype == np.uint8 and frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        return _MASK_GENERATOR.generate(frame_rgb)
