"""Shared helpers for command-line scripts."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def apply_cuda_lib_dir(env_name: str = "CUDA_LIB_DIR") -> None:
    """Optionally prepend an external CUDA library directory to LD_LIBRARY_PATH."""
    lib_dir = os.environ.get(env_name)
    if lib_dir and os.path.isdir(lib_dir):
        os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def configure_logging(level: str, logger_name: str) -> logging.Logger:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    return logging.getLogger(logger_name)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_configs(processing_config: str | Path, memory_config: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return load_json(processing_config), load_json(memory_config)


def resolve_model_path(processing_config: dict[str, Any], key: str) -> str:
    """Resolve a model path from config.

    Absolute values are used as-is. Relative values are interpreted under
    `models_root` for backward-compatible configs.
    """
    value = Path(processing_config[key])
    if value.is_absolute():
        return str(value)
    return str(Path(processing_config.get("models_root", "")) / value)


def discover_clip_ids(clips_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in clips_dir.glob("*.mp4"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(ids)


def build_sam2_from_config(processing_config: dict[str, Any], memory_config: dict[str, Any]) -> None:
    from mmagent.utils import sam2_wrapper

    sam2_wrapper.build_mask_generator(
        cfg_rel_path=processing_config["sam2_cfg"],
        ckpt_abs_path=resolve_model_path(processing_config, "sam2_ckpt"),
        device=processing_config["sam2_device"],
        points_per_side=memory_config["sam2_points_per_side"],
        pred_iou_thresh=memory_config["sam2_pred_iou_thresh"],
        stability_score_thresh=memory_config["sam2_stability_score_thresh"],
        box_nms_thresh=memory_config["sam2_box_nms_thresh"],
        min_mask_region_area=memory_config["min_mask_area_px"],
    )


def build_dinov2_from_config(processing_config: dict[str, Any]) -> None:
    from mmagent.utils import dinov2_wrapper

    dinov2_wrapper.build_dinov2(
        model_dir=resolve_model_path(processing_config, "dinov2_ckpt"),
        device=processing_config["dinov2_device"],
    )


def build_text_embedder_from_config(processing_config: dict[str, Any]) -> None:
    from mmagent.utils import bge_wrapper

    bge_wrapper.build_text_embedder(
        model_dir=resolve_model_path(processing_config, "text_embed_ckpt"),
        device=processing_config["text_embed_device"],
    )


def build_qwen3vl_from_config(processing_config: dict[str, Any]) -> None:
    from mmagent.utils import qwen3vl_wrapper

    qwen3vl_wrapper.build_qwen3vl(
        model_dir=resolve_model_path(processing_config, "vlm_ckpt"),
        device_map=processing_config.get("vlm_device", "cuda:0"),
    )
