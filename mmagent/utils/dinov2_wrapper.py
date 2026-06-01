"""DINOv2 embedding for object crops.

The pipeline passes masked object crops and unit-normalizes the resulting
descriptor. For legacy black-filled crops, patch pooling focuses on non-black
foreground patches; for white-filled crops, the descriptor behaves closer to a
crop-level patch mean while preserving the same public interface.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

_PROC = None
_MODEL = None
_DEVICE = None
_PATCH = 14

# ImageNet normalization (matches DINOv2 preprocessor_config.json).
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_dinov2(model_dir: str, device: str = "cuda:0") -> None:
    global _PROC, _MODEL, _DEVICE, _PATCH
    if _MODEL is not None:
        return
    from transformers import AutoImageProcessor, AutoModel

    logger.info(f"Loading DINOv2 from {model_dir} on {device}")
    _PROC = AutoImageProcessor.from_pretrained(model_dir)
    _MODEL = AutoModel.from_pretrained(model_dir).half().to(device).eval()
    _DEVICE = device
    _PATCH = int(getattr(_MODEL.config, "patch_size", 14))


@torch.inference_mode()
def embed(
    crops_rgb: Iterable[np.ndarray],
    batch_size: int = 16,
    min_patch_mask_frac: float = 0.25,
) -> np.ndarray:
    """Embed masked crops via patch-mean pooling.

    Legacy black-filled crops recover an approximate foreground mask from
    non-black pixels, then mean-pool patch tokens whose footprint is at least
    `min_patch_mask_frac` foreground. White-filled crops naturally keep most
    patches, which is compatible with the current white-background crop policy.
    Fallback to CLS when no patch qualifies.

    Returns (N, D) unit-normalized float32. D = model hidden size (1024 for
    dinov2-large).
    """
    if _MODEL is None:
        raise RuntimeError("call build_dinov2(...) first")

    crops = list(crops_rgb)
    dim = int(_MODEL.config.hidden_size)
    if not crops:
        return np.zeros((0, dim), dtype=np.float32)

    pil_images = [Image.fromarray(c) for c in crops]

    outs = []
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start:start + batch_size]
        inputs = _PROC(images=batch, return_tensors="pt").to(_DEVICE)
        px = inputs["pixel_values"]                               # (B, 3, H, W)
        inputs_h = {
            k: v.half() if v.dtype == torch.float32 else v
            for k, v in inputs.items()
        }
        hs = _MODEL(**inputs_h).last_hidden_state                 # (B, 1+N, D) fp16

        B, _, H, W = px.shape
        Hp, Wp = H // _PATCH, W // _PATCH
        N = Hp * Wp

        # Recover "not-black" binary mask from the normalized input.
        mean_t = _MEAN.to(px.device, px.dtype)
        std_t = _STD.to(px.device, px.dtype)
        px_un = px * std_t + mean_t                               # back to [0,1]
        mask_pix = (px_un.sum(dim=1, keepdim=True) > 0.03).float()  # (B, 1, H, W)

        # Fraction of each 14x14 patch that is inside the mask.
        mask_patch = F.avg_pool2d(mask_pix, kernel_size=_PATCH, stride=_PATCH)  # (B,1,Hp,Wp)
        patch_keep = (mask_patch >= min_patch_mask_frac).float()                # (B,1,Hp,Wp)
        w = patch_keep.view(B, N, 1).to(hs.dtype)                               # (B,N,1)

        patch_feat = hs[:, 1:, :]                                  # (B, N, D)
        total = w.sum(dim=1).clamp(min=1e-6)                       # (B, 1)
        pooled = (patch_feat * w).sum(dim=1) / total               # (B, D)

        # Fallback to CLS for crops whose mask is entirely below min_patch_mask_frac.
        empty = (w.sum(dim=(1, 2)) == 0)
        if empty.any():
            pooled[empty] = hs[empty, 0, :]

        pooled = F.normalize(pooled, dim=-1).float().cpu().numpy()
        outs.append(pooled)

    return np.concatenate(outs, axis=0)
