"""Qwen3-VL wrapper for video+image prompting.

Interface:
    build_qwen3vl(model_dir, device_map="auto") -> None
    generate(video_path, object_crops, prompt, max_new_tokens=1024) -> str

`object_crops` is a list of (label, rgb_ndarray | PIL.Image) pairs — rendered
as "<object_N>:" + image tiles in a single multimodal user turn. The prompt
string is appended after all image tiles and the video (if provided).

The model is loaded in bf16 by default. Use `device_map` and `max_memory`
from the caller when the selected checkpoint needs multi-GPU placement.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from . import video_io

logger = logging.getLogger(__name__)

_PROC = None
_MODEL = None


def build_qwen3vl(
    model_dir: str,
    device_map: str | dict = "auto",
    dtype: torch.dtype = torch.bfloat16,
    max_memory: dict | None = None,
) -> None:
    global _PROC, _MODEL
    if _MODEL is not None:
        return
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    logger.info(f"Loading Qwen3-VL from {model_dir} (device_map={device_map})")
    _PROC = AutoProcessor.from_pretrained(model_dir)
    kwargs = {"dtype": dtype, "device_map": device_map}
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    _MODEL = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, **kwargs).eval()


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def _build_messages(
    video_frames: list[Image.Image] | None,
    object_crops: Iterable[tuple[str, object]],
    prompt: str,
) -> list[dict]:
    """Build a single user turn: video (opt), per-object (label text + image), then prompt."""
    content: list[dict] = []
    if video_frames is not None and len(video_frames) > 0:
        content.append({"type": "video", "video": video_frames})

    crops = list(object_crops)
    if crops:
        content.append({"type": "text", "text": "Object features:"})
        for label, img in crops:
            content.append({"type": "text", "text": f"<{label}>:"})
            content.append({"type": "image", "image": _to_pil(img)})

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


@torch.inference_mode()
def _run(messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    if _MODEL is None:
        raise RuntimeError("call build_qwen3vl(...) first")
    inputs = _PROC.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    try:
        first_device = next(_MODEL.parameters()).device
    except StopIteration:
        first_device = torch.device("cuda:0")
    inputs = {k: (v.to(first_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=float(temperature))
    else:
        gen_kwargs.update(do_sample=False)

    output_ids = _MODEL.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[:, input_len:]
    text = _PROC.batch_decode(new_ids, skip_special_tokens=True)[0]
    return text.strip()


def generate(
    video_path: str | None,
    object_crops: Iterable[tuple[str, object]],
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    video_fps: float = 2.0,
    max_video_frames: int = 32,
) -> str:
    """Pre-decode `video_path` at `video_fps`, then call the VLM once."""
    video_frames = None
    if video_path is not None:
        raw = video_io.sample_uniform_frames(
            video_path, fps=video_fps, max_frames=max_video_frames
        )
        video_frames = [Image.fromarray(f).convert("RGB") for f in raw]
    messages = _build_messages(video_frames, object_crops, prompt)
    return _run(messages, max_new_tokens=max_new_tokens, temperature=temperature)


def generate_from_frames(
    video_frames: list[Image.Image] | None,
    object_crops: Iterable[tuple[str, object]],
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Variant that takes already-decoded PIL frames (or None for text-only video)."""
    messages = _build_messages(video_frames, object_crops, prompt)
    return _run(messages, max_new_tokens=max_new_tokens, temperature=temperature)
