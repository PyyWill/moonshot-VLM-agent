"""Single-round QA controller.

Composes:
  [LONG-TERM MEMORY] = BGE-top-K retrieval over episodic+semantic text nodes
                       (cap at `max_clip` so we don't leak post-query clips)
  [OBJECT FEATURES]  = cropped images for every object referenced by the hits
                       (expanded through equivalence groups)
  [CURRENT SCENE]    = raw frames from the ongoing clip, sampled at clip_fps
                       over the first `short_term_seconds` of that clip
  [QUESTION]         = user's question text

and calls Qwen3-VL once to produce a concise natural-language answer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .prompts import prompt_answer_question
from .retrieve import (
    RetrievedMemory,
    collect_object_crops,
    format_memory_block,
    retrieve_long_term,
)
from .utils import qwen3vl_wrapper, video_io
from .videograph import VideoGraph

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    question: str
    answer: str
    n_long_term_hits: int
    n_object_crops: int
    n_short_term_frames: int
    max_clip: int | None
    hits: list[RetrievedMemory] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def _sample_short_term_frames(
    clip_path: str | None,
    short_term_seconds: float,
    fps: float,
    max_frames: int,
) -> list[Image.Image]:
    """Sample the first `short_term_seconds` of `clip_path` at `fps`. Returns []
    when short-term is disabled (k=0) or clip_path is None."""
    if not clip_path or short_term_seconds <= 0 or fps <= 0:
        return []
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(clip_path)
    src_fps = float(vr.get_avg_fps())
    if src_fps <= 0:
        return []
    total = len(vr)
    end_idx = min(total, int(round(short_term_seconds * src_fps)))
    if end_idx <= 0:
        return []
    n = max(1, min(int(round(short_term_seconds * fps)), max_frames))
    if n == 1:
        idxs = [end_idx // 2]
    else:
        idxs = [int(round(i * (end_idx - 1) / (n - 1))) for i in range(n)]
    return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in idxs]


def answer_question_single_round(
    graph: VideoGraph,
    question: str,
    *,
    current_clip_path: str | None = None,
    short_term_seconds: float = 0.0,
    max_clip: int | None = None,
    top_k: int = 8,
    include_episodic: bool = True,
    include_semantic: bool = True,
    clip_fps: float = 2.0,
    max_short_term_frames: int = 16,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> QAResult:
    """Run one VLM turn combining long-term memory + short-term frames.

    Time convention: if the question arrives at t = N*10 + k, set
    max_clip = N - 1 (last finished clip) and short_term_seconds = k.
    Set current_clip_path to the ongoing (unfinished) clip, or None if k=0.
    """
    hits = retrieve_long_term(
        graph,
        question=question,
        top_k=top_k,
        include_episodic=include_episodic,
        include_semantic=include_semantic,
        max_clip=max_clip,
    )

    ref_labels: list[str] = []
    seen = set()
    for h in hits:
        for r in h.expanded_refs or h.refs:
            if r not in seen:
                seen.add(r)
                ref_labels.append(r)
    object_crops = collect_object_crops(graph, ref_labels, max_crops_per_object=1)

    short_term_frames = _sample_short_term_frames(
        current_clip_path, short_term_seconds, clip_fps, max_short_term_frames
    )

    memory_block = format_memory_block(hits)
    prompt = (
        "[LONG-TERM MEMORY]\n"
        f"{memory_block}\n\n"
        "[OBJECT FEATURES]\n"
        f"(images of {len(object_crops)} referenced objects follow as <object_N> tiles)\n\n"
        "[CURRENT SCENE]\n"
        f"{'(the video frames that follow are the first '+ f'{short_term_seconds:g}s of the ongoing clip)' if short_term_frames else '(no current-scene video provided — answer from memory only)'}\n\n"
        + prompt_answer_question.format(question=question.strip())
    )

    raw = qwen3vl_wrapper.generate_from_frames(
        video_frames=short_term_frames or None,
        object_crops=object_crops,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return QAResult(
        question=question,
        answer=raw.strip(),
        n_long_term_hits=len(hits),
        n_object_crops=len(object_crops),
        n_short_term_frames=len(short_term_frames),
        max_clip=max_clip,
        hits=hits,
        debug={
            "prompt_preview": prompt[:800],
            "ref_labels": ref_labels,
        },
    )
