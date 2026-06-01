"""Retrieval over the object-centric VideoGraph.

At query time we embed the question with BGE-M3 and rank every text node
(episodic + semantic) by cosine similarity. For each retrieved hit we also
expand the referenced <object_N> tokens through the equivalence union-find so
the downstream QA prompt sees all equivalent IDs for the same physical object.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .utils import bge_wrapper
from .videograph import VideoGraph, _extract_refs

logger = logging.getLogger(__name__)


@dataclass
class RetrievedMemory:
    node_id: int
    type: str           # 'episodic' | 'semantic'
    clip_id: int
    text: str
    score: float
    refs: list[str]          # raw <object_N> tokens in text
    expanded_refs: list[str] # refs + their equivalence-group members (unique, ordered)


def _expand_refs(graph: VideoGraph, refs: list[str]) -> list[str]:
    seen, out = set(), []
    for r in refs:
        canon = graph.reverse_character_mappings.get(r, r)
        group = graph.character_mappings.get(canon, {r})
        for m in sorted(group, key=lambda x: int(x.split("_")[1])):
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def retrieve_long_term(
    graph: VideoGraph,
    question: str,
    top_k: int = 8,
    include_episodic: bool = True,
    include_semantic: bool = True,
    max_clip: int | None = None,
    min_score: float = 0.0,
) -> list[RetrievedMemory]:
    """BGE-embed `question`, rank text nodes by cosine sim, return top_k hits.

    `max_clip` (inclusive) lets the caller enforce long-term = clips 0..N-1 so
    the retrieval cannot leak information from clips strictly after the query
    time. Set to None to retrieve across the whole graph.
    """
    q_emb = bge_wrapper.embed_one(question).astype(np.float32)

    hits: list[tuple[float, int]] = []
    for tid in graph.text_nodes:
        node = graph.nodes[tid]
        if node.type == "episodic" and not include_episodic:
            continue
        if node.type == "semantic" and not include_semantic:
            continue
        if max_clip is not None and node.clip_id is not None and node.clip_id > max_clip:
            continue
        emb = node.metadata.get("embedding")
        if emb is None:
            continue
        sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-9))
        if sim < min_score:
            continue
        hits.append((sim, tid))

    hits.sort(key=lambda x: -x[0])
    hits = hits[:top_k]

    out: list[RetrievedMemory] = []
    for sim, tid in hits:
        node = graph.nodes[tid]
        text = node.metadata["contents"][-1]
        refs = _extract_refs(text)
        out.append(
            RetrievedMemory(
                node_id=tid,
                type=node.type,
                clip_id=int(node.clip_id) if node.clip_id is not None else -1,
                text=text,
                score=sim,
                refs=refs,
                expanded_refs=_expand_refs(graph, refs),
            )
        )
    return out


def collect_object_crops(
    graph: VideoGraph,
    object_labels: Iterable[str],
    max_crops_per_object: int = 1,
):
    """Return [(label, PIL.Image), ...] crops for each unique <object_N> label.

    `object_labels` are tokens like "object_7" (no angle brackets). We decode
    the stored base64 PNGs. Only takes the most recent crop per object node.
    """
    import base64
    from io import BytesIO

    from PIL import Image

    seen = set()
    out = []
    for lbl in object_labels:
        canon = graph.reverse_character_mappings.get(lbl, lbl)
        if canon in seen:
            continue
        seen.add(canon)
        try:
            oid = int(canon.split("_")[1])
        except (IndexError, ValueError):
            continue
        node = graph.nodes.get(oid)
        if node is None or node.type != "object":
            continue
        crops_b64 = node.metadata.get("contents") or []
        if not crops_b64:
            continue
        for b64 in crops_b64[-max_crops_per_object:]:
            try:
                img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            except Exception as e:
                logger.warning(f"decode crop for {canon} failed: {e}")
                continue
            out.append((canon, img))
    return out


def format_memory_block(hits: list[RetrievedMemory]) -> str:
    """Render retrieved memories as a single text block for the QA prompt.

    Episodic lines are grouped by clip and listed in clip order; semantic lines
    are listed separately. Equivalence groups referenced by any hit are
    surfaced explicitly so the VLM knows which IDs are the same object.
    """
    if not hits:
        return "(no prior memory retrieved)"

    epi = [h for h in hits if h.type == "episodic"]
    sem = [h for h in hits if h.type == "semantic"]

    epi.sort(key=lambda h: (h.clip_id, h.node_id))
    sem.sort(key=lambda h: -h.score)

    lines: list[str] = []
    if epi:
        lines.append("Episodic notes (per clip, oldest first):")
        current_clip = None
        for h in epi:
            if h.clip_id != current_clip:
                lines.append(f"  [clip {h.clip_id}]")
                current_clip = h.clip_id
            lines.append(f"    - {h.text}")
    if sem:
        lines.append("Semantic notes:")
        for h in sem:
            lines.append(f"  - {h.text}")

    equiv_lines: list[str] = []
    seen_groups: set[tuple[str, ...]] = set()
    for h in hits:
        if len(h.expanded_refs) <= 1:
            continue
        key = tuple(sorted(h.expanded_refs))
        if key in seen_groups:
            continue
        seen_groups.add(key)
        equiv_lines.append("  - " + " = ".join(f"<{m}>" for m in h.expanded_refs))
    if equiv_lines:
        lines.append("Known identity groups (same physical object):")
        lines.extend(equiv_lines)

    return "\n".join(lines)
