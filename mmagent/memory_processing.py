"""Per-clip memory generation for Moonshot VLM Agent.

Pipeline (runs AFTER object_processing has populated the graph for a clip):
  1. Assemble object features: list of (label=f"object_{gid}", crop) for every
     object node whose first_clip or last_clip == clip_id.
  2. Call Qwen3-VL with the clip video + object tiles + the memory-generation
     prompt; parse the JSON-ish {"episodic_memory": [...], "semantic_memory": [...]}.
  3. Embed each memory string with BGE-M3.
  4. For each episodic line: add an `episodic` text node + `mention` edges to
     every <object_N> referenced.
  5. For each semantic line: try to merge with an existing semantic node that
     shares the same <object_N> reference set using cosine-sim; otherwise add a
     new node with mention edges.
  6. Call `graph.refresh_equivalences()` so union-find is up-to-date for retrieval.
"""
from __future__ import annotations

import base64
import logging
import re
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from .prompts import prompt_generate_object_memory
from .utils import bge_wrapper, qwen3vl_wrapper
from .utils.general import validate_and_fix_dict
from .videograph import OBJECT_REF_RE, VideoGraph

logger = logging.getLogger(__name__)


# ---------- feature assembly ----------

def _decode_b64_png(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def collect_clip_object_features(
    graph: VideoGraph,
    clip_id: int,
    max_objects: int | None = None,
) -> list[tuple[str, Image.Image]]:
    """Return [(f"object_{gid}", PIL.Image), ...] for every object node that
    was seen in this clip. One crop per node (the most recent one).
    """
    out: list[tuple[str, Image.Image]] = []
    oids_here = list(graph.object_nodes_by_clip.get(clip_id, []))
    for oid in oids_here:
        node = graph.nodes.get(oid)
        if node is None or node.type != "object":
            continue
        crops = node.metadata.get("contents") or []
        if not crops:
            continue
        try:
            img = _decode_b64_png(crops[-1])
        except Exception as e:
            logger.warning(f"decode crop for object_{oid} failed: {e}")
            continue
        out.append((f"object_{oid}", img))
    if max_objects is not None and len(out) > max_objects:
        out = out[:max_objects]
    return out


# ---------- ref parsing ----------

def parse_object_refs(graph: VideoGraph, text: str) -> list[int]:
    """Return unique integer object ids in `text` (in insertion order) that
    correspond to actual object nodes in `graph`."""
    seen, out = set(), []
    for m in OBJECT_REF_RE.finditer(text):
        tok = m.group(1)  # "object_17"
        try:
            oid = int(tok.split("_")[1])
        except (IndexError, ValueError):
            continue
        if oid in seen:
            continue
        node = graph.nodes.get(oid)
        if node is None or node.type != "object":
            continue
        seen.add(oid)
        out.append(oid)
    return out


# ---------- VLM call ----------

def generate_memories_for_clip(
    clip_path: str,
    object_features: list[tuple[str, Image.Image]],
    max_new_tokens: int = 1536,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> tuple[list[str], list[str]]:
    """Run Qwen3-VL once, return (episodic_list, semantic_list). Empty lists on failure."""
    valid_ids = ", ".join(f"<{lbl}>" for lbl, _ in object_features)
    prompt = (
        prompt_generate_object_memory
        + f"\n\nAllowed object IDs for this clip (you MUST NOT reference any ID outside this list): {valid_ids}\n"
    )
    for attempt in range(max_retries):
        raw = qwen3vl_wrapper.generate(
            video_path=clip_path,
            object_crops=object_features,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        parsed = validate_and_fix_dict(raw)
        if isinstance(parsed, dict):
            epi = parsed.get("episodic_memory") or []
            sem = parsed.get("semantic_memory") or []
            if isinstance(epi, list) and isinstance(sem, list):
                epi = [str(x).strip() for x in epi if str(x).strip()]
                sem = [str(x).strip() for x in sem if str(x).strip()]
                return epi, sem
        logger.warning(
            f"memory-gen parse failed (attempt {attempt + 1}/{max_retries}); "
            f"raw len={len(raw)}, tail={raw[-200:]!r}"
        )
    return [], []


# ---------- graph writes ----------

def _insert_text_node(
    graph: VideoGraph,
    node_type: str,
    clip_id: int,
    text: str,
    embedding: np.ndarray,
    edge_weight: int = 1,
) -> int:
    tid = graph.add_text_node(node_type, clip_id, text, embedding)
    for oid in parse_object_refs(graph, text):
        graph.add_edge(tid, oid, relation="mention", weight=edge_weight)
    return tid


def _update_or_insert_semantic(
    graph: VideoGraph,
    clip_id: int,
    text: str,
    embedding: np.ndarray,
    positive_threshold: float,
    negative_threshold: float,
) -> int | None:
    """Try to reinforce/weaken an existing semantic node with the same <object_N>
    reference set; fall back to inserting a new one."""
    refs = parse_object_refs(graph, text)
    if not refs:
        return _insert_text_node(graph, "semantic", clip_id, text, embedding)

    ref_tokens = [f"object_{oid}" for oid in refs]
    candidates = graph.search_semantic_nodes_by_refs(ref_tokens)
    for tid in candidates:
        node = graph.nodes[tid]
        existing_emb = node.metadata.get("embedding")
        if existing_emb is None:
            continue
        sim = float(
            np.dot(embedding, existing_emb)
            / (np.linalg.norm(embedding) * np.linalg.norm(existing_emb) + 1e-9)
        )
        if sim > positive_threshold:
            graph.reinforce_text_node(tid, new_content=text)
            return tid
        if sim < negative_threshold:
            graph.weaken_text_node(tid)
            # still create a fresh node for the new variant
            return _insert_text_node(graph, "semantic", clip_id, text, embedding)

    return _insert_text_node(graph, "semantic", clip_id, text, embedding)


def write_memories_to_graph(
    graph: VideoGraph,
    clip_id: int,
    episodic: list[str],
    semantic: list[str],
    positive_threshold: float,
    negative_threshold: float,
) -> dict[str, Any]:
    """Embed every line with BGE-M3 and insert into the graph.
    Returns a summary record for debugging."""
    all_texts = list(episodic) + list(semantic)
    if all_texts:
        embs = bge_wrapper.embed(all_texts, batch_size=16)
    else:
        embs = np.zeros((0, 1024), dtype=np.float32)

    epi_ids: list[int] = []
    for i, text in enumerate(episodic):
        tid = _insert_text_node(graph, "episodic", clip_id, text, embs[i])
        epi_ids.append(tid)

    sem_ids: list[int | None] = []
    for j, text in enumerate(semantic):
        emb = embs[len(episodic) + j]
        tid = _update_or_insert_semantic(
            graph, clip_id, text, emb,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )
        sem_ids.append(tid)

    graph.refresh_equivalences()

    return {
        "clip_id": clip_id,
        "n_episodic": len(episodic),
        "n_semantic": len(semantic),
        "episodic_node_ids": epi_ids,
        "semantic_node_ids": sem_ids,
        "equivalence_groups": {k: sorted(v) for k, v in graph.character_mappings.items()},
    }


# ---------- orchestrator ----------

def process_memories_for_clip(
    graph: VideoGraph,
    clip_path: str,
    clip_id: int,
    memory_config: dict,
    max_new_tokens: int = 1536,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """End-to-end: gather object features → VLM gen → embed → graph write."""
    features = collect_clip_object_features(graph, clip_id)
    logger.info(f"clip {clip_id}: {len(features)} object features for VLM")

    episodic, semantic = generate_memories_for_clip(
        clip_path=clip_path,
        object_features=features,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    logger.info(
        f"clip {clip_id}: VLM returned {len(episodic)} episodic + {len(semantic)} semantic"
    )

    rec = write_memories_to_graph(
        graph,
        clip_id=clip_id,
        episodic=episodic,
        semantic=semantic,
        positive_threshold=float(memory_config.get("positive_threshold", 0.85)),
        negative_threshold=float(memory_config.get("negative_threshold", 0.0)),
    )
    rec["raw_episodic"] = episodic
    rec["raw_semantic"] = semantic
    rec["n_object_features"] = len(features)
    return rec
