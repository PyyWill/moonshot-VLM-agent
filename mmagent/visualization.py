"""Render VideoGraph memory as a clean, self-contained HTML report.

The report uses one tab per clip plus one full-graph tab. Clip tabs use a
quiet bento layout: object nodes, facts, reasoning, and a compact summary.
Hovering an object tile or any `<object_N>` reference previews the object crop.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import struct
from pathlib import Path

from mmagent.utils.general import load_video_graph
from mmagent.videograph import _extract_refs

REPO_ROOT = Path(__file__).resolve().parents[1]

NODE_COLOR = {"object": "#7C53A6", "memory": "#D99AD5"}
EDGE_COLOR = {"mention": "#C9C2D2", "equivalence": "#9F7EBF"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", type=str, default=str(REPO_ROOT / "runs" / "memory" / "graph.pkl"))
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "runs" / "memory" / "grounded_memory.html"))
    p.add_argument("--clip-min", type=int, default=None)
    p.add_argument("--clip-max", type=int, default=None)
    p.add_argument("--thumb-px", type=int, default=128)
    p.add_argument("--no-graph-tab", action="store_true")
    return p.parse_args()


def _sort_object_token(label: str) -> int:
    try:
        return int(label.split("_")[1])
    except Exception:
        return 10**9


def _canon_label(graph, label: str) -> str:
    return graph.reverse_character_mappings.get(label, label)


def _object_node(graph, label: str):
    canon = _canon_label(graph, label)
    try:
        oid = int(canon.split("_")[1])
    except Exception:
        return None
    node = graph.nodes.get(oid)
    if node is None or node.type != "object":
        return None
    return node


def _png_size_from_b64(b64: str) -> tuple[int, int] | None:
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    if len(raw) >= 24 and raw.startswith(b"\x89PNG\r\n\x1a\n") and raw[12:16] == b"IHDR":
        return struct.unpack(">II", raw[16:24])
    return None


def _valid_display_crop_b64(b64: str | None) -> str | None:
    if not b64:
        return None
    size = _png_size_from_b64(b64)
    if size is not None and max(size) > 480:
        return None
    return b64


def _object_crop_b64(
    graph,
    label: str,
    clip_id: int | None = None,
    *,
    inherit_previous: bool = False,
) -> str | None:
    node = _object_node(graph, label)
    if node is None:
        return None
    if clip_id is not None:
        crops_by_clip = node.metadata.get("contents_by_clip") or {}
        crops = crops_by_clip.get(str(clip_id)) or crops_by_clip.get(clip_id) or []
        crop = _valid_display_crop_b64(crops[-1] if crops else None)
        if crop or not inherit_previous:
            return crop
        previous: list[tuple[int, str]] = []
        for key, values in crops_by_clip.items():
            if not values:
                continue
            try:
                cid = int(key)
            except Exception:
                continue
            if cid < clip_id:
                previous.append((cid, values[-1]))
        for _, candidate in sorted(previous, reverse=True):
            crop = _valid_display_crop_b64(candidate)
            if crop:
                return crop
        return None
    crops = node.metadata.get("contents") or []
    return _valid_display_crop_b64(crops[-1] if crops else None)


def _latest_object_crop_b64(graph, label: str) -> str | None:
    node = _object_node(graph, label)
    if node is None:
        return None
    crops_by_clip = node.metadata.get("contents_by_clip") or {}
    candidates: list[tuple[int, str]] = []
    for key, crops in crops_by_clip.items():
        if not crops:
            continue
        try:
            cid = int(key)
        except Exception:
            continue
        candidates.append((cid, crops[-1]))
    for _, crop in sorted(candidates, reverse=True):
        valid_crop = _valid_display_crop_b64(crop)
        if valid_crop:
            return valid_crop
    return _object_crop_b64(graph, label)


def _object_display_name(graph, label: str) -> str:
    node = _object_node(graph, label)
    if node is None:
        return label
    return str(node.metadata.get("label") or node.metadata.get("name") or label)


def _inline_text(text: str, clip_id: int | None = None) -> str:
    import re

    parts = re.split(r"(<object_\d+>)", text)
    out: list[str] = []
    for part in parts:
        if part.startswith("<object_") and part.endswith(">"):
            label = part[1:-1]
            preview_key = f"clip_{clip_id}:{label}" if clip_id is not None else label
            out.append(
                f'<span class="objref" data-obj="{preview_key}" tabindex="0">'
                f'&lt;{label}&gt;</span>'
            )
        else:
            out.append(html.escape(part))
    return "".join(out)


def _memory_kind(text: str) -> str:
    if text.startswith("[f]"):
        return "fact"
    if text.startswith("[r]"):
        return "reasoning"
    return "memory"


def _clean_memory_text(text: str) -> str:
    if text.startswith("[f]") or text.startswith("[r]"):
        return text[3:].strip()
    return text


def _memory_timestamp_label(graph, tid: int) -> str:
    timestamp = graph.nodes[tid].metadata.get("timestamp") or {}
    label = timestamp.get("label")
    if label:
        return str(label)
    time_range = graph.nodes[tid].metadata.get("time_range") or {}
    if "clip_start_sec" in time_range and "clip_end_sec" in time_range:
        start = float(time_range.get("clip_start_sec", 0.0))
        end = float(time_range.get("clip_end_sec", start))
        return f"{((start + end) / 2.0):.1f}s"
    return "unknown"


def _task_status_label(graph, tid: int) -> str:
    status = graph.nodes[tid].metadata.get("task_status", None)
    if status is None:
        return "None"
    if isinstance(status, list):
        return ", ".join(str(item) for item in status) if status else "None"
    return str(status)


def _memory_item_html(graph, tid: int) -> str:
    text = graph.nodes[tid].metadata["contents"][-1]
    clip_id = graph.nodes[tid].clip_id
    kind = _memory_kind(text)
    label = "F" if kind == "fact" else "R" if kind == "reasoning" else "M"
    timestamp = html.escape(_memory_timestamp_label(graph, tid))
    task_status = html.escape(_task_status_label(graph, tid))
    return (
        f'<li class="memory-item {kind}" data-timestamp="{timestamp}" aria-label="Timestamp {timestamp}">'
        f'<span class="memory-badge">{label}</span>'
        f'<span><span class="memory-text">{_inline_text(_clean_memory_text(text), clip_id=clip_id)}</span>'
        f'<span class="memory-meta"><span>Task {task_status}</span></span></span>'
        f'</li>'
    )


def _safety_warning_item_html(warning: str) -> str:
    label = "Safety"
    body = warning
    if warning.startswith("[") and "]" in warning:
        label, body = warning[1:].split("]", 1)
        body = body.strip()
    return (
        '<li class="safety-item">'
        f'<span class="safety-badge">{html.escape(label)}</span>'
        f'<span class="safety-text">{html.escape(body)}</span>'
        '</li>'
    )


def _clip_ids(graph) -> list[int]:
    ids = set(graph.text_nodes_by_clip.keys()) | set(graph.object_nodes_by_clip.keys())
    return sorted(ids)


def _clip_text_ids(graph, cid: int) -> tuple[list[int], list[int], list[int]]:
    all_ids = graph.text_nodes_by_clip.get(cid, [])
    facts = [t for t in all_ids if graph.nodes[t].metadata["contents"][-1].startswith("[f]")]
    reasoning = [t for t in all_ids if graph.nodes[t].metadata["contents"][-1].startswith("[r]")]
    other = [t for t in all_ids if t not in set(facts) | set(reasoning)]
    return facts, reasoning, other


def _referenced_objects_for_clip(graph, cid: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tid in graph.text_nodes_by_clip.get(cid, []):
        text = graph.nodes[tid].metadata["contents"][-1]
        for ref in _extract_refs(text):
            canon = _canon_label(graph, ref)
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
    for oid in graph.object_nodes_by_clip.get(cid, []):
        canon = f"object_{oid}"
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return sorted(out, key=_sort_object_token)


def _render_object_tile(graph, canon: str, thumb_px: int, clip_id: int | None = None) -> str:
    b64 = _object_crop_b64(graph, canon, clip_id=clip_id, inherit_previous=True)
    name = html.escape(_object_display_name(graph, canon))
    try:
        object_mark = f"O{int(canon.split('_')[1])}"
    except Exception:
        object_mark = canon
    img = (
        f'<img src="data:image/png;base64,{b64}" alt="{canon}" '
        f'style="max-width:{thumb_px}px;max-height:{thumb_px}px" />'
        if b64 else '<div class="no-img">No crop</div>'
    )
    node = _object_node(graph, canon)
    seen = node.metadata.get("seen_count", 1) if node is not None else 1
    return f"""
<div class="object-tile">
  <span class="object-index">{html.escape(object_mark)}</span>
  <span class="object-image">{img}</span>
  <span class="object-copy">
    <span class="object-token">&lt;{canon}&gt;</span>
    <span class="object-name">{name}</span>
    <span class="object-meta">seen {seen}</span>
  </span>
</div>
"""


def _render_clip(graph, cid: int, thumb_px: int) -> str:
    fact_ids, reasoning_ids, other_ids = _clip_text_ids(graph, cid)
    objects = _referenced_objects_for_clip(graph, cid)
    safety_warnings = list(getattr(graph, "safety_warnings_by_clip", {}).get(cid, []))

    object_tiles = "\n".join(_render_object_tile(graph, obj, thumb_px, clip_id=cid) for obj in objects)
    if not object_tiles:
        object_tiles = '<div class="empty">No object references for this clip.</div>'

    facts = "\n".join(_memory_item_html(graph, tid) for tid in fact_ids)
    if not facts:
        facts = '<li class="empty">No fact lines.</li>'

    reasoning = "\n".join(_memory_item_html(graph, tid) for tid in reasoning_ids)
    if not reasoning:
        reasoning = '<li class="empty">No reasoning lines.</li>'

    safety_items = "\n".join(_safety_warning_item_html(warning) for warning in safety_warnings)
    if not safety_items:
        safety_items = '<li class="empty safety-empty">No safety warnings.</li>'

    other = ""
    if other_ids:
        other_items = "\n".join(_memory_item_html(graph, tid) for tid in other_ids)
        other = f"""
<section class="bento-block other-block">
  <div class="block-heading">
    <h2>Other Memory</h2>
    <span>{len(other_ids)} lines</span>
  </div>
  <ul class="memory-list">{other_items}</ul>
</section>
"""

    return f"""
<section class="clip-view view" id="view-clip-{cid}">
  <div class="clip-header">
    <div>
      <p class="eyebrow">Clip-by-clip memory</p>
      <h1>Clip {cid}</h1>
    </div>
    <div class="clip-stats">
      <span><b>{len(objects)}</b> objects</span>
      <span><b>{len(fact_ids)}</b> facts</span>
      <span><b>{len(reasoning_ids)}</b> reasoning</span>
      <span><b>{len(safety_warnings)}</b> warnings</span>
    </div>
  </div>

  <div class="bento-grid">
    <section class="bento-block objects-block">
      <div class="block-heading">
        <h2>Object Nodes</h2>
        <span>part crops</span>
      </div>
      <div class="object-grid">{object_tiles}</div>
    </section>

    <section class="bento-block facts-block">
      <div class="block-heading">
        <h2>Facts</h2>
        <span>[f]</span>
      </div>
      <ul class="memory-list">{facts}</ul>
    </section>

    <section class="bento-block reasoning-block">
      <div class="block-heading">
        <h2>Reasoning</h2>
        <span>[r]</span>
      </div>
      <ul class="memory-list">{reasoning}</ul>
    </section>

    <section class="bento-block safety-block">
      <div class="block-heading">
        <h2>Safety Warnings</h2>
      </div>
      <ul class="safety-list">{safety_items}</ul>
    </section>

    {other}
  </div>
</section>
"""


def _graph_payload(graph) -> dict[str, list[dict]]:
    nodes: list[dict] = []
    for oid in graph.object_nodes:
        node = graph.nodes[oid]
        md = node.metadata
        name = str(md.get("label") or md.get("name") or f"object_{oid}")
        nodes.append(
            {
                "id": oid,
                "kind": "object",
                "token": f"object_{oid}",
                "label": f"O{oid}",
                "name": name,
                "seen": int(md.get("seen_count", 1)),
                "firstClip": md.get("first_clip"),
                "lastClip": md.get("last_clip"),
            }
        )

    for tid in graph.text_nodes:
        node = graph.nodes[tid]
        text = node.metadata["contents"][-1]
        kind = _memory_kind(text)
        prefix = "F" if kind == "fact" else "R" if kind == "reasoning" else "M"
        nodes.append(
            {
                "id": tid,
                "kind": "memory",
                "memoryKind": kind,
                "label": f"{prefix}{tid}",
                "clip": node.clip_id,
                "text": _clean_memory_text(text),
                "refs": _extract_refs(text),
            }
        )

    edges: list[dict] = []
    for (src, dst, rel), weight in graph.edges.items():
        if src not in graph.nodes or dst not in graph.nodes:
            continue
        edges.append(
            {
                "source": src,
                "target": dst,
                "relation": rel,
                "weight": int(weight),
            }
        )
    return {"nodes": nodes, "edges": edges}


_CSS = """
:root {
  --bg: #FCFBFD;
  --surface: #FFFFFF;
  --soft: #F8F5F9;
  --soft-2: #F2F0F4;
  --line: #E8E2EC;
  --text: #282A2D;
  --muted: #716C76;
  --purple: #7C53A6;
  --purple-soft: #9F7EBF;
  --pink: #D99AD5;
  --pink-soft: #F2D0F0;
  --shadow: 0 18px 48px rgba(63, 64, 64, 0.055);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Arial, Helvetica, sans-serif;
  line-height: 1.55;
}
.app-shell {
  width: min(1480px, calc(100% - 56px));
  margin: 0 auto;
  padding: 36px 0 48px;
}
.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(252, 251, 253, 0.92);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--line);
}
.topbar-inner {
  width: min(1480px, calc(100% - 56px));
  margin: 0 auto;
  padding: 16px 0 13px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 18px;
  align-items: center;
}
.brand h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
  line-height: 1.14;
  letter-spacing: 0;
}
.brand h1 span {
  display: block;
  margin-top: 2px;
  font-weight: 700;
}
.brand p {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 13px;
}
.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}
.tabs button {
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--text);
  border-radius: 8px;
  padding: 7px 12px;
  cursor: pointer;
  font-size: 13px;
  box-shadow: 0 1px 0 rgba(63, 64, 64, 0.02);
}
.tabs button:hover {
  border-color: var(--pink);
  background: #FFF8FE;
}
.tabs button.active {
  color: #fff;
  background: var(--purple);
  border-color: var(--purple);
}
.view { display: none; }
.view.active { display: block; }
.clip-header {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: end;
  margin: 34px 0 22px;
}
.eyebrow {
  margin: 0 0 6px;
  color: var(--purple);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}
.clip-header h1 {
  margin: 0;
  font-size: 36px;
  line-height: 1.15;
  letter-spacing: 0;
}
.clip-stats {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.clip-stats span {
  border: 1px solid var(--line);
  background: var(--soft);
  border-radius: 8px;
  padding: 8px 11px;
  color: var(--muted);
  font-size: 13px;
}
.clip-stats b { color: var(--text); }
.bento-grid {
  display: grid;
  grid-template-columns: minmax(440px, 0.82fr) minmax(560px, 1.18fr);
  gap: 18px;
  align-items: start;
}
.bento-block {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  padding: 19px;
}
.objects-block {
  grid-column: 1;
  grid-row: 1 / span 99;
  position: sticky;
  top: 104px;
  max-height: calc(100vh - 120px);
  overflow-y: auto;
  scrollbar-gutter: stable;
}
.facts-block,
.safety-block,
.reasoning-block,
.other-block {
  grid-column: 2;
}
.block-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 16px;
}
.block-heading h2 {
  margin: 0;
  font-size: 16px;
  font-weight: 650;
}
.block-heading span {
  color: var(--muted);
  font-size: 12px;
}
.object-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.object-tile {
  position: relative;
  min-height: 112px;
  border: 1px solid var(--line);
  background: #FBF8FC;
  border-radius: 8px;
  padding: 11px;
  display: grid;
  grid-template-columns: 86px 1fr;
  gap: 11px;
  align-items: center;
  text-align: left;
}
.object-index {
  position: absolute;
  top: 8px;
  left: 8px;
  min-width: 28px;
  height: 22px;
  padding: 0 7px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: #FFFFFF;
  border: 1px solid #E9DDED;
  color: var(--purple);
  font-size: 12px;
  font-weight: 700;
}
.object-image {
  width: 86px;
  height: 90px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #FFFFFF;
  border: 1px solid #EFE9F1;
  border-radius: 8px;
  padding: 7px;
}
.object-image img {
  display: block;
  object-fit: contain;
  max-width: 100% !important;
  max-height: 100% !important;
}
.object-copy {
  min-width: 0;
  display: grid;
  gap: 4px;
}
.object-token {
  width: max-content;
  max-width: 100%;
  font-size: 12px;
  font-family: Arial, Helvetica, sans-serif;
  color: var(--purple);
  font-weight: 700;
  background: #F6ECF7;
  border: 1px solid #EBD7EE;
  border-radius: 999px;
  padding: 1px 7px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.object-name {
  color: var(--text);
  font-size: 13px;
  line-height: 1.25;
  overflow-wrap: anywhere;
}
.object-meta {
  color: var(--muted);
  font-size: 12px;
}
.memory-list {
  list-style: none;
  padding: 0;
  margin: 0;
  border: 1px solid #ECE7EF;
  border-radius: 8px;
  overflow: hidden;
  background: #FFFFFF;
}
.memory-item {
  display: grid;
  grid-template-columns: 26px 1fr;
  gap: 12px;
  align-items: start;
  padding: 14px 15px;
  border-radius: 0;
  border: 0;
  border-bottom: 1px solid #EFEAF1;
  background: #fff;
  color: var(--text);
  font-size: 14px;
}
.memory-item:last-child {
  border-bottom: 0;
}
.memory-item.reasoning {
  background: #FFF9FE;
}
.memory-badge {
  width: 23px;
  height: 23px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 12px;
  font-weight: 700;
  background: var(--pink);
}
.memory-item.reasoning .memory-badge {
  background: var(--purple-soft);
}
.memory-text {
  display: block;
}
.memory-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 7px;
  color: var(--muted);
  font-size: 12px;
}
.memory-meta span {
  display: inline-flex;
  align-items: center;
  border: 1px solid #E8E2EC;
  background: #FAF8FB;
  border-radius: 999px;
  padding: 1px 8px;
}
.safety-block {
  background: #F6F5F7;
  border-color: #E2DEE6;
}
.safety-list {
  list-style: none;
  padding: 0;
  margin: 0;
  border: 1px solid #E2DEE6;
  border-radius: 8px;
  overflow: hidden;
  background: #F1F0F3;
}
.safety-item,
.safety-empty {
  padding: 13px 15px;
  border-bottom: 1px solid #E1DDE5;
  color: var(--text);
  font-size: 14px;
}
.safety-item {
  display: grid;
  grid-template-columns: minmax(54px, auto) 1fr;
  gap: 12px;
  align-items: start;
}
.safety-item:last-child,
.safety-empty:last-child {
  border-bottom: 0;
}
.safety-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 54px;
  height: 24px;
  padding: 0 8px;
  border-radius: 999px;
  border: 1px solid #CDC6D2;
  background: #FFFFFF;
  color: #5F5966;
  font-size: 12px;
  font-weight: 700;
}
.safety-text {
  display: block;
}
.objref {
  color: var(--purple);
  background: #F8F0F8;
  border: 1px solid #EBD5EA;
  padding: 1px 6px;
  border-radius: 6px;
  font-weight: 700;
  cursor: help;
  white-space: nowrap;
}
.objref:hover,
.objref:focus-visible {
  outline: none;
  background: var(--pink-soft);
  border-color: var(--pink);
}
.empty {
  color: var(--muted);
  font-style: italic;
}
.no-img {
  color: var(--muted);
  font-size: 12px;
}
.graph-view {
  margin-top: 30px;
}
.graph-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  padding: 18px;
}
.graph-meta {
  display: flex;
  justify-content: space-between;
  align-items: start;
  gap: 20px;
  margin-bottom: 14px;
}
.graph-note {
  color: var(--muted);
  margin: 0;
  font-size: 13px;
  max-width: 760px;
}
.graph-legend {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: var(--muted);
  font-size: 12px;
  border: 1px solid var(--line);
  background: var(--soft);
  border-radius: 8px;
  padding: 6px 8px;
}
.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
}
.legend-dot.object { background: var(--purple); }
.legend-dot.object-linked { background: var(--pink-soft); border: 1px solid var(--pink); }
.legend-dot.unlinked { background: #F2F2F2; border: 1px solid #BFC0C0; }
.graph-stage {
  position: relative;
  min-height: 680px;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}
.inline-graph {
  width: 100%;
  height: 680px;
  display: block;
  touch-action: none;
}
.graph-link {
  stroke: #D8D8D9;
  stroke-linecap: round;
  opacity: 0.86;
}
.graph-link.object-linked {
  stroke: var(--pink);
  opacity: 0.78;
}
.graph-link.memory-linked {
  stroke: #D8D8D9;
  opacity: 0.88;
}
.graph-node {
  cursor: grab;
}
.graph-node:active {
  cursor: grabbing;
}
.graph-node circle {
  transition: stroke 140ms ease, stroke-width 140ms ease, filter 140ms ease;
}
.graph-node text {
  pointer-events: none;
  text-anchor: middle;
  dominant-baseline: central;
  font-family: Arial, Helvetica, sans-serif;
  letter-spacing: 0;
  user-select: none;
}
.graph-node.object text {
  fill: #fff;
  font-size: 14px;
  font-weight: 700;
}
.graph-node.memory text {
  fill: var(--text);
  font-size: 10px;
  font-weight: 700;
}
.graph-node:hover circle,
.graph-node.focused circle {
  stroke: var(--purple);
  stroke-width: 3px;
  filter: drop-shadow(0 8px 12px rgba(124, 83, 166, 0.18));
}
.graph-hover {
  position: absolute;
  left: 18px;
  bottom: 18px;
  width: min(440px, calc(100% - 36px));
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.94);
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 14px 40px rgba(63, 64, 64, 0.10);
  color: var(--muted);
  font-size: 13px;
}
.graph-hover strong {
  display: block;
  color: var(--text);
  margin-bottom: 4px;
}
.graph-hover p {
  margin: 0;
  overflow-wrap: anywhere;
}
#preview {
  position: fixed;
  display: none;
  pointer-events: none;
  z-index: 1000;
  width: 260px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 20px 52px rgba(63, 64, 64, 0.18);
  padding: 10px;
}
#preview img {
  width: 100%;
  max-height: 240px;
  object-fit: contain;
  display: block;
  background: transparent;
  border-radius: 8px;
}
#preview .plabel {
  margin-top: 8px;
  color: var(--purple);
  font-size: 13px;
  font-weight: 700;
  text-align: center;
}
#memory-preview {
  position: fixed;
  display: none;
  pointer-events: none;
  z-index: 999;
  min-width: 136px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 16px 42px rgba(63, 64, 64, 0.14);
  padding: 9px 11px;
}
#memory-preview .mtime-label {
  display: block;
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0;
  margin-bottom: 3px;
}
#memory-preview .mtime-value {
  display: block;
  color: var(--purple);
  font-size: 15px;
  font-weight: 800;
}
@media (max-width: 900px) {
  .topbar-inner, .clip-header {
    grid-template-columns: 1fr;
    display: grid;
  }
  .tabs, .clip-stats {
    justify-content: flex-start;
  }
  .bento-grid {
    grid-template-columns: 1fr;
  }
  .objects-block {
    grid-column: auto;
    grid-row: auto;
    position: static;
    max-height: none;
    overflow: visible;
  }
  .facts-block,
  .safety-block,
  .reasoning-block,
  .other-block {
    grid-column: auto;
  }
  .object-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 560px) {
  .app-shell, .topbar-inner {
    width: min(100% - 28px, 1480px);
  }
  .clip-header h1 {
    font-size: 28px;
  }
  .object-grid {
    grid-template-columns: 1fr;
  }
  .object-tile {
    min-height: 104px;
    grid-template-columns: 84px 1fr;
  }
  .memory-item {
    padding: 13px 12px;
  }
}
"""


_JS_TEMPLATE = """
const OBJ_CROPS = __OBJ_CROPS_JSON__;
const GRAPH = __GRAPH_JSON__;

const preview = document.createElement('div');
preview.id = 'preview';
preview.innerHTML = '<img id="pimg" alt="" /><div id="plabel" class="plabel"></div>';
document.body.appendChild(preview);

const memoryPreview = document.createElement('div');
memoryPreview.id = 'memory-preview';
memoryPreview.innerHTML = '<span class="mtime-label">Timestamp</span><span id="mtime-value" class="mtime-value"></span>';
document.body.appendChild(memoryPreview);

function previewFor(label) {
  const b64 = OBJ_CROPS[label];
  if (!b64) return false;
  document.getElementById('pimg').src = 'data:image/png;base64,' + b64;
  const displayLabel = label.includes(':') ? label.split(':').pop() : label;
  document.getElementById('plabel').textContent = '<' + displayLabel + '>';
  preview.style.display = 'block';
  return true;
}
function hidePreview() {
  preview.style.display = 'none';
}
function movePreview(e) {
  const pad = 18;
  let left = e.clientX + pad;
  let top = e.clientY + pad;
  const rectW = 280;
  const rectH = 300;
  if (left + rectW > window.innerWidth) left = e.clientX - rectW - pad;
  if (top + rectH > window.innerHeight) top = e.clientY - rectH - pad;
  preview.style.left = Math.max(12, left) + 'px';
  preview.style.top = Math.max(12, top) + 'px';
}
function memoryPreviewFor(timestamp) {
  if (!timestamp) return false;
  document.getElementById('mtime-value').textContent = timestamp;
  memoryPreview.style.display = 'block';
  return true;
}
function hideMemoryPreview() {
  memoryPreview.style.display = 'none';
}
function moveMemoryPreview(e) {
  const pad = 14;
  let left = e.clientX + pad;
  let top = e.clientY + pad;
  const rectW = 168;
  const rectH = 62;
  if (left + rectW > window.innerWidth) left = e.clientX - rectW - pad;
  if (top + rectH > window.innerHeight) top = e.clientY - rectH - pad;
  memoryPreview.style.left = Math.max(12, left) + 'px';
  memoryPreview.style.top = Math.max(12, top) + 'px';
}
document.querySelectorAll('[data-obj]').forEach(el => {
  el.addEventListener('mouseenter', e => {
    if (previewFor(el.dataset.obj)) movePreview(e);
  });
  el.addEventListener('mousemove', movePreview);
  el.addEventListener('mouseleave', hidePreview);
  el.addEventListener('focus', () => {
    previewFor(el.dataset.obj);
    const r = el.getBoundingClientRect();
    preview.style.left = Math.min(window.innerWidth - 292, r.right + 12) + 'px';
    preview.style.top = Math.max(12, r.top) + 'px';
  });
  el.addEventListener('blur', hidePreview);
});
document.querySelectorAll('[data-timestamp]').forEach(el => {
  el.addEventListener('mouseenter', e => {
    if (memoryPreviewFor(el.dataset.timestamp)) moveMemoryPreview(e);
  });
  el.addEventListener('mousemove', moveMemoryPreview);
  el.addEventListener('mouseleave', hideMemoryPreview);
  el.addEventListener('focus', () => {
    memoryPreviewFor(el.dataset.timestamp);
    const r = el.getBoundingClientRect();
    memoryPreview.style.left = Math.min(window.innerWidth - 180, r.right + 12) + 'px';
    memoryPreview.style.top = Math.max(12, r.top) + 'px';
  });
  el.addEventListener('blur', hideMemoryPreview);
});

let graphFrame = null;

function setGraphPanel(title, body) {
  const panel = document.getElementById('graph-hover');
  if (!panel) return;
  panel.replaceChildren();
  const strong = document.createElement('strong');
  strong.textContent = title;
  const p = document.createElement('p');
  p.textContent = body;
  panel.appendChild(strong);
  panel.appendChild(p);
}

function resetGraphPanel() {
  setGraphPanel('Interactive VideoGraph', 'Hover an object node to preview its segmentation crop. Drag nodes to inspect local structure.');
}

function svgPoint(svg, event) {
  const pt = svg.createSVGPoint();
  pt.x = event.clientX;
  pt.y = event.clientY;
  const matrix = svg.getScreenCTM();
  return matrix ? pt.matrixTransform(matrix.inverse()) : { x: event.offsetX, y: event.offsetY };
}

function drawMemoryGraph() {
  const svg = document.getElementById('memory-graph');
  if (!svg || !GRAPH || !GRAPH.nodes || !GRAPH.nodes.length) return;
  if (graphFrame) cancelAnimationFrame(graphFrame);
  graphFrame = null;
  svg.replaceChildren();

  const parentWidth = svg.parentElement ? svg.parentElement.clientWidth : 0;
  const width = Math.max(760, Math.floor(svg.clientWidth || parentWidth || 1000));
  const height = 680;
  svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);

  const ns = 'http://www.w3.org/2000/svg';
  const linkLayer = document.createElementNS(ns, 'g');
  const nodeLayer = document.createElementNS(ns, 'g');
  svg.appendChild(linkLayer);
  svg.appendChild(nodeLayer);

  const nodes = GRAPH.nodes.map(n => ({ ...n, x: 0, y: 0, vx: 0, vy: 0 }));
  const byId = new Map(nodes.map(n => [n.id, n]));
  const objects = nodes.filter(n => n.kind === 'object');
  const memories = nodes.filter(n => n.kind !== 'object');
  const cx = width / 2;
  const cy = height / 2;
  const objectRing = Math.min(width, height) * 0.30;

  objects.forEach((node, i) => {
    const angle = objects.length === 1 ? 0 : (-Math.PI / 2) + (i * Math.PI * 2 / objects.length);
    node.x = cx + Math.cos(angle) * objectRing;
    node.y = cy + Math.sin(angle) * objectRing;
  });
  memories.forEach((node, i) => {
    const refs = (node.refs || [])
      .map(ref => byId.get(Number(ref.split('_')[1])))
      .filter(Boolean);
    const anchor = refs.length ? refs.reduce((acc, n) => {
      acc.x += n.x;
      acc.y += n.y;
      return acc;
    }, { x: 0, y: 0 }) : { x: cx, y: cy };
    const ax = refs.length ? anchor.x / refs.length : cx;
    const ay = refs.length ? anchor.y / refs.length : cy;
    const angle = (-Math.PI / 2) + (i * Math.PI * 2 / Math.max(1, memories.length));
    node.x = ax + Math.cos(angle) * 78;
    node.y = ay + Math.sin(angle) * 78;
  });

  const links = GRAPH.edges
    .map(edge => ({ ...edge, sourceNode: byId.get(edge.source), targetNode: byId.get(edge.target) }))
    .filter(edge => edge.sourceNode && edge.targetNode);

  function linkConnectionKind(edge) {
    const touchesObject = edge.sourceNode.kind === 'object' || edge.targetNode.kind === 'object';
    return touchesObject ? 'object-linked' : 'memory-linked';
  }

  const linkEls = links.map(edge => {
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('class', 'graph-link ' + linkConnectionKind(edge));
    line.setAttribute('stroke-width', String(Math.max(1.1, Math.min(3.2, 1 + edge.weight * 0.35))));
    line.setAttribute('data-relation', edge.relation);
    linkLayer.appendChild(line);
    return line;
  });

  function radius(node) {
    return node.kind === 'object' ? 23 : 14;
  }
  function fill(node) {
    if (node.kind === 'object') return '#7C53A6';
    if ((node.refs || []).length > 0) return '#F2D0F0';
    return '#F2F2F2';
  }
  function stroke(node) {
    if (node.kind === 'object') return '#68438F';
    if ((node.refs || []).length > 0) return '#D99AD5';
    return '#BFC0C0';
  }

  const nodeEls = nodes.map(node => {
    const g = document.createElementNS(ns, 'g');
    g.setAttribute('class', 'graph-node ' + node.kind + (node.memoryKind ? ' ' + node.memoryKind : ''));
    g.setAttribute('tabindex', '0');

    const circle = document.createElementNS(ns, 'circle');
    circle.setAttribute('r', String(radius(node)));
    circle.setAttribute('fill', fill(node));
    circle.setAttribute('stroke', stroke(node));
    circle.setAttribute('stroke-width', node.kind === 'object' ? '1.4' : '1.2');
    g.appendChild(circle);

    const label = document.createElementNS(ns, 'text');
    label.textContent = node.label;
    label.setAttribute('dy', '0.04em');
    g.appendChild(label);

    function describe(e) {
      document.querySelectorAll('.graph-node.focused').forEach(el => el.classList.remove('focused'));
      g.classList.add('focused');
      if (node.kind === 'object') {
        setGraphPanel('<' + node.token + '>', node.name + ' · seen ' + node.seen + ' · clips ' + node.firstClip + '–' + node.lastClip);
        if (previewFor(node.token) && e && e.clientX !== undefined) movePreview(e);
      } else {
        const prefix = node.memoryKind === 'reasoning' ? '[r]' : node.memoryKind === 'fact' ? '[f]' : '[m]';
        setGraphPanel(prefix + ' memory ' + node.id, node.text);
      }
    }
    function clear() {
      g.classList.remove('focused');
      if (node.kind === 'object') hidePreview();
      resetGraphPanel();
    }

    g.addEventListener('mouseenter', describe);
    g.addEventListener('mousemove', e => {
      if (node.kind === 'object') movePreview(e);
    });
    g.addEventListener('mouseleave', clear);
    g.addEventListener('focus', describe);
    g.addEventListener('blur', clear);

    let dragging = false;
    g.addEventListener('pointerdown', e => {
      dragging = true;
      g.setPointerCapture(e.pointerId);
      const p = svgPoint(svg, e);
      node.dragging = true;
      node.dragOffsetX = node.x - p.x;
      node.dragOffsetY = node.y - p.y;
      node.dragX = node.x;
      node.dragY = node.y;
      node.vx = 0;
      node.vy = 0;
      e.preventDefault();
    });
    g.addEventListener('pointermove', e => {
      if (!dragging) return;
      const p = svgPoint(svg, e);
      node.dragX = p.x + (node.dragOffsetX || 0);
      node.dragY = p.y + (node.dragOffsetY || 0);
    });
    function stopDrag(e) {
      dragging = false;
      node.dragging = false;
      try { g.releasePointerCapture(e.pointerId); } catch (_) {}
    }
    g.addEventListener('pointerup', stopDrag);
    g.addEventListener('pointercancel', stopDrag);

    nodeLayer.appendChild(g);
    return g;
  });

  function update() {
    links.forEach((edge, i) => {
      const line = linkEls[i];
      line.setAttribute('x1', edge.sourceNode.x);
      line.setAttribute('y1', edge.sourceNode.y);
      line.setAttribute('x2', edge.targetNode.x);
      line.setAttribute('y2', edge.targetNode.y);
    });
    nodes.forEach((node, i) => {
      nodeEls[i].setAttribute('transform', 'translate(' + node.x.toFixed(2) + ',' + node.y.toFixed(2) + ')');
    });
  }

  let tick = 0;
  function step() {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dist2 = dx * dx + dy * dy;
        if (dist2 < 0.01) {
          dx = (j - i) * 0.1;
          dy = (i + j) * 0.1;
          dist2 = dx * dx + dy * dy;
        }
        const dist = Math.sqrt(dist2);
        const strength = (a.kind === 'object' || b.kind === 'object') ? 4300 : 2100;
        const force = strength / Math.max(80, dist2);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }
    links.forEach(edge => {
      const a = edge.sourceNode;
      const b = edge.targetNode;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const ideal = (a.kind === 'object' && b.kind === 'object') ? 190 : 112;
      const force = (dist - ideal) * 0.012 * Math.min(2.5, Math.max(1, edge.weight));
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    });
    nodes.forEach(node => {
      const refCount = Array.isArray(node.refs) ? node.refs.length : 0;
      if (node.dragging) {
        node.vx += ((node.dragX || node.x) - node.x) * 0.26;
        node.vy += ((node.dragY || node.y) - node.y) * 0.26;
      } else {
        const centerStrength = node.kind === 'object' ? 0.0012 : (refCount ? 0.0022 : 0.006);
        node.vx += (cx - node.x) * centerStrength;
        node.vy += (cy - node.y) * centerStrength;
      }
      const drift = node.kind === 'object' ? 0.004 : 0.006;
      node.vx += Math.sin(tick * 0.018 + node.id * 0.73) * drift;
      node.vy += Math.cos(tick * 0.015 + node.id * 0.61) * drift;
      node.vx *= node.dragging ? 0.64 : 0.78;
      node.vy *= node.dragging ? 0.64 : 0.78;
      node.x += node.vx;
      node.y += node.vy;
      const r = radius(node) + 12;
      node.x = Math.max(r, Math.min(width - r, node.x));
      node.y = Math.max(r, Math.min(height - r, node.y));
    });
    update();
    tick += 1;
    const graphView = document.getElementById('view-graph');
    if (graphView && graphView.classList.contains('active')) {
      graphFrame = requestAnimationFrame(step);
    } else {
      graphFrame = null;
    }
  }
  resetGraphPanel();
  update();
  graphFrame = requestAnimationFrame(step);
}

function showView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  const view = document.getElementById(id);
  const button = document.querySelector('.tabs button[data-target="' + id + '"]');
  if (view) view.classList.add('active');
  if (button) button.classList.add('active');
  if (id === 'view-graph') {
    setTimeout(drawMemoryGraph, 0);
  } else if (graphFrame) {
    cancelAnimationFrame(graphFrame);
    graphFrame = null;
  }
  window.scrollTo({ top: 0, behavior: 'auto' });
}
document.querySelectorAll('.tabs button').forEach(button => {
  button.addEventListener('click', () => showView(button.dataset.target));
});
window.addEventListener('resize', () => {
  const graphView = document.getElementById('view-graph');
  if (graphView && graphView.classList.contains('active')) drawMemoryGraph();
});
const first = document.querySelector('.tabs button');
if (first) showView(first.dataset.target);
"""


def _crop_lookup(graph) -> dict[str, str]:
    lookup: dict[str, str] = {}
    clip_ids = _clip_ids(graph)
    for oid in graph.object_nodes:
        node = graph.nodes[oid]
        token = f"object_{oid}"
        crop = _latest_object_crop_b64(graph, token)
        if crop:
            lookup[token] = crop
        for clip_id in clip_ids:
            clip_crop = _object_crop_b64(graph, token, clip_id=clip_id, inherit_previous=True)
            if clip_crop:
                lookup[f"clip_{clip_id}:{token}"] = clip_crop
    for canon, group in graph.character_mappings.items():
        crop = lookup.get(canon)
        if not crop:
            continue
        for alias in group:
            lookup.setdefault(alias, crop)
            for key, value in list(lookup.items()):
                if key.endswith(f":{canon}"):
                    prefix = key.rsplit(":", 1)[0]
                    lookup.setdefault(f"{prefix}:{alias}", value)
    return lookup


def _json_for_script(data) -> str:
    return json.dumps(data).replace("</", "<\\/")


def render_memory_html(
    graph,
    out_path: str | Path,
    *,
    clip_min: int | None = None,
    clip_max: int | None = None,
    thumb_px: int = 128,
    include_graph_tab: bool = True,
) -> None:
    clip_ids = _clip_ids(graph)
    if clip_min is not None:
        clip_ids = [c for c in clip_ids if c >= clip_min]
    if clip_max is not None:
        clip_ids = [c for c in clip_ids if c <= clip_max]
    if not clip_ids:
        raise SystemExit("no clips in selected range")

    nav_buttons = [
        f'<button data-target="view-clip-{cid}" type="button">Clip {cid}</button>'
        for cid in clip_ids
    ]
    sections = [_render_clip(graph, cid, thumb_px) for cid in clip_ids]

    if include_graph_tab:
        nav_buttons.append('<button data-target="view-graph" type="button">Graph</button>')
        sections.append(f"""
<section class="graph-view view" id="view-graph">
  <div class="clip-header">
    <div>
      <p class="eyebrow">Full memory graph</p>
      <h1>VideoGraph</h1>
    </div>
    <div class="clip-stats">
      <span><b>{len(graph.object_nodes)}</b> objects</span>
      <span><b>{len(graph.text_nodes)}</b> memory</span>
      <span><b>{len(graph.edges)}</b> edges</span>
    </div>
  </div>
  <div class="graph-card">
    <div class="graph-meta">
      <p class="graph-note">Object nodes are compact purple circles labeled O0, O1. Memory nodes and edges connected to object nodes are pink; memory nodes without object references are grey. Hover object nodes for transparent segmentation crops.</p>
      <div class="graph-legend" aria-label="Graph legend">
        <span class="legend-item"><span class="legend-dot object"></span>object</span>
        <span class="legend-item"><span class="legend-dot object-linked"></span>object-linked memory</span>
        <span class="legend-item"><span class="legend-dot unlinked"></span>unlinked memory</span>
      </div>
    </div>
    <div class="graph-stage">
      <svg id="memory-graph" class="inline-graph" role="img" aria-label="Interactive VideoGraph"></svg>
      <div id="graph-hover" class="graph-hover">
        <strong>Interactive VideoGraph</strong>
        <p>Hover an object node to preview its segmentation crop. Drag nodes to inspect local structure.</p>
      </div>
    </div>
  </div>
</section>
""")

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ARISTOS Agent — Graph-Based Memory</title>
  <style>{_CSS}</style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <h1>ARISTOS Agent<span>Graph-Based Memory</span></h1>
      </div>
      <nav class="tabs" aria-label="Memory views">
        {''.join(nav_buttons)}
      </nav>
    </div>
  </header>
  <main class="app-shell">
    {''.join(sections)}
  </main>
  <script>{_JS_TEMPLATE.replace("__OBJ_CROPS_JSON__", _json_for_script(_crop_lookup(graph))).replace("__GRAPH_JSON__", _json_for_script(_graph_payload(graph)))}</script>
</body>
</html>
"""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    graph = load_video_graph(args.graph)
    if graph is None:
        raise SystemExit(f"no graph at {args.graph}")
    print(f"loaded: {graph.summary()}")
    render_memory_html(
        graph,
        args.out,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        thumb_px=args.thumb_px,
        include_graph_tab=not args.no_graph_tab,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
