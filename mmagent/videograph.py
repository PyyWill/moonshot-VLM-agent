"""
Object-centric VideoGraph.

Node types:
    object     — stores DINOv2 embeddings + masked crops for one visual entity
    episodic   — per-clip event description text (contains <object_N> refs)
    semantic   — durable fact text (contains <object_N> refs)

Edges:
    text_node  --mention--> object_node     (weight int, >0)
    object_node <-> object_node             (weight int; derived from
                                             "Equivalence: <object_x>, <object_y>"
                                             semantic nodes via refresh_equivalences)

No edges between two text nodes.
"""
from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

OBJECT_REF_RE = re.compile(r"<(object_\d+)>")
EQUIVALENCE_RE = re.compile(
    r"^\s*Equivalence\s*:\s*(<object_\d+>(?:\s*,\s*<object_\d+>)+)\s*$",
    re.IGNORECASE,
)


def _extract_refs(text: str) -> list[str]:
    """Return unique object_N tokens in insertion order."""
    seen, out = set(), []
    for m in OBJECT_REF_RE.finditer(text):
        tok = m.group(1)
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


class Node:
    __slots__ = ("id", "type", "clip_id", "metadata")

    def __init__(self, node_id: int, node_type: str, clip_id: int | None = None):
        self.id = node_id
        self.type = node_type  # 'object' | 'episodic' | 'semantic'
        self.clip_id = clip_id
        self.metadata: dict[str, Any] = {}


class VideoGraph:
    def __init__(
        self,
        max_object_embeddings: int = 10,
        max_object_crops: int = 10,
        object_matching_threshold: float = 0.60,
    ):
        self.nodes: dict[int, Node] = {}

        # edges[(src, dst)] = weight  (stored with src <= dst for object-object,
        # and src=text, dst=object for mention edges; we normalize via _edge_key)
        self.edges: dict[tuple[int, int, str], int] = {}

        self.text_nodes: list[int] = []
        self.object_nodes: list[int] = []

        self.text_nodes_by_clip: dict[int, list[int]] = defaultdict(list)
        self.object_nodes_by_clip: dict[int, list[int]] = defaultdict(list)
        self.event_sequence_by_clip: dict[int, list[int]] = defaultdict(list)

        self.max_object_embeddings = max_object_embeddings
        self.max_object_crops = max_object_crops
        self.object_matching_threshold = object_matching_threshold

        # Equivalence state (populated by refresh_equivalences)
        #   "object_k" -> {"object_x", "object_y", ...}
        self.character_mappings: dict[str, set[str]] = {}
        #   "object_x" -> "object_k"
        self.reverse_character_mappings: dict[str, str] = {}

        self._next_node_id = 0

    # ---------- object nodes ----------

    def add_object_node(self, payload: dict) -> int:
        """
        payload = {
          "embeddings": [np.ndarray],         # 1+ DINOv2 vecs (unit norm)
          "contents":   [str],                # 1+ base64 png crops
          "name":       str | None,
          "first_clip": int,
          "last_clip":  int,
        }
        """
        node = Node(self._next_node_id, "object")
        emb = list(payload["embeddings"])[: self.max_object_embeddings]
        crops = list(payload.get("contents", []))[: self.max_object_crops]
        node.metadata = {
            "embeddings": emb,
            "contents": crops,
            "name": payload.get("name"),
            "first_clip": payload.get("first_clip"),
            "last_clip": payload.get("last_clip"),
            "seen_count": int(payload.get("seen_count", 1)),
        }
        self.nodes[node.id] = node
        self.object_nodes.append(node.id)
        clip = payload.get("first_clip")
        if clip is not None:
            self.object_nodes_by_clip[clip].append(node.id)
        self._next_node_id += 1
        logger.debug(f"added object node {node.id}")
        return node.id

    def search_object_nodes(
        self,
        embedding: np.ndarray,
        threshold: float | None = None,
        exclude_clip_id: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return [(object_node_id, max_sim), ...] sorted desc, filtered by threshold.

        If `exclude_clip_id` is set, skip nodes whose `first_clip` equals it.
        Prevents the pathology where intra-clip dedupe keeps two clusters
        separate, but the looser cross-clip match threshold re-merges them
        against a node just added from the same clip.
        """
        th = self.object_matching_threshold if threshold is None else threshold
        out = []
        e = np.asarray(embedding).reshape(1, -1)
        for oid in self.object_nodes:
            node = self.nodes[oid]
            if exclude_clip_id is not None and node.metadata.get("first_clip") == exclude_clip_id:
                continue
            existing = node.metadata["embeddings"]
            if not existing:
                continue
            sim = float(cosine_similarity(e, np.stack(existing)).max())
            if sim >= th:
                out.append((oid, sim))
        out.sort(key=lambda x: -x[1])
        return out

    def update_object_node(
        self,
        node_id: int,
        embedding: np.ndarray | None = None,
        crop_b64: str | None = None,
        clip_id: int | None = None,
    ) -> None:
        node = self.nodes[node_id]
        assert node.type == "object"
        md = node.metadata

        if embedding is not None:
            embs: list = md["embeddings"]
            if len(embs) < self.max_object_embeddings:
                embs.append(embedding)
            else:
                # Random replacement keeps memory bounded while preserving older samples.
                idx = random.randrange(len(embs))
                embs[idx] = embedding

        if crop_b64 is not None:
            crops: list = md["contents"]
            if len(crops) < self.max_object_crops:
                crops.append(crop_b64)
            else:
                idx = random.randrange(len(crops))
                crops[idx] = crop_b64

        if clip_id is not None:
            md["last_clip"] = clip_id
            md["seen_count"] = int(md.get("seen_count", 0)) + 1
            if node_id not in self.object_nodes_by_clip[clip_id]:
                self.object_nodes_by_clip[clip_id].append(node_id)

    # ---------- text nodes ----------

    def add_text_node(
        self,
        node_type: str,
        clip_id: int,
        contents: str,
        embedding: np.ndarray,
    ) -> int:
        assert node_type in ("episodic", "semantic")
        node = Node(self._next_node_id, node_type, clip_id)
        node.metadata = {"contents": [contents], "embedding": embedding}
        self.nodes[node.id] = node
        self.text_nodes.append(node.id)
        self.text_nodes_by_clip[clip_id].append(node.id)
        if node_type == "episodic":
            self.event_sequence_by_clip[clip_id].append(node.id)
        self._next_node_id += 1
        return node.id

    def reinforce_text_node(self, node_id: int, new_content: str | None = None) -> None:
        """Bump weights of all mention edges for a semantic node; optionally append content variant."""
        node = self.nodes[node_id]
        if new_content is not None:
            node.metadata["contents"].append(new_content)
        for key in list(self.edges.keys()):
            src, dst, rel = key
            if rel == "mention" and src == node_id:
                self.edges[key] += 1

    def weaken_text_node(self, node_id: int) -> None:
        for key in list(self.edges.keys()):
            src, dst, rel = key
            if rel == "mention" and src == node_id:
                self.edges[key] -= 1
                if self.edges[key] <= 0:
                    del self.edges[key]

    def search_semantic_nodes_by_refs(self, refs: list[str]) -> list[int]:
        """Return semantic node ids whose referenced object set equals `refs` (as a set)."""
        target = set(refs)
        out = []
        for tid in self.text_nodes:
            n = self.nodes[tid]
            if n.type != "semantic":
                continue
            text = n.metadata["contents"][-1]
            if set(_extract_refs(text)) == target:
                out.append(tid)
        return out

    # ---------- edges ----------

    def add_edge(self, src: int, dst: int, relation: str = "mention", weight: int = 1) -> None:
        """
        Relation kinds:
            'mention'     : text_node -> object_node  (src=text, dst=object)
            'equivalence' : object_node <-> object_node (order-insensitive)

        Disallow text_node<->text_node.
        """
        s_type = self.nodes[src].type
        d_type = self.nodes[dst].type
        if s_type in ("episodic", "semantic") and d_type in ("episodic", "semantic"):
            raise ValueError(f"text-to-text edge not allowed: {src}({s_type}) -> {dst}({d_type})")
        if relation == "equivalence" and src > dst:
            src, dst = dst, src
        key = (src, dst, relation)
        self.edges[key] = self.edges.get(key, 0) + weight

    # ---------- truncation (streaming) ----------

    def truncate_memory_by_clip(self, clip_id: int, refresh: bool = True) -> None:
        """Drop everything produced at clip_id or later (for `before_clip` semantics)."""
        kill = {
            nid for nid, n in self.nodes.items()
            if n.clip_id is not None and n.clip_id >= clip_id
        }
        # also drop object_nodes whose first_clip >= clip_id
        for oid in list(self.object_nodes):
            n = self.nodes[oid]
            if n.metadata.get("first_clip", -1) >= clip_id:
                kill.add(oid)
            else:
                # clamp seen_count / last_clip by removing higher-clip entries
                last = n.metadata.get("last_clip")
                if last is not None and last >= clip_id:
                    n.metadata["last_clip"] = clip_id - 1

        for nid in kill:
            self.nodes.pop(nid, None)
        self.text_nodes = [t for t in self.text_nodes if t not in kill]
        self.object_nodes = [o for o in self.object_nodes if o not in kill]
        for d in (self.text_nodes_by_clip, self.object_nodes_by_clip, self.event_sequence_by_clip):
            for c in list(d.keys()):
                if c >= clip_id:
                    del d[c]
        self.edges = {
            k: w for k, w in self.edges.items()
            if k[0] not in kill and k[1] not in kill
        }
        if refresh:
            self.refresh_equivalences()

    # ---------- equivalences ----------

    def refresh_equivalences(self) -> None:
        """Union-find over every semantic node of the form 'Equivalence: <object_x>, <object_y>'."""
        parent: dict[str, str] = {}
        def find(a):
            while parent.get(a, a) != a:
                parent[a] = parent.get(parent[a], parent[a])
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for tid in self.text_nodes:
            n = self.nodes[tid]
            if n.type != "semantic":
                continue
            text = n.metadata["contents"][-1]
            m = EQUIVALENCE_RE.match(text)
            if not m:
                continue
            refs = _extract_refs(text)
            if len(refs) < 2:
                continue
            first = refs[0]
            parent.setdefault(first, first)
            for r in refs[1:]:
                parent.setdefault(r, r)
                union(first, r)

        # Build forward / reverse mappings
        groups: dict[str, set[str]] = defaultdict(set)
        for tok in list(parent.keys()):
            groups[find(tok)].add(tok)

        self.character_mappings = {}
        self.reverse_character_mappings = {}
        for i, (root, members) in enumerate(sorted(groups.items())):
            # Use deterministic id: "object_<min_member_id>"
            ids = sorted(int(m.split("_")[1]) for m in members)
            canonical = f"object_{ids[0]}"
            self.character_mappings[canonical] = members
            for m in members:
                self.reverse_character_mappings[m] = canonical

    # ---------- fusion ----------

    def fuse_equivalence_groups(self) -> int:
        """Merge every equivalence group's non-canonical members INTO the
        canonical object node: union embeddings/crops (capped), sum seen_count,
        min first_clip, max last_clip, rewire every mention edge (and residual
        equivalence edge) to the canonical oid, then delete the merged-away
        nodes. Returns the number of deleted nodes.

        Call this AFTER all M4 memory writes are done. Idempotent: running
        twice is a no-op because character_mappings becomes single-member
        groups after a fuse.
        """
        if not self.character_mappings:
            return 0

        canonical_of: dict[int, int] = {}
        for canon_tok, members in self.character_mappings.items():
            canon_oid = int(canon_tok.split("_")[1])
            for m in members:
                oid = int(m.split("_")[1])
                canonical_of[oid] = canon_oid

        to_delete = {oid for oid, c in canonical_of.items() if oid != c and oid in self.nodes}
        if not to_delete:
            return 0

        for canon_tok, members in self.character_mappings.items():
            if len(members) < 2:
                continue
            canon_oid = int(canon_tok.split("_")[1])
            canon_node = self.nodes.get(canon_oid)
            if canon_node is None or canon_node.type != "object":
                continue
            md = canon_node.metadata
            embs = list(md["embeddings"])
            crops = list(md["contents"])
            seen = int(md.get("seen_count", 0))
            first = md.get("first_clip")
            last = md.get("last_clip")
            for m in members:
                oid = int(m.split("_")[1])
                if oid == canon_oid or oid not in self.nodes:
                    continue
                n = self.nodes[oid]
                embs.extend(n.metadata.get("embeddings", []))
                crops.extend(n.metadata.get("contents", []))
                seen += int(n.metadata.get("seen_count", 0))
                f = n.metadata.get("first_clip")
                if f is not None:
                    first = f if first is None else min(first, f)
                l = n.metadata.get("last_clip")
                if l is not None:
                    last = l if last is None else max(last, l)
            md["embeddings"] = embs[-self.max_object_embeddings:]
            md["contents"] = crops[-self.max_object_crops:]
            md["seen_count"] = seen
            if first is not None:
                md["first_clip"] = first
            if last is not None:
                md["last_clip"] = last

        new_edges: dict[tuple[int, int, str], int] = {}
        for (s, d, r), w in self.edges.items():
            ns = canonical_of.get(s, s)
            nd = canonical_of.get(d, d)
            if r == "equivalence" and ns == nd:
                continue
            if r == "equivalence" and ns > nd:
                ns, nd = nd, ns
            key = (ns, nd, r)
            new_edges[key] = new_edges.get(key, 0) + w
        self.edges = new_edges

        self.object_nodes = [o for o in self.object_nodes if o not in to_delete]
        for cid, lst in list(self.object_nodes_by_clip.items()):
            kept: list[int] = []
            for o in lst:
                canon_o = canonical_of.get(o, o)
                if canon_o in to_delete:
                    continue
                if canon_o not in kept:
                    kept.append(canon_o)
            self.object_nodes_by_clip[cid] = kept

        for oid in to_delete:
            self.nodes.pop(oid, None)

        logger.info(f"fused {len(to_delete)} object nodes into canonicals")
        return len(to_delete)

    # ---------- debugging helpers ----------

    def summary(self) -> str:
        return (
            f"VideoGraph(objects={len(self.object_nodes)}, "
            f"episodic={sum(1 for t in self.text_nodes if self.nodes[t].type=='episodic')}, "
            f"semantic={sum(1 for t in self.text_nodes if self.nodes[t].type=='semantic')}, "
            f"edges={len(self.edges)})"
        )
