"""M1 object extraction: run SAM2 + DINOv2 + dedupe + cross-clip match
across every clip in `data/videos/test_sample/` into one shared VideoGraph.

Dumps per-clip crops to `<debug_root>/clip<NNN>/` and an aggregate
`<debug_root>/_all_clips_summary.json` so we can confirm the same physical
object gets a single global id across clips.

Usage:
    python scripts/run_m1_objects.py
    python scripts/run_m1_objects.py --clips 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from common import (
    REPO_ROOT,
    build_dinov2_from_config,
    build_sam2_from_config,
    configure_logging,
    discover_clip_ids,
    load_configs,
)

from mmagent.object_processing import process_objects_for_clip, save_debug_record
from mmagent.utils.general import save_video_graph
from mmagent.videograph import VideoGraph


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clips-dir", type=str,
                   default=str(REPO_ROOT / "data" / "videos" / "test_sample"))
    p.add_argument("--clips", type=int, nargs="*", default=None,
                   help="Subset of clip indices to run (default: all .mp4 in clips-dir).")
    p.add_argument("--processing-config", type=str,
                   default=str(REPO_ROOT / "configs" / "processing_config.json"))
    p.add_argument("--memory-config", type=str,
                   default=str(REPO_ROOT / "configs" / "memory_config.json"))
    p.add_argument("--debug-root", type=str,
                   default=str(REPO_ROOT / "data" / "debug" / "test_sample"))
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log = configure_logging(args.log_level, "run_m1_objects")

    pcfg, mcfg = load_configs(args.processing_config, args.memory_config)

    clips_dir = Path(args.clips_dir)
    clip_ids = args.clips if args.clips is not None else discover_clip_ids(clips_dir)
    if not clip_ids:
        raise SystemExit(f"no clips found under {clips_dir}")
    log.info(f"will process clip_ids = {clip_ids}")

    debug_root = Path(args.debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)

    # --- build models once ---
    log.info("Loading SAM2 ...")
    build_sam2_from_config(pcfg, mcfg)
    log.info("Loading DINOv2 ...")
    build_dinov2_from_config(pcfg)

    # --- single shared graph across all clips ---
    graph = VideoGraph(
        max_object_embeddings=mcfg["max_object_embeddings"],
        max_object_crops=mcfg["max_object_crops"],
        object_matching_threshold=mcfg["object_matching_threshold"],
    )

    per_clip_summary = []
    pbar = tqdm(clip_ids, desc="M1 clips", unit="clip")
    for cid in pbar:
        clip_path = clips_dir / f"{cid}.mp4"
        if not clip_path.is_file():
            log.warning(f"skip missing clip {clip_path}")
            continue
        debug_dir = debug_root / f"clip{cid:03d}"
        debug_dir.mkdir(parents=True, exist_ok=True)

        pbar.set_postfix_str(f"cid={cid} graph_objs={len(graph.object_nodes)}")
        log.info(f"=== clip {cid} ===")
        record = process_objects_for_clip(
            graph,
            clip_path=str(clip_path),
            clip_id=cid,
            keyframe_timestamps_sec=mcfg["keyframe_timestamps_sec"],
            memory_config=mcfg,
            debug_dir=str(debug_dir),
        )
        save_debug_record(record, str(debug_dir / "record.json"))

        n_new = sum(1 for c in record["clusters"] if c["is_new"])
        n_match = len(record["clusters"]) - n_new
        per_clip_summary.append({
            "clip_id": cid,
            "n_masks_raw": record["n_masks_raw"],
            "n_masks_kept": record["n_masks_kept"],
            "n_clusters": record["n_clusters"],
            "n_new_objects": n_new,
            "n_matched_objects": n_match,
            "graph_total_objects": len(graph.object_nodes),
        })
        log.info(
            f"clip {cid}: kept={record['n_masks_kept']} "
            f"clusters={record['n_clusters']} "
            f"new={n_new} matched={n_match} "
            f"graph_total={len(graph.object_nodes)}"
        )

    # --- aggregate cross-clip stats ---
    obj_occurrence = []  # (oid, n_clips_seen, seen_count, clip_span)
    for oid in graph.object_nodes:
        n = graph.nodes[oid]
        md = n.metadata
        seen_clips = set()
        for c_id, olist in graph.object_nodes_by_clip.items():
            if oid in olist:
                seen_clips.add(c_id)
        obj_occurrence.append({
            "object_id": oid,
            "n_clips_seen": len(seen_clips),
            "clips": sorted(seen_clips),
            "seen_count": md.get("seen_count", 1),
            "first_clip": md.get("first_clip"),
            "last_clip": md.get("last_clip"),
        })
    obj_occurrence.sort(key=lambda x: -x["n_clips_seen"])

    reid_count = sum(1 for o in obj_occurrence if o["n_clips_seen"] >= 2)

    agg = {
        "clips_processed": clip_ids,
        "per_clip": per_clip_summary,
        "total_object_nodes": len(graph.object_nodes),
        "objects_seen_in_ge2_clips": reid_count,
        "top_reid_objects": obj_occurrence[:20],
    }
    out_path = debug_root / "_all_clips_summary.json"
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    log.info(f"wrote aggregate summary -> {out_path}")

    graph_path = debug_root / "_graph.pkl"
    save_video_graph(graph, str(graph_path))
    log.info(f"wrote graph -> {graph_path}")

    # --- console report ---
    print("\n=== M1 CROSS-CLIP REPORT ===")
    print(f"clips processed       : {clip_ids}")
    print(f"total object nodes    : {len(graph.object_nodes)}")
    print(f"objects seen in >=2   : {reid_count}")
    print(f"graph summary         : {graph.summary()}")
    print("\nper-clip:")
    print(f"{'cid':>3} {'raw':>4} {'kept':>4} {'clus':>4} {'new':>4} {'mat':>4} {'total':>5}")
    for s in per_clip_summary:
        print(
            f"{s['clip_id']:>3} {s['n_masks_raw']:>4} {s['n_masks_kept']:>4} "
            f"{s['n_clusters']:>4} {s['n_new_objects']:>4} "
            f"{s['n_matched_objects']:>4} {s['graph_total_objects']:>5}"
        )
    if reid_count:
        print("\ntop cross-clip objects (seen in >=2 clips):")
        for o in obj_occurrence:
            if o["n_clips_seen"] < 2:
                break
            print(
                f"  gid={o['object_id']:04d} n_clips={o['n_clips_seen']} "
                f"seen_count={o['seen_count']} clips={o['clips']}"
            )


if __name__ == "__main__":
    main()
