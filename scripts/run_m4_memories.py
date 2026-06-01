"""M4 memory generation: run Qwen3-VL on every clip, accumulating into a
single shared VideoGraph produced by `scripts/run_m1_objects.py`.

Loads BGE-M3 + Qwen3-VL once, then iterates clips in order so that semantic
reinforcement / equivalence detection can operate across the full timeline.

Usage:
    python scripts/run_m1_objects.py                # build & save graph.pkl (once)
    python scripts/run_m4_memories.py               # run M4 on all clips
    python scripts/run_m4_memories.py --clips 0 1 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from common import (
    REPO_ROOT,
    apply_cuda_lib_dir,
    build_qwen3vl_from_config,
    build_text_embedder_from_config,
    configure_logging,
    discover_clip_ids,
    load_configs,
)

apply_cuda_lib_dir()

from mmagent.memory_processing import process_memories_for_clip
from mmagent.utils.general import load_video_graph, save_video_graph


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clips-dir", type=str,
                   default=str(REPO_ROOT / "data" / "videos" / "test_sample"))
    p.add_argument("--clips", type=int, nargs="*", default=None,
                   help="Subset of clip indices to run (default: all .mp4 in clips-dir).")
    p.add_argument("--graph", type=str,
                   default=str(REPO_ROOT / "data" / "debug" / "test_sample" / "_graph.pkl"))
    p.add_argument("--processing-config", type=str,
                   default=str(REPO_ROOT / "configs" / "processing_config.json"))
    p.add_argument("--memory-config", type=str,
                   default=str(REPO_ROOT / "configs" / "memory_config.json"))
    p.add_argument("--save-graph", type=str, default=None,
                   help="Where to write the final graph pickle (default: overwrite --graph).")
    p.add_argument("--records-dir", type=str, default=None,
                   help="Dir for per-clip memory records (default: alongside graph).")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="Save graph.pkl after every N clips (0 = only at end).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip clips whose _m4_clip<NNN>.json already exists.")
    p.add_argument("--fuse", action="store_true",
                   help="Fuse equivalence groups into canonical nodes after all clips are processed.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log = configure_logging(args.log_level, "run_m4_memories")

    pcfg, mcfg = load_configs(args.processing_config, args.memory_config)

    clips_dir = Path(args.clips_dir)
    clip_ids = args.clips if args.clips is not None else discover_clip_ids(clips_dir)
    if not clip_ids:
        raise SystemExit(f"no clips found under {clips_dir}")
    log.info(f"will process clip_ids = {clip_ids}")

    graph = load_video_graph(args.graph)
    if graph is None:
        raise SystemExit(
            f"no graph at {args.graph}; run scripts/run_m1_objects.py first"
        )
    log.info(f"loaded graph: {graph.summary()}")

    save_graph_path = args.save_graph or args.graph
    records_dir = Path(args.records_dir) if args.records_dir else Path(save_graph_path).parent
    records_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading BGE-M3 ...")
    build_text_embedder_from_config(pcfg)
    vlm_device = pcfg.get("vlm_device", "cuda:0")
    log.info(f"Loading {pcfg['vlm_ckpt']} on {vlm_device} ...")
    build_qwen3vl_from_config(pcfg)

    per_clip: list[dict] = []
    t0_all = time.time()
    pbar = tqdm(list(enumerate(clip_ids)), desc="M4 clips", unit="clip")
    for i, cid in pbar:
        clip_path = clips_dir / f"{cid}.mp4"
        if not clip_path.is_file():
            log.warning(f"skip missing clip {clip_path}")
            continue

        rec_path = records_dir / f"_m4_clip{cid:03d}.json"
        if args.skip_existing and rec_path.is_file():
            log.info(f"skip clip {cid}: {rec_path.name} already exists")
            continue

        n_objs = len(graph.object_nodes_by_clip.get(cid, []))
        pbar.set_postfix_str(f"cid={cid} objs={n_objs}")
        log.info(f"=== clip {cid} ({i+1}/{len(clip_ids)}) — {n_objs} object nodes ===")
        t0 = time.time()
        record = process_memories_for_clip(
            graph,
            clip_path=str(clip_path),
            clip_id=cid,
            memory_config=mcfg,
            max_new_tokens=args.max_new_tokens,
            temperature=float(pcfg.get("temperature", 0.0)),
        )
        dt = time.time() - t0

        with open(rec_path, "w") as f:
            json.dump(record, f, indent=2)
        log.info(
            f"clip {cid}: epi={record['n_episodic']} sem={record['n_semantic']} "
            f"feats={record['n_object_features']} eq_groups={len(record['equivalence_groups'])} "
            f"dt={dt:.1f}s -> {rec_path.name}"
        )

        per_clip.append({
            "clip_id": cid,
            "n_object_features": record["n_object_features"],
            "n_episodic": record["n_episodic"],
            "n_semantic": record["n_semantic"],
            "seconds": round(dt, 2),
        })

        if args.checkpoint_every and ((i + 1) % args.checkpoint_every == 0):
            save_video_graph(graph, save_graph_path)
            log.info(f"checkpoint -> {save_graph_path}")

    if args.fuse:
        n_deleted = graph.fuse_equivalence_groups()
        graph.refresh_equivalences()
        log.info(f"fusion: merged {n_deleted} non-canonical object nodes; {graph.summary()}")

    save_video_graph(graph, save_graph_path)
    log.info(f"final graph -> {save_graph_path}")

    agg = {
        "clips_processed": clip_ids,
        "per_clip": per_clip,
        "graph_summary": graph.summary(),
        "equivalence_groups": {k: sorted(v) for k, v in graph.character_mappings.items()},
        "total_seconds": round(time.time() - t0_all, 2),
    }
    agg_path = records_dir / "_m4_all_clips_summary.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    log.info(f"wrote aggregate summary -> {agg_path}")

    print("\n=== M4 BATCH REPORT ===")
    print(f"clips processed   : {clip_ids}")
    print(f"graph summary     : {graph.summary()}")
    print(f"equivalence groups: {len(agg['equivalence_groups'])}")
    print(f"total time        : {agg['total_seconds']}s")
    print(f"\n{'cid':>3} {'feats':>5} {'epi':>3} {'sem':>3} {'sec':>6}")
    for r in per_clip:
        print(
            f"{r['clip_id']:>3} {r['n_object_features']:>5} "
            f"{r['n_episodic']:>3} {r['n_semantic']:>3} {r['seconds']:>6.1f}"
        )
    if agg["equivalence_groups"]:
        print("\nequivalence groups:")
        for root, members in agg["equivalence_groups"].items():
            print(f"  root={root}: {members}")


if __name__ == "__main__":
    main()
