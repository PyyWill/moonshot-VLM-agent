"""M6 single-round QA.

Loads the graph produced by `scripts/run_m1_objects.py` +
`scripts/run_m4_memories.py`, takes a question, and runs one VLM turn using
BGE-top-K long-term retrieval plus (optionally) the first k seconds of a
still-ongoing clip as short-term context.

Time convention: if the user asks at t = N*10 + k seconds into the stream,
pass `--max-clip N-1` (last finished clip), `--current-clip data/videos/.../N.mp4`
and `--short-term-seconds k`. Set k=0 to disable short-term.

Usage:
    python scripts/run_m6_qa.py -q "What did the person pick up first?"
    python scripts/run_m6_qa.py -q "What tool am I holding right now?" \
        --current-clip data/videos/test_sample/5.mp4 \
        --short-term-seconds 3 --max-clip 4
"""
from __future__ import annotations

import argparse
import json

from common import (
    REPO_ROOT,
    apply_cuda_lib_dir,
    build_qwen3vl_from_config,
    build_text_embedder_from_config,
    configure_logging,
    load_json,
)

apply_cuda_lib_dir()

from mmagent.control import answer_question_single_round
from mmagent.utils.general import load_video_graph


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--question", type=str, required=True)
    p.add_argument("--graph", type=str,
                   default=str(REPO_ROOT / "data" / "debug" / "test_sample" / "_graph.pkl"))
    p.add_argument("--processing-config", type=str,
                   default=str(REPO_ROOT / "configs" / "processing_config.json"))
    p.add_argument("--current-clip", type=str, default=None,
                   help="Path to the ongoing (unfinished) clip for short-term context.")
    p.add_argument("--short-term-seconds", type=float, default=0.0)
    p.add_argument("--max-clip", type=int, default=None,
                   help="Restrict long-term retrieval to clips <= max-clip.")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--no-episodic", action="store_true")
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--dump-json", type=str, default=None,
                   help="Optional path to dump the full QA record.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    log = configure_logging(args.log_level, "run_m6_qa")

    pcfg = load_json(args.processing_config)

    graph = load_video_graph(args.graph)
    if graph is None:
        raise SystemExit(f"no graph at {args.graph}")
    log.info(f"loaded graph: {graph.summary()}")

    log.info("Loading BGE-M3 ...")
    build_text_embedder_from_config(pcfg)
    vlm_device = pcfg.get("vlm_device", "cuda:0")
    log.info(f"Loading {pcfg['vlm_ckpt']} on {vlm_device} ...")
    build_qwen3vl_from_config(pcfg)

    result = answer_question_single_round(
        graph,
        question=args.question,
        current_clip_path=args.current_clip,
        short_term_seconds=args.short_term_seconds,
        max_clip=args.max_clip,
        top_k=args.top_k,
        include_episodic=not args.no_episodic,
        include_semantic=not args.no_semantic,
        max_new_tokens=args.max_new_tokens,
        temperature=float(pcfg.get("temperature", 0.0)),
    )

    print("\n=== M6 QA ===")
    print(f"Q: {result.question}")
    print(f"A: {result.answer}")
    print(
        f"\n[retrieved {result.n_long_term_hits} long-term hits, "
        f"{result.n_object_crops} object crops, "
        f"{result.n_short_term_frames} short-term frames, "
        f"max_clip={result.max_clip}]"
    )
    print("\ntop hits:")
    for h in result.hits[:10]:
        print(f"  [{h.type:8s} clip={h.clip_id:>2} score={h.score:.3f}] {h.text}")

    if args.dump_json:
        out = {
            "question": result.question,
            "answer": result.answer,
            "n_long_term_hits": result.n_long_term_hits,
            "n_object_crops": result.n_object_crops,
            "n_short_term_frames": result.n_short_term_frames,
            "max_clip": result.max_clip,
            "hits": [
                {
                    "node_id": h.node_id, "type": h.type, "clip_id": h.clip_id,
                    "score": h.score, "text": h.text,
                    "refs": h.refs, "expanded_refs": h.expanded_refs,
                }
                for h in result.hits
            ],
            "debug": result.debug,
        }
        with open(args.dump_json, "w") as f:
            json.dump(out, f, indent=2)
        log.info(f"wrote QA record -> {args.dump_json}")


if __name__ == "__main__":
    main()
