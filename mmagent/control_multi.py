"""Multi-round (agent-style) QA controller.

Each turn the VLM outputs a JSON `{thought, action, args}`. The system executes
the tool against the VideoGraph / BGE retriever and appends the observation as
plain text to the transcript. The loop terminates when the VLM emits `finish`
or when `max_rounds` is reached (fallback: synthesize an answer from the
gathered evidence).

Tools:
  - search_memory(query, type, top_k, max_clip)
  - expand_object(object_id)
  - get_clip_timeline(clip_id)
  - finish(answer)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .prompts import prompt_multi_round_system
from .retrieve import retrieve_long_term
from .utils import qwen3vl_wrapper
from .utils.general import validate_and_fix_dict
from .videograph import VideoGraph, _extract_refs

logger = logging.getLogger(__name__)


@dataclass
class MultiRoundResult:
    question: str
    answer: str
    n_rounds: int
    terminated: str   # 'finish' | 'max_rounds' | 'parse_error'
    transcript: list[dict] = field(default_factory=list)


# ---------- tools ----------

def _tool_search_memory(graph: VideoGraph, args: dict) -> Any:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "empty query"}
    mem_type = str(args.get("type", "both")).lower()
    top_k = int(args.get("top_k", 5))
    max_clip = args.get("max_clip")
    if max_clip is not None:
        max_clip = int(max_clip)
    hits = retrieve_long_term(
        graph,
        question=query,
        top_k=top_k,
        include_episodic=(mem_type in ("both", "episodic")),
        include_semantic=(mem_type in ("both", "semantic")),
        max_clip=max_clip,
    )
    return [
        {
            "node_id": h.node_id,
            "type": h.type,
            "clip_id": h.clip_id,
            "score": round(h.score, 3),
            "text": h.text,
            "refs": h.refs,
        }
        for h in hits
    ]


def _tool_expand_object(graph: VideoGraph, args: dict) -> Any:
    tok = str(args.get("object_id", "")).strip().strip("<>")
    if not tok.startswith("object_"):
        return {"error": f"invalid object_id: {tok!r}"}
    canon = graph.reverse_character_mappings.get(tok, tok)
    equivalents = sorted(
        graph.character_mappings.get(canon, {tok}),
        key=lambda x: int(x.split("_")[1]),
    )

    mentions: list[dict] = []
    canon_oid = int(canon.split("_")[1])
    for tid in graph.text_nodes:
        node = graph.nodes[tid]
        text = node.metadata["contents"][-1]
        refs = _extract_refs(text)
        refs_resolved = {graph.reverse_character_mappings.get(r, r) for r in refs}
        if canon in refs_resolved:
            mentions.append({
                "node_id": tid,
                "type": node.type,
                "clip_id": int(node.clip_id) if node.clip_id is not None else -1,
                "text": text,
            })
    mentions.sort(key=lambda m: (m["clip_id"], m["type"], m["node_id"]))
    return {
        "canonical": canon,
        "equivalents": equivalents,
        "n_mentions": len(mentions),
        "mentions": mentions[:30],  # cap for context budget
    }


def _tool_get_clip_timeline(graph: VideoGraph, args: dict) -> Any:
    cid = int(args.get("clip_id", -1))
    epi_ids = graph.event_sequence_by_clip.get(cid, [])
    all_ids = graph.text_nodes_by_clip.get(cid, [])
    epi = [{"node_id": t, "text": graph.nodes[t].metadata["contents"][-1]} for t in epi_ids]
    sem = [
        {"node_id": t, "text": graph.nodes[t].metadata["contents"][-1]}
        for t in all_ids
        if graph.nodes[t].type == "semantic"
    ]
    return {"clip_id": cid, "episodic": epi, "semantic": sem}


_TOOLS = {
    "search_memory": _tool_search_memory,
    "expand_object": _tool_expand_object,
    "get_clip_timeline": _tool_get_clip_timeline,
}


# ---------- loop ----------

def _render_observation(obs: Any, char_limit: int = 2500) -> str:
    s = json.dumps(obs, ensure_ascii=False, indent=2)
    if len(s) > char_limit:
        s = s[:char_limit] + "\n... (truncated)"
    return s


def answer_question_multi_round(
    graph: VideoGraph,
    question: str,
    *,
    max_rounds: int = 5,
    max_new_tokens_per_round: int = 512,
    fallback_top_k: int = 10,
    temperature: float = 0.0,
) -> MultiRoundResult:
    system_prompt = (
        prompt_multi_round_system
        .replace("__MAX_ROUNDS__", str(max_rounds))
        .replace("__QUESTION__", question.strip())
    )
    transcript: list[dict] = []
    trace_text = ""

    for round_idx in range(max_rounds):
        full_prompt = (
            system_prompt
            + ("\n\nPrevious tool calls and observations:\n" + trace_text if trace_text else "")
            + f"\n\nRound {round_idx + 1}/{max_rounds}. Reply with one JSON action."
        )
        raw = qwen3vl_wrapper.generate_from_frames(
            video_frames=None,
            object_crops=[],
            prompt=full_prompt,
            max_new_tokens=max_new_tokens_per_round,
            temperature=temperature,
        )
        parsed = validate_and_fix_dict(raw)
        if not isinstance(parsed, dict) or "action" not in parsed:
            transcript.append({"round": round_idx, "raw": raw, "parse_error": True})
            logger.warning(f"round {round_idx}: JSON parse failed; tail={raw[-200:]!r}")
            return MultiRoundResult(
                question=question,
                answer=f"(multi-round parse error on round {round_idx + 1}) raw: {raw[:200]}",
                n_rounds=round_idx + 1,
                terminated="parse_error",
                transcript=transcript,
            )

        action = str(parsed.get("action", "")).strip()
        args = parsed.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        thought = str(parsed.get("thought", ""))

        if action == "finish":
            answer = str(args.get("answer", "")).strip()
            transcript.append({
                "round": round_idx, "thought": thought, "action": action, "args": args,
            })
            return MultiRoundResult(
                question=question,
                answer=answer or "(empty answer from finish)",
                n_rounds=round_idx + 1,
                terminated="finish",
                transcript=transcript,
            )

        tool = _TOOLS.get(action)
        if tool is None:
            observation: Any = {"error": f"unknown action: {action}"}
        else:
            try:
                observation = tool(graph, args)
            except Exception as e:
                observation = {"error": f"tool {action} raised: {e!r}"}

        obs_text = _render_observation(observation)
        transcript.append({
            "round": round_idx, "thought": thought,
            "action": action, "args": args, "observation": observation,
        })
        trace_text += (
            f"\n--- round {round_idx + 1} ---\n"
            f"thought: {thought}\n"
            f"action: {action}\n"
            f"args: {json.dumps(args, ensure_ascii=False)}\n"
            f"observation: {obs_text}\n"
        )

    logger.warning(f"multi-round hit max_rounds={max_rounds} without finish; falling back to summary")
    fallback_hits = retrieve_long_term(graph, question=question, top_k=fallback_top_k)
    evidence = "\n".join(f"- [{h.type} clip {h.clip_id}] {h.text}" for h in fallback_hits)
    synth_prompt = (
        "The agent used up its tool-call rounds. Here is the gathered evidence "
        "(plus a fresh top-retrieval over the full graph). Produce a concise 1-3 "
        "sentence answer.\n\n"
        f"Question: {question.strip()}\n\n"
        f"Agent transcript:\n{trace_text[-3000:]}\n\n"
        f"Top evidence:\n{evidence}\n\nAnswer:"
    )
    answer = qwen3vl_wrapper.generate_from_frames(
        video_frames=None, object_crops=[],
        prompt=synth_prompt, max_new_tokens=256, temperature=temperature,
    )
    return MultiRoundResult(
        question=question,
        answer=answer.strip(),
        n_rounds=max_rounds,
        terminated="max_rounds",
        transcript=transcript,
    )
