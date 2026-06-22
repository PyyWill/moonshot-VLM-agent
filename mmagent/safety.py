"""Safety-warning generation from clip keyframes.

This stage is intentionally independent from graph-memory generation. It uses
only chronological keyframes and the closed-set safety taxonomy, then returns
conservative warning strings such as ``[S-H4] ...``. Empty output means no
visible warning is warranted for the clip.
"""
from __future__ import annotations

import ast
import json
import os
import re
from io import BytesIO
from typing import Any

from PIL import Image


SAFETY_TAXONOMY_SUMMARY = """
Active irreversible safety monitor:
- S-H1 Human safety: thermal contact or burn.
- S-H2 Human safety: sharp, pinch, or puncture hazard.
- S-H3 Human safety: chemical, fume, or residue exposure.
- S-H4 Human/workspace safety: battery physical damage.
- S-D1 Device safety: ESD or EOS latent damage.
- S-D2 Device safety: thermal component damage.
- S-D3 Device safety: mechanical overstress or wrong force direction.
- S-D4 Device safety: connector or pin damage.
- S-D5 Device safety: foreign object, conductive debris, or contamination.
- S-D6 Device/task safety: irreversible wrong placement before commit.
- S-D7 Device safety: fastener, thread, or mount damage.
- S-W1 Workspace safety: fire, heat transfer, or unstable hot tool.
- S-W2 Workspace/process safety: occlusion or instability during hazardous action.

On-demand flexible task-correctness monitor:
- C-T1 Task correctness: step omission.
- C-T2 Task correctness: extra or unnecessary step.
- C-T3 Task correctness: step modification.
- C-T4 Task correctness: wrong part or hardware.
- C-T5 Task correctness: reversible orientation or placement mismatch.
- C-T6 Task correctness: reversible fastening-quality issue.
- C-T7 Task correctness: reversible cable-routing or clearance issue.
- C-T8 Task correctness: missing validation or unresolved uncertainty.
"""


PROMPT_GENERATE_SAFETY_WARNINGS = """
You are a conservative safety monitor for first-person small-quadcopter assembly.
You receive chronological keyframes from one clip. Generate safety warnings
ONLY from visible evidence in these keyframes.

Use this closed-set safety taxonomy:
__SAFETY_TAXONOMY__

Rules:
1. Output warnings only for concrete visible concerns in the current clip.
2. Use both S-* safety classes and C-* task-correctness classes when warranted.
3. Be conservative: ordinary staging, loose screws on a board, visible tools,
   or a user's hand near parts are not warnings by themselves.
4. Do not infer future hazards from the manual or from general knowledge if the
   current keyframes do not show a plausible current concern.
5. Do not mention keyframe numbers.
6. Each warning must be one concise English string formatted exactly as:
   "[TYPE] description"
   where TYPE is one of S-H1..S-H4, S-D1..S-D7, S-W1..S-W2, or C-T1..C-T8.
7. If there is no visible warning, return an empty list.

Return ONLY a valid JSON object:
{
  "safety_warnings": []
}
"""


SAFETY_WARNING_RE = re.compile(r"^\[(S-(?:H[1-4]|D[1-7]|W[1-2])|C-T[1-8])\]\s+\S.+$")


def _pil_to_jpeg_bytes(image: Image.Image, *, max_width: int = 1280) -> bytes:
    rgb = image.convert("RGB")
    if rgb.width > max_width:
        new_h = max(1, int(rgb.height * max_width / rgb.width))
        rgb = rgb.resize((max_width, new_h), Image.Resampling.LANCZOS)
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=86, optimize=True)
    return buf.getvalue()


def _get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY to generate safety warnings.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install google-genai to use Gemini safety-warning generation.") from exc
    return genai.Client(api_key=api_key), types


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_safety_response(raw: str) -> list[str]:
    text = _strip_code_fence(raw)
    parsed: Any
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        raise ValueError("safety response must be a JSON object")
    raw_warnings = parsed.get("safety_warnings")
    if not isinstance(raw_warnings, list):
        raise ValueError("safety response must contain a safety_warnings list")
    warnings: list[str] = []
    for item in raw_warnings:
        warning = str(item).strip()
        if SAFETY_WARNING_RE.match(warning):
            warnings.append(warning)
    return warnings


def generate_safety_warnings_with_gemini(
    keyframes,
    *,
    model: str,
    max_retries: int = 3,
) -> list[str]:
    if not keyframes:
        return []
    client, types = _get_gemini_client()
    prompt = PROMPT_GENERATE_SAFETY_WARNINGS.replace(
        "__SAFETY_TAXONOMY__",
        SAFETY_TAXONOMY_SUMMARY.strip(),
    )
    contents: list[Any] = [prompt]
    for keyframe in keyframes:
        contents.append(f"Keyframe {keyframe.frame_index} at {keyframe.timestamp_sec:.2f}s")
        contents.append(
            types.Part.from_bytes(
                data=_pil_to_jpeg_bytes(keyframe.image),
                mime_type="image/jpeg",
            )
        )

    last_raw = ""
    for _ in range(max_retries):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.1,
                seed=1,
                response_mime_type="application/json",
            ),
        )
        last_raw = getattr(response, "text", "") or ""
        try:
            return _parse_safety_response(last_raw)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            continue
    raise RuntimeError(f"Gemini did not return valid safety warnings. Last response: {last_raw[:500]!r}")
