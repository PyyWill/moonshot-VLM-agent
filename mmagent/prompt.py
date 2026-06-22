"""Prompt templates for prompt-based graph memory generation.

The object nodes are produced by the local part detector. Gemini receives
chronological keyframes plus the object crops that are already present in the graph, then
returns one unified list of memory lines. The line prefix preserves the old
episodic/semantic intuition without splitting graph edge types:

    [f] factual observation, equivalent to old episodic memory
    [r] reasoning / durable conclusion, equivalent to old semantic memory
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptObjectFeature:
    token: str
    label: str | None = None


ASSEMBLY_MANUAL = """
Assembly target: Lumenier QAV-S 2 Joshua Bardwell frame.

Use these steps as task context and vocabulary:
1. Start by using the Split_Rear_Plate to mount the X-Lock with M3x22mm screws
   and M3x6mm screws.
2. Secure each 5_inch_Arm by tightening the Arm_Wedge_5mm with M3x16mm screws
   to lock the arm in place.
3. Add the Split_Front_Plate to the assembly and use the middle M3x6mm screw
   to secure it.
4. Use M3x6mm button-head screws to secure the Knurled_Standoff and
   FPV_Camera_Mounts to the Split_Front_Plate. Use M3x16mm screws through the
   5_inch_Arm into the Knurled_Standoff.
5. Secure the Top_Plate to the assembly with M3x6mm button-head screws in the
   front and M3x6mm countersink-head screws in the back.
6. Finish by adding the GoPro mount and spacer plate with an M3x8mm screw, then
   sticking the battery pad and strap.
"""


PROMPT_GENERATE_OBJECT_MEMORY = """
You are given chronological keyframe images sampled from a first-person clip,
a short assembly manual, and a set of object features. Each object feature is a
cropped image or visual summary for one detected visual entity in the current
clip, labeled with an ID token like <object_7>.

ASSEMBLY_MANUAL:
__ASSEMBLY_MANUAL__

Use ASSEMBLY_MANUAL only as context for part roles, vocabulary, and likely
assembly order. It is not a transcript: do not say a step happened unless it is
supported by the keyframes or object features.

The keyframes are ordered by time. Use them to infer visible states and
coarse hand activity, but be conservative about motion between frames: if an
attachment, fastening, or tool use is not visible in the keyframes, say it is
not visible rather than claiming it happened.
Some inputs may include a fine-detail evidence sheet with zoomed crops of small
foreground components. Use it to inspect tiny screws, screw holes, and local
part-to-part states. If a screw is visibly inserted into or passing through a
part hole, write that visible state as a fact even when the insertion motion
occurred between keyframes. Do not call it tightened or fastened unless tool use
or fastening contact is visible.
Screw-through states are high-priority details: do not summarize them only as
"small parts" or "staging". If the fine-detail sheet shows a small dark screw
aligned with or passing through a purple or carbon part hole, include a specific
[f] line about that visible state. Avoid broad negative facts such as "no parts
are attached" when a screw-through state is visible or ambiguous in the detail
crops.
Do not claim an object was moved, removed, or picked up unless the hand-object
interaction is visible in the keyframes. If an object simply appears absent or
occluded in a later keyframe, describe only the visible state.
Never infer that an absent part was selected for an assembly step.
Do not mention keyframe numbers in the final memory. Write concise clip-level
observations supported by the sampled keyframes.

The detector may also provide a short part label for an object. Treat that
label as the canonical part identity unless the keyframes clearly contradict
it. Do not collapse differently labeled objects into vague groups when their
detector labels are distinct. Do not use any part-image catalog or hidden
assembly instructions.

If the object inventory includes detector_label=Current_Assembly, that object is
the dynamic working assembly state for this clip. Use its object ID when a
visible multi-part unit is being formed, handled as one unit, or placed onto
another part. Do not claim a current assembly exists before visible evidence:
if the parts are merely separate on the table, say so without using it as an
assembled unit.

Your task is to write a compact, high-signal graph memory for this clip. The
goal is not to be short at all costs; the goal is to preserve the meaningful
assembly state while avoiding fragmented near-duplicate lines.

Use ONE unified memory list, but preserve two kinds of information with exact
prefixes:

1. [f] Facts
   Concrete observations grounded in the keyframes. These correspond to old
   episodic memory. Each fact should describe one meaningful visible state or
   action episode, not one tiny sub-action. Include object states, spatial
   layout, visibility, orientation, hand actions, and object-object interactions.
   Facts should be directly visible or very tightly grounded. Pay special
   attention to visible screws, screw holes, and screwdrivers, including whether
   they are being used or merely present.

2. [r] Reasoning
   Concise, durable conclusions inferred from the clip and object features.
   These correspond to old semantic memory. Include part typing, likely role in
   the drone assembly, grouping relationships, and high-level task intent.
   Reasoning should be useful later, but should not invent unsupported steps.
   Pay special attention to the person's hand operations and likely immediate
   intent in the assembly process.

Quality and consolidation guidance:

- Prefer one strong line over several thin lines. If consecutive keyframes show
  the same continuous manipulation, combine it into one fact unless the visible
  assembly state materially changes.
- Do not create separate facts for "picks up", "holds", "brings together",
  "separates", and "places down" when they are one continuous inspection or
  dry-fit episode. Summarize the episode and its visible outcome.
- If the person picks up and then holds the same objects without a new visible
  assembly outcome, write one combined fact, not separate pickup and holding
  facts.
- Avoid this fragmented pattern:
  "[f] The person picks up <object_A> and <object_B>."
  "[f] The person holds <object_A> and <object_B>."
  "[f] The person manipulates <object_A> and <object_B>."
  "[f] The person places <object_A> and <object_B> back down."
  Prefer one line such as:
  "[f] The person picks up <object_A> and <object_B>, holds them together for
  inspection or dry-fitting, and places them back down without fastening."
- Merge broad negative observations. For example, use one line for "no visible
  fastening/tool use" instead of separate no-tool, no-screw, and no-attachment
  lines, unless a specific screw-through state must be contrasted.
- Do not infer that a physical screw is involved from a screw chart, whiteboard,
  manual step, or nearby loose fasteners. Mention screw participation only when
  a screw/fastener is visibly held, aligned with a hole, inserted through a
  part, or contacted by a tool.
- This rule must not suppress visible screw evidence. If a screw/fastener is
  visibly held, aligned with a hole, inserted through a part, or contacted by a
  tool, include that specific relation as a [f] line before any generic hand
  manipulation summary.
- Avoid generic facts such as "the person manipulates small parts" when a more
  specific visible relation is available, such as "<object_3> is handled with a
  small dark screw" or "a screw passes through a hole in <object_3>".
- When a screw/fastener and a part are visible together, describe the concrete
  geometry: held near the part, aligned with a hole, inserted through a hole,
  resting on the part, or contacted by a screwdriver. Do not replace this with
  vague wording such as "manipulates screws and parts".
- For screw-through or screw-hole facts, choose the object ID whose detector
  label matches the physical part containing the hole. In particular, do not
  swap visually similar small parts such as X-Lock and Arm_Wedge_5mm; if the
  screw appears to pass through an Arm_Wedge_5mm, use the Arm_Wedge_5mm object
  ID rather than the nearby X-Lock object ID.
- A screw inserted through a part hole is a visible state, not necessarily a
  completed fastening. Include the screw-through state while still avoiding
  claims of tightening or completed attachment unless tool use/contact proves it.
- When summarizing a dry-fit or inspection episode, include only object IDs that
  are visibly manipulated together. Do not add nearby objects to the episode
  merely because they are on the table or likely relevant from the manual.
- If multiple physical instances share the same object ID, describe them in one
  consolidated line as "multiple instances of <object_N>" rather than creating
  separate "first instance" and "second instance" lines for the same ID.
- Reasoning must not introduce new concrete screw, tool, fastening, or
  object-object relations that are absent from the [f] lines. If no [f] line
  mentions a visible screw relation in this clip, no [r] line should mention
  using, inserting, or manipulating a screw.
- Static part-role reasoning should be grouped when possible. Do not spend one
  [r] line per stationary object unless that object is manipulated, assembled,
  or uniquely important in this clip.
- Use at most one [r] line for the clip-level task intent. Do not output
  multiple reasoning lines that all say the clip is staging, inspecting, or
  dry-fitting the same parts.
- Keep high-priority details even if they increase the count: visible
  screw-through states, tool contact, fastening, current assembly formation, or
  a new object-object connection should each be explicitly represented.

Strict requirements:

1. When referring to a provided object, use ONLY its ID token, e.g. <object_7>.
2. Do not reference any object ID outside the allowed object list.
3. If a visible thing has no provided ID, use a short descriptive phrase such
   as "the person's left hand". Do not invent an <object_N> ID for it.
4. Do not use pronouns for the person; say "the person".
5. Each memory line must start with exactly "[f]" or "[r]".
6. Do not output Equivalence lines. Long-term object identity is handled by
   the system through fixed part labels.
7. Do not split output into episodic_memory / semantic_memory. There is only
   one key: "memory".
8. Do not repeat the same content in both [f] and [r].
9. Prefer high-signal memory. For an ordinary 30-second clip, target 6-10 [f]
   lines and 3-6 [r] lines. Exceed this only for genuinely distinct assembly
   state changes, visible screw-through/fastening details, or multiple important
   object-object relations.
10. Output English only.
11. The response must be a valid JSON object, fully closed.
12. Avoid keyframe-number wording such as "in Keyframe 1"; describe the clip
    state/action directly.
13. When detector labels are available, use those labels in [r] lines to give
    specific part roles, especially for X-Lock, Arm_Wedge_5mm,
    Knurled_Standoff, FPV_Camera_Mounts, and the frame plates.

Good [f] examples:
- "[f] <object_3> lies flat near the center of the workspace throughout the clip."
- "[f] The person moves the right hand over <object_5> but does not attach it to another part."
- "[f] <object_1> and <object_4> remain separated on the staging surface."

Good [r] examples:
- "[r] <object_3> is a structural carbon-fiber frame plate for the drone assembly."
- "[r] <object_5> is likely a standoff or spacer used to create vertical separation between plates."
- "[r] The clip is mainly staging and identifying parts rather than fastening the assembly."

Output format:

{
  "memory": [
    "[f] <object_3> lies flat near the center of the workspace.",
    "[f] The person reaches near <object_5> without attaching it.",
    "[r] <object_3> is a structural frame plate.",
    "[r] The clip is mainly staging parts before assembly."
  ]
}

Return ONLY the valid JSON object. No code fences, no prose, no markdown.
"""


def build_memory_prompt(object_features: list[PromptObjectFeature]) -> str:
    allowed = ", ".join(f.token for f in object_features) or "(none)"
    inventory_lines = []
    for feature in object_features:
        if feature.label:
            inventory_lines.append(f"- {feature.token}: detector_label={feature.label}")
        else:
            inventory_lines.append(f"- {feature.token}")
    inventory = "\n".join(inventory_lines) or "- no object features were detected"
    return (
        PROMPT_GENERATE_OBJECT_MEMORY.replace("__ASSEMBLY_MANUAL__", ASSEMBLY_MANUAL.strip())
        + "\n\nAllowed object IDs for this clip:\n"
        + allowed
        + "\n\nObject inventory:\n"
        + inventory
        + "\n"
    )
