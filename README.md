# Moonshot VLM Memory System

Graph-based memory construction for first-person drone assembly videos.

This repository keeps the memory pipeline only: input videos are split into
clips, converted into grounded object nodes, summarized as prompt-based memory,
merged into a cumulative `VideoGraph`, and rendered as a self-contained
interactive HTML report. The previous QA interface, local model server code,
SAM/DINO wrappers, and runtime outputs are intentionally not part of this
version.

![Moonshot VLM memory system pipeline](assets/pipeline.svg)

## What It Builds

For each processed video, the system produces:

- `graph.pkl`: cumulative graph-based memory;
- `grounded_memory.html`: an interactive report with clip-by-clip memory tabs
  and a full graph tab.

The graph stores object nodes, memory nodes, and mention edges:

```text
memory node --mention--> object node
```

Memory lines use one unified text format with explicit prefixes:

- `[f]` facts: visible states, hand actions, screw relations, contacts, and
  clip-local outcomes;
- `[r]` reasoning: durable conclusions, part roles, intent, and assembly-state
  interpretation.

Every memory node stores its `clip_id`, timestamp metadata, embedding, and a
`task_status=None` placeholder for later task-state annotation.

## Pipeline

### 1. Input Video -> Fixed Clips

`scripts/segment_video.py` splits a `.mp4` into fixed-duration clips and writes
a `manifest.json`.

The formal assembly setting uses `30s` clips:

```bash
python scripts/segment_video.py \
  --video data/correct_assemble_v1.mp4 \
  --out-dir runs/correct_assemble_v1/clips_30s \
  --clip-seconds 30 \
  --overwrite
```

### 2. Clips -> VLM-Enhanced Object Nodes

`mmagent/memory_builder.py` samples 15 chronological keyframes per clip by
default. Object node recognition is no longer geometry-only. The current
pipeline uses a lightweight CV proposal stage followed by VLM closed-set label
assignment:

- foreground masking proposes transparent candidate crops from each keyframe;
- repeated candidates are clustered across the clip to keep the VLM input
  compact;
- candidate crops are arranged into an unlabeled crop sheet;
- `docs/Part Images` reference views are arranged into a reference sheet;
- Gemini assigns every candidate to one of the closed labels:
  the 8 manual part classes, `Current_Assembly`, or `Reject`;
- a focused plate-refinement pass separates `Top_Plate`,
  `Split_Front_Plate`, and `Split_Rear_Plate` by silhouette and hole layout;
- small general guards handle ambiguous tiny parts and obvious assembly crops.

The fixed part vocabulary comes from `docs/Manual/manual.txt`. The dynamic
`Current_Assembly` node represents visible connected or inserted subassemblies.
It is not forced to appear as a crop in every clip.

### 3. Object Nodes -> Prompt-Based Clip Memory

Gemini receives chronological keyframes, a fine-detail evidence sheet, object
crops tagged as `<object_N>`, detector labels, and compact manual context from
`mmagent/prompt.py`.

The model returns high-signal `[f]` and `[r]` memory lines. The builder embeds
these lines, links object mentions to object nodes, and reinforces similar
reasoning memories across clips.

### 4. Clip Memory -> Cumulative VideoGraph

`mmagent/videograph.py` maintains the cumulative long-term memory.

The graph contains:

- object nodes for the known parts and `Current_Assembly`;
- memory nodes for `[f]` facts and `[r]` reasoning;
- mention edges from memory nodes to referenced object nodes.

Object nodes also store `contents_by_clip` for visualization. In the HTML clip
view, object images come from the current clip when available; if a clip has no
representative crop for that object, the display inherits the latest previous
crop without changing the underlying graph evidence.

### 5. Keyframes -> Safety Warnings

Safety warnings are generated separately from memory generation. The safety
call uses only clip keyframes and the closed-set taxonomy summarized in
`mmagent/safety.py`.

Warnings follow this format:

```text
[TYPE] description
```

`TYPE` belongs to `S-*` safety classes or `C-*` task-correctness classes. If no
concern is visible, the clip stores an empty list.

### 6. VideoGraph -> Interactive HTML

`mmagent/visualization.py` renders the default HTML report:

- one tab per processed clip;
- object nodes, facts, reasoning, and safety warnings in the clip view;
- a full graph tab with object/memory nodes and mention edges;
- hover previews for object references;
- hover timestamps for memory lines;
- graph object previews use the latest available object crop.

## Install

```bash
pip install -r requirements.txt
```

Gemini calls use either environment variable:

```bash
export GEMINI_API_KEY="..."
# or
export GOOGLE_API_KEY="..."
```

Optional model overrides:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
export OBJECT_GEMINI_MODEL="gemini-2.5-flash"
export SAFETY_GEMINI_MODEL="gemini-2.5-flash-lite"
export GEMINI_EMBEDDING_MODEL="gemini-embedding-001"
```

## Run

### Process `0.mp4`

```bash
python scripts/segment_video.py \
  --video 0.mp4 \
  --out-dir runs/0mp4/clips \
  --clip-seconds 30 \
  --overwrite
```

```bash
python scripts/build_memory.py \
  --videos-dir runs/0mp4/clips \
  --out-dir runs/0mp4/memory
```

Open:

```text
runs/0mp4/memory/grounded_memory.html
```

### Process A Prefix Of The Formal Video

```bash
python scripts/segment_video.py \
  --video data/correct_assemble_v1.mp4 \
  --out-dir runs/correct_assemble_v1_clip0_1/clips_30s \
  --clip-seconds 30 \
  --prefix-seconds 60 \
  --overwrite
```

```bash
python scripts/build_memory.py \
  --videos-dir runs/correct_assemble_v1_clip0_1/clips_30s \
  --out-dir runs/correct_assemble_v1_clip0_1/memory \
  --frame-workers 4
```

## Outputs

```text
runs/<video_id>/clips/manifest.json
runs/<video_id>/clips/clip_000.mp4
runs/<video_id>/clips/clip_001.mp4
runs/<video_id>/memory/graph.pkl
runs/<video_id>/memory/grounded_memory.html
```

After memory construction, only `graph.pkl` and `grounded_memory.html` are
needed to reload and inspect the generated memory.

## Examples

The repository includes static HTML examples under `examples/`:

```text
examples/grounded_memory_clip0_1.html
examples/grounded_memory_clip0_7.html
```

These examples are generated from the formal assembly video prefix and are safe
to open directly in a browser. They are examples only; runtime `runs/` outputs
remain ignored by git.

## Code Map

```text
assets/pipeline.svg         project-level pipeline diagram
examples/                   static HTML examples
scripts/segment_video.py    CLI wrapper for video segmentation
scripts/build_memory.py     CLI wrapper for memory construction
mmagent/video_segmenter.py  input video -> fixed-duration clips
mmagent/memory_builder.py   keyframes -> object nodes -> memory -> graph -> HTML
mmagent/prompt.py           Gemini prompt for [f] facts and [r] reasoning
mmagent/safety.py           keyframe-only safety warning generation
mmagent/videograph.py       graph data structure and fusion logic
mmagent/visualization.py    clip tabs and interactive graph HTML
mmagent/utils/general.py    graph save/load helpers
```

## Notes

- `docs/Manual/manual.txt` defines the assembly part vocabulary.
- `docs/Part Images` provides visual references for VLM object recognition.
- `docs/Assembly Graph/` is kept as an empty tracked directory placeholder.
- `runs/`, `outputs/`, and `data/` are runtime/local-data directories and are
  ignored by git.
