# Moonshot VLM Memory System

Graph-based memory construction for first-person drone assembly videos.

This repository keeps only the memory pipeline: video clips are converted into
grounded part nodes, prompt-based text memory, safety warnings, a cumulative
`VideoGraph`, and a self-contained interactive HTML report.

It does not include the previous QA interface, local model servers, SAM/DINO
object-discovery wrappers, or runtime outputs.

![Moonshot VLM memory system pipeline](assets/pipeline.svg)

## What It Builds

For each input video, the system produces:

- `graph.pkl`: the cumulative graph-based memory;
- `grounded_memory.html`: the default visualization with clip-by-clip memory
  tabs and a full graph tab.

The graph stores object nodes, text memory nodes, and mention edges:

```text
memory node --mention--> object node
```

Memory text is unified but typed by prefix:

- `[f]` facts: visible states, hand actions, screw relations, contacts, and
  clip-local outcomes;
- `[r]` reasoning: durable conclusions, part roles, intent, and assembly-state
  interpretation.

Every memory node also stores `clip_id`, timestamp metadata, embeddings, and a
`task_status=None` placeholder for later task-state annotation.

## Pipeline

### 1. Input Video -> Split Clips

`scripts/segment_video.py` splits one `.mp4` into fixed-duration clips and a
`manifest.json`.

The formal setting uses `30s` clips. The CLI default is `10s`, so set
`--clip-seconds 30` explicitly for the assembly experiments.

### 2. Split Clips -> Local Part Nodes

`mmagent/memory_builder.py` samples 15 chronological keyframes per clip by
default. It extracts local part candidates with lightweight CV:

- foreground masking and transparent crops;
- cached reference features from `docs/Part Images`;
- SIFT, shape, color, and geometry matching;
- parallel keyframe processing.

Part identity is closed-set for the drone-frame task:

- 8 fixed part classes from `docs/Manual/manual.txt`;
- 1 dynamic `Current_Assembly` node.

### 3. Local Part Nodes -> Gemini Memory

Gemini receives chronological keyframes, a fine-detail evidence sheet, object
crops tagged as `<object_N>`, detector labels, and compact manual context from
`mmagent/prompt.py`.

The model returns high-signal `[f]` and `[r]` memory lines. The builder embeds
these lines, links object mentions to object nodes, and reinforces similar
reasoning memories across clips.

### 4. Gemini Memory -> VideoGraph Long-Term Memory

`mmagent/videograph.py` maintains the cumulative `VideoGraph`.

The graph contains:

- object nodes for known parts and `Current_Assembly`;
- memory nodes for `[f]` facts and `[r]` reasoning;
- mention edges from memory nodes to referenced object nodes.

### 5. Keyframes -> Safety Warnings

Safety warnings are generated separately from memory generation. The safety
call uses only clip keyframes and the closed-set taxonomy summarized in
`mmagent/safety.py`.

Warnings follow:

```text
[TYPE] description
```

`TYPE` belongs to `S-*` safety classes or `C-*` task-correctness classes. If no
concern is visible, the clip stores an empty list.

### 6. VideoGraph -> Interactive HTML

`mmagent/visualization.py` renders the default HTML:

- one clip tab per processed clip;
- object nodes, facts, reasoning, and safety warnings in the clip view;
- a full graph tab with object/memory nodes and mention edges;
- hover previews for object references and memory timestamps.

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

### Process The Formal Assembly Video

```bash
python scripts/segment_video.py \
  --video data/correct_assemble_v1.mp4 \
  --out-dir runs/correct_assemble_v1/clips_30s \
  --clip-seconds 30 \
  --overwrite
```

```bash
python scripts/build_memory.py \
  --videos-dir runs/correct_assemble_v1/clips_30s \
  --out-dir runs/correct_assemble_v1/memory
```

For a quick prefix test:

```bash
python scripts/segment_video.py \
  --video data/correct_assemble_v1.mp4 \
  --out-dir runs/correct_assemble_v1/clips_30s_prefix \
  --clip-seconds 30 \
  --prefix-seconds 60 \
  --overwrite
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

## Code Map

```text
assets/pipeline.svg         project-level pipeline diagram
scripts/segment_video.py    CLI wrapper for video segmentation
scripts/build_memory.py     CLI wrapper for memory construction
mmagent/video_segmenter.py  input video -> fixed-duration clips
mmagent/memory_builder.py   keyframes -> parts -> memory -> graph -> HTML
mmagent/prompt.py           Gemini prompt for [f] facts and [r] reasoning
mmagent/safety.py           keyframe-only safety warning generation
mmagent/videograph.py       graph data structure and fusion logic
mmagent/visualization.py    clip tabs and interactive graph HTML
mmagent/utils/general.py    graph save/load helpers
```

## Notes

- `docs/Manual/manual.txt` defines the assembly part vocabulary.
- `docs/Part Images` provides reference views for local part matching.
- `docs/Part Images/.reference_features.pkl` is a rebuildable feature cache.
- `runs/`, `outputs/`, and `data/` are runtime/local-data directories and are
  ignored by git.
