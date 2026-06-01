#!/bin/bash
# Split a source video into fixed-length clips (default 10s).
# Usage: ./split_video_clips.sh <src.mp4> <video_id> [prefix_seconds] [interval]
set -euo pipefail

SRC="${1:?src video path required}"
VIDEO_ID="${2:?video_id required}"
PREFIX_SEC="${3:-0}"          # 0 = whole video; otherwise first N seconds
INTERVAL="${4:-10}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/data/videos/${VIDEO_ID}"
mkdir -p "${OUT_DIR}"

if [[ "${PREFIX_SEC}" -gt 0 ]]; then
    TRIMMED="${OUT_DIR}/_full.mp4"
    echo "[1/2] trimming first ${PREFIX_SEC}s of ${SRC} -> ${TRIMMED}"
    ffmpeg -y -i "${SRC}" -t "${PREFIX_SEC}" -c copy "${TRIMMED}"
    INPUT="${TRIMMED}"
else
    INPUT="${SRC}"
fi

echo "[2/2] splitting ${INPUT} into ${INTERVAL}s segments under ${OUT_DIR}"
ffmpeg -y -i "${INPUT}" -c copy -map 0 \
       -segment_time "${INTERVAL}" -f segment -reset_timestamps 1 \
       "${OUT_DIR}/%d.mp4"

# Drop the trimmed full file (we keep only clip files)
if [[ "${PREFIX_SEC}" -gt 0 ]]; then
    rm -f "${TRIMMED}"
fi

echo "done. clips:"
ls -la "${OUT_DIR}"
