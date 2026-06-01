"""General parsing and graph persistence helpers."""
import json
import ast
import logging
import os
import pickle
import shutil

logger = logging.getLogger(__name__)


def refine_json_str(raw: str) -> str:
    """Strip markdown fences and surrounding whitespace from LLM JSON output."""
    return raw.strip("```json").strip("```python").strip("```").strip()


def validate_and_fix_json(raw: str):
    fixed = refine_json_str(raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode failed: {e}; raw prefix: {raw[:200]!r}")
        return None


def validate_and_fix_dict(raw: str):
    """Parse a dict returned by an LLM. Tolerates Python-dict literal output
    (single quotes, None/True/False) as well as JSON-strict output."""
    s = refine_json_str(raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        result = ast.literal_eval(s)
        if isinstance(result, dict):
            return result
    except (SyntaxError, ValueError):
        pass
    logger.error(f"dict parse failed; raw prefix: {raw[:200]!r}")
    return None


def validate_and_fix_python_list(raw: str):
    try:
        s = raw.strip("```json").strip("```python").strip("```").strip()
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
        raise ValueError("not a list")
    except (SyntaxError, ValueError) as e:
        logger.error(f"list parse failed: {e}; raw: {raw[:200]!r}")
        return None


def save_video_graph(graph, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp = save_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(graph, f)
    shutil.move(tmp, save_path)
    logger.info(f"Saved graph to {save_path}")


def load_video_graph(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
