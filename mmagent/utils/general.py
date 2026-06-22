"""Graph persistence helpers."""
import logging
import os
import pickle
import shutil

logger = logging.getLogger(__name__)


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
