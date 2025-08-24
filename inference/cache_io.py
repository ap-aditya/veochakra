import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

CACHE_ROOT = Path("artifacts/cache_dets")

def ensure_cache_dir():
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

def stable_cache_name(video_path: Path) -> str:
    try:
        size = video_path.stat().st_size
        return f"{video_path.stem}_{size}.jsonl"
    except Exception:
        h = hashlib.sha256(str(video_path).encode()).hexdigest()[:16]
        return f"{video_path.stem}_{h}.jsonl"

def cache_path_for(video_path: Path) -> Path:
    ensure_cache_dir()
    return CACHE_ROOT / stable_cache_name(video_path)

def read_cache(video_path: Path) -> Optional[List[List[Dict]]]:
    p = cache_path_for(video_path)
    if not p.exists():
        return None
    frames = []
    with open(p, "r") as f:
        for line in f:
            frames.append(json.loads(line))
    return frames

def write_cache(video_path: Path, per_frame_dets: List[List[Dict]]):
    p = cache_path_for(video_path)
    with open(p, "w") as f:
        for dets in per_frame_dets:
            f.write(json.dumps(dets) + "\n")
