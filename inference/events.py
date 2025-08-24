from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import time
import json
import shutil
import pandas as pd

from .draw import draw_info_panel

def smooth_scores(scores: List[float], k: int = 9) -> np.ndarray:
    k = max(1, int(k) | 1)
    out = np.zeros(len(scores), dtype=np.float32)
    s = 0.0
    q = []
    for i, v in enumerate(scores):
        q.append(v); s += v
        if len(q) > k:
            s -= q.pop(0)
        out[i] = s / len(q)
    return out

@dataclass
class EventConfig:
    threshold: float
    hysteresis_low_ratio: float = 0.8
    end_patience_frames: int = 5
    cooldown_seconds: float = 2.0
    save_screenshots: bool = True
    info_on_screenshots: bool = True

@dataclass
class EventState:
    in_event: bool = False
    start_idx: int = -1
    last_above_idx: int = -1
    last_end_time: float = 0.0
    peak_score: float = 0.0
    peak_frame_idx: int = -1
    peak_tags: List[str] = field(default_factory=list)
    peak_hints: List[Tuple[str, float]] = field(default_factory=list)

    events: List[Dict[str, Any]] = field(default_factory=list)

def _in_cooldown(state: EventState, cfg: EventConfig) -> bool:
    return (time.time() - state.last_end_time) < cfg.cooldown_seconds

def compute_event_tags(features: Dict[str, float]) -> List[str]:
    tags = []
    SPEED_THRESH_PX = 20.0
    loiter_ratio = float(features.get("loiter_ratio", 0.0))
    avg_speed = float(features.get("avg_speed_px", 0.0))
    fast_ratio = float(features.get("fast_ratio", 0.0))
    max_speed = float(features.get("max_speed_px", 0.0))
    min_person_obj_norm = float(features.get("min_person_obj_norm", 1.0))
    grid_max = float(features.get("grid_max", 0.0))
    num_people = float(features.get("num_people", 0.0))
    if loiter_ratio >= 0.5 and avg_speed < SPEED_THRESH_PX:
        tags.append("Loitering-like")

    if fast_ratio >= 0.4 or max_speed >= 1.5 * SPEED_THRESH_PX:
        tags.append("Rapid movement")

    if min_person_obj_norm <= 0.03:
        tags.append("Close to object")

    if grid_max >= 4 or num_people >= 8:
        tags.append("Crowding")

    return tags

def top_feature_hints(features: Dict[str, float], tags: List[str]) -> List[Tuple[str, float]]:
    hints: List[Tuple[str, float]] = []
    if "Loitering-like" in tags:
        hints.append(("loiter_ratio", float(features.get("loiter_ratio", 0.0))))
        hints.append(("avg_speed_px", float(features.get("avg_speed_px", 0.0))))
    if "Rapid movement" in tags:
        hints.append(("max_speed_px", float(features.get("max_speed_px", 0.0))))
        hints.append(("fast_ratio", float(features.get("fast_ratio", 0.0))))
    if "Close to object" in tags:
        hints.append(("min_person_obj_norm", float(features.get("min_person_obj_norm", 1.0))))
    if "Crowding" in tags:
        hints.append(("grid_max", float(features.get("grid_max", 0.0))))
        hints.append(("num_people", float(features.get("num_people", 0.0))))

    if len(hints) < 3:
        candidates = [
            ("stationary_ratio", features.get("stationary_ratio", 0.0)),
            ("min_inter_person_norm", features.get("min_inter_person_norm", 1.0)),
            ("object_conf_max", features.get("object_conf_max", 0.0)),
        ]
        for k, v in candidates:
            if all(k != hk for hk, _ in hints):
                hints.append((k, float(v)))
            if len(hints) >= 3:
                break
    return hints[:3]

def update_events(
    state: EventState,
    cfg: EventConfig,
    frame_idx: int,
    score: float,
    frame_bgr,
    tracked,
    out_dir: Path,
    features: Dict[str, float] | None = None,
):
    hi = cfg.threshold
    lo = cfg.threshold * cfg.hysteresis_low_ratio

    if not state.in_event:
        if score >= hi and not _in_cooldown(state, cfg):
            state.in_event = True
            state.start_idx = frame_idx
            state.last_above_idx = frame_idx
            state.peak_score = score
            state.peak_frame_idx = frame_idx
            cur_tags = compute_event_tags(features or {})
            hints = top_feature_hints(features or {}, cur_tags)
            state.peak_tags = cur_tags
            state.peak_hints = hints
            if cfg.save_screenshots:
                save_screenshot(frame_bgr, tracked, out_dir, frame_idx, score, cur_tags, hints, cfg.info_on_screenshots)
    else:
        if score >= lo:
            state.last_above_idx = frame_idx
            if score > state.peak_score:
                state.peak_score = score
                state.peak_frame_idx = frame_idx
                cur_tags = compute_event_tags(features or {})
                hints = top_feature_hints(features or {}, cur_tags)
                state.peak_tags = cur_tags
                state.peak_hints = hints
        else:
            if frame_idx - state.last_above_idx >= cfg.end_patience_frames:
                state.events.append({
                    "start_frame": state.start_idx,
                    "end_frame": frame_idx,
                    "peak_frame": state.peak_frame_idx,
                    "peak_score": float(state.peak_score),
                    "tags": state.peak_tags.copy(),
                })
                state.in_event = False
                state.start_idx = -1
                state.last_end_time = time.time()
                state.peak_score = 0.0
                state.peak_frame_idx = -1
                state.peak_tags = []
                state.peak_hints = []

def save_screenshot(
    frame_bgr,
    tracked,
    out_dir: Path,
    frame_idx: int,
    score: float = 0.0,
    tags: List[str] | None = None,
    feature_hints: List[Tuple[str, float]] | None = None,
    draw_info: bool = True,
):
    shots_dir = out_dir / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    img = frame_bgr.copy()
    for t in tracked:
        x1, y1, x2, y2 = map(int, t["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(img, f'ID{t["id"]}', (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    if draw_info:
        img = draw_info_panel(
            img,
            score=score,
            tags=tags or [],
            feature_hints=feature_hints or [],
            pos=(10, 10),
        )
    fn = shots_dir / f"frame_{frame_idx:06d}.png"
    cv2.imwrite(str(fn), img)

def finalize_report(run_dir: Path, per_frame_rows: List[Dict[str, Any]], events: List[Dict[str, Any]], settings: Dict[str, Any], zip_images: bool = True):
    run_dir.mkdir(parents=True, exist_ok=True)
    pf = pd.DataFrame(per_frame_rows)
    pf.to_csv(run_dir / "per_frame.csv", index=False)
    ev = pd.DataFrame(events)
    ev.to_csv(run_dir / "events.csv", index=False)
    with open(run_dir / "settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    shots_dir = run_dir / "screenshots"
    if zip_images and shots_dir.exists():
        shutil.make_archive(str(run_dir / "screenshots"), "zip", str(shots_dir))