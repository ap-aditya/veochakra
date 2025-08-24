from typing import List, Dict
import cv2
import numpy as np

def color_for_id(i: int):
    rng = np.random.default_rng(seed=i)
    c = rng.integers(60, 255, size=3).tolist()
    return (int(c[0]), int(c[1]), int(c[2]))

def draw_overlays(frame_bgr, tracked: List[Dict], draw_ids: bool = True):
    img = frame_bgr.copy()
    for t in tracked:
        x1, y1, x2, y2 = map(int, t["box"])
        col = color_for_id(t["id"])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        if draw_ids:
            label = f'ID{t["id"]}'
            cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img