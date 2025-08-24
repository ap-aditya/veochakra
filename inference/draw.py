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

def draw_info_panel(img, score: float, tags: List[str] = None, feature_hints: List[str] = None, pos: tuple = (10, 10)):
    if tags is None:
        tags = []
    if feature_hints is None:
        feature_hints = []
    
    overlay = img.copy()
    x, y = pos
    
    panel_width = 300
    panel_height = 60 + len(tags) * 20 + len(feature_hints) * 15
    
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    
    alpha = 0.7
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    text_y = y + 25
    
    score_text = f"Anomaly Score: {score:.3f}"
    cv2.putText(img, score_text, (x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    text_y += 25
    
    if tags:
        cv2.putText(img, "Tags:", (x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        text_y += 20
        for tag in tags:
            cv2.putText(img, f"  - {tag}", (x + 20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            text_y += 20
    
    if feature_hints:
        cv2.putText(img, "Features:", (x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        text_y += 20
        for hint in feature_hints[:3]: 
            cv2.putText(img, f"  {hint}", (x + 20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            text_y += 15
    
    return img