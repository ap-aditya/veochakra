import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import joblib
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class Config:
    CLASSES_TO_DETECT = [0, 24, 26, 28]  # person, backpack, handbag, suitcase
    CONF_THRESHOLD = 0.35
    IOU_NMS = 0.5
    DEVICE = 0 if torch.cuda.is_available() else "cpu"

    # DeepSORT
    MAX_AGE = 30
    N_INIT = 3
    MAX_DIST = 0.2
    MAX_IOU_DISTANCE = 0.7
    NN_BUDGET = 100

    # Feature windows (seconds)
    STATIONARY_WINDOW_SEC = 2.0
    LOITER_WINDOW_SEC = 2.0
    LOITERING_DIST_PX = 40.0
    SPEED_THRESH_PX = 20.0


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union if union > 0 else 0.0


class Detector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = Config.CONF_THRESHOLD, iou: float = Config.IOU_NMS):
        self.model = YOLO(weights)
        self.model.fuse()
        self.names = self.model.model.names
        self.conf = conf
        self.iou = iou

    def detect(self, frame_bgr: np.ndarray):
        res = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou, device=Config.DEVICE, verbose=False)
        dets = []
        if len(res) == 0 or res[0].boxes is None or len(res[0].boxes) == 0:
            return dets, self.names
        boxes = res[0].boxes.xyxy.cpu().numpy()
        confs = res[0].boxes.conf.cpu().numpy()
        clss = res[0].boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(boxes, confs, clss):
            if k in Config.CLASSES_TO_DETECT:
                dets.append((float(x1), float(y1), float(x2), float(y2), float(c), int(k)))
        return dets, self.names


class Tracker:
    def __init__(self):
        self.trk = DeepSort(
            max_age=Config.MAX_AGE,
            n_init=Config.N_INIT,
            max_iou_distance=Config.MAX_IOU_DISTANCE,
            max_cosine_distance=Config.MAX_DIST,
            nn_budget=Config.NN_BUDGET,
            embedder="mobilenet",
            half=True if torch.cuda.is_available() else False,
            bgr=True
        )

    def update(self, frame_bgr: np.ndarray, dets: List[Tuple[float,float,float,float,float,int]], class_names: Dict[int,str]):
        inputs = []
        for (x1, y1, x2, y2, c, k) in dets:
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            inputs.append([(float(x1), float(y1), float(w), float(h)), float(c), str(class_names[k])])

        tracks = self.trk.update_tracks(inputs, frame=frame_bgr)

        det_boxes = [(float(x1), float(y1), float(x2), float(y2)) for (x1, y1, x2, y2, _, _) in dets]
        det_classes = [int(k) for *_, k in dets]

        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = t.to_tlbr()
            cls_id = 0
            if det_boxes:
                ious = [iou_xyxy((x1, y1, x2, y2), b) for b in det_boxes]
                j = int(np.argmax(ious)) if len(ious) else 0
                if ious and ious[j] > 0.3:
                    cls_id = det_classes[j]
            out.append({
                "box": (float(x1), float(y1), float(x2), float(y2)),
                "id": int(t.track_id),
                "center": (float((x1+x2)/2), float((y1+y2)/2)),
                "cls_id": int(cls_id)
            })
        return out


class FeatureLogger:
    def __init__(self, fps: float, w: int, h: int, feature_order: List[str]):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.w, self.h = int(w), int(h)
        self.img_diag = max(1.0, float(math.hypot(w, h)))
        from collections import defaultdict, deque
        self.hist = defaultdict(lambda: deque(maxlen=int(self.fps * 10)))
        self.first_seen = {}
        self.frame_idx = 0
        self.stationary_window = int(Config.STATIONARY_WINDOW_SEC * self.fps)
        self.loiter_window = int(Config.LOITER_WINDOW_SEC * self.fps)
        self.last_detections = []  # [(cls_id, conf)]
        self.feature_order = feature_order

    def set_last_detections(self, dets):
        self.last_detections = [(int(k), float(c)) for (*_, c, k) in dets]

    def step(self, tracked_objects):
        self.frame_idx += 1
        for obj in tracked_objects:
            tid = obj["id"]
            self.hist[tid].append(obj["center"])
            if tid not in self.first_seen:
                self.first_seen[tid] = self.frame_idx
        return self._aggregate(tracked_objects)

    def _aggregate(self, objs):
        people = [o for o in objs if o["cls_id"] == 0]
        others = [o for o in objs if o["cls_id"] != 0]

        num_people, num_objects = len(people), len(others)

        speeds, accels, loiter_flags, stationary_flags = [], [], [], []
        for p in people:
            tid = p["id"]; h = self.hist[tid]
            if len(h) >= 2:
                s = float(np.linalg.norm(np.array(h[-1]) - np.array(h[-2])))
                speeds.append(s)
            if len(h) >= 3:
                s1 = float(np.linalg.norm(np.array(h[-1]) - np.array(h[-2])))
                s2 = float(np.linalg.norm(np.array(h[-2]) - np.array(h[-3])))
                accels.append(abs(s1 - s2))
            if len(h) >= self.loiter_window:
                disp = float(np.linalg.norm(np.array(h[-1]) - np.array(h[-self.loiter_window])))
                loiter_flags.append(1.0 if disp < Config.LOITERING_DIST_PX else 0.0)
            if len(h) >= self.stationary_window:
                disp = float(np.linalg.norm(np.array(h[-1]) - np.array(h[-self.stationary_window])))
                stationary_flags.append(1.0 if disp < (0.6 * Config.LOITERING_DIST_PX) else 0.0)

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        avg_accel = float(np.mean(accels)) if accels else 0.0
        fast_ratio = float(np.mean([s > Config.SPEED_THRESH_PX for s in speeds])) if speeds else 0.0
        loiter_ratio = float(np.mean(loiter_flags)) if loiter_flags else 0.0
        stationary_ratio = float(np.mean(stationary_flags)) if stationary_flags else 0.0

        min_person_obj = 1e6
        for p in people:
            px, py = p["center"]
            for o in others:
                ox, oy = o["center"]
                d = float(np.hypot(px - ox, py - oy))
                if d < min_person_obj: min_person_obj = d
        if not people or not others:
            min_person_obj = 1e6

        pair_dists = []
        for i in range(len(people)):
            p1 = people[i]["center"]
            for j in range(i+1, len(people)):
                p2 = people[j]["center"]
                pair_dists.append(float(np.hypot(p1[0]-p2[0], p1[1]-p2[1])))
        min_inter_person = float(np.min(pair_dists)) if pair_dists else 1e6
        avg_inter_person = float(np.mean(pair_dists)) if pair_dists else 1e6

        ages = []
        for o in objs:
            tid = o["id"]
            ages.append(self.frame_idx - self.first_seen.get(tid, self.frame_idx) + 1)
        mean_age = float(np.mean(ages)) if ages else 0.0
        med_age = float(np.median(ages)) if ages else 0.0
        max_age = float(np.max(ages)) if ages else 0.0

        p_confs = [c for cid, c in self.last_detections if cid == 0]
        o_confs = [c for cid, c in self.last_detections if cid != 0]
        p_conf_mean = float(np.mean(p_confs)) if p_confs else 0.0
        p_conf_max = float(np.max(p_confs)) if p_confs else 0.0
        o_conf_mean = float(np.mean(o_confs)) if o_confs else 0.0
        o_conf_max = float(np.max(o_confs)) if o_confs else 0.0

        grid = np.zeros((3,3), dtype=float)
        if self.w > 0 and self.h > 0:
            for p in people:
                x, y = p["center"]
                gi = min(2, max(0, int((y / self.h) * 3)))
                gj = min(2, max(0, int((x / self.w) * 3)))
                grid[gi, gj] += 1.0
        grid_mean, grid_max, grid_std = float(grid.mean()), float(grid.max()), float(grid.std())

        diag = self.img_diag
        features = {
            "num_people": float(num_people),
            "num_objects": float(num_objects),
            "avg_speed_px": avg_speed,
            "max_speed_px": max_speed,
            "avg_accel_px": avg_accel,
            "fast_ratio": fast_ratio,
            "loiter_ratio": loiter_ratio,
            "stationary_ratio": stationary_ratio,
            "min_person_obj_px": min_person_obj,
            "min_inter_person_px": min_inter_person,
            "avg_inter_person_px": avg_inter_person,
            "min_person_obj_norm": min_person_obj/diag,
            "min_inter_person_norm": min_inter_person/diag,
            "avg_inter_person_norm": avg_inter_person/diag,
            "mean_track_age": mean_age,
            "median_track_age": med_age,
            "max_track_age": max_age,
            "person_conf_mean": p_conf_mean,
            "person_conf_max": p_conf_max,
            "object_conf_mean": o_conf_mean,
            "object_conf_max": o_conf_max,
            "grid_mean": grid_mean,
            "grid_max": grid_max,
            "grid_std": grid_std,
        }
        vec = np.array([features[k] for k in self.feature_order], dtype=np.float32)
        vec[~np.isfinite(vec)] = 0.0
        return features, vec


class ScorerAdapter:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        with open(self.artifacts_dir / "meta.json", "r") as f:
            self.meta = json.load(f)
        self.feature_order = self.meta["feature_order"]
        self.threshold = float(self.meta["threshold"])
        self.model = joblib.load(self.artifacts_dir / "lgbm_frame_scorer.joblib")
        self.detector = Detector()
        self.tracker = Tracker()
        self.logger = None

    def start_stream(self, fps: float, w: int, h: int):
        self.logger = FeatureLogger(fps=fps, w=w, h=h, feature_order=self.feature_order)

    def score_frame(self, frame_bgr: np.ndarray, cached_dets: Optional[List[Dict]] = None):
        if self.logger is None:
            h, w = frame_bgr.shape[:2]
            self.start_stream(fps=30.0, w=w, h=h)

        if cached_dets is None:
            dets, class_names = self.detector.detect(frame_bgr)
        else:
            dets = [(float(d["box"][0]), float(d["box"][1]), float(d["box"][2]), float(d["box"][3]),
                     float(d["conf"]), int(d["cls"])) for d in cached_dets]
            class_names = self.detector.names

        self.logger.set_last_detections(dets)
        tracked = self.tracker.update(frame_bgr, dets, class_names)
        self.logger.step(tracked)
        features, vec = self.logger._aggregate(tracked)
        score = float(self.model.predict_proba(vec.reshape(1, -1))[:, 1])
        return tracked, features, score, dets

