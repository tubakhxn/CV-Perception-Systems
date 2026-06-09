"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          MONOCULAR 3D VEHICLE PERCEPTION SYSTEM                             ║
║          Tesla-Inspired Autonomous Driving Perception                       ║
║          Dev/Creator : tubakhxn | github.com/tubakhxn                      ║
║          Research & Educational Purposes Only                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import subprocess
import importlib

# ─────────────────────────────────────────────────────────────────────────────
#  AUTO DEPENDENCY INSTALLER
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED = {
    "cv2":          "opencv-python",
    "numpy":        "numpy",
    "matplotlib":   "matplotlib",
    "scipy":        "scipy",
    "ultralytics":  "ultralytics",
    "PIL":          "Pillow",
    "tqdm":         "tqdm",
    "torch":        "torch",
    "torchvision":  "torchvision",
    "open3d":       "open3d",
    "huggingface_hub": "huggingface_hub",
    "transformers": "transformers",
}

def install_missing():
    missing = []
    for mod, pkg in REQUIRED.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[SETUP] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
        print("[SETUP] All dependencies installed.\n")
    else:
        print("[SETUP] All dependencies satisfied.\n")

install_missing()

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from PIL import Image
from tqdm import tqdm
import torch
import time
import math
import warnings
import traceback
warnings.filterwarnings("ignore")

from ultralytics import YOLO
from collections import defaultdict, deque

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║   MONOCULAR 3D VEHICLE PERCEPTION SYSTEM  v1.0                             ║
║   Dev/Creator : tubakhxn | github.com/tubakhxn                            ║
║   Research & Educational Purposes Only                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# HUD Colors (BGR for OpenCV, RGB for matplotlib)
NEON_GREEN   = (0, 255, 128)
NEON_CYAN    = (0, 255, 255)
NEON_YELLOW  = (0, 255, 200)
NEON_RED     = (0, 80, 255)
NEON_ORANGE  = (0, 165, 255)
NEON_BLUE    = (255, 200, 0)
DARK_BG      = (8, 12, 20)
GRID_COLOR   = (30, 60, 40)

# Matplotlib RGB equivalents
M_GREEN  = (0.0, 1.0, 0.5)
M_CYAN   = (0.0, 1.0, 1.0)
M_RED    = (1.0, 0.2, 0.1)
M_ORANGE = (1.0, 0.6, 0.0)
M_YELLOW = (1.0, 0.9, 0.0)
M_BLUE   = (0.3, 0.7, 1.0)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FOCAL_LENGTH_PX = 720.0
REAL_CAR_HEIGHT  = 1.5   # metres
REAL_CAR_WIDTH   = 1.9
REAL_CAR_LENGTH  = 4.5
CAMERA_HEIGHT    = 1.2   # metres above ground

# ─────────────────────────────────────────────────────────────────────────────
#  DEPTH ESTIMATOR  (Depth Anything V2 tiny → MiDaS fallback → geometric)
# ─────────────────────────────────────────────────────────────────────────────
class DepthEstimator:
    def __init__(self):
        self.model = None
        self.mode  = "geometric"
        self._load_model()

    def _load_model(self):
        # Try Depth Anything V2 via transformers
        try:
            from transformers import pipeline as hf_pipeline
            print("[DEPTH] Loading Depth Anything V2 (tiny) …")
            self.model = hf_pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if torch.cuda.is_available() else -1,
            )
            self.mode = "depth_anything"
            print("[DEPTH] Depth Anything V2 ready.")
            return
        except Exception as e:
            print(f"[DEPTH] Depth Anything V2 unavailable ({e}), trying MiDaS …")

        # Fallback: MiDaS via torch.hub
        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small",
                pretrained=True, verbose=False
            )
            self.model.eval()
            self.transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", verbose=False
            ).small_transform
            self.mode = "midas"
            print("[DEPTH] MiDaS small ready.")
            return
        except Exception as e:
            print(f"[DEPTH] MiDaS unavailable ({e}), using geometric depth.")

    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        if self.mode == "depth_anything":
            try:
                pil = Image.fromarray(frame_rgb)
                out = self.model(pil)
                depth = np.array(out["depth"], dtype=np.float32)
                depth = cv2.resize(depth, (w, h))
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                return depth
            except Exception:
                pass
        if self.mode == "midas":
            try:
                inp = self.transforms(frame_rgb)
                with torch.no_grad():
                    pred = self.model(inp)
                    pred = torch.nn.functional.interpolate(
                        pred.unsqueeze(1), size=(h, w),
                        mode="bicubic", align_corners=False
                    ).squeeze()
                depth = pred.cpu().numpy().astype(np.float32)
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                return depth
            except Exception:
                pass
        # Pure geometric fallback
        return self._geometric_depth(frame_rgb)

    def _geometric_depth(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        # Vertical gradient — lower in image = closer
        y_coords = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
        depth = 1.0 - y_coords * 0.8
        # Darken lane region estimate
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
        depth = depth - gaussian_filter(edges, sigma=5) * 0.15
        depth = np.clip((depth - depth.min()) / (depth.max() - depth.min() + 1e-6), 0, 1)
        return depth.astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
#  OBJECT TRACKER  (simple IoU-based SORT-lite)
# ─────────────────────────────────────────────────────────────────────────────
class Track:
    _next_id = 1
    def __init__(self, bbox, cls_id, dist):
        self.id       = Track._next_id; Track._next_id += 1
        self.bbox     = bbox        # x1,y1,x2,y2
        self.cls_id   = cls_id
        self.dist     = dist
        self.hits     = 1
        self.missed   = 0
        self.history  = deque(maxlen=30)  # centre points
        self.speeds   = deque(maxlen=10)
        cx = (bbox[0]+bbox[2])//2; cy = (bbox[1]+bbox[3])//2
        self.history.append((cx, cy))
        self.prev_dist = dist
        self.speed_mps = 0.0
        self.color     = self._rand_color()

    def _rand_color(self):
        palette = [NEON_GREEN, NEON_CYAN, NEON_YELLOW, NEON_ORANGE, NEON_BLUE]
        return palette[self.id % len(palette)]

    def update(self, bbox, dist, fps):
        self.bbox  = bbox
        self.prev_dist = self.dist
        self.dist  = dist
        self.hits += 1
        self.missed = 0
        cx = (bbox[0]+bbox[2])//2; cy = (bbox[1]+bbox[3])//2
        self.history.append((cx, cy))
        delta_d = self.prev_dist - dist          # +ve = approaching
        self.speed_mps = delta_d * fps
        self.speeds.append(self.speed_mps)

    def predict_trajectory(self, steps=15):
        if len(self.history) < 3:
            return []
        pts = list(self.history)[-6:]
        if len(pts) < 2:
            return []
        dx = (pts[-1][0] - pts[0][0]) / max(len(pts)-1, 1)
        dy = (pts[-1][1] - pts[0][1]) / max(len(pts)-1, 1)
        last = pts[-1]
        traj = [(int(last[0] + dx*i), int(last[1] + dy*i)) for i in range(1, steps+1)]
        return traj

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/max(ua,1e-6)

class Tracker:
    def __init__(self, iou_thresh=0.35, max_miss=8):
        self.tracks    = []
        self.iou_thresh = iou_thresh
        self.max_miss   = max_miss

    def update(self, detections, fps):
        """detections: list of (bbox, cls_id, conf, dist)"""
        # Mark all missed
        for t in self.tracks:
            t.missed += 1

        matched_track_ids = set()
        matched_det_ids   = set()

        # Match by IoU
        for di, (bbox, cls_id, conf, dist) in enumerate(detections):
            best_iou = self.iou_thresh; best_t = None
            for t in self.tracks:
                if t.missed > 1: continue
                sc = iou(bbox, t.bbox)
                if sc > best_iou:
                    best_iou = sc; best_t = t
            if best_t is not None:
                best_t.update(bbox, dist, fps)
                best_t.missed = 0
                matched_track_ids.add(id(best_t))
                matched_det_ids.add(di)

        # New tracks for unmatched dets
        for di, (bbox, cls_id, conf, dist) in enumerate(detections):
            if di not in matched_det_ids:
                self.tracks.append(Track(bbox, cls_id, dist))

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_miss]
        return self.tracks

# ─────────────────────────────────────────────────────────────────────────────
#  DISTANCE ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
def estimate_distance(bbox, frame_h, frame_w, depth_map=None):
    x1,y1,x2,y2 = bbox
    obj_h_px = max(y2 - y1, 1)
    obj_w_px = max(x2 - x1, 1)

    # Geometric via apparent height
    dist_h = (REAL_CAR_HEIGHT * FOCAL_LENGTH_PX) / obj_h_px
    dist_w = (REAL_CAR_WIDTH  * FOCAL_LENGTH_PX) / obj_w_px
    dist_geo = (dist_h + dist_w) / 2.0

    if depth_map is not None:
        # Depth map sample at bbox centre
        cy = int(np.clip((y1+y2)//2, 0, depth_map.shape[0]-1))
        cx = int(np.clip((x1+x2)//2, 0, depth_map.shape[1]-1))
        patch = depth_map[max(0,cy-5):cy+5, max(0,cx-5):cx+5]
        d_norm = float(np.median(patch)) if patch.size else 0.5
        # Invert: depth_anything gives closer=larger in 0-1 space
        dist_neural = CAMERA_HEIGHT / (d_norm + 0.05) * 3.5
        dist = 0.4 * dist_geo + 0.6 * dist_neural
    else:
        dist = dist_geo

    return float(np.clip(dist, 1.5, 120.0))

# ─────────────────────────────────────────────────────────────────────────────
#  COLLISION RISK
# ─────────────────────────────────────────────────────────────────────────────
def collision_risk(tracks):
    if not tracks:
        return 0.0
    risks = []
    for t in tracks:
        d = max(t.dist, 1.0)
        speed = np.mean(list(t.speeds)) if t.speeds else 0.0
        approach = max(speed, 0.0)
        ttc = d / (approach + 0.1)  # time to collision
        risk = 1.0 / (ttc * 0.2 + 1) + 1.0 / (d * 0.05 + 1)
        risks.append(np.clip(risk, 0, 1))
    return float(np.clip(max(risks), 0, 1))

# ─────────────────────────────────────────────────────────────────────────────
#  DRAW HELPERS  (OpenCV panel)
# ─────────────────────────────────────────────────────────────────────────────
def draw_scanlines(img, alpha=0.06):
    h, w = img.shape[:2]
    for y in range(0, h, 4):
        cv2.line(img, (0,y), (w,y), (0,0,0), 1)
    overlay = img.copy()
    cv2.addWeighted(overlay, 1-alpha, img, alpha, 0, img)

def draw_hud_border(img, color=NEON_GREEN, thickness=2):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0,0), (w-1,h-1), color, thickness)
    L = 20
    # Corner brackets
    for (cx,cy,sx,sy) in [(0,0,1,1),(w-1,0,-1,1),(0,h-1,1,-1),(w-1,h-1,-1,-1)]:
        cv2.line(img,(cx,cy),(cx+sx*L,cy), color,2)
        cv2.line(img,(cx,cy),(cx,cy+sy*L), color,2)

def draw_3d_bbox(img, bbox, dist, track_id, cls_name, color, conf):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    bh = y2-y1; bw = x2-x1

    # Perspective offset for pseudo-3D
    off_x = int(bw * 0.18); off_y = int(-bh * 0.22)
    pts_front = np.array([[x1,y2],[x2,y2],[x2,y1],[x1,y1]], dtype=np.int32)
    pts_back  = pts_front + np.array([off_x, off_y])

    # Draw back face (dim)
    dim = tuple(int(c*0.45) for c in color)
    cv2.polylines(img, [pts_back], True, dim, 1, cv2.LINE_AA)
    # Connectors
    for a, b in zip(pts_front, pts_back):
        cv2.line(img, tuple(a), tuple(b), dim, 1, cv2.LINE_AA)
    # Front face
    cv2.polylines(img, [pts_front], True, color, 2, cv2.LINE_AA)

    # Label
    risk_col = NEON_RED if dist < 15 else (NEON_ORANGE if dist < 30 else NEON_GREEN)
    label = f"#{track_id} {cls_name} {dist:.1f}m"
    lx = x1; ly = y1 - 8 if y1 > 20 else y2 + 18
    cv2.rectangle(img, (lx-2, ly-14), (lx+len(label)*8+2, ly+4), (0,0,0), -1)
    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, risk_col, 1, cv2.LINE_AA)

def draw_trajectory_cv(img, track):
    traj = track.predict_trajectory()
    hist = list(track.history)
    # Draw history trail
    for i in range(1, len(hist)):
        alpha = i / len(hist)
        c = tuple(int(v*alpha) for v in track.color)
        cv2.line(img, hist[i-1], hist[i], c, 1, cv2.LINE_AA)
    # Draw prediction dashes
    for i in range(1, len(traj)):
        cv2.line(img, traj[i-1], traj[i], NEON_YELLOW, 1, cv2.LINE_AA)

def draw_analytics_bar(img, tracks, frame_no, fps, risk_score):
    h, w = img.shape[:2]
    bar_h = 56
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (10, 16, 28)
    # Separator line
    cv2.line(bar, (0, 0), (w, 0), NEON_GREEN, 1)

    n_vehicles  = len(tracks)
    closest     = min((t.dist for t in tracks), default=999.0)
    avg_density = n_vehicles / max(1, w/100)
    risk_pct    = int(risk_score * 100)
    risk_color  = NEON_RED if risk_score > 0.6 else (NEON_ORANGE if risk_score > 0.3 else NEON_GREEN)

    stats = [
        (f"VEHICLES: {n_vehicles}", NEON_CYAN),
        (f"CLOSEST: {closest:.1f}m", NEON_YELLOW if closest<20 else NEON_GREEN),
        (f"RISK: {risk_pct}%", risk_color),
        (f"TRACKS: {len(tracks)}", NEON_CYAN),
        (f"FPS: {fps:.1f}", NEON_GREEN),
        (f"FRAME: {frame_no}", (120,120,120)),
        ("tubakhxn", (60,80,60)),
    ]
    col_w = w // len(stats)
    for i, (txt, col) in enumerate(stats):
        x = i * col_w + 8
        cv2.putText(bar, txt, (x, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)
    img[h-bar_h:h] = bar

def draw_lane_overlay(img):
    h, w = img.shape[:2]
    overlay = img.copy()
    vanish_y = int(h * 0.44)
    pts = np.array([
        [int(w*0.44), vanish_y],
        [int(w*0.56), vanish_y],
        [int(w*0.75), h],
        [int(w*0.25), h],
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (0, 60, 20))
    cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
    cv2.polylines(img, [pts[[0,3]]], False, NEON_GREEN, 2, cv2.LINE_AA)
    cv2.polylines(img, [pts[[1,2]]], False, NEON_GREEN, 2, cv2.LINE_AA)
    # Dashed centre
    for i in range(10):
        t = i / 10.0; t2 = (i+0.5)/10.0
        x1 = int(w*0.44 + (w*0.25 - w*0.44)*t)
        y1 = int(vanish_y + (h - vanish_y)*t)
        x2 = int(w*0.44 + (w*0.25 - w*0.44)*t2)
        y2 = int(vanish_y + (h - vanish_y)*t2)
        mid_x1 = (x1 + int(w*0.56 + (w*0.75 - w*0.56)*t))//2
        mid_x2 = (x2 + int(w*0.56 + (w*0.75 - w*0.56)*t2))//2
        cv2.line(img, (mid_x1, y1), (mid_x2, y2), NEON_YELLOW, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
#  3D PERCEPTION PANEL  (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def build_3d_panel(tracks, depth_map, panel_w, panel_h):
    """Return an np.uint8 BGR image of the 3D perception panel."""
    dpi = 100
    fig_w = panel_w / dpi; fig_h = panel_h / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=(0.03,0.05,0.08))
    ax  = fig.add_axes([0,0.12,1,0.88], projection='3d')
    ax.set_facecolor((0.03,0.05,0.08))
    fig.patch.set_facecolor((0.03,0.05,0.08))

    # ── Ground grid ──────────────────────────────────────────────────────────
    gx = np.linspace(-12, 12, 20)
    gz = np.linspace(0, 80, 30)
    GX, GZ = np.meshgrid(gx, gz)
    GY = np.zeros_like(GX)
    ax.plot_wireframe(GX, GZ, GY, color=(0.0, 0.28, 0.12), linewidth=0.3, alpha=0.55)

    # ── Pseudo point cloud from depth map ────────────────────────────────────
    if depth_map is not None:
        h_dm, w_dm = depth_map.shape
        step = max(1, h_dm // 55)
        ys_idx = np.arange(0, h_dm, step)
        xs_idx = np.arange(0, w_dm, step)
        YY, XX = np.meshgrid(ys_idx, xs_idx, indexing='ij')
        D = depth_map[YY, XX]
        # Convert to world coords
        z_world = (1.0 - D + 0.05) * 80.0  # far=0→z=80
        x_world = (XX / w_dm - 0.5) * z_world * 0.9
        y_world = np.zeros_like(z_world)

        # Mask ground plane and sky
        valid = (D > 0.05) & (D < 0.97) & (ys_idx[:, None] > h_dm * 0.35)
        z_v = z_world[valid]; x_v = x_world[valid]; y_v = y_world[valid]
        d_v = D[valid]

        if len(z_v) > 0:
            colors = plt.cm.plasma(1.0 - d_v)
            ax.scatter(x_v, z_v, y_v, c=colors, s=0.6, alpha=0.55, linewidths=0)

    # ── 3D Vehicle boxes ──────────────────────────────────────────────────────
    for t in tracks:
        x1,y1,x2,y2 = t.bbox
        d = t.dist
        # World position
        img_cx_norm = ((x1+x2)/2 - panel_w/2) / (panel_w/2)
        wx = img_cx_norm * d * 0.5
        wz = d
        wy = 0.0

        cw = REAL_CAR_WIDTH; cl = REAL_CAR_LENGTH; ch = REAL_CAR_HEIGHT

        # 8 corners of 3D box
        corners = np.array([
            [wx-cw/2, wz-cl/2, wy],
            [wx+cw/2, wz-cl/2, wy],
            [wx+cw/2, wz+cl/2, wy],
            [wx-cw/2, wz+cl/2, wy],
            [wx-cw/2, wz-cl/2, wy+ch],
            [wx+cw/2, wz-cl/2, wy+ch],
            [wx+cw/2, wz+cl/2, wy+ch],
            [wx-cw/2, wz+cl/2, wy+ch],
        ])
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        col = M_GREEN if d > 30 else (M_ORANGE if d > 15 else M_RED)
        for a,b in edges:
            ax.plot([corners[a,0],corners[b,0]],
                    [corners[a,1],corners[b,1]],
                    [corners[a,2],corners[b,2]],
                    color=col, lw=1.2, alpha=0.9)

        # Trajectory in 3D
        traj = t.predict_trajectory(10)
        if len(traj) > 1 and len(t.history) >= 2:
            hist = list(t.history)
            # Project hist to world
            tx = [img_cx_norm * (t.dist + i*0.5) * 0.5 for i in range(-len(hist),0)]
            tz = [t.dist + i*0.5 for i in range(-len(hist),0)]
            ty = [0.0]*len(tx)
            ax.plot(tx, tz, ty, color=M_CYAN, lw=0.8, alpha=0.6, linestyle='--')

        # Distance label
        ax.text(wx, wz, ch+0.5,
                f"#{t.id} {t.dist:.1f}m",
                color='#00ffaa', fontsize=5.5, ha='center')

    # ── Ego vehicle indicator ─────────────────────────────────────────────────
    ax.scatter([0], [0], [0], c=[[1,1,0]], s=60, marker='^', zorder=10)
    ax.text(0, -3, 0.5, "EGO", color='yellow', fontsize=6, ha='center')

    # ── Axes styling ─────────────────────────────────────────────────────────
    ax.set_xlim(-15, 15); ax.set_ylim(0, 85); ax.set_zlim(0, 6)
    ax.set_xlabel("X (m)", color=(0,1,0.5), fontsize=6, labelpad=1)
    ax.set_ylabel("Depth (m)", color=(0,1,0.5), fontsize=6, labelpad=1)
    ax.set_zlabel("Z (m)", color=(0,1,0.5), fontsize=6, labelpad=1)
    ax.tick_params(colors=(0.2,0.8,0.3), labelsize=5)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor((0,0.3,0.1,0.3))
    ax.view_init(elev=22, azim=-65)

    # ── Bottom info bar ───────────────────────────────────────────────────────
    ax2 = fig.add_axes([0, 0, 1, 0.1])
    ax2.set_facecolor((0.02,0.06,0.10))
    ax2.axis('off')
    n_v = len(tracks)
    closest = min((t.dist for t in tracks), default=0.0)
    ax2.text(0.02, 0.5, f"VEHICLES: {n_v}   CLOSEST: {closest:.1f}m   TRACKS: {n_v}",
             color=(0,1,0.5), fontsize=7, va='center', transform=ax2.transAxes)
    ax2.text(0.98, 0.5, "3D PERCEPTION | tubakhxn",
             color=(0.3,0.5,0.3), fontsize=6, va='center', ha='right',
             transform=ax2.transAxes)

    # ── FIX: use buffer_rgba() instead of removed tostring_rgb() ─────────────
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]   # RGBA → RGB, drop alpha
    plt.close(fig)
    panel = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    panel = cv2.resize(panel, (panel_w, panel_h))
    return panel

# ─────────────────────────────────────────────────────────────────────────────
#  DEPTH MAP COLORIZER  (for output PNG)
# ─────────────────────────────────────────────────────────────────────────────
def colorize_depth(depth_map):
    dm = (depth_map * 255).astype(np.uint8)
    colored = cv2.applyColorMap(dm, cv2.COLORMAP_PLASMA)
    return colored

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_video(input_path: str):
    print(BANNER)
    print(f"[INIT] Input  : {input_path}")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}"); sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INIT] Video  : {src_w}x{src_h} @ {src_fps:.1f}fps  |  {total_frames} frames")

    # Output dimensions
    LEFT_W  = 640; LEFT_H  = 400
    RIGHT_W = 640; RIGHT_H = 400
    OUT_W   = LEFT_W + RIGHT_W
    OUT_H   = LEFT_H + 60        # +60 for analytics bar

    # ── Load YOLO ─────────────────────────────────────────────────────────────
    print("[YOLO] Loading YOLOv8n …")
    yolo = YOLO("yolov8n.pt")
    print("[YOLO] Ready.")

    # ── Load Depth Estimator ──────────────────────────────────────────────────
    depth_est = DepthEstimator()

    # ── Setup output writer ───────────────────────────────────────────────────
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (OUT_W, OUT_H))
    print(f"[OUT]  Writing to {output_path}  ({OUT_W}x{OUT_H})")

    tracker = Tracker()
    frame_no = 0
    t_prev   = time.time()
    fps_display = src_fps

    # Dashboard accumulator
    dashboard_frame = None
    saved_depth     = None

    # ── DEPTH SKIP: run depth every N frames for speed ────────────────────────
    DEPTH_EVERY = 4
    cached_depth = None

    print(f"\n[RUN]  Processing {total_frames} frames …\n")
    pbar = tqdm(total=total_frames, unit="frame",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_no += 1
        pbar.update(1)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── Depth estimation ──────────────────────────────────────────────────
        if frame_no % DEPTH_EVERY == 1 or cached_depth is None:
            small = cv2.resize(frame_rgb, (320, 192))
            cached_depth = depth_est.estimate(small)
            cached_depth = cv2.resize(cached_depth, (src_w, src_h))

        depth_map = cached_depth

        # ── YOLO detection ────────────────────────────────────────────────────
        results = yolo(frame_rgb, verbose=False, conf=0.30,
                       classes=list(VEHICLE_CLASSES.keys()))
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES: continue
                conf   = float(box.conf[0])
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                dist = estimate_distance((x1,y1,x2,y2), src_h, src_w, depth_map)
                detections.append(((x1,y1,x2,y2), cls_id, conf, dist))

        # ── Update tracker ────────────────────────────────────────────────────
        now = time.time(); fps_display = 0.9*fps_display + 0.1*(1.0/(now-t_prev+1e-6)); t_prev = now
        tracks = tracker.update(detections, src_fps)
        risk   = collision_risk(tracks)

        # ─────────────────────────────────────────────────────────────────────
        #  LEFT PANEL: Camera + detections
        # ─────────────────────────────────────────────────────────────────────
        left = cv2.resize(frame_bgr, (LEFT_W, LEFT_H))
        draw_lane_overlay(left)

        for t in tracks:
            if t.missed > 0: continue
            # Scale bbox to panel
            sx = LEFT_W / src_w; sy = LEFT_H / src_h
            bx1 = int(t.bbox[0]*sx); by1 = int(t.bbox[1]*sy)
            bx2 = int(t.bbox[2]*sx); by2 = int(t.bbox[3]*sy)
            cls_name = VEHICLE_CLASSES.get(t.cls_id, "vehicle")
            draw_3d_bbox(left, (bx1,by1,bx2,by2), t.dist, t.id, cls_name, t.color, 1.0)
            # Trajectory
            tscaled = Track.__new__(Track)
            tscaled.__dict__ = t.__dict__.copy()
            tscaled.history = deque([(int(x*sx), int(y*sy)) for x,y in t.history], maxlen=30)
            draw_trajectory_cv(left, tscaled)

        # Depth overlay (subtle)
        depth_colored = colorize_depth(cv2.resize(depth_map, (LEFT_W, LEFT_H)))
        cv2.addWeighted(left, 0.82, depth_colored, 0.18, 0, left)

        # Corner label
        cv2.putText(left, "CAMERA VIEW", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_CYAN, 1, cv2.LINE_AA)
        draw_hud_border(left, NEON_GREEN)
        draw_scanlines(left)

        # ─────────────────────────────────────────────────────────────────────
        #  RIGHT PANEL: 3D perception
        # ─────────────────────────────────────────────────────────────────────
        right = build_3d_panel(
            [t for t in tracks if t.missed == 0],
            cv2.resize(depth_map, (320,192)) if depth_map is not None else None,
            RIGHT_W, RIGHT_H
        )
        cv2.putText(right, "3D PERCEPTION", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_CYAN, 1, cv2.LINE_AA)
        draw_hud_border(right, NEON_CYAN)

        # ─────────────────────────────────────────────────────────────────────
        #  COMPOSITE
        # ─────────────────────────────────────────────────────────────────────
        composite = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        composite[:LEFT_H, :LEFT_W] = left
        composite[:RIGHT_H, LEFT_W:LEFT_W+RIGHT_W] = right

        draw_analytics_bar(composite, [t for t in tracks if t.missed==0],
                           frame_no, fps_display, risk)

        writer.write(composite)

        # Save dashboard snapshot at frame 60 (or last available)
        if frame_no == 60 or (frame_no == total_frames):
            dashboard_frame = composite.copy()
            saved_depth     = depth_map.copy() if depth_map is not None else None

    pbar.close()
    cap.release()
    writer.release()
    print(f"\n[DONE] Wrote {output_path}")

    # ── Save perception_dashboard.png ─────────────────────────────────────────
    if dashboard_frame is not None:
        cv2.imwrite("perception_dashboard.png", dashboard_frame)
        print("[DONE] Wrote perception_dashboard.png")

    # ── Save depth_map.png ────────────────────────────────────────────────────
    if saved_depth is not None:
        depth_png = colorize_depth(saved_depth)
        cv2.imwrite("depth_map.png", depth_png)
        print("[DONE] Wrote depth_map.png")

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  All outputs saved.  tubakhxn / github.com/tubakhxn ║")
    print("╚══════════════════════════════════════════════════════╝\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        print("Example: python main.py dashcam.mp4")
        sys.exit(1)

    input_video = sys.argv[1]
    if not os.path.isfile(input_video):
        print(f"[ERROR] File not found: {input_video}")
        sys.exit(1)

    try:
        process_video(input_video)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)