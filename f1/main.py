"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        FORMULA 1 RACE INTELLIGENCE SYSTEM  v1.0                            ║
║        AI-Powered Motorsport Analytics Platform                             ║
║        Dev/Creator : tubakhxn | github.com/tubakhxn                       ║
║        Research & Educational Purposes Only                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import subprocess
import importlib

# ─────────────────────────────────────────────────────────────────────────────
#  AUTO DEPENDENCY INSTALLER
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_PACKAGES = {
    "cv2":         "opencv-python",
    "numpy":       "numpy",
    "matplotlib":  "matplotlib",
    "scipy":       "scipy",
    "ultralytics": "ultralytics",
    "PIL":         "Pillow",
    "tqdm":        "tqdm",
    "torch":       "torch",
    "torchvision": "torchvision",
}

def install_missing_dependencies():
    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        print(f"\n[SETUP] Installing: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("[SETUP] All dependencies installed.\n")
    else:
        print("[SETUP] All dependencies satisfied.\n")

install_missing_dependencies()

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Wedge, Circle
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from collections import defaultdict, deque
import time
import math
import warnings
import traceback
import random

warnings.filterwarnings("ignore")
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║   FORMULA 1 RACE INTELLIGENCE SYSTEM  v1.0                                ║
║   Dev/Creator : tubakhxn | github.com/tubakhxn                           ║
║   Research & Educational Purposes Only                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Detection classes treated as F1 cars (vehicles)
VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck in COCO
# We map all to "car" in F1 context

# BGR colours for OpenCV
C_RED       = (0,   40,  220)
C_ORANGE    = (0,  165,  255)
C_YELLOW    = (0,  220,  255)
C_GREEN     = (0,  210,   80)
C_CYAN      = (255, 220,   0)
C_BLUE      = (255, 100,   0)
C_WHITE     = (230, 230,  230)
C_MAGENTA   = (200,   0,  200)
C_DARK      = ( 10,  14,  22)
C_GRID      = ( 28,  48,  28)
C_NEON_G    = (  0, 255, 128)
C_NEON_C    = (255, 255,   0)
C_NEON_Y    = ( 30, 240, 255)

# Matplotlib RGB
M_RED     = (0.9,  0.15, 0.05)
M_ORANGE  = (1.0,  0.55, 0.00)
M_YELLOW  = (1.0,  0.90, 0.10)
M_GREEN   = (0.05, 0.85, 0.35)
M_CYAN    = (0.10, 0.90, 1.00)
M_BLUE    = (0.20, 0.55, 1.00)
M_WHITE   = (0.90, 0.90, 0.90)
M_PURPLE  = (0.75, 0.20, 1.00)

TRACK_PALETTE = [
    M_RED, M_ORANGE, M_YELLOW, M_GREEN, M_CYAN, M_BLUE, M_WHITE, M_PURPLE,
    (1.0,0.4,0.7),(0.4,1.0,0.6),(1.0,0.7,0.3),(0.5,0.8,1.0),
]

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def bbox_centre(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)//2, (y1+y2)//2)

def bbox_area(bbox):
    x1,y1,x2,y2 = bbox
    return max(0,(x2-x1)*(y2-y1))

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/max(union,1)

def lerp_color(c1, c2, t):
    return tuple(int(c1[i]*(1-t)+c2[i]*t) for i in range(3))

def draw_scanlines(img, step=4, alpha=0.07):
    h,w = img.shape[:2]
    scan = img.copy()
    for y in range(0,h,step):
        cv2.line(scan,(0,y),(w,y),(0,0,0),1)
    cv2.addWeighted(scan,alpha,img,1-alpha,0,img)

def draw_corner_brackets(img, color, size=18, thickness=2):
    h,w = img.shape[:2]
    for (cx,cy,sx,sy) in [(0,0,1,1),(w-1,0,-1,1),(0,h-1,1,-1),(w-1,h-1,-1,-1)]:
        cv2.line(img,(cx,cy),(cx+sx*size,cy),color,thickness,cv2.LINE_AA)
        cv2.line(img,(cx,cy),(cx,cy+sy*size),color,thickness,cv2.LINE_AA)

def put_label(img, text, pos, color, scale=0.42, thickness=1, bg=True):
    x,y = pos
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if bg:
        cv2.rectangle(img,(x-2,y-th-3),(x+tw+2,y+3),(0,0,0),-1)
    cv2.putText(img, text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
#  RACE CAR TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class RaceCar:
    _next_id = 1

    def __init__(self, bbox, frame_no):
        self.id        = RaceCar._next_id; RaceCar._next_id += 1
        self.bbox      = bbox
        self.missed    = 0
        self.hits      = 1
        self.birth     = frame_no

        # History
        self.trail     = deque(maxlen=80)   # (cx,cy) world pixels
        self.speed_px  = deque(maxlen=15)   # pixel displacement per frame
        self.area_hist = deque(maxlen=10)

        cx,cy = bbox_centre(bbox)
        self.trail.append((cx,cy))
        self.area_hist.append(bbox_area(bbox))

        # Race state
        self.position       = self.id          # estimated race pos
        self.prev_position  = self.id
        self.lap_sector     = 0
        self.pit_candidate  = False
        self.overtake_flag  = 0                # frames since last overtake
        self.speed_score    = 0.0
        self.color_rgb      = TRACK_PALETTE[(self.id-1) % len(TRACK_PALETTE)]
        self.color_bgr      = tuple(int(c*255) for c in reversed(self.color_rgb))

    def update(self, bbox, frame_no):
        prev_cx,prev_cy = self.trail[-1] if self.trail else bbox_centre(bbox)
        cx,cy = bbox_centre(bbox)
        disp = math.hypot(cx-prev_cx, cy-prev_cy)
        self.speed_px.append(disp)
        self.speed_score = float(np.mean(self.speed_px)) if self.speed_px else 0.0
        self.bbox   = bbox
        self.missed = 0
        self.hits  += 1
        self.trail.append((cx,cy))
        self.area_hist.append(bbox_area(bbox))

    def avg_speed(self):
        return float(np.mean(self.speed_px)) if self.speed_px else 0.0

    def trajectory_points(self, n=25):
        pts = list(self.trail)[-n:]
        return pts

    def predict_next(self, steps=12):
        pts = list(self.trail)[-6:]
        if len(pts) < 2: return []
        dx = (pts[-1][0]-pts[0][0])/max(len(pts)-1,1)
        dy = (pts[-1][1]-pts[0][1])/max(len(pts)-1,1)
        last = pts[-1]
        return [(int(last[0]+dx*i), int(last[1]+dy*i)) for i in range(1,steps+1)]


class RaceTracker:
    def __init__(self, iou_thresh=0.30, max_miss=12):
        self.cars       = []
        self.iou_thresh = iou_thresh
        self.max_miss   = max_miss
        self.overtakes  = 0
        self.pit_events = 0

    def update(self, detections, frame_no):
        """detections: list of (x1,y1,x2,y2,conf)"""
        for car in self.cars:
            car.missed += 1

        matched_cars = set()
        matched_dets = set()

        for di,(bbox,conf) in enumerate(detections):
            best_score = self.iou_thresh
            best_car   = None
            for car in self.cars:
                if car.missed > 2: continue
                sc = iou(bbox, car.bbox)
                if sc > best_score:
                    best_score = sc; best_car = car
            if best_car is not None:
                best_car.update(bbox, frame_no)
                matched_cars.add(id(best_car))
                matched_dets.add(di)

        for di,(bbox,conf) in enumerate(detections):
            if di not in matched_dets:
                self.cars.append(RaceCar(bbox, frame_no))

        self.cars = [c for c in self.cars if c.missed <= self.max_miss]
        self._update_positions()
        self._detect_pit_activity()
        return self.active_cars()

    def active_cars(self):
        return [c for c in self.cars if c.missed == 0]

    def _update_positions(self):
        active = self.active_cars()
        # Sort by vertical position (lower y = further ahead on typical circuit cams)
        active_sorted = sorted(active, key=lambda c: bbox_centre(c.bbox)[1])
        for rank, car in enumerate(active_sorted):
            car.prev_position = car.position
            car.position      = rank + 1
            if car.prev_position != car.position:
                if car.prev_position > car.position:
                    car.overtake_flag = 45
                    self.overtakes += 1

        # Decay overtake flags
        for car in self.cars:
            if car.overtake_flag > 0:
                car.overtake_flag -= 1

    def _detect_pit_activity(self):
        for car in self.active_cars():
            spd = car.avg_speed()
            area = float(np.mean(car.area_hist)) if car.area_hist else 0
            if spd < 1.2 and car.hits > 10:
                if not car.pit_candidate:
                    car.pit_candidate = True
                    self.pit_events  += 1
            elif spd > 3.0:
                car.pit_candidate = False

# ─────────────────────────────────────────────────────────────────────────────
#  HEATMAP ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class TrackHeatmap:
    def __init__(self, w, h):
        self.w = w; self.h = h
        self.heat = np.zeros((h, w), dtype=np.float32)

    def accumulate(self, cars):
        for car in cars:
            cx,cy = bbox_centre(car.bbox)
            cx = int(np.clip(cx, 0, self.w-1))
            cy = int(np.clip(cy, 0, self.h-1))
            r = max(int(math.sqrt(bbox_area(car.bbox))*0.4), 6)
            cv2.circle(self._heat_u8_view(), (cx,cy), r, 1, -1)
        self.heat[self.heat > 0] += 0.0  # just accumulate via direct add

    def add_point(self, cx, cy, radius=12):
        cx = int(np.clip(cx, 0, self.w-1))
        cy = int(np.clip(cy, 0, self.h-1))
        y0=max(0,cy-radius); y1=min(self.h,cy+radius)
        x0=max(0,cx-radius); x1=min(self.w,cx+radius)
        self.heat[y0:y1, x0:x1] += 1.0

    def _heat_u8_view(self):
        return (np.clip(self.heat,0,255)).astype(np.uint8)

    def render(self):
        sm = gaussian_filter(self.heat, sigma=18)
        if sm.max() < 1e-6: return np.zeros((self.h,self.w,3),dtype=np.uint8)
        norm = (sm / sm.max() * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        alpha_mask = (norm.astype(np.float32)/255.0)[:,:,None]
        bg = np.zeros((self.h,self.w,3),dtype=np.uint8)
        result = (bg*(1-alpha_mask) + colored*alpha_mask).astype(np.uint8)
        return result

    def save_png(self, path, ref_frame=None):
        sm = gaussian_filter(self.heat, sigma=22)
        if sm.max() < 1e-6:
            blank = np.zeros((self.h,self.w,3),dtype=np.uint8)
            cv2.imwrite(path,blank); return
        norm = (sm/sm.max()*255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        if ref_frame is not None:
            gray_bg = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            gray_bg = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR)
            gray_bg = (gray_bg * 0.35).astype(np.uint8)
            alpha = (norm.astype(np.float32)/255.0*0.78)[:,:,None]
            out = (gray_bg*(1-alpha)+colored*alpha).astype(np.uint8)
        else:
            out = colored
        # Add title
        cv2.putText(out,"TRACK ACTIVITY HEATMAP",(12,34),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,C_NEON_Y,2,cv2.LINE_AA)
        cv2.putText(out,"tubakhxn | github.com/tubakhxn",(12,out.shape[0]-14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(80,120,80),1,cv2.LINE_AA)
        cv2.imwrite(path, out)

# ─────────────────────────────────────────────────────────────────────────────
#  DRAW — LEFT PANEL (Camera + overlays)
# ─────────────────────────────────────────────────────────────────────────────
def draw_racing_trail(img, car):
    pts = list(car.trail)[-40:]
    if len(pts) < 2: return
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = lerp_color((10,10,10), car.color_bgr, alpha)
        cv2.line(img, pts[i-1], pts[i], c, 2, cv2.LINE_AA)

def draw_car_box(img, car, sx, sy):
    x1,y1,x2,y2 = car.bbox
    bx1=int(x1*sx); by1=int(y1*sy); bx2=int(x2*sx); by2=int(y2*sy)

    color = car.color_bgr

    # Overtake flash — cycle between white and car color
    if car.overtake_flag > 0 and (car.overtake_flag//5)%2==0:
        color = C_WHITE

    # Main box
    cv2.rectangle(img,(bx1,by1),(bx2,by2),color,2,cv2.LINE_AA)

    # Corner ticks
    t=8
    for (cx,cy,sx2,sy2) in [(bx1,by1,1,1),(bx2,by1,-1,1),(bx1,by2,1,-1),(bx2,by2,-1,-1)]:
        cv2.line(img,(cx,cy),(cx+sx2*t,cy),C_WHITE,1,cv2.LINE_AA)
        cv2.line(img,(cx,cy),(cx,cy+sy2*t),C_WHITE,1,cv2.LINE_AA)

    # Speed indicator bar (bottom of box)
    spd = min(car.avg_speed()/15.0, 1.0)
    bar_w = int((bx2-bx1)*spd)
    spd_col = lerp_color(C_GREEN, C_RED, spd)
    cv2.rectangle(img,(bx1,by2+2),(bx1+bar_w,by2+5),spd_col,-1)

    # Label
    pit_txt = " PIT" if car.pit_candidate else ""
    ot_txt  = " ▲OT" if car.overtake_flag > 0 else ""
    label   = f"CAR#{car.id} P{car.position}{pit_txt}{ot_txt}"
    put_label(img, label, (bx1, by1-6), color, scale=0.42)

    # Sector indicator
    sector_cols = [C_GREEN, C_YELLOW, C_RED]
    scol = sector_cols[car.lap_sector % 3]
    cv2.circle(img,(bx2+5,(by1+by2)//2),4,scol,-1,cv2.LINE_AA)

    return (bx1,by1,bx2,by2)

def draw_speed_vector(img, car, sx, sy):
    pts = list(car.trail)[-4:]
    if len(pts) < 2: return
    p1 = (int(pts[0][0]*sx), int(pts[0][1]*sy))
    p2 = (int(pts[-1][0]*sx), int(pts[-1][1]*sy))
    dx=p2[0]-p1[0]; dy=p2[1]-p1[1]
    mag = math.hypot(dx,dy)
    if mag < 2: return
    scale = 3.0
    end = (int(p2[0]+dx*scale), int(p2[1]+dy*scale))
    spd = car.avg_speed()
    col = C_GREEN if spd<5 else (C_YELLOW if spd<10 else C_RED)
    cv2.arrowedLine(img, p2, end, col, 2, cv2.LINE_AA, tipLength=0.4)

def draw_overtake_flash(img, car, sx, sy):
    if car.overtake_flag <= 0: return
    x1,y1,x2,y2 = car.bbox
    cx=int((x1+x2)/2*sx); cy=int((y1+y2)/2*sy)
    radius = int(40*(car.overtake_flag/45.0))
    alpha  = car.overtake_flag/45.0
    overlay = img.copy()
    cv2.circle(overlay,(cx,cy),radius,C_YELLOW,-1,cv2.LINE_AA)
    cv2.addWeighted(overlay,alpha*0.25,img,1-alpha*0.25,0,img)
    cv2.circle(img,(cx,cy),radius,C_YELLOW,2,cv2.LINE_AA)
    put_label(img,"OVERTAKE",(cx-30,cy-radius-6),C_YELLOW,scale=0.52)

def draw_racing_line_predict(img, car, sx, sy):
    traj = car.predict_next(14)
    if len(traj) < 2: return
    for i in range(1,len(traj)):
        t = i/len(traj)
        c = lerp_color(car.color_bgr,(40,40,40),t)
        p1=(int(traj[i-1][0]*sx),int(traj[i-1][1]*sy))
        p2=(int(traj[i][0]*sx),int(traj[i][1]*sy))
        cv2.line(img,p1,p2,c,1,cv2.LINE_AA)

def draw_sector_zones(img, w, h):
    # Draw 3 translucent sector regions at bottom
    overlay = img.copy()
    s1=w//3; s2=2*w//3
    cv2.rectangle(overlay,(0,h-8),(s1,h),(0,200,0),-1)
    cv2.rectangle(overlay,(s1,h-8),(s2,h),(0,200,200),-1)
    cv2.rectangle(overlay,(s2,h-8),(w,h),(200,50,0),-1)
    cv2.addWeighted(overlay,0.6,img,0.4,0,img)
    for x,lbl in [(4,"S1"),(s1+4,"S2"),(s2+4,"S3")]:
        cv2.putText(img,lbl,(x,h-1),cv2.FONT_HERSHEY_SIMPLEX,0.38,(220,220,220),1,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
#  DRAW — RIGHT PANEL (Analytics dashboard via matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def build_analytics_panel(cars, tracker, frame_no, fps, panel_w, panel_h):
    dpi = 100
    fig = plt.figure(figsize=(panel_w/dpi, panel_h/dpi), dpi=dpi,
                     facecolor=(0.03,0.04,0.08))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.38,
                            left=0.08, right=0.97, top=0.93, bottom=0.06)

    BG   = (0.03,0.04,0.08)
    LINE = (0.1, 0.9, 0.45)

    active_cars  = len(cars)
    leader_id    = cars[0].id if cars else 0
    overtakes    = tracker.overtakes
    pit_events   = tracker.pit_events
    avg_speed    = float(np.mean([c.avg_speed() for c in cars])) if cars else 0.0
    intensity    = min(1.0, avg_speed/12.0 + active_cars*0.07)
    track_activity = min(1.0, active_cars/10.0)
    risk_score   = min(1.0, overtakes*0.08 + intensity*0.5)

    # ── Mini track map (scatter of trails) ───────────────────────────────────
    ax_map = fig.add_subplot(gs[0:2, 0:2])
    ax_map.set_facecolor((0.02,0.04,0.07))
    ax_map.set_title("TRACK OCCUPANCY MAP", color=LINE, fontsize=7, pad=3,
                     fontfamily='monospace')
    for car in cars:
        pts = car.trajectory_points(60)
        if len(pts) < 2: continue
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        c_rgb = car.color_rgb
        # trail gradient
        for i in range(1,len(xs)):
            alpha = i/len(xs)
            ax_map.plot([xs[i-1],xs[i]], [ys[i-1],ys[i]],
                        color=c_rgb, alpha=alpha*0.9, linewidth=1.2)
        ax_map.scatter([xs[-1]],[ys[-1]],color=c_rgb,s=45,zorder=5,
                       edgecolors='white',linewidths=0.5)
        ax_map.text(xs[-1]+4, ys[-1]-4, f"#{car.id}", color=c_rgb,
                    fontsize=5.5, fontfamily='monospace')

    ax_map.invert_yaxis()
    ax_map.axis('off')
    ax_map.set_xlim(0, 1280); ax_map.set_ylim(720, 0)

    # ── Speed bars ────────────────────────────────────────────────────────────
    ax_spd = fig.add_subplot(gs[2, 0])
    ax_spd.set_facecolor(BG)
    ax_spd.set_title("SPEED SCORES", color=LINE, fontsize=6.5, pad=2,
                     fontfamily='monospace')
    if cars:
        ids   = [f"#{c.id}" for c in cars[:8]]
        spds  = [c.avg_speed() for c in cars[:8]]
        max_s = max(spds) if spds else 1
        colors_bar = [car.color_rgb for car in cars[:8]]
        bars = ax_spd.barh(ids, spds, color=colors_bar, height=0.65)
        ax_spd.set_xlim(0, max_s*1.25)
        ax_spd.axvline(avg_speed, color=M_YELLOW, linewidth=0.8, linestyle='--', alpha=0.7)
        for bar, spd in zip(bars, spds):
            ax_spd.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                        f"{spd:.1f}", va='center', color=M_WHITE, fontsize=5.5)
    ax_spd.tick_params(colors=M_WHITE, labelsize=5.5)
    ax_spd.set_facecolor(BG)
    for spine in ax_spd.spines.values():
        spine.set_edgecolor((0.2,0.4,0.2))

    # ── Position changes ─────────────────────────────────────────────────────
    ax_pos = fig.add_subplot(gs[2, 1])
    ax_pos.set_facecolor(BG)
    ax_pos.set_title("POSITION DELTA", color=LINE, fontsize=6.5, pad=2,
                     fontfamily='monospace')
    if cars:
        ids2   = [f"#{c.id}" for c in cars[:8]]
        deltas = [c.prev_position - c.position for c in cars[:8]]
        bar_colors = [M_GREEN if d>0 else M_RED if d<0 else M_YELLOW for d in deltas]
        ax_pos.barh(ids2, deltas, color=bar_colors, height=0.65)
        ax_pos.axvline(0, color=M_WHITE, linewidth=0.6)
        ax_pos.tick_params(colors=M_WHITE, labelsize=5.5)
    for spine in ax_pos.spines.values():
        spine.set_edgecolor((0.2,0.4,0.2))

    # ── KPI gauges (text-based) ───────────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[3, 0:2])
    ax_kpi.set_facecolor(BG); ax_kpi.axis('off')

    kpis = [
        ("ACTIVE CARS",   f"{active_cars}",         M_CYAN),
        ("LEADER",        f"CAR#{leader_id}",        M_YELLOW),
        ("OVERTAKES",     f"{overtakes}",            M_GREEN),
        ("PIT EVENTS",    f"{pit_events}",           M_ORANGE),
        ("INTENSITY",     f"{intensity*100:.0f}%",   M_RED if intensity>0.7 else M_YELLOW),
        ("TRACK ACT.",    f"{track_activity*100:.0f}%", M_CYAN),
        ("FPS",           f"{fps:.1f}",              M_WHITE),
        ("FRAME",         f"{frame_no}",             (0.5,0.5,0.5)),
    ]
    cols = 4; rows = 2
    for i,(label,value,col) in enumerate(kpis):
        xi = (i%cols)/cols
        yi = 1.0 - (i//cols)/(rows) - 0.12
        ax_kpi.text(xi+0.01, yi+0.22, label, transform=ax_kpi.transAxes,
                    color=(0.5,0.7,0.5), fontsize=5.5, fontfamily='monospace')
        ax_kpi.text(xi+0.01, yi-0.02, value, transform=ax_kpi.transAxes,
                    color=col, fontsize=9.5, fontweight='bold', fontfamily='monospace')

    # ── Title bar ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.975, "F1 RACE INTELLIGENCE  |  tubakhxn",
             ha='center', color=(0.15,0.85,0.4), fontsize=7.5,
             fontfamily='monospace', transform=fig.transFigure)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1]+(4,))[:, :, :3]
    plt.close(fig)
    panel = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    panel = cv2.resize(panel,(panel_w,panel_h))
    return panel

# ─────────────────────────────────────────────────────────────────────────────
#  BOTTOM ANALYTICS BAR
# ─────────────────────────────────────────────────────────────────────────────
def draw_analytics_bar(img, cars, tracker, frame_no, fps):
    h,w = img.shape[:2]; bar_h=54
    bar = np.zeros((bar_h,w,3),dtype=np.uint8); bar[:] = (8,14,24)
    cv2.line(bar,(0,0),(w,0),(0,180,80),1)

    active     = len(cars)
    overtakes  = tracker.overtakes
    pit_events = tracker.pit_events
    avg_spd    = float(np.mean([c.avg_speed() for c in cars])) if cars else 0.0
    leader_id  = cars[0].id if cars else 0
    intensity  = min(100, int(avg_spd*6 + active*4))
    risk_col   = C_RED if intensity>70 else (C_YELLOW if intensity>40 else C_GREEN)

    stats = [
        (f"CARS: {active}",         C_NEON_C),
        (f"LEADER: #{leader_id}",   C_NEON_Y),
        (f"OVERTAKES: {overtakes}", C_GREEN),
        (f"PIT EVT: {pit_events}",  C_ORANGE),
        (f"INTENSITY: {intensity}%",risk_col),
        (f"AVG SPD: {avg_spd:.1f}", C_CYAN),
        (f"FPS: {fps:.1f}",         C_NEON_G),
        ("tubakhxn",                (50,80,50)),
    ]
    col_w = w // len(stats)
    for i,(txt,col) in enumerate(stats):
        cv2.putText(bar,txt,(i*col_w+8,36),
                    cv2.FONT_HERSHEY_SIMPLEX,0.46,col,1,cv2.LINE_AA)

    img[h-bar_h:h] = bar

# ─────────────────────────────────────────────────────────────────────────────
#  SAVE RACE DASHBOARD PNG
# ─────────────────────────────────────────────────────────────────────────────
def save_race_dashboard(frame, cars, tracker, frame_no, out_path):
    dpi = 130
    fig = plt.figure(figsize=(14,8), dpi=dpi, facecolor=(0.02,0.03,0.07))
    gs  = gridspec.GridSpec(3,3, figure=fig, hspace=0.5, wspace=0.35,
                            left=0.06,right=0.97,top=0.91,bottom=0.07)

    LINE = (0.1,0.9,0.45); BG=(0.02,0.03,0.07)

    # ── Frame snapshot ────────────────────────────────────────────────────────
    ax_frame = fig.add_subplot(gs[0:2,0:2])
    ax_frame.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    ax_frame.set_title("RACE FEED SNAPSHOT", color=LINE, fontsize=9, pad=3,
                       fontfamily='monospace')
    ax_frame.axis('off')

    # ── Speed history lines ───────────────────────────────────────────────────
    ax_spd = fig.add_subplot(gs[0,2])
    ax_spd.set_facecolor(BG)
    ax_spd.set_title("SPEED HISTORY", color=LINE, fontsize=7, pad=2, fontfamily='monospace')
    for car in cars[:6]:
        if len(car.speed_px)>1:
            ax_spd.plot(list(car.speed_px), color=car.color_rgb, lw=1.2, alpha=0.85,
                        label=f"#{car.id}")
    ax_spd.tick_params(colors=(0.6,0.8,0.6), labelsize=5)
    ax_spd.legend(fontsize=5, facecolor=BG, labelcolor=M_WHITE, framealpha=0.5)
    for sp in ax_spd.spines.values(): sp.set_edgecolor((0.15,0.4,0.2))

    # ── Position table ────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1,2])
    ax_tbl.set_facecolor(BG); ax_tbl.axis('off')
    ax_tbl.set_title("STANDINGS", color=LINE, fontsize=7, pad=2, fontfamily='monospace')
    for i,car in enumerate(cars[:8]):
        delta = car.prev_position - car.position
        sym   = "▲" if delta>0 else ("▼" if delta<0 else "–")
        col   = M_GREEN if delta>0 else (M_RED if delta<0 else M_WHITE)
        ax_tbl.text(0.05, 0.92-i*0.115, f"P{car.position}  CAR#{car.id}  {sym}{abs(delta)}",
                    transform=ax_tbl.transAxes, color=car.color_rgb,
                    fontsize=7, fontfamily='monospace')
    for sp in ax_tbl.spines.values(): sp.set_edgecolor((0.15,0.4,0.2))

    # ── KPI row ───────────────────────────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[2,0:3])
    ax_kpi.set_facecolor(BG); ax_kpi.axis('off')

    avg_spd = float(np.mean([c.avg_speed() for c in cars])) if cars else 0.0
    intensity = min(100,int(avg_spd*6+len(cars)*4))

    kpis = [
        ("ACTIVE CARS",  str(len(cars)),             M_CYAN),
        ("LEADER",       f"CAR#{cars[0].id if cars else 0}", M_YELLOW),
        ("OVERTAKES",    str(tracker.overtakes),     M_GREEN),
        ("PIT EVENTS",   str(tracker.pit_events),    M_ORANGE),
        ("AVG SPEED",    f"{avg_spd:.2f} px/f",      M_BLUE),
        ("INTENSITY",    f"{intensity}%",            M_RED if intensity>70 else M_YELLOW),
        ("FRAME",        str(frame_no),              (0.5,0.5,0.5)),
        ("STATUS",       "LIVE ●",                   M_GREEN),
    ]
    for i,(lbl,val,col) in enumerate(kpis):
        xi = i/len(kpis)
        ax_kpi.text(xi+0.005,0.85,lbl,transform=ax_kpi.transAxes,
                    color=(0.4,0.7,0.4),fontsize=6.5,fontfamily='monospace')
        ax_kpi.text(xi+0.005,0.28,val,transform=ax_kpi.transAxes,
                    color=col,fontsize=11,fontweight='bold',fontfamily='monospace')

    fig.suptitle("FORMULA 1 RACE INTELLIGENCE SYSTEM  |  tubakhxn / github.com/tubakhxn",
                 color=(0.1,0.85,0.4), fontsize=10, fontfamily='monospace', y=0.97)

    fig.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OUT]  Wrote {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def process_video(input_path):
    print(BANNER)
    print(f"[INIT] Input  : {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}"); sys.exit(1)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps= cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INIT] Video  : {src_w}x{src_h} @ {src_fps:.1f}fps  |  {total} frames")

    # Output layout
    LEFT_W=640; LEFT_H=400; RIGHT_W=640; RIGHT_H=400
    OUT_W=LEFT_W+RIGHT_W; OUT_H=LEFT_H+54

    print(f"[INIT] Output : {OUT_W}x{OUT_H}  output.mp4")

    # YOLO
    print("[YOLO] Loading YOLOv8n …")
    yolo = YOLO("yolov8n.pt")
    print("[YOLO] Ready.\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output.mp4", fourcc, src_fps,(OUT_W,OUT_H))

    tracker  = RaceTracker()
    heatmap  = TrackHeatmap(src_w, src_h)

    frame_no = 0; t_prev=time.time(); fps_d=src_fps
    dashboard_frame = None; dash_cars = []; snap_frame = None

    ANALYTICS_EVERY = 3   # rebuild matplotlib panel every N frames
    cached_panel    = None

    print(f"[RUN]  Processing {total} frames …\n")
    pbar = tqdm(total=total, unit="frame",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1; pbar.update(1)

        # ── YOLO ──────────────────────────────────────────────────────────────
        results = yolo(frame, verbose=False, conf=0.28,
                       classes=list(VEHICLE_CLASSES))
        detections = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) not in VEHICLE_CLASSES: continue
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                detections.append(((x1,y1,x2,y2), conf))

        # ── Tracker ───────────────────────────────────────────────────────────
        now=time.time(); fps_d=0.9*fps_d+0.1/(now-t_prev+1e-6); t_prev=now
        active_cars = tracker.update(detections, frame_no)

        # Assign lap sectors by horizontal position
        for car in active_cars:
            cx = bbox_centre(car.bbox)[0]
            car.lap_sector = 0 if cx<src_w//3 else (1 if cx<2*src_w//3 else 2)

        # Accumulate heatmap
        for car in active_cars:
            cx,cy=bbox_centre(car.bbox)
            r=max(int(math.sqrt(bbox_area(car.bbox))*0.35),8)
            heatmap.add_point(cx,cy,r)

        # ── LEFT PANEL ────────────────────────────────────────────────────────
        left = cv2.resize(frame,(LEFT_W,LEFT_H))
        sx=LEFT_W/src_w; sy=LEFT_H/src_h

        # Heatmap blend
        heat_vis = cv2.resize(heatmap.render(),(LEFT_W,LEFT_H))
        cv2.addWeighted(left,0.78,heat_vis,0.22,0,left)

        # Sector zones
        draw_sector_zones(left,LEFT_W,LEFT_H)

        # Trails + predictions
        for car in active_cars:
            # Scale trail
            scaled_car = type('C',(),{})()
            scaled_car.trail = deque(
                [(int(x*sx),int(y*sy)) for x,y in car.trail], maxlen=80)
            scaled_car.color_bgr=car.color_bgr
            draw_racing_trail(left, scaled_car)
            draw_racing_line_predict(left, car, sx, sy)

        # Boxes + overlays
        for car in active_cars:
            draw_car_box(left, car, sx, sy)
            draw_overtake_flash(left, car, sx, sy)
            draw_speed_vector(left, car, sx, sy)

        put_label(left,"RACE FEED",(8,20),C_NEON_C,scale=0.55)
        draw_corner_brackets(left,C_NEON_G)
        draw_scanlines(left)

        # ── RIGHT PANEL ───────────────────────────────────────────────────────
        if frame_no % ANALYTICS_EVERY == 1 or cached_panel is None:
            cached_panel = build_analytics_panel(
                active_cars, tracker, frame_no, fps_d, RIGHT_W, RIGHT_H)
        right = cached_panel.copy()
        draw_corner_brackets(right, C_NEON_C)

        # ── COMPOSITE ─────────────────────────────────────────────────────────
        composite = np.zeros((OUT_H,OUT_W,3),dtype=np.uint8)
        composite[:LEFT_H,:LEFT_W]               = left
        composite[:RIGHT_H,LEFT_W:LEFT_W+RIGHT_W] = right
        draw_analytics_bar(composite, active_cars, tracker, frame_no, fps_d)

        writer.write(composite)

        # Save dashboard snapshot
        if frame_no == min(90, total) or frame_no == total:
            snap_frame = frame.copy()
            dash_cars  = list(active_cars)

    pbar.close(); cap.release(); writer.release()
    print(f"\n[DONE] Wrote output.mp4")

    # ── race_dashboard.png ────────────────────────────────────────────────────
    if snap_frame is not None:
        save_race_dashboard(snap_frame, dash_cars, tracker,
                            frame_no, "race_dashboard.png")

    # ── race_heatmap.png ──────────────────────────────────────────────────────
    ref = cv2.resize(snap_frame,(src_w,src_h)) if snap_frame is not None else None
    heatmap.save_png("race_heatmap.png", ref)
    print("[DONE] Wrote race_heatmap.png")

    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║  All outputs saved.  tubakhxn / github.com/tubakhxn  ║")
    print("╚═══════════════════════════════════════════════════════╝\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_video.mp4>")
        print("Example: python main.py race_footage.mp4")
        sys.exit(1)

    video = sys.argv[1]
    if not os.path.isfile(video):
        print(f"[ERROR] File not found: {video}"); sys.exit(1)

    try:
        process_video(video)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1).p