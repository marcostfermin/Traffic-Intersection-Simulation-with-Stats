import os
import sys
import math
import json
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import pygame

# =============================================================================
#  LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("simulation.log", mode="w", encoding="utf-8")]
)

# =============================================================================
#  MATH / GEOMETRY
# =============================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def vec_add(a: Tuple[float,float], b: Tuple[float,float]) -> Tuple[float,float]:
    return (a[0]+b[0], a[1]+b[1])

def vec_sub(a: Tuple[float,float], b: Tuple[float,float]) -> Tuple[float,float]:
    return (a[0]-b[0], a[1]-b[1])

def vec_mul(a: Tuple[float,float], s: float) -> Tuple[float,float]:
    return (a[0]*s, a[1]*s)

def vec_len(a: Tuple[float,float]) -> float:
    return math.hypot(a[0], a[1])

def vec_norm(a: Tuple[float,float]) -> Tuple[float,float]:
    L = vec_len(a)
    if L <= 1e-9:
        return (0.0, 0.0)
    return (a[0]/L, a[1]/L)

def angle_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(-dy, dx))

def rect_from_center(cx: float, cy: float, w: float, h: float) -> pygame.Rect:
    r = pygame.Rect(0,0,int(w),int(h))
    r.center = (int(cx), int(cy))
    return r

# =============================================================================
#  SPLINE PATHS (Bezier)
# =============================================================================

class BezierCubic:
    def __init__(self, p0, p1, p2, p3):
        self.p0 = (float(p0[0]), float(p0[1]))
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.p3 = (float(p3[0]), float(p3[1]))
        self._arc_table = []
        self._arc_len = 0.0
        self._build_arc_table()

    def point(self, t: float) -> Tuple[float,float]:
        t = clamp(t, 0.0, 1.0)
        u = 1.0 - t
        p = vec_mul(self.p0, u*u*u)
        p = vec_add(p, vec_mul(self.p1, 3*u*u*t))
        p = vec_add(p, vec_mul(self.p2, 3*u*t*t))
        p = vec_add(p, vec_mul(self.p3, t*t*t))
        return p

    def tangent(self, t: float) -> Tuple[float,float]:
        t = clamp(t, 0.0, 1.0)
        u = 1.0 - t
        a = vec_mul(vec_sub(self.p1, self.p0), 3*u*u)
        b = vec_mul(vec_sub(self.p2, self.p1), 6*u*t)
        c = vec_mul(vec_sub(self.p3, self.p2), 3*t*t)
        return vec_add(vec_add(a,b),c)

    def _build_arc_table(self, steps: int = 220):
        self._arc_table = [(0.0, 0.0)]  # (t, cumulative_length)
        prev = self.point(0.0)
        cum = 0.0
        for i in range(1, steps+1):
            t = i/steps
            cur = self.point(t)
            cum += vec_len(vec_sub(cur, prev))
            self._arc_table.append((t, cum))
            prev = cur
        self._arc_len = cum

    @property
    def length(self) -> float:
        return self._arc_len

    def t_at_s(self, s: float) -> float:
        if self._arc_len <= 1e-9:
            return 1.0
        s = clamp(s, 0.0, self._arc_len)
        lo = 0
        hi = len(self._arc_table)-1
        while lo < hi:
            mid = (lo + hi)//2
            if self._arc_table[mid][1] < s:
                lo = mid + 1
            else:
                hi = mid
        idx = lo
        if idx <= 0:
            return self._arc_table[0][0]
        t1, s1 = self._arc_table[idx-1]
        t2, s2 = self._arc_table[idx]
        if abs(s2 - s1) <= 1e-9:
            return t2
        f = (s - s1) / (s2 - s1)
        return lerp(t1, t2, f)

class PathSegment:
    def __init__(self, bezier: BezierCubic, meta: Optional[dict] = None):
        self.curve = bezier
        self.meta = meta or {}

class RoutePath:
    def __init__(self, segments: List[PathSegment]):
        self.segments = segments[:]
        self.seg_lengths = [seg.curve.length for seg in self.segments]
        self.total_length = sum(self.seg_lengths)

    def sample(self, s: float) -> Tuple[Tuple[float,float], float, int]:
        if self.total_length <= 1e-9:
            p = self.segments[-1].curve.point(1.0)
            tan = self.segments[-1].curve.tangent(1.0)
            ang = angle_deg(tan[0], tan[1])
            return p, ang, len(self.segments)-1

        s = clamp(s, 0.0, self.total_length)
        acc = 0.0
        for i, L in enumerate(self.seg_lengths):
            if s <= acc + L:
                local = s - acc
                t = self.segments[i].curve.t_at_s(local)
                p = self.segments[i].curve.point(t)
                tan = self.segments[i].curve.tangent(t)
                ang = angle_deg(tan[0], tan[1])
                return p, ang, i
            acc += L
        i = len(self.segments)-1
        p = self.segments[i].curve.point(1.0)
        tan = self.segments[i].curve.tangent(1.0)
        ang = angle_deg(tan[0], tan[1])
        return p, ang, i

# =============================================================================
#  CONFIG
# =============================================================================

@dataclass
class Config:
    screen_w: int = 1600
    screen_h: int = 900
    fps: int = 60
    sim_seconds: int = 300

    # Traffic demand (Poisson rates vehicles/sec) by entry edge
    lambda_left: float = 0.20
    lambda_right: float = 0.20
    lambda_top: float = 0.18
    lambda_bottom: float = 0.18

    # Time-of-day shaping (sinusoidal multiplier)
    demand_wave_period: float = 120.0
    demand_wave_amp: float = 0.45

    # Vehicle dynamics (IDM-ish)
    v0_mean: float = 18.0
    v0_std: float = 3.0
    a_max: float = 2.0
    b_comf: float = 2.5
    T_headway: float = 1.2
    s0: float = 2.0
    delta: float = 4.0

    # Lane change (MOBIL-lite)
    lane_change_check_dist: float = 35.0
    lane_change_min_gain: float = 0.12
    lane_change_politeness: float = 0.25
    lane_change_cooldown: float = 2.0

    # Collision robustness
    max_dt: float = 0.05
    broad_cell: int = 140
    resolve_iters: int = 14
    min_clearance: float = 1.0

    # Intersection control
    min_green: float = 8.0
    max_green: float = 25.0
    yellow: float = 3.0
    all_red: float = 1.0
    ped_walk: float = 6.0
    ped_flash: float = 4.0

    # Sensors
    detector_noise_std: float = 0.25
    detector_drop_prob: float = 0.03

    # Conflict-zone reservation
    conflict_shrink: int = 46                 # shrink intersection box inward to define conflict zone
    reservation_timeout: float = 6.0          # safety release if token gets stuck

    # Spillback blocking
    spillback_lookahead: int = 120            # region after conflict zone checked for occupancy
    spillback_min_speed: float = 10.0         # treat slow vehicles as blocking
    spillback_margin: int = 8                 # extra padding to prevent box blocking

    # Drawing toggles
    draw_debug: bool = True
    draw_paths: bool = False
    draw_detectors: bool = True
    draw_bboxes: bool = False

    # Multi-intersection network (2 in a row)
    intersection_count: int = 2
    intersection_spacing: int = 520
    road_y: int = 450
    road_lanes: int = 2
    lane_width: int = 28
    approach_len: int = 280
    junction_box: int = 180

    # Files
    export_vehicle_csv: str = "vehicles.csv"
    export_summary_csv: str = "summary.csv"
    config_json: str = "config.json"

def load_config(path: str, base: Config) -> Config:
    if not os.path.exists(path):
        return base
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(base.__dict__)
        merged.update(data)
        return Config(**merged)
    except Exception as e:
        logging.error("Failed to load %s: %s", path, str(e))
        return base

# =============================================================================
#  METRICS
# =============================================================================

@dataclass
class VehicleRecord:
    vid: int
    spawn_t: float
    exit_t: float
    entry: str
    exit_edge: str
    route: str
    stops: int
    lane_changes: int
    delay: float

class Metrics:
    def __init__(self):
        self.vehicle_records: List[VehicleRecord] = []
        self.spawned = 0
        self.exited = 0
        self.collision_avoids = 0
        self.hard_clamps = 0
        self.soft_clamps = 0
        self.queue_samples: List[Tuple[float,int,int,int,int]] = []

    def record_vehicle_exit(self, rec: VehicleRecord):
        self.vehicle_records.append(rec)
        self.exited += 1

    def export(self, cfg: Config):
        try:
            with open(cfg.export_vehicle_csv, "w", encoding="utf-8") as f:
                f.write("vid,spawn_t,exit_t,entry,exit_edge,route,stops,lane_changes,delay\n")
                for r in self.vehicle_records:
                    f.write(f"{r.vid},{r.spawn_t:.3f},{r.exit_t:.3f},{r.entry},{r.exit_edge},{r.route},{r.stops},{r.lane_changes},{r.delay:.3f}\n")
            logging.info("Wrote %s", cfg.export_vehicle_csv)
        except Exception as e:
            logging.error("Failed to write vehicle CSV: %s", str(e))

        try:
            delays = [r.delay for r in self.vehicle_records]
            stops = [r.stops for r in self.vehicle_records]
            lcs = [r.lane_changes for r in self.vehicle_records]
            p50 = 0.0
            p90 = 0.0
            if delays:
                ds = sorted(delays)
                p50 = ds[int(0.50*(len(ds)-1))]
                p90 = ds[int(0.90*(len(ds)-1))]
            avg_delay = (sum(delays)/len(delays)) if delays else 0.0
            avg_stops = (sum(stops)/len(stops)) if stops else 0.0
            avg_lc = (sum(lcs)/len(lcs)) if lcs else 0.0

            with open(cfg.export_summary_csv, "w", encoding="utf-8") as f:
                f.write("metric,value\n")
                f.write(f"spawned,{self.spawned}\n")
                f.write(f"exited,{self.exited}\n")
                f.write(f"avg_delay,{avg_delay:.3f}\n")
                f.write(f"p50_delay,{p50:.3f}\n")
                f.write(f"p90_delay,{p90:.3f}\n")
                f.write(f"avg_stops,{avg_stops:.3f}\n")
                f.write(f"avg_lane_changes,{avg_lc:.3f}\n")
                f.write(f"collision_avoids,{self.collision_avoids}\n")
                f.write(f"hard_clamps,{self.hard_clamps}\n")
                f.write(f"soft_clamps,{self.soft_clamps}\n")
            logging.info("Wrote %s", cfg.export_summary_csv)
        except Exception as e:
            logging.error("Failed to write summary CSV: %s", str(e))

# =============================================================================
#  SIGNAL PHASES (including protected turns + pedestrian)
# =============================================================================

@dataclass
class Phase:
    name: str
    greens: Dict[str, Dict[str, bool]]  # approach -> movement -> allowed (movements: "S","L","R")
    ped_walk: Dict[str, bool]          # crosswalk id -> walk allowed
    min_t: float
    max_t: float

class Detector:
    def __init__(self, rect: pygame.Rect, name: str):
        self.rect = rect
        self.name = name

    def measure(self, vehicles: List["Vehicle"], noise_std: float, drop_prob: float) -> float:
        if random.random() < drop_prob:
            return float("nan")
        count = 0
        for v in vehicles:
            if self.rect.colliderect(v.aabb()):
                count += 1
        noisy = count + random.gauss(0.0, noise_std)
        return max(0.0, noisy)

class SignalController:
    def __init__(self, cfg: Config, phases: List[Phase]):
        self.cfg = cfg
        self.phases = phases
        self.phase_idx = 0
        self.state = "GREEN"
        self.timer = 0.0
        self.green_elapsed = 0.0

        self.yellow_t = cfg.yellow
        self.all_red_t = cfg.all_red
        self.ped_walk_t = cfg.ped_walk
        self.ped_flash_t = cfg.ped_flash

        self.active_phase = self.phases[self.phase_idx]
        self.timer = self.active_phase.min_t
        self.green_elapsed = 0.0

    def current(self) -> Phase:
        return self.active_phase

    def _next_phase(self):
        self.phase_idx = (self.phase_idx + 1) % len(self.phases)
        self.active_phase = self.phases[self.phase_idx]
        self.state = "GREEN"
        self.timer = self.active_phase.min_t
        self.green_elapsed = 0.0

    def tick(self, dt: float, pressure_scores: Optional[List[float]] = None):
        if dt <= 0:
            return

        if self.state == "GREEN":
            self.timer -= dt
            self.green_elapsed += dt

            if pressure_scores is not None and self.green_elapsed >= self.active_phase.min_t:
                if self.timer <= 0.25 and self.green_elapsed < self.active_phase.max_t:
                    cur_p = pressure_scores[self.phase_idx]
                    best_p = max(pressure_scores) if pressure_scores else cur_p
                    if best_p > 1e-9:
                        if cur_p >= 0.85 * best_p:
                            extend = min(1.5, self.active_phase.max_t - self.green_elapsed)
                            if extend > 0:
                                self.timer += extend
                                return

            if self.timer <= 0.0 or self.green_elapsed >= self.active_phase.max_t:
                self.state = "YELLOW"
                self.timer = self.yellow_t

        elif self.state == "YELLOW":
            self.timer -= dt
            if self.timer <= 0.0:
                self.state = "ALL_RED"
                self.timer = self.all_red_t

        elif self.state == "ALL_RED":
            self.timer -= dt
            if self.timer <= 0.0:
                self.state = "PED_WALK"
                self.timer = self.ped_walk_t

        elif self.state == "PED_WALK":
            self.timer -= dt
            if self.timer <= 0.0:
                self.state = "PED_FLASH"
                self.timer = self.ped_flash_t

        elif self.state == "PED_FLASH":
            self.timer -= dt
            if self.timer <= 0.0:
                self._next_phase()

    def allow(self, approach: str, movement: str) -> bool:
        if self.state != "GREEN":
            return False
        ph = self.active_phase
        if approach not in ph.greens:
            return False
        return bool(ph.greens[approach].get(movement, False))

    def ped_allow(self, cross_id: str) -> bool:
        if self.state == "PED_WALK":
            return bool(self.active_phase.ped_walk.get(cross_id, False))
        return False

# =============================================================================
#  CONFLICT ZONE RESERVATION (per intersection)
# =============================================================================

class ReservationManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.owner: Dict[int, Optional[int]] = {}
        self.last_touch: Dict[int, float] = {}

    def init_intersection(self, iid: int):
        self.owner[iid] = None
        self.last_touch[iid] = -1e9

    def get_owner(self, iid: int) -> Optional[int]:
        return self.owner.get(iid, None)

    def release_if_stale(self, iid: int, now_t: float):
        own = self.owner.get(iid, None)
        if own is None:
            return
        if (now_t - self.last_touch.get(iid, now_t)) > self.cfg.reservation_timeout:
            self.owner[iid] = None

    def touch(self, iid: int, now_t: float):
        self.last_touch[iid] = now_t

    def release(self, iid: int):
        self.owner[iid] = None

    def try_acquire(self, iid: int, vid: int, now_t: float) -> bool:
        self.release_if_stale(iid, now_t)
        own = self.owner.get(iid, None)
        if own is None or own == vid:
            self.owner[iid] = vid
            self.touch(iid, now_t)
            return True
        return False

# =============================================================================
#  NETWORK: INTERSECTIONS + ROADS + ROUTING + LEFT-TURN LANES
# =============================================================================

@dataclass
class Lane:
    lane_id: str
    centerline: RoutePath
    width: float
    approach: str
    movement: str          # "S","L","R"
    intersection_id: int
    kind: str              # "approach" | "connector" | "exit" | "turn"

@dataclass
class Intersection:
    iid: int
    cx: int
    cy: int
    box: pygame.Rect
    conflict: pygame.Rect
    detectors: Dict[str, Detector] = field(default_factory=dict)
    signal: Optional[SignalController] = None

class Network:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.intersections: List[Intersection] = []
        self.lanes: Dict[str, Lane] = {}
        self.entry_lanes: Dict[str, List[str]] = {}
        self.exit_edges: Dict[str, pygame.Rect] = {}
        self._build()

    def _build(self):
        cfg = self.cfg
        base_x = 520
        y = cfg.road_y
        for i in range(cfg.intersection_count):
            cx = base_x + i * cfg.intersection_spacing
            cy = y
            box = pygame.Rect(0,0,cfg.junction_box,cfg.junction_box)
            box.center = (cx, cy)
            conflict = box.inflate(-cfg.conflict_shrink, -cfg.conflict_shrink)
            inter = Intersection(iid=i, cx=cx, cy=cy, box=box, conflict=conflict)
            self.intersections.append(inter)

        self.exit_edges = {
            "LEFT": pygame.Rect(-80, 0, 80, cfg.screen_h),
            "RIGHT": pygame.Rect(cfg.screen_w, 0, 80, cfg.screen_h),
            "TOP": pygame.Rect(0, -80, cfg.screen_w, 80),
            "BOTTOM": pygame.Rect(0, cfg.screen_h, cfg.screen_w, 80),
        }

        self.entry_lanes = {"LEFT":[], "RIGHT":[], "TOP":[], "BOTTOM":[]}

        lw = cfg.lane_width
        box = cfg.junction_box

        def make_straight(p0, p3):
            p0 = (float(p0[0]), float(p0[1]))
            p3 = (float(p3[0]), float(p3[1]))
            v = vec_sub(p3, p0)
            p1 = vec_add(p0, vec_mul(v, 1/3))
            p2 = vec_add(p0, vec_mul(v, 2/3))
            return BezierCubic(p0,p1,p2,p3)

        def make_turn(p0, p3, bend: Tuple[float,float]):
            p0 = (float(p0[0]), float(p0[1]))
            p3 = (float(p3[0]), float(p3[1]))
            b = (float(bend[0]), float(bend[1]))
            p1 = (lerp(p0[0], b[0], 0.75), lerp(p0[1], b[1], 0.75))
            p2 = (lerp(p3[0], b[0], 0.75), lerp(p3[1], b[1], 0.75))
            return BezierCubic(p0,p1,p2,p3)

        # Corridor lanes (eastbound from LEFT, westbound from RIGHT)
        corridor_y0 = cfg.road_y - lw/2
        corridor_y1 = cfg.road_y + lw/2

        for li, cy_lane in enumerate([corridor_y0, corridor_y1]):
            lid = f"EW_E_L{li}"
            p0 = (0 - 60, cy_lane)
            p3 = (cfg.screen_w + 60, cy_lane)
            seg = PathSegment(make_straight(p0,p3), meta={"type":"corridor"})
            rp = RoutePath([seg])
            self.lanes[lid] = Lane(
                lane_id=lid, centerline=rp, width=lw,
                approach="EWCORR", movement="S", intersection_id=-1, kind="connector"
            )
            self.entry_lanes["LEFT"].append(lid)

        for li, cy_lane in enumerate([corridor_y0+lw*2, corridor_y1+lw*2]):
            lid = f"EW_W_L{li}"
            p0 = (cfg.screen_w + 60, cy_lane)
            p3 = (0 - 60, cy_lane)
            seg = PathSegment(make_straight(p0,p3), meta={"type":"corridor"})
            rp = RoutePath([seg])
            self.lanes[lid] = Lane(
                lane_id=lid, centerline=rp, width=lw,
                approach="WWCORR", movement="S", intersection_id=-1, kind="connector"
            )
            self.entry_lanes["RIGHT"].append(lid)

        # NS approach lanes: straight and left-turn at each intersection
        for inter in self.intersections:
            cx, cy = inter.cx, inter.cy

            # TOP -> BOTTOM (southbound) approach: N{iid}
            # Straight lane
            lid = f"N{inter.iid}_S_S"
            p0 = (cx - lw, 0 - 60)
            p_enter = (cx - lw, cy - box/2 - 12)
            p_exit = (cx - lw, cfg.screen_h + 60)
            seg1 = PathSegment(make_straight(p0, p_enter), meta={"type":"approach"})
            seg2 = PathSegment(make_straight(p_enter, p_exit), meta={"type":"exit"})
            self.lanes[lid] = Lane(lane_id=lid, centerline=RoutePath([seg1,seg2]), width=lw, approach=f"NS{inter.iid}", movement="S", intersection_id=inter.iid, kind="approach")
            self.entry_lanes["TOP"].append(lid)

            # Left-turn lane from TOP: turn into eastbound corridor (to the RIGHT)
            lid = f"N{inter.iid}_S_L"
            p0 = (cx - lw*2.2, 0 - 60)
            p_enter = (cx - lw*2.2, cy - box/2 - 12)
            p_turn_end = (cx + box/2 + 60, corridor_y0)  # merge into upper eastbound lane
            bend = (cx - box/2, cy - box/2)
            segA = PathSegment(make_straight(p0, p_enter), meta={"type":"approach"})
            segB = PathSegment(make_turn(p_enter, p_turn_end, bend), meta={"type":"turn"})
            self.lanes[lid] = Lane(lane_id=lid, centerline=RoutePath([segA,segB]), width=lw, approach=f"NS{inter.iid}", movement="L", intersection_id=inter.iid, kind="turn")
            self.entry_lanes["TOP"].append(lid)

            # BOTTOM -> TOP (northbound) approach: S{iid}
            # Straight lane
            lid = f"S{inter.iid}_N_S"
            p0 = (cx + lw, cfg.screen_h + 60)
            p_enter = (cx + lw, cy + box/2 + 12)
            p_exit = (cx + lw, 0 - 60)
            seg1 = PathSegment(make_straight(p0, p_enter), meta={"type":"approach"})
            seg2 = PathSegment(make_straight(p_enter, p_exit), meta={"type":"exit"})
            self.lanes[lid] = Lane(lane_id=lid, centerline=RoutePath([seg1,seg2]), width=lw, approach=f"NS{inter.iid}", movement="S", intersection_id=inter.iid, kind="approach")
            self.entry_lanes["BOTTOM"].append(lid)

            # Left-turn lane from BOTTOM: turn into westbound corridor (to the LEFT)
            lid = f"S{inter.iid}_N_L"
            p0 = (cx + lw*2.2, cfg.screen_h + 60)
            p_enter = (cx + lw*2.2, cy + box/2 + 12)
            p_turn_end = (cx - box/2 - 60, corridor_y0 + lw*2)  # merge into upper westbound lane
            bend = (cx + box/2, cy + box/2)
            segA = PathSegment(make_straight(p0, p_enter), meta={"type":"approach"})
            segB = PathSegment(make_turn(p_enter, p_turn_end, bend), meta={"type":"turn"})
            self.lanes[lid] = Lane(lane_id=lid, centerline=RoutePath([segA,segB]), width=lw, approach=f"NS{inter.iid}", movement="L", intersection_id=inter.iid, kind="turn")
            self.entry_lanes["BOTTOM"].append(lid)

        # EW left-turn lanes (explicit queueing): approach segments from corridor edges that turn to TOP/BOTTOM exits
        # These lanes are spawned from LEFT/RIGHT edges as their own lanes (so they can queue separately).
        for inter in self.intersections:
            cx, cy = inter.cx, inter.cy

            # Eastbound-left: from LEFT edge to this intersection, then turn UP (TOP exit)
            lid = f"EW_E_I{inter.iid}_L"
            p0 = (0 - 60, corridor_y1)  # lower eastbound lane start (more realistic left-turn pocket)
            p_enter = (cx - box/2 - 24, corridor_y1)
            p_turn_end = (cx + lw*0.2, 0 - 60)
            bend = (cx - box/2, cy - box/2)
            segA = PathSegment(make_straight(p0, p_enter), meta={"type":"approach"})
            segB = PathSegment(make_turn(p_enter, p_turn_end, bend), meta={"type":"turn"})
            self.lanes[lid] = Lane(lane_id=lid, centerline=RoutePath([segA,segB]), width=lw, approach=f"EW{inter.iid}", movement="L", intersection_id=inter.iid, kind="turn")
            self.entry_lanes["LEFT"].append(lid)

            # Westbound-left: from RIGHT edge to this intersection, then turn DOWN (BOTTOM exit)
            lid = f"EW_W_I{inter.iid}_L"
            p0 = (cfg.screen_w + 60, corridor_y1 + lw*2)  # lower westbound lane start
            p_enter = (cx + box/2 + 24, corridor_y1 + lw*2)
            p_turn_end = (cx - lw*0.2, cfg.screen_h + 60)
            bend = (cx + box/2, cy + box/2)
            segA = PathSegment(make_straight(p0, p_enter), meta={"type":"approach"})
            segB = PathSegment(make_turn(p_enter, p_turn_end, bend), meta={"type":"turn"})
            self.lanes[lid] = Lane(lane_id=lid, centerline=RoutePath([segA,segB]), width=lw, approach=f"EW{inter.iid}", movement="L", intersection_id=inter.iid, kind="turn")
            self.entry_lanes["RIGHT"].append(lid)

        # Detectors + signal plans per intersection
        for inter in self.intersections:
            inter.detectors = self._make_detectors(inter)
            inter.signal = self._make_signal_plan(inter)

    def _make_detectors(self, inter: Intersection) -> Dict[str, Detector]:
        cfg = self.cfg
        box = inter.box
        dets = {}

        w = cfg.junction_box
        d = 42

        # Straight demand detectors (EW + NS)
        dets["WB_S"] = Detector(pygame.Rect(box.right+10, box.top+20, d, w-40), "WB_S")
        dets["EB_S"] = Detector(pygame.Rect(box.left-10-d, box.top+20, d, w-40), "EB_S")
        dets["NB_S"] = Detector(pygame.Rect(box.left+20, box.bottom+10, w-40, d), "NB_S")
        dets["SB_S"] = Detector(pygame.Rect(box.left+20, box.top-10-d, w-40, d), "SB_S")

        # Left-turn queue detectors (more localized near box corners / pockets)
        dets["EW_L"] = Detector(pygame.Rect(box.left-120, box.centery-18, 120, 36), "EW_L")
        dets["NS_L"] = Detector(pygame.Rect(box.centerx-18, box.bottom+10, 36, 120), "NS_L")

        return dets

    def _make_signal_plan(self, inter: Intersection) -> SignalController:
        cfg = self.cfg

        phases = []

        phases.append(Phase(
            name=f"I{inter.iid}_EW_STRAIGHT",
            greens={
                f"EW{inter.iid}": {"S": True, "L": False, "R": True},
                f"NS{inter.iid}": {"S": False, "L": False, "R": False},
            },
            ped_walk={f"PED_NS{inter.iid}": False, f"PED_EW{inter.iid}": True},
            min_t=cfg.min_green,
            max_t=cfg.max_green
        ))

        phases.append(Phase(
            name=f"I{inter.iid}_NS_STRAIGHT",
            greens={
                f"EW{inter.iid}": {"S": False, "L": False, "R": False},
                f"NS{inter.iid}": {"S": True, "L": False, "R": True},
            },
            ped_walk={f"PED_NS{inter.iid}": True, f"PED_EW{inter.iid}": False},
            min_t=cfg.min_green,
            max_t=cfg.max_green
        ))

        phases.append(Phase(
            name=f"I{inter.iid}_EW_LEFT_PROT",
            greens={
                f"EW{inter.iid}": {"S": False, "L": True, "R": False},
                f"NS{inter.iid}": {"S": False, "L": False, "R": False},
            },
            ped_walk={f"PED_NS{inter.iid}": False, f"PED_EW{inter.iid}": True},
            min_t=max(6.0, cfg.min_green*0.75),
            max_t=max(12.0, cfg.max_green*0.7)
        ))

        phases.append(Phase(
            name=f"I{inter.iid}_NS_LEFT_PROT",
            greens={
                f"EW{inter.iid}": {"S": False, "L": False, "R": False},
                f"NS{inter.iid}": {"S": False, "L": True, "R": False},
            },
            ped_walk={f"PED_NS{inter.iid}": True, f"PED_EW{inter.iid}": False},
            min_t=max(6.0, cfg.min_green*0.75),
            max_t=max(12.0, cfg.max_green*0.7)
        ))

        return SignalController(cfg, phases)

# =============================================================================
#  VEHICLES
# =============================================================================

class Vehicle:
    _VID = 1

    def __init__(self, cfg: Config, net: Network, metrics: Metrics, lane_id: str, entry_edge: str, spawn_t: float):
        self.cfg = cfg
        self.net = net
        self.metrics = metrics

        self.vid = Vehicle._VID
        Vehicle._VID += 1

        self.entry_edge = entry_edge
        self.exit_edge = ""
        self.route_name = ""

        self.lane_id = lane_id
        self.s = 0.0
        self.v = max(0.0, random.gauss(cfg.v0_mean, cfg.v0_std))
        self.a = 0.0

        self.v_des = max(6.0, random.gauss(cfg.v0_mean, cfg.v0_std)) * 8.0
        self.a_max = cfg.a_max * 80.0
        self.b_comf = cfg.b_comf * 80.0
        self.T = cfg.T_headway
        self.s0 = cfg.s0 * 10.0
        self.delta = cfg.delta

        self.length = 34.0
        self.width = 18.0

        self.spawn_t = float(spawn_t)
        self.stops = 0
        self._was_moving = True

        self.lane_changes = 0
        self.last_lane_change_t = -1e9

        self.alive = True
        self.finished = False

        self.color = (20, 20, 20)
        self._assign_color()

        self.intent = self._choose_intent(entry_edge)

        # For explicit lane spawn routing:
        # If we spawned into a lane that is already movement L, keep intent aligned.
        ln = self.lane()
        if ln.movement == "L":
            self.intent = "LEFT"
        if ln.movement == "S":
            if self.intent == "LEFT" and (entry_edge in ("TOP","BOTTOM")):
                # let TOP/BOTTOM truly use left lanes by default (Engine tries to select, but keep safe)
                pass

    def _assign_color(self):
        palette = [
            (30,30,30),(10,50,120),(120,30,30),(20,120,60),
            (120,80,20),(60,20,120),(10,110,110),(130,10,90)
        ]
        self.color = palette[self.vid % len(palette)]

    def _choose_intent(self, entry_edge: str) -> str:
        r = random.random()
        if r < 0.70:
            return "STRAIGHT"
        if r < 0.85:
            return "RIGHT"
        return "LEFT"

    def lane(self) -> Lane:
        return self.net.lanes[self.lane_id]

    def pos_ang(self) -> Tuple[Tuple[float,float], float, int]:
        return self.lane().centerline.sample(self.s)

    def aabb(self) -> pygame.Rect:
        (p, ang, _) = self.pos_ang()
        return rect_from_center(p[0], p[1], self.length, self.width)

    # ---------------------------
    # IDM-like acceleration
    # ---------------------------

    def _idm_accel(self, gap: float, dv: float) -> float:
        v = max(0.0, self.v)
        v0 = max(1.0, self.v_des)
        amax = self.a_max
        b = self.b_comf
        T = self.T
        s0 = self.s0
        delta = self.delta

        s_star = s0 + max(0.0, v*T + (v*dv)/(2*math.sqrt(max(1e-6, amax*b))))
        if gap <= 1e-6:
            return -b * 2.0
        term_free = 1.0 - (v/v0)**delta
        term_int = (s_star/gap)**2
        a = amax * (term_free - term_int)
        return clamp(a, -b*2.0, amax)

    # ---------------------------
    # Intersection helpers
    # ---------------------------

    def _near_intersection(self) -> Optional[Intersection]:
        r = self.aabb()
        for inter in self.net.intersections:
            if inter.box.inflate(180,180).colliderect(r):
                return inter
        return None

    def _inside_rect(self, rect: pygame.Rect) -> bool:
        return rect.colliderect(self.aabb())

    def _signal_allows(self, inter: Intersection) -> bool:
        ln = self.lane()
        return inter.signal.allow(ln.approach, ln.movement)

    def _projected_aabb(self, ds: float) -> pygame.Rect:
        s0 = self.s
        self.s = s0 + ds
        rr = self.aabb()
        self.s = s0
        return rr

    # ---------------------------
    # Lane changing (corridor only)
    # ---------------------------

    def _is_corridor_lane(self) -> bool:
        return self.lane_id.startswith("EW_") and ("_I" not in self.lane_id)

    def _adjacent_corridor_lane(self) -> Optional[str]:
        if not self._is_corridor_lane():
            return None
        if self.lane_id.endswith("L0"):
            return self.lane_id[:-2] + "L1"
        if self.lane_id.endswith("L1"):
            return self.lane_id[:-2] + "L0"
        return None

    def try_lane_change(self, now_t: float, vehicles_by_lane: Dict[str,List["Vehicle"]]):
        if not self._is_corridor_lane():
            return
        if (now_t - self.last_lane_change_t) < self.cfg.lane_change_cooldown:
            return

        adj = self._adjacent_corridor_lane()
        if adj is None or adj not in vehicles_by_lane:
            return

        cur_lane_list = vehicles_by_lane.get(self.lane_id, [])
        adj_lane_list = vehicles_by_lane.get(adj, [])

        cur_leader_gap = leader_gap_ahead(self, cur_lane_list)
        adj_leader_gap = leader_gap_ahead(self, adj_lane_list)

        safe_ahead = (adj_leader_gap is None) or (adj_leader_gap > self.cfg.lane_change_check_dist)
        safe_behind = follower_gap_behind(self, adj_lane_list) > self.cfg.lane_change_check_dist

        if not (safe_ahead and safe_behind):
            return

        gcur = (cur_leader_gap if cur_leader_gap is not None else 9999.0)
        gadj = (adj_leader_gap if adj_leader_gap is not None else 9999.0)
        gain = (gadj - gcur) / max(1.0, self.cfg.lane_change_check_dist)

        if gain > self.cfg.lane_change_min_gain:
            self.lane_id = adj
            self.last_lane_change_t = now_t
            self.lane_changes += 1

    # ---------------------------
    # Dynamics update
    # ---------------------------

    def update(self, dt: float, vehicles_same_lane: List["Vehicle"], blocked_virtual_gap: Optional[float] = None) -> Tuple[float,float]:
        if not self.alive:
            return (0.0, 0.0)

        leader = None
        min_gap = 1e18
        dv = 0.0
        for other in vehicles_same_lane:
            if other.vid == self.vid or (not other.alive):
                continue
            if other.s > self.s:
                gap = other.s - self.s - self.length*0.5 - other.length*0.5
                if gap < min_gap:
                    min_gap = gap
                    leader = other
        if leader is None:
            min_gap = 1e18
            dv = 0.0
        else:
            dv = self.v - leader.v

        if blocked_virtual_gap is not None:
            # Treat as virtual leader close ahead
            a_cmd = self._idm_accel(gap=max(self.cfg.min_clearance, blocked_virtual_gap), dv=self.v)
        else:
            if min_gap > 1e9:
                a_cmd = self._idm_accel(gap=1e12, dv=0.0)
            else:
                a_cmd = self._idm_accel(gap=max(self.cfg.min_clearance, min_gap), dv=dv)

        self.a = a_cmd
        self.v = max(0.0, self.v + self.a * dt)
        ds = self.v * dt

        moving = (self.v > 5.0)
        if (not moving) and self._was_moving:
            self.stops += 1
        self._was_moving = moving

        self.s += ds
        return (ds, self.v)

    def maybe_exit_world(self, now_t: float) -> Optional[VehicleRecord]:
        if not self.alive:
            return None

        r = self.aabb()

        for edge, erect in self.net.exit_edges.items():
            if erect.colliderect(r):
                self.alive = False
                self.finished = True
                if not self.exit_edge:
                    self.exit_edge = edge
                if not self.route_name:
                    ln = self.lane()
                    self.route_name = f"{ln.approach}_{ln.movement}"
                delay = (now_t - self.spawn_t)
                rec = VehicleRecord(
                    vid=self.vid,
                    spawn_t=self.spawn_t,
                    exit_t=now_t,
                    entry=self.entry_edge,
                    exit_edge=self.exit_edge,
                    route=self.route_name,
                    stops=self.stops,
                    lane_changes=self.lane_changes,
                    delay=delay
                )
                return rec
        return None

def leader_gap_ahead(me: Vehicle, lane_list: List[Vehicle]) -> Optional[float]:
    best = None
    for v in lane_list:
        if v.vid == me.vid or (not v.alive):
            continue
        if v.s > me.s:
            gap = v.s - me.s - me.length*0.5 - v.length*0.5
            if best is None or gap < best:
                best = gap
    return best

def follower_gap_behind(me: Vehicle, lane_list: List[Vehicle]) -> float:
    best = 1e18
    for v in lane_list:
        if v.vid == me.vid or (not v.alive):
            continue
        if v.s < me.s:
            gap = me.s - v.s - me.length*0.5 - v.length*0.5
            if gap < best:
                best = gap
    return best if best < 1e17 else 1e18

# =============================================================================
#  COLLISION MANAGER
# =============================================================================

class CollisionManager:
    def __init__(self, cfg: Config, metrics: Metrics):
        self.cfg = cfg
        self.metrics = metrics

    def _cell(self, x: float, y: float) -> Tuple[int,int]:
        s = self.cfg.broad_cell
        return (int(x)//s, int(y)//s)

    def build_grid(self, vehicles: List[Vehicle]) -> Dict[Tuple[int,int], List[Vehicle]]:
        grid: Dict[Tuple[int,int], List[Vehicle]] = {}
        for v in vehicles:
            if not v.alive:
                continue
            r = v.aabb()
            c1 = self._cell(r.left, r.top)
            c2 = self._cell(r.right, r.bottom)
            for cx in range(c1[0], c2[0]+1):
                for cy in range(c1[1], c2[1]+1):
                    grid.setdefault((cx,cy), []).append(v)
        return grid

    def nearby(self, grid: Dict[Tuple[int,int], List[Vehicle]], rect: pygame.Rect) -> List[Vehicle]:
        c1 = self._cell(rect.left, rect.top)
        c2 = self._cell(rect.right, rect.bottom)
        out = []
        seen = set()
        for cx in range(c1[0]-1, c2[0]+2):
            for cy in range(c1[1]-1, c2[1]+2):
                for v in grid.get((cx,cy), []):
                    if v.vid not in seen:
                        seen.add(v.vid)
                        out.append(v)
        return out

    def safe_step(self, v: Vehicle, ds: float, vehicles: List[Vehicle], grid: Dict[Tuple[int,int], List[Vehicle]]) -> float:
        if ds <= 0.0:
            return 0.0

        s0 = v.s
        v.s = s0 + ds
        r_full = v.aabb()
        v.s = s0

        candidates = self.nearby(grid, r_full)
        if self._rect_ok(v, r_full, candidates):
            return ds

        lo = 0.0
        hi = 1.0
        for _ in range(self.cfg.resolve_iters):
            mid = (lo + hi) * 0.5
            v.s = s0 + ds*mid
            r_mid = v.aabb()
            v.s = s0
            if self._rect_ok(v, r_mid, candidates):
                lo = mid
            else:
                hi = mid

        safe_ds = ds * lo
        if safe_ds < ds * 0.999:
            self.metrics.collision_avoids += 1
            if lo < 0.35:
                self.metrics.hard_clamps += 1
            else:
                self.metrics.soft_clamps += 1
        return safe_ds

    def _rect_ok(self, v: Vehicle, rect: pygame.Rect, candidates: List[Vehicle]) -> bool:
        for o in candidates:
            if o.vid == v.vid or (not o.alive):
                continue
            if rect.colliderect(o.aabb()):
                return False
        return True

# =============================================================================
#  DEMAND MODEL (Poisson + time-varying)
# =============================================================================

class Demand:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.t = 0.0
        self.next_left = 0.0
        self.next_right = 0.0
        self.next_top = 0.0
        self.next_bottom = 0.0
        self._schedule_initial()

    def _schedule_initial(self):
        self.next_left = self._sample_next(self.cfg.lambda_left)
        self.next_right = self._sample_next(self.cfg.lambda_right)
        self.next_top = self._sample_next(self.cfg.lambda_top)
        self.next_bottom = self._sample_next(self.cfg.lambda_bottom)

    def _wave_multiplier(self, t: float) -> float:
        a = self.cfg.demand_wave_amp
        if self.cfg.demand_wave_period <= 1e-6:
            return 1.0
        return 1.0 + a * math.sin(2.0*math.pi*t/self.cfg.demand_wave_period)

    def _sample_next(self, lam: float) -> float:
        lam_eff = max(1e-6, lam)
        return random.expovariate(lam_eff)

    def tick(self, dt: float) -> Dict[str,int]:
        self.t += dt
        mult = self._wave_multiplier(self.t)

        events = {"LEFT":0, "RIGHT":0, "TOP":0, "BOTTOM":0}

        def step(edge: str, attr: str, base_lam: float):
            nonlocal mult
            tnext = getattr(self, attr)
            tnext -= dt
            count = 0
            while tnext <= 0.0:
                count += 1
                lam_eff = max(1e-6, base_lam * mult)
                tnext += random.expovariate(lam_eff)
            setattr(self, attr, tnext)
            events[edge] = count

        step("LEFT", "next_left", self.cfg.lambda_left)
        step("RIGHT","next_right", self.cfg.lambda_right)
        step("TOP","next_top", self.cfg.lambda_top)
        step("BOTTOM","next_bottom", self.cfg.lambda_bottom)

        return events

# =============================================================================
#  ENGINE
# =============================================================================

class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        pygame.init()
        self.screen = pygame.display.set_mode((cfg.screen_w, cfg.screen_h))
        pygame.display.set_caption("Traffic Simulation (Reservations + Left Lanes + Spillback)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.small = pygame.font.Font(None, 20)

        self.metrics = Metrics()
        self.net = Network(cfg)
        self.demand = Demand(cfg)
        self.collision = CollisionManager(cfg, self.metrics)
        self.resv = ReservationManager(cfg)
        for inter in self.net.intersections:
            self.resv.init_intersection(inter.iid)

        self.vehicles: List[Vehicle] = []
        self.paused = False
        self.t = 0.0
        self._last_queue_sample = 0.0

        self.debug = cfg.draw_debug
        self.draw_paths = cfg.draw_paths
        self.draw_detectors = cfg.draw_detectors
        self.draw_bboxes = cfg.draw_bboxes

    # =============================================================================
    #  SPAWN with intent-aware lane selection (supports dedicated left lanes)
    # =============================================================================

    def _choose_entry_lane_for_edge(self, edge: str) -> Optional[str]:
        lanes = self.net.entry_lanes.get(edge, [])
        if not lanes:
            return None

        # Intent-aware: for TOP/BOTTOM, use explicit left lanes sometimes.
        # For LEFT/RIGHT, sometimes spawn into EW left-turn approach lanes (explicit queue lanes).
        intent = self._sample_intent()

        if edge in ("TOP","BOTTOM"):
            # Prefer NS left lanes when intent is LEFT
            if intent == "LEFT":
                cand = [lid for lid in lanes if lid.endswith("_L")]
                if cand:
                    return random.choice(cand)
            # Prefer straight lanes otherwise
            cand = [lid for lid in lanes if lid.endswith("_S")]
            if cand:
                return random.choice(cand)
            return random.choice(lanes)

        if edge == "LEFT":
            if intent == "LEFT":
                cand = [lid for lid in lanes if ("EW_E_I" in lid and lid.endswith("_L"))]
                if cand:
                    return random.choice(cand)
            # corridor otherwise
            cand = [lid for lid in lanes if lid.startswith("EW_E_L")]
            if cand:
                return random.choice(cand)
            return random.choice(lanes)

        if edge == "RIGHT":
            if intent == "LEFT":
                cand = [lid for lid in lanes if ("EW_W_I" in lid and lid.endswith("_L"))]
                if cand:
                    return random.choice(cand)
            cand = [lid for lid in lanes if lid.startswith("EW_W_L")]
            if cand:
                return random.choice(cand)
            return random.choice(lanes)

        return random.choice(lanes)

    def _sample_intent(self) -> str:
        r = random.random()
        if r < 0.70:
            return "STRAIGHT"
        if r < 0.85:
            return "RIGHT"
        return "LEFT"

    def spawn_vehicle(self, edge: str, count: int):
        if count <= 0:
            return
        for _ in range(count):
            lane_id = self._choose_entry_lane_for_edge(edge)
            if lane_id is None:
                continue
            v = Vehicle(self.cfg, self.net, self.metrics, lane_id, edge, self.t)

            # safe spawn check
            r = v.aabb()
            ok = True
            for o in self.vehicles:
                if o.alive and r.colliderect(o.aabb()):
                    ok = False
                    break
            if ok:
                self.vehicles.append(v)
                self.metrics.spawned += 1

    # =============================================================================
    #  EVENTS
    # =============================================================================

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return False
                if e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if e.key == pygame.K_d:
                    self.debug = not self.debug
                if e.key == pygame.K_p:
                    self.draw_paths = not self.draw_paths
                if e.key == pygame.K_b:
                    self.draw_bboxes = not self.draw_bboxes
                if e.key == pygame.K_t:
                    self.draw_detectors = not self.draw_detectors
                if e.key == pygame.K_PERIOD:
                    if self.paused:
                        self.tick(1.0/self.cfg.fps, stepping=True)
            return True
        return True

    # =============================================================================
    #  ADAPTIVE PRESSURE: now includes left-turn demand separately
    # =============================================================================

    def compute_pressure_scores(self, inter: Intersection) -> List[float]:
        dets = inter.detectors
        allv = [v for v in self.vehicles if v.alive]
        measures = {}
        for k, det in dets.items():
            measures[k] = det.measure(allv, self.cfg.detector_noise_std, self.cfg.detector_drop_prob)

        def nz(x):
            return 0.0 if (x != x) else x

        ew_s = nz(measures.get("EB_S",0.0)) + nz(measures.get("WB_S",0.0))
        ns_s = nz(measures.get("NB_S",0.0)) + nz(measures.get("SB_S",0.0))
        ew_l = nz(measures.get("EW_L",0.0))
        ns_l = nz(measures.get("NS_L",0.0))

        scores = [0.0]*len(inter.signal.phases)
        if len(scores) >= 1:
            scores[0] = ew_s
        if len(scores) >= 2:
            scores[1] = ns_s
        if len(scores) >= 3:
            scores[2] = ew_l * 1.10
        if len(scores) >= 4:
            scores[3] = ns_l * 1.10
        return scores

    # =============================================================================
    #  RESERVATION + SPILLBACK LOGIC (virtual blocking)
    # =============================================================================

    def _find_intersection_for_vehicle(self, v: Vehicle) -> Optional[Intersection]:
        r = v.aabb()
        for inter in self.net.intersections:
            if inter.box.inflate(200,200).colliderect(r):
                return inter
        return None

    def _spillback_blocks(self, inter: Intersection, v: Vehicle, ds: float, grid: Dict[Tuple[int,int], List[Vehicle]]) -> bool:
        # Prevent entering conflict if downstream region is occupied by a slow/queued vehicle.
        # Conservative: project to see if we'd enter conflict; if yes, check "after conflict" region.
        cur = v.aabb()
        proj = v._projected_aabb(ds)
        entering = (not inter.conflict.colliderect(cur)) and inter.conflict.colliderect(proj)
        if not entering:
            return False

        # Determine direction of travel near this location using tangent angle from path.
        (p, ang, _) = v.pos_ang()
        # Create a rectangle ahead of conflict zone in direction of travel.
        # We'll approximate by using angle quadrants.
        look = self.cfg.spillback_lookahead
        pad = self.cfg.spillback_margin
        cz = inter.conflict

        ahead = None
        if -45 <= ang <= 45:
            # moving right
            ahead = pygame.Rect(cz.right+pad, cz.top, look, cz.height)
        elif 135 <= ang or ang <= -135:
            # moving left
            ahead = pygame.Rect(cz.left-look-pad, cz.top, look, cz.height)
        elif 45 < ang < 135:
            # moving up
            ahead = pygame.Rect(cz.left, cz.top-look-pad, cz.width, look)
        else:
            # moving down
            ahead = pygame.Rect(cz.left, cz.bottom+pad, cz.width, look)

        # If any slow vehicle occupies ahead region, block.
        # Use grid broadphase by scanning relevant cells with CollisionManager.nearby pattern.
        # We'll reuse a simple scan from all vehicles because counts are moderate; still efficient enough.
        for o in self.vehicles:
            if (not o.alive) or (o.vid == v.vid):
                continue
            if ahead.colliderect(o.aabb()) and o.v < self.cfg.spillback_min_speed:
                return True
        return False

    def _reservation_blocks(self, inter: Intersection, v: Vehicle, ds: float) -> bool:
        # Vehicles must hold reservation when entering conflict zone.
        cur = v.aabb()
        proj = v._projected_aabb(ds)

        entering = (not inter.conflict.colliderect(cur)) and inter.conflict.colliderect(proj)
        inside = inter.conflict.colliderect(cur)

        if inside:
            # keep token touched if we own it
            if self.resv.get_owner(inter.iid) == v.vid:
                self.resv.touch(inter.iid, self.t)
            return False

        if not entering:
            return False

        # Must have signal green as well (enforced separately), but reservation must be acquired now.
        ok = self.resv.try_acquire(inter.iid, v.vid, self.t)
        return (not ok)

    def _release_reservations(self):
        # Clear reservation when owner has fully exited conflict zone.
        for inter in self.net.intersections:
            own = self.resv.get_owner(inter.iid)
            if own is None:
                continue
            self.resv.release_if_stale(inter.iid, self.t)
            own2 = self.resv.get_owner(inter.iid)
            if own2 is None:
                continue
            # if owner not inside conflict anymore, release
            owner_vehicle = None
            for v in self.vehicles:
                if v.alive and v.vid == own2:
                    owner_vehicle = v
                    break
            if owner_vehicle is None:
                self.resv.release(inter.iid)
                continue
            if not inter.conflict.colliderect(owner_vehicle.aabb()):
                self.resv.release(inter.iid)

    # =============================================================================
    #  SIM TICK
    # =============================================================================

    def tick(self, dt: float, stepping: bool = False):
        if dt <= 0.0:
            return
        if dt > self.cfg.max_dt:
            steps = int(math.ceil(dt/self.cfg.max_dt))
            sub = dt/steps
            for _ in range(steps):
                self.tick(sub, stepping=stepping)
            return

        self.t += dt

        events = self.demand.tick(dt)
        self.spawn_vehicle("LEFT", events["LEFT"])
        self.spawn_vehicle("RIGHT", events["RIGHT"])
        self.spawn_vehicle("TOP", events["TOP"])
        self.spawn_vehicle("BOTTOM", events["BOTTOM"])

        for inter in self.net.intersections:
            scores = self.compute_pressure_scores(inter)
            inter.signal.tick(dt, pressure_scores=scores)

        vehicles_by_lane: Dict[str, List[Vehicle]] = {}
        for v in self.vehicles:
            if v.alive:
                vehicles_by_lane.setdefault(v.lane_id, []).append(v)

        for lane_id, lane_veh in vehicles_by_lane.items():
            lane_veh.sort(key=lambda x: x.s)
        for v in self.vehicles:
            if v.alive:
                v.try_lane_change(self.t, vehicles_by_lane)

        vehicles_by_lane = {}
        for v in self.vehicles:
            if v.alive:
                vehicles_by_lane.setdefault(v.lane_id, []).append(v)
        for lane_id, lane_veh in vehicles_by_lane.items():
            lane_veh.sort(key=lambda x: x.s)

        grid = self.collision.build_grid([v for v in self.vehicles if v.alive])

        # Longitudinal update + signal/reservation/spillback blocking:
        for lane_id, lane_veh in vehicles_by_lane.items():
            lane_veh.sort(key=lambda x: x.s, reverse=True)
            for v in lane_veh:
                if not v.alive:
                    continue

                inter = self._find_intersection_for_vehicle(v)
                blocked_virtual_gap = None

                # Signal must be GREEN for entering conflict zone (not merely near it).
                # Reservation must be acquired for entering conflict zone.
                if inter is not None:
                    # Estimate intended ds ignoring collisions first: we need a ds candidate.
                    # We'll compute ds from current speed guess by doing a provisional update, then undo and redo safely.
                    s_before = v.s
                    ds_guess = max(0.0, v.v * dt)
                    if ds_guess < 1e-6:
                        ds_guess = 0.0

                    cur = v.aabb()
                    proj = v._projected_aabb(ds_guess)
                    entering_conflict = (not inter.conflict.colliderect(cur)) and inter.conflict.colliderect(proj)

                    if entering_conflict:
                        if not v._signal_allows(inter):
                            blocked_virtual_gap = max(v.s0, 10.0)
                        else:
                            if self._spillback_blocks(inter, v, ds_guess, grid):
                                blocked_virtual_gap = max(v.s0, 10.0)
                            else:
                                if self._reservation_blocks(inter, v, ds_guess):
                                    blocked_virtual_gap = max(v.s0, 10.0)

                ds, vv = v.update(dt, lane_veh, blocked_virtual_gap=blocked_virtual_gap)

                safe_ds = self.collision.safe_step(v, ds, self.vehicles, grid)
                v.s -= ds
                v.s += safe_ds

        self._release_reservations()

        for v in self.vehicles:
            if not v.alive:
                continue
            rec = v.maybe_exit_world(self.t)
            if rec is not None:
                self.metrics.record_vehicle_exit(rec)

        if (self.t - self._last_queue_sample) >= 1.0:
            qL = self._queue_count_edge("LEFT")
            qR = self._queue_count_edge("RIGHT")
            qT = self._queue_count_edge("TOP")
            qB = self._queue_count_edge("BOTTOM")
            self.metrics.queue_samples.append((self.t, qL, qR, qT, qB))
            self._last_queue_sample = self.t

        if int(self.t) >= int(self.cfg.sim_seconds) and (not stepping):
            raise StopIteration

    def _queue_count_edge(self, edge: str) -> int:
        c = 0
        for v in self.vehicles:
            if not v.alive:
                continue
            if v.entry_edge != edge:
                continue
            if v.s < 200.0 and v.v < 10.0:
                c += 1
        return c

    # =============================================================================
    #  DRAWING
    # =============================================================================

    def draw_background(self):
        self.screen.fill((235, 235, 235))

        y = self.cfg.road_y
        lw = self.cfg.lane_width
        road_h = lw * 4
        pygame.draw.rect(self.screen, (60,60,60), pygame.Rect(0, int(y-road_h/2), self.cfg.screen_w, int(road_h)))

        for k in range(-1, 2):
            yy = y + k*lw
            pygame.draw.line(self.screen, (220,220,220), (0, int(yy)), (self.cfg.screen_w, int(yy)), 2)

        for inter in self.net.intersections:
            pygame.draw.rect(self.screen, (90,90,90), inter.box, 2)
            pygame.draw.rect(self.screen, (140,140,140), inter.conflict, 1)

            pygame.draw.line(self.screen, (230,230,230), (inter.box.left, inter.cy-40), (inter.box.right, inter.cy-40), 2)
            pygame.draw.line(self.screen, (230,230,230), (inter.box.left, inter.cy+40), (inter.box.right, inter.cy+40), 2)

    def draw_paths_debug(self):
        if not self.draw_paths:
            return
        for lid, lane in self.net.lanes.items():
            pts = []
            total = lane.centerline.total_length
            steps = 60
            for i in range(steps+1):
                s = total * (i/steps)
                p, ang, segi = lane.centerline.sample(s)
                pts.append((int(p[0]), int(p[1])))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (120,180,255), False, pts, 1)

    def draw_detectors_debug(self):
        if not self.draw_detectors:
            return
        for inter in self.net.intersections:
            for k, det in inter.detectors.items():
                pygame.draw.rect(self.screen, (255, 200, 0), det.rect, 1)
                s = self.small.render(k, True, (10,10,10))
                self.screen.blit(s, (det.rect.x+2, det.rect.y+2))

    def draw_signals(self):
        for inter in self.net.intersections:
            sig = inter.signal
            ph = sig.current().name
            st = sig.state
            txt = f"I{inter.iid} {st} {ph.split('_')[-1]} t={sig.timer:.1f}"
            s = self.small.render(txt, True, (10,10,10))
            self.screen.blit(s, (inter.box.left, inter.box.top-18))

            ew_ok = sig.allow(f"EW{inter.iid}", "S")
            ns_ok = sig.allow(f"NS{inter.iid}", "S")
            ew_l_ok = sig.allow(f"EW{inter.iid}", "L")
            ns_l_ok = sig.allow(f"NS{inter.iid}", "L")

            col_ew = (0,180,0) if ew_ok else (180,0,0)
            col_ns = (0,180,0) if ns_ok else (180,0,0)
            col_ewl = (0,180,0) if ew_l_ok else (180,0,0)
            col_nsl = (0,180,0) if ns_l_ok else (180,0,0)

            pygame.draw.circle(self.screen, col_ew, (inter.cx-40, inter.box.top-35), 7)
            pygame.draw.circle(self.screen, col_ns, (inter.cx+40, inter.box.top-35), 7)
            pygame.draw.circle(self.screen, col_ewl, (inter.cx-15, inter.box.top-35), 6)
            pygame.draw.circle(self.screen, col_nsl, (inter.cx+15, inter.box.top-35), 6)

            ped_ew = sig.ped_allow(f"PED_EW{inter.iid}")
            ped_ns = sig.ped_allow(f"PED_NS{inter.iid}")
            col_p1 = (0,120,220) if ped_ew else (140,140,140)
            col_p2 = (0,120,220) if ped_ns else (140,140,140)
            pygame.draw.rect(self.screen, col_p1, pygame.Rect(inter.cx-45, inter.box.bottom+8, 18, 10))
            pygame.draw.rect(self.screen, col_p2, pygame.Rect(inter.cx+27, inter.box.bottom+8, 18, 10))

            own = self.resv.get_owner(inter.iid)
            if own is not None:
                rr = self.small.render(f"RESV:{own}", True, (50,50,50))
                self.screen.blit(rr, (inter.box.right-70, inter.box.top-18))

    def draw_vehicles(self):
        for v in self.vehicles:
            if not v.alive:
                continue
            p, ang, _ = v.pos_ang()

            surf = pygame.Surface((int(v.length), int(v.width)), pygame.SRCALPHA)
            surf.fill((*v.color, 255))
            rot = pygame.transform.rotate(surf, ang)
            rr = rot.get_rect(center=(int(p[0]), int(p[1])))
            self.screen.blit(rot, rr.topleft)

            if self.draw_bboxes:
                pygame.draw.rect(self.screen, (0,0,0), v.aabb(), 1)

    def draw_hud(self, fps: float):
        total = self.metrics.spawned
        exited = self.metrics.exited
        txt = f"t={int(self.t)}s  FPS={fps:.1f}  spawned={total}  exited={exited}  avoids={self.metrics.collision_avoids}"
        s = self.font.render(txt, True, (0,0,0))
        self.screen.blit(s, (10, 10))

        keys = "SPACE pause | . step | D debug | P paths | T detectors | B bboxes | ESC quit"
        s2 = self.small.render(keys, True, (0,0,0))
        self.screen.blit(s2, (10, 36))

        if self.paused:
            s3 = self.font.render("PAUSED", True, (200,0,0))
            self.screen.blit(s3, (10, 58))

        if self.metrics.queue_samples:
            t,qL,qR,qT,qB = self.metrics.queue_samples[-1]
            s4 = self.small.render(f"Queues (approx): L={qL} R={qR} T={qT} B={qB}", True, (0,0,0))
            self.screen.blit(s4, (10, 82))

    # =============================================================================
    #  RUN LOOP
    # =============================================================================

    def run(self):
        logging.info("Simulation start.")
        try:
            while True:
                ok = self.handle_events()
                if not ok:
                    break

                dt = self.clock.tick(self.cfg.fps)/1000.0
                fps = self.clock.get_fps()

                if not self.paused:
                    try:
                        self.tick(dt)
                    except StopIteration:
                        break

                self.draw_background()
                self.draw_paths_debug()
                self.draw_detectors_debug()
                self.draw_signals()
                self.draw_vehicles()
                self.draw_hud(fps)

                pygame.display.flip()

        finally:
            pygame.quit()
            self.metrics.export(self.cfg)
            logging.info("Simulation end. spawned=%d exited=%d avoids=%d hard=%d soft=%d",
                         self.metrics.spawned, self.metrics.exited, self.metrics.collision_avoids,
                         self.metrics.hard_clamps, self.metrics.soft_clamps)

# =============================================================================
#  MAIN
# =============================================================================

base = Config()
cfg = load_config(base.config_json, base)
eng = Engine(cfg)
eng.run()
