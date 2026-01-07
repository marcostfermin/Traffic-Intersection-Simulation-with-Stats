import os
import sys
import math
import json
import time
import queue
import random
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pygame

# -----------------------------------------------------------------------------
#  Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("simulation.log", mode="w", encoding="utf-8")
    ],
)

# -----------------------------------------------------------------------------
#  Config + safe reload
# -----------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    screenWidth: int = 1400
    screenHeight: int = 800
    fps: int = 60
    simulationTime: int = 300

    # signal timing
    defaultYellow: int = 5
    defaultRedBuffer: int = 0
    randomGreenSignalTimer: bool = True
    randomGreenSignalTimerRange: Tuple[int, int] = (10, 20)
    defaultGreen: Dict[int, int] = None

    # geometry and UI
    signalCoods: List[Tuple[int, int]] = None
    signalTimerCoods: List[Tuple[int, int]] = None
    vehicleCountCoods: List[Tuple[int, int]] = None
    timeElapsedCoods: Tuple[int, int] = (1100, 50)

    stopLines: Dict[str, int] = None
    defaultStop: Dict[str, int] = None

    stoppingGap: int = 25
    movingGap: int = 25

    # spawn points (constant)
    spawnX: Dict[str, List[int]] = None
    spawnY: Dict[str, List[int]] = None

    # turning geometry anchors
    mid: Dict[str, Dict[str, int]] = None

    # spawn/traffic
    spawnIntervalSec: float = 1.0
    turnProbabilityLane1: float = 0.40
    turnProbabilityLane2: float = 0.40
    directionDistribution: List[int] = None  # cumulative percent thresholds (len=4)

    # asset paths
    backgroundPath: str = "images/intersection.png"
    redSignalPath: str = "images/signals/red.png"
    yellowSignalPath: str = "images/signals/yellow.png"
    greenSignalPath: str = "images/signals/green.png"
    vehicleImageTemplate: str = "images/{direction}/{vehicleClass}.png"

    # speeds in px/sec
    speedsPxPerSec: Dict[str, float] = None

    # collision/physics robustness
    maxDt: float = 0.05          # cap dt to reduce "jump" collisions
    broadphaseCell: int = 160    # spatial hash cell size
    resolveIters: int = 14       # binary search iterations for clamping
    intersectionRect: Tuple[int, int, int, int] = (610, 340, 180, 180)  # conservative conflict zone
    conservativeIntersection: bool = True  # if True, only 1 vehicle allowed inside conflict zone

    def __post_init__(self):
        if self.defaultGreen is None:
            self.defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}

        if self.signalCoods is None:
            self.signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]

        if self.signalTimerCoods is None:
            self.signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

        if self.vehicleCountCoods is None:
            self.vehicleCountCoods = [(480, 210), (880, 210), (880, 550), (480, 550)]

        if self.stopLines is None:
            self.stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}

        if self.defaultStop is None:
            self.defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

        if self.spawnX is None:
            self.spawnX = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}

        if self.spawnY is None:
            self.spawnY = {'right': [348, 370, 398], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

        if self.mid is None:
            self.mid = {'right': {'x': 705, 'y': 445}, 'down': {'x': 695, 'y': 450}, 'left': {'x': 695, 'y': 425}, 'up': {'x': 695, 'y': 400}}

        if self.directionDistribution is None:
            self.directionDistribution = [25, 50, 75, 100]

        if self.speedsPxPerSec is None:
            self.speedsPxPerSec = {'car': 135.0, 'bus': 108.0, 'truck': 108.0, 'bike': 150.0}

def load_config_from_file(path: str, base: SimulationConfig) -> SimulationConfig:
    if not os.path.exists(path):
        return base
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = SimulationConfig(**{**base.__dict__, **data})
        logging.info("Loaded config from %s", path)
        return cfg
    except Exception as e:
        logging.error("Failed to load config %s: %s", path, str(e))
        return base

# -----------------------------------------------------------------------------
#  Assets validation
# -----------------------------------------------------------------------------

class AssetManager:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.cache: Dict[str, pygame.Surface] = {}

    def validate_or_die(self, required_vehicle_types: List[str], directions: List[str]):
        paths = [self.config.backgroundPath, self.config.redSignalPath, self.config.yellowSignalPath, self.config.greenSignalPath]
        for d in directions:
            for vt in required_vehicle_types:
                paths.append(self.config.vehicleImageTemplate.format(direction=d, vehicleClass=vt))

        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            msg = "Missing required asset(s):\n" + "\n".join(missing)
            logging.error(msg)
            raise FileNotFoundError(msg)

    def image(self, path: str) -> pygame.Surface:
        if path in self.cache:
            return self.cache[path]
        img = pygame.image.load(path).convert_alpha()
        self.cache[path] = img
        return img

# -----------------------------------------------------------------------------
#  Signals: tick-based controller
# -----------------------------------------------------------------------------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0]-a[0], b[1]-a[1])

def angle_deg_from_vec(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(-dy, dx))

class SignalController:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.noOfSignals = 4
        self.currentGreen = 0
        self.currentYellow = 0

        self._phase_state = "GREEN"  # GREEN, YELLOW, ALL_RED
        self._green_remaining = 0.0
        self._yellow_remaining = 0.0
        self._all_red_remaining = 0.0

        self._set_new_green(self.currentGreen)

    def _pick_green_time(self, idx: int) -> float:
        if self.config.randomGreenSignalTimer:
            return float(random.randint(self.config.randomGreenSignalTimerRange[0], self.config.randomGreenSignalTimerRange[1]))
        return float(self.config.defaultGreen[idx])

    def _set_new_green(self, idx: int):
        self.currentGreen = idx
        self.currentYellow = 0
        self._phase_state = "GREEN"
        self._green_remaining = self._pick_green_time(idx)
        self._yellow_remaining = float(self.config.defaultYellow)
        self._all_red_remaining = float(self.config.defaultRedBuffer)

    def tick(self, dt: float):
        if dt <= 0:
            return

        if self._phase_state == "GREEN":
            self._green_remaining -= dt
            if self._green_remaining <= 0:
                self._phase_state = "YELLOW"
                self.currentYellow = 1

        elif self._phase_state == "YELLOW":
            self._yellow_remaining -= dt
            if self._yellow_remaining <= 0:
                self.currentYellow = 0
                if self._all_red_remaining > 0:
                    self._phase_state = "ALL_RED"
                else:
                    self._advance_to_next_green()

        elif self._phase_state == "ALL_RED":
            self._all_red_remaining -= dt
            if self._all_red_remaining <= 0:
                self._advance_to_next_green()

    def _advance_to_next_green(self):
        next_green = (self.currentGreen + 1) % self.noOfSignals
        self._set_new_green(next_green)

    def get_state(self) -> Tuple[int, int]:
        return self.currentGreen, self.currentYellow

    def display_value_for_signal(self, idx: int) -> str:
        if idx == self.currentGreen:
            if self.currentYellow == 1:
                return str(max(0, int(math.ceil(self._yellow_remaining))))
            return str(max(0, int(math.ceil(self._green_remaining))))
        return "---"

# -----------------------------------------------------------------------------
#  Paths (for turning)
# -----------------------------------------------------------------------------

class Path:
    def __init__(self, points: List[Tuple[float, float]]):
        self.points = points[:]
        self._seg_len: List[float] = []
        self._cum: List[float] = [0.0]
        total = 0.0
        for i in range(len(points)-1):
            L = dist(points[i], points[i+1])
            self._seg_len.append(L)
            total += L
            self._cum.append(total)
        self.length = total

    def sample(self, s: float) -> Tuple[float, float]:
        if self.length <= 1e-9:
            return self.points[-1]
        s = clamp(s, 0.0, self.length)
        for i in range(len(self._seg_len)):
            if s <= self._cum[i+1]:
                seg_s = s - self._cum[i]
                L = self._seg_len[i]
                if L <= 1e-9:
                    return self.points[i+1]
                t = seg_s / L
                ax, ay = self.points[i]
                bx, by = self.points[i+1]
                return (ax + (bx-ax)*t, ay + (by-ay)*t)
        return self.points[-1]

    def heading(self, s: float) -> float:
        if len(self.points) < 2:
            return 0.0
        s = clamp(s, 0.0, self.length)
        eps = 6.0
        p1 = self.sample(s)
        p2 = self.sample(clamp(s+eps, 0.0, self.length))
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        return angle_deg_from_vec(dx, dy)

# -----------------------------------------------------------------------------
#  World state + stats
# -----------------------------------------------------------------------------

class WorldState:
    def __init__(self, config: SimulationConfig, signalController: SignalController):
        self.config = config
        self.signalController = signalController

        self.directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

        self.vehicles: Dict[str, Dict[int, List["Vehicle"]]] = {
            'right': {1: [], 2: []},
            'down': {1: [], 2: []},
            'left': {1: [], 2: []},
            'up': {1: [], 2: []},
        }

        self.crossed = {'right': 0, 'down': 0, 'left': 0, 'up': 0}

        self.simulation = pygame.sprite.Group()

        self.queue_len_samples = {'right': [], 'down': [], 'left': [], 'up': []}
        self.throughput_events: List[Tuple[float, str]] = []

        # collision/reservation
        self.intersection_rect = pygame.Rect(*self.config.intersectionRect)
        self.intersection_owner_id: Optional[int] = None

    def all_vehicles(self) -> List["Vehicle"]:
        out = []
        for d in ['right', 'down', 'left', 'up']:
            for lane in [1, 2]:
                out.extend(self.vehicles[d][lane])
        return out

    def lane_list(self, direction: str, lane: int) -> List["Vehicle"]:
        return self.vehicles[direction][lane]

    def register_vehicle(self, v: "Vehicle"):
        self.vehicles[v.direction][v.lane].append(v)
        self.simulation.add(v)

    def unregister_vehicle(self, v: "Vehicle"):
        lane = self.vehicles[v.direction][v.lane]
        if v in lane:
            lane.remove(v)
        if v in self.simulation:
            self.simulation.remove(v)

        if self.intersection_owner_id == v.vid:
            self.intersection_owner_id = None

    def record_crossed(self, direction: str, now_s: float):
        self.crossed[direction] += 1
        self.throughput_events.append((now_s, direction))

    def sample_queue_lengths(self):
        for d in ['right', 'down', 'left', 'up']:
            q = 0
            for lane in [1, 2]:
                for v in self.vehicles[d][lane]:
                    if not v.has_crossed:
                        q += 1
            self.queue_len_samples[d].append(q)

# -----------------------------------------------------------------------------
#  Strong collision handling:
#   - dt cap
#   - spawn clearance (handled in Spawner)
#   - same-lane clamping (prevents rear from overtaking/overlapping front)
#   - broadphase spatial hash + binary-search clamping against ALL vehicles
#   - conservative intersection reservation (prevents cross-direction conflicts)
# -----------------------------------------------------------------------------

class CollisionManager:
    def __init__(self, world: WorldState):
        self.world = world
        self.config = world.config

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        s = self.config.broadphaseCell
        return (int(x) // s, int(y) // s)

    def _build_spatial_hash(self) -> Dict[Tuple[int, int], List["Vehicle"]]:
        grid: Dict[Tuple[int, int], List["Vehicle"]] = {}
        for v in self.world.all_vehicles():
            r = v.rect()
            c1 = self._cell(r.left, r.top)
            c2 = self._cell(r.right, r.bottom)
            for cx in range(c1[0], c2[0] + 1):
                for cy in range(c1[1], c2[1] + 1):
                    grid.setdefault((cx, cy), []).append(v)
        return grid

    def _nearby_candidates(self, grid: Dict[Tuple[int, int], List["Vehicle"]], rect: pygame.Rect) -> List["Vehicle"]:
        c1 = self._cell(rect.left, rect.top)
        c2 = self._cell(rect.right, rect.bottom)
        out = []
        seen = set()
        for cx in range(c1[0] - 1, c2[0] + 2):
            for cy in range(c1[1] - 1, c2[1] + 2):
                for v in grid.get((cx, cy), []):
                    if v.vid not in seen:
                        seen.add(v.vid)
                        out.append(v)
        return out

    def _intersection_entering(self, v: "Vehicle", new_rect: pygame.Rect) -> bool:
        ir = self.world.intersection_rect
        return (not v.rect().colliderect(ir)) and new_rect.colliderect(ir)

    def _intersection_inside(self, rect: pygame.Rect) -> bool:
        return rect.colliderect(self.world.intersection_rect)

    def _intersection_allowed(self, v: "Vehicle", new_rect: pygame.Rect) -> bool:
        if not self.config.conservativeIntersection:
            return True

        ir = self.world.intersection_rect

        if not new_rect.colliderect(ir):
            if self.world.intersection_owner_id == v.vid:
                self.world.intersection_owner_id = None
            return True

        # If already owner, always allowed
        if self.world.intersection_owner_id == v.vid:
            return True

        # If nobody owns it, acquire when entering (or when trying to be inside)
        if self.world.intersection_owner_id is None:
            self.world.intersection_owner_id = v.vid
            return True

        # Otherwise blocked
        return False

    def resolve_all(self, dt: float, now_s: float):
        # Build broadphase grid once per frame
        grid = self._build_spatial_hash()

        # Update in a stable order:
        # front-to-back per direction lane so clamping becomes consistent.
        # If vehicles overlap already, this still avoids the worst "rear pushes front" issues.
        ordered = self._ordered_update_list()
        for v in ordered:
            if not v.alive:
                continue
            self.resolve_vehicle(v, dt, now_s, grid)

    def _ordered_update_list(self) -> List["Vehicle"]:
        out = []
        for d in ['right', 'down', 'left', 'up']:
            for lane in [1, 2]:
                lane_list = self.world.lane_list(d, lane)
                # sort by along-lane progress so front vehicles update first
                lane_sorted = sorted(lane_list, key=lambda vv: vv.length_along_lane())
                out.extend(lane_sorted)
        return out

    def resolve_vehicle(self, v: "Vehicle", dt: float, now_s: float, grid: Dict[Tuple[int, int], List["Vehicle"]]):
        # 1) Cap dt at config.maxDt by splitting into substeps
        # This prevents large dt jumps creating unavoidable overlaps.
        remaining = dt
        while remaining > 1e-9 and v.alive:
            step_dt = remaining if remaining <= self.config.maxDt else self.config.maxDt
            remaining -= step_dt
            self._resolve_vehicle_substep(v, step_dt, now_s, grid)

    def _resolve_vehicle_substep(self, v: "Vehicle", dt: float, now_s: float, grid: Dict[Tuple[int, int], List["Vehicle"]]):
        # Compute intended motion scalar (lane step or path step)
        v.prepare_motion(dt)

        if v.pending_kind is None or v.pending_amount <= 0.0:
            v.post_motion(now_s)  # still allow crossed check / cleanup
            return

        # 2) Same-lane hard clamp to prevent rear overlapping front (robust for straight & turning alike)
        max_fraction_lane = self._lane_clamp_fraction(v)

        # 3) Intersection reservation clamp
        max_fraction_intersection = self._intersection_clamp_fraction(v, grid)

        # 4) Broad collision clamp against ALL nearby vehicles using binary search
        max_fraction_broad = self._broad_collision_clamp_fraction(v, grid)

        # apply the most conservative clamp
        f = min(1.0, max_fraction_lane, max_fraction_intersection, max_fraction_broad)
        if f < 0.0:
            f = 0.0

        v.apply_motion_fraction(f)
        v.post_motion(now_s)

    def _lane_clamp_fraction(self, v: "Vehicle") -> float:
        # Strong lane clamping: ensure proposed motion doesn't violate movingGap relative to front.
        front = v.front_vehicle()
        if front is None or (not front.alive):
            return 1.0

        # If front is in same lane, enforce gap using current front rect (conservative).
        mg = float(self.config.movingGap)

        # compute my current and intended
        if v.pending_kind == "lane":
            step = v.pending_amount
            if v.direction == 'right':
                my_front_now = v.x + v.image.get_rect().width
                front_back = front.x - mg
                max_move = front_back - my_front_now
                if max_move <= 0:
                    return 0.0
                return clamp(max_move / step, 0.0, 1.0)

            if v.direction == 'left':
                my_back_now = v.x
                front_front = front.x + front.image.get_rect().width + mg
                max_move = my_back_now - front_front
                if max_move <= 0:
                    return 0.0
                return clamp(max_move / step, 0.0, 1.0)

            if v.direction == 'down':
                my_front_now = v.y + v.image.get_rect().height
                front_back = front.y - mg
                max_move = front_back - my_front_now
                if max_move <= 0:
                    return 0.0
                return clamp(max_move / step, 0.0, 1.0)

            if v.direction == 'up':
                my_back_now = v.y
                front_front = front.y + front.image.get_rect().height + mg
                max_move = my_back_now - front_front
                if max_move <= 0:
                    return 0.0
                return clamp(max_move / step, 0.0, 1.0)

            return 1.0

        # For path movement, do broadphase clamp; lane clamp doesn't apply cleanly.
        return 1.0

    def _intersection_clamp_fraction(self, v: "Vehicle", grid: Dict[Tuple[int, int], List["Vehicle"]]) -> float:
        # If conservative intersection is on, do NOT allow entering/being inside unless you own it.
        # We clamp motion via binary search to stop at the border if not allowed.
        if not self.config.conservativeIntersection:
            return 1.0

        test_rect_full = v.proposed_rect_for_fraction(1.0)
        if not test_rect_full.colliderect(self.world.intersection_rect):
            return 1.0

        # If fully allowed, no clamp.
        if self._intersection_allowed(v, test_rect_full):
            return 1.0

        # Otherwise, binary search largest fraction that does NOT violate intersection rule.
        lo = 0.0
        hi = 1.0
        for _ in range(self.config.resolveIters):
            mid = (lo + hi) * 0.5
            r = v.proposed_rect_for_fraction(mid)
            if self._intersection_allowed(v, r):
                lo = mid
            else:
                hi = mid
        return lo

    def _broad_collision_clamp_fraction(self, v: "Vehicle", grid: Dict[Tuple[int, int], List["Vehicle"]]) -> float:
        # Binary search fraction that avoids colliding with any other vehicle rect.
        # This handles:
        # - straight/straight collisions across lanes
        # - turning/straight overlaps
        # - turning/turning overlaps
        # - any weird geometry or dt corner cases
        #
        # It is conservative because it tests AABB overlap; rotation changes rect size but
        # pygame uses rotated surface rect; this is still safer than ignoring rotation.

        test_full = v.proposed_rect_for_fraction(1.0)
        candidates = self._nearby_candidates(grid, test_full)

        # quick check if full move is safe
        if self._rect_safe_against_candidates(v, test_full, candidates):
            return 1.0

        lo = 0.0
        hi = 1.0
        for _ in range(self.config.resolveIters):
            mid = (lo + hi) * 0.5
            r = v.proposed_rect_for_fraction(mid)
            if self._rect_safe_against_candidates(v, r, candidates):
                lo = mid
            else:
                hi = mid
        return lo

    def _rect_safe_against_candidates(self, v: "Vehicle", rect: pygame.Rect, candidates: List["Vehicle"]) -> bool:
        # also include intersection reservation rule for safety
        if not self._intersection_allowed(v, rect):
            return False

        for other in candidates:
            if other.vid == v.vid:
                continue
            if not other.alive:
                continue
            if rect.colliderect(other.rect()):
                return False
        return True

# -----------------------------------------------------------------------------
#  Vehicle
# -----------------------------------------------------------------------------

class Vehicle(pygame.sprite.Sprite):
    _VID_COUNTER = 1

    def __init__(
        self,
        assets: AssetManager,
        world: WorldState,
        vehicleClass: str,
        direction_number: int,
        direction: str,
        lane: int,
        will_turn: int,
    ):
        pygame.sprite.Sprite.__init__(self)
        self.assets = assets
        self.world = world
        self.config = world.config

        self.vid = Vehicle._VID_COUNTER
        Vehicle._VID_COUNTER += 1

        self.vehicleClass = vehicleClass
        self.direction_number = direction_number
        self.direction = direction
        self.lane = lane
        self.willTurn = will_turn

        self.speed = float(self.config.speedsPxPerSec[vehicleClass])

        path = self.config.vehicleImageTemplate.format(direction=direction, vehicleClass=vehicleClass)
        self.originalImage = self.assets.image(path)
        self.image = self.originalImage

        self.x = float(self.config.spawnX[direction][lane])
        self.y = float(self.config.spawnY[direction][lane])

        self.has_crossed = False
        self.in_turn_path = False
        self.turn_progress = 0.0
        self.turn_path: Optional[Path] = None

        self.turn_start_condition = None
        self._turn_points_template: List[Tuple[float, float]] = []

        self.stop = float(self.config.defaultStop[self.direction])
        self._compute_stop_from_front()

        self._configure_turn_gate_and_path()

        self.pending_kind: Optional[str] = None     # "lane" or "path"
        self.pending_amount: float = 0.0           # px (lane) or path distance increment
        self.pending_dt: float = 0.0

        self.alive = True

        self.world.register_vehicle(self)

    def rect(self) -> pygame.Rect:
        r = self.image.get_rect()
        r.topleft = (int(self.x), int(self.y))
        return r

    def length_along_lane(self) -> float:
        if self.direction == 'right':
            return self.x
        if self.direction == 'left':
            return -self.x
        if self.direction == 'down':
            return self.y
        if self.direction == 'up':
            return -self.y
        return 0.0

    def front_vehicle(self) -> Optional["Vehicle"]:
        lane = self.world.lane_list(self.direction, self.lane)
        ordered = sorted(lane, key=lambda v: v.length_along_lane())
        try:
            idx = ordered.index(self)
        except ValueError:
            return None
        if idx == 0:
            return None
        return ordered[idx-1]

    def _compute_stop_from_front(self):
        front = self.front_vehicle()
        if front is None:
            self.stop = float(self.config.defaultStop[self.direction])
            return
        if front.has_crossed:
            self.stop = float(self.config.defaultStop[self.direction])
            return

        fg = float(self.config.stoppingGap)
        if self.direction == 'right':
            self.stop = float(front.stop - front.image.get_rect().width - fg)
        elif self.direction == 'left':
            self.stop = float(front.stop + front.image.get_rect().width + fg)
        elif self.direction == 'down':
            self.stop = float(front.stop - front.image.get_rect().height - fg)
        elif self.direction == 'up':
            self.stop = float(front.stop + front.image.get_rect().height + fg)
        else:
            self.stop = float(self.config.defaultStop[self.direction])

    def _signal_allows(self) -> bool:
        currentGreen, currentYellow = self.world.signalController.get_state()
        my_green_index = self.direction_number
        if self.has_crossed:
            return True
        if currentGreen == my_green_index and currentYellow == 0:
            return True
        return False

    def _is_before_stop_line(self) -> bool:
        sl = self.config.stopLines[self.direction]
        if self.direction == 'right':
            return (self.x + self.image.get_rect().width) <= sl
        if self.direction == 'left':
            return self.x >= sl
        if self.direction == 'down':
            return (self.y + self.image.get_rect().height) <= sl
        if self.direction == 'up':
            return self.y >= sl
        return True

    def _at_or_past_stop_point(self) -> bool:
        if self.direction == 'right':
            return (self.x + self.image.get_rect().width) >= self.stop
        if self.direction == 'left':
            return self.x <= self.stop
        if self.direction == 'down':
            return (self.y + self.image.get_rect().height) >= self.stop
        if self.direction == 'up':
            return self.y <= self.stop
        return False

    def _configure_turn_gate_and_path(self):
        m = self.config.mid[self.direction]
        cx, cy = float(m['x']), float(m['y'])

        if self.direction == 'right':
            if self.lane == 1:
                enter_x = self.config.stopLines['right'] + 40
                self.turn_start_condition = ("x_gt", float(enter_x))
                pts = [
                    (cx, float(self.y)),
                    (cx, 250.0),
                    (cx, 160.0)
                ]
            else:
                enter_x = float(self.config.mid['right']['x'])
                self.turn_start_condition = ("x_gt", enter_x)
                pts = [
                    (cx, float(self.y)),
                    (cx, 650.0),
                    (cx, 760.0)
                ]

        elif self.direction == 'down':
            if self.lane == 1:
                enter_y = self.config.stopLines['down'] + 50
                self.turn_start_condition = ("y_gt", float(enter_y))
                pts = [
                    (float(self.x), cy),
                    (900.0, cy),
                    (1050.0, cy)
                ]
            else:
                enter_y = float(self.config.mid['down']['y'])
                self.turn_start_condition = ("y_gt", enter_y)
                pts = [
                    (float(self.x), cy),
                    (450.0, cy),
                    (300.0, cy)
                ]

        elif self.direction == 'left':
            if self.lane == 1:
                enter_x = self.config.stopLines['left'] - 70
                self.turn_start_condition = ("x_lt", float(enter_x))
                pts = [
                    (cx, float(self.y)),
                    (cx, 650.0),
                    (cx, 760.0)
                ]
            else:
                enter_x = float(self.config.mid['left']['x'])
                self.turn_start_condition = ("x_lt", enter_x)
                pts = [
                    (cx, float(self.y)),
                    (cx, 250.0),
                    (cx, 160.0)
                ]

        elif self.direction == 'up':
            if self.lane == 1:
                enter_y = self.config.stopLines['up'] - 60
                self.turn_start_condition = ("y_lt", float(enter_y))
                pts = [
                    (float(self.x), cy),
                    (450.0, cy),
                    (300.0, cy)
                ]
            else:
                enter_y = float(self.config.mid['up']['y'])
                self.turn_start_condition = ("y_lt", enter_y)
                pts = [
                    (float(self.x), cy),
                    (900.0, cy),
                    (1050.0, cy)
                ]

        else:
            self.turn_start_condition = None
            pts = []

        self._turn_points_template = [(float(px), float(py)) for (px, py) in pts]

    def _should_enter_turn(self) -> bool:
        if self.willTurn != 1:
            return False
        if self.in_turn_path:
            return False
        if self.turn_start_condition is None:
            return False
        kind, threshold = self.turn_start_condition
        if kind == "x_gt":
            return (self.x + self.image.get_rect().width) >= threshold
        if kind == "x_lt":
            return self.x <= threshold
        if kind == "y_gt":
            return (self.y + self.image.get_rect().height) >= threshold
        if kind == "y_lt":
            return self.y <= threshold
        return False

    def _enter_turn_path(self):
        self.in_turn_path = True
        self.turn_progress = 0.0
        start = (float(self.x), float(self.y))
        pts = [start] + self._turn_points_template
        self.turn_path = Path(pts)

    def _mark_crossed_if_needed(self, now_s: float):
        if self.has_crossed:
            return
        sl = self.config.stopLines[self.direction]
        if self.direction == 'right' and (self.x + self.image.get_rect().width) > sl:
            self.has_crossed = True
            self.world.record_crossed(self.direction, now_s)
        elif self.direction == 'left' and self.x < sl:
            self.has_crossed = True
            self.world.record_crossed(self.direction, now_s)
        elif self.direction == 'down' and (self.y + self.image.get_rect().height) > sl:
            self.has_crossed = True
            self.world.record_crossed(self.direction, now_s)
        elif self.direction == 'up' and self.y < sl:
            self.has_crossed = True
            self.world.record_crossed(self.direction, now_s)

    def prepare_motion(self, dt: float):
        self.pending_dt = dt
        self.pending_kind = None
        self.pending_amount = 0.0

        # Decide whether to initiate turn this tick (based on current position)
        if self._should_enter_turn():
            self._enter_turn_path()

        # Determine intended motion
        if self.in_turn_path and self.turn_path is not None:
            # Path motion: always attempt to advance along path
            self.pending_kind = "path"
            self.pending_amount = self.speed * dt
            return

        # Lane motion: obey signals and stop behavior BEFORE crossing stop line
        # The collision manager will further clamp by front + broad collisions.
        if (not self.has_crossed) and self._is_before_stop_line():
            if not self._signal_allows():
                # if at stop point, don't move
                if self._at_or_past_stop_point():
                    return

        self.pending_kind = "lane"
        self.pending_amount = self.speed * dt

    def proposed_state_for_fraction(self, fraction: float) -> Tuple[float, float, pygame.Surface]:
        # Return (x, y, image) if motion applied partially
        fraction = clamp(fraction, 0.0, 1.0)
        if self.pending_kind is None or self.pending_amount <= 0.0 or fraction <= 0.0:
            return (self.x, self.y, self.image)

        if self.pending_kind == "lane":
            step = self.pending_amount * fraction
            nx, ny = self.x, self.y
            if self.direction == 'right':
                nx += step
            elif self.direction == 'left':
                nx -= step
            elif self.direction == 'down':
                ny += step
            elif self.direction == 'up':
                ny -= step
            return (nx, ny, self.image)

        if self.pending_kind == "path":
            if self.turn_path is None:
                return (self.x, self.y, self.image)
            # sample at hypothetical progress
            p = self.turn_progress + (self.pending_amount * fraction)
            if p >= self.turn_path.length:
                end = self.turn_path.sample(self.turn_path.length)
                return (end[0], end[1], self.originalImage)
            pos = self.turn_path.sample(p)
            heading = self.turn_path.heading(p)
            img = pygame.transform.rotate(self.originalImage, heading)
            return (pos[0], pos[1], img)

        return (self.x, self.y, self.image)

    def proposed_rect_for_fraction(self, fraction: float) -> pygame.Rect:
        nx, ny, img = self.proposed_state_for_fraction(fraction)
        r = img.get_rect()
        r.topleft = (int(nx), int(ny))
        return r

    def apply_motion_fraction(self, fraction: float):
        fraction = clamp(fraction, 0.0, 1.0)
        if self.pending_kind is None or self.pending_amount <= 0.0 or fraction <= 0.0:
            return

        if self.pending_kind == "lane":
            step = self.pending_amount * fraction
            if self.direction == 'right':
                self.x += step
            elif self.direction == 'left':
                self.x -= step
            elif self.direction == 'down':
                self.y += step
            elif self.direction == 'up':
                self.y -= step
            return

        if self.pending_kind == "path":
            if self.turn_path is None:
                return
            self.turn_progress += self.pending_amount * fraction
            if self.turn_progress >= self.turn_path.length:
                end = self.turn_path.sample(self.turn_path.length)
                self.x, self.y = end[0], end[1]
                self.in_turn_path = False
                self.turn_path = None
                self.image = self.originalImage
                return
            pos = self.turn_path.sample(self.turn_progress)
            self.x, self.y = pos[0], pos[1]
            heading = self.turn_path.heading(self.turn_progress)
            self.image = pygame.transform.rotate(self.originalImage, heading)
            return

    def post_motion(self, now_s: float):
        # mark crossed for stats
        self._mark_crossed_if_needed(now_s)

        # cleanup if offscreen
        if (self.x < -400) or (self.x > self.config.screenWidth + 400) or (self.y < -400) or (self.y > self.config.screenHeight + 400):
            self.alive = False
            self.world.unregister_vehicle(self)

    def render(self, screen: pygame.Surface):
        screen.blit(self.image, (self.x, self.y))

# -----------------------------------------------------------------------------
#  VehicleFactory
# -----------------------------------------------------------------------------

class VehicleFactory:
    def __init__(self, assets: AssetManager, world: WorldState):
        self.assets = assets
        self.world = world
        self.vehicleTypesByIndex = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}

    def create(self, vehicle_type_index: int, lane: int, direction_number: int, will_turn: int):
        vehicleClass = self.vehicleTypesByIndex[vehicle_type_index]
        direction = self.world.directionNumbers[direction_number]
        Vehicle(self.assets, self.world, vehicleClass, direction_number, direction, lane, will_turn)

# -----------------------------------------------------------------------------
#  Spawner with queue + spawn clearance
# -----------------------------------------------------------------------------

class SpawnRequest:
    def __init__(self, vehicle_type_index: int, lane: int, direction_number: int, will_turn: int):
        self.vehicle_type_index = vehicle_type_index
        self.lane = lane
        self.direction_number = direction_number
        self.will_turn = will_turn

class Spawner:
    def __init__(self, config: SimulationConfig, world: WorldState, factory: VehicleFactory, assets: AssetManager):
        self.config = config
        self.world = world
        self.factory = factory
        self.assets = assets

        self.spawn_queue: "queue.Queue[SpawnRequest]" = queue.Queue()

        self.allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}
        self.allowedVehicleTypesList = []
        i = 0
        for vehicleType in ['car', 'bus', 'truck', 'bike']:
            if self.allowedVehicleTypes[vehicleType]:
                self.allowedVehicleTypesList.append(i)
            i += 1

        self._producer_running = True
        self._producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self._producer_thread.start()

    def stop(self):
        self._producer_running = False

    def _producer_loop(self):
        while self._producer_running:
            req = self._make_request()
            self.spawn_queue.put(req)
            time.sleep(max(0.01, self.config.spawnIntervalSec))

    def _make_request(self) -> SpawnRequest:
        vehicle_type_index = random.choice(self.allowedVehicleTypesList)
        lane_number = random.randint(1, 2)

        will_turn = 0
        temp = random.randint(0, 99)
        if lane_number == 1:
            if temp < int(self.config.turnProbabilityLane1 * 100):
                will_turn = 1
        else:
            if temp < int(self.config.turnProbabilityLane2 * 100):
                will_turn = 1

        temp = random.randint(0, 99)
        direction_number = 0
        dist = self.config.directionDistribution
        if temp < dist[0]:
            direction_number = 0
        elif temp < dist[1]:
            direction_number = 1
        elif temp < dist[2]:
            direction_number = 2
        else:
            direction_number = 3

        return SpawnRequest(vehicle_type_index, lane_number, direction_number, will_turn)

    def _spawn_area_clear(self, direction: str, lane: int, vehicle_img: pygame.Surface) -> bool:
        sx = self.config.spawnX[direction][lane]
        sy = self.config.spawnY[direction][lane]
        w = vehicle_img.get_rect().width
        h = vehicle_img.get_rect().height

        buffer = self.config.stoppingGap + 20
        spawn_rect = pygame.Rect(sx - buffer, sy - buffer, w + 2*buffer, h + 2*buffer)

        for v in self.world.lane_list(direction, lane):
            if spawn_rect.colliderect(v.rect()):
                return False
        return True

    def consume_and_spawn(self, max_per_tick: int = 3):
        count = 0
        while count < max_per_tick:
            try:
                req = self.spawn_queue.get_nowait()
            except queue.Empty:
                break

            direction = self.world.directionNumbers[req.direction_number]
            vehicleClass = self.factory.vehicleTypesByIndex[req.vehicle_type_index]
            img_path = self.config.vehicleImageTemplate.format(direction=direction, vehicleClass=vehicleClass)
            img = self.assets.image(img_path)

            if self._spawn_area_clear(direction, req.lane, img):
                self.factory.create(req.vehicle_type_index, req.lane, req.direction_number, req.will_turn)
            else:
                # keep trying later to avoid "burst overlap" under heavy queue
                if random.random() < 0.50:
                    self.spawn_queue.put(req)

            count += 1

# -----------------------------------------------------------------------------
#  Simulation Engine
# -----------------------------------------------------------------------------

class SimulationEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config

        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.config.screenWidth, self.config.screenHeight))
        pygame.display.set_caption("SIMULATION")

        self.assets = AssetManager(self.config)
        self.assets.validate_or_die(required_vehicle_types=['car','bus','truck','bike'], directions=['right','down','left','up'])

        self.background = self.assets.image(self.config.backgroundPath)
        self.redSignal = self.assets.image(self.config.redSignalPath)
        self.yellowSignal = self.assets.image(self.config.yellowSignalPath)
        self.greenSignal = self.assets.image(self.config.greenSignalPath)

        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 22)

        self.signalController = SignalController(self.config)
        self.world = WorldState(self.config, self.signalController)
        self.factory = VehicleFactory(self.assets, self.world)
        self.spawner = Spawner(self.config, self.world, self.factory, self.assets)

        self.collisionManager = CollisionManager(self.world)

        self.running = True
        self.paused = False
        self.sim_time = 0.0

        self._last_stats_sample = 0.0
        self.show_overlay = True

    def reset(self):
        logging.info("Resetting simulation...")
        self.spawner.stop()

        self.signalController = SignalController(self.config)
        self.world = WorldState(self.config, self.signalController)
        self.factory = VehicleFactory(self.assets, self.world)
        self.spawner = Spawner(self.config, self.world, self.factory, self.assets)
        self.collisionManager = CollisionManager(self.world)

        self.sim_time = 0.0
        self.paused = False

    def reload_config(self):
        logging.info("Reloading config from config.json...")
        self.config = load_config_from_file("config.json", self.config)

        self.assets = AssetManager(self.config)
        self.assets.validate_or_die(required_vehicle_types=['car','bus','truck','bike'], directions=['right','down','left','up'])

        self.background = self.assets.image(self.config.backgroundPath)
        self.redSignal = self.assets.image(self.config.redSignalPath)
        self.yellowSignal = self.assets.image(self.config.yellowSignalPath)
        self.greenSignal = self.assets.image(self.config.greenSignalPath)

        self.reset()

    def _write_stats_csv(self):
        try:
            with open("stats.csv", "w", encoding="utf-8") as f:
                f.write("metric,value\n")
                total = sum(self.world.crossed.values())
                f.write(f"total_vehicles,{total}\n")
                f.write(f"time_seconds,{int(self.sim_time)}\n")
                for d in ['right','down','left','up']:
                    f.write(f"crossed_{d},{self.world.crossed[d]}\n")
                for d in ['right','down','left','up']:
                    samples = self.world.queue_len_samples[d]
                    avgq = (sum(samples)/len(samples)) if samples else 0.0
                    f.write(f"avg_queue_{d},{avgq:.3f}\n")
            logging.info("Wrote stats.csv")
        except Exception as e:
            logging.error("Failed writing stats.csv: %s", str(e))

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    logging.info("Paused=%s", self.paused)
                if event.key == pygame.K_r:
                    self.reset()
                if event.key == pygame.K_c:
                    self.reload_config()
                if event.key == pygame.K_o:
                    self.show_overlay = not self.show_overlay

    def _draw_signals(self):
        currentGreen, currentYellow = self.signalController.get_state()
        for i in range(4):
            if i == currentGreen:
                if currentYellow == 1:
                    self.screen.blit(self.yellowSignal, self.config.signalCoods[i])
                else:
                    self.screen.blit(self.greenSignal, self.config.signalCoods[i])
            else:
                self.screen.blit(self.redSignal, self.config.signalCoods[i])

        for i in range(4):
            text = self.signalController.display_value_for_signal(i)
            surf = self.font.render(text, True, (255,255,255), (0,0,0))
            self.screen.blit(surf, self.config.signalTimerCoods[i])

    def _draw_counts_and_time(self, fps: float):
        dirs = {0:'right', 1:'down', 2:'left', 3:'up'}
        for i in range(4):
            d = dirs[i]
            cnt = self.world.crossed[d]
            surf = self.font.render(str(cnt), True, (0,0,0), (255,255,255))
            self.screen.blit(surf, self.config.vehicleCountCoods[i])

        timeElapsedText = self.font.render(("Time Elapsed: " + str(int(self.sim_time))), True, (0,0,0), (255,255,255))
        self.screen.blit(timeElapsedText, self.config.timeElapsedCoods)

        if self.show_overlay:
            total = sum(self.world.crossed.values())
            overlay_lines = [
                f"FPS: {fps:.1f}",
                f"Total Passed: {total}",
                f"SpawnQ: {self.spawner.spawn_queue.qsize()}",
                f"Paused: {self.paused}",
                f"IntersectionOwner: {self.world.intersection_owner_id}",
                "Keys: SPACE pause | R reset | C reload config.json | O overlay | ESC quit",
            ]
            y = 10
            for line in overlay_lines:
                s = self.small_font.render(line, True, (0,0,0), (255,255,255))
                self.screen.blit(s, (10, y))
                y += 20

            now_s = self.sim_time
            recent = [t for (t, _) in self.world.throughput_events if (now_s - t) <= 60.0]
            tpm = len(recent)
            s = self.small_font.render(f"Throughput (last 60s): {tpm} vehicles/min", True, (0,0,0), (255,255,255))
            self.screen.blit(s, (10, y))

    def run(self):
        logging.info("Simulation started.")
        try:
            while self.running:
                self._handle_events()

                dt_ms = self.clock.tick(self.config.fps)
                dt = dt_ms / 1000.0
                fps = self.clock.get_fps()

                if not self.paused:
                    # signals tick
                    self.sim_time += dt
                    self.signalController.tick(dt)

                    # spawn vehicles (main thread only)
                    self.spawner.consume_and_spawn(max_per_tick=3)

                    # strong collision-aware update (replaces per-vehicle naive move)
                    self.collisionManager.resolve_all(dt, self.sim_time)

                    if (self.sim_time - self._last_stats_sample) >= 1.0:
                        self.world.sample_queue_lengths()
                        self._last_stats_sample = self.sim_time

                    if int(self.sim_time) >= int(self.config.simulationTime):
                        logging.info("Simulation time reached: %s seconds", self.config.simulationTime)
                        self.running = False

                self.screen.blit(self.background, (0, 0))

                # draw intersection conflict rect in overlay mode (debug)
                if self.show_overlay:
                    pygame.draw.rect(self.screen, (255, 0, 0), self.world.intersection_rect, 2)

                self._draw_signals()

                for v in self.world.simulation:
                    v.render(self.screen)

                self._draw_counts_and_time(fps)
                pygame.display.update()

        finally:
            self.spawner.stop()
            self._write_stats_csv()
            pygame.quit()

            totalVehicles = sum(self.world.crossed.values())
            logging.info("Direction-wise Vehicle Counts:")
            logging.info("Right: %d | Down: %d | Left: %d | Up: %d",
                         self.world.crossed['right'], self.world.crossed['down'], self.world.crossed['left'], self.world.crossed['up'])
            logging.info("Total vehicles passed: %d", totalVehicles)
            logging.info("Total time: %d", int(self.sim_time))

# -----------------------------------------------------------------------------
#  Entry
# -----------------------------------------------------------------------------

base_config = SimulationConfig()
config = load_config_from_file("config.json", base_config)

engine = SimulationEngine(config)
engine.run()
