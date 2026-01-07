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
#  Logging (improvement #10)
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
#  Config + safe reload (improvement #8, #9)
# -----------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    screenWidth: int = 1400
    screenHeight: int = 800
    fps: int = 60
    simulationTime: int = 300

    # signal timing
    defaultYellow: int = 5
    defaultRedBuffer: int = 0  # optional all-red between phases (seconds)
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

    # spawn points (constant; spawner checks space, no drifting coords)
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

    # speeds in px/sec (dt-based; improvement #6)
    speedsPxPerSec: Dict[str, float] = None

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
            self.directionDistribution = [25, 50, 75, 100]  # right, down, left, up

        if self.speedsPxPerSec is None:
            # Original was ~2 px/frame at variable FPS. Use stable px/sec.
            # 2.25 px/frame @ 60 fps ~ 135 px/sec (car)
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
#  Assets validation (improvement #9)
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
#  Signals: event-driven tick-based controller (improvement #5)
# -----------------------------------------------------------------------------

class SignalPhase:
    def __init__(self, green_dir_index: int, green_sec: int, yellow_sec: int, all_red_sec: int):
        self.green_dir_index = green_dir_index
        self.green_sec = green_sec
        self.yellow_sec = yellow_sec
        self.all_red_sec = all_red_sec

class SignalController:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.noOfSignals = 4
        self.currentGreen = 0
        self.currentYellow = 0
        self.time_in_phase = 0.0

        self._phase_state = "GREEN"  # GREEN, YELLOW, ALL_RED
        self._green_remaining = 0
        self._yellow_remaining = 0
        self._all_red_remaining = 0

        self._set_new_green(self.currentGreen)

    def _pick_green_time(self, idx: int) -> int:
        if self.config.randomGreenSignalTimer:
            return random.randint(self.config.randomGreenSignalTimerRange[0], self.config.randomGreenSignalTimerRange[1])
        return self.config.defaultGreen[idx]

    def _set_new_green(self, idx: int):
        self.currentGreen = idx
        self.currentYellow = 0
        self._phase_state = "GREEN"
        self._green_remaining = self._pick_green_time(idx)
        self._yellow_remaining = self.config.defaultYellow
        self._all_red_remaining = self.config.defaultRedBuffer
        self.time_in_phase = 0.0

    def tick(self, dt: float):
        if dt <= 0:
            return

        self.time_in_phase += dt

        if self._phase_state == "GREEN":
            self._green_remaining -= dt
            if self._green_remaining <= 0:
                self._phase_state = "YELLOW"
                self.currentYellow = 1
                self.time_in_phase = 0.0

        elif self._phase_state == "YELLOW":
            self._yellow_remaining -= dt
            if self._yellow_remaining <= 0:
                self.currentYellow = 0
                if self._all_red_remaining > 0:
                    self._phase_state = "ALL_RED"
                else:
                    self._advance_to_next_green()
                self.time_in_phase = 0.0

        elif self._phase_state == "ALL_RED":
            self._all_red_remaining -= dt
            if self._all_red_remaining <= 0:
                self._advance_to_next_green()
                self.time_in_phase = 0.0

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
        # show remaining red only when close (like original), but red is implicit now
        # approximate "red remaining" as total time until that signal gets green again:
        # sum of remaining in current phase + phases of others.
        # For UI simplicity (no dead complexity), keep "---" always for red.
        return "---"

# -----------------------------------------------------------------------------
#  Paths + movement strategies (improvement #1, #2, #3)
# -----------------------------------------------------------------------------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0]-a[0], b[1]-a[1])

def angle_deg_from_vec(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(-dy, dx))

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
        # find segment
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
        # small lookahead to compute heading
        eps = 5.0
        p1 = self.sample(s)
        p2 = self.sample(clamp(s+eps, 0.0, self.length))
        dx = p2[0]-p1[0]
        dy = p2[1]-p1[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        return angle_deg_from_vec(dx, dy)

class MovementStrategy:
    def step(self, vehicle, dt: float):
        raise NotImplementedError

class StraightMovement(MovementStrategy):
    def step(self, vehicle, dt: float):
        vehicle._advance_along_lane(dt)

class TurningMovement(MovementStrategy):
    def step(self, vehicle, dt: float):
        # while not entered path, still advance along lane until "turn start"
        if not vehicle.in_turn_path:
            vehicle._advance_along_lane(dt)
            if vehicle._should_enter_turn():
                vehicle._enter_turn_path()
        else:
            vehicle._advance_along_path(dt)

# -----------------------------------------------------------------------------
#  World state (encapsulation) + lane management (improvement #3, #7)
# -----------------------------------------------------------------------------

class WorldState:
    def __init__(self, config: SimulationConfig, signalController: SignalController):
        self.config = config
        self.signalController = signalController

        self.directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
        self.vehicleTypes = ['car', 'bus', 'truck', 'bike']

        # lane registry
        self.vehicles: Dict[str, Dict[int, List["Vehicle"]]] = {
            'right': {1: [], 2: []},
            'down': {1: [], 2: []},
            'left': {1: [], 2: []},
            'up': {1: [], 2: []},
        }

        # crossed counts by direction
        self.crossed = {'right': 0, 'down': 0, 'left': 0, 'up': 0}

        # pygame group
        self.simulation = pygame.sprite.Group()

        # stats sampling
        self.queue_len_samples = {'right': [], 'down': [], 'left': [], 'up': []}
        self.throughput_events: List[Tuple[float, str]] = []  # (time, direction)

    def lane_list(self, direction: str, lane: int) -> List["Vehicle"]:
        return self.vehicles[direction][lane]

    def register_vehicle(self, v: "Vehicle"):
        self.vehicles[v.direction][v.lane].append(v)
        self.simulation.add(v)

    def unregister_vehicle(self, v: "Vehicle"):
        # safe remove if present
        lane = self.vehicles[v.direction][v.lane]
        if v in lane:
            lane.remove(v)
        if v in self.simulation:
            self.simulation.remove(v)

    def record_crossed(self, direction: str, now_s: float):
        self.crossed[direction] += 1
        self.throughput_events.append((now_s, direction))

    def sample_queue_lengths(self):
        # define queue length as vehicles that have not crossed stop line
        for d in ['right', 'down', 'left', 'up']:
            q = 0
            for lane in [1, 2]:
                for v in self.vehicles[d][lane]:
                    if not v.has_crossed:
                        q += 1
            self.queue_len_samples[d].append(q)

# -----------------------------------------------------------------------------
#  Vehicle (inheritance + polymorphism via subclasses & strategy) (improvement #1)
# -----------------------------------------------------------------------------

class Vehicle(pygame.sprite.Sprite):
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

        self.vehicleClass = vehicleClass
        self.direction_number = direction_number
        self.direction = direction
        self.lane = lane
        self.willTurn = will_turn

        self.speed = self.config.speedsPxPerSec[vehicleClass]

        # images
        path = self.config.vehicleImageTemplate.format(direction=direction, vehicleClass=vehicleClass)
        self.originalImage = self.assets.image(path)
        self.image = self.originalImage

        # position
        sx = self.config.spawnX[direction][lane]
        sy = self.config.spawnY[direction][lane]
        self.x = float(sx)
        self.y = float(sy)

        # state
        self.has_crossed = False
        self.in_turn_path = False
        self.turn_progress = 0.0
        self.turn_path: Optional[Path] = None
        self.turn_start_condition = None  # set below
        self._rotation = 0.0

        # movement strategy
        self.mover: MovementStrategy = TurningMovement() if self.willTurn == 1 else StraightMovement()

        # compute stop position based on front vehicle (centralized spacing)
        self.stop = self._compute_initial_stop()

        # determine when to enter turn path
        self._configure_turn_gate_and_path()

        # register
        self.world.register_vehicle(self)

    def rect(self) -> pygame.Rect:
        r = self.image.get_rect()
        r.topleft = (int(self.x), int(self.y))
        return r

    def length_along_lane(self) -> float:
        # used for ordering and spacing (higher means further along movement direction)
        if self.direction == 'right':
            return self.x
        if self.direction == 'left':
            return -self.x
        if self.direction == 'down':
            return self.y
        if self.direction == 'up':
            return -self.y
        return 0.0

    def _front_vehicle(self) -> Optional["Vehicle"]:
        lane = self.world.lane_list(self.direction, self.lane)
        # sort by progress
        ordered = sorted(lane, key=lambda v: v.length_along_lane())
        # find self in ordered list
        try:
            idx = ordered.index(self)
        except ValueError:
            return None
        if idx == 0:
            return None
        return ordered[idx-1]

    def _compute_initial_stop(self) -> float:
        front = self._front_vehicle()
        if front is None or front.has_crossed:
            return float(self.config.defaultStop[self.direction])

        fg = float(self.config.stoppingGap)
        if self.direction == 'right':
            return float(front.stop - front.image.get_rect().width - fg)
        if self.direction == 'left':
            return float(front.stop + front.image.get_rect().width + fg)
        if self.direction == 'down':
            return float(front.stop - front.image.get_rect().height - fg)
        if self.direction == 'up':
            return float(front.stop + front.image.get_rect().height + fg)
        return float(self.config.defaultStop[self.direction])

    def _gap_ok(self) -> bool:
        front = self._front_vehicle()
        if front is None:
            return True

        mg = float(self.config.movingGap)
        if self.direction == 'right':
            return (self.x + self.image.get_rect().width) < (front.x - mg) or front.in_turn_path
        if self.direction == 'left':
            return self.x > (front.x + front.image.get_rect().width + mg) or front.in_turn_path
        if self.direction == 'down':
            return (self.y + self.image.get_rect().height) < (front.y - mg) or front.in_turn_path
        if self.direction == 'up':
            return self.y > (front.y + front.image.get_rect().height + mg) or front.in_turn_path
        return True

    def _signal_allows(self) -> bool:
        currentGreen, currentYellow = self.world.signalController.get_state()
        # mapping: 0 right, 1 down, 2 left, 3 up
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

    def _configure_turn_gate_and_path(self):
        # Parametric pathing using waypoints (improvement #2).
        # Keep it simple and robust: enter path near "mid" anchors.
        m = self.config.mid[self.direction]
        cx, cy = float(m['x']), float(m['y'])

        # Define exit points for turning (approximate based on existing visuals).
        # These are tuned to the provided intersection image coordinates.
        # Lane 1 and lane 2 differ slightly to mimic original behavior.
        if self.direction == 'right':
            # right -> up (lane1) or right -> down (lane2)
            if self.lane == 1:
                enter_x = self.config.stopLines['right'] + 40
                self.turn_start_condition = ("x_gt", float(enter_x))
                points = [
                    (self.x, self.y),
                    (cx, float(self.y)),
                    (cx, 250.0),
                    (cx, 160.0)
                ]
            else:
                enter_x = float(self.config.mid['right']['x'])
                self.turn_start_condition = ("x_gt", enter_x)
                points = [
                    (self.x, self.y),
                    (cx, float(self.y)),
                    (cx, 650.0),
                    (cx, 760.0)
                ]

        elif self.direction == 'down':
            # down -> right (lane1) or down -> left (lane2)
            if self.lane == 1:
                enter_y = self.config.stopLines['down'] + 50
                self.turn_start_condition = ("y_gt", float(enter_y))
                points = [
                    (self.x, self.y),
                    (float(self.x), cy),
                    (900.0, cy),
                    (1050.0, cy)
                ]
            else:
                enter_y = float(self.config.mid['down']['y'])
                self.turn_start_condition = ("y_gt", enter_y)
                points = [
                    (self.x, self.y),
                    (float(self.x), cy),
                    (450.0, cy),
                    (300.0, cy)
                ]

        elif self.direction == 'left':
            # left -> down (lane1) or left -> up (lane2)
            if self.lane == 1:
                enter_x = self.config.stopLines['left'] - 70
                self.turn_start_condition = ("x_lt", float(enter_x))
                points = [
                    (self.x, self.y),
                    (cx, float(self.y)),
                    (cx, 650.0),
                    (cx, 760.0)
                ]
            else:
                enter_x = float(self.config.mid['left']['x'])
                self.turn_start_condition = ("x_lt", enter_x)
                points = [
                    (self.x, self.y),
                    (cx, float(self.y)),
                    (cx, 250.0),
                    (cx, 160.0)
                ]

        elif self.direction == 'up':
            # up -> left (lane1) or up -> right (lane2)
            if self.lane == 1:
                enter_y = self.config.stopLines['up'] - 60
                self.turn_start_condition = ("y_lt", float(enter_y))
                points = [
                    (self.x, self.y),
                    (float(self.x), cy),
                    (450.0, cy),
                    (300.0, cy)
                ]
            else:
                enter_y = float(self.config.mid['up']['y'])
                self.turn_start_condition = ("y_lt", enter_y)
                points = [
                    (self.x, self.y),
                    (float(self.x), cy),
                    (900.0, cy),
                    (1050.0, cy)
                ]

        else:
            self.turn_start_condition = None
            points = [(self.x, self.y)]

        # path will be rebuilt at entry with current start; store template waypoints beyond start
        self._turn_points_template = points[1:] if len(points) > 1 else [(self.x, self.y)]

    def _should_enter_turn(self) -> bool:
        if self.turn_start_condition is None:
            return False
        kind, threshold = self.turn_start_condition
        if kind == "x_gt":
            return self.x + self.image.get_rect().width >= threshold
        if kind == "x_lt":
            return self.x <= threshold
        if kind == "y_gt":
            return self.y + self.image.get_rect().height >= threshold
        if kind == "y_lt":
            return self.y <= threshold
        return False

    def _enter_turn_path(self):
        self.in_turn_path = True
        self.turn_progress = 0.0
        start = (float(self.x), float(self.y))
        points = [start] + [(float(px), float(py)) for (px, py) in self._turn_points_template]
        self.turn_path = Path(points)

    def _advance_along_lane(self, dt: float):
        # obey signals and spacing before crossing stop line
        if (not self.has_crossed) and self._is_before_stop_line():
            if (not self._signal_allows()):
                # can still roll up to stop point if spaced ok
                if not self._gap_ok():
                    return
                # move until stop coordinate
                if self.direction == 'right':
                    if (self.x + self.image.get_rect().width) >= self.stop:
                        return
                elif self.direction == 'left':
                    if self.x <= self.stop:
                        return
                elif self.direction == 'down':
                    if (self.y + self.image.get_rect().height) >= self.stop:
                        return
                elif self.direction == 'up':
                    if self.y <= self.stop:
                        return
            else:
                if not self._gap_ok():
                    return

        # move
        step = self.speed * dt
        if self.direction == 'right':
            self.x += step
        elif self.direction == 'left':
            self.x -= step
        elif self.direction == 'down':
            self.y += step
        elif self.direction == 'up':
            self.y -= step

    def _advance_along_path(self, dt: float):
        if self.turn_path is None:
            self.in_turn_path = False
            return
        self.turn_progress += self.speed * dt
        if self.turn_progress >= self.turn_path.length:
            end = self.turn_path.sample(self.turn_path.length)
            self.x, self.y = end[0], end[1]
            self.in_turn_path = False
            self.turn_path = None
            self.image = self.originalImage
            return
        p = self.turn_path.sample(self.turn_progress)
        self.x, self.y = p[0], p[1]

        # rotate sprite to face heading
        heading = self.turn_path.heading(self.turn_progress)
        # pygame rotates counterclockwise, and our heading uses screen coords (y down),
        # this works well enough for visual alignment.
        self.image = pygame.transform.rotate(self.originalImage, heading)

    def update(self, dt: float, now_s: float):
        self.mover.step(self, dt)
        self._mark_crossed_if_needed(now_s)

        # cleanup if offscreen
        if (self.x < -300) or (self.x > self.config.screenWidth + 300) or (self.y < -300) or (self.y > self.config.screenHeight + 300):
            self.world.unregister_vehicle(self)

    def render(self, screen: pygame.Surface):
        screen.blit(self.image, (self.x, self.y))

# -----------------------------------------------------------------------------
#  VehicleFactory (improvement #1 without dead code)
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
#  Spawner with queue + optional generator thread (improvement #4, #7)
# -----------------------------------------------------------------------------

class SpawnRequest:
    def __init__(self, vehicle_type_index: int, lane: int, direction_number: int, will_turn: int):
        self.vehicle_type_index = vehicle_type_index
        self.lane = lane
        self.direction_number = direction_number
        self.will_turn = will_turn

class Spawner:
    def __init__(self, config: SimulationConfig, world: WorldState, factory: VehicleFactory):
        self.config = config
        self.world = world
        self.factory = factory
        self.spawn_queue: "queue.Queue[SpawnRequest]" = queue.Queue()
        self._spawn_accum = 0.0

        self.allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}
        self.allowedVehicleTypesList = []
        i = 0
        for vehicleType in ['car', 'bus', 'truck', 'bike']:
            if self.allowedVehicleTypes[vehicleType]:
                self.allowedVehicleTypesList.append(i)
            i += 1

        # Optional separate producer thread (kept minimal, safe via queue)
        self._producer_running = True
        self._producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self._producer_thread.start()

    def stop(self):
        self._producer_running = False

    def _producer_loop(self):
        # produces at approximate interval; actual spawn happens in main thread
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
        # checks if any vehicle overlaps a spawn "buffer" rectangle
        sx = self.config.spawnX[direction][lane]
        sy = self.config.spawnY[direction][lane]
        w = vehicle_img.get_rect().width
        h = vehicle_img.get_rect().height

        buffer = self.config.stoppingGap + 10
        spawn_rect = pygame.Rect(sx - buffer, sy - buffer, w + 2*buffer, h + 2*buffer)

        for v in self.world.lane_list(direction, lane):
            if spawn_rect.colliderect(v.rect()):
                return False
        return True

    def consume_and_spawn(self, assets: AssetManager, max_per_tick: int = 3):
        # consume up to N requests per frame; spawn if safe, else drop (or requeue lightly)
        count = 0
        while count < max_per_tick:
            try:
                req = self.spawn_queue.get_nowait()
            except queue.Empty:
                break

            direction = self.world.directionNumbers[req.direction_number]
            vehicleClass = self.factory.vehicleTypesByIndex[req.vehicle_type_index]
            img_path = self.config.vehicleImageTemplate.format(direction=direction, vehicleClass=vehicleClass)
            img = assets.image(img_path)

            if self._spawn_area_clear(direction, req.lane, img):
                self.factory.create(req.vehicle_type_index, req.lane, req.direction_number, req.will_turn)
            else:
                # requeue once with a short delay probability to avoid permanent drop bursts
                if random.random() < 0.25:
                    self.spawn_queue.put(req)

            count += 1

# -----------------------------------------------------------------------------
#  Simulation Engine (improvement #6, #8, #9, #10)
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
        self.spawner = Spawner(self.config, self.world, self.factory)

        self.running = True
        self.paused = False
        self.start_time_real = time.time()
        self.sim_time = 0.0

        self._last_stats_sample = 0.0

        # overlay toggles
        self.show_overlay = True

    def reset(self):
        logging.info("Resetting simulation...")
        # stop spawner producer cleanly
        self.spawner.stop()

        # rebuild everything with current config
        self.signalController = SignalController(self.config)
        self.world = WorldState(self.config, self.signalController)
        self.factory = VehicleFactory(self.assets, self.world)
        self.spawner = Spawner(self.config, self.world, self.factory)

        self.sim_time = 0.0
        self.start_time_real = time.time()
        self.paused = False

    def reload_config(self):
        logging.info("Reloading config from config.json...")
        self.config = load_config_from_file("config.json", self.config)

        # refresh dependent objects
        self.assets = AssetManager(self.config)
        self.assets.validate_or_die(required_vehicle_types=['car','bus','truck','bike'], directions=['right','down','left','up'])

        self.background = self.assets.image(self.config.backgroundPath)
        self.redSignal = self.assets.image(self.config.redSignalPath)
        self.yellowSignal = self.assets.image(self.config.yellowSignalPath)
        self.greenSignal = self.assets.image(self.config.greenSignalPath)

        # rebuild controller/world/spawner with new config
        self.reset()

    def _write_stats_csv(self):
        # improvement #10: save summary + timeseries
        try:
            with open("stats.csv", "w", encoding="utf-8") as f:
                f.write("metric,value\n")
                total = sum(self.world.crossed.values())
                f.write(f"total_vehicles,{total}\n")
                f.write(f"time_seconds,{int(self.sim_time)}\n")
                for d in ['right','down','left','up']:
                    f.write(f"crossed_{d},{self.world.crossed[d]}\n")

                # avg queue lengths
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

        # timers
        for i in range(4):
            text = self.signalController.display_value_for_signal(i)
            surf = self.font.render(text, True, (255,255,255), (0,0,0))
            self.screen.blit(surf, self.config.signalTimerCoods[i])

    def _draw_counts_and_time(self, fps: float):
        # vehicle count per direction number order: right, down, left, up
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
                "Keys: SPACE pause | R reset | C reload config.json | O overlay | ESC quit",
            ]
            y = 10
            for line in overlay_lines:
                s = self.small_font.render(line, True, (0,0,0), (255,255,255))
                self.screen.blit(s, (10, y))
                y += 20

            # throughput last 60s
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
                    self.sim_time += dt
                    self.signalController.tick(dt)

                    # spawn safely in main thread (improvement #7) using queue (improvement #4)
                    self.spawner.consume_and_spawn(self.assets, max_per_tick=3)

                    # update vehicles (dt-based movement; improvement #6)
                    now_s = self.sim_time
                    for v in list(self.world.simulation):
                        v.update(dt, now_s)

                    # sample queues for stats once per second (improvement #8 instrumentation)
                    if (self.sim_time - self._last_stats_sample) >= 1.0:
                        self.world.sample_queue_lengths()
                        self._last_stats_sample = self.sim_time

                    if int(self.sim_time) >= int(self.config.simulationTime):
                        logging.info("Simulation time reached: %s seconds", self.config.simulationTime)
                        self.running = False

                # render
                self.screen.blit(self.background, (0, 0))
                self._draw_signals()
                for v in self.world.simulation:
                    v.render(self.screen)
                self._draw_counts_and_time(fps)
                pygame.display.update()

        finally:
            # graceful shutdown
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
