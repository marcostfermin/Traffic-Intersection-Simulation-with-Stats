import random
import time
import threading
import pygame
import sys
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class SimulationConfig:
    screenWidth: int = 1400
    screenHeight: int = 800
    simulationTime: int = 300

    defaultGreen: dict = None
    defaultRed: int = 150
    defaultYellow: int = 5

    randomGreenSignalTimer: bool = True
    randomGreenSignalTimerRange: tuple = (10, 20)

    signalCoods: tuple = ((530,230),(810,230),(810,570),(530,570))
    signalTimerCoods: tuple = ((530,210),(810,210),(810,550),(530,550))
    vehicleCountCoods: tuple = ((480,210),(880,210),(880,550),(480,550))
    timeElapsedCoods: tuple = (1100,50)

    stopLines: dict = None
    defaultStop: dict = None

    stoppingGap: int = 25
    movingGap: int = 25

    x: dict = None
    y: dict = None

    mid: dict = None
    rotationAngle: int = 3

    def __post_init__(self):
        object.__setattr__(self, "defaultGreen", self.defaultGreen if self.defaultGreen is not None else {0:10, 1:10, 2:10, 3:10})
        object.__setattr__(self, "stopLines", self.stopLines if self.stopLines is not None else {'right': 590, 'down': 330, 'left': 800, 'up': 535})
        object.__setattr__(self, "defaultStop", self.defaultStop if self.defaultStop is not None else {'right': 580, 'down': 320, 'left': 810, 'up': 545})
        object.__setattr__(self, "x", self.x if self.x is not None else {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]})
        object.__setattr__(self, "y", self.y if self.y is not None else {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]})
        object.__setattr__(self, "mid", self.mid if self.mid is not None else {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}})

class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""

class SignalController:
    def __init__(self, config):
        self.config = config
        self.signals = []
        self.noOfSignals = 4
        self.currentGreen = 0
        self.nextGreen = (self.currentGreen+1) % self.noOfSignals
        self.currentYellow = 0
        self._lock = threading.Lock()

    def initialize(self):
        minTime = self.config.randomGreenSignalTimerRange[0]
        maxTime = self.config.randomGreenSignalTimerRange[1]
        if(self.config.randomGreenSignalTimer):
            ts1 = TrafficSignal(0, self.config.defaultYellow, random.randint(minTime,maxTime))
            self.signals.append(ts1)
            ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, self.config.defaultYellow, random.randint(minTime,maxTime))
            self.signals.append(ts2)
            ts3 = TrafficSignal(self.config.defaultRed, self.config.defaultYellow, random.randint(minTime,maxTime))
            self.signals.append(ts3)
            ts4 = TrafficSignal(self.config.defaultRed, self.config.defaultYellow, random.randint(minTime,maxTime))
            self.signals.append(ts4)
        else:
            ts1 = TrafficSignal(0, self.config.defaultYellow, self.config.defaultGreen[0])
            self.signals.append(ts1)
            ts2 = TrafficSignal(ts1.yellow+ts1.green, self.config.defaultYellow, self.config.defaultGreen[1])
            self.signals.append(ts2)
            ts3 = TrafficSignal(self.config.defaultRed, self.config.defaultYellow, self.config.defaultGreen[2])
            self.signals.append(ts3)
            ts4 = TrafficSignal(self.config.defaultRed, self.config.defaultYellow, self.config.defaultGreen[3])
            self.signals.append(ts4)
        self._repeat_loop()

    def _printStatus(self):
        for i in range(0, 4):
            if(self.signals[i] != None):
                if(i==self.currentGreen):
                    if(self.currentYellow==0):
                        print(" GREEN TS",i+1,"-> r:",self.signals[i].red," y:",self.signals[i].yellow," g:",self.signals[i].green)
                    else:
                        print("YELLOW TS",i+1,"-> r:",self.signals[i].red," y:",self.signals[i].yellow," g:",self.signals[i].green)
                else:
                    print("   RED TS",i+1,"-> r:",self.signals[i].red," y:",self.signals[i].yellow," g:",self.signals[i].green)
        print()

    def _updateValues(self):
        for i in range(0, self.noOfSignals):
            if(i==self.currentGreen):
                if(self.currentYellow==0):
                    self.signals[i].green-=1
                else:
                    self.signals[i].yellow-=1
            else:
                self.signals[i].red-=1

    def _repeat_loop(self):
        while(True):
            while(self.signals[self.currentGreen].green>0):
                with self._lock:
                    self._printStatus()
                    self._updateValues()
                time.sleep(1)

            with self._lock:
                self.currentYellow = 1

            while(self.signals[self.currentGreen].yellow>0):
                with self._lock:
                    self._printStatus()
                    self._updateValues()
                time.sleep(1)

            with self._lock:
                self.currentYellow = 0

                if(self.config.randomGreenSignalTimer):
                    self.signals[self.currentGreen].green = random.randint(self.config.randomGreenSignalTimerRange[0], self.config.randomGreenSignalTimerRange[1])
                else:
                    self.signals[self.currentGreen].green = self.config.defaultGreen[self.currentGreen]
                self.signals[self.currentGreen].yellow = self.config.defaultYellow
                self.signals[self.currentGreen].red = self.config.defaultRed

                self.currentGreen = self.nextGreen
                self.nextGreen = (self.currentGreen+1) % self.noOfSignals
                self.signals[self.nextGreen].red = self.signals[self.currentGreen].yellow + self.signals[self.currentGreen].green

    def get_state(self):
        with self._lock:
            return self.currentGreen, self.currentYellow

    def get_signals(self):
        with self._lock:
            return self.signals

class WorldState:
    def __init__(self, config):
        self.config = config
        self.directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

        self.x = self._copy_spawn_coords(config.x)
        self.y = self._copy_spawn_coords(config.y)

        self.vehicles = {
            'right': {0:[], 1:[], 2:[], 'crossed':0},
            'down': {0:[], 1:[], 2:[], 'crossed':0},
            'left': {0:[], 1:[], 2:[], 'crossed':0},
            'up': {0:[], 1:[], 2:[], 'crossed':0}
        }

        self.vehiclesTurned = {'right': {1:[], 2:[]}, 'down': {1:[], 2:[]}, 'left': {1:[], 2:[]}, 'up': {1:[], 2:[]}}
        self.vehiclesNotTurned = {'right': {1:[], 2:[]}, 'down': {1:[], 2:[]}, 'left': {1:[], 2:[]}, 'up': {1:[], 2:[]}}

        self.simulation = pygame.sprite.Group()
        self.signalController = SignalController(config)

    def _copy_spawn_coords(self, d):
        out = {}
        for k in d:
            out[k] = [d[k][0], d[k][1], d[k][2]]
        return out

class BaseVehicle(pygame.sprite.Sprite):
    TYPE = "base"
    SPEED = 2.0

    def __init__(self, lane, direction_number, direction, will_turn, world):
        pygame.sprite.Sprite.__init__(self)
        self.world = world
        self.config = world.config

        self.lane = lane
        self.direction_number = direction_number
        self.direction = direction
        self.willTurn = will_turn

        self.speed = self.SPEED

        self.x = self.world.x[self.direction][self.lane]
        self.y = self.world.y[self.direction][self.lane]

        self.crossed = 0
        self.turned = 0
        self.rotateAngle = 0
        self.crossedIndex = 0

        path = "images/" + self.direction + "/" + self.TYPE + ".png"
        self.originalImage = pygame.image.load(path)
        self.image = pygame.image.load(path)

        self.world.vehicles[self.direction][self.lane].append(self)
        self.index = len(self.world.vehicles[self.direction][self.lane]) - 1

        if(len(self.world.vehicles[self.direction][self.lane])>1 and self.world.vehicles[self.direction][self.lane][self.index-1].crossed==0):
            prev = self.world.vehicles[self.direction][self.lane][self.index-1]
            if(self.direction=='right'):
                self.stop = (prev.stop - prev.image.get_rect().width - self.config.stoppingGap)
            elif(self.direction=='left'):
                self.stop = (prev.stop + prev.image.get_rect().width + self.config.stoppingGap)
            elif(self.direction=='down'):
                self.stop = (prev.stop - prev.image.get_rect().height - self.config.stoppingGap)
            elif(self.direction=='up'):
                self.stop = (prev.stop + prev.image.get_rect().height + self.config.stoppingGap)
        else:
            self.stop = self.config.defaultStop[self.direction]

        if(self.direction=='right'):
            temp = self.image.get_rect().width + self.config.stoppingGap
            self.world.x[self.direction][self.lane] -= temp
        elif(self.direction=='left'):
            temp = self.image.get_rect().width + self.config.stoppingGap
            self.world.x[self.direction][self.lane] += temp
        elif(self.direction=='down'):
            temp = self.image.get_rect().height + self.config.stoppingGap
            self.world.y[self.direction][self.lane] -= temp
        elif(self.direction=='up'):
            temp = self.image.get_rect().height + self.config.stoppingGap
            self.world.y[self.direction][self.lane] += temp

        self.world.simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        currentGreen, currentYellow = self.world.signalController.get_state()

        if(self.direction=='right'):
            if(self.crossed==0 and self.x+self.image.get_rect().width>self.config.stopLines[self.direction]):
                self.crossed = 1
                self.world.vehicles[self.direction]['crossed'] += 1
                if(self.willTurn==0):
                    self.world.vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(self.world.vehiclesNotTurned[self.direction][self.lane]) - 1

            if(self.willTurn==1):
                if(self.lane == 1):
                    if(self.crossed==0 or self.x+self.image.get_rect().width<self.config.stopLines[self.direction]+40):
                        if((self.x+self.image.get_rect().width<=self.stop or (currentGreen==0 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x+self.image.get_rect().width<(self.world.vehicles[self.direction][self.lane][self.index-1].x - self.config.movingGap) or self.world.vehicles[self.direction][self.lane][self.index-1].turned==1)):
                            self.x += self.speed
                    else:
                        if(self.turned==0):
                            self.rotateAngle += self.config.rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x += 2.4
                            self.y -= 2.8
                            if(self.rotateAngle==90):
                                self.turned = 1
                                self.world.vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(self.world.vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if(self.crossedIndex==0 or (self.y>(self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].y + self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].image.get_rect().height + self.config.movingGap))):
                                self.y -= self.speed

                elif(self.lane == 2):
                    if(self.crossed==0 or self.x+self.image.get_rect().width<self.config.mid[self.direction]['x']):
                        if((self.x+self.image.get_rect().width<=self.stop or (currentGreen==0 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x+self.image.get_rect().width<(self.world.vehicles[self.direction][self.lane][self.index-1].x - self.config.movingGap) or self.world.vehicles[self.direction][self.lane][self.index-1].turned==1)):
                            self.x += self.speed
                    else:
                        if(self.turned==0):
                            self.rotateAngle += self.config.rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x += 2
                            self.y += 1.8
                            if(self.rotateAngle==90):
                                self.turned = 1
                                self.world.vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(self.world.vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if(self.crossedIndex==0 or ((self.y+self.image.get_rect().height)<(self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].y - self.config.movingGap))):
                                self.y += self.speed

            else:
                if(self.crossed == 0):
                    if((self.x+self.image.get_rect().width<=self.stop or (currentGreen==0 and currentYellow==0)) and (self.index==0 or self.x+self.image.get_rect().width<(self.world.vehicles[self.direction][self.lane][self.index-1].x - self.config.movingGap))):
                        self.x += self.speed
                else:
                    if((self.crossedIndex==0) or (self.x+self.image.get_rect().width<(self.world.vehiclesNotTurned[self.direction][self.lane][self.crossedIndex-1].x - self.config.movingGap))):
                        self.x += self.speed

        elif(self.direction=='down'):
            if(self.crossed==0 and self.y+self.image.get_rect().height>self.config.stopLines[self.direction]):
                self.crossed = 1
                self.world.vehicles[self.direction]['crossed'] += 1
                if(self.willTurn==0):
                    self.world.vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(self.world.vehiclesNotTurned[self.direction][self.lane]) - 1

            if(self.willTurn==1):
                if(self.lane == 1):
                    if(self.crossed==0 or self.y+self.image.get_rect().height<self.config.stopLines[self.direction]+50):
                        if((self.y+self.image.get_rect().height<=self.stop or (currentGreen==1 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.y+self.image.get_rect().height<(self.world.vehicles[self.direction][self.lane][self.index-1].y - self.config.movingGap) or self.world.vehicles[self.direction][self.lane][self.index-1].turned==1)):
                            self.y += self.speed
                    else:
                        if(self.turned==0):
                            self.rotateAngle += self.config.rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x += 1.2
                            self.y += 1.8
                            if(self.rotateAngle==90):
                                self.turned = 1
                                self.world.vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(self.world.vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if(self.crossedIndex==0 or ((self.x + self.image.get_rect().width) < (self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].x - self.config.movingGap))):
                                self.x += self.speed

                elif(self.lane == 2):
                    if(self.crossed==0 or self.y+self.image.get_rect().height<self.config.mid[self.direction]['y']):
                        if((self.y+self.image.get_rect().height<=self.stop or (currentGreen==1 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.y+self.image.get_rect().height<(self.world.vehicles[self.direction][self.lane][self.index-1].y - self.config.movingGap) or self.world.vehicles[self.direction][self.lane][self.index-1].turned==1)):
                            self.y += self.speed
                    else:
                        if(self.turned==0):
                            self.rotateAngle += self.config.rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x -= 2.5
                            self.y += 2
                            if(self.rotateAngle==90):
                                self.turned = 1
                                self.world.vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(self.world.vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if(self.crossedIndex==0 or (self.x>(self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].x + self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].image.get_rect().width + self.config.movingGap))):
                                self.x -= self.speed

            else:
                if(self.crossed == 0):
                    if((self.y+self.image.get_rect().height<=self.stop or (currentGreen==1 and currentYellow==0)) and (self.index==0 or self.y+self.image.get_rect().height<(self.world.vehicles[self.direction][self.lane][self.index-1].y - self.config.movingGap))):
                        self.y += self.speed
                else:
                    if((self.crossedIndex==0) or (self.y+self.image.get_rect().height<(self.world.vehiclesNotTurned[self.direction][self.lane][self.crossedIndex-1].y - self.config.movingGap))):
                        self.y += self.speed

        elif(self.direction=='left'):
            if(self.crossed==0 and self.x<self.config.stopLines[self.direction]):
                self.crossed = 1
                self.world.vehicles[self.direction]['crossed'] += 1
                if(self.willTurn==0):
                    self.world.vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(self.world.vehiclesNotTurned[self.direction][self.lane]) - 1

            if(self.willTurn==1):
                if(self.lane == 1):
                    if(self.crossed==0 or self.x>self.config.stopLines[self.direction]-70):
                        if((self.x>=self.stop or (currentGreen==2 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x>(self.world.vehicles[self.direction][self.lane][self.index-1].x + self.world.vehicles[self.direction][self.lane][self.index-1].image.get_rect().width + self.config.movingGap) or self.world.vehicles[self.direction][self.lane][self.index-1].turned==1)):
                            self.x -= self.speed
                    else:
                        if(self.turned==0):
                            self.rotateAngle += self.config.rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x -= 1
                            self.y += 1.2
                            if(self.rotateAngle==90):
                                self.turned = 1
                                self.world.vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(self.world.vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if(self.crossedIndex==0 or ((self.y + self.image.get_rect().height) <(self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].y  -  self.config.movingGap))):
                                self.y += self.speed

                elif(self.lane == 2):
                    if(self.crossed==0 or self.x>self.config.mid[self.direction]['x']):
                        if((self.x>=self.stop or (currentGreen==2 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x>(self.world.vehicles[self.direction][self.lane][self.index-1].x + self.world.vehicles[self.direction][self.lane][self.index-1].image.get_rect().width + self.config.movingGap) or self.world.vehicles[self.direction][self.lane][self.index-1].turned==1)):
                            self.x -= self.speed
                    else:
                        if(self.turned==0):
                            self.rotateAngle += self.config.rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x -= 1.8
                            self.y -= 2.5
                            if(self.rotateAngle==90):
                                self.turned = 1
                                self.world.vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(self.world.vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if(self.crossedIndex==0 or (self.y>(self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].y + self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].image.get_rect().height +  self.config.movingGap))):
                                self.y -= self.speed

            else:
                if(self.crossed == 0):
                    if((self.x>=self.stop or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(self.world.vehicles[self.direction][self.lane][self.index-1].x + self.world.vehicles[self.direction][self.lane][self.index-1].image.get_rect().width + self.config.movingGap))):
                        self.x -= self.speed
                else:
                    if((self.crossedIndex==0) or (self.x>(self.world.vehiclesNotTurned[self.direction][self.lane][self.crossedIndex-1].x + self.world.vehiclesNotTurned[self.direction][self.lane][self.crossedIndex-1].image.get_rect().width + self.config.movingGap))):
                        self.x -= self.speed

        elif(self.direction=='up'):
            if(self.crossed==0 and self.y<self.config.stopLines[self.direction]):
                self.crossed = 1
                self.world.vehicles[self.direction]['crossed'] += 1
                if(self.willTurn==0):
                    self.world.vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(self.world.vehiclesNotTurned[self.direction][self.lane]) - 1

            if(self.willTurn==1):
                if(self.lane == 1):
                    if(self.crossed==0 or self.y>self.config.stopLines[self.direction]-60):
                        if((self.y>=self.stop or (currentGreen==3 and currentYellow==0) or self.crossed == 1) and (self.index==0 or self.y>(self.world.vehicles[self.direction][self.lane][self.index-1].y + self.world.vehicles[self.direction][self.lane][self.index-1].image.get_rect().height +  self.config.movingGap) or self.world.vehicles[self.direction][self.lane][self.index-1].turned==1)):
                            self.y -= self.speed
                    else:
                        if(self.turned==0):
                            self.rotateAngle += self.config.rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x -= 2
                            self.y -= 1.2
                            if(self.rotateAngle==90):
                                self.turned = 1
                                self.world.vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(self.world.vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if(self.crossedIndex==0 or (self.x>(self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].x + self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].image.get_rect().width + self.config.movingGap))):
                                self.x -= self.speed

                elif(self.lane == 2):
                    if(self.crossed==0 or self.y>self.config.mid[self.direction]['y']):
                        if((self.y>=self.stop or (currentGreen==3 and currentYellow==0) or self.crossed == 1) and (self.index==0 or self.y>(self.world.vehicles[self.direction][self.lane][self.index-1].y + self.world.vehicles[self.direction][self.lane][self.index-1].image.get_rect().height +  self.config.movingGap) or self.world.vehicles[self.direction][self.lane][self.index-1].turned==1)):
                            self.y -= self.speed
                    else:
                        if(self.turned==0):
                            self.rotateAngle += self.config.rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x += 1
                            self.y -= 1
                            if(self.rotateAngle==90):
                                self.turned = 1
                                self.world.vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(self.world.vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if(self.crossedIndex==0 or (self.x<(self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].x - self.world.vehiclesTurned[self.direction][self.lane][self.crossedIndex-1].image.get_rect().width - self.config.movingGap))):
                                self.x += self.speed

            else:
                if(self.crossed == 0):
                    if((self.y>=self.stop or (currentGreen==3 and currentYellow==0)) and (self.index==0 or self.y>(self.world.vehicles[self.direction][self.lane][self.index-1].y + self.world.vehicles[self.direction][self.lane][self.index-1].image.get_rect().height + self.config.movingGap))):
                        self.y -= self.speed
                else:
                    if((self.crossedIndex==0) or (self.y>(self.world.vehiclesNotTurned[self.direction][self.lane][self.crossedIndex-1].y + self.world.vehiclesNotTurned[self.direction][self.lane][self.crossedIndex-1].image.get_rect().height + self.config.movingGap))):
                        self.y -= self.speed

class Car(BaseVehicle):
    TYPE = "car"
    SPEED = 2.25

class Bus(BaseVehicle):
    TYPE = "bus"
    SPEED = 1.8

class Truck(BaseVehicle):
    TYPE = "truck"
    SPEED = 1.8

class Bike(BaseVehicle):
    TYPE = "bike"
    SPEED = 2.5

class VehicleFactory:
    def __init__(self, world):
        self.world = world
        self.vehicle_map = {"car": Car, "bus": Bus, "truck": Truck, "bike": Bike}

    def create(self, vehicleClass, lane, direction_number, direction, will_turn):
        cls = self.vehicle_map.get(vehicleClass)
        if(cls is None):
            raise ValueError("Unknown vehicle type: " + str(vehicleClass))
        return cls(lane, direction_number, direction, will_turn, self.world)

class SimulationEngine:
    def __init__(self, config):
        self.config = config
        self.world = WorldState(config)
        self.vehicleFactory = VehicleFactory(self.world)

        self.allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}
        self.allowedVehicleTypesList = []
        self._build_allowed_vehicle_list()

        self.timeElapsed = 0

        self.black = (0, 0, 0)
        self.white = (255, 255, 255)

        self.screenSize = (self.config.screenWidth, self.config.screenHeight)
        self.screen = pygame.display.set_mode(self.screenSize)
        pygame.display.set_caption("SIMULATION")

        self.background = pygame.image.load('images/intersection.png')
        self.redSignal = pygame.image.load('images/signals/red.png')
        self.yellowSignal = pygame.image.load('images/signals/yellow.png')
        self.greenSignal = pygame.image.load('images/signals/green.png')
        self.font = pygame.font.Font(None, 30)

        self.vehicleCountTexts = ["0", "0", "0", "0"]

        self._threads_started = False

    def _build_allowed_vehicle_list(self):
        i = 0
        for vehicleType in self.allowedVehicleTypes:
            if(self.allowedVehicleTypes[vehicleType]):
                self.allowedVehicleTypesList.append(i)
            i += 1

    def start(self):
        if(self._threads_started):
            return
        self._threads_started = True

        thread1 = threading.Thread(name="initialization",target=self.world.signalController.initialize, args=())
        thread1.daemon = True
        thread1.start()

        thread2 = threading.Thread(name="generateVehicles",target=self._generateVehicles_loop, args=())
        thread2.daemon = True
        thread2.start()

        thread3 = threading.Thread(name="simTime",target=self._simTime_loop, args=())
        thread3.daemon = True
        thread3.start()

    def _generateVehicles_loop(self):
        vehicleTypesByIndex = {0:'car', 1:'bus', 2:'truck', 3:'bike'}

        while(True):
            vehicle_type_index = random.choice(self.allowedVehicleTypesList)
            lane_number = random.randint(1,2)

            will_turn = 0
            temp = random.randint(0,99)
            if(temp<40):
                will_turn = 1

            temp = random.randint(0,99)
            direction_number = 0
            dist = [25,50,75,100]
            if(temp<dist[0]):
                direction_number = 0
            elif(temp<dist[1]):
                direction_number = 1
            elif(temp<dist[2]):
                direction_number = 2
            elif(temp<dist[3]):
                direction_number = 3

            direction = self.world.directionNumbers[direction_number]
            vehicleClass = vehicleTypesByIndex[vehicle_type_index]

            self.vehicleFactory.create(vehicleClass, lane_number, direction_number, direction, will_turn)
            time.sleep(1)

    def _simTime_loop(self):
        while(True):
            self.timeElapsed += 1
            time.sleep(1)
            if(self.timeElapsed==self.config.simulationTime):
                self.showStats()
                os._exit(1)

    def showStats(self):
        totalVehicles = 0
        print('Direction-wise Vehicle Counts')
        signals = self.world.signalController.get_signals()
        for i in range(0,4):
            if(i < len(signals) and signals[i]!=None):
                print('Direction',i+1,':',self.world.vehicles[self.world.directionNumbers[i]]['crossed'])
                totalVehicles += self.world.vehicles[self.world.directionNumbers[i]]['crossed']
        print('Total vehicles passed:',totalVehicles)
        print('Total time:',self.timeElapsed)

    def _render_signals_and_text(self):
        signals = self.world.signalController.get_signals()
        if(len(signals) < 4):
            return False

        currentGreen, currentYellow = self.world.signalController.get_state()

        for i in range(0,4):
            if(i==currentGreen):
                if(currentYellow==1):
                    signals[i].signalText = signals[i].yellow
                    self.screen.blit(self.yellowSignal, self.config.signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    self.screen.blit(self.greenSignal, self.config.signalCoods[i])
            else:
                if(signals[i].red<=10):
                    signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                self.screen.blit(self.redSignal, self.config.signalCoods[i])

        signalTexts = ["","","",""]
        for i in range(0,4):
            signalTexts[i] = self.font.render(str(signals[i].signalText), True, self.white, self.black)
            self.screen.blit(signalTexts[i], self.config.signalTimerCoods[i])

        for i in range(0,4):
            displayText = self.world.vehicles[self.world.directionNumbers[i]]['crossed']
            self.vehicleCountTexts[i] = self.font.render(str(displayText), True, self.black, self.white)
            self.screen.blit(self.vehicleCountTexts[i], self.config.vehicleCountCoods[i])

        timeElapsedText = self.font.render(("Time Elapsed: "+str(self.timeElapsed)), True, self.black, self.white)
        self.screen.blit(timeElapsedText, self.config.timeElapsedCoods)

        return True

    def run(self):
        self.start()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.showStats()
                    sys.exit()

            self.screen.blit(self.background,(0,0))

            ok = self._render_signals_and_text()
            if(not ok):
                pygame.display.update()
                time.sleep(0.01)
                continue

            for vehicle in self.world.simulation:
                vehicle.render(self.screen)
                vehicle.move()

            pygame.display.update()

pygame.init()
config = SimulationConfig()
engine = SimulationEngine(config)
engine.run()
