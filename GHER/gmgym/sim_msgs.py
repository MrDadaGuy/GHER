#!/usr/bin/env python3

class Observation:
    def __init__(self, done=False, info="", box_x=None, box_y=None, box_z=None, ball_x=None, ball_y=None, ball_z=None):
        self.done = done
        self.info = info
        self.Box = _Box(box_x, box_y, box_z)
        self.Ball = _Ball(ball_x, ball_y, ball_z)
    
class _Box:
    def __init__ (self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

class _Ball:
    def __init__ (self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z