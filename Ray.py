from util import Vector


class Ray:
    def __init__(self, ray_source:Vector, ray_direction:Vector):
        self.source = ray_source
        self.direction = ray_direction
