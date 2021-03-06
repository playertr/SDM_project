# World.py
# February 16, 2021
# Tim Player playert@oregonstate.edu

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from shapely import affinity as af

class World:
    """2D continuous map world to hold a 2D shape. """

    def __init__(self, shape, x, y, theta, map_size=(10,10)):
        self.shape      = shape # a Polygon object
        self.map_size   = map_size
        self.x          = x
        self.y          = y
        self.theta      = theta
        self.tf_shape   = af.translate(af.rotate(self.shape,
                            theta, use_radians=True, origin=(0,0)), x, y)
        self.centroid = self.tf_shape.centroid

    def measure(self, x, y):
        """ Determine whether point (x,y) is within the shape."""
        p = Point(x, y)
        return (x, y, p.within(self.tf_shape))

    def plot(self, ax):
        """ Plot the shape on the map """
        ax.set_title('World Map')
        ax.plot(*self.tf_shape.exterior.xy)
        ax.set_xlim([0, self.map_size[0]])
        ax.set_ylim([0, self.map_size[1]])
        ax.set_aspect('equal')
