#!/usr/bin/env python3

import numpy as np
from ActionController import ActionController, ParticleFilter
import statistics as stats

if __name__ == '__main__':
    # mean particle x,y,theta are represented as instance.x/y/theta
    # We need to get variance for each of these to make sure there is actual convergence
    a = ActionController
    while True:
        std_x = stats.stdev([p.x for p in a.ParticleFilter.particles])
        std_y = stats.stdev([p.y for p in a.ParticleFilter.particles])
        std_theta = stats.stdev([p.theta for p in a.ParticleFilter.particles])
        std_thresh_xy = 2
        std_thresh_theta = np.pi/8
        mean_x = a.ParticleFilter.x
        mean_y = a.ParticleFilter.y
        mean_theta = a.ParticleFileter.theta
        # stop if mean is within ~5-10% of true centroid and
        # variance is less than the largest distance to outer edge of object from centroid
        if abs(mean_x - a.world.centroid.x) < 0.5 and abs(mean_y - a.world.centroid.y) < 0.5 and abs(mean_theta - a.world.theta) < np.pi/6:
            if (std_x and std_y) < std_thresh_xy and (std_theta < std_thresh_theta):
                break

    err = 'go away'