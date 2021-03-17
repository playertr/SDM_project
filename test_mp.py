import numpy as np
from shapely.geometry import Polygon
import time

from ActionController import ActionController

def main():
    # World
    shape = Polygon([[0, 0], [6, 0], [0, 3], [0, 0]]) # triangle

    x = 5
    y = 5
    theta = np.pi/3
    map_size = 10, 10

    # ParticleFilter
    num_particles = 1000

    # Experiment
    lookahead_depth = 2
    discount = 0.9

    action_controller = ActionController(
        shape, x, y, theta, map_size, 
        num_particles, 
        lookahead_depth, discount, 
    )

    # Simulation
    max_iterations = 1000

    # Stopping criteria
    # Mean thresholds
    mean_xy_dif = 1
    mean_theta_dif = np.pi/8
    # Standard deviation thresholds
    std_xy_threshold = 2
    std_theta_threshold = np.pi/8

    #################################################################### TEST

    # Run
    iteration = 0
    start_time = time.perf_counter()
    while iteration < 1:
        iteration += 1
        print(f"Iteration {iteration}")
        action_controller.tick()

        if iteration == max_iterations:
            break

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(duration)

if __name__ == '__main__':
    main()