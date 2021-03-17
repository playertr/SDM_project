import numpy as np
from shapely.geometry import Point, Polygon
from shapely import affinity as af
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import display
import time
import imageio
from pathlib import Path
from ActionController import ActionController
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # World
    # shape = Polygon([[0, 0], [6, 0], [6, 1],[0, 1], [0, 0]]) # rectangle
    shape = Polygon([[0, 0], [6, 0], [0, 3], [0, 0]]) # triangle
    # Center shape
    shape
    x = 5
    y = 5
    theta = np.pi/3
    map_size = 10, 10

    # ParticleFilter
    num_particles = 1000
                                                    
    # Experiment
    lookahead_depth = 1 ########################################################################## important shit right here
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


    # Visuals

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(nrows=3, ncols=3)

    ax1 = fig.add_subplot(gs[:,0])
    ax2 = fig.add_subplot(gs[:,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,2])
    ax5 = fig.add_subplot(gs[2,2])

    action_controller.world.plot(ax1)
    action_controller.world.plot(ax2)
    ax1.set_title('Particles')
    ax2.set_title('Mean of Particles')
    fig.tight_layout()


    # Run
    iteration = 0
    start_time = time.perf_counter()
    while iteration < 4:
        iteration += 1
        action_controller.tick()
        
        # Visuals
        s = ax1.scatter(*action_controller.previous_action, c='r')
        q = action_controller.particle_filter.draw_particles(ax1)
        ax3.clear(); ax4.clear(); ax5.clear();
        h1, h2, h3 = action_controller.particle_filter.draw_histogram(ax3, ax4, ax5)

        x,y,t = action_controller.particle_filter.get_state() # access estimated state
        q2 = ax2.quiver(x, y, np.cos(t), np.sin(t), label='Estimate')

        fig.suptitle(f"iteration {iteration}")
        plt.pause(0.005)

        s.remove()
        q.remove()
        q2.remove()
        
        if iteration == max_iterations:
            break

    end_time = time.perf_counter()
    duration = end_time - start_time
    print(duration / 4)