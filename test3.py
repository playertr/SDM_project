from shapely.geometry import Point, Polygon
from shapely import affinity as af
import multiprocessing
import numpy as np
import time

class Particle:
    """ Single particle in a particle filter. State is x, y, theta."""

    def __init__(self):
        self.x = None
        self.y = None
        self.theta = None
        self._vertices = None # array of homogeneous coords
        self.shape = None

def transition_fnc(p):
    """ Makes a new particle, moved by a random amount. """
    shape = Polygon([[0, 0], [6, 0], [0, 3], [0, 0]])
    new_p = Particle()
    new_p.x = np.random.normal(loc=p.x, scale=1)
    new_p.y = np.random.normal(loc=p.y, scale=1)
    new_p.theta = np.random.normal(loc=p.theta, scale=1) % (2 * np.pi)
    new_p.shape = af.translate(af.rotate(shape, new_p.theta, use_radians=True,
                                            origin=(0, 0)), new_p.x, new_p.y)
    return new_p

def reset_particles(particles):
    """ Initialize particles to random dist over map. """
    shape = Polygon([[0, 0], [6, 0], [0, 3], [0, 0]])
    for p in particles:
        p.x = np.random.uniform(0, 10)
        p.y = np.random.uniform(0, 10)
        p.theta = np.random.uniform(0, 2 * np.pi)
        p.shape = af.translate(af.rotate(shape, p.theta, use_radians=True,
                                            origin=(0, 0)), p.x, p.y)

def main():
    particles = [Particle() for i in range(1000)]
    reset_particles(particles)

    tic = time.time()

    for _ in range(100):
        x_bar = [transition_fnc(p) for p in particles]
        particles = x_bar
    print(f"Elapsed time: {time.time() - tic}")

    
def parallel_main():
    particles = [Particle() for i in range(1000)]
    reset_particles(particles)

    tic = time.time()

    for _ in range(100):
        with multiprocessing.Pool(None) as pool:
            new_particles = pool.map(
                transition_fnc,
                particles)

        particles = new_particles

    print(f"Elapsed time: {time.time() - tic}")


if __name__ == '__main__':
    main()