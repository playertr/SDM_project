from shapely.geometry import Point, Polygon
from shapely import affinity as af
import multiprocessing
import numpy as np
import time
from numba import jit
import matplotlib.pyplot as plt

class Particle:
    """ Single particle in a particle filter. State is x, y, theta."""

    def __init__(self):
        self.x = None
        self.y = None
        self.theta = None
        self._vertices = None # array of homogeneous coords
        self.shape = None

def transition_fnc(p):
    """ Moves this particle by a random amount. """
    new_p = Particle()
    new_p.x = np.random.normal(loc=p.x, scale=1)
    new_p.y = np.random.normal(loc=p.y, scale=1)
    new_p.theta = np.random.normal(loc=p.theta, scale=1) % (2 * np.pi)
    new_p._vertices = tfmat(p.theta, p.x, p.y).dot(p._vertices)
    new_p.shape = Polygon(p._vertices[0:2,:].T)
    return new_p

def reset_particles(particles, shape):
    """ Initialize particles to random dist over map. """

    # Create array of homogeneous coords from shape vertices
    pts = np.array(shape.exterior.coords)
    for p in particles:
        p._vertices = np.zeros((3, len(pts)))
        p._vertices[2,:] = 1
        p._vertices[0:2,:] = np.array(shape.exterior.coords).T

    # Randomly transform, then make new Shape objects
    for p in particles:
        p.x = np.random.uniform(0, 10)
        p.y = np.random.uniform(0, 10)
        p.theta = np.random.uniform(0, 2 * np.pi)
        p._vertices = tfmat(p.theta, p.x, p.y).dot(p._vertices)
        p.shape = Polygon(p._vertices[0:2,:].T)

@jit(nopython=True)
def tfmat(theta, x, y):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, x],
        [s, c,  y],
        [0.0, 0.0,  1.0]
        ])

def fast_main():
    shape = Polygon([[0, 0], [6, 0], [0, 3], [0, 0]])

    particles = [Particle() for i in range(1000)]
    reset_particles(particles, shape)

    tic = time.time()

    for _ in range(100):
        particles = [transition_fnc(p) for p in particles]
        # for p in particles:
        #     transition_fnc(p)

    print(f"Elapsed time: {time.time() - tic}")

def main():
    shape = Polygon([[0, 0], [6, 0], [0, 3], [0, 0]])

    pts = np.array(shape.exterior.coords)
    vertices = np.zeros((3, len(pts)))
    vertices[2,:] = 1
    vertices[0:2,:] = pts.T

    print(vertices)

    shape2 = Polygon(vertices[0:2,:].T)

    theta = -np.pi/6
    x = 1
    y = 0.1
    tf = tfmat(-np.pi/6, 1, 0.1)
    vert2 = tf.dot(vertices)
    print(vert2)
    shape3 = Polygon(vert2[0:2,:].T)

    shape4 = af.translate(af.rotate(shape, theta, use_radians=True,
                                             origin=(0, 0)), x, y)

    plt.plot(*shape2.exterior.xy, 'r')
    plt.plot(*shape3.exterior.xy)
    plt.plot(*shape4.exterior.xy, 'g.')
    plt.axes().set_aspect('equal')
    plt.pause(0.05)

    breakpoint()
    print("hello")
    
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