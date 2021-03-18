# StateEstimator.py
# February 16, 2021
# Tim Player playert@oregonstate.edu

from abc import ABC, abstractmethod
import numpy as np
from shapely import affinity as af
from shapely.geometry import Point
from scipy.stats import entropy
from copy import deepcopy
from numba import jit
from shapely.geometry import Polygon
import random

class StateEstimator(ABC):
    """ State Estimator abstract base class. StateEstimator
    may be implemented as, e.g., a particle filter or Kalman
    filter. """

    def __init__(self, map_size):
        self.x = None
        self.y = None
        self.theta = None
        self.map_size = map_size
        self.centroid = None

    def get_state(self):
        """ Return the estimated most-likely state. """
        return self.x, self.y, self.theta

    @abstractmethod
    def get_uncertainty(self):
        """ Return the uncertainty. The return type may vary. """
        pass

@jit(nopython=True)
def tfmat(theta, x, y):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, x],
        [s, c, y],
        [0.0, 0.0, 1.0]
        ])

class ParticleFilter(StateEstimator):
    """ Particle filter to estimate object x, y, and theta. """

    def __init__(self, shape, num_particles, map_size):
        super().__init__(map_size)
        self.shape = af.translate(shape, -shape.centroid.xy[0].tolist()[0], -shape.centroid.xy[1].tolist()[0]) # shape of object

        # Make array of homogeneous vertex coordinates
        pts = np.array(self.shape.exterior.coords) 
        verts = np.zeros((3, len(pts)))
        verts[2,:] = 1
        verts[0:2,:] = pts.T
        self._vertices = verts

        self.particles = [Particle() for i in range(num_particles)]
        self.reset_particles()

        self.XY_TRANSITION_SIGMA = 0.1
        self.THETA_TRANSITION_SIGMA = 7.5 * np.pi/180
        self.OBS_ACCURACY = 0.99
    
    def copy(self):
        return deepcopy(self)

    def noisy_measure(self, x, y, truth):
        hit = truth
        if not np.random.random() < self.OBS_ACCURACY:
            hit = not hit
        return x, y, hit

    def update(self, z):
        """ Updates particles and estimate with new measurement z=(x,y,CONTACT)"""
        # sample new particles according to transition probabilities
        x_bar = [self.transition_fnc(p) for p in self.particles]

        # weight new particles according to observation likelihood
        weights = [self.obs_likelihood(p, z) for p in x_bar]
        sumw = sum(weights)
        weights = [w / sumw for w in weights]

        # draw particles according to their weights
        self.particles = np.random.choice(x_bar, len(x_bar), p=weights, replace=True)

        # update MLE estimate as mean of particles. Could also do k-means.
        self.x = np.mean([p.x for p in self.particles])
        self.y = np.mean([p.y for p in self.particles])
        self.theta = np.mean([p.theta for p in self.particles])

    def obs_likelihood(self, p, z):
        """ Returns likelihood of getting measurement z given that the
        object has state (p.x, p.y, p.theta)."""
        tf_shape = p.shape
        point = Point(z[0], z[1])
        within = point.within(tf_shape)
        return self.OBS_ACCURACY if within == z[2] else (1 - self.OBS_ACCURACY)

    def transition_fnc(self, p):
        """ Makes a new particle, moved by a random amount. """
        new_p = Particle()
        new_p.x = np.random.normal(loc=p.x, scale=self.XY_TRANSITION_SIGMA)
        new_p.y = np.random.normal(loc=p.y, scale=self.XY_TRANSITION_SIGMA)
        new_p.theta = np.random.normal(loc=p.theta, scale=self.THETA_TRANSITION_SIGMA) % (2 * np.pi)
        verts = tfmat(new_p.theta, new_p.x, new_p.y).dot(self._vertices)
        new_p.shape = Polygon(verts[0:2,:].T)
        return new_p

    def reset_particles(self):
        """ Initialize particles to random dist over map. """
        for p in self.particles:
            p.x = np.random.uniform(0, self.map_size[0])
            p.y = np.random.uniform(0, self.map_size[1])
            p.theta = np.random.uniform(0, 2 * np.pi)
            p.shape = af.translate(af.rotate(self.shape, p.theta, use_radians=True,
                                             origin=(0, 0)), p.x, p.y)

    def xyt_to_poly(self, x, y, theta):
        """ Efficiently convert an x, y, theta state to a Polygon object. """
        verts = tfmat(theta, x, y).dot(self._vertices)
        return Polygon(verts[0:2,:].T)

    def draw_particles(self, ax):
        """ Draw all the particles on this plt Axis. """
        X = np.array([p.x for p in self.particles])
        Y = np.array([p.y for p in self.particles])
        U = np.array([np.cos(p.theta) for p in self.particles])
        V = np.array([np.sin(p.theta) for p in self.particles])
        vecs = np.stack([X, Y, U, V])
        W, counts = np.unique(vecs, axis=1, return_counts=True)
        q = ax.quiver(W[0, :], W[1, :], W[2, :], W[3, :], counts)
        return q

    def draw_histogram(self, ax1, ax2, ax3):
        """ Draws a histogram of particle x, y, theta on thes plt Axes."""
        X = np.array([p.x for p in self.particles])
        weights = np.ones_like(X) / float(len(X))
        h1 = ax1.hist(X, weights=weights)
        ax1.set_xlabel('x')
        Y = np.array([p.y for p in self.particles])
        h2 = ax2.hist(Y, weights=weights)
        ax2.set_xlabel('y')
        T = np.array([p.theta for p in self.particles])
        h3 = ax3.hist(T, weights=weights)
        ax3.set_xlabel('theta')
        return h1, h2, h3

    def get_uncertainty(self):
        pass

    def get_entropy(self, bin_sizes=(50, 50, 10)):
        """
        Given the current particle distribution returns the normalized shannon entropy
        :param bin_sizes: number of bins for x, y, theta
        :return: Shannon entropy
        """
        # Gather particle poses into lists
        x = [p.x for p in self.particles]
        y = [p.y for p in self.particles]
        theta = [p.theta % 2 * np.pi for p in self.particles]

        # Use 3D histogram to bin particles
        bounds = ((0, self.map_size[0]), (0, self.map_size[1]), (0, 2 * np.pi))
        H, edges = np.histogramdd((x, y, theta), bins=bin_sizes, range=bounds)

        return entropy((H / np.sum(H)).flatten()) / np.log(H.size)

    # def get_candidate_actions(self, p_samples=10, tot_samples=10, sigma=4):
    #     samples = []
    #     for p in self.particles:
    #         # Double check that this is the correct syntax for getting the centroid -----------
    #         cen_loc = p.shape.centroid
    #         for take_samples in range(p_samples):
    #             samples.append((np.random.normal(loc=cen_loc.x, scale=sigma),
    #                             (np.random.normal(loc=cen_loc.y, scale=sigma))))
    #     return [samples[i] for i in np.random.randint(0, len(samples), size=tot_samples)]
    
    def get_candidate_actions(self, p_samples=1, tot_samples=10, sigma=1):
        samples = []
        for p in self.particles:
            # Double check that this is the correct syntax for getting the centroid -----------
            # cen_loc = p.shape.centroid
            random_vertex = random.choice(list(p.shape.exterior.coords))
            for take_samples in range(p_samples):
                samples.append((np.random.normal(loc=random_vertex[0], scale=sigma),
                                (np.random.normal(loc=random_vertex[1], scale=sigma))))
        return [samples[i] for i in np.random.randint(0, len(samples), size=tot_samples)]



    def get_expectation_of_hit(self, sample):
        # For sample, check hit against all particles, then average hits vs misses
        hits = 0
        misses = 0
        samplePoint = Point(sample[0], sample[1])
        for p in self.particles:
            if samplePoint.within(p.shape):
                hits += 1
            else:
                misses += 1
        return hits / (hits + misses)



class Particle:
    """ Single particle in a particle filter. State is x, y, theta."""

    def __init__(self):
        self.x = None
        self.y = None
        self.theta = None
        self.shape = None
