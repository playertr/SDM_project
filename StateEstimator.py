# StateEstimator.py
# February 16, 2021
# Tim Player playert@oregonstate.edu

from abc import ABC, abstractmethod
import numpy as np
from shapely import affinity as af
from shapely.geometry import Point

class StateEstimator(ABC):
    """ State Estimator abstract base class. StateEstimator
    may be implemented as, e.g., a particle filter or Kalman
    filter. """

    def __init__(self, map_size):
        self.x          = None
        self.y          = None
        self.theta      = None
        self.map_size   = map_size

    def get_state(self):
        """ Return the estimated most-likely state. """
        return self.x, self.y, self.theta

    @abstractmethod
    def get_uncertainty(self):
        """ Return the uncertainty. The return type may vary. """
        pass

class ParticleFilter(StateEstimator):
    """ Particle filter to estimate object x, y, and theta. """

    def __init__(self, shape, num_particles, map_size):
        super().__init__(map_size)
        self.shape     = shape # shape of object
        self.particles = [Particle() for i in range(num_particles)]
        self.reset_particles(map_size)

        self.TRANSITION_SIGMA    = 0.1
        self.OBS_ACCURACY     = 0.99

    def update(self, z):
        """ Updates particles and estimate with new measurement z=(x,y,CONTACT)"""
        # sample new particles according to transition probabilities
        x_bar   = [self.transition_fnc(p) for p in self.particles]

        # weight new particles according to observation likelihood
        weights = [self.obs_likelihood(p, z) for p in x_bar]
        sumw    = sum(weights)
        weights = [w / sumw for w in weights]

        # draw particles according to their weights
        self.particles = np.random.choice(x_bar, len(x_bar), p=weights, replace=True)

        # update MLE estimate as mean of particles. Could also do k-means.
        self.x      = np.mean([p.x for p in self.particles])
        self.y      = np.mean([p.y for p in self.particles])
        self.theta  = np.mean([p.theta for p in self.particles])

    def obs_likelihood(self, p, z):
        """ Returns likelihood of getting measurement z given that the
        object has state (p.x, p.y, p.theta)."""
        tf_shape = af.translate(af.rotate(self.shape, 
                            p.theta, use_radians=True), p.x, p.y)
        point  = Point(z[0], z[1])
        within = point.within(tf_shape)
        return self.OBS_ACCURACY if within == z[2] else (1 - self.OBS_ACCURACY)

    def transition_fnc(self, p):
        """ Makes a new particle, moved by a random amount. """
        new_p = Particle()
        new_p.x     = np.random.normal(loc=p.x, scale=self.TRANSITION_SIGMA)
        new_p.y     = np.random.normal(loc=p.y, scale=self.TRANSITION_SIGMA)
        new_p.theta = np.random.normal(loc=p.theta, scale=self.TRANSITION_SIGMA)
        return new_p

    def reset_particles(self, map_size):
        """ Initialize particles to random dist over map. """
        for p in self.particles:
            p.x     = np.random.uniform(0, self.map_size[0])
            p.y     = np.random.uniform(0, self.map_size[1])
            p.theta = np.random.uniform(0, 2*np.pi)

    def draw_particles(self, ax):
        """ Draw all the particles on this plt Axis. """
        X = np.array([p.x for p in self.particles])
        Y = np.array([p.y for p in self.particles])
        U = np.array([np.cos(p.theta) for p in self.particles])
        V = np.array([np.sin(p.theta) for p in self.particles])
        vecs = np.stack([X,Y,U,V])
        W, counts = np.unique(vecs, axis=1, return_counts=True)
        q = ax.quiver(W[0,:], W[1,:], W[2,:], W[3,:], counts)
        return q

    def get_uncertainty(self):
        pass

class Particle:
    """ Single particle in a particle filter. State is x, y, theta."""
    
    def __init__(self):
        self.x      = None
        self.y      = None
        self.theta  = None