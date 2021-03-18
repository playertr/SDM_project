import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import euclidean

from StateEstimator import ParticleFilter
from World import World
import multiprocessing
from functools import partial

import os

class ActionController:

    def __init__(self, 
    # World
    shape, x, y, theta, map_size, 
    # ParticleFilter
    num_particles,
    # Experiment
    lookahead_depth, discount):
        self.world = World(shape, x, y, theta, map_size)
        self.particle_filter = ParticleFilter(shape, num_particles, map_size)

        self.lookahead_depth = lookahead_depth
        self.discount = discount

        # Convergence criteria
        self.mean_xy_threshold = 1
        self.mean_theta_threshold = np.pi / 8
        self.std_xy_threshold = 2
        self.std_theta_threshold = np.pi / 8

        # Accumulators
        self.entropies = []
        self.costs = []

        self.previous_action = (0,0)
        self.converged = None
    

    def tick(self):
        action = self.get_action()
        measurement = self.particle_filter.noisy_measure(*self.world.measure(*action)) # measurement: x, y, within_object
        self.particle_filter.update(measurement)

        self.update_convergence()
        self.entropies.append(self.particle_filter.get_entropy())
        self.costs.append(euclidean(self.previous_action or action, action))
    
        self.previous_action = action


    def update_convergence(self):
        
        avg_x = self.particle_filter.x
        avg_y = self.particle_filter.y
        avg_theta = self.particle_filter.theta

        std_x = np.std([particle.x for particle in self.particle_filter.particles])
        std_y = np.std([particle.y for particle in self.particle_filter.particles])
        std_theta = np.std([particle.theta for particle in self.particle_filter.particles])

        if (
            np.abs(avg_x - self.world.x) <= self.mean_xy_threshold
            and np.abs(avg_y - self.world.y) <= self.mean_xy_threshold
            and np.abs(avg_theta - self.world.theta) <= self.mean_theta_threshold
            and std_x <= self.std_xy_threshold
            and std_y <= self.std_xy_threshold
            and std_theta <= self.std_theta_threshold
        ):
            self.converged = True
        else:
            self.converged = False


    def get_action(self):
        # return self._get_action_multiprocessing(self.previous_action, self.particle_filter, self.lookahead_depth)
        return self._get_action_multiprocessing(self.previous_action, self.particle_filter, self.lookahead_depth)

    def _get_action_multiprocessing(self, previous_action, particle_filter, lookahead_depth):
        # The FIRST recursive call starts separate processes. 

        # If the original lookahead_depth > 0, this output is not used.
        # If this is the first call and the lookahead_depth == 0, produce a random action.
        if lookahead_depth == 0:
            return (
                np.random.uniform(0, self.world.map_size[0]), 
                np.random.uniform(0, self.world.map_size[1])
            )

        current_entropy = particle_filter.get_entropy()
        candidate_actions = particle_filter.get_candidate_actions()

        with multiprocessing.Pool(None) as pool:
            candidate_action_values = pool.map(
                partial(predict_value,
                    previous_action=self.previous_action,
                    previous_entropy=current_entropy,
                    particle_filter=particle_filter,
                    lookahead_depth=lookahead_depth-1,
                    discount=self.discount,
                    map_size=self.world.map_size
                    ),
                candidate_actions
            )
        
        return candidate_actions[np.argmax(candidate_action_values)]
    
    # def _get_action(self, previous_action, particle_filter, lookahead_depth):
    #     # Just to test without multiprocessing

    #     # If the original lookahead_depth > 0, this output is not used.
    #     # If this is the first call and the lookahead_depth == 0, produce a random action.
    #     if lookahead_depth == 0:
    #         return (
    #             np.random.uniform(0, self.world.map_size[0]), 
    #             np.random.uniform(0, self.world.map_size[1])
    #         )

    #     current_entropy = particle_filter.get_entropy()
    #     candidate_actions = particle_filter.get_candidate_actions()

    #     candidate_action_values = [pv_wrapper(
    #         action, 
    #         particle_filter=particle_filter,
    #         lookahead_depth=lookahead_depth-1,
    #         previous_entropy=current_entropy,
    #         discount=self.discount,
    #         map_size=self.world.map_size)
    #         for action in candidate_actions
    #         ]
        
    #     return candidate_actions[np.argmax(candidate_action_values)]
    

# Modified get_action and predict_entropy_and_cost to be global, stateless functions for multiprocessing.
# def pv_wrapper(action, particle_filter, previous_entropy, lookahead_depth, discount, map_size):
#     return predict_value(
#         this_action=action, 
#         previous_action=action,
#         previous_entropy=previous_entropy,
#         particle_filter=particle_filter,
#         lookahead_depth=lookahead_depth,
#         discount=discount,
#         map_size=map_size
#     )


def get_action(previous_action, particle_filter, lookahead_depth, discount, map_size):
    """
    Returns the action, a Point object, to be taken next given the belief state 
    represented by particle_filter. Recursively looks ahead in concert with predict_entropy_and_cost.

    Uses Information Gain / Cost as the discerning objective between candidate_actions.
    """
    if lookahead_depth == 0:
        print("Should never happen by recursion.")
        return (
            np.random(0, map_size[0]),
            np.random(0, map_size[1])
        )

    current_entropy = particle_filter.get_entropy()
    candidate_actions = particle_filter.get_candidate_actions()
    candidate_values = []
        
    # An action is just a tuple (x, y).

    for action in candidate_actions:

        value = predict_value(
            this_action=action, 
            previous_action=previous_action, 
            previous_entropy=current_entropy, 
            particle_filter=particle_filter, 
            lookahead_depth=lookahead_depth-1, 
            discount=discount, 
            map_size=map_size)

        candidate_values.append(value)
    
    return candidate_actions[np.argmax(candidate_values)], np.max(candidate_values)

def predict_value(this_action, previous_action, previous_entropy, particle_filter, lookahead_depth, discount, map_size):
    """
    Returns the predicted value of taking this_action after having taken the previous_action, 
    considered across lookahead_depth future actions.
    """
    # print(f"({previous_action[0]:.2f},{previous_action[1]:.2f}),({this_action[0]:.2f},{this_action[1]:.2f})")
    # Compute current value of this node from its information gain and distance

    alpha = particle_filter.get_expectation_of_hit(this_action)

    outcome_entropies = []
    pfs = []

    # Determine the entropy after recieving either observation.

    for outcome in [True, False]: # True implies a hit, False implies a miss.
        hypothetical_particle_filter = particle_filter.copy()
        hypothetical_particle_filter.update((*this_action, outcome))

        outcome_entropy = hypothetical_particle_filter.get_entropy()
        
        pfs.append(hypothetical_particle_filter)
        outcome_entropies.append(outcome_entropy)

    current_expected_entropy = np.dot(outcome_entropies, [alpha, 1 - alpha])
    current_cost = euclidean(previous_action, this_action)
    
    current_value = (previous_entropy - current_expected_entropy) / (current_cost or 1)

    # Base case
    if lookahead_depth == 0:
        return current_value
    
    # Recursive case: find discounted expected future value
    next_values = []
    for hypothetical_particle_filter,outcome_entropy in zip(pfs, outcome_entropies):
    
        # Compute the next_action that would be taken given the hypothetical_particle_filter belief state, 
        # and recursively find the entropy and cost of that action for a decremented lookahead_depth.
        next_action, next_value = get_action(this_action, hypothetical_particle_filter, lookahead_depth, discount, map_size)

        # Store expected values for each outcome of this_action.
        next_values.append(next_value)
    
    expected_next_value = np.dot(next_values, [alpha, 1 - alpha])

    return current_value + discount * expected_next_value