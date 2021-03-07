import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import euclidean

from StateEstimator import ParticleFilter
from World import World


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

        self.previous_action = None
        self.converged = None
    

    def tick(self):
        action = self.get_action()
        self.previous_action = action
        measurement = self.particle_filter.noisy_measure(*self.world.measure(*action)) # measurement: x, y, within_object
        self.particle_filter.update(measurement)

        self.update_convergence()
        self.entropies.append(self.particle_filter.get_entropy())
        self.costs.append(euclidean(self.previous_action or action, action))
    

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
        return self._get_action(self.previous_action, self.particle_filter, self.lookahead_depth)


    def _get_action(self, previous_action, particle_filter, lookahead_depth):
        """
        Returns the action, a Point object, to be taken next given the belief state 
        represented by particle_filter. Recursively looks ahead in concert with predict_entropy_and_cost.

        Uses Information Gain / Cost as the discerning objective between candidate_actions.
        """

        # If the original lookahead_depth > 0, this output is not used.
        # If this is the first call and the lookahead_depth == 0, produce a random action.
        if lookahead_depth == 0:
            return (
                np.random.uniform(0, self.world.map_size[0]), 
                np.random.uniform(0, self.world.map_size[1])
            )

        current_entropy = particle_filter.get_entropy()
        candidate_actions = particle_filter.get_candidate_actions()
        candidate_action_objectives = []
        
        # An action is just a tuple (x, y).

        for action in candidate_actions:
            if previous_action is None: previous_action = action # For the first action in the simulation.

            action_entropy, action_cost = self._predict_entropy_and_cost(action, previous_action, particle_filter, 1, lookahead_depth)
            objective = (current_entropy - action_entropy) / (action_cost or 1)
            candidate_action_objectives.append(objective)
        
        return candidate_actions[np.argmax(candidate_action_objectives)]


    def _predict_entropy_and_cost(self, this_action, previous_action, particle_filter, steps_into_future, lookahead_depth):
        """
        Returns a tuple of the expected entropy and cost of taking this_action after having taken previous_action, 
        considered across lookahead_depth future actions.
        """

        # Base case.
        if lookahead_depth == 0:
            return (particle_filter.get_entropy(), 0)

        # Recursive step.
        alpha = particle_filter.get_expectation_of_hit(this_action)
        outcome_entropies = []
        outcome_costs = []
        for outcome in [True, False]: # True implies a hit, False implies a miss.
            # Create an updated copy of particle_filter as though this_action had been taken and resulted in outcome.
            hypothetical_particle_filter = particle_filter.copy()
            hypothetical_particle_filter.update((*this_action, outcome))
            
            # Compute the next_action that would be taken given the hypothetical_particle_filter belief state, 
            # and recursively find the entropy and cost of that action for a decremented lookahead_depth.
            # Note that if lookahead_depth <= 1, this next_action will not be used.
            next_action = self._get_action(this_action, hypothetical_particle_filter, lookahead_depth - 1) # Only used if lookahead_depth > 1.
            entropy, cost = self._predict_entropy_and_cost(next_action, this_action, hypothetical_particle_filter, steps_into_future + 1, lookahead_depth - 1)
            
            # Store entropies and costs for each outcome of this_action.
            outcome_entropies.append(entropy)
            outcome_costs.append(cost)

        # For both the entropies and costs of each outcome for this_action, 
        # the expected future value is the calculated value times the expectation of that outcome, alpha or 1 - alpha.
        expected_entropy = np.dot(outcome_entropies, [alpha, 1 - alpha])
        expected_cost = np.dot(outcome_costs, [alpha, 1 - alpha])
        # Add the cost of getting from previous_action to this_action.
        # This known cost is added after taking the alpha-weighted expectation of the future costs.
        expected_cost += euclidean(previous_action, this_action)

        return expected_entropy * self.discount**steps_into_future, expected_cost * self.discount**steps_into_future
