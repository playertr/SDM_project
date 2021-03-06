import numpy as np
from shapely.geometry import Point

from StateEstimator import ParticleFilter
from World import World



def get_action(previous_action, particles, lookahead_depth):
    """
    Returns the action, a Point object, to be taken next given the belief state 
    represented by particles. Recursively looks ahead in concert with predict_entropy_and_cost.

    Uses Information Gain / Cost as the discerning objective between candidate_actions.
    """

    # The output is not used in this base case, so this is a shortcut.
    if lookahead_depth == 0:
        return None

    current_entropy = get_entropy(particles)
    candidate_actions = get_candidate_actions(particles)
    candidate_action_objectives = []
    
    # An action is just a tuple (x, y).

    for action in candidate_actions:
        if previous_action is None: previous_action = action # For the first action in the simulation.

        action_entropy, action_cost = predict_entropy_and_cost(action, previous_action, particles, 1, lookahead_depth)
        objective = (current_entropy - action_entropy) / action_cost
        candidate_action_objectives.append(objective)
    
    return candidate_actions[np.argmax(candidate_action_objectives)]


def predict_entropy_and_cost(this_action, previous_action, particles, steps_into_future, lookahead_depth):
    """
    Returns a tuple of the expected entropy and cost of taking this_action after having taken previous_action, 
    considered across lookahead_depth future actions.
    """

    # Base case.
    if lookahead_depth == 0:
        return (get_entropy(particles), 0)

    # Recursive step.
    alpha = get_expectation_of_hit(particles, this_action)
    outcome_entropies = []
    outcome_costs = []
    for outcome in [True, False]: # True implies a hit, False implies a miss.
        # Create an updated copy of particles as though this_action had been taken and resulted in outcome.
        hypothetical_particles = particles.copy()
        hypothetical_particles.update((this_action.x, this_action.y, outcome))
        
        # Compute the next_action that would be taken given the hypothetical_particles belief state, 
        # and recursively find the entropy and cost of that action for a decremented lookahead_depth.
        # Note that if lookahead_depth <= 1, this next_action will not be used.
        next_action = get_action(this_action, hypothetical_particles, lookahead_depth - 1) # Only used if lookahead_depth > 1.
        entropy, cost = predict_entropy_and_cost(next_action, this_action, hypothetical_particles, steps_into_future + 1, lookahead_depth - 1)
        
        # Store entropies and costs for each outcome of this_action.
        outcome_entropies.append(entropy)
        outcome_costs.append(cost)

    # For both the entropies and costs of each outcome for this_action, 
    # the expected future value is the calculated value times the expectation of that outcome, alpha or 1 - alpha.
    expected_entropy = np.dot(outcome_entropies, [alpha, 1 - alpha])
    expected_cost = np.dot(outcome_costs, [alpha, 1 - alpha])
    # Add the cost of getting from previous_action to this_action.
    # This known cost is added after taking the alpha-weighted expectation of the future costs.
    expected_cost += get_cost(previous_action, this_action)

    return expected_entropy * discount**steps_into_future, expected_cost * discount**steps_into_future
