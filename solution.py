import sys
import time
from typing import Dict

import numpy as np
from constants import *
from environment import *
from state import State
from collections import deque 
from functools import lru_cache

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

"""


class Solver:

    def __init__(self, environment: Environment):
        self.environment = environment
        #
        # TODO: Define any class instance variables you require (e.g. dictionary mapping state to VI value) here.
        #
        self.states = []
        self.state_values = dict()
        self.policy = dict()
        self.epsilon = environment.epsilon
        self.gamma = environment.gamma
        self.iteration = 0
        self.previous_state_values = {}
        self.transition_cache = {}

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5, 6]

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        #
        # TODO: Implement any initialisation for Value Iteration (e.g. building a list of states) here. You should not
        #  perform value iteration in this method.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #

        self.states = self.get_reachable_states()
    
        state_distances = {}
        max_distance = 0
        for state in self.states:
            distance = self.hex_manhattan_distance(state)
            state_distances[state] = distance
            max_distance = max(max_distance, distance)
            normalized_distance = state_distances[state] / max_distance
            proximity_value = 1 - normalized_distance
            self.state_values[state] = proximity_value
        
        self.policy = {state: REVERSE for state in self.states}
        
    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Value Iteration has reached convergence here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        if self.iteration == 0:
            return False
        
        differences = [abs(self.state_values[state] - self.previous_state_values.get(state, 0)) for state in self.states]
        max_diff = max(differences)
        
        return max_diff < self.epsilon

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once) with in-place updates.
        """
        self.previous_state_values = self.state_values.copy()
        self.iteration += 1
        
        for state in self.states:
            action_values = {}
            for action in BEE_ACTIONS:
                action_value = 0
                for probability, next_state, reward in self.get_transition_outcome(state, action):
                    action_value += probability * (reward + (self.gamma * self.state_values[next_state]))
                action_values[action] = action_value
            
            self.state_values[state] = max(action_values.values())
            self.policy[state] = max(action_values, key=action_values.get)

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while True:
            self.vi_iteration()

            # NOTE: vi_iteration is always called before vi_is_converged
            if self.vi_is_converged():
                break

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        #
        # TODO: Implement code to return the value V(s) for the given state (based on your stored VI values) here. If a
        #  value for V(s) has not yet been computed, this function should return 0.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        return self.state_values.get(state, 0)

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored VI values) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        return self.policy.get(state, REVERSE)

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        #
        # TODO: Implement any initialisation for Policy Iteration (e.g. building a list of states) here. You should not
        #  perform policy iteration in this method. You should assume an initial policy of always move FORWARDS.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        self.previous_la_policy = {}
        self.previous_policy = {}
        self.states = self.get_reachable_states()

        self.state_values = {state: 0 for state in self.states}

        self.policy = {state: FORWARD for state in self.states}
    
    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Policy Iteration has reached convergence here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        if self.iteration == 0:
            return False

        return self.policy == self.previous_policy

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        #
        # TODO: Implement code to perform a single iteration of Policy Iteration (evaluation + improvement) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        self.previous_policy = self.policy.copy()
        self.iteration += 1

        self.policy_evaluation()
        self.policy_improvement()
            
    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while True:
            self.pi_iteration()

            # NOTE: pi_iteration is always called before pi_is_converged
            if self.pi_is_converged():
                break

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored PI policy) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        return self.policy.get(state, FORWARD)

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: Add any additional methods here
    #
    #   
    def policy_evaluation(self, max_iterations=80):
        iteration = 0
        while iteration < max_iterations:
            max_diff = 0
            self.previous_state_values = self.state_values.copy()
            
            for state in self.states:
                old_value = self.state_values[state]
                action = self.policy[state]
                new_value = 0
                
                for probability, next_state, reward in self.get_transition_outcome(state, action):
                    new_value += probability * (reward + self.gamma * self.previous_state_values[next_state])
                
                self.state_values[state] = new_value
                max_diff = max(max_diff, abs(old_value - new_value))            
            if max_diff < self.epsilon:
                return
            
            iteration += 1
        
    def policy_improvement(self):
        for state in self.states:
            # Keep track of maximum value
            action_values = dict()
            for action in BEE_ACTIONS:
                action_value = 0
                for probability, next_state, reward in self.get_transition_outcome(state, action):
                    action_value += probability * (reward + self.gamma * self.state_values[next_state])

                action_values[action] = action_value

            # Update policy
            self.policy[state] = max(action_values, key=action_values.get)

    @lru_cache(maxsize=None)
    def hex_manhattan_distance(self, state: State):
        """
        Calculate the total minimum distance of the hexagonal Manhattan distance 
        between the widget centers from the state and the target, considering orientation.
        """
        total_distance = 0
        widget_cells = [widget_get_occupied_cells(self.environment.widget_types[i], 
                                                                state.widget_centres[i],
                                                                state.widget_orients[i]) 
                        for i in range(self.environment.n_widgets)]
        
        for target in self.environment.target_list:
            min_distance = float('inf')
            for i, cells in enumerate(widget_cells):
                if target in cells:
                    min_distance = 0
                    break
                else:
                    # Find the closest cell in the widget to the target
                    for cell in cells:
                        distance = self.hex_distance(cell, target)
                        if distance < min_distance:
                            min_distance = distance
            
            total_distance += min_distance
        
        return total_distance

    @lru_cache(maxsize=None)
    def hex_distance(self, pos1, pos2):
        """Calculate the hexagonal Manhattan distance between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        dx = x2 - x1
        dy = y2 - y1
        
        return max(abs(dx), abs(dy)) + (abs(dx + dy) - max(abs(dx), abs(dy))) // 2
        
    def get_reachable_states(self):
        environment = self.environment
        initial_state = environment.get_init_state()
        visited_states = set()
        dq = deque()

        visited_states.add(initial_state)
        dq.append(initial_state)
        while dq:
            s = dq.popleft()
            for action in BEE_ACTIONS:
                _, next_state = environment.apply_dynamics(s, action)
                if next_state not in visited_states:
                    dq.append(next_state)
                    visited_states.add(next_state)

        return list(visited_states)
    
    def stoch_actions(self, action):
        """
        Takes in an action and returns a map of movements with their probabilites
        :param: action: the action the agent takes
        :return: {[movements]: probability, ...}
        """

        double_mv_probs = self.environment.double_move_probs[action]
        drift_cw_probs = self.environment.drift_cw_probs[action]
        drift_ccw_probs = self.environment.drift_ccw_probs[action]
        no_drift_probs = 1 - drift_cw_probs - drift_ccw_probs
        no_double_mv_probs = 1 - double_mv_probs
        
        return [
            # No double and No cw/ccw
            ([action], no_drift_probs * no_double_mv_probs), 
            # No double and cw
            ([SPIN_RIGHT, action], no_double_mv_probs * drift_cw_probs),
            # No double and ccw
            ([SPIN_LEFT, action], no_double_mv_probs * drift_cw_probs),
            # Double and No cw/ccw
            ([action, action], double_mv_probs),
            # Double and cw
            ([SPIN_RIGHT, action , action], double_mv_probs * drift_cw_probs),
            # Double and ccw
            ([SPIN_LEFT, action, action], double_mv_probs * drift_ccw_probs)
        ]
    
    @lru_cache(maxsize=None)
    def get_transition_outcome(self, state: State, action):
        """
        Retrieve the possible outcomes for the given state and action.
        :param state: the current state
        :param action: the action to transition to another state
        :return [probability, next_state, reward]: A list containing the probability, the next state and 
        the reward from taking the action
        """
        # terminal state
        if self.environment.is_solved(state):
            return ((1.0, state, 0.0),)

        if (state, action) not in self.transition_cache:
            outcomes = []
            for actual_actions, probability in self.stoch_actions(action):
                next_state = state
                reward = 0
                for m in actual_actions:
                    new_reward, next_state = self.environment.apply_dynamics(next_state, m)
                    reward += new_reward
                outcomes.append((probability, next_state, reward))
            self.transition_cache[(state, action)] = tuple(outcomes)

        return self.transition_cache[(state, action)]