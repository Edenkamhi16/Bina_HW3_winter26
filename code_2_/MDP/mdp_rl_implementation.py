from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #

    U_final = None
    # TODO:
    # ====== YOUR CODE: ======

    # ========================
    return U_final


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ====== 
    policy = None
    for r in mdp.num_row:
        for c in mdp.num_col:
            if mdp.board[r][c] == ' WALL' or (r, c) in mdp.terminal_states:
                policy[r][c] = None
            else:
                best_action = None
                best_value = float('-inf')
                for action in mdp.actions:
                    probs = mdp.transition_function[action]
                    cur_value = np.sum([probs[next_action] * U[mdp.step((r, c), mdp.actions[next_action])] for next_action in probs.keys()])
                    if cur_value > best_value:
                        best_value = cur_value
                        best_action = action
                policy[r][c] = best_action
    # ========================
    return policy


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    U = None
    # TODO:
    # ====== YOUR CODE: ======
    for r in mdp.num_row:
        for c in mdp.num_col:
            if mdp.board[r][c] == ' WALL' or (r, c) in mdp.terminal_states:
                U[r][c] = 0
            else:
                action = policy[r][c]
                probs = mdp.transition_function[action]
                U[r][c] = float(mdp.get_reward(r, c)) + np.sum([probs[next_action] * ( + mdp.gamma * U[mdp.step((r, c), next_action)]) for next_action in probs.keys()])

    # ========================
    return U


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======

    # ========================
    return optimal_policy


def mc_algorithm(
        sim,
        num_episodes,
        gamma,
        num_rows=3,
        num_cols=4,
        actions=[Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT],
        policy=None,
):
    # Given a simulator, the number of episodes to run, the number of rows and columns in the MDP, the possible actions,
    # and an optional policy, run the Monte Carlo algorithm to estimate the utility of each state.
    # Return the utility of each state.

    V = None

    # ====== YOUR CODE: ======

    # =========================

    return V
