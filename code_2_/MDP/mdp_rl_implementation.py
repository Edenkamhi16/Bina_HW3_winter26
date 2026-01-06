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
    U_final = U_init.copy()
    threshold = epsilon * (1 - mdp.gamma) / mdp.gamma
    
    while(True):
        delta = 0
        U_new = [row[:] for row in U_final]
        
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                state = (row, col)
                #if it is a WALL
                if mdp.board[row][col] == "WALL":
                    U_new[row][col] = None
                    continue

                #if state is TERMINAL get its reward
                if state in mdp.terminal_states:
                    U_new[row][col] = float(mdp.get_reward(state))
                    continue
                
                R_state = float(mdp.get_reward(state))

                U_temp = R_state + mdp.gamma * get_max_sum_action(mdp, U_final, (row, col))
                
                if abs(U_temp - U_final[row][col]) > delta:
                    delta = abs(U_temp - U_final[row][col])

                U_new[row][col] = U_temp
        
        U_final = U_new
        if delta < threshold:
            break
    # ========================
    return U_final

def get_max_sum_action(mdp, U, state):
    best_sum = float('-inf')
    for action in mdp.actions:
        action_sum = 0.0

        probs = mdp.transition_function[action]
        
        for actual_action, prob in zip(mdp.actions, probs):
            next_state = mdp.step(state, actual_action)
            u_next = U[next_state[0]][next_state[1]]
            
            if u_next is None:
                u_next = 0.0

            action_sum += prob * u_next


        if action_sum > best_sum:
            best_sum = action_sum

    return best_sum

def get_max_action(mdp, U, state):
    best_action = None
    best_sum = float('-inf')
    for action in mdp.actions:
        action_sum = 0.0

        probs = mdp.transition_function[action]
        
        for actual_action, prob in zip(mdp.actions, probs):
            next_state = mdp.step(state, actual_action)
            u_next = U[next_state[0]][next_state[1]]
            
            if u_next is None:
                u_next = 0.0

            action_sum += prob * u_next


        if action_sum > best_sum:
            best_sum = action_sum
            best_action = action

    return best_action

def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ====== 
    policy = [[None for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    for r in range (mdp.num_row):
        for c in range (mdp.num_col):
            if mdp.board[r][c] == 'WALL' or (r, c) in mdp.terminal_states:
                policy[r][c] = None
            else:
                policy[r][c] = get_max_action(mdp, U, (r, c))
    # ========================
    return policy


def policy_evaluation(mdp, policy):
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # Flatten state indices for matrix representation
    state_indices = {}
    idx = 0
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if mdp.board[r][c] != 'WALL' : #and (r, c) not in mdp.terminal_states:
                state_indices[(r, c)] = idx
                idx += 1
    n = len(state_indices)

    # Build R vector
    R = np.zeros(n)
    for (r, c), i in state_indices.items():
        R[i] = float(mdp.get_reward((r, c)))

    # Build transition matrix P_pi
    P_pi = np.zeros((n, n))
    for (r, c), i in state_indices.items():
        if (r, c) in mdp.terminal_states:
            continue
        action = policy[r][c]
        probs = mdp.transition_function[action]
        for actual_action, prob in zip(mdp.actions, probs):
            next_state = mdp.step((r, c), actual_action)
            if next_state in state_indices:
                j = state_indices[next_state]
                P_pi[i, j] += prob

    # Solve (I - gamma * P_pi) U = R
    I = np.eye(n)
    A = I - mdp.gamma * P_pi
    U_vec = np.linalg.solve(A, R)

    # Map back to grid
    U = [[None if mdp.board[r][c] == 'WALL' else 0.0 for c in range(mdp.num_col)] for r in range(mdp.num_row)]
    for (r, c), i in state_indices.items():
        U[r][c] = U_vec[i]


    return U


def policy_iteration(mdp, policy_init):
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    new_policy = [row[:] for row in policy_init]
    # ====== YOUR CODE: ======
    while(True):
        U = policy_evaluation(mdp, new_policy)
        unchanged = True

        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                state = (row, col)

                # skip WALL
                if mdp.board[row][col] == "WALL":
                    continue

                # skip TERMINAL
                if state in mdp.terminal_states:
                    continue

                if get_max_sum_action(mdp, U, (row, col)) > get_sum_policy(mdp, U, (row, col), new_policy[row][col]):
                    new_policy[row][col] = get_max_action(mdp, U, state)
                    unchanged = False
        if unchanged:
            break
    
    optimal_policy = new_policy        
    # ========================
    return optimal_policy

def get_sum_policy(mdp, U, state, action):
    sum = 0.0
    probs = mdp.transition_function[action]
    
    for actual_action, prob in zip(mdp.actions, probs):
        next_state = mdp.step(state, actual_action)
        u_next = U[next_state[0]][next_state[1]]
        
        if u_next is None:
            u_next = 0.0

        sum += prob * u_next

    return sum

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
