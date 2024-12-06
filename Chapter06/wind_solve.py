import gymnasium as gym
import envs
import os
import numpy as np
import time

from loguru import logger

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "wind.log")
logger.add(log_file_path, mode="w")

"""
Credits to https://github.com/vojtamolda/reinforcement-learning-an-introduction/blob/main/chapter06/sarsa.py
"""


def run_episode(env, policy, render=True):
    """Follow policy through an environment's episode and return an array of collected rewards"""
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    state, _ = env.reset()
    if render:
        env.render()

    done = False
    rewards = []
    while not done:
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        action = np.argmax(policy[state_ridx])
        state, reward, done, _, info = env.step(action)
        rewards += [reward]

        if render:
            env.render()

    if render:
        import matplotlib.pyplot as plt

        plt.show()

    return rewards


def sarsa(env, num_episodes, eps0=0.5, alpha=0.5, debug=False):
    """On-policy Sarsa algorithm per Chapter 6.4 (with exploration rate decay)"""
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    # Number of available actions and maximal state ravel index
    n_action = env.action_space.n
    n_state_ridx = (
        np.ravel_multi_index(env.observation_space.nvec - 1, env.observation_space.nvec)
        + 1
    )

    # Initialization of action value function
    q = np.zeros([n_state_ridx, n_action], dtype=np.float64)

    # Initialize policy to equal-probable random
    policy = np.ones([n_state_ridx, n_action], dtype=np.float64) / n_action

    # I want to observe the q_values of the starting position change start = (0, 3)
    # I also want to observe the q_values of the positions close to the goal goal = (7, 3)
    # I also want to observe the policy probabilities change close to those positions
    starting_tiles = [[0, 3], [0, 4], [1, 3], [0, 2]]
    starting_tiles_r = [
        np.ravel_multi_index(tile, env.observation_space.nvec)
        for tile in starting_tiles
    ]
    end_pod = [[7, 3], [6, 3], [8, 3], [7, 4], [7, 2]]
    end_idx = [
        np.ravel_multi_index(tile, env.observation_space.nvec) for tile in end_pod
    ]

    history = [0] * num_episodes
    for episode in range(num_episodes):

        if episode % 1000 == 0 and debug:
            logger.debug(f"=============================================")
            logger.debug(f"Ep: {episode}")
            logger.debug(f"Q_start_values: {q[starting_tiles_r,:]}")
            logger.debug(f"pi start: {policy[starting_tiles_r]}")
            logger.debug(f"Q_end_values: {q[end_idx,:]}")
            logger.debug(f"pi_end: {policy[end_idx]}")

        # Reset the environment
        state, _ = env.reset()
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        action = np.random.choice(n_action, p=policy[state_ridx])

        done = False
        while not done:

            # Step the environment forward and check for termination
            next_state, reward, done, _, info = env.step(action)
            next_state_ridx = np.ravel_multi_index(
                next_state, env.observation_space.nvec
            )
            next_action = np.random.choice(n_action, p=policy[next_state_ridx])

            # Update q values
            q[state_ridx, action] += alpha * (
                reward + q[next_state_ridx, next_action] - q[state_ridx, action]
            )

            # Extract eps-greedy policy from the updated q values
            eps = eps0 / (episode + 1)
            policy[state_ridx, :] = eps / n_action
            policy[state_ridx, np.argmax(q[state_ridx])] = 1 - eps + eps / n_action
            assert np.allclose(np.sum(policy, axis=1), 1)

            # Prepare the next q update
            state_ridx = next_state_ridx
            action = next_action
            history[episode] += 1

    return q, policy, history


if __name__ == "__main__":
    env = gym.make("WindyGridworldEnv-v0")

    q, policy, history = sarsa(env, num_episodes=500)
    run_episode(env, policy)
