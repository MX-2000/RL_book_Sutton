import gymnasium as gym
import envs
import os
import numpy as np
import math
import time
from gymnasium.spaces import flatten, unflatten

from loguru import logger
import matplotlib.pyplot as plt

import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "mc_control.log")
logger.add(log_file_path, mode="w")

mapped_action = [-1, 0, 1]


def get_episode(env, policy, render=False):
    """Simulate an episode following a policy and returns the history of actions, states and returns"""
    states_idx = []  # Index of 1D flattened state
    actions_idx = []  # Index of 1D flattened actions
    rewards = []

    possible_actions = np.arange(int(np.prod(env.action_space.nvec)))  # [0,1,2,3,4...]

    car_pos_space = env.observation_space["car_position"]
    velocity_space = env.observation_space["velocity"]

    combined_nvec = np.concatenate(
        [car_pos_space.nvec, velocity_space.nvec]
    )  # [max_x, max_y, max_vx, max_vy]

    terminated = False
    state, info = env.reset()

    if render:
        env.render()

    while not terminated:
        car_position = state["car_position"]
        velocity = state["velocity"]

        combined_state = np.concatenate([car_position, velocity])
        # Transform multi-dimensional state into 1D index based on all possible states
        state_ridx = np.ravel_multi_index(combined_state, combined_nvec)

        states_idx.append(state_ridx)

        # Get action from policy. This returns the value of the array but possible_action contains indices, so we get an idx
        action_idx = np.random.choice(possible_actions, p=policy[state_ridx])
        actions_idx.append(action_idx)

        # We need to transform the idx into an action the environment can understand
        action = np.unravel_index(action_idx, env.action_space.nvec, order="C")
        # MultiDiscrete samples are > 0 so we need to map the action back to real added velocity
        action = mapped_action[action[0]], mapped_action[action[1]]

        state, reward, terminated, _, _ = env.step(action)

        # print(state, action_idx, reward)
        rewards.append(reward)

        if render:
            env.render()

    assert len(states_idx) == len(rewards) == len(actions_idx)
    return states_idx, actions_idx, rewards


def argmax_last(arr):
    """Taking first indices in argmax doesn't work well with what I want because in starting states first indice actions are doing nothing"""
    arr = np.asarray(arr)
    max_value = np.max(arr)
    max_indices = np.where(arr == max_value)[0]
    return max_indices[-1]


def mc_control(num_episodes, gamma=1, epsilon=1):
    file_path = os.path.join("Chapter05", "racetrack1.txt")
    env = gym.make("Racetrack-v0", grid_file_path=file_path, logging=False)
    obs_sp = env.observation_space
    act_sp = env.action_space

    # We know obs_sp is Dict with MultiDiscrete spaces
    num_states = math.prod([math.prod(space.nvec) for key, space in obs_sp.items()])
    # We know act_sp is MultiDiscrete
    num_actions = math.prod(act_sp.nvec)

    print(f"Possible states: {num_states}")

    # Optimistic Q values
    Q = np.random.uniform(low=0.99, high=1.01, size=(num_states, num_actions))
    # Q = np.ones(shape=(num_states, num_actions), dtype=np.float64)

    C = np.zeros(shape=(Q.shape))

    # Init our greedy policy, taking the argmax of Q means random action at first because ou Q values are sligthly random.
    pi = np.zeros(shape=(num_states, num_actions), dtype=np.float64)
    pi[np.arange(num_states), np.argmax(Q, axis=1)] = 1.0

    # For debug
    tot_rs = []
    updates = 0

    for episode in range(num_episodes):
        epsilon = 0.2 + (1 - 0.2) * (1 - episode / num_episodes) ** 1
        # Initialise a policy to be epsilon soft regarding to pi
        b = np.ones(shape=(num_states, num_actions), dtype=np.float64) * (
            epsilon / num_actions
        )
        greedy_actions = np.argmax(pi, axis=1)
        b[np.arange(num_states), greedy_actions] += 1 - epsilon

        # Completely random
        # b = np.ones(shape=(num_states, num_actions), dtype=np.float64) / (num_actions)

        states, actions_idx, rewards = get_episode(env, b)

        steps_nb = len(rewards)
        tot_rs.append(steps_nb)

        G = 0
        W = 1

        for step in range(len(states) - 1, -1, -1):
            state = states[step]
            action = actions_idx[step]
            reward = rewards[step]

            G = gamma * G + reward
            C[state, action] = C[state, action] + W

            q_update = (W / C[state, action]) * (G - Q[state, action])
            Q[state, action] = Q[state, action] + q_update

            if q_update:
                updates += 1

            policy_action = np.argmax(Q[state])

            pi[state] = np.zeros(shape=num_actions)
            pi[state, policy_action] = 1.0

            if policy_action != action:
                break
            W = W * (1 / b[state, action])

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, epsilon: {epsilon}")
            print(f"Mean steps over 100 ep: {np.mean(tot_rs)}")
            print(f"Tot updates: {updates}")
            tot_rs = []

    return pi


if __name__ == "__main__":
    pi = mc_control(100_000)

    with open("policy.pkl", "wb") as f:
        pickle.dump(pi, f)

    file_path = os.path.join("Chapter05", "racetrackTest.txt")
    env = gym.make(
        "Racetrack-v0",
        grid_file_path=file_path,
        noise=False,
        logging=False,
    )
    print("Running episode")
    states, actions_idx, rewards = get_episode(env, pi, render=True)
    print("Episode over, plotting")
    plt.show()
