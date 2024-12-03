import gymnasium as gym
import envs
import os
import numpy as np
import math
import time
from gymnasium.spaces import flatten, unflatten

from loguru import logger

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "mc_control.log")
logger.add(log_file_path, mode="w")


def get_episode(env, policy):
    """Simulate an episode following a policy and returns the history of actions, states and returns"""
    states = []
    actions_idx = []
    rewards = []

    possible_actions = np.arange(int(np.prod(env.action_space.nvec)))

    car_pos_space = env.observation_space["car_position"]
    velocity_space = env.observation_space["velocity"]

    combined_nvec = np.concatenate([car_pos_space.nvec, velocity_space.nvec])

    terminated = False
    state, info = env.reset()

    while not terminated:
        car_position = state["car_position"]
        velocity = state["velocity"]

        combined_state = np.concatenate([car_position, velocity])
        state_ridx = np.ravel_multi_index(combined_state, combined_nvec)

        states.append(state_ridx)

        # Get action from policy
        action_idx = np.random.choice(possible_actions, p=policy[state_ridx])
        actions_idx.append(action_idx)

        state, reward, terminated, _, _ = env.step(action_idx)

        # print(state, action_idx, reward)
        rewards.append(reward)

    assert len(states) == len(rewards) == len(actions_idx)
    return states, actions_idx, rewards


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

    # Optimistic Q values
    Q = np.random.uniform(low=0.99, high=1.01, size=(num_states, num_actions))
    # Q = np.ones(shape=(num_states, num_actions), dtype=np.float64)

    C = np.zeros(shape=(Q.shape))

    # We initialize it to take action uniformely at first so it doesn't get stuck in taking the first action
    pi = np.array([argmax_last(Q[state]) for state in range(num_states)])

    # For debug
    start_q = Q[[13381, 13356, 13331, 13306, 13281, 13256], :]
    tot_rs = []

    for episode in range(num_episodes):
        epsilon = 0.2 + (1 - 0.2) * (1 - episode / num_episodes) ** 1
        # Initialise a policy to be epsilon soft regarding to pi
        b = np.ones(shape=(num_states, num_actions), dtype=np.float64) * (
            epsilon / num_actions
        )
        greedy_actions = pi
        b[np.arange(num_states), greedy_actions] += 1 - epsilon

        states, actions_idx, rewards = get_episode(env, b)

        tot_r = np.sum(rewards)
        tot_rs.append(tot_r)

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

            # policy_action = np.argmax(Q[state])
            policy_action = argmax_last(Q[state])

            pi[state] = policy_action
            if policy_action != action:
                break
            W = W * (1 / b[state, action])

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}")
            print(f"Mean steps over 100 ep: {np.mean(tot_rs)}")
            tot_rs = []

            if not np.all(Q[[13381, 13356, 13331, 13306, 13281, 13256], :] == start_q):
                logger.debug(
                    f"NEW Start Qs: {Q[[13381,13356,13331,13306,13281,13256],:]}"
                )
                logger.debug(f"Pi Start: {pi[[13381,13356,13331,13306,13281,13256]]}")
        # Debug
        start_q = Q[[13381, 13356, 13331, 13306, 13281, 13256], :]

    return pi


if __name__ == "__main__":
    # pi = mc_control(10_000)

    file_path = os.path.join("Chapter05", "racetrack1.txt")
    env = gym.make("Racetrack-v0", grid_file_path=file_path, logging=False)
    obs, _ = env.reset()
    fo = flatten(env.observation_space, obs)
    print(fo)
    a = env.action_space.sample()
    print(a)
    print(flatten(env.action_space, a))

    # TODO how to work with these flatten/unflatten to keep track of states / Q values (because I need indices)
    raise "read here"
    # car_pos_space = env.observation_space["car_position"]
    # velocity_space = env.observation_space["velocity"]

    # combined_nvec = np.concatenate([car_pos_space.nvec, velocity_space.nvec])
    # combined_state = np.concatenate([[3, 16], [2, 1]])
    # state_ridx = np.ravel_multi_index(combined_state, combined_nvec)
    # print(state_ridx)
    # 13225 - 13375
