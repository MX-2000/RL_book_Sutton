import gymnasium
import envs
import numpy as np


def run_simple_bandit(steps=100, epsilon=0.1):

    arms_nb = 10
    env = gymnasium.make("KArmedBandit-v0", k=arms_nb, stationnary=True)
    _, info = env.reset()

    q_estimates = [0 for i in range(arms_nb)]
    action_count = [0 for i in range(arms_nb)]

    rewards = []
    optimal_actions = []

    for i in range(steps):

        if np.random.random() < epsilon:
            action = np.random.randint(0, arms_nb)
        else:
            action = np.argmax(q_estimates)

        optimal = 1 if action == np.argmax(info["q_star"]) else 0

        _, reward, _, _, info = env.step(action)
        action_count[action] += 1
        q_estimates[action] = q_estimates[action] + (1 / action_count[action]) * (
            reward - q_estimates[action]
        )
        rewards.append(reward)
        optimal_actions.append(optimal)

    return rewards, optimal_actions


def get_metrics(rewards, optimal_actions):
    average_rewards = [sum(rewards[: i + 1]) / (i + 1) for i in range(len(rewards))]
    optimal_pct = [
        sum(optimal_actions[: i + 1]) * 100 / (i + 1)
        for i in range(len(optimal_actions))
    ]

    return average_rewards, optimal_pct


if __name__ == "__main__":
    rewards, optimal_actions = run_simple_bandit(steps=1000, epsilon=0.1)
    rewards, optimal_actions = get_metrics(rewards, optimal_actions)
    print(rewards[-1], optimal_actions[-1])
