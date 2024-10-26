import gymnasium
import envs
import numpy as np
import matplotlib.pyplot as plt


def run_bandit(
    steps=100,
    epsilon=0.1,
    stationnary=True,
    k=10,
    method="sample average",
    alpha=0.1,
    initial_estimates=0,
):

    arms_nb = k
    env = gymnasium.make("KArmedBandit-v0", k=arms_nb, stationnary=stationnary)
    _, info = env.reset(seed=6)

    q_estimates = [initial_estimates for i in range(arms_nb)]
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

        if method == "sample average":

            q_estimates[action] = q_estimates[action] + (1 / action_count[action]) * (
                reward - q_estimates[action]
            )
        else:
            q_estimates[action] = q_estimates[action] + alpha * (
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
    STEPS = 25_000
    rewards, optimal_actions = run_bandit(
        steps=STEPS,
        epsilon=0.1,
        stationnary=False,
        method="constant",
        alpha=0.1,
        initial_estimates=0,
    )
    rewards1, optimal_actions1 = get_metrics(rewards, optimal_actions)
    # print(f"Epsilon 0.1: {rewards1[-1]}, {optimal_actions1[-1]}")

    rewards, optimal_actions = run_bandit(
        steps=STEPS,
        epsilon=0.1,
        stationnary=False,
        method="constant",
        alpha=0.1,
        initial_estimates=5,
    )
    rewards2, optimal_actions2 = get_metrics(rewards, optimal_actions)
    # print(f"Epsilon 0.01: {rewards2[-1]}, {optimal_actions2[-1]}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(list(range(STEPS)), rewards1, color="blue", label="Initial Q = 0")
    ax1.plot(list(range(STEPS)), rewards2, color="red", label="Initial Q = +5")
    # Adding labels and title
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average Reward")
    ax1.legend()  # Show the legend to distinguish between the lines

    ax2.plot(
        list(range(STEPS)),
        optimal_actions1,
        color="blue",
        label="Initial Q = 0",
    )
    ax2.plot(
        list(range(STEPS)),
        optimal_actions2,
        color="red",
        label="Initial Q = +5",
    )
    # Adding labels and title
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("%Optimal action")
    ax2.legend()  # Show the legend to distinguish between the lines

    plt.tight_layout()
    # Display the plot

    plt.savefig("Diff_init_Q_values_nonstationnary.png", format="png", dpi=300)

    plt.show()
