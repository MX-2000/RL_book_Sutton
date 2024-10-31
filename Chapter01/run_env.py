from cProfile import label
import gymnasium
import envs
import numpy as np
import matplotlib.pyplot as plt

# SEED = 6
# np.random.seed(SEED)


def run_bandit(
    steps=100,
    epsilon=0.1,
    ucb=False,
    c=2,
    gradient=False,
    alpha_grad=0.1,
    stationnary=True,
    k=10,
    method="sample average",
    alpha=0.1,
    initial_estimates=0,
):

    arms_nb = k
    env = gymnasium.make("KArmedBandit-v0", k=arms_nb, stationnary=stationnary)
    _, info = env.reset()

    q_estimates = np.zeros(shape=k) + initial_estimates
    # [initial_estimates for i in range(arms_nb)]

    action_count = np.zeros(arms_nb)
    # [0 for i in range(arms_nb)]

    rewards = np.zeros((steps,))
    optimal_actions = np.zeros((steps,))

    H_ = np.zeros((k,))
    grad_probs = np.exp(H_) / np.sum(np.exp(H_))
    baseline = 0

    for i in range(steps):
        if gradient:
            action = np.random.choice(k, p=grad_probs)
        elif ucb:
            ucb_estimates = np.where(
                action_count == 0,
                np.inf,
                q_estimates + c * np.sqrt(np.log(i + 1) / action_count),
            )
            action = np.argmax(ucb_estimates)
        else:
            if np.random.random() < epsilon:
                action = np.random.randint(0, arms_nb)
            else:
                action = np.argmax(q_estimates)

        optimal = 1 if action == np.argmax(info["q_star"]) else 0

        _, reward, _, _, info = env.step(action)
        action_count[action] += 1

        if gradient:
            baseline += alpha_grad * (reward - baseline)
            for j in range(k):
                if j == action:
                    H_[action] = H_[action] + alpha_grad * (reward - baseline) * (
                        1 - grad_probs[j]
                    )
                else:
                    H_[j] = H_[j] - alpha_grad * (reward - baseline) * grad_probs[j]
            grad_probs = np.exp(H_) / np.sum(np.exp(H_))

        elif method == "sample average":

            q_estimates[action] = q_estimates[action] + (1 / action_count[action]) * (
                reward - q_estimates[action]
            )
        else:
            q_estimates[action] = q_estimates[action] + alpha * (
                reward - q_estimates[action]
            )

        rewards[i] = reward
        optimal_actions[i] = optimal

    return rewards, optimal_actions


def get_metrics(rewards, optimal_actions):
    average_rewards = [sum(rewards[: i + 1]) / (i + 1) for i in range(len(rewards))]
    optimal_pct = [
        sum(optimal_actions[: i + 1]) * 100 / (i + 1)
        for i in range(len(optimal_actions))
    ]

    return average_rewards, optimal_pct


def average_bandits(n=500, **kwargs):
    rewards = np.zeros((n, kwargs["steps"]))
    optimal_actions = np.zeros((n, kwargs["steps"]))

    for i in range(n):
        rewards[i:], optimal_actions[i] = run_bandit(**kwargs)

    mean_rewards = rewards.mean(axis=0)
    optimal_perc = optimal_actions.mean(axis=0)

    return mean_rewards, optimal_perc


if __name__ == "__main__":
    STEPS = 1000

    label1 = "gradient"
    label2 = "e greedy"
    rewards1, optimal_actions1 = average_bandits(
        steps=STEPS,
        epsilon=0.1,
        gradient=True,
        alpha_grad=0.1,
        stationnary=True,
        method="constant",
        alpha=0.1,
        initial_estimates=0,
    )

    rewards2, optimal_actions2 = average_bandits(
        steps=STEPS,
        epsilon=0.1,
        stationnary=True,
        method="constant",
        alpha=0.1,
        initial_estimates=5,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(list(range(STEPS)), rewards1, color="blue", label=label1)
    ax1.plot(list(range(STEPS)), rewards2, color="red", label=label2)
    # Adding labels and title
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average Reward")
    ax1.legend()  # Show the legend to distinguish between the lines

    ax2.plot(
        list(range(STEPS)),
        optimal_actions1,
        color="blue",
        label=label1,
    )
    ax2.plot(
        list(range(STEPS)),
        optimal_actions2,
        color="red",
        label=label2,
    )
    # Adding labels and title
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("%Optimal action")
    ax2.legend()  # Show the legend to distinguish between the lines

    plt.tight_layout()
    # Display the plot

    # plt.savefig("Ex2_6_diff_init_q_values_nonstationnary.png", format="png", dpi=300)

    plt.show()
