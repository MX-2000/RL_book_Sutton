import numpy as np
import matplotlib.pyplot as plt

P_HEAD = 0.4
REWARD_WIN = 1
TARGET = 100


def get_best_action_value(V, state, gamma=1):

    biggest_bet = min(state, TARGET - state)
    expected_returns = np.zeros(shape=(biggest_bet + 1,))

    # All the betting possibilities from 1 to biggest_bet
    for a in range(1, biggest_bet + 1):
        r_win = (
            REWARD_WIN if a + state >= 100 else 0
        )  # We only get a reward if we reach 100
        s_prime_win = min(100, state + a)
        s_prime_lose = max(0, state - a)
        return_a = P_HEAD * (r_win + gamma * V[s_prime_win]) + (1 - P_HEAD) * (
            0 + gamma * V[s_prime_lose]
        )
        expected_returns[a] = return_a

    # Rounding trick taken from https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
    return np.argmax(np.round(expected_returns, 5)), expected_returns[1:].max()


def value_iteration(V, pi, gamma=1):
    delta = 1
    threshold = 1e-55

    values = [V.copy()]

    i = 0

    while delta > threshold:

        delta = 0

        for state in range(1, TARGET):
            v = V[state]

            _, expected_return = get_best_action_value(V, state)

            V[state] = expected_return

            delta = max(delta, abs(v - V[state]))

        values.append(V.copy())

        print(f"Iteration {i} delta: {delta}")
        i += 1

    for state in range(1, TARGET):
        max_action, _ = get_best_action_value(V, state)
        pi[state] = max_action

    return values, pi


def main():
    V = np.zeros(shape=(TARGET + 1,), dtype=np.float64)
    pi = np.zeros(shape=(TARGET + 1,), dtype=np.int32)

    values, pi = value_iteration(V, pi)

    xticks = [1, 25, 50, 75, 99]
    plt.subplot(2, 1, 1)
    for i, value_function in enumerate(values[:-1]):
        plt.plot(range(1, TARGET), value_function[1:-1], label=f"Sweep {i+1}")
        if i > 3:
            break
    # Add the final value function
    plt.plot(
        range(1, TARGET),
        values[-1][1:-1],
        label="Final value function",
        color="black",
        linewidth=1.5,
    )

    plt.xlabel("Capital")
    plt.ylabel("Value estimates")
    plt.legend()
    plt.title("Value Estimates during Value Iteration")
    plt.xticks(xticks)

    plt.subplot(2, 1, 2)
    plt.step(range(1, TARGET), pi[1:-1], where="mid")
    plt.xlabel("Capital")
    plt.ylabel("Final policy (stake)")
    plt.title("Final Policy Derived from Value Iteration")
    plt.xticks(xticks)

    plt.tight_layout()

    plt.savefig(f"04_09_P_{P_HEAD}.png")

    plt.show()


if __name__ == "__main__":
    main()
