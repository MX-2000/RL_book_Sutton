import gymnasium as gym
import envs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import time
from mpl_toolkits.mplot3d import Axes3D


def highest_poisson(l, p):
    n = 0
    while poisson.cdf(n, l) < p:
        n += 1
    return n


MAX_MOVE = 5
MAX_CARS = 20
RENT_REWARD = 10
MOVING_COST = 2
LAMBDAS = np.array([[3, 4], [3, 2]], dtype=np.int32)
MAX_POISSON = highest_poisson(np.max(LAMBDAS), 0.99)
GAMMA = 0.9
MAX_PARKING = 10
OVERPARK_COST = 4
ORIGINAL = False


def get_expected_return(V, state, pi_s, poisson_probs, gamma):
    # Apply action and calculate moving cost
    cars_loc1 = min(state[0] - pi_s, MAX_CARS)
    cars_loc2 = min(state[1] + pi_s, MAX_CARS)

    moving_cost = MOVING_COST * abs(pi_s)

    parking_cost = 0
    if not ORIGINAL:
        if pi_s > 0:
            # First car to move from loc1 to loc2 is free
            moving_cost -= MOVING_COST

        # Any cars kept above MAX_PARKING cost an extra 4$ for all cars in a location
        if cars_loc1 > MAX_PARKING:
            parking_cost += 4
        if cars_loc2 > MAX_PARKING:
            parking_cost += 4

    expected_return = -(moving_cost + parking_cost)

    # Iterate over possible rental requests at both locations
    for req1 in range(0, MAX_POISSON + 1):
        prob_req1 = poisson_probs[LAMBDAS[0, 0]][req1]
        num_rentals_loc1 = min(cars_loc1, req1)
        reward_loc1 = RENT_REWARD * num_rentals_loc1
        cars_loc1_end = cars_loc1 - num_rentals_loc1

        for req2 in range(0, MAX_POISSON + 1):
            prob_req2 = poisson_probs[LAMBDAS[1, 0]][req2]
            num_rentals_loc2 = min(cars_loc2, req2)
            reward_loc2 = RENT_REWARD * num_rentals_loc2
            cars_loc2_end = cars_loc2 - num_rentals_loc2

            prob_req = prob_req1 * prob_req2
            immediate_reward = reward_loc1 + reward_loc2

            # Iterate over possible returns at both locations
            for ret1 in range(0, MAX_POISSON + 1):
                prob_ret1 = poisson_probs[LAMBDAS[0, 1]][ret1]
                cars_loc1_next = min(cars_loc1_end + ret1, MAX_CARS)

                for ret2 in range(0, MAX_POISSON + 1):
                    prob_ret2 = poisson_probs[LAMBDAS[1, 1]][ret2]
                    cars_loc2_next = min(cars_loc2_end + ret2, MAX_CARS)

                    prob = prob_req * prob_ret1 * prob_ret2
                    next_state = (cars_loc1_next, cars_loc2_next)
                    expected_return += prob * (immediate_reward + gamma * V[next_state])

    return expected_return


def build_poisson_probs(max, lambdas):

    poisson_probs = {}
    for rate in [LAMBDAS[0, 0], LAMBDAS[0, 1], LAMBDAS[1, 0], LAMBDAS[1, 1]]:
        probs = poisson.pmf(np.arange(0, MAX_POISSON + 1), rate)
        probs[-1] = 1 - probs[:-1].sum()  # Adjust the last probability
        poisson_probs[rate] = probs

    return poisson_probs


def policy_evaluation(V, pi, poisson_probs, gamma):
    delta = 1
    threshold = 1e-2

    while delta > threshold:
        delta = 0
        for state in np.ndindex(V.shape):
            v = V[state]
            pi_s = pi[state]

            expected_return = get_expected_return(V, state, pi_s, poisson_probs, gamma)

            V[state] = expected_return
            delta = max(delta, abs(v - V[state]))
        print("Iteration delta:", delta)

    return V


def policy_improvment(V, pi, poisson_probs, gamma):
    policy_stable = True

    for state in np.ndindex(V.shape):
        old_action = pi[state]

        action_returns = np.full(shape=(2 * MAX_MOVE + 1,), fill_value=-np.inf)

        # We need to find the argmax for this
        for action in range(2 * MAX_MOVE + 1):
            real_action = (
                action - MAX_MOVE
            )  # Real actions range from -MAX_MOVE to MAX_MOVE

            expected_return = get_expected_return(
                V, state, real_action, poisson_probs, gamma
            )

            action_returns[action] = expected_return

        new_action = (
            np.argmax(action_returns) - MAX_MOVE
        )  # Real actions range from -MAX_MOVE to MAX_MOVE

        pi[state] = new_action

        if old_action != new_action:
            policy_stable = False

    return pi, policy_stable


def policy_iteration(gamma):
    policies = []
    values = []

    V = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.float32)
    pi = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.int32)

    values.append(V.copy())
    policies.append(pi.copy())

    poisson_probs = build_poisson_probs(MAX_POISSON, LAMBDAS)

    i = 0
    while True:
        V = policy_evaluation(V, pi, poisson_probs, gamma)
        pi, policy_stable = policy_improvment(V, pi, poisson_probs, gamma)

        print(f"Iteration {i}, Policy stable? {policy_stable}")
        values.append(V.copy())
        policies.append(pi.copy())

        i += 1
        if policy_stable:
            break
        else:
            print(pi)

    return values, policies


def random_strat(env, days):
    obs, info = env.reset()

    car_moves = np.zeros((days,))
    cars = np.zeros((days, 2))
    rewards = np.zeros((days,))

    for day in range(days):

        possible_actions = np.arange(
            max(-obs[1], -MAX_MOVE), min(obs[0], MAX_MOVE) + 1, dtype=np.int32
        )
        # action = env.observation_space.sample()
        action = np.random.choice(possible_actions)

        obs, reward, terminated, truncated, info = env.step(action)

        car_moves[day] = action
        rewards[day] = reward
        cars[day] = obs

        if terminated or truncated:
            break

    return car_moves, cars, rewards


def main():

    # env = gym.make(
    #     "JacksRental-v0",
    #     max_move=MAX_MOVE,
    #     max_cars=MAX_CARS,
    #     rent_r=RENT_REWARD,
    #     moving_cost=MOVING_COST,
    #     max_poisson=MAX_POISSON,
    #     p_lambdas=LAMBDAS,
    #     logging=False,
    # )

    # car_moves, cars, rewards = random_strat(env, days)
    # return car_moves, cars, rewards

    values, policies = policy_iteration(GAMMA)

    print(f"{len(policies)} policies")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(5):
        ax = axes[i]
        policy = policies[i]
        contour = ax.contour(
            np.arange(MAX_CARS + 1),
            np.arange(MAX_CARS + 1),
            policy,
            levels=np.arange(-5, 6),
        )
        ax.clabel(contour, inline=True, fontsize=8)
        ax.set_title(f"$\\pi_{{{i}}}$")
        ax.set_xlabel("#Cars at second location")
        ax.set_ylabel("#Cars at first location")

    # 3D Surface plot for the value function
    ax = axes[-1]
    X, Y = np.meshgrid(np.arange(MAX_CARS + 1), np.arange(MAX_CARS + 1))
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    surf = ax.plot_surface(X, Y, values[-1], cmap="viridis", edgecolor="k")
    ax.set_xlabel("#Cars at second location")
    ax.set_ylabel("#Cars at first location")
    ax.set_zlabel("Value")
    ax.set_title(f"$V(\\pi_{{{len(policies)}}})$")

    # Adjust layout
    plt.tight_layout()
    plt.show()

    fig.canvas.draw()

    # Save the figure with tight bounding box
    plt.savefig(
        f"04_07_{'original' if ORIGINAL else 'modified'}.png", bbox_inches="tight"
    )


def plot_agent(car_moves, rewards, cars):

    fig, axs = plt.subplots(3, 1, figsize=(15, 8))

    axs[0].plot(car_moves)
    axs[0].set_title("Car moves overtime")
    axs[0].set_xlabel("days")
    axs[0].set_ylabel("Car moves")

    axs[1].bar(range(days), rewards)
    axs[1].set_title("Rewards overtime")
    axs[1].set_xlabel("days")
    axs[1].set_ylabel("Rewrad")

    axs[2].bar(range(days), cars[:, 0], color="blue", label="nLoc1")
    axs[2].bar(range(days), cars[:, 1], color="green", label="nLoc2")
    axs[2].set_title("Cars in locations")
    axs[2].set_xlabel("days")
    axs[2].set_ylabel("number of cars")
    axs[2].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
