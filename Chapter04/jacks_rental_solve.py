import gymnasium as gym
import envs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import time


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


def build_pmf_poisson(max, lambdas):
    prob_dict = {
        "loc1_req": np.arange(max + 1),
        "loc1_ret": np.arange(max + 1),
        "loc2_req": np.arange(max + 1),
        "loc2_ret": np.arange(max + 1),
    }
    prob_dict["loc1_req"] = poisson.pmf(prob_dict["loc1_req"], lambdas[0, 0])
    prob_dict["loc1_ret"] = poisson.pmf(prob_dict["loc1_ret"], lambdas[0, 1])
    prob_dict["loc2_req"] = poisson.pmf(prob_dict["loc2_req"], lambdas[1, 0])
    prob_dict["loc2_ret"] = poisson.pmf(prob_dict["loc2_ret"], lambdas[1, 1])

    # dims are (loc1_req, loc1_ret, loc2_req, loc2_ret)
    result = np.zeros(
        shape=(MAX_POISSON + 1, MAX_POISSON + 1, MAX_POISSON + 1, MAX_POISSON + 1)
    )
    for pos in np.ndindex(result.shape):

        result[pos] = (
            prob_dict["loc1_req"][pos[0]]
            * prob_dict["loc1_ret"][pos[1]]
            * prob_dict["loc2_req"][pos[2]]
            * prob_dict["loc2_ret"][pos[3]]
        )

    return result


def policy_evaluation(env, V, pi):
    obs, info = env.reset()

    delta = 1
    threshold = 1e-2

    while delta > threshold:
        for state in np.ndindex(V):
            v = V[state]
            pi_s = pi[state]
            V[state] = 0  # TODO


def policy_improvment(env, V, pi):
    pass


def policy_iteration(env):
    V = np.zeros((MAX_CARS, MAX_CARS))
    pi = np.zeros((MAX_CARS, MAX_CARS))

    poisson_PMF = build_pmf_poisson(MAX_POISSON, LAMBDAS)

    V = policy_evaluation(env, V, pi)
    pi = policy_improvment(env, V, pi)

    return V, pi


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


def main(days):

    env = gym.make(
        "JacksRental-v0",
        max_move=MAX_MOVE,
        max_cars=MAX_CARS,
        rent_r=RENT_REWARD,
        moving_cost=MOVING_COST,
        max_poisson=MAX_POISSON,
        p_lambdas=LAMBDAS,
        logging=False,
    )
    car_moves, cars, rewards = random_strat(env, days)
    return car_moves, cars, rewards


if __name__ == "__main__":
    days = 100
    car_moves, cars, rewards = main(days)

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
