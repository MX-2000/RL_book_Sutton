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
GAMMA = 0.9


def get_expected_return(V, state, pi_s, poisson_probs, gamma):
    # Apply action and calculate moving cost
    cars_loc1 = min(state[0] - pi_s, MAX_CARS)
    cars_loc2 = min(state[1] + pi_s, MAX_CARS)
    moving_cost = MOVING_COST * abs(pi_s)

    expected_return = -moving_cost

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
    # prob_dict = {
    #     "loc1_req": np.arange(max + 1),
    #     "loc1_ret": np.arange(max + 1),
    #     "loc2_req": np.arange(max + 1),
    #     "loc2_ret": np.arange(max + 1),
    # }
    # prob_dict["loc1_req"] = poisson.pmf(prob_dict["loc1_req"], lambdas[0, 0])
    # prob_dict["loc1_ret"] = poisson.pmf(prob_dict["loc1_ret"], lambdas[0, 1])
    # prob_dict["loc2_req"] = poisson.pmf(prob_dict["loc2_req"], lambdas[1, 0])
    # prob_dict["loc2_ret"] = poisson.pmf(prob_dict["loc2_ret"], lambdas[1, 1])

    # # dims are (loc1_req, loc1_ret, loc2_req, loc2_ret)
    # result = np.zeros(
    #     shape=(MAX_POISSON + 1, MAX_POISSON + 1, MAX_POISSON + 1, MAX_POISSON + 1)
    # )
    # for pos in np.ndindex(result.shape):

    #     result[pos] = (
    #         prob_dict["loc1_req"][pos[0]]
    #         * prob_dict["loc1_ret"][pos[1]]
    #         * prob_dict["loc2_req"][pos[2]]
    #         * prob_dict["loc2_ret"][pos[3]]
    #     )

    # return result


def policy_evaluation(env, V, pi, poisson_probs, gamma):
    obs, info = env.reset()

    delta = 1
    threshold = 1e-2

    while delta > threshold:
        delta = 0
        for state in np.ndindex(V.shape):
            v = V[state]
            pi_s = pi[state]
            new_estimate = 0
            for prob_state in np.ndindex(poisson_probs.shape):
                prob = poisson_probs[prob_state]
                # For shape compatibility
                poisson_gen = np.array(prob_state).reshape(2, 2)
                next_state, reward, _, _, _ = env.unwrapped.simulate_step(
                    state, pi_s, poisson_gen
                )
                v_next = V[tuple(next_state)]
                new_estimate += prob * (reward + gamma * v_next)
            V[state] = new_estimate
            delta = max(delta, abs(v - new_estimate))
        print("Iteration over, ", delta)

    return V


def policy_evaluation_faster(V, pi, poisson_probs, gamma):
    delta = 100
    threshold = 10

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
        for action in range(-MAX_MOVE, MAX_MOVE + 1):
            pi_s = action

            expected_return = get_expected_return(V, state, pi_s, poisson_probs, gamma)

            action_returns[action] = expected_return

        pi[state] = np.argmax(action_returns)

        if old_action != pi[state]:
            policy_stable = False

    return pi, policy_stable


def policy_iteration(gamma):
    policies = []
    values = []

    V = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.float32)
    pi = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=np.int32)

    values.append(V)
    policies.append(pi)

    poisson_probs = build_poisson_probs(MAX_POISSON, LAMBDAS)

    while True:
        V = policy_evaluation_faster(V, pi, poisson_probs, gamma)
        pi, policy_stable = policy_improvment(V, pi, poisson_probs, gamma)

        print(f"Policy stable? {policy_stable}")
        values.append(V)
        policies.append(pi)

        if policy_stable:
            break

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


def main(days):

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

    V, pi = policy_iteration(GAMMA)


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
