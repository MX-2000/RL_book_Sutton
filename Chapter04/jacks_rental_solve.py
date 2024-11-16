import gymnasium as gym
import envs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
TODO

Main question: 
* Faut-il initialiser l'env avec un nombre de voitures dans chaque emplacement ?
* Quel est l'ordre exact des choses ? Move-Returns-Request ?
* Comment le V((0,0)) peut-il être positif ? Alors qu'il y a de grande chances qu'une request n'ait pas de voiture ?
"""


def main(days):
    MAX_MOVE = 5

    env = gym.make("JacksRental-v0", max_move=MAX_MOVE, logging=False)
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
