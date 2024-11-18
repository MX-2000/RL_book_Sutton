import os

from typing import Optional
import numpy as np
import gymnasium as gym

from loguru import logger

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "JacksRental.log")
logger.add(log_file_path, mode="w")


class JacksRental(gym.Env):

    def __init__(
        self,
        max_move,
        max_cars,
        rent_r,
        moving_cost,
        max_poisson,
        p_lambdas,
        logging=False,
    ):

        self.moving_cost = moving_cost
        self.max_move = max_move
        self.rent_r = rent_r
        self.max_poisson = max_poisson
        self.max_cars = max_cars

        self.logging = logging

        self.action_space = gym.spaces.Discrete(
            2 * self.max_move + 1, start=-self.max_move
        )  # Negative values are from loc2 to loc1

        self.observation_space = gym.spaces.Box(
            0, 20, shape=(2,), dtype=np.int32
        )  # We observe the number of cars at location A and B

        self.cars = np.zeros((2,), dtype=np.int32)  # Number of cars in each location

        # First col is the renting request, second col is returns
        self.lambdas = p_lambdas

    def _get_obs(self):
        if self.logging:
            logger.debug(f"N Cars: {self.cars}")

        return self.cars

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.cars = np.zeros((2,), dtype=np.int32)

        return self._get_obs(), self._get_info()

    def step(self, action):
        if self.logging:
            logger.debug(f"Action: {action}")

        assert self.action_space.contains(action)

        reward = -self.moving_cost * abs(action)  # Every move costs

        # action < 0 means we move from loc2 to loc1
        self.cars[0] -= action
        self.cars[1] += action

        assert np.all(self.cars >= 0)

        request_gen = np.random.poisson(self.lambdas)
        request_gen = np.clip(request_gen, None, self.max_poisson)

        tot_cars_rented = min(request_gen[0, 0], self.cars[0]) + min(
            request_gen[1, 0], self.cars[1]
        )

        # Cars get rented
        self.cars -= request_gen[:, 0]

        # If Jack has a car available, he rents it out and is credited self.rent_r
        reward += tot_cars_rented * self.rent_r

        # Cars become available for renting the day after they are returned.
        self.cars += request_gen[:, 1]

        self.cars = np.clip(self.cars, 0, self.max_cars)

        return self._get_obs(), reward, False, False, self._get_info()

    def simulate_step(self, pstate, action, poisson_gen):
        """
        Used for dynamic programming algorithms
        """
        state = np.array(
            [pstate[0], pstate[1]], dtype=np.int32
        )  # Trick for pstate being tuple item assignement

        if self.logging:
            logger.debug(f"Action: {action}")

        assert self.action_space.contains(action)
        assert self.observation_space.contains(state)

        reward = -self.moving_cost * abs(action)  # Every move costs

        # action < 0 means we move from loc2 to loc1
        state[0] -= action
        state[1] += action

        assert np.all(state >= 0)

        # request_gen = np.random.poisson(self.lambdas)
        # request_gen = np.clip(request_gen, None, self.max_poisson)
        request_gen = poisson_gen

        # Cars become available for renting the day after they are returned.
        state += request_gen[:, 1]

        tot_cars_rented = min(request_gen[0, 0], state[0]) + min(
            request_gen[1, 0], state[1]
        )

        # Cars get rented
        state -= request_gen[:, 0]

        # If Jack has a car available, he rents it out and is credited self.rent_r
        reward += tot_cars_rented * self.rent_r

        state = np.clip(state, 0, self.max_cars)

        return state, reward, False, False, {}


if __name__ == "__main__":
    env = JacksRental()
