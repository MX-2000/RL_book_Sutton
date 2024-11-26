import os

from typing import Optional
import numpy as np
import gymnasium as gym

from loguru import logger

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "Racetrack.log")
logger.add(log_file_path, mode="w")


class Racetrack(gym.Env):

    def __init__(
        self,
        grid_file_path,
        max_velocity=5,
        logging=False,
    ):

        with open(grid_file_path, "r") as file:
            lines = file.readlines()

        grid = [list(line.strip("\n")) for line in lines]
        self.grid = np.array(grid)

        self.max_velocity = max_velocity
        self.logging = logging

        # We can decide of the velocity for horizontal and vertical moves
        self.action_space = gym.spaces.MultiDiscrete([3, 3])

        # Because we actually can increment -1, 0 or 1
        self.action_mapping = [-1, 0, 1]

        # The agent observes the map, its own position and velocity vector
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(
                    low=0,
                    high=3,
                    shape=(self.grid.shape[0], self.grid.shape[1]),
                    dtype=np.int32,
                ),
                "car_position": gym.spaces.MultiDiscrete(
                    [max(self.grid.shape), max(self.grid.shape)]
                ),
                "velocity": gym.spaces.MultiDiscrete(
                    [self.max_velocity + 1, self.max_velocity + 1]
                ),
            }
        )

    def _get_obs(self):
        if self.logging:
            logger.debug(f"TODO")

        return "Nothing here"

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        return self._get_obs(), self._get_info()

    def step(self, action):
        pass


if __name__ == "__main__":
    grid_file = os.path.join("Chapter05", "racetrack1.txt")
    env = Racetrack(grid_file)
