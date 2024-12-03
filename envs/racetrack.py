import os

from typing import Optional
import numpy as np
import gymnasium as gym

from loguru import logger

import skimage.draw

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(script_dir, "Racetrack.log")
logger.add(log_file_path, mode="w")


class Racetrack(gym.Env):

    def __init__(
        self,
        grid_file_path,
        max_velocity=4,
        noise=True,
        logging=False,
    ):

        with open(grid_file_path, "r") as file:
            lines = file.readlines()

        grid = [list(line.strip("\n")) for line in lines]
        self.grid_mapping = {"T": 0, "S": 1, "F": 2, "X": 3}
        self.grid = np.vectorize(self.grid_mapping.get)(grid)

        self.max_velocity = max_velocity
        self.logging = logging
        self.noise = noise

        # We can decide of the velocity for horizontal and vertical moves
        self.action_space = gym.spaces.MultiDiscrete([3, 3])
        self.actions = [-1, 0, 1]  # that's the only possible added velocity

        # The agent observes the map, its own position and velocity vector
        self.observation_space = gym.spaces.Dict(
            {
                "car_position": gym.spaces.MultiDiscrete(
                    [self.grid.shape[0], self.grid.shape[1]]
                ),
                "velocity": gym.spaces.MultiDiscrete(
                    [self.max_velocity + 1, self.max_velocity + 1]
                ),
            }
        )

        self.car_position = None
        self.car_velocity = None

    def _get_obs(self):
        if self.logging:
            pass

        return {
            "car_position": self.car_position,
            "velocity": self.car_velocity,
        }

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        start_tiles_idx = np.where(self.grid == self.grid_mapping["S"])
        idx = self.np_random.integers(start_tiles_idx[0].size)
        self.car_position = np.array(
            [start_tiles_idx[0][idx], start_tiles_idx[1][idx]], dtype=np.int32
        )

        self.car_velocity = np.zeros(shape=(2,), dtype=np.int32)

        if self.logging:
            logger.debug(f"Reset state: {self.car_position}, {self.car_velocity}")

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Args:
            action (tuple): dx,dy for the added velocity

        """

        if self.logging:
            logger.debug(f"Old position: {self._get_obs()['car_position']}")

        # Reward is always -1 no matter what
        reward = -1

        action = np.array(action, dtype=np.int32)
        assert action.size == 2
        assert action.all() in self.actions

        if self.noise:
            # 0.1 prob that velocity increments are both zero
            prob = self.np_random.random()
            if prob < 0.1:
                action = np.zeros(shape=(2,), dtype=np.int32)

                if self.logging:
                    logger.debug("Randomly nulling velocity")

        if self.logging:
            logger.debug(f"Action is {action}")

        # Clip velocity to 0,max_velocity and increments of -1,1 or 0 and not 0,0
        added_velocity = action

        new_vel = self.car_velocity + added_velocity
        new_vel = np.clip(new_vel, 0, self.max_velocity)

        # New vel can't be 0,0 unless we are on the starting line
        start_tiles_idx = np.where(self.grid == self.grid_mapping["S"])
        if np.isin(self.car_position, start_tiles_idx).all() and (new_vel == 0).all():
            new_vel = self.car_velocity.copy()

        if self.logging:
            logger.debug(f"Old vel: {self.car_velocity}, new vel: {new_vel}")

        self.car_velocity = new_vel

        # Because the first axis are inverted
        new_position = np.array(
            [
                self.car_position[0] - self.car_velocity[0],
                self.car_position[1] + self.car_velocity[1],
            ]
        )

        if self.logging:
            logger.debug(f"New position: {new_position}")

        # Calculate intersection of speed vector and racetrack
        # Credits to https://github.com/vojtamolda/reinforcement-learning-an-introduction/blob/main/chapter05/racetrack.py
        xs, ys = skimage.draw.line(*self.car_position, *(new_position))
        xs_within_track = np.clip(xs, 0, self.grid.shape[0] - 1)
        ys_within_track = np.clip(ys, 0, self.grid.shape[1] - 1)
        collisions = self.grid[xs_within_track, ys_within_track]

        if self.logging:
            logger.debug(
                f"Collisions coordinates: {[(xs_within_track[i], ys_within_track[i]) for i in range(len(xs_within_track))]}"
            )

        # Check whether the car is on the road, out of track or crossing the finish line
        within_track_limits = (xs == xs_within_track).all() and (
            ys == ys_within_track
        ).all()
        crossing_finish = (collisions == self.grid_mapping["F"]).any()
        on_grass = (collisions == self.grid_mapping["X"]).any()

        if self.logging:
            logger.debug(f"Is within limits: {within_track_limits}")
            logger.debug(f"Is crossing finish: {crossing_finish}")
            logger.debug(f"Is on grass: {on_grass}")

        # Restart when on grass or out of racetrack limits without crossing
        if on_grass or (not within_track_limits and not crossing_finish):
            self.reset()
            return self._get_obs(), reward, False, False, self._get_info()

        # Clip the new_position to the end of track in case we cross the finish
        new_position = np.clip(new_position, 0, np.array(self.grid.shape) - 1)

        self.car_position = new_position
        return self._get_obs(), reward, crossing_finish, False, self._get_info()


if __name__ == "__main__":
    grid_file = os.path.join("Chapter05", "racetrack1.txt")
    env = Racetrack(grid_file, logging=True)
    obs, _ = env.reset()
    # print(obs)
    action_mapping = [-1, 0, 1]
    action = env.action_space.sample()
    action = action_mapping[action[0]], action_mapping[action[1]]
    print(action)
    # action = np.array([1, 1])
    # print(action)
    # # action = env.action_mapping[action[0]], env.action_mapping[action[1]]
    # # print(action)
    obs, r, terminated, truncated, info = env.step(action)
    # print(obs)
