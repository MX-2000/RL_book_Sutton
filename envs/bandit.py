from typing import Optional
import numpy as np
import gymnasium as gym


class KBanditEnv(gym.Env):

    def __init__(self, k: int = 10, stationnary=True):

        self.stationnary = stationnary

        # Number of arms
        self.k = k
        self.action_space = gym.spaces.Discrete(k)

        # We don't need observation space in nonassociative setup
        # So we make it empty
        self.observation_space = gym.spaces.Box(
            low=0.0, high=0.0, shape=(1,), dtype=np.float32
        )

        true_q_values = []
        if self.stationnary:
            for i in range(self.k):

                # We generate the true value according to normal distribution with mean 0 and unit variance
                true_q = self.np_random.normal(loc=0, scale=1)
                true_q_values.append(true_q)
        else:
            START_Q = 0
            true_q_values = [START_Q for i in range(self.k)]

        self.true_q_values = true_q_values

    def _get_info(self):
        return {"q_star": self.true_q_values}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Return a fixed observation since it's non-associative
        return np.array([0.0], dtype=np.float32), self._get_info()

    def step(self, action):

        # Continuous task
        terminated = False
        truncated = False
        reward = self.np_random.normal(loc=self.true_q_values[action], scale=1)
        obs = np.array([0.0], dtype=np.float32)

        if not self.stationnary:
            # We update all the q*(a) by a independant random walk
            for i in range(self.k):
                increment = self.np_random.normal(loc=0.0, scale=0.01)
                self.true_q_values[i] += increment

        return obs, reward, terminated, truncated, self._get_info()


if __name__ == "__main__":
    env = KBanditEnv()
