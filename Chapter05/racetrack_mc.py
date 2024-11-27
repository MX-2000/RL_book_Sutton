import gymnasium as gym
import envs
import os

file_path = os.path.join("Chapter05", "racetrack1.txt")
env = gym.make("Racetrack-v0", grid_file_path=file_path)
obs, _ = env.reset()
print(obs)
