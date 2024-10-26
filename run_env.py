import gymnasium
import envs

env = gymnasium.make("KArmedBandit-v0", stationnary=False)
env.reset()
_, reward, _, _, info = env.step(0)
print(reward)
print(info)
