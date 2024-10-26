from gymnasium.envs.registration import register
from envs.bandit import KBanditEnv

register(
    id="KArmedBandit-v0",
    entry_point="envs:KBanditEnv",
)
