from gymnasium.envs.registration import register
from envs.bandit import KBanditEnv
from envs.jacks_rental import JacksRental
from envs.racetrack import Racetrack
from envs.wind import WindyGridworldEnv

register(id="Racetrack-v0", entry_point="envs:Racetrack")
register(id="KArmedBandit-v0", entry_point="envs:KBanditEnv")
register(id="JacksRental-v0", entry_point="envs:JacksRental")
register(id="WindyGridworldEnv-v0", entry_point="envs:WindyGridworldEnv")
