import gym
import numpy as np
from stable_baselines3 import PPO
from gym import spaces

class EnergyEnv(gym.Env):

    def __init__(self):
        super(EnergyEnv, self).__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(4,), dtype=np.float32
        )

    def reset(self):
        self.state = np.random.rand(4)*100
        return self.state

    def step(self, action):

        energy = self.state[0]

        if action == 0:
            energy -= 5
        elif action == 2:
            energy += 5

        reward = -energy

        self.state = np.random.rand(4)*100

        done = False

        return self.state, reward, done, {}

env = EnergyEnv()

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)