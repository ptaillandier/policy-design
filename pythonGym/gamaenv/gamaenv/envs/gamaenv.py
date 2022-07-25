import gym
import numpy as np
from typing import Optional
from gym import spaces

class GamaEnv(gym.Env):

    def __init__(self):
        self.n_observations = 3 #Number of observations
        self.n_execution_steps = 9
        self.steps_before_done = self.n_execution_steps
        obs_high_bounds = np.array([50.0, 1.0, 20000000.0])
        obs_low_bounds = np.array([-50.0, 0.0, 0.0])
        self.observation_space = spaces.Box(obs_low_bounds, obs_high_bounds)
        action_high_bounds = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        action_low_bounds = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.action_space = spaces.Box(action_low_bounds, action_high_bounds)
            
    def step(self, action):

        self.state = np.random.rand(1, self.n_observations).flatten()
        done = False
        if self.steps_before_done == 0:
            done = True
        else:
            self.steps_before_done -= 1

        reward = np.random.rand()
        return np.array(self.state, dtype=np.float32), reward, done, {} 

    def reset(self,*,seed: Optional[int] = None,return_info: bool = False,options: Optional[dict] = None ): 
        self.state = np.random.rand(1, self.n_observations).flatten()
        self.steps_before_done = self.n_execution_steps
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
