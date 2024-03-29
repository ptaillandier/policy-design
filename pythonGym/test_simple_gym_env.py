import unittest 
import numpy as np
import gym
import simplegymenv
from gym.utils import env_checker as env_checker_gym
from ray.rllib.utils.pre_checks import env as env_checker_ray

class TestSimpleGymEnv(unittest.TestCase):
    def test_simple_gym_env(self):
        env = gym.make('SimpleGymEnv-v0')
        print('env._max_episode_steps', env._max_episode_steps)
        n_iters = 3
        done = False
        for iter in range(n_iters):
            env.reset()
            while not done:
                action = np.random.rand(1,4).flatten()
                next_observation, reward, done, info = env.step(action)
                print('done', done)

    def test_simple_gym_env2(self):
        env = gym.make('SimpleGymEnv-v0')
        env_checker_gym.check_env(env)

    def test_simple_gym_env3(self):
        env = gym.make('SimpleGymEnv-v0')
        env_checker_ray.check_env(env)
if __name__ == '__main__':
    unittest.main()

