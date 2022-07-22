from gym.envs.registration import register

register(id='SimpleGymEnv-v0', entry_point='simplegymenv.envs:SimpleGymEnv', max_episode_steps=10)
