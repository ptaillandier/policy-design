from gym.envs.registration import register

register(id='GamaGymEnv-v0', entry_point='gamagymenv.envs:GamaGymEnv', max_episode_steps=11)

