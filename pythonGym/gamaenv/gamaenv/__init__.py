from gym.envs.registration import register

register(id='GamaEnv-v0', entry_point='gamaenv.envs:GamaEnv', max_episode_steps=10)
