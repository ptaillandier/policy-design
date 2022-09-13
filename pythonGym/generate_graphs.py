import pandas as pd
import os.path
fname = r'/home/mvinyalssal/ray_results/PPO_GamaEnv-v0_2022-09-13_09-42-074bae5s1h/result.json'
print("File exists ", os.path.isfile(fname))
results = pd.read_json(fname, lines=True)
print('results.head()', results.head())
print('results.describe()',results.describe())
print('results.columns', results.columns)
# This iterates through each training iteration 
for ind in results.index:
    print(results['hist_stats'][ind]['episode_reward'])
