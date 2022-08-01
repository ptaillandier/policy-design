import socket
import time
import sys
import gym
import gamaenv
from keras import Sequential
import os
from policy_dirichlet import Policy
from training_dirichlet import Training
import numpy as np
from typing import List, TextIO, IO
import argparse
import numpy.typing as npt
from user_local_variables import *
import utils
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser(description='Runs the experiment for the gama policy design environment')
parser.add_argument(
    "--iters",
    type=int,
    default=3,
    help="Number of iterations.",
)

parser.add_argument(
    "--num-batch-episodes",
    type=int,
    default=1,
    help="Number of episodes par batch",
)

parser.add_argument(
    "--discount-factor",
    type=float,
    default=0.99,
    help="Discount factor for the reinforcement learning",
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=3e-4,
    help="The learning rate of the optimizer.",
)

parser.add_argument(
    "--epsilon",
    type=float,
    default=1e-7,
    help="The epsilon parameter of the optimizer.",
)

parser.add_argument(
    "--sizes",
    type=int,
    nargs = "+",
    default=[32,32],
    help="Size of the different layers of the neural network.",
)
args = parser.parse_args()

### Start configuration variables ###
max_training_iters = args.iters # Number of training iterations (times that we run training)
batch_size = args.num_batch_episodes
discount_factor = args.discount_factor
learning_rate = args.learning_rate
epsilon = args.epsilon
layers_sizes = args.sizes 
### End configuration variables ###
n_episodes = max_training_iters*batch_size #The total number of episodes explored will be the number of iterations for the training par the size of batch examples processed on each training
print("Total number of episodes =", n_episodes)
print("max_training_iters", max_training_iters)
# Actions (5) 
# 1. Thetaeconomy - Fraction of financial support [0,1] 
# 2. Thetamanagement - Fraction of increment on the skill of trained agents [0,1]
# 3. Fmanagement - Fraction of individuals chosen randomly to be trained [0,1]
# 4. Thetaenvironment - Fraction of environmental awareness [0,1]
# 5. Fenvironment - Fraction of individuals chosen randomly to increase environmental awaraness [0,1]
# Cost computed as Nman(100*Fman)*thetamanagement + N_new_adopters(max100,observed in next state)*thetaeconomy+0.5*Nenv(100*Fenv)*thetaenv
n_actions       = 5    
n_dimensions = 3 #3 dimensions: economy, management, environment
layers_sizes.append(n_dimensions*3) #Add the last output layer considering 3 dimensions to split the budget the number of actions

# Observations (3) 
# 1. Remaining budget - Remaining budget available to implement public policies
# 2. Fraction of adopters - Fraction of adopters [0,1]
# 3. Remaining time before ending the simulation - Unit (in seconds)

n_observations  = 3     # Number of observations from the state of the social planner, can be modified for testing
n_times_4_action = 10 # Number of times in which the policy maker can change the public policy (time horizon:     5 years)


# Rewards
# 1. Evolution of the intention of adoption (mean_intention - previous_mean_intention) / previous_mean_intention)
MODELPATH                   = 'nngamma' # Path to the file where to store the neural network
results_filepath            = 'results_sum_rewards.csv'
results2_filepath           = 'results_number_adopters.csv'
results3_filepath           = 'results_actions.csv'
results4_filepath           = 'results_nnoutputs.csv'


if __name__ == "__main__":
    #Create the environment
    env = gym.make('GamaEnv-v0',
            headless_directory      = headless_dir,
            headless_script_path    = run_headless_script_path,
            gaml_experiment_path    = gaml_file_path,
            gaml_experiment_name    = experiment_name)
    #Check that the result file for evaluation does not exist
    try:
      os.remove(results_filepath)
    except OSError:
          pass
    #First line contains the title
    with open(results_filepath, 'a') as f:
          f.write('sum_of_episode_rewards\n')

    #Check that the result2 file for evaluation does not exist
    try:
      os.remove(results2_filepath)
    except OSError:
          pass
    #First line contains the title
    with open(results2_filepath, 'a') as f:
          f.write('number_adopters_end_episode\n')

    #Check that the result3 file for evaluation does not exist
    try:
      os.remove(results3_filepath)
    except OSError:
          pass
    #First line contains the title
    with open(results3_filepath, 'a') as f:
          f.write('iteration,decision_step,thetaeconomy,thetamanagement,fmanagement,thetaenviron,fenviron\n')

    #Check that the result3 file for evaluation does not exist
    try:
      os.remove(results4_filepath)
    except OSError:
          pass
    #First line contains the title
    with open(results4_filepath, 'a') as f:
          f.write('iteration,decision_step,budget_obs,fadopters_obs,cman,cenv,cleft,mean_thetaeco,mean_thetaman,mean_thetaenv,std_thetaeco,std_thetaman,std_thetaenv\n')

    model = utils.mlp(n_observations, layers_sizes)
    print('model.summary()', model.summary())
    policy_manager: Policy = Policy(model)
    print('max_training_iters', max_training_iters)
    i_episode = 0
    #For each training iteration
    for i_iter in range(max_training_iters):
        print('i_iter', i_iter)
        tic_b_iter = time.time()
        batch_episodes = []
 
        for i_batch in range(batch_size):
            time_updating_policy = 0
            time_simulation = 0
            i_experience = 0
            episode = utils.Episode()
            episode.set_id(i_episode)
            tic_b = time.time()
            obs = env.reset()
            time_simulation = time_simulation + time.time()-tic_b
            obs[2] = float(n_times_4_action-i_experience) #We change the last observation to be the number of times tha    t remain for changing the policy
            print("model initially received:", obs)
            done = False
            while not done:
                # we then compute a policy and send it back to gama
                tic_b = time.time()
                action, processed_observations, action_env, nn_outputs = policy_manager.choose_action(obs, n_actions)
                time_updating_policy = time_updating_policy + time.time() - tic_b
                #store in the result file the outputs of the nn 
                with open(results4_filepath, 'a') as f:
                    f.write(str(episode.id)+','+str(i_experience)+','+str(obs[0])+','+str(obs[1])+','+','.join([str(output)     for output in nn_outputs])+'\n')
                str_action = ",".join([str(action) for action in action_env]) + "\n" 
                #store in the result file the actions taken
                with open(results3_filepath, 'a') as f:
                    f.write(str(episode.id)+','+str(i_experience)+','+ str_action)
 
                print("model sending policy:(thetaeconomy,thetamanagement,fmanagement,thetaenvironment,fenvironment)",str_action)
                tic_b = time.time()
                next_obs, reward, done, info = env.step(action_env)
                time_simulation = time_simulation + time.time() - tic_b
                print("model received reward:", reward)
                episode.add_experience(obs, action, float(reward))
                obs = next_obs
                i_experience = i_experience + 1
            
            episode.set_last_observation(obs)
            print('\t','updating policy time', time_updating_policy)
            print('\t','simulation time', time_simulation)
            batch_episodes.append(episode)
        
        i_batch_episode = 0
        for episode in batch_episodes:
            sum_episode_rewards = sum(episode.rewards) #the sum of discounted? rewards of an episode is the basic component for all performance stadistics
            #store in the resuls file the sum of rewards for this episode
            with open(results_filepath, 'a') as f:
                f.write(str(sum_episode_rewards)+'\n')
            # Save the number of adopters end of each episode for statistics
            with open(results2_filepath, 'a') as f:
                f.write(str(episode.last_observation[1])+'\n')
            print('episode.observations[-2][0]', episode.observations[-2][0])
            print('episode.observations[-1][0]', episode.observations[-1][0])
            print('Batch episode:', i_batch_episode,'\t', ' reward:', sum_episode_rewards, ' fraction of adopters ', str(episode.last_observation[1]), 'remaining_budget', episode.last_observation[0], '\n')
            i_batch_episode = i_batch_episode + 1

        tic_b = time.time()
        print('discount_factor', discount_factor)
        # Create a training based on model with the desired parameters.
        tr = Training(model, discount_factor=discount_factor, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon))
        tr.train(batch_episodes)

        
        training_time = time.time() - tic_b
        print('\t','training time', training_time)
        print('it:',i_iter,'\t time:',time.time()-tic_b_iter)
        i_episode = i_episode + 1
    #Store model to reuse to pursue learning       
    model.save(MODELPATH, include_optimizer=False)
