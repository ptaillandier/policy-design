import gym
import gamaenv
import socket
import time
import sys
from keras import Sequential
import os
from policy_dirichlet import Policy
from ppo_training import PPOTraining
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
    "--num-iters",
    type=int,
    default=3,
    help="Number of iterations.",
)

parser.add_argument(
    "--num-batch-episodes",
    type=int,
    default=200,
    help="Number of episodes par batch",
)

parser.add_argument(
    "--num-update-epochs",
    type=int,
    default=30,
    help="Number of training epochs per update",
)

parser.add_argument(
    "--num-mini-batches",
    type=int,
    default=100,
    help="Number of training minibatches per update/epoch. Default to 100",
)

parser.add_argument(
    "--discount-factor",
    type=float,
    default=0.999,
    help="Discount factor for the reinforcement learning",
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=3e-4,
    help="The learning rate of the optimizer. Same for policy and value networks",
)

parser.add_argument(
    "--epsilon",
    type=float,
    default=1e-7,
    help="The epsilon parameter of the optimizer. Same for policy and value networks",
)



parser.add_argument(
    "--gae-lambda",
    type=float,
    default=0.95,
    help="Lambda factor for the computation of GAE",
)

parser.add_argument(
    "--activation",
    type=str,
    default="tanh",
    help="Activation function descriptor for hidden layers. Supported values are: tanh or relu",
)

parser.add_argument(
    "--minibatch-splitting",
    type=str,
    default="SHUFFLE_TRANSITIONS",
    help="The way the data is split into mini-batches. Supported values are: FIXED_TRAJECTORIES, SHUFFLE_TRAJECTORIES, SHUFFLE_TRANSITIONS (default), SHUFFLE_TRANSITIONS_RECOMPUTE_ADVANTAGES",
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
max_training_iters = args.num_iters # Number of training iterations (times that we run training)
batch_size = args.num_batch_episodes
discount_factor = args.discount_factor
layers_sizes = args.sizes 
activation_function = tf.tanh
if args.activation == "relu":
    activation_function = "relu"
elif args.activation == "tanh":
    activation_fuction = tf.tanh

### End configuration variables ###
### Start configuration specific ppo variables ###
clipping_ratio = 0.2
policy_learning_rate = args.learning_rate
critic_learning_rate = args.learning_rate
policy_epsilon = args.epsilon
critic_epsilon = args.epsilon
target_kl = 0.01 #Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)
gae_lambda = args.gae_lambda #Lambda parameter for the Generalized Advantage Estimation (GAE) 
n_update_epochs = args.num_update_epochs #Number of epochs to update the policy (default set to 10,30) and it is the same number for actor and critic policy
n_mini_batches = args.num_mini_batches #Number of training minibatches par update/epoch (default set to 32-128)
minibatch_splitting_method = args.minibatch_splitting
### End configuration specific ppo variables ###
## From others implementation
#args.batch_size = int(args.num_envs * args.num_steps)
#args.minibatch_size = int(args.batch_size // args.num_minibatches)
#where num_envs is the number of parallel game environments default = 1
#where num_steps is the number of steps to run in each environment per policy rollout. Default 2048
# In our case each episode has 10 experiences/steps and we specify the number of episodes via batch_size
# so the conversion is num_steps = 10*batch_size, for a default num_steps 2048 will be batch_size=200
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
layers_sizes.append(n_dimensions*3) #Add the last output layer considering  dimensions to split the budget the number of actions

# Observations (3) 
# 1. Remaining budget - Remaining budget available to implement public policies
# 2. Fraction of adopters - Fraction of adopters [0,1]
# 3. Remaining time before ending the simulation - Unit (in seconds)

n_observations  = 3     # Number of observations from the state of the social planner, can be modified for testing
n_times_4_action = 10 # Number of times in which the policy maker can change the public policy (time horizon:     5 years) 

# Rewards
# 1. Evolution of the intention of adoption (mean_intention - previous_mean_intention) / previous_mean_intention)
ACTORMODELPATH              = 'actor_nngamma' # Path to the file where to store the actor neural network
CRITICMODELPATH             = 'critic_nngamma' # Path to the file where to store the critic neural network
results2_filepath           = 'results_number_adopters.csv'
results3_filepath           = 'results_actions.csv'
results4_filepath           = 'results_nnoutputs.csv'

# The loop of interaction between the gama simulation and the model
def gama_interaction_loop(gama_simulation: socket, episode: utils.Episode) -> None:

    global results3_filepath    
    global actor_model
    policy_manager: Policy = Policy(actor_model)
    gama_socket_as_file: TextIO[IO[str]] = gama_simulation.makefile(mode='rw')

    try:
        n_times_4_action = 10 # Number of times in which the policy maker can change the public policy (time horizon: 5 years)
        time_updating_policy = 0
        time_simulation = 0
        i_experience = 0
        last_obs = -1
        while True:
           # we wait for the simulation to send the observations
           #print("waiting for observations")
           tic_b = time.time()
           received_observations: str = gama_socket_as_file.readline()
           time_simulation = time_simulation + time.time()-tic_b
           if "END\n" in received_observations:
               last_obs: npt.NDArray[np.float64] = gamainteraction.string_to_nparray(received_observations.replace("END", ""))
               last_obs[2] = float(n_times_4_action-i_experience) #We change the last observation to be the number of times that remain for changing the policy
               episode.set_last_observation(last_obs)
               break

           print("model received:", received_observations)
           obs: npt.NDArray[np.float64] = gamainteraction.string_to_nparray(received_observations)
           obs[2] = float(n_times_4_action-i_experience) #We change the last observation to be the number of times that remain for changing the policy
            
           # we then compute a policy and send it back to gama
           tic_b = time.time()
           action, action_env, nn_outputs = gamainteraction.process_observations(policy_manager, obs, n_actions)
           time_updating_policy = time_updating_policy + time.time() - tic_b

           #store in the result file the outputs of the nn 
           with open(results4_filepath, 'a') as f:
               f.write(str(episode.id)+','+str(i_experience)+','+str(obs[0])+','+str(obs[1])+','+','.join([str(output) for output in nn_outputs])+'\n')

           str_action = gamainteraction.action_to_string(np.array(action_env))
           #store in the result file the actions taken
           with open(results3_filepath, 'a') as f:
               f.write(str(episode.id)+','+str(i_experience)+','+ str_action)

           print("model sending policy:(thetaeconomy ,thetamanagement,fmanagement,thetaenvironment,fenvironment)", str_action)
           gama_socket_as_file.write(str_action)
           gama_socket_as_file.flush()

           tic_b = time.time()
           # we finally wait for the reward
           #print("The model is waiting for the reward")
           policy_reward = gama_socket_as_file.readline()
           time_simulation = time_simulation + time.time() - tic_b
           print("model received reward:", policy_reward)
               
           gamainteraction.process_reward(policy_reward, action_env, received_observations)
           episode.add_experience(obs, action, float(policy_reward))
           i_experience = i_experience + 1
           # new line for better understanding of the logs
           #print()
    except ConnectionResetError:
       print("connection reset, end of simulation")
    except:
        print("EXCEPTION pendant l'execution")
        print(sys.exc_info()[0])
        sys.exit(-1)

    gama_socket_as_file.write("over\n") #we send a message for the simulation to wait before closing
    gama_socket_as_file.flush()
      
    print('\t','updating policy time', time_updating_policy)
    print('\t','simulation time', time_simulation)
    
    return episode



if __name__ == "__main__":
    #Create the environment
    env = gym.make('GamaEnv-v0', 
                   headless_directory       = headless_dir,
                   headless_script_path     = run_headless_script_path,
                   gaml_experiment_path     = gaml_file_path, 
                   gaml_experiment_name     = experiment_name)

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
          f.write('iteration,decision_step,budget_obs,fadopters_obs,cleft,cman,cenv,mean_thetaeco,mean_thetaman,mean_thetaenv,std_thetaeco,std_thetaman,std_thetaenv\n')
    
    actor_model = utils.mlp(n_observations, layers_sizes, activation = activation_function, last_layer_scaling=0.01)
    print('actor_model.summary()')
    utils.full_summary(actor_model)
    critic_model = utils.mlp(n_observations, np.append(layers_sizes[:-1],  1), activation = activation_function, last_layer_scaling=1.0)
    print('critic_model.summary()')
    utils.full_summary(critic_model)
    policy_manager: Policy = Policy(actor_model)
    

    batch_episodes = []
    i_episode = 0
    #For each training iteration
    for i_iter in range(max_training_iters):
        print('i_iter', i_iter)
        tic_b_iter = time.time()
        batch_episodes.clear()
 
        for i_batch in range(batch_size):
            time_updating_policy = 0
            time_simulation = 0
            i_experience = 0
            episode = utils.Episode()
            episode.set_id(i_episode)
            tic_b = time.time()
            obs = env.reset()
            time_simulation = time_simulation + time.time()-tic_b
            obs[2] = float(n_times_4_action-i_experience) #We change the last observation to be the number of times tha    t remain for changing the     policy
            print("model initially received:", obs)
            done = False
            while not done:
                # we then compute a policy and send it back to gama
                tic_b = time.time()
                action, processed_observations, action_env, nn_outputs = policy_manager.choose_action(obs, n_actions)
                time_updating_policy = time_updating_policy + time.time() - tic_b
                #store in the result file the outputs of the nn
                with open(results4_filepath, 'a') as f:
                    f.write(str(episode.id)+','+str(i_experience)+','+str(obs[0])+','+str(obs[1])+','+','.join([str(output)
                                                                                                                for
                                                                                                                output
                                                                                                                in
                                                                                                                nn_outputs])+'\n')
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
            # Add episode to the batch of episodes
            batch_episodes.append(episode)
                    
        i_batch_episode = 0
        for episode in batch_episodes:
            sum_episode_rewards = sum(episode.rewards) #the sum of discounted? rewards of an episode is the basic component for all performance stadistics
            # Save the number of adopters end of each episode for statistics
            with open(results2_filepath, 'a') as f:
                f.write(str(episode.last_observation[1])+'\n')
            print('episode.observations[-2][0]', episode.observations[-2][0])
            print('episode.observations[-1][0]', episode.observations[-1][0])
            print('Batch episode:', i_batch_episode,'\t', ' reward:', sum_episode_rewards, ' fraction of adopters ', str(episode.last_observation[1]), 'remaining_budget', episode.last_observation[0], '\n')
            i_batch_episode = i_batch_episode + 1

        tic_b = time.time()
        print('discount_factor', discount_factor)
        print('Optimiser Adam with learning rate', policy_learning_rate, " epsilon ", policy_epsilon)
        # Create a training based on model with the desired parameters.
        tr = PPOTraining(actor_model, critic_model, actor_optimizer= tf.keras.optimizers.Adam(learning_rate=policy_learning_rate, epsilon=policy_epsilon), critic_optimizer= tf.keras.optimizers.Adam(learning_rate=critic_learning_rate, epsilon=critic_epsilon), clipping_ratio=clipping_ratio, target_kl=target_kl, discount_factor=discount_factor, gae_lambda=gae_lambda, minibatch_splitting_method=minibatch_splitting_method)
        tr.train(batch_episodes, n_update_epochs, n_mini_batches)
        training_time = time.time() - tic_b
        print('\t','training time', training_time)
        print('it:',i_iter,'\t time:',time.time()-tic_b_iter)   
        i_episode = i_episode + 1  
    #Store model to reuse to pursue learning       
    actor_model.save(ACTORMODELPATH, include_optimizer=False)
    critic_model.save(CRITICMODELPATH, include_optimizer=False)
