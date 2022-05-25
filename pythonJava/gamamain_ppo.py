import socket
import time
import sys
from keras import Sequential
import os
import gamainteraction
from policy_dirichlet import Policy
from actor_training import ActorTraining
from critic_training import CriticTraining
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
layers_sizes = args.sizes 
### End configuration variables ###
### Start configuration specific ppo variables ###
clipping_ratio = 0.2
policy_learning_rate = 1e-7
critic_learning_rate = 1e-7
target_kl = 0.01 #Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)
gae_lambda = 0.97 #Lambda parameter for the Generalized Advantage Estimation (GAE) 
n_update_epochs = 10 #Number of epochs to update the policy (default set to 10,30) and it is the same number for policy and critic
n_mini_batches = 100 #Number of training minibatches par update/epoch (default set to 32-128)
train_critic_iterations = 80 #Number of gradient descent steps to take on value function per epoch (aka training iteration)
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
layers_sizes.append(n_dimensions*3+1) #Add the last output layer considering 4 dimensions to split the budget the number of actions

# Observations (3) 
# 1. Remaining budget - Remaining budget available to implement public policies
# 2. Fraction of adopters - Fraction of adopters [0,1]
# 3. Remaining time before ending the simulation - Unit (in seconds)

n_observations  = 3     # Number of observations from the state of the social planner, can be modified for testing


# Rewards
# 1. Evolution of the intention of adoption (mean_intention - previous_mean_intention) / previous_mean_intention)
ACTORMODELPATH              = 'actor_nngamma' # Path to the file where to store the actor neural network
CRITICMODELPATH             = 'critic_nngamma' # Path to the file where to store the critic neural network
results_filepath            = 'results_sum_rewards.csv'
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
        n_times_4_action = 9 # Number of times in which the policy maker can change the public policy (time horizon: 5 years)
        time_updating_policy = 0
        time_simulation = 0
        i_experience = 0
        while True:
           # we wait for the simulation to send the observations
           #print("waiting for observations")
           tic_b = time.time()
           received_observations: str = gama_socket_as_file.readline()
           time_simulation = time_simulation + time.time()-tic_b
           if received_observations == "END\n":
               #print("simulation has ended")
               break

           print("model received:", received_observations)
           obs: npt.NDArray[np.float64] = gamainteraction.string_to_nparray(received_observations.replace("END", ""))
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
               f.write(str(episode.id)+','+str(i_experience)+','+ str_action+'\n')

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

def train_actor_model(_model: Sequential, _batch_episodes: List[utils.Episode], _batch_deltas: List[float],  _clipping_ratio:float, _target_kl:float, _train_policy_iterations:int, _policy_learning_rate:float):
    # Create a training based on model with the desired parameters.
    policy_optimizer = tf.keras.optimizers.Adam(learning_rate=_policy_learning_rate)
    tr = ActorTraining(_model, optimizer= policy_optimizer, clipping_ratio=_clipping_ratio, target_kl=_target_kl)
    tr.train(_batch_episodes, _batch_deltas, _train_policy_iterations)


def train_critic_model(_model: Sequential, _batch_episodes: List[utils.Episode],  _discount_factor:float,_train_critic_iterations:int, _critic_learning_rate:float):
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=_critic_learning_rate)
    # Create a training based on model with the desired parameters.
    tr = CriticTraining(_model, optimizer=critic_optimizer, discount_factor=_discount_factor)
    tr.train(_batch_episodes, _train_critic_iterations)



if __name__ == "__main__":
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
          f.write('iteration, decision step, thetaeconomy, thetamanagement, fmanagement, thetaenviron, fenviron\n')

    #Check that the result3 file for evaluation does not exist
    try:
      os.remove(results4_filepath)
    except OSError:
          pass
    #First line contains the title
    with open(results4_filepath, 'a') as f:
          f.write('iteration, decision step, budget_obs, fadopters_obs, ceco, cman, cenv, cleft, mean_thetaeco, mean_thetaman, mean_thetaenv, std_thetaeco, std_thetaman, std_thetaenv\n')

    
    actor_model = utils.mlp(n_observations, layers_sizes)
    print('actor_model.summary()', actor_model.summary())
    critic_model = utils.mlp(n_observations, np.append(layers_sizes[:-1],  1))
    print('critic_model.summary()', critic_model.summary())
    

    batch_episodes = []
    batch_deltas = []
    i_episode = 0
    #For each training iteration
    for i_iter in range(max_training_iters):
        print('i_iter', i_iter)
        tic_b_iter = time.time()
        batch_episodes.clear()
        batch_deltas.clear()
 
        for i_batch in range(batch_size):
            episode = utils.Episode()
            episode.set_id(i_episode)
            tic_setting_gama = time.time()
            #init server
            listener_port = gamainteraction.listener_init(gama_interaction_loop, episode)

            #generate xml for the headless
            xml_path = gamainteraction.generate_gama_xml(   headless_dir,
                                                            listener_port,
                                                            gaml_file_path,
                                                            experiment_name
                                                            )
            print('\t','setting up gama', time.time()-tic_setting_gama)
            # run gama
            if gamainteraction.run_gama_headless(xml_path,
                                                 headless_dir,
                                                 run_headless_script_path) == 2:
                print("simulation " + str(i_episode) + " ended in error, stopping everything")
                sys.exit(-1)
            batch_episodes.append(episode)
            #
            # 1. Compute discounted deltas for this episode
            #
            # 1.1 Compute the values for observations present in the episode
            episode_values = critic_model(np.vstack(episode.observations))
            episode_values = np.append(episode_values, 0)
            # 1.2 Compute discounted deltas for this episode
            episode_deltas = utils.discount_rewards(episode.rewards + discount_factor*episode_values[1:] - episode_values[:-1], discount_factor * gae_lambda)
            episode_deltas = episode_deltas.astype('float32')
            # 1.3 Store deltas to the batch
            batch_deltas.append(episode_deltas)

                    
        i_batch_episode = 0
        for episode in batch_episodes:
            sum_episode_rewards = sum(episode.rewards) #the sum of discounted? rewards of an episode is the basic component for all performance stadistics
            #store in the resuls file the sum of rewards for this episode
            with open(results_filepath, 'a') as f:
                f.write(str(sum_episode_rewards)+'\n')
            # Save the number of adopters end of each episode for statistics
            with open(results2_filepath, 'a') as f:
                f.write(str(episode.observations[-1][1])+'\n')
            print('episode.observations[-2][0]', episode.observations[-2][0])
            print('episode.observations[-1][0]', episode.observations[-1][0])
            print('Batch episode:', i_batch_episode,'\t', ' reward:', sum_episode_rewards, ' fraction of adopters ', str(episode.observations[-1][1]), '\n')
            i_batch_episode = i_batch_episode + 1

        tic_b = time.time()
        print('discount_factor', discount_factor)
        train_actor_model(actor_model, batch_episodes, batch_deltas, clipping_ratio, target_kl, train_policy_iterations, policy_learning_rate)
        train_critic_model(critic_model, batch_episodes, discount_factor, train_critic_iterations, critic_learning_rate)
        training_time = time.time() - tic_b
        print('\t','training time', training_time)
        print('it:',i_iter,'\t time:',time.time()-tic_b_iter)   
        i_episode = i_episode + 1  
    #Store model to reuse to pursue learning       
    actor_model.save(ACTORMODELPATH, include_optimizer=False)
    critic_model.save(CRITICMODELPATH, include_optimizer=False)
