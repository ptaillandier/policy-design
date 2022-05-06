import socket
import time
import sys
from keras import Sequential
import os
import gamainteraction
import policy_discrete 
import training_discrete
import numpy as np
from typing import List, TextIO, IO
import argparse
import numpy.typing as npt
from user_local_variables import *
import utils

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
n_episodes = max_training_iters*batch_size #The total number of episodes explored will be the number of iterations for the training par the size of batch examples processed on each training
print("Total number of episodes =", n_episodes)
print("max_training_iters", max_training_iters)
# Actions (5) 
# 1. Thetaeconomy - Fraction of financial support [0,1] 
# 2. Thetamanagement - Fraction of increment on the skill of trained agents [0,1]
# 3. Fmanagement - Fraction of individuals chosen randomly to be trained [0,1]
# 4. Thetaenvironment - Fraction of environmental awareness [0,1]
# 5. Fenvironment - Fraction of individuals chosen randomly to increase environmental awaraness [0,1]
# Cost computed as Nman(100*Fman)*thetamanagement + N_new_adopters(max100,observed in next state)*thetaeconomy+0.5*Nenv(100*Fenv)*thetaenv*100
n_actions       = 5    
discretization_step = 0.1
layers_sizes.append(n_actions*len(np.arange(0,1+discretization_step,discretization_step))) #Add the last output layer considering the number of actions
print('layers_sizes', layers_sizes)
# Observations (3) 
# 1. Remaining budget - Remaining budget available to implement public policies
# 2. Fraction of adopters - Fraction of adopters [0,1]
# 3. Remaining time before ending the simulation - Unit (in seconds)

n_observations  = 3     # Number of observations from the state of the social planner, can be modified for testing


# Rewards
# 1. Evolution of the intention of adoption (mean_intention - previous_mean_intention) / previous_mean_intention)
MODELPATH                   = 'nngamma' # Path to the file where to store the neural network
results_filepath            = 'results_sum_rewards_discretized.csv'
results2_filepath           = 'results_number_adopters_discretized.csv'

# Process actions
def process_actions(index_actions, discretization_step):
    values_actions = np.arange(0,1+discretization_step,discretization_step)
    actions = [values_actions[index_action.numpy()[0]] for index_action in index_actions]
    print('actions after processing', actions)
    return actions

# The loop of interaction between the gama simulation and the model
def gama_interaction_loop(gama_simulation: socket, episode: utils.Episode) -> None:

    
    global model
    global discretization_step
    policy_manager = policy_discrete.PolicyDiscrete(model)
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
           print('calling policy_manager')
           model_actions, _ = policy_manager.choose_action(obs, n_actions)
           print('model_actions', model_actions)
           simulation_actions = process_actions(model_actions, discretization_step)
           print('simulation_actions', simulation_actions)
           time_updating_policy = time_updating_policy + time.time() - tic_b

           str_action = gamainteraction.action_to_string(np.array(simulation_actions))
           print("model sending policy:(thetaeconomy ,thetamanagement,fmanagement,thetaenvironment,fenvironment)", str_action)
           gama_socket_as_file.write(str_action)
           gama_socket_as_file.flush()

           tic_b = time.time()
           # we finally wait for the reward
           #print("The model is waiting for the reward")
           policy_reward = gama_socket_as_file.readline()
           time_simulation = time_simulation + time.time() - tic_b
           print("model received reward:", policy_reward)
               
           gamainteraction.process_reward(policy_reward, simulation_actions, received_observations)
           episode.add_experience(obs, model_actions, float(policy_reward))
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

def train_model(_model: Sequential, _batch_episodes: List[utils.Episode], _discount_factor:float):
    # Create a training based on model with the desired parameters.
    tr = training_discrete.TrainingDiscrete(_model, discount_factor=_discount_factor)

    tr.train(_batch_episodes)




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


    #create neural network model for the environment
    #model = gama.create_model(n_observations, n_actions)
    
    model = utils.mlp(n_observations, layers_sizes)
    print('model.summary()', model.summary())
    #save this initial model to the disk
    model.save(MODELPATH, include_optimizer=False)
    print('max_training_iters', max_training_iters)
    #For each training iteration
    for i_iter in range(max_training_iters):
        print('i_iter', i_iter)
        tic_b_iter = time.time()
        batch_episodes = []
        for i_batch in range(batch_size):
            episode = utils.Episode()
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
        
        i_episode = 0
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
            print('Episode:', i_episode,'\t', ' reward:', sum_episode_rewards, ' fraction of adopters ', str(episode.observations[-1][1]), '\n')
            i_episode = i_episode + 1

        tic_b = time.time()
        print('discount_factor', discount_factor)
        train_model(model, batch_episodes, discount_factor)
        training_time = time.time() - tic_b
        print('\t','training time', training_time)
        print('it:',i_iter,'\t time:',time.time()-tic_b_iter)       
