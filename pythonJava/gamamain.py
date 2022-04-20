import socket
import time
import sys
from keras import Sequential
import os
import gamainteraction
from policy import Policy
import training
import numpy as np
import matplotlib.pyplot as plt
import gama
import argparse
from numpy.random import seed
import numpy.typing as npt
from typing import List
from user_local_variables import *

parser = argparse.ArgumentParser(description='Runs the experiment for the gama policy design environment')
parser.add_argument(
    "--iters",
    type=int,
    default=3,
    help="Number of iterations.",
)

args = parser.parse_args()

### Global variables ###
n_episodes      = args.iters     # Number of episodes for the training
# Actions (5) 
# 1. Fmanagement - Fraction of individuals chosen randomly to be trained [0,1]
# 2. Thetamanagement - Fraction of increment on the skill of trained agents [0,1]
# 3. Thetaeconomy - Fraction of financial support [0,1] 
# 4. Fenvironment - Fraction of individuals chosen randomly to increase environmental awaraness [0,1]
# 5. Thetaenvironment - Fraction of environmental awareness [0,1]
# The cost of actions is: 2*Nman(100*Fman)*thetamanagement + N_new_adopters(max100,observed in next state)*thetaeconomy+Nenv(100*Fenv)*thetaenv*100
n_actions       = 5    
# Observations (3) 
# 1. Remaining budget - Remaining budget available to implement public policies
# 2. Fraction of adopters - Fraction of adopters [0,1]
# 3. Remaining time before ending the simulation - Unit (in seconds)

n_observations  = 3     # Number of observations from the state of the social planner, can be modified for testing


# Rewards
# 1. Evolution of the intention of adoption (mean_intention - previous_mean_intention) / previous_mean_intention)
MODELPATH                   = 'nngamma' # Path to the file where to store the neural network
results_filepath            = 'results_sum_of_rewards_gama.csv'
results2_filepath           = 'results_number_of_adopters_gama.csv'

# The loop of interaction between the gama simulation and the model
def gama_interaction_loop(gama_simulation: socket) -> None:

    
    global model
    policy_manager: Policy = Policy(model)
    try:

        observations:   List[npt.NDArray[np.float64]]   = []
        actions:        List[np.float64]                = []
        rewards:        List[np.float64]                = []
        n_times_4_action = 9 # Number of times in which the policy maker can change the public policy (time horizon: 5 years)
        while True:

            # we wait for the simulation to send the observations
            print("waiting for observations")
            received_observations: str = gama_simulation.recv(1024).decode()

            if received_observations == "END":
                print("simulation has ended")
                break

            print("model received:", received_observations)
            obs: npt.NDArray[np.float64] = gamainteraction.string_to_nparray(received_observations.replace("END", ""))
            obs[2] =  float(n_times_4_action-len(rewards)) #We change the last observation to be the number of times that remain for changing the policy
            observations += [obs]
            
            # we then compute a policy and send it back to gama
            policy = gamainteraction.process_observations(policy_manager, obs, n_actions)
            actions += [policy]

            str_action = gamainteraction.action_to_string(np.array(policy))
            print("model sending policy:(nman,thetaman,thetaeconomy,nenv,thetaenv)", str_action)
            gama_simulation.send(str_action.encode())

            # we finally wait for the reward
            #print("The model is waiting for the reward")
            policy_reward = gama_simulation.recv(1024).decode()
            print("model received reward:", policy_reward)
            rewards += [float(policy_reward)]
            gamainteraction.process_reward(policy_reward, policy, received_observations)

            # new line for better understanding of the logs
            print()
    except ConnectionResetError:
        print("connection reset, end of simulation")
    except:
        print(sys.exc_info()[0])

    gama_simulation.send("over\n".encode()) #we send a message for the simulation to wait before closing
    train_model(model, observations, actions, rewards)
    # Save the sum of rewards of each episode for statistics
    with open(results_filepath, 'a') as f:
        f.write(str(sum(rewards))+'\n')
    # Save the number of adopters end of each episode for statistics
    with open(results2_filepath, 'a') as f:
        f.write(str(observations[-1][1])+'\n')
    print('observations[-2][0]', observations[-2][0])
    print('observations[-1][0]', observations[-1][0])
    


def train_model(_model: Sequential, _observations: List[npt.NDArray[np.float64]], _actions: List[int], _rewards: List[float]):
    # Create a training based on model with the desired parameters.
    tr = training.Training(_model)
    

    tr.train_step(np.vstack(_observations),
                  np.array(_actions),
                  np.array(_rewards))


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
    model = gama.create_model(n_observations, n_actions)
    #save this initial model to the disk
    model.save(MODELPATH, include_optimizer=False)
    #For each episode
    for i_episode in range(n_episodes):
        tic_b_iter = time.time()
        #init server
        listener_port = gamainteraction.listener_init(gama_interaction_loop)

        #generate xml for the headless
        xml_path = gamainteraction.generate_gama_xml(   headless_dir,
                                                        listener_port,
                                                        gaml_file_path,
                                                        experiment_name
                                                        )
        # run gama
        gamainteraction.run_gama_headless(xml_path,
                                          headless_dir,
                                          run_headless_script_path)
        print('it:',i_episode,'\t time:',time.time()-tic_b_iter)       
