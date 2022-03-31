import socket
import sys
from keras import Sequential

import gamainteraction
from policy import Policy
import training
import numpy as np
import matplotlib.pyplot as plt
import gama
from numpy.random import seed
import numpy.typing as npt
from typing import List
from user_local_variables import *

seed(1)


### Global variables ###
n_episodes      = 1     # Number of episodes for the training
n_actions       = 5     # Number of actions discrete actions of the social planner, can be modified for testing
n_observations  = 3     # Number of observations from the state of the social planner, can be modified for testing

MODELPATH                   = 'nngamma' # Path to the file where to store the neural network
sumrewards: List[float]     = []


# The loop of interaction between the gama simulation and the model
def gama_interaction_loop(gama_simulation: socket) -> None:

    global sumrewards
    global model
    policy_manager: Policy = Policy(model)
    try:

        observations:   List[npt.NDArray[np.float64]]   = []
        actions:        List[np.float64]                = []
        rewards:        List[np.float64]                = []

        while True:

            # we wait for the simulation to send the observations
            print("waiting for observations")
            received_observations: str = gama_simulation.recv(1024).decode()

            if received_observations == "END":
                print("simulation has ended")
                break

            print("the model received:", received_observations, "and will start processing it")
            obs: npt.NDArray[np.float64] = gamainteraction.string_to_nparray(received_observations.replace("END", ""))
            observations += [obs]

            # we then compute a policy and send it back to gama
            policy = gamainteraction.process_observations(policy_manager, obs, n_actions)
            actions += [policy]

            str_action = gamainteraction.action_to_string(np.array(policy))
            print("the model is sending policy", str_action)
            gama_simulation.send(str_action.encode())

            # we finally wait for the reward
            print("The model is waiting for the reward")
            policy_reward = gama_simulation.recv(1024).decode()
            print("the model received the reward", policy_reward)
            rewards += [float(policy_reward)]
            gamainteraction.process_reward(policy_reward, policy, received_observations)

            # new line for better understanding of the logs
            print()
    except ConnectionResetError:
        print("connection reset, end of simulation")
    except:
        print(sys.exc_info()[0])

    gama_simulation.send("over".encode()) #we send a message for the simulation to wait before closing
    train_model(model, observations, actions, rewards)
    sumrewards += [sum(rewards)]


def train_model(_model: Sequential, _observations: List[npt.NDArray[np.float64]], _actions: List[int], _rewards: List[float]):
    # Create a training based on model with the desired parameters.
    tr = training.Training(_model)
    print("observations:", _observations)
    print("actions:", _actions)
    print("rewards:", _rewards)
    tr.train_step(np.vstack(_observations),
                  np.array(_actions),
                  np.array(_rewards))


if __name__ == "__main__":

    #create neural network model for the environment
    #by now is just a dummy network for integration testing on the number of observations and number of actions
    model = gama.create_model(n_observations, n_actions)
    #save this initial  model to the disk
    model.save(MODELPATH, include_optimizer=False)

    # # Save the sum of rewards of each episode for statistics
    # sumrewards = []
    #For each episode
    for i_episode in range(n_episodes):

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

    input() # to wait
    print("sum rewards", sumrewards)

    # Plot the results
    plt.plot(range(n_episodes), sumrewards)
    plt.xlabel('Episode'); plt.ylabel('Sum of rewards')
    plt.savefig('gama_results.png')
    plt.show()
    # Print statistics measurements
    print('Mean:', np.mean(sumrewards), ' Median:', np.median(sumrewards), ' Standard dev:', np.std(sumrewards))
