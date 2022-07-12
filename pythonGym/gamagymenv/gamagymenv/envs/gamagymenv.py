import os
import socket
from _thread import *

import gym
import numpy as np
from typing import Optional, TextIO, IO, List
from gym import spaces
import numpy.typing as npt
from keras import Sequential
from ray.experimental.tf_utils import tf

from episode import Episode
from gamagymenv.user_local_variables import *
from policy_dirichlet import Policy


def string_to_nparray(array_as_string: str) -> npt.NDArray[np.float64]:

    # first we remove brackets and parentheses
    clean = "".join([c if c not in "()[]{}" else '' for c in str(array_as_string)])

    # then we split into numbers
    nbs = [float(nb) for nb in filter(lambda s: s.strip() != "", clean.split(','))]

    return np.array(nbs)


# Generates an XML file that can be used to run gama in headless mode, listener_port is used as a parameter of
# the simulation to communicate through tcp
# returns the path of the XML
def generate_gama_xml(headless_dir: str, listener_port: int, gaml_experiment_path: str, experiment_name: str) -> str:
    # TODO: ajouter nb de simulations dans le xml

    sim_file_path = os.path.join(headless_dir, "tmp_sim.xml")
    sim_file = open(sim_file_path, "w+")

    #We fill the file with the appropriate xml needed to run the gama experiment
    sim_file.write(f"""<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>
    <Experiment_plan>
        <Simulation id=\"2\" sourcePath=\"{gaml_experiment_path}\" experiment=\"{experiment_name}\" seed="0">
            <Parameters>
                <Parameter name=\"port\" type=\"INT\" value=\"{listener_port}\"/>
            </Parameters>
            <Outputs />
        </Simulation>
    </Experiment_plan>""")

    sim_file.close()

    return sim_file_path


# Runs gama simulations in headless mode following the description of the xml file
def run_gama_headless(xml_file_path: str, headless_dir: str, gama_headless_script_path: str) -> int:
    cmd = f"cd \"{headless_dir}\" && \"{gama_headless_script_path}\" \"{xml_file_path}\" out"
    print("running gama with command: ", cmd)
    return os.system(cmd)


# Function that builds a customized multilayer perceptron (MLP) neural network
def mlp(n_observations, sizes, activation='relu', output_activation=None, last_layer_scaling=np.sqrt(2)):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(n_observations,)))

    for size in sizes[:-1]:
        x = tf.keras.layers.Dense(units = size,
                                  activation = activation,
                                  kernel_initializer = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))
                                  )
        model.add(x)

    model.add(tf.keras.layers.Dense(units=sizes[-1],
                                    activation=output_activation,
                                    kernel_initializer = tf.keras.initializers.Orthogonal(gain=last_layer_scaling)
                                    ))
    return model


# Converts an action to a string to be sent to the simulation
def action_to_string(actions: npt.NDArray[np.float64]) -> str:
    return ",".join([str(action) for action in actions]) + "\n"


# Takes the observations from the simulation, sent as a string
# Computes the appropriate policy and returns it
def process_observations(policy_manager: Policy, observations: npt.NDArray[np.float64]):
    action, processed_observations, bounds, nn_outputs = policy_manager.choose_action(observations)
    return action, bounds, nn_outputs


class GamaGymEnv(gym.Env):

    # ENVIRONMENT CONSTANTS
    init_budget:        float   = 10.0
    n_observations:     int     = 3
    n_execution_steps:  int     = 9
    steps_before_done:  int     = n_execution_steps
    _max_episode_steps: int     = 11
    n_times_4_action:   int     = 10  # Number of times in which the policy maker can change the public policy (time horizon: 5 years)
    results3_filepath:  str     = 'results_actions.csv'
    results4_filepath:  str     = 'results_nnoutputs.csv'

    # Simulation execution variables
    state:                      npt.NDArray[np.float64]
    current_step:               int
    model:                      Sequential
    policy_manager:             Policy
    gama_socket:                socket
    gama_simulation_as_file     = None # For some reason the typing doesn't work
    time_simulation:            int
    i_experience:               int
    last_obs:                   int

    # MODEL CREATION PARAMETERS
    layers_sizes:           List[int]   = [32, 32]
    last_layer_scaling:     float       = 0.01
    activation_function:    str         = 'tanh'
    episode:                Episode

    def __init__(self):

        # OBSERVATION SPACE:
        # 1. Remaining budget                               - Remaining budget available to implement public policies
        # 2. Fraction of adopters                           - Fraction of adopters [0,1]
        # 3. Remaining time before ending the simulation    - Unit (in steps)
        obs_high_bounds = np.array([np.inf, 1.0, self._max_episode_steps])
        obs_low_bounds  = np.array([0.0, 0.0, 0.0])
        self.observation_space = spaces.Box(obs_low_bounds, obs_high_bounds, dtype=np.float32)

        # ACTIONS:
        # 1. Thetaeconomy       - Fraction of financial support [0,1]
        # 2. Thetamanagement    - Fraction of increment on the skill of trained agents [0,1]
        # 3. Fmanagement        - Fraction of individuals chosen randomly to be trained [0,1]
        # 4. Thetaenvironment   - Fraction of environmental awareness [0,1]
        # 5. Fenvironment       - Fraction of individuals chosen randomly to increase environmental awareness [0,1]
        action_high_bounds  = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        action_low_bounds   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.action_space   = spaces.Box(action_low_bounds, action_high_bounds, dtype=np.float32)

    # Execute one step of the simulation
    def step(self, action):

        print("step")
        self.state, end = self.read_observations()
        print("observations received", self.state, end)

        action, action_env, nn_outputs = process_observations(self.policy_manager, self.state)

        # sending actions
        str_action = action_to_string(np.array(action_env))
        print("model sending policy:(thetaeconomy ,thetamanagement,fmanagement,thetaenvironment,fenvironment)", str_action)
        self.gama_simulation_as_file.write(str_action)
        self.gama_simulation_as_file.flush()

        # we finally wait for the reward
        policy_reward = self.gama_simulation_as_file.readline()
        reward = float(policy_reward)
        print("model received reward:", policy_reward, " as a float: ", reward)
        self.episode.add_experience(self.state, action, reward)

        # We store everything in files
        self.store_results(nn_outputs, str_action)

        # end of step, check if over
        self.i_experience += 1

        # TODO: is steps_before_done used somewhere else ? we can just use i_experience
        if self.steps_before_done == 0:
            end = True
            self.gama_simulation_as_file.write("over\n")  # we send a message for the simulation to wait before closing
            self.gama_simulation_as_file.flush()
        else:
            self.steps_before_done -= 1

        return np.array(self.state, dtype=np.float32), reward, end, {}

    # Save the action and nn_outputs to files
    def store_results(self, nn_outputs, str_action: str):
        # store in the result file the actions taken
        with open(self.results3_filepath, 'a') as f:
            f.write(str(self.episode.id) + ',' + str(self.i_experience) + ',' + str_action)
        # store in the result file the outputs of the nn
        with open(self.results4_filepath, 'a') as f:
            f.write(
                str(self.episode.id) + ',' + str(self.i_experience) + ',' + str(self.state[0]) + ',' + str(
                    self.state[1]) + ',' + ','.join(
                    [str(output) for output in nn_outputs]) + '\n')

    # Connect with the current running gama simulation
    def wait_for_gama_to_connect(self):

        #The server is waiting for clients to connect
        conn, addr = self.gama_socket.accept()
        print("gama connected:", conn, addr)
        self.gama_simulation_as_file = conn.makefile(mode='rw')
        print(self.gama_simulation_as_file)

    def read_observations(self):

        received_observations: str = self.gama_simulation_as_file.readline()
        print("model received:", received_observations)

        obs     = string_to_nparray(received_observations.replace("END", ""))
        obs[2]  = float(self.n_times_4_action - self.i_experience)  # We change the last observation to be the number of times that remain for changing the policy

        over = "END\n" in received_observations

        return obs, over


    # Must reset the simulation to its initial state
    # Should return the initial observations
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):

        print("reset")
        self.i_experience           = 0
        self.last_obs               = -1
        self.current_step           = 0
        self.model                  = self.init_model()
        self.policy_manager         = Policy(self.model)
        self.episode                = Episode()

        # Starts gama and get initial state
        self.run_gama_simulation()
        self.wait_for_gama_to_connect()
        self.state, end = self.read_observations() #TODO: probably useless
        if end:
            self.episode.set_last_observation(self.state)

        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            #TODO ?
            return np.array(self.state, dtype=np.float32), {}

    def init_model(self):
        return mlp(self.n_observations,
                   self.layers_sizes,
                   activation=self.activation_function,
                   last_layer_scaling=self.last_layer_scaling
                   )

    # Initialize the socket to communicate with gama
    def listener_init(self) -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket successfully created")

        s.bind(('', 0))  # localhost + port given by the os
        port = s.getsockname()[1]
        print("Socket bound to %s" % port)

        s.listen()
        print("Socket started listening")

        self.gama_socket = s
        return port

    # Init the server + run gama
    def run_gama_simulation(self):
        port = self.listener_init()
        xml_path = generate_gama_xml(headless_dir, port, gaml_file_path, experiment_name)
        start_new_thread(run_gama_headless, (xml_path, headless_dir, run_headless_script_path))

