import gym
import os
import socket
from _thread import *
import numpy as np
import numpy.typing as npt
from typing import Optional
from gym import spaces
from gamaenv.user_local_variables import *

class GamaEnv(gym.Env):
    # ENVIRONMENT CONSTANTS
    init_budget:        float   = 10.0
    n_observations:     int     = 3
    n_execution_steps:  int     = 9
    steps_before_done:  int     = n_execution_steps
    max_episode_steps: int     = 11
    n_times_4_action:   int     = 10  # Number of times in which the policy maker can change the public policy (time horizon: 5 years)

    # Simulation execution variables
    gama_socket:                socket
    gama_simulation_as_file     = None # For some reason the typing doesn't work

    def __init__(self):
        print("INIT")
        # OBSERVATION SPACE:
        # 1. Remaining budget                               - Remaining budget available to implement public policies
        # 2. Fraction of adopters                           - Fraction of adopters [0,1]
        # 3. Remaining time before ending the simulation    - Unit (in steps)
        obs_high_bounds = np.array([50.0, 1.0, np.Inf])
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
        print("END INIT")
            
    def step(self, action):
        print("STEP")
        print("np.array(action)", np.array(action))
        # sending actions
        str_action = GamaEnv.action_to_string(np.array(action))
        print("model sending policy:(thetaeconomy ,thetamanagement,fmanagement,thetaenvironment,fenvironment)", str_action)
        self.gama_simulation_as_file.write(str_action)
        self.gama_simulation_as_file.flush()
        # we wait for the reward
        policy_reward = self.gama_simulation_as_file.readline()
        reward = float(policy_reward)
        print("model received reward:", policy_reward, " as a float: ", reward)
        self.state, end = self.read_observations()
        print("observations received", self.state, end)
        print("END STEP")
        return np.array(self.state, dtype=np.float32), reward, end, {} 

    # Must reset the simulation to its initial state
    # Should return the initial observations
    def reset(self,*,seed: Optional[int] = None,return_info: bool = False,options: Optional[dict] = None ): 
        print("RESET")
        # Starts gama and get initial state
        self.run_gama_simulation()
        self.wait_for_gama_to_connect()
        #self.state = np.random.rand(1, self.n_observations).flatten()
        self.state, end = self.read_observations() #TODO: probably useless
        print('self.state', self.state)
        print('type(self.state)', type(self.state))
        print("END RESET")
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
     
        # OLD
        #self.steps_before_done = self.n_execution_steps


    # Init the server + run gama
    def run_gama_simulation(self):
        port = self.listener_init()
        xml_path = GamaEnv.generate_gama_xml(headless_dir, port, gaml_file_path, experiment_name)
        start_new_thread(GamaEnv.run_gama_headless, (xml_path, headless_dir, run_headless_script_path))

    # Generates an XML file that can be used to run gama in headless mode, listener_port is used as a parameter of
    # the simulation to communicate through tcp
    # returns the path of the XML
    @classmethod
    def generate_gama_xml(cls, headless_dir: str, listener_port: int, gaml_experiment_path: str, experiment_name: str) -> str:
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

    # Runs gama simulations in headless mode following the description of the xml file
    @classmethod
    def run_gama_headless(cls, xml_file_path: str, headless_dir: str, gama_headless_script_path: str) -> int:
        cmd = f"cd \"{headless_dir}\" && \"{gama_headless_script_path}\" \"{xml_file_path}\" out"
        print("running gama with command: ", cmd)
        return os.system(cmd)
  
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
        #obs[2]  = float(self.n_times_4_action - self.i_experience)  # We change the last observation to be the number of times that remain for changing the policy
        over = "END\n" in received_observations

        return obs, over

    # Converts an action to a string to be sent to the simulation
    @classmethod
    def action_to_string(cls, actions: npt.NDArray[np.float64]) -> str:
        return ",".join([str(action) for action in actions]) + "\n"
