import shlex
import subprocess
from subprocess import CompletedProcess

import gym
import time
import os
import sys
import socket
from _thread import *
import numpy as np
import numpy.typing as npt
from typing import Optional
from gym import spaces
import psutil

class GamaEnv(gym.Env):


    # USER LOCAL VARIABLES
    headless_dir: str               # Root directory for gama headless
    run_headless_script_path: str   # Path to the script that runs gama headless
    gaml_file_path: str             # Path to the gaml file containing the experiment/simulation to run
    experiment_name: str            # Name of the experiment to run


    # ENVIRONMENT CONSTANTS
    max_episode_steps:  int     = 11

    # Simulation execution variables
    gama_pid: int               = -1 # The pid of the command running gama
    gama_socket                 = None
    gama_simulation_as_file     = None # For some reason the typing doesn't work
    gama_simulation_connection  = None # Resulting from socket create connection
    def __init__(self, headless_directory: str, headless_script_path: str, gaml_experiment_path: str, gaml_experiment_name: str):

        self.headless_dir               = headless_directory
        self.run_headless_script_path   = headless_script_path
        self.gaml_file_path             = gaml_experiment_path
        self.experiment_name            = gaml_experiment_name

        #print("INIT")
        # OBSERVATION SPACE:
        # 1. Remaining budget                               - Remaining budget available to implement public policies
        # 2. Fraction of adopters                           - Fraction of adopters [0,1]
        # 3. Number of times policy applied action    - Unit (in steps)
        obs_high_bounds = np.array([50.0, 1.0, self.max_episode_steps])
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
        #print("END INIT")
            
    def step(self, action):
        try:
            #print("STEP")
            # sending actions
            str_action = GamaEnv.action_to_string(np.array(action))
            #print("model sending policy:(thetaeconomy ,thetamanagement,fmanagement,thetaenvironment,fenvironment)", str_action)
            self.gama_simulation_as_file.write(str_action)
            self.gama_simulation_as_file.flush()
            #print("model sent policy, now waiting for reward")
            # we wait for the reward
            policy_reward = self.gama_simulation_as_file.readline()
            reward = float(policy_reward)

            #print("model received reward:", policy_reward, " as a float: ", reward)
            self.state, end = self.read_observations()
            #print("observations received", self.state, end)
            # If it was the final step, we need to send a message back to the simulation once everything done to acknowledge that it can now close
            if end:
                self.gama_simulation_as_file.write("END\n")
                self.gama_simulation_as_file.flush()
                self.gama_simulation_as_file.close()
                self.gama_simulation_connection.shutdown(socket.SHUT_RDWR)
                self.gama_simulation_connection.close()
                self.gama_socket.shutdown(socket.SHUT_RDWR)
                self.gama_socket.close()
        except ConnectionResetError:
            print("connection reset, end of simulation")
        except:
            print("EXCEPTION pendant l'execution")
            print(sys.exc_info()[0])
            sys.exit(-1)
        #print("END STEP")
        return np.array(self.state, dtype=np.float32), reward, end, {} 

    # Must reset the simulation to its initial state
    # Should return the initial observations
    def reset(self,*,seed: Optional[int] = None,return_info: bool = False,options: Optional[dict] = None ): 
        #print("RESET")
        #print("self.gama_simulation_as_file", self.gama_simulation_as_file)
        #print("self.gama_simulation_connection",
        #      self.gama_simulation_connection)
        #Check if the environment terminated 
        if self.gama_simulation_connection is not None:
            #print("self.gama_simulation_connection.fileno()",
            #      self.gama_simulation_connection.fileno())
            if self.gama_simulation_connection.fileno() != -1:
                self.gama_simulation_connection.shutdown(socket.SHUT_RDWR)
                self.gama_simulation_connection.close()
                self.gama_socket.shutdown(socket.SHUT_RDWR)
                self.gama_socket.close()
        if self.gama_simulation_as_file is not None:
            self.gama_simulation_as_file.close()
            self.gama_simulation_as_file = None

        self.clean_subprocesses()


        tic_setting_gama = time.time()
        # Starts gama and get initial state
        self.run_gama_simulation()
        self.wait_for_gama_to_connect()
        self.state, end = self.read_observations()
        print('\t','setting up gama', time.time()-tic_setting_gama)
        #print('after reset self.state', self.state)
        #print('after reset end', end)
        #print("END RESET")
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
     
        # OLD
        #self.steps_before_done = self.n_execution_steps

    def clean_subprocesses(self):
        print("calling clean_subprocesses")
        print("self.gama_pid", self.gama_pid)
        if self.gama_pid > 0:
            parent = psutil.Process(self.gama_pid)
            for child in parent.children(recursive=True):  # or parent.children() for recursive=False
                child.kill()
            parent.kill()

    def __del__(self):
        print("calling __del__")
        self.clean_subprocesses()

    # Init the server + run gama
    def run_gama_simulation(self):
        port = self.listener_init()
        xml_path = GamaEnv.generate_gama_xml(self.headless_dir, port, self.gaml_file_path, self.experiment_name)
        self.thread_id = start_new_thread(GamaEnv.run_gama_headless, (self, xml_path, self.headless_dir, self.run_headless_script_path))

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
    def run_gama_headless(self, xml_file_path: str, headless_dir: str, gama_headless_script_path: str) -> int:
        cmd = f"cd \"{headless_dir}\" && \"{gama_headless_script_path}\" \"{xml_file_path}\" out"
        print("running gama with command: ", cmd)
        #cmd_to_args = shlex.split(cmd)
        res = subprocess.Popen(cmd, shell=True)
        self.gama_pid = res.pid
        return res.returncode
  
    # Connect with the current running gama simulation
    def wait_for_gama_to_connect(self):

        #The server is waiting for clients to connect
        self.gama_simulation_connection, addr = self.gama_socket.accept()
        print("gama connected:", self.gama_simulation_connection, addr)
        self.gama_simulation_as_file = self.gama_simulation_connection.makefile(mode='rw')
        print("self.gama_simulation_as_file", self.gama_simulation_as_file)

    def read_observations(self):

        received_observations: str = self.gama_simulation_as_file.readline()
        #print("model received:", received_observations)

        over = "END" in received_observations
        obs  = GamaEnv.string_to_nparray(received_observations.replace("END", ""))
        #obs[2]  = float(self.n_times_4_action - self.i_experience)  # We change the last observation to be the number of times that remain for changing the policy

        return obs, over

    # Converts a string to a numpy array of floats
    @classmethod
    def string_to_nparray(cls, array_as_string: str) -> npt.NDArray[np.float64]:
        # first we remove brackets and parentheses
        clean = "".join([c if c not in "()[]{}" else '' for c in str(array_as_string)])
        # then we split into numbers
        nbs = [float(nb) for nb in filter(lambda s: s.strip() != "", clean.split(','))]
        return np.array(nbs)


    # Converts an action to a string to be sent to the simulation
    @classmethod
    def action_to_string(cls, actions: npt.NDArray[np.float64]) -> str:
        return ",".join([str(action) for action in actions]) + "\n"
