import os
import socket
from _thread import *
from typing import Callable, TextIO
import numpy.typing as npt
import numpy as np
import utils
from policy import Policy


def simulation_loop(sock_server, gama_interaction_loop, episode) -> None:
    #The server is waiting for clients to connect
    conn, addr = sock_server.accept()
    #print("connected", conn, addr)
    #One client connected = one gama simulation
    gama_interaction_loop(conn, episode)

def listener_init(gama_interaction_loop_function, episode) -> int:

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #print("Listening Socket successfully created")

    s.bind(('', 0)) #localhost + port given by the os
    port = s.getsockname()[1]
    #print("Listening socket bound to %s" % port)

    s.listen()
    #print("Listening socket started listening")

    start_new_thread(simulation_loop, (s, gama_interaction_loop_function, episode))

    return port


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
def run_gama_headless(sim_file_path: str, headless_dir: str, run_headless_script_path: str) -> int:
    return os.system(f"cd \"{headless_dir}\" && \"{run_headless_script_path}\" \"{sim_file_path}\" out")


def string_to_nparray(array_as_string: str) -> npt.NDArray[np.float64]:

    #first we remove brackets and parentheses
    clean = "".join([c if c not in "()[]{}" else '' for c in str(array_as_string)])

    #then we split into numbers
    nbs = [float(nb) for nb in filter(lambda s: s.strip() != "",  clean.split(','))]

    return np.array(nbs)

#Converts an action to a string to be sent to the simulation
def action_to_string(actions: npt.NDArray[np.float64]) -> str:
    return ",".join([str(action) for action in actions]) + "\n"


# Takes the observations from the simulation, sent as a string
# Computes the appropriate policy and returns it
def process_observations(policy_manager: Policy, observations: npt.NDArray[np.float64], n_actions: int):
    action, processed_observations, bounds = policy_manager.choose_action(observations, n_actions)
    return action, bounds


# Takes the reward returned after applying the policy given the observations and update the model accordingly
def process_reward(policy_reward: str, policy_applied: str, received_observations: str) -> None:
    pass




