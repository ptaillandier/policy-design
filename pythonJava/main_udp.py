import os
import random
import socket
from _thread import *
import tensorflow as tf
import numpy as np
import scipy


print("TensorFlow version:", tf.__version__)


def server_loop(sock_listen: socket) -> None:

    port: int = -1
    sock_send: socket = None

    obj = {"salut": 123, 3: 3.0, "oui": [1, 2, 3], "non": {"a": 1, "b": "cdef"}}
    while True:
        data = sock_listen.recv(10240)
        print("reçu:", data)
        #The first message is send to set the sender port
        if port == -1:
            port = int(data)
            sock_send = sender_init(port)
        else:
            #the simulation sends input data for the model
            pass

        sock_send.sendto(str(random.random()).encode(encoding='ascii'), ("localhost", port))

    sock.close()


def listener_init() -> int:

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Listening Socket successfully created")

    s.bind(('', 0))
    port = s.getsockname()[1]
    print("Listening socket binded to %s" % (port))

    start_new_thread(server_loop, (s,))

    return port


def sender_init(port: int) -> socket:

    sending_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Sending socket successfully created")

    return sending_socket


if __name__ == '__main__':

    #create a model

    #init server
    listener_port = listener_init()

    #training parameters
    nb_simulation_training  = 1
    nb_step_per_simulation  = 500
    experiment_xml_path     = r"C:\GAMA_1.8.2_Windows_with_JDK_01.05.22_15839b2d\headless\samples\communication_python_test.xml"
    headless_script_path    = r"C:\GAMA_1.8.2_Windows_with_JDK_01.05.22_15839b2d\headless\gama-headless.bat"
    headless_dir            = r"C:\GAMA_1.8.2_Windows_with_JDK_01.05.22_15839b2d\headless"
    simulation_out          = r"C:\GAMA_1.8.2_Windows_with_JDK_01.05.22_15839b2d\headless\out"

    #TODO: we must configure the xml file with at least the port number

    sim_file_path = os.path.join(headless_dir, "tmp_sim.xml")
    sim_file = open(sim_file_path, "w+")
    sim_file.write(f"""<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>
    <Experiment_plan>
        <Simulation id=\"2\" sourcePath=\"samples/TestPythonCommunication/models/PythonCommunicationUDP.gaml\" finalStep=\"10\" experiment=\"PythonCommunication\">
            <Parameters>
                <Parameter name=\"port data\" type=\"INT\" value=\"{listener_port}\"/>
            </Parameters>
        </Simulation>
    </Experiment_plan>""")

    # we move the cursor at the beginning of the file
    sim_file.seek(0)

    for sim in range(nb_simulation_training):
        #run a gama simulation
        #the simulation will interact with the model through the server
        os.system(f"cd {headless_dir} && {headless_script_path} {sim_file_path} out")

