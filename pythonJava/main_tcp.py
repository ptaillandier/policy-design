import os
import socket
from _thread import *


def server_loop(sock_listen: socket) -> None:

    port: int = -1

    while True:
        conn, addr = sock_listen.accept()
        print("connected", conn, addr)
        while True:
            print("waiting for data")
            data = conn.recv(1024)
            print("python received", data)
            print("python sending answer")
            conn.send("pong\n".encode())
            conn.send("pang\n".encode())
            conn.send("peng\n".encode())
            conn.send("on teste des trucs FIN pour le prochain".encode())
        sock.close()



def listener_init() -> int:

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Listening Socket successfully created")

    s.bind(('', 0))
    port = s.getsockname()[1]
    print("Listening socket binded to %s" % (port))

    s.listen()
    print("Listening socket started listening")

    start_new_thread(server_loop, (s,))

    return port


# def sender_init(port: int) -> socket:
#
#     sending_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     print("Sending socket successfully created")
#
#     return sending_socket



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

    sim_file_path = os.path.join(headless_dir, "tmp_sim.xml")
    sim_file = open(sim_file_path, "w+")
    sim_file.write(f"""<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>
    <Experiment_plan>
        <Simulation id=\"2\" sourcePath=\"samples/TestPythonCommunication/models/PythonCommunicationTCP.gaml\" finalStep=\"10\" experiment=\"PythonCommunication\">
            <Parameters>
                <Parameter name=\"port\" type=\"INT\" value=\"{listener_port}\"/>
            </Parameters>
        </Simulation>
    </Experiment_plan>""")

    # we move the cursor at the beginning of the file
    sim_file.seek(0)

    for sim in range(nb_simulation_training):
        #run a gama simulation
        #the simulation will interact with the model through the server
        #os.system(f"cd {headless_dir} && {headless_script_path} {sim_file_path} out")
        input()




