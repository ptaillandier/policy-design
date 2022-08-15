import json

import websockets
import asyncio

from typing import List, Dict


class GamaServerHandler:

	socket = None
	url: str
	port: int

	def __init__(self, url: str, port: int):
		self.url 	= url
		self.port 	= port

	async def send_message(self, message: str):
		await self.socket.send(message)

	async def read_message(self) -> str:
		return await self.socket.recv()

	async def connect(self) -> str:
		try:
			self.socket = await websockets.connect(f"ws://{self.url}:{self.port}")
			return await self.socket.recv()
		except Exception as e:
			print(e)
			return ""

	def open(self) -> bool:
		return self.socket is not None and self.socket.open()

	def create_gama_command(self, command_type: str, gaml_file_path: str="", experiment_name: str = "",
									socket_id: str="", exp_id: str = "", end_condition: str = "", params=None):
		return {
			"type": command_type,
			"model": gaml_file_path,
			"experiment": experiment_name,
			"socket_id": socket_id,
			"exp_id": exp_id,
			"auto-export": False,
			"parameters": params,
			"until": end_condition,
		}

	async def send_command(self, command_type: str, gaml_file_path: str = "", experiment_name: str = "",
									socket_id: str = "", exp_id: str = "", end_condition: str = "", params=None):
		cmd = self.create_gama_command(command_type, gaml_file_path, experiment_name, socket_id, exp_id, end_condition, params)
		cmd_to_str = json.dumps(cmd, indent=0)
		print("sending", cmd_to_str, self.socket)
		await self.socket.send(cmd_to_str)

	async def send_command_return(self, command_type: str, gaml_file_path: str = "", experiment_name: str = "",
									socket_id: str = "", exp_id: str = "", end_condition: str = "", params: List[Dict] = None, unpack_json=False):
		await self.send_command(command_type, gaml_file_path, experiment_name, socket_id, exp_id, end_condition, params)
		res = await self.socket.recv()
		print("received back", res)
		if unpack_json:
			res = json.loads(res)
		return res

	# Load the experiment "experiment_name" in the file "gaml_file_path" through the socket "socket_id" and returns the experiment's id
	async def init_experiment(self, gaml_file_path: str, experiment_name: str, socket_id: str, params: List[Dict] = [{}], callback = None) -> str:
		res = await self.send_command_return("launch",
											 gaml_file_path=gaml_file_path,
											 experiment_name=experiment_name,
											 socket_id=socket_id,
											 params=params,
											 unpack_json=True,
											 )
		if callback is not None:
			callback(res)

		return res["exp_id"]

	async def play(self, socket_id: str, experiment_id: str, callback=None) -> bool:
		res = await self.send_command_return("play", socket_id=socket_id, exp_id=experiment_id)
		return res == "play"

	async def pause(self, socket_id: str, experiment_id: str, callback=None) -> bool:
		res = await self.send_command_return("pause", socket_id=socket_id, exp_id=experiment_id)
		return res == "pause"

	async def step(self, socket_id: str, experiment_id: str, callback=None) -> bool:
		res = await self.send_command_return("step", socket_id=socket_id, exp_id=experiment_id)
		return res == "step"

	async def reload(self, socket_id: str, experiment_id: str, params: list = [], callback=None) -> bool:
		res = await self.send_command_return("reload", socket_id=socket_id, exp_id=experiment_id, params=params)
		return res == "reload"

	async def stop(self, socket_id: str, experiment_id: str, callback=None) -> bool:
		res = await self.send_command_return("stop", socket_id=socket_id, exp_id=experiment_id)
		return res == "stop"

async def tests():
	h = GamaServerHandler("localhost", 6868)
	socket_id = await h.connect()
	print(socket_id)
	exp_id = await h.init_experiment(	r"/home/baptiste/Documents/GitHub/new-policy-design/Diffusion Innovation - Reinforcement learning/models/TCP_model_env_rc2.gaml",
										"one_simulation",
										socket_id,
										 [{"type": "int", "name" : "port", "value": 1234}])
	await h.play(socket_id, exp_id)
	#print(await h.send_command_return("nonsense", socket_id, exp_id))
	await h.pause(socket_id, exp_id)
	await h.step(socket_id, exp_id)
	await h.step(socket_id, exp_id)
	await h.reload(socket_id, exp_id) # equivalent to go to init and then play
	await h.stop(socket_id, exp_id) #raises exceptions in java but seems to work #TODO: investigate, it seems to be linked to the "do die" bug



if __name__ == "__main__":

	asyncio.run(tests())





