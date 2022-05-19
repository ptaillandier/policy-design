# Import the RL algorithm (Trainer) we would like to use.
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from simplegymenv.envs.simplegymenv import SimpleGymEnv

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "SimpleGymEnv-v0",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [32],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
}

# register the custom environment in ray
env = 'SimpleGymEnv-v0'
register_env(env, lambda config: SimpleGymEnv())
# Create our RLlib Trainer.
trainer = PPOTrainer(config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(3):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print('result')
    print(result)    

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
trainer.evaluate()
