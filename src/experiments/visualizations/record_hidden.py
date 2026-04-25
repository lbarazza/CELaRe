# Leonardo Barazza, acse-lb1223

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from experiments.utils import make_env_continuous, make_env, load_checkpoint
import tyro
from experiments.config.config import ArgsContinuous, ArgsDiscrete
from celare import AgentContinuous, AgentDiscrete

# Determine whether to use continuous or discrete
is_continuous = None
if len(sys.argv) > 1:
    first_arg = sys.argv[1]
    if first_arg == "--d":
        is_continuous = False
    elif first_arg == "--c":
        is_continuous = True
    else:
        raise ValueError("Invalid argument. Please provide 'd' for discrete or 'c' for continuous.")

# Load the correct arguments and agent classes based on the mode
args = tyro.cli((ArgsContinuous if is_continuous else ArgsDiscrete))

path = args.weights_path

NUM_CODEBOOK_VECTORS = args.num_codebook_vectors

# Initialize the environment
if is_continuous:
    env = make_env_continuous(env_id=args.env_id,
                              idx=0,
                              capture_video=False,
                              run_name="data_collection",
                              gamma=args.gamma,
                              rgb_array=True)
    agent_class = AgentContinuous
else:
    env = make_env(env_id=args.env_id,
                   idx=0,
                   capture_video=False,
                   run_name="data_collection",
                   rgb_array=True)
    agent_class = AgentDiscrete

envs = gym.vector.SyncVectorEnv([env])  # dummy vectorized env used for loading the agent
env = env()  # evaluate thunk to get the environment

# Initialize the agent
agent = agent_class(envs, n_codebook_vectors=NUM_CODEBOOK_VECTORS)
checkpoint = load_checkpoint(path)
agent.load_state_dict(checkpoint)
agent.eval()

# Initialize the environment and get the initial observation
obs = env.reset()[0]

# Initialize the DataFrame for hidden states
hidden_state_dim = 64  # Dimension of the hidden state vector
distance_dim = NUM_CODEBOOK_VECTORS  # Dimension of the distance vector

columns = ['trajectory_id', 'time_step']
for i in range(1):
    for j in range(hidden_state_dim):
        columns.append(f'hidden-{i}_{j}')
    columns.append(f'hidden-{i}_codebook-id')
    for j in range(distance_dim):
        columns.append(f'distance-{i}_{j}')

df = pd.DataFrame(columns=columns)

done = False
trajectory_id = 0
time_step = 0

def collect_step():
    global obs, done, df, trajectory_id, time_step

    obs = torch.Tensor(obs).unsqueeze(0)

    # Get hidden states from the agent
    hidden_states_raw = agent.get_hidden_states(obs)
    hidden_states = [hidden_state.detach().numpy().flatten() for hidden_state in hidden_states_raw]

    # Get action, indices, and distances from the agent
    action, _, _, _, _, _, _, indices_raw, distances_raw = agent.get_action_and_value(obs)
    action = action.detach().numpy().squeeze()
    obs, _, done, _, _ = env.step(action)

    indices = [index.flatten().numpy() for index in indices_raw]
    distances = [distance.detach().numpy().flatten() for distance in distances_raw]

    # construct the row for the DataFrame
    row = np.concatenate((np.array(trajectory_id).flatten(), np.array(time_step).flatten(), hidden_states[0], indices[0], distances[0]))
    df.loc[len(df)] = row

    time_step += 1

    if done:
        time_step = 0
        trajectory_id += 1
        obs = env.reset()[0]

N = args.collection_steps  # Number of hidden states to collect

for step in range(N):
    collect_step()
    if step % 1000 == 0:
        print(f"Collected {step} steps")

# Save the DataFrame to a file
df.to_csv('experiments/visualizations/data/trajectories.csv', index=False)
print("Saved hidden states to visualizations/data/trajectories.csv")

env.close()
