# Leonardo Barazza, acse-lb1223

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from experiments.utils import make_env_continuous, make_env, load_checkpoint
import tyro
from experiments.config.config import ArgsContinuous, ArgsDiscrete
from celare import AgentContinuous, AgentDiscrete
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

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
                              run_name="visualization",
                              gamma=args.gamma,
                              rgb_array=True)
    agent_class = AgentContinuous
else:
    env = make_env(env_id=args.env_id,
                   idx=0,
                   capture_video=False,
                   run_name="visualization",
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

# Get the number of sets of indices and probabilities
with torch.no_grad():
    action, _, _, _, _, _, _, indices, probs = agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0))
num_sets = len(indices)

# Create the main plot
fig, ax = plt.subplots(figsize=(8, 8))

# Initial environment image plot setup
img = env.render()
im = ax.imshow(img)
ax.axis('off')

# Create the inset axes for the histogram in the top left corner
inset_ax = fig.add_axes([0.15, 0.6, 0.15, 0.15])

# Initialize the histograms in the inset axes
bars_list = []
tab10_colors = plt.cm.tab10(np.arange(10))

probs_set = np.zeros(NUM_CODEBOOK_VECTORS)
colors = tab10_colors[:probs_set.shape[0]]
bars = inset_ax.bar(range(probs_set.shape[0]), probs_set, color=colors)
bars_list.append(bars)

inset_ax.axis('off')
inset_ax.xaxis.set_visible(True)
inset_ax.yaxis.set_ticks([])

# Store the text objects for each bar
text_objects = []
for j, bar in enumerate(bars):
    text = inset_ax.text(
        bar.get_x() + bar.get_width() / 2,
        -0.05,
        f'{j}',
        ha='center',
        va='top',
        fontsize=10,
        color='white'
    )
    text_objects.append(text)

inset_ax.set_ylim(0, 1)
inset_ax.set_xticks(range(probs_set.shape[0]))

done = False

def update(frame):
    global obs, done

    if done:
        return []

    obs = torch.Tensor(obs).unsqueeze(0)

    # Get action, indices, and probabilities from the agent
    action, _, _, _, _, _, _, indices, probs = agent.get_action_and_value(obs)

    # Adjust temperature for better visualizations
    if is_continuous:
        probs = (F.softmax(probs[0] / 0.05, dim=2),)
    else:
        probs = (F.softmax(probs[0] / 0.015, dim=2),)

    action = action.detach().numpy().squeeze()
    obs, _, done, _, _ = env.step(action)

    # Update environment image
    img = env.render()
    im.set_array(img)

    # Update histogram
    probs_np = probs[0].detach().numpy().flatten()
    for j, bar in enumerate(bars_list[0]):
        bar.set_height(probs_np[j] if j < len(probs_np) else 0)
        bar.set_alpha(1.0 if j == indices[0].item() else 0.3)

    return [im] + [bar for bars in bars_list for bar in bars] + text_objects

# Create and save the animation
ani = FuncAnimation(fig, update, frames=range(1000), repeat=False, blit=True)
ani.save('experiments/visualizations/animation.mp4', writer=animation.FFMpegWriter(fps=30))
env.close()
