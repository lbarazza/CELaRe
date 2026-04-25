# Leonardo Barazza, acse-lb1223

import gymnasium as gym
import numpy as np
import torch

# Helper function for creating discrete action environments. Returns a thunk that creates the environment.
def make_env(env_id, idx, capture_video, run_name, rgb_array=False):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        elif rgb_array:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        return env

    return thunk

# Helper function for creating continuous action environments. Returns a thunk that creates the environment.
def make_env_continuous(env_id, idx, capture_video, run_name, gamma, rgb_array=False):
    def thunk():

        # handle special continuous LunarLander-v2 case
        env_args = {}
        if env_id == "LunarLander-v2":
            env_args = {'continuous': True,}

        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_args)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        elif rgb_array:
            env = gym.make(env_id, render_mode="rgb_array", **env_args)
        else:
            env = gym.make(env_id, **env_args)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

# Helper function to save checkpoint throughout training
def save_checkpoint(agent, run_name, global_step, verbose=False):
    # Construct the paths for saving
    checkpoint_path = f"experiments/weights/{run_name}__step-{global_step}_checkpoint.pth"

    # Create a dictionary to hold the model state and obs_rms
    checkpoint = {
        'model_state_dict': agent.state_dict(),
    }

    # Save the dictionary to a single file
    torch.save(checkpoint, checkpoint_path)
    
    if verbose:
        print(f"Checkpoint saved to {checkpoint_path}")

# Helper function to load checkpoint
def load_checkpoint(checkpoint_path):
    # Load the checkpoint from the file
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']

    return state_dict