# Leonardo Barazza, acse-lb1223

import os
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ArgsDiscrete:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "discrete_experiment"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model in the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    latent_dim: int = 2
    """the latent dimension of the VQ-VAE"""

    # clustering
    agent_mode: int = 2
    """the mode of the agent. 0 is the baseline, 1 uses the constant coefficient, mode 2 uses the oscillating coefficient"""
    oscill_freq: float = 1.0/100.0
    """the frequency of the oscillation"""
    oscill_mu: float = 0.1
    """the mean of the oscillation"""
    oscill_amp: float = 0.05
    """the amplitude of the oscillation"""
    oscill_n: int = 7
    """the number of discrete values in the oscillation"""

    # to be filled at runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    detach: bool = False
    """detach gradient from main network"""
    alpha: float = 1.0
    """the alpha value for the extra reconstruction loss term"""
    clust_entropy_coef: float = 1.0
    """the entropy coefficient for the clustering loss"""

    # checkpoints and logging
    print_interval: int = 5
    """interval between training status logs in the terminal (in terms of main loop steps)"""
    checkpoint_interval: int = 75_000
    """interval between saving model weights (in terms of global env interaction steps)"""

    # for later visualization
    d: bool = False
    """keeps using discrete arguments if turned on"""
    c: bool = False
    """switches to continuous arguments if turned on"""
    weights_path: str = None
    """the path to the weights file"""
    collection_steps: int = 10_000
    """the number of steps to collect data for the hidden layer visualization"""
    num_codebook_vectors: int = 3
    """the number of codebook vectors for the clustering"""

# defines default environment arguments
def env_args():
    return {
        # 'continuous': True, # uncomment for continuous LunarLander-v2 as default
    }

@dataclass
class ArgsContinuous:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 6#2
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "continuous_experiment"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model in the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "LunarLander-v2"
    """the id of the environment"""
    env_args: Dict = field(default_factory=env_args)
    """the arguments for the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    latent_dim: int = 2
    """the latent dimension of the VQ-VAE"""

    # clustering
    agent_mode: int = 2
    """the mode of the agent. 0 is the baseline, 1 uses the constant coefficient, mode 2 uses the oscillating coefficient"""
    oscill_freq: float = 1.0/640.0
    """the frequency of the oscillation"""
    oscill_mu: float = 0.1
    """the mean of the oscillation"""
    oscill_amp: float = 0.05
    """the amplitude of the oscillation"""
    oscill_n: int = 7
    """the number of discrete values in the oscillation"""

    # to be filled at runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    detach: bool = False
    """detach gradient from main network"""
    alpha: float = 1.0
    """the alpha value for the extra reconstruction loss term"""
    clust_entropy_coef: float = 0.1
    """the entropy coefficient for the clustering loss"""

    # checkpoints and logging
    print_interval: int = 5
    """interval between training status logs in the terminal (in terms of main loop steps)"""
    checkpoint_interval: int = 75_000
    """interval between saving model weights (in terms of global env interaction steps)"""

    # for later visualization
    d: bool = False
    """keeps using discrete arguments if turned on"""
    c: bool = False
    """switches to continuous arguments if turned on"""
    weights_path: str = None
    """the path to the weights file"""
    collection_steps: int = 10_000
    """the number of steps to collect data for the hidden layer visualization"""
    num_codebook_vectors: int = 3
    """the number of codebook vectors for the clustering"""
