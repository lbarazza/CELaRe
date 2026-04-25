# Leonardo Barazza, acse-lb1223

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F
from .model import layer_init, ClustLinear

######################## CONTINUOUS ACTION AGENT ########################

class AgentContinuous(nn.Module):
    def __init__(self, envs, n_codebook_vectors=3, latent_dim=2, detach=False, alpha=1.0):
        super().__init__()
        
        self.n_codebook_vectors = n_codebook_vectors
        self.latent_dim = latent_dim
        self.detach = detach
        self.alpha = alpha
        obs_shape = np.array(envs.single_observation_space.shape).prod()

        # shared network
        self.fc1 = layer_init(nn.Linear(obs_shape, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.fc3 = layer_init(nn.Linear(64, 64))
        self.clust1 = ClustLinear(dim=64, codebook_size=n_codebook_vectors, latent_dim=latent_dim, detach=detach, alpha=alpha)
        self.fc4 = layer_init(nn.Linear(64, 64))

        # actor and critic heads
        self.critic_out = layer_init(nn.Linear(64, 1), std=1.0)
        self.actor_out = layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
    
    # return the hidden states for visualization
    def get_hidden_states(self, x):
        z = F.mish(self.fc1(x))
        z = F.mish(self.fc2(z))
        z = F.mish(self.fc3(z))
        z, quant_loss1, recons_loss1, entropy_loss1, indices1, probs1 = self.clust1(z)
        hidden1 = z
        hidden_states = (hidden1,)
        return hidden_states

    # run the actor and critic
    def actor_and_value_forward(self, x):

        z = F.mish(self.fc1(x))
        z = F.mish(self.fc2(z))
        z = F.mish(self.fc3(z))

        # apply the CELaRe layer
        z, quant_loss1, recons_loss1, entropy_loss1, indices1, probs1 = self.clust1(z)

        z = F.mish(self.fc4(z))

        action_mean = self.actor_out(z)
        value = self.critic_out(z)

        action_logstd = self.actor_logstd.expand_as(action_mean)

        quant_loss = quant_loss1
        recons_loss = recons_loss1
        entropy_loss = entropy_loss1

        indices = (indices1,)
        probs = (probs1,)

        return action_mean, action_logstd, value, quant_loss, recons_loss, entropy_loss, indices, probs

    # get the value of the state from the critic
    def get_value(self, x):
        action_mean, action_logstd, value, quant_loss, \
            recons_loss, entropy_loss, indices, clust_probs = self.actor_and_value_forward(x)
        return value

    # get the action and value from the actor and critic
    def get_action_and_value(self, x, action=None):
        action_mean, action_logstd, value, quant_loss, \
            recons_loss, entropy_loss, indices, clust_probs = self.actor_and_value_forward(x)
        action_std = torch.exp(action_logstd)

        # sample the action from the normal distribution constructed from the network output
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value,\
            quant_loss, recons_loss, entropy_loss, indices, clust_probs


######################## DISCRETE ACTION AGENT ########################

class AgentDiscrete(nn.Module):
    def __init__(self, envs, n_codebook_vectors=3, latent_dim=2, detach=False, alpha=1.0):
        super().__init__()

        self.n_codebook_vectors = n_codebook_vectors
        self.latent_dim = latent_dim
        self.detach = detach
        self.alpha = alpha
        obs_shape = np.array(envs.single_observation_space.shape).prod()

        # shared network
        self.fc1 = layer_init(nn.Linear(obs_shape, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.fc3 = layer_init(nn.Linear(64, 64))
        self.clust1 = ClustLinear(dim=64, codebook_size=n_codebook_vectors, latent_dim=latent_dim, \
                                  detach=detach, alpha=alpha)
        self.fc4 = layer_init(nn.Linear(64, 64))

        # actor and critic heads
        self.critic_out = layer_init(nn.Linear(64, 1), std=1.0)
        self.actor_out = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
    
    # return the hidden states for visualization
    def get_hidden_states(self, x):
        z = F.relu(self.fc1(x))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z, quant_loss1, recons_loss1, entropy_loss1, indices1, probs1 = self.clust1(z)
        hidden1 = z
        hidden_states = (hidden1,)
        return hidden_states

    # run the actor and critic
    def actor_and_value_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x, quant_loss, recons_loss, entropy_loss, indices, probs = self.clust1(x)
        x = F.relu(self.fc4(x))
        logits = self.actor_out(x)
        value = self.critic_out(x)

        indices = (indices,)
        probs = (probs,) 

        return logits, value, quant_loss, recons_loss, entropy_loss, indices, probs
    
    # get the value of the state from the critic
    def get_value(self, x):
        _, value, _, _, _, _, _ = self.actor_and_value_forward(x)
        return value

    # get the action and value from the actor and critic
    def get_action_and_value(self, x, action=None):

        logits, value, quant_loss, recons_loss, entropy_loss, indices, clust_probs = self.actor_and_value_forward(x)

        # sample the action from the categorical distribution constructed from the network output
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value, quant_loss, recons_loss, entropy_loss, indices, clust_probs
