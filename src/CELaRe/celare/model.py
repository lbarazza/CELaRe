# Leonardo Barazza, acse-lb1223

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

######################## UTILITIES #######################

# Initialize the weights of a layer
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

######################## VQ-VAE ########################

# Encoder for the VQ-VAE
class Encoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, latent_dim=16):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.mish(self.fc1(x))
        x = self.fc2(x)
        return x

# Vector Quantizer for the VQ-VAE
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    # Calculate the squared distances between input features and embeddings
    def calculate_distances(self, x_flat):
        distances = (torch.sum(x_flat ** 2, dim=2, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(x_flat, self.embedding.weight.t()))
        return distances

    # Find the indices of the closest embeddings
    def find_closest_indices(self, x):
        # Flatten input and calculate distances
        # the -1 in view introduces an extra dimension in the case
        # that x is only 1D (excluding batch size). This allows to treat it
        # as a normal 3D input which is actually of shape 1 x N. This allows
        # to use the original VQ-VAE code with no modifications. (and is also
        # ready to be used with higher-dimensional inputs in the future).
        # The original code expects an input of shape B x C x N x M. In our case,
        # C corresponds to the latent dimensions and N and M are both 1. So it's
        # like we are acting on only one 'pixel'
        x_flat = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        distances = self.calculate_distances(x_flat)

        # Find the indices of the closest embeddings
        closest_indices = torch.argmin(distances, dim=2)
        return closest_indices, distances

    # Retrieve quantized embeddings given indices
    def get_quantized_embeddings(self, indices):
        quantized = self.embedding(indices).permute(0, 2, 1)
        return quantized

    # The forward pass now only finds and returns the closest_indices
    def forward(self, x):
        closest_indices, distances = self.find_closest_indices(x)
        return closest_indices, distances
    
    # Return the codebook vectors
    def get_codebook_vectors(self):
        return self.embedding.weight

# Decoder for the VQ-VAE
class Decoder(nn.Module):
    def __init__(self, output_dim=64, hidden_dim=64, latent_dim=16):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.mish(self.fc1(x))
        x = self.fc2(x)
        return x
    
# VQ-VAE model
class VQVAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, codebook_size=3, latent_dim=8, commitment_cost=0.25, lam=0.0, temp=2.0):
        super(VQVAE, self).__init__()

        # entropy parameters
        self.lam = lam
        self.temp = temp

        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=latent_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(output_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

    def forward(self, x):

        encoded = self.encoder(x)
        closest_indices, distances = self.quantizer(encoded)
        
        # The view_as is necessary to reshape the quantized tensor to the same shape as the encoded tensor.
        # This is necessary in the 1D input case because the quantizer returns a 3D tensor (excluding batch size)
        # while the encoded tensor is 1D (excluding batch size) and I want the output to be 1D as well.
        quantized = self.quantizer.get_quantized_embeddings(closest_indices).view_as(encoded)

        # Calculate Entropy Regularization
        probabilities = F.softmax(-distances / self.temp, dim=2)  # Negate distances for softmax
        codebook_usage = torch.mean(probabilities, dim=0)  # Average over batch
        entropy_loss = -torch.sum(codebook_usage * torch.log(codebook_usage + 1e-10))  # Add small value to avoid log(0)

        # Straight-Through Estimator
        # We first calculate the quantization loss using the detached quantized embeddings
        e_latent_loss = F.mse_loss(quantized.detach(), encoded)
        q_latent_loss = F.mse_loss(quantized, encoded.detach())
        quantization_loss = q_latent_loss + self.quantizer.commitment_cost * e_latent_loss \
                            + self.lam * entropy_loss

        # Then, apply the straight-through estimator by using quantized values for forward pass
        # and the gradient of encoded for the backward pass
        quantized = encoded + (quantized - encoded).detach()

        decoded = self.decoder(quantized)

        return decoded, quantization_loss, entropy_loss, closest_indices, probabilities


######################## CLUSTERING LAYER ########################

# CELaRe layer
class ClustLinear(nn.Module):
    def __init__(self, dim=64, codebook_size=3, latent_dim=8, detach=False, alpha=1.0, commitment_cost=1.0):
        super(ClustLinear, self).__init__()

        self.layernorm = nn.LayerNorm(dim)
        self.vqvae = VQVAE(input_dim=dim, codebook_size=codebook_size, latent_dim=latent_dim,
                           commitment_cost=commitment_cost)
        self.detach = detach
        self.alpha = alpha
    
    def forward(self, x):
        x = self.layernorm(x)

        # Detach the input if necessary (for the baseline)
        if self.detach:
            decoded, quant_loss, entropy_loss, indices, probs = self.vqvae(x.detach())
        else:
            decoded, quant_loss, entropy_loss, indices, probs = self.vqvae(x)

        # add second term in recons_loss when not using the baseline
        recons_loss = 1.0 * F.mse_loss(decoded, x.detach())
        if not self.detach:
            recons_loss = recons_loss + self.alpha * F.mse_loss(decoded.detach(), x) # symmetric term

        return x, quant_loss, recons_loss, entropy_loss, indices, probs


######################## OSCILLATING COEFFICIENT ########################

# Oscillating coefficient
def oscill_coef(update_step, freq=0.0, mean=0.1, amp=0.0, n=None):

    # Determine the range of vae_coef
    min_value = mean - amp
    max_value = mean + amp

    coef = mean - amp * np.cos(2 * np.pi * freq * update_step)

    if n is not None:
        # Create an array of n possible discrete values
        discrete_values = np.linspace(min_value, max_value, n)
        coef = discrete_values[np.abs(discrete_values - coef).argmin()]

    return coef
