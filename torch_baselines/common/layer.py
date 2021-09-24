import numpy as np
import jax.numpy as jnp
from torch_baselines.common.utils import get_flatten_size
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        # Uniform Distribution bounds: 
        #     U(-1/sqrt(p), 1/sqrt(p))
        self.lowerU          = -1.0 / np.sqrt(in_features) # 
        self.upperU          =  1.0 / np.sqrt(in_features) # 
        self.sigma_0         = std_init
        self.sigma_ij_in     = self.sigma_0 / np.sqrt(self.in_features)
        self.sigma_ij_out    = self.sigma_0 / np.sqrt(self.out_features)

        """
        Registre_Buffer: Adds a persistent buffer to the module.
            A buffer that is not to be considered as a model parameter -- like "running_mean" in BatchNorm
            It is a "persistent state" and can be accessed as attributes --> self.weight_epsilon
        """
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        self.weight_mu.data.uniform_(self.lowerU, self.upperU)
        self.weight_sigma.data.fill_(self.sigma_ij_in)

        self.bias_mu.data.uniform_(self.lowerU, self.upperU)
        self.bias_sigma.data.fill_(self.sigma_ij_out)

    def sample_noise(self):
        eps_in  = self.func_f(self.in_features)
        eps_out = self.func_f(self.out_features)
        # Take the outter product 
        """
            >>> v1 = torch.arange(1., 5.) [1, 2, 3, 4]
            >>> v2 = torch.arange(1., 4.) [1, 2, 3]
            >>> torch.ger(v1, v2)
            tensor([[  1.,   2.,   3.],
                    [  2.,   4.,   6.],
                    [  3.,   6.,   9.],
                    [  4.,   8.,  12.]])
        """
        eps_ij = eps_out.ger(eps_in)
        self.weight_epsilon.copy_(eps_ij)
        self.bias_epsilon.copy_(eps_out)

    def func_f(self, n): # size
        # sign(x) * sqrt(|x|) as in paper
        x = torch.rand(n)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma*self.weight_epsilon, 
                               self.bias_mu   + self.bias_sigma  *self.bias_epsilon)

        else:
            return F.linear(x, self.weight_mu,
                               self.bias_mu)