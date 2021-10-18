import numpy as np
import jax.numpy as jnp
from torch_baselines.common.utils import get_flatten_size, visual_embedding
from torch_baselines.common.layer import NoisyLinear
from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_STD_SCALE = (LOG_STD_MAX - LOG_STD_MIN)/2.0
LOG_STD_MEAN = (LOG_STD_MAX + LOG_STD_MIN)/2.0

class Actor(nn.Module):
    def __init__(self,state_size,action_size,node=256,hidden_n=1,noisy=False,cnn_mode="normal"):
        super(Actor, self).__init__()
        self.noisy = noisy
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            visual_embedding(st,cnn_mode)
            if len(st) == 3 else nn.Identity()
            for st in state_size 
        ])
        
        flatten_size = np.sum(
                       np.asarray(
                        [
                        get_flatten_size(pr,st)
                        for pr,st in zip(self.preprocess,state_size)
                        ]
                        ))
        
        self.linear = nn.Sequential(
            *([
            lin(flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else lin(node,node) for i in range(2*(hidden_n-1))
            ]
            )
        )
        self.act_mu  = lin(node, action_size[0])
        self.log_std = lin(node, action_size[0])

    def forward(self, xs):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        lin = self.linear(cated)
        mu = self.act_mu(lin)
        log_std = torch.tanh(self.log_std(lin))*LOG_STD_SCALE + LOG_STD_MEAN
        return mu, log_std
        
    def action(self, xs):
        mu, log_std = self(xs)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        pi = torch.tanh(x_t)
        return pi
    
    def sample_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m,NoisyLinear):
                m.sample_noise()
                
    def update_data(self, xs):
        mu, log_std = self(xs)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        pi = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - pi.pow(2) + EPS)
        log_prob = log_prob.sum(1, keepdim=True)
        return pi, log_prob, mu, log_std, std
    

class Critic(nn.Module):
    def __init__(self,state_size,action_size,node=256,hidden_n=1,noisy=False,cnn_mode="normal"):
        super(Critic, self).__init__()
        self.noisy = noisy
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            visual_embedding(st,cnn_mode)
            if len(st) == 3 else nn.Identity()
            for st in state_size 
        ])
        
        flatten_size = np.sum(
                       np.asarray(
                        [
                        get_flatten_size(pr,st)
                        for pr,st in zip(self.preprocess,state_size)
                        ]
                        )) + action_size[0]
        
        self.q1 = nn.Sequential(
            *([
            lin(flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else lin(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            lin(node, 1)
            ]
            )
        )
        
        self.q2 = nn.Sequential(
            *([
            lin(flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else lin(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            lin(node, 1)
            ]
            )
        )

    def forward(self, xs,action):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)] + [action]
        cated = torch.cat(flats,dim=-1)
        q1 = self.q1(cated)
        q2 = self.q2(cated)
        return q1, q2
        
    def sample_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m,NoisyLinear):
                m.sample_noise()
                
class Value(nn.Module):
    def __init__(self,state_size,node=256,hidden_n=1,noisy=False,cnn_mode="normal"):
        super(Value, self).__init__()
        self.noisy = noisy
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            visual_embedding(st,cnn_mode)
            if len(st) == 3 else nn.Identity()
            for st in state_size 
        ])
        
        flatten_size = np.sum(
                       np.asarray(
                        [
                        get_flatten_size(pr,st)
                        for pr,st in zip(self.preprocess,state_size)
                        ]
                        ))
        
        self.linear = nn.Sequential(
            *([
            lin(flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else lin(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            lin(node, 1)
            ]
            )
        )

    def forward(self, xs):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        v = self.linear(cated)
        return v
        
    def sample_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m,NoisyLinear):
                m.sample_noise()