import numpy as np
import jax.numpy as jnp
from torch.nn.modules.activation import Tanh
from torch_baselines.common.utils import get_flatten_size, visual_embedding
from torch_baselines.common.layer import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,preprocess,action_size,node=256,hidden_n=1,rnn='None'):
        super(Actor, self).__init__()
        
        self.preprocess = preprocess
        
        self.linear = nn.Sequential(
            *([
            nn.Linear(self.preprocess.flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else nn.Linear(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            nn.Linear(node, action_size[0]),
            nn.Tanh()
            ]
            )
        )

    def forward(self, xs):
        feature = self.preprocess(xs)
        return self.linear(feature)
                
class Critic(nn.Module):
    def __init__(self,preprocess,node=256,hidden_n=1,rnn='None'):
        super(Critic, self).__init__()
        
        self.preprocess = preprocess
    
        self.linear = nn.Sequential(
            *([
            nn.Linear(self.preprocess.flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else nn.Linear(node,node) for i in range(2*(hidden_n-1))
            ] + 
            [
            nn.Linear(node,1)
            ]
            )
        )

    def forward(self, xs):
        feature = self.preprocess(xs)
        return self.linear(feature)
    
class AC(nn.Module):
    def __init__(self,preprocess,action_size,node=256,hidden_n=1):
        super(Critic, self).__init__()
        
        self.preprocess = preprocess
        
        self.linear = nn.Sequential(
            *([
            nn.Linear(self.preprocess.flatten_size,node),
            nn.ReLU()] + 
            [
            nn.ReLU() if i%2 else nn.Linear(node,node) for i in range(2*(hidden_n-1))
            ]
            )
        )
        
        self.actor = nn.Sequential(
            nn.Linear(node,action_size),
            nn.Softmax(1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(node,1)
        )

    def forward(self, xs):
        feature = self.preprocess(xs)
        h = self.linear(feature)
        a = self.actor(h)
        v = self.critic(h)
        return a, v