import numpy as np
import jax.numpy as jnp
from numpy.core.fromnumeric import shape
from torch_baselines.common.utils import get_flatten_size
from torch_baselines.common.layer import NoisyLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,state_size,action_size,node=256,noisy=False,dualing=False,ModelOptions=None,n_support = 64):
        super(Model, self).__init__()
        self.dualing = dualing
        self.noisy = noisy
        self.n_support = n_support
        self.action_size = action_size
        self.node = node
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(st[0],32,kernel_size=3,stride=1,padding=1,padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=3,stride=1),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
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
        
        self.state_embedding = nn.Sequential(
            lin(flatten_size,node),
            nn.ReLU()
        )
        
        self.register_buffer('pi_mtx', torch.from_numpy(np.expand_dims(np.pi * np.arange(0, 128), axis=0))) # for non updating constant values
        self.quantile_embedding = nn.Sequential(
            lin(128,node),
            nn.ReLU()
        )
        
        if not self.dualing:
            self.q_linear = nn.Sequential(
                lin(node,node),
                nn.ReLU(),
                lin(node, action_size[0])
            )
        else:
            self.advatage_linear = nn.Sequential(
                lin(node,node),
                nn.ReLU(),
                lin(node, action_size[0])
            )
            
            self.value_linear = nn.Sequential(
                lin(node,node),
                nn.ReLU(),
                lin(node, 1)
            )
        

    def forward(self, xs, quantile):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        
        state_embed = self.state_embedding(cated).unsqueeze(2).repeat_interleave(self.n_support, dim=2).view(-1,self.node)
        costau = quantile.view(-1,1)*self.pi_mtx
        quantile_embed = self.quantile_embedding(costau)
        
        mul_embed = torch.multiply(state_embed,quantile_embed)
        
        if not self.dualing:
            q = self.q_linear(mul_embed).view(-1,self.action_size[0],self.n_support)
        else:
            a = self.advatage_linear(mul_embed).view(-1,self.action_size[0],self.n_support)
            v = self.value_linear(mul_embed).view(-1,1,self.n_support)
            q = v + a - a.mean(1, keepdim=True)
        return q
    
    def get_action(self,xs, quantile):
        with torch.no_grad():
            return self(xs,quantile).mean(2).max(1)[1].view(-1,1).detach().cpu().clone()
        
    def sample_noise(self):
        if not self.noisy:
            return
        for m in self.modules():
            if isinstance(m,NoisyLinear):
                m.sample_noise()
                
class QuantileFunction(nn.Module):
    def __init__(self,state_size,node=256,noisy=False,ModelOptions=None,n_support = 64):
        super(QuantileFunction, self).__init__()
        self.noisy = noisy
        self.n_support = n_support
        self.node = node
        if noisy:
            lin = NoisyLinear
        else:
            lin = nn.Linear
        self.preprocess = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(st[0],32,kernel_size=3,stride=1,padding=1,padding_mode='replicate'),
                nn.ReLU(),
                nn.Conv2d(32,64,kernel_size=3,stride=1),
                nn.ReLU(),
                nn.Conv2d(64,64,kernel_size=3,stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
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
                lin(node,node),
                nn.ReLU(),
                lin(node, n_support),
                nn.Softmax(1)
            )
        
    
    def forward(self, xs):
        flats = [pre(x) for pre,x in zip(self.preprocess,xs)]
        cated = torch.cat(flats,dim=-1)
        softmax = self.linear(cated)
        quantile = torch.cat([torch.zeros([cated.shape(0),1]),torch.cumsum(softmax,1)],1)
        quantile = (quantile[:][1:] + quantile[:][:-1])/2.0
        return quantile
        
    
'''
    with tf.variable_scope(scope, reuse=reuse):
        if self.feature_extraction == "cnn":
            critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
        else:
            critics_h = tf.layers.flatten(obs)

        # Concatenate preprocessed state and action
        qi_h = tf.concat([critics_h, action], axis=-1)

        qi_h = mlp(qi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
        #qi_h = tf.layers.dense(qi_h, n_support, name="qi")
        #logqi = tf.nn.log_softmax(qi_h,axis=1)
        #qi = tf.exp(logqi)
        qi = tf.layers.dense(qi_h, n_support, tf.nn.softmax, name="qi")
        logqi = tf.log(qi)
        tau = tf.math.cumsum(qi,axis=1)
        tau = tf.concat([tf.zeros([tf.shape(tau)[0],1]),tau],axis=-1)
        tau_hats = tf.stop_gradient((tau[:, :-1] + tau[:, 1:]) / 2.0)
        entropies = -tf.reduce_sum(logqi * qi,axis=-1,keepdims=True)
    return qi, tau, tau_hats, entropies
'''