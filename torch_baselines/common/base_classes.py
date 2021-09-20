import os
import glob
import json
import zipfile
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Union, List, Callable, Optional

import gym
import cloudpickle
import numpy as np
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRLModel(ABC):
    def __init__(self, policy, env, verbose=0, *, requires_vec_env, policy_base,
                 policy_kwargs=None, seed=None, n_cpu_tf_sess=None):
        if isinstance(policy, str) and policy_base is not None:
            self.policy = get_policy_from_name(policy_base, policy)
        else:
            self.policy = policy
        self.env = env
        self.verbose = verbose
        self._requires_vec_env = requires_vec_env
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self._vectorize_action = False
        self.num_timesteps = 0
        self.graph = None
        self.sess = None
        self.params = None
        self.seed = seed
        self._param_load_ops = None
        self.n_cpu_tf_sess = n_cpu_tf_sess
        self.episode_reward = None
        self.ep_info_buf = None