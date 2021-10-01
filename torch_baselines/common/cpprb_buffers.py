import random
from typing import Optional, List, Union

import numpy as np
import cpprb

class ReplayBuffer(object):
    
    def __init__(self, size: int, observation_space: list):
        self.buffer_size = size
        self.obsdict = dict(("obs{}".format(idx),{"shape": o}) for idx,o in enumerate(observation_space))
        self.nextobsdict = dict(("nextobs{}".format(idx),{"shape": o}) for idx,o in enumerate(observation_space))
        self.buffer = cpprb.ReplayBuffer(size,
                    env_dict={**self.obsdict,
                        "action": {"shape": 1},
                        "reward": {},
                        **self.nextobsdict,
                        "done": {}
                    })

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def storage(self):
        return self.buffer

    @property
    def buffer_size(self) -> int:
        return self.buffer_size

    def can_sample(self, n_samples: int) -> bool:
        return len(self) >= n_samples

    def is_full(self) -> int:
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, nxtobs_t, done):
        obsdict = zip(self.obsdict.keys,obs_t)
        nextobsdict = zip(self.nextobsdict.keys,nxtobs_t)
        self.buffer.add(**obsdict,action=action,reward=reward,**nextobsdict,done=done)

    def sample(self, batch_size: int):
        smpl = self.buffer.sample(batch_size)
        obses_t = [smpl[o] for o in self.obsdict.keys()]
        actions = smpl['action']
        rewards = smpl['reward']
        nxtobses_t = [smpl[no] for no in self.nextobsdict.keys()]
        dones = smpl['done']
        return (obses_t,
                actions,
                rewards,
                nxtobses_t,
                dones)
        
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size: int, observation_space: list, alpha: float):
        self.buffer_size = size
        self.obsdict = dict(("obs{}".format(idx),{"shape": o}) for idx,o in enumerate(observation_space))
        self.nextobsdict = dict(("nextobs{}".format(idx),{"shape": o}) for idx,o in enumerate(observation_space))
        self.buffer = cpprb.PrioritizedReplayBuffer(size,
                    env_dict={**self.obsdict,
                        "action": {"shape": 1},
                        "reward": {},
                        **self.nextobsdict,
                        "done": {}
                    },
                    alpha=0.5)

    def sample(self, batch_size: int):
        smpl = self.buffer.sample(batch_size)
        obses_t = [smpl[o] for o in self.obsdict.keys()]
        actions = smpl['action']
        rewards = smpl['reward']
        nxtobses_t = [smpl[no] for no in self.nextobsdict.keys()]
        dones = smpl['done']
        weights = smpl['weights']
        indexes = smpl['indexes']
        return (obses_t,
                actions,
                rewards,
                nxtobses_t,
                dones,
                weights,
                indexes)
        
    def update_priorities(self, indexes, priorities):
        self.buffer.update_priorities(indexes,priorities)