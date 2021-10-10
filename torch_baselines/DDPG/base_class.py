import gym
import torch
import numpy as np
import jax.numpy as jnp
import torch.multiprocessing as mp

from tqdm.auto import trange
from collections import deque

from torch_baselines.common.base_classes import TensorboardWriter
from torch_baselines.common.cpprb_buffers import ReplayBuffer, PrioritizedReplayBuffer
from torch_baselines.common.utils import convert_states

from mlagents_envs.environment import UnityEnvironment, ActionTuple

class Deterministic_Policy_Gradient_Family(object):
    def __init__(self, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, train_freq=1, gradient_steps=1, batch_size=32,
                 n_step = 1, learning_starts=1000, target_network_tau=0.99, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-6, 
                 param_noise=False, max_grad_norm = 1.0, log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None):
        
        self.env = env
        self.log_interval = log_interval
        self.policy_kwargs = policy_kwargs
        self.seed = seed
        
        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_tau = target_network_tau
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._gamma = self.gamma**n_step #n_step gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.n_step_method = (n_step > 1)
        self.n_step = n_step
        self.max_grad_norm = max_grad_norm
        self.train_process = None
        
        self.get_device_setup()
        self.get_env_setup()
        self.get_memory_setup()
        
    def get_device_setup(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        #torch.cuda.synchronize()
        print("----------------------device---------------------")
        print(self.device)
        print("-------------------------------------------------")
        
        
    def get_env_setup(self):
        print("----------------------env------------------------")
        if isinstance(self.env,UnityEnvironment):
            print("unity-ml agent environmet")
            self.env.reset()
            group_name = list(self.env.behavior_specs.keys())[0]
            group_spec = self.env.behavior_specs[group_name]
            self.env.step()
            dec, term = self.env.get_steps(group_name)
            self.group_name = group_name
            
            self.observation_space = [list(spec.shape) for spec in group_spec.observation_specs]
            self.observation_space = [[sp[2], sp[0], sp[1]] if len(sp) == 3 else sp for sp in self.observation_space]
            self.action_size = [group_spec.action_spec.continuous_size]
            self.worker_size = len(dec.agent_id)
            self.env_type = "unity"
            
        elif isinstance(self.env,gym.Env) or isinstance(self.env,gym.Wrapper):
            print("openai gym environmet")
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.observation_space = [list(observation_space.shape)]
            self.action_size = [action_space.shape[0]]
            self.worker_size = 1
            self.env_type = "gym"
        
        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")
        
    def get_memory_setup(self):
        buffer_obs = [[sp[1], sp[2], sp[0]] if len(sp) == 3 else sp for sp in self.observation_space]
        if not self.prioritized_replay:
            self.replay_buffer = ReplayBuffer(self.buffer_size,buffer_obs,action_space=self.action_size[0])
        else:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size,buffer_obs,self.prioritized_replay_alpha,action_space=self.action_size[0])
    
    def setup_model(self):
        pass
    
    def _train_step(self, steps):
        pass
    
    def actions(self,obs,befor_train):
        pass
        
    def learn(self, total_timesteps, callback=None, log_interval=1000, tb_log_name="Q_network",
              reset_num_timesteps=True, replay_wrapper=None):
        pbar = trange(total_timesteps, miniters=log_interval)
        with TensorboardWriter(self.tensorboard_log, tb_log_name) as self.summary:
            if self.env_type == "unity":
                self.learn_unity(pbar, callback, log_interval)
            if self.env_type == "gym":
                self.learn_gym(pbar, callback, log_interval)
    
    def learn_unity(self, pbar, callback=None, log_interval=100):
        self.env.reset()
        self.env.step()
        dec, term = self.env.get_steps(self.group_name)
        self.scores = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        befor_train = True
        for steps in pbar:
            len_dec = len(dec)
            if len_dec:
                actions = self.actions(dec.obs,befor_train)
                action_tuple = ActionTuple(continuous=actions)
                self.env.set_actions(self.group_name, action_tuple)
                old_dec = dec
            self.env.step()
            old_term_id = term.agent_id
            dec, term = self.env.get_steps(self.group_name)
            for id in term.agent_id:
                obs = old_dec[id].obs
                nxtobs = term[id].obs
                reward = term[id].reward
                done = not term[id].interrupted
                terminal = True
                act = actions[id]
                self.replay_buffer.add(obs, act, reward, nxtobs, done, id, terminal)
                self.scores[id] += reward
                self.scoreque.append(self.scores[id])
                if self.summary:
                    self.summary.add_scalar("episode_reward", self.scores[id], steps)
                self.scores[id] = 0
            for id in dec.agent_id:
                if (id in old_term_id and not len_dec) or id in term.agent_id:
                    continue #if in old_term_id -> start dec but not in old_dec_id -> second dec, if in term.agent_id -> start dec
                obs = old_dec[id].obs
                nxtobs = dec[id].obs
                reward = dec[id].reward
                done = False
                terminal = False
                act = actions[id]
                self.replay_buffer.add(obs, act, reward, nxtobs, done, id, terminal)
                self.scores[id] += reward

            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description("score : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque),np.mean(self.lossque)
                                    )
                                    )
            
            if steps > self.learning_starts and steps % self.train_freq == 0:
                befor_train = False
                if self.train_process is not None:
                    for p in self.train_process:
                        p.join()
                        p.close()
                self.train_process = []
                for i in np.arange(self.gradient_steps):
                    p = mp.Process(target=self._train_step, args=(steps,))
                    p.start()
                    self.train_process.append(p)
                    
                
        
    def learn_gym(self, pbar, callback=None, log_interval=100):
        state = self.env.reset()
        self.scores = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        befor_train = True
        for steps in pbar:
            actions = self.actions([state],befor_train)
            next_state, reward, terminal, info = self.env.step(actions[0])
            done = terminal
            if "TimeLimit.truncated" in info:
                done = not info["TimeLimit.truncated"]
            self.replay_buffer.add([state], actions, reward, [next_state], done, 0, terminal)
            self.scores[0] += reward
            state = next_state
            if terminal:
                self.scoreque.append(self.scores[0])
                if self.summary:
                    self.summary.add_scalar("episode_reward", self.scores[0], steps)
                self.scores[0] = 0
                state = self.env.reset()
                
            if steps > self.learning_starts and steps % self.train_freq == 0:
                befor_train = False
                if self.train_process is not None:
                    for p in self.train_process:
                        p.join()
                self.train_process = []
                for i in np.arange(self.gradient_steps):
                    p = mp.Process(target=self._train_step, args=(steps,))
                    p.start()
                    self.train_process.append(p)
            
            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description("score : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque),np.mean(self.lossque)
                                    )
                                    )