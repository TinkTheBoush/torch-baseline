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
        
        self.get_device_setup()
        self.get_env_setup()
        self.get_memory_setup()
        
    def get_device_setup(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        obses = convert_states(dec.obs)
        for steps in pbar:
            
            actions = self.actions(dec.obs,befor_train)
            action_tuple = ActionTuple(continuous=actions)
            old_obses = obses

            self.env.set_actions(self.group_name, action_tuple)
            self.env.step()
            
            dec, term = self.env.get_steps(self.group_name)
            term_ids = list(term.agent_id)
            term_obses = convert_states(term.obs)
            term_rewards = list(term.reward)
            term_done = list(term.interrupted)
            while len(dec) == 0:
                self.env.step()
                dec, term = self.env.get_steps(self.group_name)
                if len(term.agent_id) > 0:
                    term_ids += list(term.agent_id)
                    term_obses = [np.stack(to,o) for to,o in zip(term_obses,convert_states(term.obs))]
                    term_rewards += list(term.reward)
                    term_done += list(term.interrupted)
            obses = convert_states(dec.obs)
            nxtobs = [np.copy(o) for o in obses]
            done = np.full((self.worker_size),False)
            terminal = np.full((self.worker_size),False)
            term_ids = np.asarray(term_ids)
            reward = dec.reward
            term_on = term_ids.shape[0] > 0
            if term_on:
                term_rewards = np.asarray(term_rewards)
                term_done = np.asarray(term_done)
                for n,t in zip(nxtobs,term_obses):
                    n[term_ids] = t
                done[term_ids] = ~term_done
                terminal[term_ids] = True
                reward[term_ids] = term_rewards
            self.scores += reward
            self.replay_buffer.add(old_obses, actions, reward, nxtobs, done, 0, terminal)
            if term_on:
                if self.summary:
                    for tid in term_ids:
                        self.summary.add_scalar("env/episode_reward", self.scores[tid], steps)
                        self.summary.add_scalar("env/time over",float(not done[tid]),steps)
                self.scoreque.extend(self.scores[term_ids])
                self.scores[term_ids] = 0
            
            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description("score : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque),np.mean(self.lossque)
                                    )
                                    )
            
            if steps > self.learning_starts and steps % self.train_freq == 0:
                befor_train = False
                for i in np.arange(self.gradient_steps):
                    loss = self._train_step(steps,i) 
        
    def learn_gym(self, pbar, callback=None, log_interval=100):
        state = convert_states([self.env.reset()])
        self.scores = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=self.worker_size*10)
        self.lossque = deque(maxlen=self.worker_size*10)
        befor_train = True
        for steps in pbar:
            actions = self.actions(state,befor_train)
            next_state, reward, terminal, info = self.env.step(actions[0])
            next_state = convert_states([next_state])
            done = terminal
            if "TimeLimit.truncated" in info:
                done = not info["TimeLimit.truncated"]
            self.replay_buffer.add(state, actions, reward, next_state, done, 0, terminal)
            self.scores[0] += reward
            state = next_state
            if terminal:
                self.scoreque.append(self.scores[0])
                if self.summary:
                    self.summary.add_scalar("env/episode_reward", self.scores[0], steps)
                    self.summary.add_scalar("env/time over",float(not done),steps)
                self.scores[0] = 0
                state = self.env.reset()
                
            if steps > self.learning_starts and steps % self.train_freq == 0:
                befor_train = False
                for i in np.arange(self.gradient_steps):
                    self._train_step(steps,i)
            
            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description("score : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque),np.mean(self.lossque)
                                    )
                                    )