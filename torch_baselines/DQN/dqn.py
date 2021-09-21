from torch_baselines.DQN.network import Model
from torch_baselines.common.buffers import ReplayBuffer
from torch_baselines.common.schedules import LinearSchedule
import numpy as np
from mlagents_envs.environment import UnityEnvironment,ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import gym
from collections import deque
import torch


class DQN:
    def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False, verbose=0, tensorboard_log=None, 
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None):
        
        self.policy = policy
        self.env = env
        self.verbose = verbose
        self.policy_kwargs = policy_kwargs
        self.seed = seed
        
        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.double_q = double_q

        self.graph = None
        self.sess = None
        self.step_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None
        
        self.get_env_setup()
        
        if _init_setup_model:
            self.setup_model()
        
    def get_env_setup(self):
        if isinstance(self.env,UnityEnvironment):
            self.env.reset()
            group_name = list(self.env.behavior_specs.keys())[0]
            group_spec = self.env.behavior_specs[group_name]
            self.env.step()
            dec, term = self.env.get_steps(group_name)
            self.group_name = group_name
            
            self.observation_space = [list(spec.shape) for spec in group_spec.observation_specs]
            self.observation_space = [[sp[2], sp[0], sp[1]] if len(sp) == 3 else sp for sp in self.observation_space]
            self.action_size = [branch for branch in group_spec.action_spec.discrete_branches]
            self.worker_size = len(dec.agent_id)
            self.env_type = "unity"
            
        elif isinstance(self.env,gym.Env):
            from gym import spaces
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.observation_space = [list(observation_space.shape)]
            self.action_size = [action_space.n]
            self.worker_size = 1
            self.env_type = "gym"
            
        if self.prioritized_replay:
            pass
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)

            
            
    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        self.model = Model(self.observation_space,self.action_size,**self.policy_kwargs)
        self.model.eval()
        self.target_model = Model(self.observation_space,self.action_size,**self.policy_kwargs)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        
        
    def _train_step(self, step, learning_rate):
        # Sample a batch from the replay buffer
        data = self.replay_buffer.sample(self.batch_size)
        obses = [torch.from_numpy(o).float() for o in data[0]]
        obses = [o.permute(0,3,1,2) if len(o.shape) == 4 else o for o in obses]
        actions = torch.from_numpy(data[1])
        rewards = torch.from_numpy(data[2])
        nxtobses = [torch.from_numpy(o).float() for o in data[3]]
        nxtobses = [no.permute(0,3,1,2) if len(no.shape) == 4 else no for no in nxtobses]
        dones = (~torch.from_numpy(data[4])).float()
        vals = self.model(obses).gather(-1,actions.view(-1,1))
        '''
        print(vals.shape)
        print(actions.shape)
        vals = vals
        '''
        with torch.no_grad():
            next_vals = dones*torch.max(self.target_model(nxtobses),dim=-1)[0]
            targets = rewards + self.gamma*next_vals
        loss = torch.mean(torch.square(targets - vals))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if step % self.target_network_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss.detach()

    
    def actions(self,obs,epsilon):
        if epsilon > np.random.uniform():
            actions = np.random.choice(self.action_size[0], [self.worker_size])
        else:
            self.model.eval()
            obs = [torch.from_numpy(o).float() for o in obs]
            obs = [o.permute(0,3,1,2) if len(o.shape) == 4 else o for o in obs]
            actions = np.expand_dims(self.model.get_action(obs).numpy(), axis=-1)
        return actions
        #pass
        
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):
        if self.env_type == "unity":
            self.learn_unity(total_timesteps, callback, log_interval, tb_log_name,
              reset_num_timesteps, replay_wrapper)
        if self.env_type == "gym":
            self.learn_gym(total_timesteps, callback, log_interval, tb_log_name,
              reset_num_timesteps, replay_wrapper)
    
    def learn_unity(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):
        self.env.reset()
        self.env.step()
        dec, term = self.env.get_steps(self.group_name)
        self.scores = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        
        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                            initial_p=self.exploration_initial_eps,
                                            final_p=self.exploration_final_eps)
        for steps in range(total_timesteps):
            update_eps = self.exploration.value(steps)
            actions = self.actions(dec.obs,update_eps)
            
            action_tuple = ActionTuple(discrete=actions)
            self.env.set_actions(self.group_name, action_tuple)
            
            if len(dec.agent_id) > 0:
                old_dec = dec
                old_action = actions
                
            self.env.step()
            dec, term = self.env.get_steps(self.group_name)
            for idx in term.agent_id:
                obs = old_dec[idx].obs
                nxtobs = term[idx].obs
                reward = term[idx].reward
                done = not term[idx].interrupted
                act = old_action[idx]
                self.replay_buffer.add(obs, act, reward, nxtobs, done)
                self.scores[idx] += reward
                self.scoreque.append(self.scores[idx])
                self.scores[idx] = 0
            for idx in dec.agent_id:
                if idx in term.agent_id:
                    continue
                obs = old_dec[idx].obs
                nxtobs = dec[idx].obs
                reward = dec[idx].reward
                done = False
                act = old_action[idx]
                self.replay_buffer.add(obs, act, reward, nxtobs, done)
                self.scores[idx] += reward
            if steps % 1000 == 0 and len(self.scoreque) > 0:
                print(np.mean(self.scoreque))
            
            can_sample = self.replay_buffer.can_sample(self.batch_size)
            if can_sample and steps > self.learning_starts/self.worker_size and steps % self.train_freq == 0:
                loss = self._train_step(steps,self.learning_rate)
                
            
        
    def learn_gym(self, total_timesteps, callback=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):
        
        
        state = self.env.reset()
        self.scores = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.scoreque.append(0)
        self.lossque = deque(maxlen=10)
        self.lossque.append(0)
        
        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                            initial_p=self.exploration_initial_eps,
                                            final_p=self.exploration_final_eps)
        for steps in range(total_timesteps):
            update_eps = self.exploration.value(steps)
            actions = self.actions([state],update_eps)
            next_state, reward, done, info = self.env.step(actions[0])
            self.replay_buffer.add([state], actions[0], reward, [next_state], done)
            self.scores[0] += reward
            if done:
                self.scoreque.append(self.scores[0])
                self.env.reset()
            state = next_state
            can_sample = self.replay_buffer.can_sample(self.batch_size)

            if can_sample and steps > self.learning_starts/self.worker_size and steps % self.train_freq == 0:
                loss = self._train_step(steps,self.learning_rate)
                self.lossque.append(loss)
            
            if steps % 1000 == 0 and len(self.scoreque) > 0:
                print("score : ", np.mean(self.scoreque), ", epsion :", update_eps, ", loss : ", np.mean(self.scoreque))