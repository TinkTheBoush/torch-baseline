import gym
import torch
import numpy as np

from tqdm.auto import trange
from collections import deque

from torch_baselines.QRDQN.network import Model
from torch_baselines.common.base_classes import TensorboardWriter
from torch_baselines.common.losses import QRHuberLosses
from torch_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer
from torch_baselines.common.schedules import LinearSchedule

from mlagents_envs.environment import UnityEnvironment, ActionTuple

class QRDQN:
    def __init__(self, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.3, n_support = 64,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True,
                 dualing_model = False, learning_starts=1000, target_network_update_freq=2000, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-6, 
                 param_noise=False, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None):
        
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
        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.double_q = double_q
        self.dualing_model = dualing_model
        self.n_support = n_support

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
        
        self.get_device_setup()
        self.get_env_setup()
        self.get_memory_setup()
        
        if _init_setup_model:
            self.setup_model()
        
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
            self.action_size = [branch for branch in group_spec.action_spec.discrete_branches]
            self.worker_size = len(dec.agent_id)
            self.env_type = "unity"
            
        elif isinstance(self.env,gym.Env) or isinstance(self.env,gym.Wrapper):
            print("openai gym environmet")
            action_space = self.env.action_space
            observation_space = self.env.observation_space
            self.observation_space = [list(observation_space.shape)]
            self.action_size = [action_space.n]
            self.worker_size = 1
            self.env_type = "gym"
        
        print("observation size : ", self.observation_space)
        print("action size : ", self.action_size)
        print("worker_size : ", self.worker_size)
        print("-------------------------------------------------")
        
    def get_memory_setup(self):
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size,self.prioritized_replay_alpha)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)   
            
    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        self.model = Model(self.observation_space,self.action_size,n_support=self.n_support,
                           dualing=self.dualing_model,noisy=self.param_noise,
                           **self.policy_kwargs)
        self.target_model = Model(self.observation_space,self.action_size,n_support=self.n_support,
                                  dualing=self.dualing_model,noisy=self.param_noise,
                                  **self.policy_kwargs)
        self.model.train()
        self.model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.train()
        self.target_model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        '''
        if self.prioritized_replay:
            self.loss = WeightedMSELoss()
        else:
            self.loss = torch.nn.MSELoss()
        '''
        self.loss = QRHuberLosses(support_size=self.n_support)
        '''
        if self.prioritized_replay:
            self.loss = WeightedHuber()
        else:
            self.loss = torch.nn.SmoothL1Loss()
        '''
        self.quantile = torch.range(0.5 / self.n_support,1, 1 / self.n_support).to(self.device)
        
        print("----------------------model----------------------")
        print(self.model)
        print("-------------------------------------------------")
    
    def _train_step(self, steps):
        # Sample a batch from the replay buffer
        if self.prioritized_replay:
            data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)
        else:
            data = self.replay_buffer.sample(self.batch_size)
        obses = [torch.from_numpy(o).to(self.device).float() for o in data[0]]
        obses = [o.permute(0,3,1,2) if len(o.shape) == 4 else o for o in obses]
        actions = torch.from_numpy(data[1]).to(self.device).view(-1,1,1).repeat_interleave(self.n_support, dim=2)
        rewards = torch.from_numpy(data[2]).to(self.device).float().view(-1,1,1)
        nxtobses = [torch.from_numpy(o).to(self.device).float() for o in data[3]]
        nxtobses = [no.permute(0,3,1,2) if len(no.shape) == 4 else no for no in nxtobses]
        dones = (~torch.from_numpy(data[4]).to(self.device)).float().view(-1,1,1)
        vals = self.model(obses).gather(1,actions)
        with torch.no_grad():
            if self.double_q:
                action = self.model(nxtobses).mean(2).max(1)[1].view(-1,1,1).repeat_interleave(self.n_support, dim=2)
            else:
                action = self.target_model(nxtobses).mean(2).max(1)[1].view(-1,1,1).repeat_interleave(self.n_support, dim=2)
            next_vals = dones*self.target_model(nxtobses).gather(1,action)
            targets = (next_vals * self.gamma) + rewards
        
        logit_valid_tile = targets.view(-1,self.n_support,1).repeat_interleave(self.n_support, dim=2)
        theta_loss_tile = vals.view(-1,1,self.n_support).repeat_interleave(self.n_support, dim=1)
        
        if self.prioritized_replay:
            indexs = data[6]
            losses = self.loss(theta_loss_tile,logit_valid_tile,self.quantile)
            new_priorities = np.sqrt(losses.detach().cpu().clone().numpy()) + self.prioritized_replay_eps
            self.replay_buffer.update_priorities(indexs,new_priorities)
            loss = losses.mean(-1)
        else:
            loss = self.loss(theta_loss_tile,logit_valid_tile,self.quantile).mean(-1)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if steps % self.target_network_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        if self.summary:
            self.summary.add_scalar("loss/qloss", loss, steps)

        return loss.detach().cpu().clone().numpy()

    
    def actions(self,obs,epsilon):
        if epsilon <= np.random.uniform(0,1):
            obs = [torch.from_numpy(o).to(self.device).float() for o in obs]
            obs = [o.permute(0,3,1,2) if len(o.shape) == 4 else o for o in obs]
            actions = self.model.get_action(obs).numpy()
        else:
            actions = np.random.choice(self.action_size[0], [self.worker_size,1])
        return actions
        #pass
        
    def learn(self, total_timesteps, callback=None, log_interval=1000, tb_log_name="QRDQN",
              reset_num_timesteps=True, replay_wrapper=None):
        if self.dualing_model:
            tb_log_name = "Dualing_" + tb_log_name
        if self.double_q:
            tb_log_name = "Double_" + tb_log_name
        if self.prioritized_replay:
            tb_log_name = tb_log_name + "+PER"
        
        with TensorboardWriter(self.tensorboard_log, tb_log_name) as self.summary:
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                initial_p=self.exploration_initial_eps,
                                                final_p=self.exploration_final_eps)
            pbar = trange(total_timesteps, miniters=log_interval)
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
        
        for steps in pbar:
            update_eps = self.exploration.value(steps)
            actions = self.actions(dec.obs,update_eps)
            
            action_tuple = ActionTuple(discrete=actions)
            self.env.set_actions(self.group_name, action_tuple)
            old_dec = dec
            self.env.step()
            dec, term = self.env.get_steps(self.group_name)
            
            for idx in term.agent_id:
                obs = old_dec[idx].obs
                nxtobs = term[idx].obs
                reward = term[idx].reward
                done = not term[idx].interrupted
                act = actions[idx]
                self.replay_buffer.add(obs, act, reward, nxtobs, done)
                self.scores[idx] += reward
                self.scoreque.append(self.scores[idx])
                if self.summary:
                    self.summary.add_scalar("episode_reward", self.scores[idx], steps)
                self.scores[idx] = 0
            for idx in dec.agent_id:
                if idx in term.agent_id:
                    continue
                obs = old_dec[idx].obs
                nxtobs = dec[idx].obs
                reward = dec[idx].reward
                done = False
                act = actions[idx]
                self.replay_buffer.add(obs, act, reward, nxtobs, done)
                self.scores[idx] += reward

            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description("score : {:.3f}, epsilon : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque),update_eps,np.mean(self.lossque)
                                    )
                                    )
            
            can_sample = self.replay_buffer.can_sample(self.batch_size)
            if can_sample and steps > self.learning_starts/self.worker_size and steps % self.train_freq == 0:
                loss = self._train_step(steps)
                self.lossque.append(loss)
                
        
    def learn_gym(self, pbar, callback=None, log_interval=100):
        state = self.env.reset()
        self.scores = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        
        for steps in pbar:
            update_eps = self.exploration.value(steps)
            actions = self.actions([state],update_eps)
            next_state, reward, done, info = self.env.step(actions[0][0])
            done_real = done
            if "TimeLimit.truncated" in info:
                done_real = not info["TimeLimit.truncated"]
            self.replay_buffer.add([state], actions[0], reward, [next_state], done_real)
            self.scores[0] += reward
            state = next_state
            if done:
                self.scoreque.append(self.scores[0])
                if self.summary:
                    self.summary.add_scalar("episode_reward", self.scores[0], steps)
                self.scores[0] = 0
                state = self.env.reset()
                
            can_sample = self.replay_buffer.can_sample(self.batch_size)
            if can_sample and steps > self.learning_starts/self.worker_size and steps % self.train_freq == 0:
                loss = self._train_step(steps)
                self.lossque.append(loss)
            
            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description("score : {:.3f}, epsilon : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque),update_eps,np.mean(self.lossque)
                                    )
                                    )