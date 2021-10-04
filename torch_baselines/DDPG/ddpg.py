import torch
import numpy as np

from torch_baselines.DDPG.base_class import Deterministic_Policy_Gradient_Family
from torch_baselines.DDPG.network import Actor, Critic
from torch_baselines.common.losses import MSELosses, HuberLosses
from torch_baselines.common.utils import convert_states, hard_update, soft_update
from torch_baselines.common.noise import OUNoise

class DDPG(Deterministic_Policy_Gradient_Family):
    def __init__(self, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.3,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32,
                 n_step = 1, learning_starts=1000, target_network_tau=0.99, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-6, 
                 param_noise=False, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None):
        
        super(DDPG, self).__init__(env, gamma, learning_rate, buffer_size, exploration_fraction,
                 exploration_final_eps, exploration_initial_eps, train_freq, batch_size, 
                 n_step, learning_starts, target_network_tau, prioritized_replay,
                 prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps, 
                 param_noise, verbose, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed)
        
        self.noise = OUNoise(0.2, action_size = self.action_size[0], worker_size= self.worker_size)
        
        if _init_setup_model:
            self.setup_model()
            
    def actions(self,obs,epsilon,befor_train):
        if not befor_train:
            actions = np.clip(self.actor(convert_states(obs,self.device)).detach().cpu().clone().numpy() + self.noise(),-1,1)
        else:
            actions = np.clip(np.random.normal(size=(self.worker_size,self.action_size[0])),-1,1)
        return actions
            
    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        self.actor = Actor(self.observation_space,self.action_size,
                           noisy=self.param_noise, **self.policy_kwargs)
        self.critic = Critic(self.observation_space,self.action_size,
                           noisy=self.param_noise, **self.policy_kwargs)
        self.target_actor = Actor(self.observation_space,self.action_size,
                           noisy=self.param_noise, **self.policy_kwargs)
        self.target_critic = Critic(self.observation_space,self.action_size,
                           noisy=self.param_noise, **self.policy_kwargs)
        self.actor.train()
        self.actor.to(self.device)
        self.critic.train()
        self.critic.to(self.device)
        self.target_actor.train()
        self.target_actor.to(self.device)
        self.target_critic.train()
        self.target_critic.to(self.device)
        hard_update(self.target_actor,self.actor)
        hard_update(self.target_critic,self.critic)
        
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(),lr=self.learning_rate)
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(),lr=self.learning_rate)
        self.critic_loss = MSELosses()
        
        print("----------------------model----------------------")
        print(self.actor)
        print(self.critic)
        print(self.critic_loss)
        print("-------------------------------------------------")
    
    def _train_step(self, steps):
        # Sample a batch from the replay buffer
        if self.prioritized_replay:
            data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)
        else:
            data = self.replay_buffer.sample(self.batch_size)
        obses = convert_states(data[0],self.device)
        actions = torch.tensor(data[1],dtype=torch.int64,device=self.device)
        rewards = torch.tensor(data[2],dtype=torch.float32,device=self.device).view(-1,1)
        nxtobses = convert_states(data[3],self.device)
        dones = (~torch.tensor(data[4],dtype=torch.bool,device=self.device)).float().view(-1,1)
        vals = self.critic(obses,actions)
        with torch.no_grad():
            next_actions = self.target_actor(nxtobses)
            next_vals = dones * self.target_critic(nxtobses,next_actions)
            targets = (next_vals * self._gamma) + rewards

        if self.prioritized_replay:
            weights = torch.from_numpy(data[5]).to(self.device)
            indexs = data[6]
            new_priorities = np.abs((targets - vals).squeeze().detach().cpu().clone().numpy()) + self.prioritized_replay_eps
            self.replay_buffer.update_priorities(indexs,new_priorities)
            critic_loss = (weights*self.critic_loss(vals,targets)).mean(-1)
        else:
            critic_loss = self.critic_loss(vals,targets).mean(-1)
         
        self.critic_optimizer.zero_grad()   
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(obses,self.actor(obses)).squeeze().mean(-1)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        soft_update(self.target_actor,self.actor,self.target_network_tau)
        soft_update(self.target_critic,self.critic,self.target_network_tau)
        
        if self.summary:
            self.summary.add_scalar("loss/critic_loss", critic_loss, steps)
            self.summary.add_scalar("loss/actor_loss", actor_loss, steps)
            self.summary.add_scalar("loss/targets", targets.mean(), steps)

        return critic_loss.detach().cpu().clone().numpy()
    
    def terminal_callback(self,workers):
        self.noise.reset(workers)
    
    def learn(self, total_timesteps, callback=None, log_interval=1000, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)