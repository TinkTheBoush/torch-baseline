import torch
import numpy as np

from torch_baselines.DDPG.base_class import Deterministic_Policy_Gradient_Family
from torch_baselines.TD4_IQN.network import Actor, Critic, Qunatile_Maker
from torch_baselines.common.losses import QRHuberLosses
from torch_baselines.common.utils import convert_tensor, hard_update, soft_update, minimum_quantile

class TD4_IQN(Deterministic_Policy_Gradient_Family):
    def __init__(self, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, n_support = 8, target_action_noise_mul = 1.5,
                 action_noise = 0.1, train_freq=1, gradient_steps=1, batch_size=32, policy_delay = 2, risk_avoidance = 0.0,
                 n_step = 1, learning_starts=1000, target_network_tau=0.99, prioritized_replay=False, 
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-6, 
                 param_noise=False, max_grad_norm = 1.0, log_interval=200, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None):
        
        super(TD4_IQN, self).__init__(env, gamma, learning_rate, buffer_size, train_freq, gradient_steps, batch_size, 
                 n_step, learning_starts, target_network_tau, prioritized_replay,
                 prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps, 
                 param_noise, max_grad_norm, log_interval, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed)
        
        self.n_support = n_support
        self.action_noise = action_noise
        target_action_noise_mul = target_action_noise_mul
        self.target_action_noise = action_noise * target_action_noise_mul       #0.2
        self.action_noise_clamp = 0.5 #self.target_action_noise*1.5
        self.risk_avoidance = risk_avoidance
        self.policy_delay = policy_delay
        self.sample_risk_avoidance = False
        self.quantile_minimum_mode = 'sort_split'
        
        if _init_setup_model:
            self.setup_model()
            
            
    def setup_model(self):
        self.policy_kwargs = {} if self.policy_kwargs is None else self.policy_kwargs
        self.actor = Actor(self.observation_space,self.action_size,
                           noisy=self.param_noise, **self.policy_kwargs)
        self.critic = Critic(self.observation_space,self.action_size,
                           noisy=self.param_noise, **self.policy_kwargs)
        self.target_actor = Actor(self.observation_space,self.action_size,
                           noisy=self.param_noise, **self.policy_kwargs)
        self.target_critic = Critic(self.observation_space,self.action_size,
                           noisy=self.param_noise,  **self.policy_kwargs)
        self.actor.train()
        self.actor.to(self.device)
        self.critic.train()
        self.critic.to(self.device)
        self.target_actor.train()
        self.target_actor.to(self.device)
        self.target_critic.train()
        self.target_critic.to(self.device)
        self.actor_param = list(self.actor.parameters())
        self.main_param = list(self.actor.parameters()) + list(self.critic.parameters())
        self.target_param = list(self.target_actor.parameters()) + list(self.target_critic.parameters())
        hard_update(self.target_param,self.main_param)
        
        #self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(),lr=self.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.learning_rate)
        self.critic_loss = QRHuberLosses()
        self.quantile = Qunatile_Maker(self.n_support)
        self.quantile.to(self.device)
        '''
        
        '''
        if self.risk_avoidance == 'auto':
            pass
        elif self.risk_avoidance == 'normal':
            self.sample_risk_avoidance = True
        else:
            self.risk_avoidance = float(self.risk_avoidance)
            #self.grad_mul = (self.quantile.view(1,self.n_support) < 0.1).float()/0.1
            #self.grad_mul = 1.0 - self.risk_avoidance*(2.0*self.quantile.view(1,self.n_support) - 1.0)
        
        print("----------------------model----------------------")
        print(self.actor)
        print(self.critic)
        print(self.critic_loss)
        print("-------------------------------------------------")
    
    def _train_step(self, steps, grad_step):
        # Sample a batch from the replay buffer
        if self.prioritized_replay:
            data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)
        else:
            data = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            obses = convert_tensor(data[0],self.device)
            actions = torch.as_tensor(data[1],dtype=torch.float32,device=self.device)
            rewards = torch.as_tensor(data[2],dtype=torch.float32,device=self.device).view(-1,1)
            nxtobses = convert_tensor(data[3],self.device)
            invdones = (~torch.as_tensor(data[4],dtype=torch.bool,device=self.device)).float().view(-1,1)
            next_actions = self.target_actor(nxtobses)
            next_actions = torch.clamp(next_actions + 
                                       torch.clamp(
                                           torch.normal(0,self.target_action_noise,size=(self.batch_size, self.action_size[0]),device=self.device),
                                            -self.action_noise_clamp,self.action_noise_clamp),
                                       -1,1)
            quantile_target = self.quantile(self.batch_size)
            next_vals1, next_vals2 = self.target_critic(nxtobses,next_actions,quantile_target)
            next_vals = invdones * minimum_quantile(next_vals1,next_vals2,self.quantile_minimum_mode)
            targets = (self._gamma * next_vals) + rewards
            logit_valid_tile = targets.unsqueeze(1).repeat_interleave(self.n_support, dim=1).detach()
        
        quantile_main = self.quantile(self.batch_size)
        vals1, vals2 = self.critic(obses,actions.detach(),quantile_main.detach())
        theta1_loss_tile = vals1.unsqueeze(2).repeat_interleave(targets.shape[1], dim=2)
        theta2_loss_tile = vals2.unsqueeze(2).repeat_interleave(targets.shape[1], dim=2)
        
        if self.prioritized_replay:
            weights = torch.from_numpy(data[5]).to(self.device)
            indexs = data[6]
            critic_losses1 = self.critic_loss(theta1_loss_tile,logit_valid_tile,quantile_main.unsqueeze(2))
            critic_losses2 = self.critic_loss(theta2_loss_tile,logit_valid_tile,quantile_main.unsqueeze(2))
            new_priorities = critic_losses1.cpu().clone().numpy() + \
                            critic_losses2.cpu().clone().numpy() + self.prioritized_replay_eps
            self.replay_buffer.update_priorities(indexs,new_priorities)
            critic_loss1 = (weights*critic_losses1).mean()
            critic_loss2 = (weights*critic_losses2).mean()
        else:
            critic_loss1 = self.critic_loss(theta1_loss_tile,logit_valid_tile,quantile_main.unsqueeze(2)).mean()
            critic_loss2 = self.critic_loss(theta2_loss_tile,logit_valid_tile,quantile_main.unsqueeze(2)).mean()
        critic_loss = critic_loss1 + critic_loss2
        self.lossque.append(critic_loss.detach().cpu().clone().numpy())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        step = (steps + grad_step)
        if step % self.policy_delay == 0:
            quantile_policy = self.quantile(self.batch_size)*0.1
            q1,_ = self.critic(obses,self.actor(obses),quantile_policy)
            actor_loss = -(q1).mean()
            #grad_mul = 1.0 - self.risk_avoidance*(2.0*quantile_policy - 1.0)
            #actor_loss = -(q1*grad_mul.detach()).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actor_param, self.max_grad_norm)
            self.actor_optimizer.step()
        
            soft_update(self.target_param,self.main_param,self.target_network_tau)
            
            if self.summary and step % self.log_interval == 0:
                self.summary.add_scalar("loss/actor_loss", actor_loss, steps)
        
        if self.summary and step % self.log_interval == 0:
            self.summary.add_scalar("loss/critic_loss", critic_loss, steps)
            self.summary.add_scalar("loss/targets", targets.mean(), steps)
            
    def actions(self,obs,befor_train):
        if not befor_train:
            with torch.no_grad():
                actions = np.clip(self.actor(convert_tensor(obs,self.device)).detach().cpu().clone().numpy() + 
                                np.random.normal(0,self.action_noise,size=(self.worker_size,self.action_size[0]))
                                ,-1,1)
        else:
            actions = np.random.uniform(-1,1,size=(self.worker_size,self.action_size[0]))
        return actions
    
    def learn(self, total_timesteps, callback=None, log_interval=1000, tb_log_name="TD4_IQN",
              reset_num_timesteps=True, replay_wrapper=None):
        if self.sample_risk_avoidance:
            tb_log_name = tb_log_name + "_{}".format(self.risk_avoidance)
        else:
            tb_log_name = tb_log_name + "_{:.2f}".format(self.risk_avoidance)
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)