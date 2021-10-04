import torch
import numpy as np

from torch_baselines.DQN.base_class import Q_Network_Family
from torch_baselines.QRDQN.network import Model
from torch_baselines.common.losses import QRHuberLosses
from torch_baselines.common.utils import convert_states, hard_update

class QRDQN(Q_Network_Family):
    def __init__(self, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.3, n_support = 200,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True,
                 dualing_model = False, n_step = 1, learning_starts=1000, target_network_update_freq=2000, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-6, 
                 param_noise=False, munchausen=False, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None):
        
        super(QRDQN, self).__init__(env, gamma, learning_rate, buffer_size, exploration_fraction,
                 exploration_final_eps, exploration_initial_eps, train_freq, batch_size, double_q,
                 dualing_model, n_step, learning_starts, target_network_update_freq, prioritized_replay,
                 prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps, 
                 param_noise, munchausen, verbose, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed)
        
        self.n_support = n_support
        
        if _init_setup_model:
            self.setup_model()
            
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
        self.target_model.train()
        self.target_model.to(self.device)
        hard_update(self.target_model,self.model)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.loss = QRHuberLosses(support_size=self.n_support)
        self.quantile = torch.arange(0.5 / self.n_support,1, 1 / self.n_support).to(self.device).view(1,1,self.n_support)
        
        print("----------------------model----------------------")
        print(self.model)
        print("-------------------------------------------------")
    
    def _train_step(self, steps):
        # Sample a batch from the replay buffer
        if self.prioritized_replay:
            data = self.replay_buffer.sample(self.batch_size,self.prioritized_replay_beta0)
        else:
            data = self.replay_buffer.sample(self.batch_size)
        obses = convert_states(data[0],self.device)
        actions = torch.tensor(data[1],dtype=torch.int64,device=self.device).view(-1,1)
        rewards = torch.tensor(data[2],dtype=torch.float32,device=self.device).view(-1,1)
        nxtobses = convert_states(data[3],self.device)
        dones = (~torch.tensor(data[4],dtype=torch.bool,device=self.device)).float().view(-1,1,1)
        self.model.sample_noise()
        self.target_model.sample_noise()
        vals = self.model(obses).gather(1,actions.view(-1,1,1).repeat_interleave(self.n_support, dim=2))
        with torch.no_grad():
            next_q = self.target_model(nxtobses)
            next_mean_q = next_q.mean(2)
            if self.double_q:
                next_actions = self.model(nxtobses).mean(2).max(1)[1].view(-1,1,1).repeat_interleave(self.n_support, dim=2)
            else:
                next_actions = next_mean_q.max(1)[1].view(-1,1,1).repeat_interleave(self.n_support, dim=2)
            if self.munchausen:
                logsum = torch.logsumexp((next_mean_q - next_mean_q.max(1)[0].unsqueeze(-1))/self.munchausen_entropy_tau , 1).unsqueeze(-1)
                tau_log_pi_next = (next_mean_q - next_mean_q.max(1)[0].unsqueeze(-1) - self.munchausen_entropy_tau*logsum).unsqueeze(-1)
                pi_target = torch.nn.functional.softmax(next_mean_q/self.munchausen_entropy_tau, dim=1).unsqueeze(-1)
                next_vals = (pi_target*dones*(next_q.gather(1,next_actions) - tau_log_pi_next)).sum(1)
                
                q_k_targets = self.target_model(obses).mean(2)
                v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
                logsum = torch.logsumexp((q_k_targets - v_k_target)/self.munchausen_entropy_tau, 1).unsqueeze(-1)
                log_pi = q_k_targets - v_k_target - self.munchausen_entropy_tau*logsum
                munchausen_addon = log_pi.gather(1, actions)
                
                rewards += self.munchausen_alpha*torch.clamp(munchausen_addon, min=-1, max=0)
            else:
                next_vals = (dones*next_q.gather(1,next_actions)).squeeze()
            targets = (next_vals * self._gamma) + rewards
        
        logit_valid_tile = targets.view(-1,self.n_support,1).repeat_interleave(self.n_support, dim=2)
        theta_loss_tile = vals.view(-1,1,self.n_support).repeat_interleave(self.n_support, dim=1)
        
        if self.prioritized_replay:
            weights = torch.from_numpy(data[5]).to(self.device)
            indexs = data[6]
            losses = self.loss(theta_loss_tile,logit_valid_tile,self.quantile)
            new_priorities = losses.detach().cpu().clone().numpy() + self.prioritized_replay_eps
            self.replay_buffer.update_priorities(indexs,new_priorities)
            loss = losses.mean(-1)
            loss = (weights*losses).mean(-1)
        else:
            loss = self.loss(theta_loss_tile,logit_valid_tile,self.quantile).mean(-1)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if steps % self.target_network_update_freq == 0:
            hard_update(self.target_model,self.model)
        
        if self.summary:
            self.summary.add_scalar("loss/qloss", loss, steps)

        return loss.detach().cpu().clone().numpy()
    
    def learn(self, total_timesteps, callback=None, log_interval=1000, tb_log_name="QRDQN",
              reset_num_timesteps=True, replay_wrapper=None):
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)