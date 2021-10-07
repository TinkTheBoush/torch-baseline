import torch
import numpy as np

from collections import deque

from torch_baselines.DDPG.base_class import Deterministic_Policy_Gradient_Family
from torch_baselines.DDPG.network import Actor, Critic
from torch_baselines.common.losses import MSELosses, HuberLosses
from torch_baselines.common.utils import convert_states, hard_update, soft_update
from torch_baselines.common.schedules import LinearSchedule
from torch_baselines.common.noise import OUNoise

from mlagents_envs.environment import UnityEnvironment, ActionTuple

class DDPG(Deterministic_Policy_Gradient_Family):
    def __init__(self, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.3,
                 exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, gradient_steps=1, batch_size=32,
                 n_step = 1, learning_starts=1000, target_network_tau=0.99, prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_eps=1e-6, 
                 param_noise=False, max_grad_norm = 1.0, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, 
                 full_tensorboard_log=False, seed=None):
        
        super(DDPG, self).__init__(env, gamma, learning_rate, buffer_size, train_freq, gradient_steps, batch_size, 
                 n_step, learning_starts, target_network_tau, prioritized_replay,
                 prioritized_replay_alpha, prioritized_replay_beta0, prioritized_replay_eps, 
                 param_noise, max_grad_norm, verbose, tensorboard_log, _init_setup_model, policy_kwargs, 
                 full_tensorboard_log, seed)

        self.exploration_final_eps = exploration_final_eps
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_fraction = exploration_fraction
        
        self.noise = OUNoise(action_size = self.action_size[0], worker_size= self.worker_size)
        
        if _init_setup_model:
            self.setup_model()
            
    def actions(self,obs,epsilon,befor_train,agent_id):
        if not befor_train:
            with torch.no_grad():
                actions = np.clip(self.actor(convert_states(obs,self.device)).detach().cpu().clone().numpy() + self.noise()[agent_id]*epsilon,-1,1)
        else:
            actions = np.clip(np.random.normal(size=(self.worker_size,self.action_size[0])),-1,1)[agent_id]
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
        
        #self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(),lr=self.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=self.learning_rate)
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
        
        with torch.no_grad():
            obses = convert_states(data[0],self.device)
            actions = torch.tensor(data[1],dtype=torch.float32,device=self.device)
            rewards = torch.tensor(data[2],dtype=torch.float32,device=self.device).view(-1,1)
            nxtobses = convert_states(data[3],self.device)
            dones = (~torch.tensor(data[4],dtype=torch.bool,device=self.device)).float().view(-1,1)
            next_vals = dones * self.target_critic(nxtobses,self.target_actor(nxtobses))
            targets = (next_vals * self._gamma) + rewards
        
        vals = self.critic(obses,actions)
        
        if self.prioritized_replay:
            weights = torch.from_numpy(data[5]).to(self.device)
            indexs = data[6]
            new_priorities = np.abs((targets - vals).squeeze().detach().cpu().clone().numpy()) + self.prioritized_replay_eps
            self.replay_buffer.update_priorities(indexs,new_priorities)
            critic_loss = (weights*self.critic_loss(vals,targets)).mean(-1)
        else:
            critic_loss = self.critic_loss(vals,targets).mean(-1)
         
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(obses,self.actor(obses)).squeeze().mean(-1)
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
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
    
    def learn(self, total_timesteps, callback=None, log_interval=1000, tb_log_name="DDPG",
              reset_num_timesteps=True, replay_wrapper=None):
        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                                initial_p=self.exploration_initial_eps,
                                                final_p=self.exploration_final_eps)
        super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, replay_wrapper)
        
    def learn_unity(self, pbar, callback=None, log_interval=100):
        self.env.reset()
        self.env.step()
        dec, term = self.env.get_steps(self.group_name)
        self.scores = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        befor_train = True
        for steps in pbar:
            update_eps = self.exploration.value(steps)
            actions = self.actions(dec.obs,update_eps,befor_train,dec.agent_id)
            action_tuple = ActionTuple(continuous=actions)
            self.env.set_actions(self.group_name, action_tuple)
            if len(dec.agent_id) > 0:
                old_dec = dec
                old_actions = actions
            self.env.step()
            dec, term = self.env.get_steps(self.group_name)
            
            for id in term.agent_id:
                obs = old_dec[id].obs
                nxtobs = term[id].obs
                reward = term[id].reward
                done = not term[id].interrupted
                terminal = True
                act = old_actions[id]
                self.replay_buffer.add(obs, act, reward, nxtobs, done, id, terminal)
                self.scores[id] += reward
                self.scoreque.append(self.scores[id])
                if self.summary:
                    self.summary.add_scalar("episode_reward", self.scores[id], steps)
                self.scores[id] = 0
            for id in dec.agent_id:
                if id in term.agent_id:
                    continue
                obs = old_dec[id].obs
                nxtobs = dec[id].obs
                reward = dec[id].reward
                done = False
                terminal = False
                act = old_actions[id]
                self.replay_buffer.add(obs, act, reward, nxtobs, done, id, terminal)
                self.scores[id] += reward

            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description("score : {:.3f}, epsilon : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque),update_eps,np.mean(self.lossque)
                                    )
                                    )
            
            self.terminal_callback(term.agent_id)
            
            if steps > self.learning_starts/self.worker_size and steps % self.train_freq == 0:
                befor_train = False
                for i in range(self.gradient_steps):
                    loss = self._train_step(steps)
                    self.lossque.append(loss)
                
        
    def learn_gym(self, pbar, callback=None, log_interval=100):
        state = self.env.reset()
        self.scores = np.zeros([self.worker_size])
        self.scoreque = deque(maxlen=10)
        self.lossque = deque(maxlen=10)
        befor_train = True
        for steps in pbar:
            update_eps = self.exploration.value(steps)
            actions = self.actions([state],update_eps,befor_train,[0])
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
                self.terminal_callback([0])
                
            if steps > self.learning_starts/self.worker_size and steps % self.train_freq == 0:
                befor_train = False
                for i in range(self.gradient_steps):
                    loss = self._train_step(steps)
                    self.lossque.append(loss)
            
            if steps % log_interval == 0 and len(self.scoreque) > 0 and len(self.lossque) > 0:
                pbar.set_description("score : {:.3f}, epsilon : {:.3f}, loss : {:.3f} |".format(
                                    np.mean(self.scoreque),update_eps,np.mean(self.lossque)
                                    )
                                    )