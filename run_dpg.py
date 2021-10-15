import os
import argparse
import gym

from torch_baselines.DDPG.ddpg import DDPG
from torch_baselines.TD3.td3 import TD3
from torch_baselines.TD4_QR.td4_qr import TD4_QR
from torch_baselines.TD4_IQN.td4_iqn import TD4_IQN
from torch_baselines.common.utils import set_random_seed
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="BipedalWalker-v3", help='environment')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--algo', type=str, default="DDPG", help='algo ID')
    parser.add_argument('--worker_id', type=int, default=0, help='verbose')
    parser.add_argument('--gamma', type=float, default=0.995, help='gamma')
    parser.add_argument('--train_freq', type=int, default=1, help='train freq')
    parser.add_argument('--grad_step', type=int, default=1, help='grad step')
    parser.add_argument('--target_update_tau', type=float, default=0.98, help='target update intervals')
    parser.add_argument('--batch', type=int, default=512, help='batch size')
    parser.add_argument('--buffer_size', type=float, default=1e6, help='buffer_size')
    parser.add_argument('--per', action='store_true')
    parser.add_argument('--noisynet', action='store_true')
    parser.add_argument('--n_step', type=int, default=1, help='n step setting when n > 1 is n step td method')
    parser.add_argument('--steps', type=float, default=1e6, help='step size')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--logdir',type=str, default='log/',help='log file dir')
    parser.add_argument('--seed', type=int, default=1234567, help='random seed')
    parser.add_argument('--max_grad', type=float,default=-1.0, help='grad clip max size')
    parser.add_argument('--risk_avoidance', type=str,default='0.0', help='risk_avoidance for TD4')
    parser.add_argument('--n_support', type=int,default=64, help='n_support for TD4')
    parser.add_argument('--node', type=int,default=256, help='network node number')
    parser.add_argument('--hidden_n', type=int,default=2, help='hidden layer number')
    parser.add_argument('--action_noise', type=float,default=0.1, help='action noise std')
    parser.add_argument('--target_noise_mul', type=float,default=2.0, help='target noise mul')
    args = parser.parse_args() 
    env_name = args.env
    env_type = ""
    
    set_random_seed(args.seed)
    if os.path.exists(env_name):
        engine_configuration_channel = EngineConfigurationChannel()
        channel = EnvironmentParametersChannel()
        engine_configuration_channel.set_configuration_parameters(time_scale=12.0,capture_frame_rate=60)
        
        env = UnityEnvironment(file_name=env_name,seed=args.seed,no_graphics=False, worker_id=args.worker_id,
                               side_channels=[engine_configuration_channel,channel],timeout_wait=1000,)
        env_name = env_name.split('/')[-1].split('.')[0]
        env_type = "unity"
    else:
        env = gym.make(env_name,seed=args.seed)
        env_type = "gym"
    policy_kwargs = {'node': args.node,
                     'hidden_n': args.hidden_n}
    
    if args.algo == "DDPG":
        agent = DDPG(env,batch_size = args.batch, learning_rate=args.lr, gamma = args.gamma, train_freq=args.train_freq, 
                    gradient_steps=args.grad_step, buffer_size= int(args.buffer_size), target_network_tau= args.target_update_tau,
                    prioritized_replay = args.per, param_noise = args.noisynet, n_step = args.n_step, max_grad_norm = args.max_grad,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs)
    elif args.algo == "TD3":
        agent = TD3(env,batch_size = args.batch, learning_rate=args.lr, gamma = args.gamma, train_freq=args.train_freq, 
                    gradient_steps=args.grad_step, buffer_size= int(args.buffer_size), target_network_tau= args.target_update_tau,
                    prioritized_replay = args.per, target_action_noise_mul= args.target_noise_mul ,action_noise = args.action_noise,
                    n_step = args.n_step, max_grad_norm = args.max_grad,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs)
    elif args.algo == "TD4_QR":
        agent = TD4_QR(env,batch_size = args.batch, learning_rate=args.lr, gamma = args.gamma, train_freq=args.train_freq, 
                       gradient_steps=args.grad_step, buffer_size= int(args.buffer_size), target_network_tau= args.target_update_tau,
                    prioritized_replay = args.per, target_action_noise_mul= args.target_noise_mul , action_noise = args.action_noise, 
                    n_step = args.n_step, max_grad_norm = args.max_grad, 
                    risk_avoidance = args.risk_avoidance, n_support=args.n_support,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs)
    elif args.algo == "TD4_IQN":
        agent = TD4_IQN(env,batch_size = args.batch, learning_rate=args.lr, gamma = args.gamma, train_freq=args.train_freq, 
                       gradient_steps=args.grad_step, buffer_size= int(args.buffer_size), target_network_tau= args.target_update_tau,
                    prioritized_replay = args.per, target_action_noise_mul= args.target_noise_mul , action_noise = args.action_noise, 
                    n_step = args.n_step, max_grad_norm = args.max_grad, 
                    risk_avoidance = args.risk_avoidance, n_support=args.n_support,
                    tensorboard_log=args.logdir + env_type + "/" +env_name, policy_kwargs=policy_kwargs)

    agent.learn(int(args.steps))
    
    env.close()