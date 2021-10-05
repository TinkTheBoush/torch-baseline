import os
import argparse
import gym

from torch_baselines.DDPG.ddpg import DDPG
from mlagents_envs.environment import UnityEnvironment,ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="Cartpole-v1", help='environment')
    parser.add_argument('--algo', type=str, default="DQN", help='algo ID')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
    parser.add_argument('--target_update_tau', type=float, default=0.99, help='target update intervals')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--buffer_size', type=float, default=50000, help='buffer_size')
    parser.add_argument('--per', action='store_true')
    parser.add_argument('--noisynet', action='store_true')
    parser.add_argument('--n_step', type=int, default=1, help='n step setting when n > 1 is n step td method')
    parser.add_argument('--steps', type=float, default=1e6, help='step size')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--logdir',type=str, default='log/',help='log file dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_grad', type=float,default=1.0, help='grad clip max size')
    args = parser.parse_args() 
    env_name = args.env
    
    if os.path.exists(env_name):
        engine_configuration_channel = EngineConfigurationChannel()
        channel = EnvironmentParametersChannel()
        
        env = UnityEnvironment(file_name=env_name,no_graphics=True, side_channels=[engine_configuration_channel,channel],timeout_wait=10000)
        engine_configuration_channel.set_configuration_parameters(time_scale=20.0)
        env_name = env_name.split('/')[-1].split('.')[0]
        
    else:
        env = gym.make(env_name)
        
    if args.algo == "DDPG":
        agent = DDPG(env,batch_size = args.batch, gamma = args.gamma, buffer_size= int(args.buffer_size), target_network_tau= args.target_update_tau,
                    prioritized_replay = args.per, param_noise = args.noisynet, n_step = args.n_step,
                    tensorboard_log=args.logdir+env_name)

    agent.learn(int(args.steps))