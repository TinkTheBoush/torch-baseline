import os
import argparse
import gym

from torch_baselines.DQN.dqn import DQN 
from mlagents_envs.environment import UnityEnvironment,ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="Cartpole-v1", help='environment')
    parser.add_argument('--algo', type=str, default="DQN", help='algo ID')
    parser.add_argument('--target_update', type=int, default=2000, help='target update intervals')
    parser.add_argument('--per', action='store_true')
    parser.add_argument('--steps', type=float, default=1e6, help='step size')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--logdir',type=str, default='log/',help='log file dir')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args() 
    env_name = args.env
    
    if os.path.exists(env_name):
        engine_configuration_channel = EngineConfigurationChannel()
        channel = EnvironmentParametersChannel()
        
        env = UnityEnvironment(file_name=env_name,no_graphics=False, side_channels=[engine_configuration_channel,channel],timeout_wait=10000)
        env_name = env_name.split('/')[-1].split('.')[0]
        
    else:
        env = gym.make(env_name)
        pass
    
    agent = DQN("asdf",env,target_network_update_freq=args.target_update,prioritized_replay=args.per,tensorboard_log=args.logdir+env_name)
    agent.learn(int(args.steps))