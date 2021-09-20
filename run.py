import os
import argparse
import gym

from torch_baselines.DQN.dqn import DQN 
from mlagents_envs.environment import UnityEnvironment,ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="BipedalWalkerHardcore-v3", help='environment ID')
    parser.add_argument('--algo', type=str, default="DQN", help='algo ID')
    parser.add_argument('--steps', type=float, default=1e6, help='step size')
    parser.add_argument('--riskfactor', type=float, default=0, help='risk factor')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--policy_delay', type=int, default=2, help='policy_delay')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--n_support', type=int, default=32, help='number of quantile support')
    parser.add_argument('--tau', type=float, default=5e-3, help='target update gain')
    parser.add_argument('--train_freq', type=int, default=10, help='random seed')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='ent_coef')
    parser.add_argument('--logdir',type=str, default='log/',help='log file dir')
    parser.add_argument('--act_fun',type=str, default='relu',help='activation function')
    args = parser.parse_args() 
    env_name = args.env
    
    if os.path.exists(env_name):
        engine_configuration_channel = EngineConfigurationChannel()
        channel = EnvironmentParametersChannel()
        
        env = UnityEnvironment(file_name=env_name,no_graphics=False, side_channels=[engine_configuration_channel,channel])
        agent = DQN("asdf",env)
        
    else:
        
        pass
    
    agent.learn(int(args.steps))