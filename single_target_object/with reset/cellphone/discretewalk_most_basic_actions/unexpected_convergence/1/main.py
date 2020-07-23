# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/Kaixhin/Rainbow

The main file needed within rainbow. Runs of the train and test functions from their respective
files.

Example of use:
`cd algorithms/rainbow`
`python main.py`

Runs Rainbow DQN on our AI2ThorEnv wrapper with default params. Optionally it can be run on any
atari environment as well using the "game" flag, e.g. --game seaquest.
"""

import argparse

from datetime import datetime

import numpy as np
import torch

from algorithms.rainbow.agent import Agent
from algorithms.rainbow.env import Env, FrameStackEnv
from algorithms.rainbow.memory import ReplayMemory
from algorithms.rainbow.test import test
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
import time
from random import randrange
import ai2thor.controller
import pickle
from gym_ai2thor.tasks import Hideandseek

    
    
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
""" Game can be ai2thor to use our wrapper or one atari rom from the list here:
    https://github.com/openai/atari-py/tree/master/atari_py/atari_roms """
parser.add_argument('--game', type=str, default='ai2thor', help='ATARI game or environment')
parser.add_argument('--max-num-steps', type=int, default=int(50e6), metavar='STEPS',
                    help='Number of training steps')
parser.add_argument('--max-episode-length', type=int, default=int(400), metavar='LENGTH',
                    help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=100, metavar='T',
                    help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE',
                    help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.8, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--num-atoms', type=int, default=51, metavar='C',
                    help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V',
                    help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V',
                    help='Maximum of value distribution support')
parser.add_argument('--model-path', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=1, metavar='k',
                    help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.1, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n',
                    help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ',
                    help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(32e3), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE',
                    help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η',
                    help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε',
                    help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE',
                    help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(30000), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate-only', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=5e8, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=5, metavar='N',
                    help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=1500, metavar='N',
                    help='Number of transitions to use for validating Q')
parser.add_argument('--log-interval', type=int, default=200, metavar='STEPS',
                    help='Number of training steps between logging status')
parser.add_argument('--render', action='store_true', default=False,
                    help='Display screen (testing only)')
parser.add_argument('--config-file', type=str, default='config_files/rainbow_example.json',
                    help='Config file used for ai2thor environment definition')

if __name__ == '__main__':
    # Setup arguments, seeds and cuda
    args = parser.parse_args()
    print('-' * 10 + '\n' + 'Options' + '\n' + '-' * 10)
    for k, v in vars(args).items():
        print(' ' * 4 + k + ': ' + str(v))
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        # Disable non deterministic ops (not sure if critical but better safe than sorry)
        torch.backends.cudnn.enabled = False
    else:
        args.device = torch.device('cpu')

    # Simple ISO 8601 timestamped logger
    def log(s):
        print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)
        
    def load_memory(memory_path):
        with open(memory_path, 'rb') as f:
            print("Loading " + memory_path)
            return pickle.load(f)
        
    def save_memory(memory, memory_path):
        with open(memory_path, 'wb') as pickle_file:
            print("saving " + memory_path)
            pickle.dump(memory, pickle_file)


    # Environment selection
    if args.game == 'ai2thor':
        env = FrameStackEnv(AI2ThorEnv(config_file=args.config_file), args.history_length,
                            args.device)
        args.resolution = env.config['resolution']
        args.img_channels = env.observation_space.shape[0]
    else:
        env = Env(args)
        env.train()
        args.resolution = (84, 84)
        args.img_channels = 1
    action_space = env.action_space

    # Agent
    dqn = Agent(args, env)
#    dqn.save(path='weights', filename='hide0.pt')
#    dqn.save(path='weights', filename='hide.pt')
#    dqn.save(path='weights', filename='seek.pt')
    mem = ReplayMemory(args, args.memory_capacity)
    """ Priority weights are linearly annealed and increase every step by priority_weight_increase from 
    args.priority_weight to 1. 
    Typically, the unbiased nature of the updates is most important near convergence at the end of 
    training, as the process is highly non-stationary anyway, due to changing policies, state 
    distributions and bootstrap targets, that small bias can be ignored in this context.
    """
    priority_weight_increase = (1 - args.priority_weight) / (args.max_num_steps - args.learn_start)

    """ Construct validation memory. The transitions stored in this memory will remain constant for 
    the whole training process. During training this gives us a "fixed" evaluation dataset to see
    how the agent's confidence of its performance is improving. 
    """
    val_mem = ReplayMemory(args, args.evaluation_size)
    mem_steps, done = 0, True
    state = env.reset()
    found =0
    flag=0
    hidden=0
    sought=0
    doflag=0
    with open('variables.pickle', 'wb') as f:
        pickle.dump([found,flag,hidden,sought], f)
    for mem_steps in range(args.evaluation_size):   
#        time.sleep(0.05) 
        with open('variables.pickle', 'rb') as f:
            found,flag,hidden,sought = pickle.load(f)
        if done and sought==1:
#            dqn.save(path='weights', filename='seek0.pt')
#            dqn.load(args, env, path='weights/hide0.pt')
            print('hide mode')
            done = False
            flag=1
            sought=0
        if done and hidden==1:
#            next_state, reward, done, _ = env.step(17)
            if found==2:
                for i in range (700):
                    next_state, _, done, _ = env.step(randrange(4))
#            dqn.save(path='weights', filename='hide0.pt')
#            dqn.load(args, env, path='weights/seek0.pt')
            print('seek mode')
            done = False
            flag=0
            hidden=0
            found=0
        with open('variables.pickle', 'wb') as f:
            pickle.dump([found,flag,hidden,sought], f)
            
        action=env.action_space.sample()        

        next_state, reward, done, _ = env.step(action)  # Step  
        # No need to store actions or rewards because we only use the state to evaluate Q
        val_mem.append(state, None, None, done)
        state = next_state

    if args.evaluate_only:
#    if True:
        dqn.eval()  # Set DQN (online network) to evaluation mode
#        time.sleep(0.05) 
        try:
            avg_reward, avg_Q = test(env, mem_steps, args, dqn, val_mem, evaluate_only=True)
            print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        except:
            print("An exception occurred when evaluate"+str(mem_steps)) 
        
    else:
        # Training loop
#        dqn = Agent(args, env)
        reward_history_hide=[]
        reward_history_seek=[]
        found =0
        flag=1
        hidden=0
        sought=0
        doflag=0
        hide_round=1
        seek_round=0
        with open('variables.pickle', 'wb') as f:
                pickle.dump([found,flag,hidden,sought], f)
        dqn.train()
        num_steps, done = 0, True
        state, done = env.reset(), False
        state = env.reset_agent_with_object()
        print('hide mode')
        while num_steps < args.max_num_steps:
            # if num_steps == args.learn_start:
                # state, done = env.reset(), False
#            time.sleep(0.05) 
            with open('variables.pickle', 'rb') as f:
                found,flag,hidden,sought = pickle.load(f)
                
            if num_steps==100:
                dqn.save(path='weights', filename='seek.pt')
                save_memory(mem, 'seekmem.pickle')
                
                    
            if done and sought==1:
                
                # if num_steps>=args.learn_start:
                reward_history_hide.append([reward,num_steps])
                with open('reward_history_hide.pickle', 'wb') as f:
                    pickle.dump([reward_history_seek], f)
                dqn.save(path='weights', filename='seek.pt')
                save_memory(mem, 'seekmem.pickle')
                dqn.load(args, env, 'weights/hide.pt')
                mem =load_memory('hidemem.pickle')
                state = env.reset_agent_with_object()
                print('hide mode')
                hide_round+=1
                print('{}th hide round!'.format(hide_round))
                done = False
                flag=1
                sought=0
                
            if done and hidden==2:
                # if num_steps>=args.learn_start:
                reward_history_hide.append([reward,num_steps])   
                with open('reward_history_hide.pickle', 'wb') as f:
                    pickle.dump([reward_history_hide], f)
                hidden=0
                state, done = env.reset(), False
                state = env.reset_agent_with_object()
                print('hiding not successful, still hide mode')
                hide_round+=1
                print('{}th hide round!'.format(hide_round))
                
            if done and hidden==1:   
                # if num_steps>=args.learn_start:
                reward_history_hide.append([reward,num_steps])   
                with open('reward_history_hide.pickle', 'wb') as f:
                    pickle.dump([reward_history_hide], f)
                dqn.save(path='weights', filename='hide.pt')
                save_memory(mem, 'hidemem.pickle')
                dqn.load(args, env, 'weights/seek.pt')
                mem =load_memory('seekmem.pickle')
                print('seek mode')
                seek_round+=1
                print('{}th seek round!'.format(seek_round))
                done = False
                flag=0
                hidden=0
                found=0
                state = env.reset_agent()
                
            with open('variables.pickle', 'wb') as f:
                pickle.dump([found,flag,hidden,sought], f)
                
            if num_steps % args.replay_frequency == 0:
                dqn.reset_noise()  # Draw a new set of noisy epsilons
            
            action = dqn.act(state)  # Choose an action greedily (with noisy weights)

                
            next_state, reward, done, _ = env.step(action) 
            
            
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
            mem.append(state, action, reward, done)  # Append transition to memory
            num_steps += 1

            if num_steps % args.log_interval == 0:
                log('num_steps = ' + str(num_steps) + ' / ' + str(args.max_num_steps))

            # Train and test
            if num_steps >= args.learn_start:
                # Anneal importance sampling weight β to 1
                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                if num_steps % args.replay_frequency == 0:
                    dqn.learn(mem)  # Train with n-step distributional double-Q learning

                if num_steps % args.evaluation_interval == 0:
                    dqn.eval()  # Set DQN (online network) to evaluation mode. Fixed linear layers
                    # Test and save best model
                    try:
                        avg_reward, avg_Q = test(env, num_steps, args, dqn, val_mem)
                        log('num_steps = ' + str(num_steps) + ' / ' + str(args.max_num_steps) +
                        ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                    except:
                        print("An exception occurred when evaluate"+str(num_steps)) 
                    dqn.train()  # Set DQN (online network) back to training mode

                # Update target network
                if num_steps % args.target_update == 0:
                    dqn.update_target_net()
            state = next_state
    env.close()
