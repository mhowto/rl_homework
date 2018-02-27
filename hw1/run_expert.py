#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import os
import gym_util 

envs = ['Hopper-v1', 'Ant-v1', 'HalfCheetah-v1', 'Humanoid-v1', 'Reacher-v1',  'Walker2d-v1']

def run_expert(expert_policy_file, envname, in_jupyter=False, render='store_true', max_timesteps=None, num_rollouts=20):
    policy_fn = load_policy.load_policy(expert_policy_file)
    print('---------------training ' + envname + '---------------')
    with tf.Session():
        tf_util.initialize()

        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            frames = []
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    if in_jupyter:
                        frames.append(env.render(mode='rgb_array'))
                    else:
                        env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            if render and in_jupyter:
                env.render(close=True)
                gym_util.display_frames_as_gif(frames)
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        save_expert_data(envname, expert_data, returns)
    print('--------------------------------------------\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        save_expert_data(args.envname, expert_data, returns)

def save_expert_data(envname, expert_data, returns=None):
    if not os.path.exists('data'):
        os.mkdir('data')
    np.save(os.path.join('data', '{}_observations'.format(envname)), expert_data['observations'])
    np.save(os.path.join('data', '{}_actions'.format(envname)), expert_data['actions'])
    if returns:
        np.save(os.path.join('data', '{}_returns'.format(envname)), returns)

if __name__ == '__main__':
    main()