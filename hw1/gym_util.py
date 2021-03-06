import gym
# from gym.configuration import undo_logger_setup
import tensorflow as tf
import tf_util
import numpy as np
import matplotlib.pyplot as plt

gym.configuration.undo_logger_setup()

def run_gym(envname, policy_fn, max_timesteps=None, num_rollouts=1, render=False):
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
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    # print('returns', returns)
    # print('mean return', np.mean(returns))
    # print('std of return', np.std(returns))
    return returns, observations, actions
