from behaviour_cloning import *
from gym_util import run_gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import logging

logging.basicConfig(format='', filename='train.log', level=logging.DEBUG)

def run_bc(env, epochs=100, num_rollouts=1):
    obs = np.load(os.path.join('data', '{}_observations.npy'.format(env)))
    actions = np.load(os.path.join('data', '{}_actions.npy'.format(env)))
    actions = np.squeeze(actions)

    with tf.Graph().as_default():
        nn = SimpleNN(obs.shape[1], actions.shape[1])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            rewards = []
            for e in tqdm(range(epochs), desc='env='+env):
                nn.train(sess, obs, actions, epoch=1)
                returns = run_gym(env, lambda x: nn.evaluate(sess, x), num_rollouts=num_rollouts)
                rewards.append(np.mean(returns))
            rewards = np.array(rewards)
            np.save(os.path.join('data', '{}_bc_rewards'.format(env)), rewards)


def get_bc_rewards(env):
    filepath = os.path.join('data', '{}_bc_rewards.npy'.format(env))
    rewards = np.load(filepath)
    x = np.arange(1, rewards.size + 1)
    return {'x': x, 'y': rewards}