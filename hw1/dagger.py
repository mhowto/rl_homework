import numpy as np
import os
import tensorflow as tf
from behaviour_cloning import *
from gym_util import run_gym
import load_policy
from tqdm import tqdm
 
def data_augment(env, epochs=100, num_rollouts=1):
    obs = np.load(os.path.join('data', '{}_observations.npy'.format(env)))
    actions = np.load(os.path.join('data', '{}_actions.npy'.format(env)))
    actions = np.squeeze(actions)

    with tf.Graph().as_default():
        policy_fn = load_policy.load_policy('experts/{}.pkl'.format(env))
        nn = SimpleNN(obs.shape[1], actions.shape[1])
        nn2 = SimpleNN(obs.shape[1], actions.shape[1], name='simple_nn2')
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            rewards_dagger = []
            rewards_no_dagger = []

            # train nn
            for e in tqdm(range(epochs), desc='env='+env):
                nn.train(sess, obs, actions, epoch=1)
                returns, obs, _ = run_gym(env, lambda x: nn.evaluate(sess, x), num_rollouts=num_rollouts)

                # dagger
                actions_dagger = [policy_fn(ob[None, :]) for ob in obs]
                actions_dagger = np.squeeze(np.array(actions_dagger))
                
                nn2.train(sess, obs, actions_dagger, epoch=1)

                returns_dagger, _, _ = run_gym(env, lambda x: nn2.evaluate(sess, x), num_rollouts=num_rollouts)
                rewards_dagger.append(np.mean(returns_dagger))
                rewards_no_dagger.append(np.mean(returns))

            np.save(os.path.join('data', '{}_bc_rewards_no_dagger'.format(env)), np.array(rewards_no_dagger))
            np.save(os.path.join('data', '{}_bc_rewards_dagger'.format(env)), np.array(rewards_dagger))
    '''            

    nn = SimpleNN(obs.shape[1], actions.shape[1])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # train nn
        for e in tqdm(range(epochs), desc='env='+env):
            nn.train(sess, obs, actions, epoch=1)
            returns, obs, _ = run_gym(env, lambda x: nn.evaluate(sess, x), num_rollouts=num_rollouts)

            # dagger
            actions_dagger = [policy_fn(ob[None, :]) for ob in obs]
            actions_dagger = np.squeeze(np.array(actions_dagger))
                
            nn.train(sess, obs, actions_dagger, epoch=1)

            returns_dagger, _, _ = run_gym(env, lambda x: nn.evaluate(sess, x), num_rollouts=num_rollouts)

            np.save(os.path.join('data', '{}_bc_rewards_no_dagger'.format(env)), np.array(returns))
            np.save(os.path.join('data', '{}_bc_rewards_dagger'.format(env)), np.array(returns_dagger))
    '''

def get_dagger_rewards(env):
    rewards_with_no_dagger = np.load(os.path.join('data', '{}_bc_rewards_no_dagger.npy'.format(env)))
    x1 = np.arange(1, rewards_with_no_dagger.size + 1)
    rewards_with_dagger = np.load(os.path.join('data', '{}_bc_rewards_dagger.npy'.format(env)))
    x2 = np.arange(1, rewards_with_dagger.size + 1)

    return {'x': x1, 'y': rewards_with_no_dagger}, {'x': x2, 'y': rewards_with_dagger}
