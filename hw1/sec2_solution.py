from behaviour_cloning import *
from gym_util import run_gym
import matplotlib.pyplot as plt
import numpy as np

# envs = ['Hopper-v1', 'Ant-v1', 'HalfCheetah-v1', 'Humanoid-v1', 'Reacher-v1',  'Walker2d-v1']
envs = ['Hopper-v1']

for env in envs:
    print('training ' + env)
    obs = np.load(os.path.join('data', '{}_observations.npy'.format(env)))
    actions = np.load(os.path.join('data', '{}_actions.npy'.format(env)))
    actions = np.squeeze(actions)

    with tf.Graph().as_default():
        nn = SimpleNN(obs.shape[1], actions.shape[1])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            rewards = []
            for e in range(100):
                nn.train(sess, obs, actions, epoch=1)
                returns = run_gym(env, lambda x: nn.evaluate(sess, x))
                rewards.append(np.mean(returns))
            print(rewards)
            plt.plot(rewards)
            plt.show()
