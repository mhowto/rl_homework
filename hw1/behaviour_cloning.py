import numpy as np
import os
import tensorflow as tf


class BatchIterator(object):
    def __init__(self, data, batch_size, epoch=None, shuffle=False):
        if isinstance(data, tuple) or isinstance(data, list):
            self.data = [np.array(d) for d in data]
            self.size = self.data[0].shape[0]
        else:
            self.data = np.array(data)
            self.size = self.data.shape[0]
        self.indexs = np.arange(self.size)

        if shuffle:
            self.indexs = np.random.shuffle(self.indexs)

        self.stop = epoch
        self._index = 0
        self._epoch_count = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop and self._epoch_count >= self.stop:
            raise StopIteration
        if self._index + self.batch_size >= self.size:
            self._epoch_count += 1
            indexs = np.concatenate((self.indexs[self._index:], self.indexs[:(self.batch_size-(self.size - self._index))]))
        else:
            indexs = self.indexs[self._index:self._index+self.size]

        self._index = (self._index + self.batch_size) % self.size

        if isinstance(self.data, tuple) or isinstance(self.data, list):
            return [d[indexs, ...] for d in self.data]
        else:
            return self.data[indexs, ...]

class SimpleNN(object):
    def __init__(self, obs_size, action_size, learning_rate=0.01):
        '''
        observations = tf.squeeze(observations)
        actions = tf.squeeze(actions)
        obs_size = observations.get_shape()[1]
        action_size = actions.get_shape()[1]
        '''
        self.obs_size = obs_size
        self.action_size = action_size

        self.inputs_ = tf.placeholder(tf.float32, shape=[None, obs_size], name='inputs_')
        self.outputs_ = tf.placeholder(tf.float32, shape=[None, action_size], name='outputs_')

        with tf.variable_scope('layer1'):
            self.l1_out = tf.layers.dense(self.inputs_, 64, activation=tf.nn.relu, name='l1_out')

        with tf.variable_scope('layer2'):
            self.outputs = tf.layers.dense(self.l1_out, action_size, name='layer2')
        self.loss = tf.reduce_mean(tf.squared_difference(self.outputs, self.outputs_))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
    
    def train(self, sess, observations, actions, epoch=100):
        iter_step = 0
        for ob, ac in BatchIterator((observations, actions), batch_size=128, epoch=epoch):
            loss, _ = sess.run([self.loss, self.optimizer], feed_dict={self.inputs_: ob, self.outputs_: ac,})
            iter_step += 1
            # if iter_step % 400 == 0:
                # print('iter: {}, loss: {}'.format(iter_step, loss))

    def evaluate(self, sess, observations):
        observations = np.reshape(observations, (-1, self.obs_size))
        actions = sess.run(self.outputs, feed_dict={self.inputs_: observations})
        return actions

if __name__ == '__main__':
    #envs = ['Hopper-v1', 'Ant-v1', 'HalfCheetah-v1', 'Humanoid-v1', 'Reacher-v1',  'Walker2d-v1']
    envs = ['Hopper-v1']

    for env in envs:
        print('training '+env)
        obs = np.load(os.path.join('data', '{}_observations.npy'.format(env)))
        actions = np.load(os.path.join('data', '{}_actions.npy'.format(env)))
        actions = np.squeeze(obs)

        with tf.Graph().as_default():
            nn = SimpleNN(obs.shape[1], actions.shape[1])
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                nn.train(obs, actions, epoch=40)
                actions = nn.evaluate(obs[0, ...])
                print(actions)
