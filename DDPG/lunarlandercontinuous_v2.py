""" Train a DDPG for LunarLanderContinuous """
__author__ = "Max Pflueger"

import argparse
import math
import os
import re
from six.moves import input

import gym
import tensorflow as tf

from ddpg import DDPG
import ddpg_helpers


def mu_net(x, action_space, trainable=True):
    for h in action_space.high:
        if h != 1:
            raise ValueError("Action space must be -1 to 1")
    for l in action_space.low:
        if l != -1:
            raise ValueError("Action space must be -1 to 1")
    if len(action_space.shape) != 1:
        raise ValueError("Action dimension {} is not supported"
            .format(action_space.shape))
    action_dim = action_space.shape[0]

    net = tf.layers.dense(x, 256, activation=tf.nn.relu, name='fc1',
        trainable=trainable)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu, name='fc2',
        trainable=trainable)
    net = tf.layers.dense(net, action_dim, activation=tf.tanh,
        name='out', trainable=trainable)
    return net


def q_net(x, a, trainable=True):
    net = tf.layers.dense(x, 256, activation=tf.nn.relu, name='Q_fc1',
        trainable=trainable)
    net = tf.concat([net, a], axis=1, name='concat_actions')
    net = tf.layers.dense(net, 256, activation=tf.nn.relu, name='Q_fc2',
        trainable=trainable)
    net = tf.layers.dense(net, 1, name='Q_out', trainable=trainable)
    return net


def r(x,r):
    return r + math.exp(-10 * abs(x[0] - 0.45))


if __name__ == "__main__":
    print("WARNING: this file is not done.")
    env_name = 'LunarLanderContinuous-v2'
    args = ddpg_helpers.parse_args(env_name)

    env = gym.make(env_name)
    env.reset()

    # Check if path exists and avoid overwriting accidentally
    save_path = ddpg_helpers.process_save_path(args.save, args.mode)

    # Print Info
    print("Observation Space: {}".format(env.observation_space))
    print("Action Space: {}".format(env.action_space))

    # Set hyperparameters
    hypers = {
        'episode_len': 200,
        'terminate_episodes': True,
        'gamma': 0.99,
        'tau': 0.01,
        'max_buffer_size': 1000000,
        'batch_size': 256,
        'Q_learning_rate': 1e-3,
        'mu_learning_rate': 1e-4,
        'clip_gradients': False,
        'noise_sigma': 0.5,
    }

    my_DDPG = DDPG(env, hypers, q_net, mu_net, r=r)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.restore:
            my_DDPG.restore(sess, args.restore)

        if args.mode == 'train':
            for _ in range(10000):
                my_DDPG.train(sess, 5, save_path=save_path, log_dir=args.log)
                my_DDPG.test_episode(sess, render=True)

        if args.mode == 'test':
            episodes = 10
            my_DDPG.test_episode(sess, render=True)
            perf = my_DDPG.test(sess, episodes=episodes, render=False)
            print("Performance over {} episodes: {}".format(episodes, perf))

