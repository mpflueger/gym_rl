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


# def process_save_path(save_path, mode):
#     """ Prevent accidentally overwritng a checkpoint file """
#     save_path = os.path.abspath(save_path)

#     checkpoint_exists = False
#     if os.path.exists(os.path.dirname(save_path)):
#         if [f for f in os.listdir(os.path.dirname(save_path))
#               if re.match(os.path.basename(save_path), f)]:
#             checkpoint_exists = True

#     if checkpoint_exists and mode == 'train':
#         resp = input("Save path \'{}\' already exists, use anyway?"
#                           " [y/N]: ".format(save_path))
#         if not re.match(r'[yY](es)?$', resp):
#             save_path = None
#             print('\033[91m' + "  Checkpoints will NOT be saved!"
#                   + '\033[0m')

#     if not checkpoint_exists and mode == 'test':
#         print("Cannot use 'test' mode without a checkpoint.")

#     return save_path


# def parse_args(name):
#     """ Provide standard argument parsing """
#     parser = argparse.ArgumentParser(description="Run DDPG for {}".format(name))
#     parser.add_argument('mode', metavar='mode', default='train',
#         choices=['train', 'test'],
#         help="What do you want to do? {train, test}")
#     parser.add_argument('--save', '-s',
#         default='checkpoints/{}.ckpt'.format(name),
#         help="Change the default checkpoint save location")
#     parser.add_argument('--restore', '-r',
#         help="Restore a checkpoint before beginning")
#     return parser.parse_args()


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

