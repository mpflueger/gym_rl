""" Train a DQN for the LunarLander """
__author__ = "Max Pflueger"

import gym
import tensorflow as tf

from dqn import DQN
import dqn_helpers


def r(x,r):
    return r


if __name__ == "__main__":
    args = dqn_helpers.parse_args('LunarLander-v2')

    env = gym.make('LunarLander-v2')
    env.reset()

    # Check if path exists and avoid overwriting accidentally
    save_path = dqn_helpers.process_save_path(args.save, args.mode)

    # Print Info
    print("Observation Space: {}".format(env.observation_space))
    print("Action Space: {}".format(env.action_space))

    # Set hyperparameters
    hypers = {
        'episode_len': 200,
        'terminate_episodes': True,
        'gamma': 0.99,
        'C': 1000,
        'max_buffer_size': 1000000,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'epsilon_0': 1.0,
        'epsilon_f': 0.1,
        'epsilon_episodes': 1000,
    }

    my_DQN = DQN(env, hypers, dqn_helpers.q_net_3, r=r)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.restore:
            my_DQN.restore(sess, args.restore)

        if args.mode == 'train':
            my_DQN.train(sess, 50000, save_path=save_path, log_dir=args.log)
        if args.mode == 'test':
            episodes = 10
            perf = my_DQN.test(sess, episodes=episodes)
            print("Performance over {} episodes: {}".format(episodes, perf))

