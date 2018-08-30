""" DDPG for simple gym problems """
__author__ = "Max Pflueger"

import numpy as np
import random
import re

import tensorflow as tf

class DDPG:
    def __init__(self, env, hypers, q_net, mu_net, r=None):
        self.env = env
        self.r_calc = r

        # Hyperparameter defaults
        self.episode_len = 200
        self.terminate_episodes = True
        self.gamma = 0.99
        self.replay_buffer_size = 1000000
        self.batch_size = 256
        self.Q_learning_rate = 1e-3
        self.mu_learning_rate = 1e-4
        self.tau = 0.001
        self.clip_gradients = True
        self.noise_theta = 0.15
        self.noise_sigma = 0.2

        # Set the supplied hyperparameters
        for k in hypers.keys():
            setattr(self, k, hypers[k])

        # Placeholders
        # state: x, action: a, reward: r, new state: xp,
        # sample is non-terminal: live
        self.x = tf.placeholder(tf.float32,
            shape=(None, ) + env.observation_space.shape,
            name='x')
        self.a = tf.placeholder(tf.float32,
            shape=(None, ) + env.action_space.shape,
            name='a')
        self.r = tf.placeholder(tf.float32,
            shape=(None, ),
            name='r')
        self.xp = tf.placeholder(tf.float32,
            shape=(None, ) + env.observation_space.shape,
            name='xp')
        self.live = tf.placeholder(tf.float32,
            shape=(None, ),
            name='live')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.inc_global_step_op = tf.assign_add(self.global_step, 1)
        self.global_episode = tf.Variable(0, trainable=False, name='global_episode')
        self.inc_global_episode_op = tf.assign_add(self.global_episode, 1)

        # Create actor and critic networks
        # Calculate critic loss
        with tf.variable_scope('mu_target'):
            self.mu_target = mu_net(self.xp, self.env.action_space,
                                    trainable=False)
        with tf.variable_scope('Q_target'):
            self.Q_target = q_net(self.xp, self.mu_target,
                                  trainable=False)
        with tf.variable_scope('Q_net'):
            self.Q_critic = q_net(self.x, self.a)
        self.critic_y = self.r + self.live * self.gamma * self.Q_target
        self.critic_loss = tf.losses.mean_squared_error(
            self.critic_y, self.Q_critic)
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope='Q_net')

        # Calculate actor loss
        with tf.variable_scope('mu_net'):
            self.mu = mu_net(self.x, self.env.action_space)
        with tf.variable_scope('Q_net', reuse=True):
            self.Q_actor = q_net(self.x, self.mu)
        self.actor_loss = - tf.reduce_mean(self.Q_actor)
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='mu_net')

        # Critic Optimizer
        critic_opt = tf.train.AdamOptimizer(self.Q_learning_rate)
        gvs = critic_opt.compute_gradients(
            self.critic_loss,
            var_list=self.critic_vars)
        if self.clip_gradients:
            clipped_gvs = [(tf.clip_by_value(g, -1, 1), v) for (g, v) in gvs]
            self.train_critic_op = critic_opt.apply_gradients(clipped_gvs)
        else:
            self.train_critic_op = critic_opt.apply_gradients(gvs)

        # Actor Optimizer
        actor_opt = tf.train.AdamOptimizer(self.mu_learning_rate)
        gvs = actor_opt.compute_gradients(
            self.actor_loss,
            var_list=self.actor_vars)
        if self.clip_gradients:
            clipped_gvs = [(tf.clip_by_value(g, -1, 1), v) for (g, v) in gvs]
            self.train_actor_op = actor_opt.apply_gradients(clipped_gvs)
        else:
            self.train_actor_op = actor_opt.apply_gradients(gvs)

        # Target Network Update
        self.target_update_op = self._get_target_soft_update_op()

        self.replay_buffer = []
        self.saver = tf.train.Saver()

    def _get_target_soft_update_op(self):
        """ Return a tf operation to do target soft update.

        Follows the formula from [Lillicrap et. al. 2016]: 
        theta' = tau * theta + (1-tau)*theta'
        """
        ops = []

        target_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='Q_target')
        for var in target_vars:
            var_from_Q = tf.get_default_graph().get_tensor_by_name(
                re.sub('Q_target', 'Q_net', var.name))
            ops.append(var.assign(var_from_Q * self.tau + (1 - self.tau) * var))
        target_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='mu_target')
        for var in target_vars:
            var_from_mu = tf.get_default_graph().get_tensor_by_name(
                re.sub('mu_target', 'mu_net', var.name))
            ops.append(var.assign(var_from_mu * self.tau + (1 - self.tau) * var))

        return tf.group(*ops, name='update_target')

    def _add_transition(self, replay):
        """ Add a transition to the replay buffer """
        self.replay_buffer.append(replay)
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def _get_minibatch(self, n):
        """ Get a minibatch of size n from the replay buffer """
        if n > len(self.replay_buffer):
            n = len(self.replay_buffer)
        return random.sample(self.replay_buffer, n)

    def restore(self, sess, checkpoint):
        """ Restore model variables from a checkpoint """
        self.saver.restore(sess, checkpoint)

    def train_episode(self, sess):
        obs = self.env.reset()
        noise = np.zeros(self.env.action_space.shape)
        noise_mu = np.zeros(self.env.action_space.shape)

        step = tf.train.global_step(sess, self.global_step)
        for _ in range(self.episode_len):
            #self.env.render()

            # Calculate action from current policy
            feed = {self.x: [obs]}
            [a, Q] = sess.run([self.mu, self.Q_actor], feed_dict=feed)
            a = np.reshape(a, self.env.action_space.shape)

            # Add action noise
            # Select noise from Ornstein-Uhlenbeck process
            # TODO: Did I do this right?
            noise_W = np.random.normal(
                scale=self.noise_sigma,
                size=self.env.action_space.shape)
            noise += self.noise_theta * (noise_mu - noise) + noise_W
            a += noise

            # Execute action and save transition to replay buffer
            obs_new, r, done, info = self.env.step(a)
            # TODO: this next line is dumb. obs has different shape from
            #   reset() and step() in the mountaincarcontinuous_v0 space,
            #   seems like a bug...
            obs_new = np.reshape(obs_new, self.env.observation_space.shape)

            if self.r_calc:
                r = self.r_calc(obs_new, r)

            live = 1
            if done:
                live = 0
            t = [obs, a, r, obs_new, live]
            self._add_transition(t)
            obs = obs_new

            # Sample a minibatch from the replay buffer
            minibatch = self._get_minibatch(self.batch_size)
            feed = {
                self.x: [],
                self.a: [],
                self.r: [],
                self.xp: [],
                self.live: [],
            }
            for t in minibatch:
                feed[self.x].append(t[0])
                feed[self.a].append(t[1])
                feed[self.r].append(t[2])
                feed[self.xp].append(t[3])
                feed[self.live].append(t[4])

            # Update critic (Q)
            sess.run(self.train_critic_op, feed_dict=feed)

            # Update actor (mu)
            sess.run(self.train_actor_op, feed_dict=feed)

            # Update target networks
            sess.run(self.target_update_op)

            # Update the global step
            step = sess.run(self.inc_global_step_op)

            if self.terminate_episodes and done:
                break

        return step

    def train(self, sess, episodes, save_path=None):
        """ Train the model for a set number of episodes """
        for _ in range(episodes):
            step = self.train_episode(sess)
            ep = sess.run(self.inc_global_episode_op)
            print(ep)

            if ep % 50 == 0:
                perf = self.test(sess)
                print("Test at episode {}, performance: {}".format(ep, perf))

                if save_path:
                    sp_prefix = self.saver.save(sess, save_path)
                    print("Model saved in path {}".format(sp_prefix))

    def test_episode(self, sess, render=False):
        """ Evaluate performance on a single episode """
        obs = self.env.reset()
        print(obs)
        r_sum = 0
        for _ in range(self.episode_len):
            if render:
                self.env.render()

            # Calculate action from current policy
            # TODO: this next line is dumb. obs has different shape from
            #   reset() and step() in the mountaincarcontinuous_v0 space,
            #   seems like a bug...
            obs = np.reshape(obs, self.env.observation_space.shape)
            feed = {self.x: [obs]}
            [a, Q] = sess.run([self.mu, self.Q_actor], feed_dict=feed)

            if render:
                print(" State: {}, Q: {}, a: {}".format(obs, Q, a))

            # Step environment
            obs, r, done, info = self.env.step(a)
            r_sum += r

            if self.terminate_episodes and done:
                break
        return r_sum


    def test(self, sess, episodes=10, render=False):
        """ Evaluate performance of the current policy """
        r_sum = 0
        for i in range(episodes):
            r_sum += self.test_episode(sess, render=render)
        r_mean = r_sum / episodes
        return r_mean

