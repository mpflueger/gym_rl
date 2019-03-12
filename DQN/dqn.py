""" DQN for simple gym problems """
__author__ = "Max Pflueger"

import numpy as np
import os
import random
import re

import tensorflow as tf

class DQN:
    def __init__(self, env, hypers, q_net, r=None):
        self.env = env
        self.r_calc = r

        # Hyperparameter defaults
        self.episode_len = 200
        self.terminate_episodes = True
        self.gamma = 0.99
        self.C = 1000
        self.max_buffer_size = 1000000
        self.batch_size = 256
        self.learning_rate = 1e-3
        self.epsilon_0 = 1.0
        self.epsilon_f = 0.1
        self.epsilon_episodes = 1000
        self.clip_gradients = True

        # Set the supplied hyperparameters
        for k in hypers.keys():
            setattr(self, k, hypers[k])

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.inc_global_step_op = tf.assign_add(self.global_step, 1)
        self.global_episode = tf.Variable(0, trainable=False, name='global_episode')
        self.inc_global_episode_op = tf.assign_add(self.global_episode, 1)

        # Placeholders
        # state: x, action: a, reward: r, new state: xp,
        # sample is non-terminal: live
        self.x = tf.placeholder(tf.float32,
            shape=(None, ) + env.observation_space.shape,
            name='DQN_x')
        self.a = tf.placeholder(tf.int32,
            shape=(None, ),
            name='DQN_a')
        self.r = tf.placeholder(tf.float32,
            shape=(None, ),
            name='DQN_r')
        self.xp = tf.placeholder(tf.float32,
            shape=(None, ) + env.observation_space.shape,
            name='DQN_xp')
        self.live = tf.placeholder(tf.float32,
            shape=(None, ),
            name='DQN_live')

        # Create the Q network and target network with separate parameters
        with tf.variable_scope('Q_net'):
            self.Q = q_net(self.x, env.action_space.n)
        with tf.variable_scope('Q_target'):
            self.Q_target = q_net(self.xp, env.action_space.n,
                                  trainable=False)

        # Reduce to the action under an optimal policy
        self.Q_action = tf.argmax(self.Q, axis=1, name='Q_action')

        # Create an operation to copy variable values from Q to Q_target
        self.target_copy_op = self._copy_to_target()

        # Define the TD-loss function
        # L = (r + gamma * max(Q_target(s_p, a_p)) - Q(s,a))^2
        self.L_target = self.r \
            + self.live * self.gamma * tf.reduce_max(self.Q_target, axis=1)
        self.L_q = tf.diag_part(tf.gather(self.Q, self.a, axis=1))
        self.L = tf.reduce_mean((self.L_target - self.L_q)**2)

        # Create the gradient update operation
        opt = tf.train.RMSPropOptimizer(self.learning_rate)
        gvs = opt.compute_gradients(self.L)
        if self.clip_gradients:
            clipped_gvs = [(tf.clip_by_value(g, -1, 1), v) for (g, v) in gvs]
            self.train_op = opt.apply_gradients(clipped_gvs)
        else:
            self.train_op = opt.apply_gradients(gvs)

        # Create replay buffer
        self.replay_buffer = []

        # Init count between target network updates
        self.c = 0

        # Create the saver 
        self.saver = tf.train.Saver()

        # Log histograms
        tf.summary.scalar("TD-loss", self.L)
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.histogram(v.name, v)
        self.merged_summaries = tf.summary.merge_all()

        # Some debug stuff
        if False:
            print("List Variables:")
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                print(v.name)
            print("List Tensors:")
            for t in tf.get_default_graph().get_operations():
                print("{} {}".format(t.name, type(t)))

    def _copy_to_target(self):
        """ Copy Q_net to Q_target

        Create an operation to copy weights from the Q network to the
        target network
        """
        ops = []
        target_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='Q_target')
        for var in target_vars:
            var_from_Q = tf.get_default_graph().get_tensor_by_name(
                re.sub('Q_target', 'Q_net', var.name))
            ops.append(var.assign(var_from_Q))
        return tf.group(*ops, name='copy_to_target')

    def _add_transition(self, replay):
        """ Add a transition to the replay buffer """
        self.replay_buffer.append(replay)
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)

    def _get_minibatch(self, n):
        """ Get a minibatch of size n from the replay buffer """
        if n > len(self.replay_buffer):
            n = len(self.replay_buffer)
        return random.sample(self.replay_buffer, n)

    def train_episode(self, sess, epsilon=0.0, writer=None):
        obs = self.env.reset()
        r_sum = 0
        for _ in range(self.episode_len):
            #self.env.render()

            # Calculate action from current policy
            feed = {self.x: [obs]}
            [a] = sess.run([self.Q_action], feed_dict=feed)
            a = a[0]

            # Apply epsilon-greedy
            if random.random() < epsilon:
                a = self.env.action_space.sample()

            # Measure a state transition
            obs_new, r, done, info = self.env.step(a)
            if self.r_calc:
                r = self.r_calc(obs_new, r)
            r_sum += r
            
            # Add transition to replay buffer
            live = 1
            if self.terminate_episodes and done:
                live = 0
            t = [obs, a, r, obs_new, live] 
            self._add_transition(t)
            obs = obs_new

            # Sample minibatch from replay buffer
            minibatch = self._get_minibatch(self.batch_size)
            feed = {self.x: [],
                    self.a: [],
                    self.r: [],
                    self.xp: [],
                    self.live: []}
            for t in minibatch:
                feed[self.x].append(t[0])
                feed[self.a].append(t[1])
                feed[self.r].append(t[2])
                feed[self.xp].append(t[3])
                feed[self.live].append(t[4])

            # Perform gradient update
            [_, summary] = sess.run([self.train_op, self.merged_summaries], feed_dict=feed)
            step = sess.run(self.inc_global_step_op)
            self.c += 1

            if writer:
                writer.add_summary(summary, step)

            # Check if time to update target network 
            if self.c >= self.C:
                #print("Updating target network at c={}".format(self.c))
                sess.run(self.target_copy_op)
                self.c = 0

            # End episode when done
            if self.terminate_episodes and done:
                break
        sess.run(self.inc_global_episode_op)
        return r_sum

    def train(self, sess, episodes, save_path=None, log_dir=None):
        """ Train the model for a set number of episodes """
        train_writer = tf.summary.FileWriter(
            os.path.join(log_dir, "train"), graph=sess.graph)

        epsilon_slope = (self.epsilon_0 - self.epsilon_f)\
                        / self.epsilon_episodes
        ep = tf.train.global_step(sess, self.global_episode)
        for _ in range(episodes):
            if ep < self.epsilon_episodes:
                epsilon = self.epsilon_0 - (ep * epsilon_slope)
            else:
                epsilon = self.epsilon_f

            r = self.train_episode(sess, epsilon=epsilon, writer=train_writer)

            ep = tf.train.global_step(sess, self.global_episode)
            r_summary = tf.Summary(value=[
                tf.Summary.Value(tag="reward", simple_value=r)])
            train_writer.add_summary(r_summary, ep)

            # Print current test performance
            if ep % 50 == 0:
                print("Test at episode {}, performance: {}".format(
                    ep, self.test(sess)))
                if save_path:
                    sp_prefix = self.saver.save(sess, save_path)
                    print("Model saved in path {}".format(sp_prefix))

    def restore(self, sess, checkpoint):
        """ Restore model variables from a checkpoint """
        self.saver.restore(sess, checkpoint)

    def test_episode(self, sess, render=False):
        obs = self.env.reset()
        r_sum = 0
        for _ in range(self.episode_len):
            if render:
                self.env.render()

            # Calculate action from current policy
            feed = {self.x: [obs]}
            [a, Q] = sess.run([self.Q_action, self.Q], feed_dict=feed)
            a = a[0]
            if render:
                print(" State: {}, Q: {}".format(obs, Q))

            # Measure a state transition
            obs, r, done, info = self.env.step(a)
            r_sum += r

            if self.terminate_episodes and done:
                break
        return r_sum

    def test(self, sess, episodes=10):
        """ Evaluate performance of the current DQN """
        r_sum = 0
        render = True
        for i in range(episodes):
            r_sum += self.test_episode(sess, render=render)
            render = False
        r_mean = r_sum / episodes
        return r_mean

