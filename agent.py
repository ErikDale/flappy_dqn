from models import DNNModel, CNNRLModel
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

"""
@author: Erik Dale
@date: 21.03.23
"""

class Agent:
    def __init__(self, gamma=0.80, lr=0.001, n_actions=2, cnn_model=True):
        self.gamma = gamma
        self.lr = lr
        if cnn_model:
            self.model = CNNRLModel(n_actions)
        else:
            self.model = DNNModel(n_actions)
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.action_memory = []
        self.reward_memory = []
        self.state_memory = []

    def choose_action(self, state):
        """
        Method that gives a state to the model, and then gets an action in return from the model
        :param state: state of the game
        :return: an action calculated by the model given a state
        """
        prob = self.model(state)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        self.action_memory.append(action)
        x = action.numpy()
        return int(action.numpy()[0])
 
    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def store_state(self, state):
        self.state_memory.append(state)

    def store_action(self, action):
        self.action_memory.append(action)

    def learn(self):
        sum_reward = 0
        discnt_rewards = []
        self.reward_memory.reverse()
        for r in self.reward_memory:
            sum_reward = r + self.gamma * sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()

        for state, action, reward in zip(self.state_memory, self.action_memory, discnt_rewards):
            with tf.GradientTape() as tape:
                p = self.model(np.array(state), training=True)
                loss = self.calc_loss(p, action, reward)
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        self.reward_memory = []
        self.action_memory = []
        self.state_memory = []

    '''def calc_loss(self, prob, action, reward):
        reward = tf.convert_to_tensor(reward, 1)
        # Add a small value to prob to avoid getting nan or inf
        dist = tfp.distributions.Categorical(probs=prob + 1e-8, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward
        return loss'''

    def calc_loss(self, prob, action, reward):
        reward = tf.convert_to_tensor(reward, 1)
        # Add a small value to prob to avoid getting nan or inf
        dist = tfp.distributions.Categorical(probs=prob + 1e-8, dtype=tf.float32)
        prob_action = dist.probs_parameter()[0][int(action)]
        loss = -tf.math.log(prob_action) * reward
        return loss
