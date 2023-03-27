from models import DNNModel, CNNRLModel
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import deque
import random

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


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent2:
    def __init__(self, gamma=0.9, lr=LR, n_actions=2, cnn_model=True):
        self.lr = lr
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = gamma  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        if cnn_model:
            self.model = CNNRLModel(n_actions)
        else:
            self.model = DNNModel(n_actions)
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.criterion = tf.keras.losses.MeanSquaredError()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Method that gives a state to the model, and then gets an action in return from the model
        :param state: state of the game
        :return: an action calculated by the model given a state
        """
        probs = self.model(state)
        dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def get_action2(self, state):
        """
        Method that gives a state to the model, and then gets an action in return from the model
        :param state: state of the game
        :return: an action calculated by the model given a state
        """
        prob = self.model(state)
        prob = tf.squeeze(prob, 0)
        action = tf.argmax(prob, axis=0)
        action = int(action[0].numpy())
        return action

    def train_step(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            reward = tf.convert_to_tensor(reward, 1)
            # (n, x)

            '''if len(state.shape) == 1:
                # (1, x)
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done,)'''

            # 1: predicted Q values with current state
            pred = self.model(state, training=True)

            pred = [tf.squeeze(pred)]

            target = [tf.identity(pred)]

            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    probs = self.model(next_state[idx], training=True)

                    # Finding the index of the action with the biggest probability
                    index = tf.argmax(probs[0], 0).numpy().item()
                    Q_new = reward[idx] + self.gamma * probs[0][index]


                # Checking if the target already contain a numpy array
                if isinstance(target[0][0], np.ndarray):
                    list = target[0][0]
                else:
                    list = target[0][0].numpy()

                if list.shape[0] > 2:
                    list[idx][action[idx]] = Q_new
                    list = tf.convert_to_tensor(list)
                    target = [list]
                else:
                    list = target[0][0].numpy()
                    list[action[idx]] = Q_new
                    list = list.reshape((1, 2))
                    list = tf.convert_to_tensor(list)
                    target = [list]


            #target = tf.Variable(target)
            #pred = tf.Variable(pred)
            loss = self.criterion(target, pred)
            # loss.backward()
            x = self.model.trainable_variables
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))