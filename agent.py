from model import CNNRLModel
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Agent:
    def __init__(self, gamma=0.95, lr=0.001, n_actions=2):
        self.gamma = gamma
        self.lr = lr
        self.model = CNNRLModel(n_actions)
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.action_memory = []
        self.reward_memory = []
        self.state_memory = []

    def choose_action(self, state):
        prob = self.model(state)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        self.action_memory.append(action)
        return int(action.numpy()[0])

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def store_state(self, state):
        self.state_memory.append(state)

    def learn(self):
        # G = np.zeros_like(self.reward_memory)
        # for i in range(len(self.reward_memory)):
        #     discount = 1
        #     g_sum = 0
        #     for j in range(i,len(self.reward_memory)):
        #         g_sum += self.reward_memory[j] * discount
        #         discount *= self.gamma
        #     G[i] = g_sum
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

    def calc_loss(self, prob, action, reward):
        print(prob)
        print(action)
        reward = tf.convert_to_tensor(reward)
        print(reward)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward
        return loss
