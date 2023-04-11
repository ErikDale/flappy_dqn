from models import DNNModel, CNNRLModel, ActorCriticModel, ActorCriticModel2
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
BATCH_SIZE = 100
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
        # Maybe loop over this?
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


class ActorCriticAgent:
    def __init__(self, num_actions, replay_buffer_size=10000, batch_size=16, discount_factor=0.99, learning_rate=0.001):
        self.model = ActorCriticModel(num_actions)
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.replay_buffer = []
        self.steps = 0
        self.num_actions = num_actions

    def learn(self):
        #if len(self.replay_buffer) < self.batch_size:
        #    return

        self.batch_size = int(len(self.replay_buffer) * 0.1)

        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)

        # Convert the batch of experiences to tensors
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])

        with tf.GradientTape() as tape:
            # Compute the expected action values for the next state using the critic model
            _, next_state_expected_action_values = self.model(next_states)

            # Compute the target Q-values for the current state
            target_q_values = rewards + self.discount_factor * np.amax(next_state_expected_action_values, axis=1)

            # Compute the expected action values for the current state using the critic model
            current_state_expected_action_values, _ = self.model(states)

            # Compute the loss between the expected action values and the target Q-values

            actions_mask = tf.one_hot(actions, self.num_actions)
            q_values = tf.reduce_sum(current_state_expected_action_values * actions_mask, axis=1)
            critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))

            # Compute the policy loss between the actor output and the estimated value of the action by the critic
            advantages = target_q_values - tf.reduce_mean(current_state_expected_action_values, axis=1)
            actor_loss = tf.keras.losses.categorical_crossentropy(actions_mask, current_state_expected_action_values, from_logits=False)
            actor_loss = tf.reduce_mean(actor_loss * tf.expand_dims(advantages, axis=-1))

            # Compute the total loss as a weighted sum of the critic and actor losses
            total_loss = critic_loss + actor_loss

            # Compute the gradients of the loss with respect to the model variables
            variables = self.model.trainable_variables
            gradients = tape.gradient(total_loss, variables)

            # Apply the gradients to the model variables to update the model
            self.optimizer.apply_gradients(zip(gradients, variables))
            self.reset()

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        actor_output, _ = self.model(state)
        action = np.random.choice(self.num_actions, p=actor_output.numpy()[0])
        return action

    def reset(self):
        self.replay_buffer = []
        self.steps = 0


class ActorCriticAgent2:
    def __init__(self, num_actions, replay_buffer_size=10000, batch_size=16, discount_factor=0.99, actor_learning_rate=1e-4, critic_learning_rate=1e-3):
        self.num_actions = num_actions
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        # Initialize actor-critic model
        self.model = ActorCriticModel2(num_actions)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def act(self, state):
        """
        First the maximum value from the actor output is removed to prevent overflow during exponentiation.
        Then the output is exponentiated and added a small epsilon value to ensure all probabilities are non-zero.
        Finally, a normalization of the probabilities is done to obtain a policy, and sample an action from it.
        """
        state = np.expand_dims(state, axis=0)
        actor_output, _ = self.model(state)
        actor_output = tf.squeeze(actor_output)
        actor_output = actor_output - tf.reduce_max(actor_output)
        exp_actor_output = tf.exp(actor_output) + 1e-7  # Add small epsilon value to ensure non-zero
        policy = exp_actor_output / tf.reduce_sum(exp_actor_output)
        action = np.random.choice(self.num_actions, p=policy.numpy())
        return action

    def learn(self):
        self.batch_size = int(len(self.replay_buffer) * 0.1)

        # Sample minibatch from replay buffer
        minibatch = np.array(random.sample(self.replay_buffer, self.batch_size), dtype=object)

        # Unpack minibatch
        states = np.stack(minibatch[:, 0])
        actions = minibatch[:, 1].astype(int)
        rewards = minibatch[:, 2]
        next_states = np.stack(minibatch[:, 3])

        # Convert actions to one-hot encoding
        actions_mask = tf.one_hot(actions, self.num_actions)

        input = (next_states, actions)
        # Compute target values for critic
        _, critic_next = self.model(input)
        target = rewards + self.discount_factor * critic_next.numpy().flatten()

        # Compute critic loss
        with tf.GradientTape() as tape:
            input = (states, actions)
            _, critic_output = self.model(input)
            critic_output = tf.squeeze(critic_output)
            target = target.astype(np.float32)
            critic_loss = tf.keras.losses.mean_squared_error(target, critic_output)

        # Compute critic gradients and update weights
        critic_grads = tape.gradient(critic_loss, self.model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.model.trainable_variables))

        # Compute actor loss
        with tf.GradientTape() as tape:
            actor_output, _ = self.model(input)
            actor_loss = -tf.reduce_mean(tf.math.log(tf.reduce_sum(actor_output * actions_mask, axis=1)) * (target - critic_output))

        # Compute actor gradients and update weights
        actor_grads = tape.gradient(actor_loss, self.model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))
        self.reset()

    def reset(self):
        self.replay_buffer.clear()

