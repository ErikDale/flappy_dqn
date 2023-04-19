from models import DNNModel, CNNRLModel, CriticModel, ActorModel
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import deque
import random
import math

"""
@author: Erik Dale
@date: 21.03.23
"""


class ActorCriticAgent:
    """
    Actor-Critic agent that learns to interact with an environment.
    :param num_actions: the number of actions available to the agent.
    :param replay_buffer_size: the maximum size of the replay buffer. Default is 20000.
    :param batch_size: the size of the minibatch sampled from the replay buffer during training. Default is 32.
    :param discount_factor: the discount factor for future rewards. Default is 0.95.
    :param actor_learning_rate: the learning rate for the actor model. Default is 0.0001.
    :param critic_learning_rate: the learning rate for the critic model. Default is 0.001.
    """
    def __init__(self, num_actions, replay_buffer_size=20000, batch_size=32, discount_factor=0.95,
                 actor_learning_rate=0.0001, critic_learning_rate=0.001):
        """
        Initialize the actor critic agent.
        """
        self.num_actions = num_actions
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.gradient_clip_norm = 1

        # Initialize actor-critic model
        self.actor_model = ActorModel(num_actions)
        self.critic_model = CriticModel()

        # self.actor_optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate=actor_learning_rate, momentum=0.95, weight_decay=0.9)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)

    def remember(self, state, action, reward, next_state, done):
        """
        Store a new experience tuple in the replay buffer.
        :param state: the current state of the environment.
        :param action: the action taken by the agent.
        :param reward: the reward received by the agent.
        :param next_state: the next state of the environment.
        :param done: whether the episode has terminated or not.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """
        Generate an action using the current policy.
        :param state: the current state of the environment.
        :return: the chosen action.
        """
        state = np.expand_dims(state, axis=0)
        actor_output = self.actor_model(state)
        actor_output = tf.squeeze(actor_output)
        actor_output = actor_output - tf.reduce_max(actor_output)
        exp_actor_output = tf.exp(actor_output) + 1e-2  # Add small epsilon value to ensure non-zero values
        policy = exp_actor_output / tf.reduce_sum(exp_actor_output)
        if math.isnan(policy[0]) or math.isnan(policy[1]):
            action = 0
        else:
            # action = np.random.choice(self.num_actions, p=policy.numpy())
            action = tf.argmax(policy)
        return action

    def learn(self):
        """
        Update the actor and critic models using a batch of experiences from the replay buffer.
        """
        # Sample minibatch from replay buffer
        minibatch = np.array(random.sample(self.replay_buffer, self.batch_size), dtype=object)

        # Unpack minibatch
        states = np.stack(minibatch[:, 0])
        actions = minibatch[:, 1].astype(int)
        rewards = minibatch[:, 2]
        next_states = np.stack(minibatch[:, 3])
        dones = minibatch[:, 4]

        # Convert inputs to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute critic loss
        with tf.GradientTape() as tape:
            inputs = (states, actions)
            values = self.critic_model(inputs)

            inputs = (next_states, actions)
            next_values = self.critic_model(inputs)

            next_values = np.array(next_values)

            td_targets = rewards + self.discount_factor * next_values * (1 - dones)
            td_errors = td_targets - values
            critic_loss = tf.reduce_mean(tf.square(td_errors))

        # Compute critic gradients and apply to optimizer
        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.gradient_clip_norm)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        # Compute actor loss
        with tf.GradientTape() as tape:
            policy = self.actor_model(states)
            actions_one_hot = tf.one_hot(actions, self.num_actions)
            advantages = tf.stop_gradient(td_errors)
            actor_loss = -tf.reduce_mean(
                tf.reduce_sum(actions_one_hot * tf.math.log(policy + 1e-10), axis=1) * advantages)

        # Compute actor gradients and apply to optimizer
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.gradient_clip_norm)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))


class Agent:
    """
    Agent that learns to interact with an environment.
    :param gamma: the discount factor for future rewards. Default is 0.95.
    :param lr: the learning rate for the optimizer. Default is 0.00001.
    :param n_actions: the number of actions available to the agent. Default is 2.
    :param cnn_model: a boolean indicating whether to use a convolutional neural network as the model. Default is True.
    :param replay_buffer_size: the maximum size of the replay buffer. Default is 20000.
    """
    def __init__(self, gamma=0.95, lr=0.000001, n_actions=2, cnn_model=True, replay_buffer_size=20000, batch_size=32):
        """
        Initializes a new Agent object.
        """
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.gamma = gamma
        self.lr = lr
        if cnn_model:
            self.model = CNNRLModel(n_actions)
        else:
            self.model = DNNModel(n_actions)
        # self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.opt = tf.keras.optimizers.experimental.RMSprop(learning_rate=self.lr, momentum=0.95, weight_decay=0.9)
        self.n_actions = n_actions
        self.gradient_clip_norm = 1
        self.batch_size = batch_size

    def choose_action(self, state):
        """
        Generate an action using the current policy.
        :param state: the current state of the environment.
        :return: the chosen action.
        """
        state = np.expand_dims(state, axis=0)
        actor_output = self.model(state)
        actor_output = tf.squeeze(actor_output)
        actor_output = actor_output - tf.reduce_max(actor_output)
        exp_actor_output = tf.exp(actor_output) + 1e-2  # Add small epsilon value to ensure non-zero
        policy = exp_actor_output / tf.reduce_sum(exp_actor_output)
        if math.isnan(policy[0]) or math.isnan(policy[1]):
            action = 0
        else:
            # action = np.random.choice(self.n_actions, p=policy.numpy())
            action = tf.argmax(policy)
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Store a new experience tuple in the replay buffer.
        :param state: the current state of the environment.
        :param action: the action taken by the agent.
        :param reward: the reward received by the agent.
        :param next_state: the next state of the environment.
        :param done: whether the episode has terminated or not.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self):
        """
        Train the agent's model using experiences from the replay buffer.
        """
        # Sample minibatch from replay buffer
        minibatch = np.array(random.sample(self.replay_buffer, self.batch_size), dtype=object)

        # Unpack minibatch
        states = np.stack(minibatch[:, 0])
        actions = minibatch[:, 1].astype(int)
        rewards = minibatch[:, 2]
        next_states = np.stack(minibatch[:, 3])
        dones = minibatch[:, 4]

        # Convert data to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.one_hot(actions, self.n_actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute target values
        target_actions = self.model(next_states)
        target_actions = tf.squeeze(target_actions)  # remove dimensions of size 1
        target_values = tf.reduce_max(target_actions, axis=1)
        target_values = rewards + self.gamma * target_values * (1 - dones)

        # Compute loss and update weights
        with tf.GradientTape() as tape:
            # Forward pass
            predicted_actions = self.model(states)
            predicted_actions = tf.squeeze(predicted_actions)
            predicted_values = tf.reduce_sum(predicted_actions * actions, axis=1)

            # Compute loss
            loss = tf.keras.losses.mean_squared_error(target_values, predicted_values)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))


'''MAX_MEMORY = 100_000
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

            if len(state.shape) == 1:
                # (1, x)
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done,)

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

            # target = tf.Variable(target)
            # pred = tf.Variable(pred)
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
        # if len(self.replay_buffer) < self.batch_size:
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
            actor_loss = tf.keras.losses.categorical_crossentropy(actions_mask, current_state_expected_action_values,
                                                                  from_logits=False)
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
        self.steps = 0'''
