from models import DNNModel, CNNRLModel, CriticModel, ActorModel, CNNRLModel2
import tensorflow as tf
import numpy as np
from collections import deque
import random
import math

"""
@author: Erik Dale
@date: 21.03.23
"""


class Agent:
    """
    Agent that learns to interact with an environment.
    :param gamma: the discount factor for future rewards. Default is 0.95.
    :param lr: the learning rate for the optimizer. Default is 0.00001.
    :param n_actions: the number of actions available to the agent. Default is 2.
    :param cnn_model: a boolean indicating whether to use a convolutional neural network as the model. Default is True.
    :param replay_buffer_size: the maximum size of the replay buffer. Default is 20000.
    """
    def __init__(self, gamma=0.95, lr=0.0001, n_actions=2, cnn_model=True, replay_buffer_size=20000, batch_size=32):
        """
        Initializes a new Agent object.
        """
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.gamma = gamma
        self.lr = lr
        if cnn_model:
            self.target_model = CNNRLModel(n_actions)
            self.behavior_model = CNNRLModel(n_actions)
            self.model = CNNRLModel(n_actions)
        else:
            self.model = DNNModel(n_actions)

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        #self.opt = tf.keras.optimizers.experimental.RMSprop(learning_rate=self.lr, momentum=0.95, weight_decay=0.9)
        self.n_actions = n_actions
        self.gradient_clip_norm = 3
        self.batch_size = batch_size

    def choose_action(self, state, model):
        """
        Generate an action using the current policy.
        :param state: the current state of the environment.
        :return: the chosen action.
        """
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        actor_output = model(state)  # Compute the output of the model
        actor_output = tf.squeeze(actor_output)  # Remove the batch dimension
        actor_output = actor_output - tf.reduce_max(actor_output)  # Subtract the maximum value for numerical stability
        exp_actor_output = tf.exp(actor_output) + 1e-2  # Exponentiate and add a small epsilon value for non-zero
        # probabilities
        policy = exp_actor_output / tf.reduce_sum(exp_actor_output)  # Compute the policy by normalizing the
        # probabilities
        if math.isnan(policy[0]) or math.isnan(policy[1]):
            action = 0  # If policy contains NaN values, choose default action (0)
        else:
            # action = np.random.choice(self.n_actions, p=policy.numpy())
            action = tf.argmax(policy)  # Choose the action with the highest probability
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

    def learn2(self, target_train):
        # Find a batch size that is 10% of the whole replay buffer.
        # self.batch_size = int(len(self.replay_buffer) * 0.1)

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

        # Compute target Q values using the target network
        target_actions = self.target_model(next_states)
        target_actions = tf.squeeze(target_actions)
        target_values = tf.reduce_max(target_actions, axis=1)
        target_values = (1 - dones) * self.gamma * target_values + rewards

        # Compute loss and update weights
        with tf.GradientTape() as tape:
            # Forward pass
            predicted_actions = self.behavior_model(states)
            predicted_actions = tf.squeeze(predicted_actions)
            predicted_values = tf.reduce_sum(predicted_actions * tf.one_hot(actions, self.n_actions), axis=1)

            # Compute loss
            loss = tf.keras.losses.mean_squared_error(target_values, predicted_values)

        # Compute gradients and apply to optimizer
        grads = tape.gradient(loss, self.behavior_model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.gradient_clip_norm)
        self.opt.apply_gradients(zip(grads, self.behavior_model.trainable_variables))

        if target_train:
            # Update the target network with the behavior network's weights
            for target, behavior in zip(self.target_model.trainable_variables, self.behavior_model.trainable_variables):
                target.assign(behavior)

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
        target_actions = self.model(next_states)  # Compute the output of the model for the next states
        target_actions = tf.squeeze(target_actions)  # Remove dimensions of size 1
        target_actions = tf.reduce_max(target_actions, axis=1)  # Compute the maximum action value for each sample
        target_values = rewards + self.gamma * target_actions * (1 - dones)  # Compute the target values

        # Compute loss and update weights
        with tf.GradientTape() as tape:
            # Forward pass
            predicted_actions = self.model(states)  # Compute the output of the model for the current states
            predicted_actions = tf.squeeze(predicted_actions)  # Remove dimensions of size 1
            predicted_values = tf.reduce_sum(predicted_actions * actions, axis=1)  # Compute the predicted values

            # Compute loss
            loss = tf.keras.losses.mean_squared_error(target_values, predicted_values)  # Compute the mean squared
            # error loss

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)  # Compute the gradients of the loss with
        # respect to the model's trainable variables
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)  # Clip the gradients to prevent
        # gradient explosion
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))  # Apply the gradients to update the
        # model's weights


class ActorCriticAgent(Agent):
    """
    Actor-Critic agent that learns to interact with an environment. Inherits from the Agent class
    :param gamma: the discount factor for future rewards. Default is 0.95.
    :param n_actions: the number of actions available to the agent. Default is 2.
    :param replay_buffer_size: the maximum size of the replay buffer. Default is 20000.
    :param batch_size: the size of the minibatch sampled from the replay buffer during training. Default is 32.
    :param actor_learning_rate: the learning rate for the actor model. Default is 0.0001.
    :param critic_learning_rate: the learning rate for the critic model. Default is 0.001.
    """
    def __init__(self, gamma=0.95, n_actions=2, replay_buffer_size=20000, batch_size=32,
                 actor_learning_rate=0.0001, critic_learning_rate=0.001):
        """
        Initialize the actor critic agent.
        """
        super().__init__(gamma, n_actions=n_actions, replay_buffer_size=replay_buffer_size, batch_size=batch_size)
        self.batch_size = batch_size
        self.gradient_clip_norm = 1

        # Initialize actor-critic model
        self.actor_model = ActorModel(n_actions)
        self.critic_model = CriticModel()

        # self.actor_optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate=actor_learning_rate, momentum=0.95, weight_decay=0.9)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)

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
            values = self.critic_model(inputs)  # Compute the critic values for the current states and actions

            inputs = (next_states, actions)
            next_values = self.critic_model(inputs)  # Compute the critic values for the next states and actions

            next_values = np.array(next_values)  # Convert the next values to a numpy array

            td_targets = rewards + self.gamma * next_values * (1 - dones)  # Compute the TD targets
            td_errors = td_targets - values  # Compute the TD errors
            critic_loss = tf.reduce_mean(tf.square(td_errors))  # Compute the mean squared TD error loss

        # Compute critic gradients and apply to optimizer
        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)  # Compute the gradients of
        # the critic loss with respect to the critic model's trainable variables
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.gradient_clip_norm)  # Clip the gradients to
        # prevent gradient explosion
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))  # Apply the
        # gradients to update the critic model's weights

        # Compute actor loss
        with tf.GradientTape() as tape:
            policy = self.actor_model(states)  # Compute the actor policy for the current states
            actions_one_hot = tf.one_hot(actions, self.n_actions)  # Convert the actions to one-hot encoding
            advantages = tf.stop_gradient(
                td_errors)  # Compute the advantages by stopping the gradient flow of TD errors
            actor_loss = -tf.reduce_mean(
                tf.reduce_sum(actions_one_hot * tf.math.log(policy + 1e-10), axis=1) * advantages
            )  # Compute the actor loss using the policy and advantages

        # Compute actor gradients and apply to optimizer
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)  # Compute the gradients of the
        # actor loss with respect to the actor model's trainable variables
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.gradient_clip_norm)  # Clip the gradients to
        # prevent gradient explosion
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))  # Apply the
        # gradients to update the actor model's weights
