import tensorflow as tf
from tensorflow import keras

"""
@author: Erik Dale
@date: 21.03.23
"""


class CNNRLModel(keras.Model):
    """
    Convolutional Neural Network model.
    :param num_actions: number of actions to output
    """

    def __init__(self, num_actions):
        super().__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=42)
        self.cnn1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', input_shape=(84, 84, 1), kernel_initializer=initializer)
        self.cnn2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializer)
        self.cnn3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer=initializer)
        self.flat1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.action = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, state, training=True):
        """
        Perform forward pass on the model with given state input.
        :param state: input state
        :return: probabilities of each action
        """
        state = tf.convert_to_tensor(state)
        # convert to right shape
        if len(state.shape) < 4:
            state = tf.expand_dims(state, 0)
        x = tf.cast(state, tf.float32)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.flat1(x)
        x = self.dense1(x)
        x = self.action(x)
        return x


class CNNRLModel2(keras.Model):
    """
    Convolutional Neural Network model.
    :param num_actions: number of actions to output
    """

    def __init__(self, num_actions):
        super().__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=42)
        self.cnn1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', input_shape=(84, 84, 4), kernel_initializer=initializer)
        self.cnn2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializer)
        self.cnn3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer=initializer)
        self.flat1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.action = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, state, training=True):
        """
        Perform forward pass on the model with given state input.
        :param state: input state
        :return: probabilities of each action
        """
        state = tf.convert_to_tensor(state)
        if len(state.shape) < 4:
            state = tf.expand_dims(state, 0)
        x = tf.cast(state, tf.float32)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.flat1(x)
        x = tf.reshape(x, (-1, 12544))

        x = self.dense1(x)
        x = self.action(x)

        if len(state.shape) == 4:
            x = tf.reduce_mean(x, axis=0, keepdims=True)

        return x



class DNNModel(keras.Model):
    """
    Deep neural network (DNN) model for a reinforcement learning agent.
    :param num_actions: number of actions to output
    """

    def __init__(self, num_actions):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(13,))
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(32, activation='relu')
        self.action = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, state):
        """
        Perform forward pass on the model with given state input.
        :param state: input state
        :return: probabilities of each action
        """
        state = tf.convert_to_tensor(state)
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.action(x)
        return x


class ActorModel(tf.keras.Model):
    """
    ActorModel class defines the actor model.
    :param num_actions: number of possible actions the agent can take
    """

    def __init__(self, num_actions):
        super(ActorModel, self).__init__()
        # Actor model
        self.actor_conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.actor_conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.actor_conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.actor_flatten = tf.keras.layers.Flatten()
        self.actor_fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.actor_output = tf.keras.layers.Dense(num_actions, activation='softmax', name='actor_output')

    def call(self, inputs):
        """
        Perform forward pass on the actor model with given state input.
        :param state: input state
        :return: probabilities of each action
        """
        # Unpack inputs
        state = inputs

        # Actor model
        x = self.actor_conv1(state)
        x = self.actor_conv2(x)
        x = self.actor_conv3(x)
        x = self.actor_flatten(x)
        x = self.actor_fc1(x)
        actor_output = self.actor_output(x)

        action1_prob = actor_output[:, 0]
        action2_prob = actor_output[:, 1]

        actor_output = tf.stack([action1_prob, action2_prob], axis=1)

        return actor_output


class CriticModel(tf.keras.Model):
    """
    CriticModel class defines the critic model.
    """
    def __init__(self):
        super(CriticModel, self).__init__()
        # Critic model
        self.critic_conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.critic_conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.critic_conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.critic_flatten = tf.keras.layers.Flatten()
        self.critic_fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.critic_concat = tf.keras.layers.Concatenate()
        self.critic_output = tf.keras.layers.Dense(1, activation='linear', name='critic_output')

    def call(self, inputs):
        """
        Perform forward pass on the critic model with given state and action inputs.
        :param inputs: tuple of state and action inputs
        :return: critic value for given state and action inputs
        """
        # Unpack inputs
        state, action = inputs

        # Critic model
        y = self.critic_conv1(state)
        y = self.critic_conv2(y)
        y = self.critic_conv3(y)
        y = self.critic_flatten(y)
        y = self.critic_fc1(y)
        action = tf.reshape(action, [-1, len(action)])
        action = tf.transpose(action)
        action = tf.cast(action, dtype=tf.float32)
        y = self.critic_concat([y, action])
        critic_output = self.critic_output(y)

        return critic_output