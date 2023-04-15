import tensorflow as tf
from tensorflow import keras

"""
@author: Erik Dale
@date: 21.03.23
"""

class CNNRLModel(keras.Model):
    """
    Convolutional Neural Network model. It takes the number of actions as input
    """
    def __init__(self, num_actions, dropout_rate=0.5):
        super().__init__()
        self.cnn1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', input_shape=(84, 84, 1))
        self.cnn2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
        self.cnn3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
        self.flat1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.action = tf.keras.layers.Dense(num_actions, activation='sigmoid')

    def call(self, state, training=True):
        state = tf.convert_to_tensor(state)
        if len(state.shape) < 4:
            state = tf.expand_dims(state, 0)
        x = tf.cast(state, tf.float32)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.flat1(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.action(x)
        return x



class DNNModel(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(13,))
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(32, activation='relu')
        self.action = tf.keras.layers.Dense(num_actions, activation='sigmoid')

    def call(self, state):
        state = tf.convert_to_tensor(state)
        #if len(state.shape) < 4:
        #    state = tf.expand_dims(state, 0)
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.action(x)
        return x


class ActorCriticModel(tf.keras.Model):
    """
    Actor-Critic model that takes an image of shape (84, 84, 1) as input and outputs an action and the expected action value
    """
    def __init__(self, num_actions):
        super().__init__()
        self.cnn1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', input_shape=(84, 84, 1))
        self.cnn2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
        self.cnn3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
        self.flat1 = tf.keras.layers.Flatten()
        self.dense_actor = tf.keras.layers.Dense(512, activation='relu')
        self.actor_output = tf.keras.layers.Dense(num_actions, activation='softmax')
        self.dense_critic = tf.keras.layers.Dense(512, activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        self.out = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, state):
        state = tf.convert_to_tensor(state)
        if len(state.shape) < 4:
            state = tf.expand_dims(state, 0)
        x = tf.cast(state, tf.float32)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.flat1(x)
        x_actor = self.dense_actor(x)
        actor_output = self.actor_output(x_actor)
        x_critic = self.dense_critic(x)
        x_concat = self.concat([x_critic, x_actor])
        output = self.out(x_concat)
        return actor_output, output


class ActorCriticModel2(tf.keras.Model):
    def __init__(self, num_actions):
        super(ActorCriticModel2, self).__init__()
        # Actor model
        self.actor_conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.actor_conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.actor_conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.actor_flatten = tf.keras.layers.Flatten()
        self.actor_fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.actor_output = tf.keras.layers.Dense(num_actions, activation='softmax', name='actor_output')

        # Critic model
        self.critic_conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.critic_conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.critic_conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.critic_flatten = tf.keras.layers.Flatten()
        self.critic_fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.critic_concat = tf.keras.layers.Concatenate()
        self.critic_output = tf.keras.layers.Dense(1, activation='linear', name='critic_output')

    def call(self, inputs):
        if len(inputs) == 2:
            # Unpack inputs
            state, action = inputs

            # Actor model
            x = self.actor_conv1(state)
            x = self.actor_conv2(x)
            x = self.actor_conv3(x)
            x = self.actor_flatten(x)
            x = self.actor_fc1(x)
            actor_output = self.actor_output(x)

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
            return actor_output, critic_output

        else:
            state = inputs
            # Actor model
            x = self.actor_conv1(state)
            x = self.actor_conv2(x)
            x = self.actor_conv3(x)
            x = self.actor_flatten(x)
            x = self.actor_fc1(x)
            actor_output = self.actor_output(x)
            return actor_output, None


class ActorModel(tf.keras.Model):
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






