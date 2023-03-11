import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_probability as tfp


class CNNRLModel(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.cnn1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(288, 512, 3))
        self.max1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.cnn2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.max2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flat1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.action = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, state):
        state = tf.convert_to_tensor(state)
        if len(state.shape) < 4:
            state = tf.expand_dims(state, 0)
        x = tf.cast(state, tf.float32)
        x = self.cnn1(x)
        x = self.max1(x)
        x = self.cnn2(x)
        x = self.max2(x)
        x = self.flat1(x)
        x = self.dense1(x)
        x = self.action(x)
        return x

