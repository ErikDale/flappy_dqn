import tensorflow as tf
from tensorflow import keras



class CNNRLModel(keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.cnn1 = tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', input_shape=(288, 512, 3))
        self.cnn2 = tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu')
        self.cnn3 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu')
        self.flat1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.action = tf.keras.layers.Dense(num_actions, activation='sigmoid')

    def call(self, state):
        state = tf.convert_to_tensor(state)
        if len(state.shape) < 4:
            state = tf.expand_dims(state, 0)
        x = tf.cast(state, tf.float32)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        #print(x)
        x = self.flat1(x)
        #print(x)
        x = self.dense1(x)
        #print(x)
        x = self.action(x)
        #print(x)
        return x

