import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

print(tf.__version__)
x = tf.random.uniform((2, 20), minval = 0, maxval = 10)


class CenteredLayer(keras.layers.Layer):
    def __init__(self):
        super(CenteredLayer, self).__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)


class myDense(keras.layers.Layer):
    def __init__(self, units):
        super(myDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name = 'w',
                                 shape = [input_shape[-1], self.units],
                                 initializer = tf.random_normal_initializer())

        self.b = self.add_weight(name = "b",
                                 shape = [self.units],
                                 initializer = tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred


dense = myDense(3)
dense(x)
print(dense.get_weights())
