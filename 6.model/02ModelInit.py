import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras

print(tf.__version__)

'''
 对于使用Sequential类构造的神经网络，我们可以通过weights属性来访问网络任一层的权重。
 回忆一下上一节中提到的Sequential类与tf.keras.Model类的继承关系。 对于Sequential
 实例中含模型参数的层，我们可以通过tf.keras.Model类的weights属性来访问该层包含的所有参数。
'''

class Linear(tf.keras.Model):
    def __init__(self):
        super(Linear, self).__init__()
        self.d1 = tf.keras.layers.Dense(
            units = 10,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01),
            bias_initializer = tf.zeros_initializer()
        )

        self.d2 = tf.keras.layers.Dense(
            units = 1,
            activation = None,
            kernel_initializer = tf.ones_initializer(),
            bias_initializer = tf.ones_initializer()
        )

    def call(self, inputs):
        output = self.d1(inputs)
        output = self.d2(output)
        return output


X = tf.random.uniform((2, 20), minval = 0, maxval = 10, dtype = tf.int32)
net = Linear()
print(net(X))
print(net.get_weights())


