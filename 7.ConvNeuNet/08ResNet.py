import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

class Residual(keras.Model):
    def __init__(self, num_channels, use_1T1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(num_channels, padding="same", kernel_size=3, strides=strides)
        self.conv2 = keras.layers.Conv2D(num_channels, kernel_size=3, padding="same")
        if use_1T1conv:
            self.conv3 = keras.layers.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, X, **kwargs):
        Y = keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return keras.activations.relu(Y + X)


"""
blk = Residual(3)
#tensorflow input shpe     (n_images, x_shape, y_shape, channels).
#mxnet.gluon.nn.conv_layers    (batch_size, in_channels, height, width)
X = tf.random.uniform((4, 6, 6 , 3))
print(blk(X).shape)
#TensorShape([4, 6, 6, 3])
"""

net = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
])


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.listLayers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.listLayers.append(Residual(num_channels, use_1T1conv=True, strides=2))
            else:
                self.listLayers.append(Residual(num_channels))

    def call(self, X, **kwargs):
        for layer in self.listLayers.layers:
            X = layer(X)
        return X


class ResNet(tf.keras.Model):
    def __init__(self, num_blocks, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.conv = keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.bn = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        self.mp = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.resnet_block1 = ResnetBlock(64, num_blocks[0], first_block=True)
        self.resnet_block2 = ResnetBlock(128, num_blocks[1])
        self.resnet_block3 = ResnetBlock(256, num_blocks[2])
        self.resnet_block4 = ResnetBlock(512, num_blocks[3])
        self.gap = keras.layers.GlobalAvgPool2D()
        self.fc = keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.resnet_block4(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


"""
X = tf.random.uniform(shape=(1, 224, 224, 1))
for layer in mynet.layers:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
"""


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.2)
    plt.show()


t1 = time.time()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
mynet = ResNet([2, 2, 2, 2])
mynet.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
history = mynet.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)
plot_learning_curves(history)
test_scores = mynet.evaluate(x_test, y_test, verbose=2)
t2 = time.time()
print("训练所用时间：%3.5f秒" % (t2 - t1))
