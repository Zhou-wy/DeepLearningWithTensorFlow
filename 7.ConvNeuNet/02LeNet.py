import tensorflow as tf
import tensorflow.keras as keras
import os
import pandas as pd
from matplotlib import pyplot as plt

# 数据集#
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
train_images = tf.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
test_images = tf.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices((train_images, y_train)).batch(batch_size).shuffle(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((test_images, y_test)).batch(batch_size).shuffle(batch_size)


# 获取标签 and 画图
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels, nrow, ncol):
    _, figs = plt.subplots(nrow, ncol, figsize = (nrow * 1.8, ncol * 1.2))
    for i in range(nrow):
        for j in range(ncol):
            index = i * ncol + j
            figs[i][j].imshow(tf.reshape(images[index], (28, 28)).numpy())
            figs[i][j].set_title(labels[index])
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


net = keras.models.Sequential()
net.add(keras.layers.Conv2D(filters = 6, kernel_size = 5, activation = "sigmoid", input_shape = (28, 28, 1)))
net.add(keras.layers.MaxPool2D(pool_size = 2, strides = 2))
net.add(keras.layers.Conv2D(filters = 16, kernel_size = 5, activation = "sigmoid"))
net.add(keras.layers.MaxPool2D(pool_size = 2, strides = 2))
net.add(keras.layers.Flatten())
net.add(keras.layers.Dense(120, activation = "sigmoid"))
net.add(keras.layers.Dense(64, activation = "sigmoid"))
net.add(keras.layers.Dense(10, activation = "sigmoid"))

optimizer = keras.optimizers.SGD(learning_rate = 0.5, momentum = 0.0, nesterov = 0.0)
net.compile(optimizer = optimizer, loss = keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
his = net.fit(train_iter, epochs = 20, batch_size = batch_size, validation_data = test_iter)


# net.evaluate(test_iter)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(his)
