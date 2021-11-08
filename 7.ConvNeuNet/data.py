import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DataLoader:
    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = np.expand_dims(self.train_images.astype(np.float32) / 255.0, axis = -1)
        self.test_images = np.expand_dims(self.test_images.astype(np.float32) / 255.0, axis = -1)
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]

    def get_batch_train(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        # need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.train_images[index], 96, 96, )
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        # need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.test_images[index], 96, 96, )
        return resized_images.numpy(), self.test_labels[index]


data = DataLoader()
x_train, y_train = data.get_batch_train(128)


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle']
    return [text_labels[int(i)] for i in labels]


print(get_fashion_mnist_labels(y_train[0:15]))


def show_fashion_mnist(images, labels, nrow, ncol):
    _, figs = plt.subplots(nrow, ncol)
    for i in range(nrow):
        for j in range(ncol):
            index = i * ncol + j
            figs[i][j].imshow(images[index])
            figs[i][j].set_title(labels[index])
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


show_fashion_mnist(x_train, get_fashion_mnist_labels(y_train[0:15]), 3, 5)

