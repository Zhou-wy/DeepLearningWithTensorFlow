import tensorflow as tf
import tensorflow.keras as keras


"""
与AlexNet和LeNet一样，VGG网络由卷积层模块后接全连接层模块构成。
卷积层模块串联数个vgg_block，其超参数由变量conv_arch定义。该变量指定了每个VGG块里卷积层个数和输出通道数。
全连接模块则跟AlexNet中的一样。
现在我们构造一个VGG网络。它有5个卷积块，前2块使用单卷积层，而后3块使用双卷积层。第一块的输出通道是64，
之后每次对输出通道数翻倍，直到变为512。因为这个网络使用了8个卷积层和3个全连接层，所以经常被称为VGG-11。
"""


def vgg_block(num_convs, num_channels):
    blk = keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(keras.layers.Conv2D(num_channels, kernel_size = 3, padding = 'same', activation = 'relu'))
    blk.add(keras.layers.MaxPool2D(pool_size = 2, strides = 2))
    return blk


def vgg(conv_arch):
    net = keras.models.Sequential()
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    net.add(keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation = 'relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation = 'relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation = 'sigmoid')
    ]))
    return net


# 它有5个卷积块，前2块使用单卷积层，而后3块使用双卷积层。第一块的输出通道是64，
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

import numpy as np

class DataLoader():
    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = np.expand_dims(self.train_images.astype(np.float32)/255.0,axis=-1)
        self.test_images = np.expand_dims(self.test_images.astype(np.float32)/255.0,axis=-1)
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]

    def get_batch_train(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        #need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.train_images[index],224,224,)
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        #need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.test_images[index],224,224,)
        return resized_images.numpy(), self.test_labels[index]

batch_size = 128
data = DataLoader()
x_batch, y_batch = data.get_batch_train(batch_size)
print("x_batch shape:",x_batch.shape,"y_batch shape:", y_batch.shape)



def train_vgg():
    epochs = 5
    num_iter = data.num_train // batch_size
    for e in range(epochs):
        for n in range(num_iter):
            x_batch, y_batch = data.get_batch_train(batch_size)
            net.fit(x_batch, y_batch)
            if n % 20 == 0:
                net.save_weights("vgg_weights.h5")


optimizer = keras.optimizers.SGD(learning_rate = 0.1, momentum = 0., nesterov = False)

net.compile(optimizer = optimizer,
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])
x_batch, y_batch = data.get_batch_train(batch_size)
x_test_batch, y_test_batch = data.get_batch_test(batch_size)
train_vgg()