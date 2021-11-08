import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import matplotlib.pyplot as plt

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
        resized_images = tf.image.resize_with_pad(self.train_images[index], 224, 224, )
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        # need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.test_images[index], 224, 224, )
        return resized_images.numpy(), self.test_labels[index]


def nin_block(num_channels, kernel_size, strides, padding):
    blk = tf.keras.models.Sequential()
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size,
                                   strides=strides, padding=padding, activation='relu'))
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=1,activation='relu'))
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=1,activation='relu'))
    return blk



net = tf.keras.models.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding='valid'))
net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
net.add(nin_block(256, kernel_size=5, strides=1, padding='same'))
net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
net.add(nin_block(384, kernel_size=3, strides=1, padding='same'))
net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2))
net.add(tf.keras.layers.Dropout(0.5))
net.add(nin_block(10, kernel_size=3, strides=1, padding='same'))
net.add(tf.keras.layers.GlobalAveragePooling2D())
net.add(tf.keras.layers.Flatten())


X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.01, 30, 128
data = DataLoader()
x_train, y_train = data.get_batch_train(batch_size)
x_test, y_test = data.get_batch_test(batch_size)
print(x_train.shape, y_train.shape)
print(type(x_train), type(y_train))

t1 = time.time()
optimizer = keras.optimizers.Adam(learning_rate = 0.001)
net.compile(optimizer = optimizer, loss = keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
net.build(input_shape=(None, 224, 224, 1))
his = net.fit(x_train, y_train, epochs = 30, batch_size = 1, validation_data = (x_test, y_test))
t2 = time.time()
print("训练所耗时间：%5.3f" % (t2 - t1))
net.save_weights("nin_weights.h5")
