import tensorflow as tf
import tensorflow.keras as keras
import os
import numpy as np
import matplotlib.pyplot as plt
from data import DataLoader


class Inception(keras.layers.Layer):
    def __init__(self, c1, c2, c3, c4):
        super(Inception, self).__init__()
        """
        :param c1: 线路1 ：单个卷积层通道数
        :param c2: 线路2 ：两个卷积层，一层卷积层通道数c2[0],二层卷积层通道数c2[1]
        :param c3: 线路3 ：两个卷积层，一层卷积层通道数c3[0],二层卷积层通道数c3[1]
        :param c4: 线路4 ：最大池化后，卷积层的通道数
        """
        # 线路1 ： 1 * 1 卷积层
        self.p1_1 = keras.layers.Conv2D(c1, kernel_size = 1, padding = "same", activation = "relu")
        # 线路2 ： 1 * 1 卷积层 + 3 * 3 卷积层
        self.p2_1 = keras.layers.Conv2D(c2[0], kernel_size = 1, padding = "same", activation = "relu")
        self.p2_2 = keras.layers.Conv2D(c2[1], kernel_size = 3, padding = "same", activation = "relu")
        # 线路3 ： 1 * 1 卷积层 + 5 * 5 卷积层
        self.p3_1 = keras.layers.Conv2D(c3[0], kernel_size = 1, activation = "relu")
        self.p3_2 = keras.layers.Conv2D(c3[1], kernel_size = 5, padding = "same", activation = "relu")
        # 线路4 ： 3 * 3 最大池化层 + 1 * 1 卷积层
        self.p4_1 = keras.layers.MaxPool2D(pool_size = 3, strides = 1, padding = "same")
        self.p4_2 = keras.layers.Conv2D(c4, kernel_size = 1, padding = "same", activation = "relu")

    def call(self, x, *args, **kwargs):
        p1 = self.p1_1(x)
        # print("p1.shape:", p1.shape)
        p2 = self.p2_2(self.p2_1(x))
        # print("p2.shape:", p2.shape)
        p3 = self.p3_2(self.p3_1(x))
        # print("p3.shape:", p2.shape)
        p4 = self.p4_2(self.p4_1(x))
        # print("p4.shape:", p4.shape)
        return tf.concat((p1, p2, p3, p4), axis = -1)


"""
GoogLeNet跟VGG一样，在主体卷积部分中使用5个模块（block），
每个模块之间使用步幅为2的3×3最大池化层来减小输出高宽。
第一模块使用一个64通道的7×7卷积层。
"""
# 第一部分#
b1 = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size = 7, strides = 2, padding = "same", activation = "relu"),
    keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = "same")
])

"""
第二模块使用2个卷积层：首先是64通道的1×1卷积层，
然后是将通道增大3倍的3×3卷积层。它对应Inception块中的第二条线路。
"""
b2 = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size = 1, padding = "same", activation = "relu"),
    keras.layers.Conv2D(64 * 3, kernel_size = 3, padding = "same", activation = "relu"),
    keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = "same")
])

"""
第三模块串联2个完整的Inception块。
第一个Inception块的输出通道数为64+128+32+32=256
其中4条线路的输出通道数比例为64:128:32:32=2:4:1:1 
其中第二、第三条线路先分别将输入通道数减小至96/192=1/2和16/192=1/12后，
再接上第二层卷积层。第二个Inception块输出通道数增至128+192+96+64=480
每条线路的输出通道数之比为128:192:96:64=4:6:3:2
其中第二、第三条线路先分别将输入通道数减小至128/256=1/2和32/256=1/8。
"""
b3 = tf.keras.models.Sequential([
    Inception(64, (96, 128), (16, 32), 32),
    Inception(128, (128, 192), (32, 96), 64),
    keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = 'same')
])

"""
第四模块更加复杂。它串联了5个Inception块，
其输出通道数分别是192+208+48+64=512、160+224+64+64=512、128+256+64+64=512、112+288+64+64=528和256+320+128+128=832。
这些线路的通道数分配和第三模块中的类似，首先含3×3卷积层的第二条线路输出最多通道，
其次是仅含1×11×1卷积层的第一条线路，之后是含5×55×5卷积层的第三条线路和含3×33×3最大池化层的第四条线路。
其中第二、第三条线路都会先按比例减小通道数。这些比例在各个Inception块中都略有不同。
"""
b4 = keras.models.Sequential([
    Inception(192, (96, 208), (16, 48), 64),
    Inception(160, (112, 224), (24, 64), 64),
    Inception(128, (128, 256), (24, 64), 64),
    Inception(112, (144, 288), (32, 64), 64),
    Inception(256, (160, 320), (32, 128), 128),
    keras.layers.MaxPool2D(pool_size = 3, strides = 2, padding = "same")
])
"""
第五模块有输出通道数为256+320+128+128=832和384+384+128+128=1024的两个Inception块。
其中每条线路的通道数的分配思路和第三、第四模块中的一致，只是在具体数值上有所不同。
需要注意的是，第五模块的后面紧跟输出层，该模块同NiN一样使用全局平均池化层来将每个通道的高和宽变成1。
后我们将输出变成二维数组后接上一个输出个数为标签类别数的全连接层。
"""
b5 = keras.models.Sequential([
    Inception(256, (160, 320), (32, 128), 128),
    Inception(384, (192, 384), (48, 128), 128),
    keras.layers.GlobalAveragePooling2D()
])

net = tf.keras.models.Sequential([b1, b2, b3, b4, b5, tf.keras.layers.Dense(10)])

"""X = tf.random.uniform(shape = (1, 96, 96, 1))
for layer in net.layers:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)"""

lr, num_epochs, batch_size = 0.5, 10, 128
dataLoader = DataLoader()
x_train, y_train = dataLoader.get_batch_train(batch_size)
x_test, y_test = dataLoader.get_batch_test(batch_size)

import datetime
import os

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join('data', 'autograph', stamp)

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq = 1)

net.compile(
    optimizer = keras.optimizers.SGD(learning_rate = lr),
    # optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = tf.keras.losses.binary_crossentropy,
    metrics = ["accuracy"]
)

history = net.fit(x_train, y_train, epochs = 10, batch_size = 1, validation_data = (x_test, y_test),
                  callbacks = [tensorboard_callback], workers = 4)
