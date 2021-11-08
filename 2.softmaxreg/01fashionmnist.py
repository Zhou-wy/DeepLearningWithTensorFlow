# softmax运算符（softmax operator）解决了以上两个问题。它通过下式将输出值变换成值为正且和为1的概率分布

import sys
import time

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

# 获取数据集
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print (len (x_train), len (x_test))


def get_fashion_mnist_labels(labels):
    '''
    Fashion-MNIST中一共包括了10个类别，分别为
    t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、
    coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。
    以下函数可以将数值标签转成相应的文本标签。
    '''
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int (i)] for i in labels]


def show_fashion_mnist(images, labels, nrow, ncol):
    """
    :param images:
    :type images:
    :param labels:
    :type labels:
    :param nrow: 行数
    :type nrow: int
    :param ncol: 列数
    :type ncol: int
    """
    _, figs = plt.subplots (nrow, ncol, figsize=(12, 12))
    index = 0
    for i in range (nrow):
        for j in range (ncol):
            index = i * ncol + j
            figs[i][j].imshow (images[index].reshape (28, 28))
            figs[i][j].set_title (labels[index])
            figs[i][j].axes.get_xaxis ().set_visible (False)
            figs[i][j].axes.get_yaxis ().set_visible (False)
    plt.show ()


"""
X, y = [], []
for i in range(15):
    X.append(x_train[i])
    y.append(y_train[i])
show_fashion_mnist(X, get_fashion_mnist_labels(y),3,5)
"""
# 这里我们直接创建 tf.data.Dataset.from_tensor_slices 实例。
# 该实例每次读取一个样本数为batch_size的小批量数据。这里的批量大小batch_size是一个超参数。

batch_size = 256
if sys.platform.startswith ("win"):
    num_works = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_works = 4

train_iter = tf.data.Dataset.from_tensor_slices ((x_train, y_train)).batch (batch_size)

start = time.time ()
for X, y in train_iter:
    continue
print ('%.2f sec' % (time.time () - start))
