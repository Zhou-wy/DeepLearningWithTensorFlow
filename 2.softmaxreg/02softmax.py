# coding:UTF-8
import sys
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

batch_size = 256
x_train = tf.cast(x_train, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
x_test = tf.cast(x_test, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size = batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size = batch_size)


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels, nrows, ncols):
    print(len(images), len(labels))
    _, figs = plt.subplots(nrows, ncols, figsize = (12, 12))
    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            figs[i][j].imshow(tf.reshape(images[index], shape = (28, 28)).numpy())
            figs[i][j].set_title(labels[index])
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


def show_fashion_mnist_n(images, labels):
    # 这⾥的_表示我们忽略（不使⽤）的变量
    _, figs = plt.subplots(1, len(images), figsize = (12, 12))  # 这里注意subplot 和subplots 的区别
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(tf.reshape(img, shape = (28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


"""
x, y = [], []
for i in range(15):
    x.append(x_train[i])
    y.append(y_train[i])
show_fashion_mnist(x, get_fashion_mnist_labels(y), 3, 5)
"""

# 初始化模型参数
num_inputs = 784
num_output = 10
w = tf.Variable(tf.random.normal(shape = (num_inputs, num_output), mean = 0, stddev = 0.01, dtype = tf.float32))
b = tf.Variable(tf.zeros(shape = (num_output)), dtype = tf.float32)


# 实现 softmax 运算
def softmax(logits, axis = -1):
    # tf.reduce_mean函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
    return tf.exp(logits) / tf.reduce_mean(tf.exp(logits), axis, keepdims = True)


# 定义模型
def net(x):
    logits = tf.matmul(tf.reshape(x, shape = (-1, w.shape[0])), w) + b
    return softmax(logits)


# 定义损失函数
def cross_entropy(y_hat, y):
    y = tf.cast(tf.reshape(y, shape = [-1, 1]), dtype = tf.int32)  # tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换
    y = tf.one_hot(y, depth = y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape = [-1, y_hat.shape[-1]]), dtype = tf.int32)
    return -tf.math.log(tf.boolean_mask(y_hat, y) + 1e-8) #tf.boolean_mask 的作用是 通过布尔值 过滤元素


# 计算分类准确率
def accuracy(y_hat, y):
    # 相等条件判断式(tf.argmax(y_hat, axis=1) == y)是一个数据类型为bool的Tensor，实际取值为：0（相等为假）或 1（相等为真）。
    return np.mean(np.argmax(y_hat, axis = 1) == y)


# 评 价模型net在数据集data_iter上的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0, 0
    for _, (x, y) in enumerate(data_iter):
        y = tf.cast(y, dtype = tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(x), axis = 1), dtype = tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n


print(evaluate_accuracy(test_iter, net))

num_epochs, lr = 10, 0.1


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params = None, lr = None, trainer = None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for x, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(x)
                l = tf.reduce_sum(loss(y_hat, y))
            grads = tape.gradient(l, params)
            if trainer == None:
                for i, param in enumerate(params):
                    param.assign_sub(lr * grads[i] / batch_size)
            else:
                trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))

            y = tf.cast(y, dtype = tf.float32)
            train_l_sum += l.numpy()
            train_acc_sum += tf.reduce_sum(
                tf.cast(tf.argmax(y_hat, axis = 1) == tf.cast(y, dtype = tf.int64), dtype = tf.int64)).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
        epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


trainer = tf.keras.optimizers.SGD(lr)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [w, b], lr)

X, y = iter(test_iter).next()
print(X.shape, y.shape)
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(tf.argmax(net(X), axis = 1).numpy())

titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

# show_fashion_mnist_n(X[0:5], titles[0:5])
show_fashion_mnist(X[0:15], titles[0:15], 3, 5)
