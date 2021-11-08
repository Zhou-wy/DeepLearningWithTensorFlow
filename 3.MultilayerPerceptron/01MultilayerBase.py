import tensorflow as tf
import numpy as np

from tensorflow import keras
from matplotlib import pyplot as plt

# 导入数据
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 归一化
x_train = x_train / 255.0
x_test = x_test / 255.0
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(batch_size).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(batch_size).batch(batch_size)


# 获取标签 and 画图
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels, nrow, ncol):
    _, figs = plt.subplots(nrow, ncol, figsize = (12, 12))
    for i in range(nrow):
        for j in range(ncol):
            index = i * ncol + j
            figs[i][j].imshow(tf.reshape(images[index], shape=(28, 28)).numpy())
            figs[i][j].set_title(labels[index])
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


class MultiLayer:
    # 定义模型参数
    def __init__(self, n_inputs, n_outputs, n_hiddens):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hiddens = n_hiddens
        self.w1 = tf.Variable(
            tf.random.normal(shape = (self.n_inputs, self.n_hiddens), mean = 0, stddev = 0.01, dtype = tf.float32))
        self.b1 = tf.Variable(tf.zeros(shape = (self.n_hiddens), dtype = tf.float32))
        self.w2 = tf.Variable(
            tf.random.normal(shape = (self.n_hiddens, self.n_outputs), mean = 0, stddev = 0.01, dtype = tf.float32))
        self.b2 = tf.Variable(tf.random.normal([self.n_outputs], stddev = 0.01))

    def relu(self, x):
        return tf.math.maximum(x, 0)

    def net(self, X):
        X = tf.cast(tf.reshape(X, shape = [-1, self.n_inputs]), dtype = tf.float32)
        h1 = self.relu(tf.matmul(X, self.w1) + self.b1)
        h2 = tf.math.softmax(tf.matmul(h1, self.w2) + self.b2)
        return h2

    def loss_scc(self, y_hat, y_true):
        return tf.losses.sparse_categorical_crossentropy(y_true, y_hat)

    def evaluate_accuracy(self, data_iter, net):
        acc_sum, n = 0.0, 0
        for _, (X, y) in enumerate(data_iter):
            y = tf.cast(y, dtype = tf.int64)
            acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis = 1), dtype = tf.int64) == y)
            n += y.shape[0]
        return acc_sum / n

    def train(self, train_iter, test_iter, n_epochs, params, lr, trainer = None):
        for epoch in range(n_epochs):
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in train_iter:
                with tf.GradientTape() as tape:
                    y_hat = self.net(X)
                    l = tf.reduce_sum(self.loss_scc(y_hat = y_hat, y_true = y))
                grads = tape.gradient(l, params)
                if trainer == None:
                    # 如果没有传入优化器，则使用原先编写的小批量随机梯度下降
                    for i, param in enumerate(params):
                        param.assign_sub(lr * grads[i] / batch_size)
                else:
                    trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))
                y = tf.cast(y, dtype = tf.float32)
                train_l_sum += l.numpy()
                train_acc_sum += tf.reduce_mean(
                    tf.cast(tf.argmax(y_hat, axis = 1) == tf.cast(y, dtype = tf.int64), dtype = tf.int64)).numpy()
                n += y.shape[0]
            test_acc = self.evaluate_accuracy(test_iter, self.net)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
                epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

    def plot_learning_curves(history):
        pd.DataFrame(history.history).plot(figsize = (8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.show()


n_inputs, n_outputs, n_hiddens = 784, 10, 256
f1 = MultiLayer(n_inputs, n_outputs, n_hiddens)
f1.train(train_iter, test_iter, 10, [f1.w1, f1.b1, f1.w2, f1.b1], 0.1)

X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(tf.argmax(f1.net(X), axis=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:15], titles[0:15],3,5)