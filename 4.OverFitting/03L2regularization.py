import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

n_train, n_test, num_inputs = 20, 100, 200
true_w, ture_b = tf.ones((num_inputs, 1)) * 0.01, 0.03

features = tf.random.normal(shape = (n_train + n_test, num_inputs))  # (120,200)
labels = keras.backend.dot(features, true_w) + ture_b
labels += tf.random.normal(mean = 0.01, shape = labels.shape)  # (120,1)
train_feature, test_feature = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


def init_parms():
    w = tf.Variable(tf.random.normal(mean = 1, shape = (num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape = (1,)))
    return [w, b]


def l2_penalty(w):
    return tf.reduce_sum(w ** 2) / 2


def net(X, w, b):
    return tf.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size, grads):
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i] / batch_size)


batch_size, num_epochs, lr = 1, 100, 0.003
optimizer = keras.optimizers.SGD()
train_iter = tf.data.Dataset.from_tensor_slices((train_feature, train_labels)).batch(batch_size).shuffle(batch_size)


def plot_show(x_vals, y_vals, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    l1 = ax.plot(x_vals, y_vals, 'ys-')  # solid line with yellow colour and square marker
    l2 = ax.plot(x_label, y_label, 'go--')  # dash line with green colour and circle marker
    ax.legend(labels = ('train loss', 'test loss'), loc = 'upper left')  # legend placed at lower right
    ax.set_title("Loss")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.show()


def fit(lambd):
    w, b = init_parms()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                l = squared_loss(net(X, w, b), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            sgd([w, b], lr, batch_size, grads)
        train_ls.append(tf.reduce_mean(squared_loss(net(train_feature, w, b), train_labels)).numpy())
        test_ls.append(tf.reduce_mean(squared_loss(net(test_feature, w, b), test_labels)).numpy())
    plot_show(range(1, num_epochs + 1), train_ls,
              range(1, num_epochs + 1), test_ls)
    print('L2 norm of w:', tf.norm(w).numpy())


fit(5)
