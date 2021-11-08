import time

import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append("..")


def get_data():  # 本函数已保存在d2lzh_tensorflow2包中方便以后使用
    data = np.genfromtxt('./data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return tf.convert_to_tensor(data[:1500, :-1], dtype=tf.float32), tf.convert_to_tensor(data[:1500, -1],
                                                                                          dtype=tf.float32)


Features, Labels = get_data()
print(Features.shape)  #

def linreg(X, w, b):
    return tf.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2


def sgd(params, states, hyperparams, grads):
    for i, p in enumerate(params):
        p.assign_sub(hyperparams["lr"] * grads[i])

def train(optimizer_fn, states, hyperparams, features, labels, batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = linreg, squared_loss
    w  = tf.Variable(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=tf.float32)
    b = tf.Variable(tf.zeros(1, dtype=tf.float32))

    def eval_loss():
        return np.array(tf.reduce_mean(loss(net(features, w, b), labels)))

    ls = [eval_loss()]
    data_iter = tf.data.Dataset.from_tensors((features, labels)).batch(batch_size)
    data_iter = data_iter.shuffle(100)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with tf.GradientTape() as tape:
                l = tf.reduce_mean(loss(net(X, w, b), y))
            grads = tape.gradient(l, [w, b])
            optimizer_fn([w, b], states, hyperparams, grads)
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())

        # 打印结果
        print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

def train_sgd(lr, batch_size, num_epochs=2):
    train(sgd, None, {'lr': lr}, Features, Labels, batch_size, num_epochs)

#train_sgd(1, 1500, 6)
"""
trainer = keras.optimizers.SGD(learning_rate=0.05)
train('trainer', {'learning_rate': 0.05}, Features, Labels, 10)
"""
