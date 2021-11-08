import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import math
import time
from data import data
import Plot

ContData = data.ControlData()
tureData = ContData.TureData()
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = ContData.load_data()
# Keras的Rnn模块提供了循环神经网络的实现。下面构造一个含单隐藏层、隐藏单元个数为256的循环神经网络层rnn_layer，并对权重做初始化。
num_hiddens = 256
cell = keras.layers.SimpleRNNCell(num_hiddens, kernel_initializer = "glorot_uniform")
rnn_layer = keras.layers.RNN(cell, time_major = True, return_sequences = True, return_state = True)
# 输出形状为(时间步数, 批量大小, 隐藏单元个数)，隐藏状态h的形状为(层数, 批量大小, 隐藏单元个数)。
batch_size = 2
state = cell.get_initial_state(batch_size = batch_size, dtype = tf.float32)

num_steps = 35
X = tf.random.uniform(shape = (num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)


class RNNModel(keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def get_initial_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)


def predict_rnn_keras(prefix, num_chars, model, vocab_size, idx_to_char,
                      char_to_idx):
    # 使用model的成员函数来初始化隐藏状态
    state = model.get_initial_state(batch_size = 1, dtype = tf.float32)
    output = [char_to_idx[prefix[0]]]
    re = []
    # print("output:",output)
    for t in range(num_chars + len(prefix) - 1):
        X = np.array([output[-1]]).reshape((1, 1))
        # print("X",X)
        Y, state = model(X, state)  # 前向计算不需要传入模型参数
        # print("Y",Y)
        # print("state:",state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
            # print(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(np.array(tf.argmax(Y, axis = -1))))
            # print(int(np.array(tf.argmax(Y[0],axis=-1))))
    re.append([idx_to_char[i] for i in output])
    return re[0]


"""
model = RNNModel(rnn_layer, vocab_size)
print(predict_rnn_keras('分开', 10, model, vocab_size, idx_to_char, char_to_idx))
"""


# 计算裁剪后的梯度
def grad_clipping(grads, theta):
    norm = np.array([0])
    for i in range(len(grads)):
        norm += tf.math.reduce_sum(grads[i] ** 2)
    # print("norm",norm)
    norm = np.sqrt(norm).item()
    new_gradient = []
    if norm > theta:
        for grad in grads:
            new_gradient.append(grad * theta / norm)
    else:
        for grad in grads:
            new_gradient.append(grad)
    # print("new_gradient",new_gradient)
    return new_gradient


def train_and_predict_rnn_keras(model, num_hiddens, vocab_size, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate = lr)

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = ContData.data_iter_consecutive(corpus_indices, batch_size, num_steps)
        state = model.get_initial_state(batch_size = batch_size, dtype = tf.float32)
        for X, Y in data_iter:
            with tf.GradientTape(persistent = True) as tape:
                (outputs, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(y, outputs)

            grads = tape.gradient(l, model.variables)
            # 梯度裁剪
            grads = grad_clipping(grads, clipping_theta)
            optimizer.apply_gradients(zip(grads, model.variables))  # 因为已经误差取过均值，梯度不用再做平均
            l_sum += np.array(l).item() * len(y)
            n += len(y)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                predict = predict_rnn_keras(
                    prefix, pred_len, model, vocab_size, idx_to_char,
                    char_to_idx)
                print(' -> ', predict, "\n")
    return predict

"""
num_epochs, batch_size, lr, clipping_theta = 250, 16, 1e2, 1e-2
pred_period, pred_len, prefixes = 100, 100, [
    [0.9999997930212569, 0.9999997656385977, 0.9999997403897148, 0.9999997172561449, 0.9999996962131402,
     0.9999996772300609, 0.9999996602707657, 0.9999996452940042, 0.9999996322538063, 0.9999996210998685,
     0.9999996117779381, 0.999999604230188, 0.9999995983955896, 0.9999995942102761, 0.9999995916078974,
     0.9999995905199677, 0.9999995908762017, 0.9999995926048424, 0.999999595632975, 0.9999995998868328,
     0.9999996052920886, 0.9999996117741351, 0.9999996192583512, 0.9999996276703567, 0.9999996369362534,
     0.9999996469828518, 0.9999996577378863, 0.9999996691302143, 0.9999996810900043, 0.9999996935489076]]
model = RNNModel(rnn_layer, vocab_size)
pData = train_and_predict_rnn_keras(model, num_hiddens, vocab_size, corpus_indices, idx_to_char, char_to_idx,
                                    num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len,
                                    prefixes)

predData = tureData + pData
print(type(tureData), len(tureData), "\n", type(predData), len(predData))
x1, x2 = np.linspace(0, 6, 600), np.linspace(0, 7.3, 730)
params = [(x1, tureData), (x2, predData)]
labels = ["Ture", "Predict"]
Plot.ShowImg(params, labels)
"""

czvuviixmsmedcge