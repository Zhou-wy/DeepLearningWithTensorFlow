import tensorflow as tf
import numpy as np
import time
import math
# import Jaydata as Jdata
from data import data

# 准备训练数据
ConData = data.ControlData()
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = ConData.load_data()
"""
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = Jdata.load_data_jay_lyrics()
"""
gqhlebfkxprzdgbh

# print(corpus_indices, "\n", char_to_idx, "\n", idx_to_char, "\n", vocab_size)


def to_onehot(X, size):
    return [tf.one_hot(x, size, dtype = tf.float32) for x in X.T]


X = np.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
len(inputs), inputs[0].shape
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size


def get_params():
    def _one(shape):
        return tf.Variable(tf.random.normal(shape = shape, stddev = 0.01, mean = 0, dtype = tf.float32))

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype = tf.float32)
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs), dtype = tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params


def init_rnn_state(batch_size, num_hiddens):
    return (tf.zeros(shape = (batch_size, num_hiddens)),)


def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X, [-1, W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


"""
state = init_rnn_state(X.shape[0], num_hiddens)
inputs = to_onehot(X, vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)

"""


def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens)
    output = [char_to_idx[prefix[0]]]
    re = []
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = tf.convert_to_tensor(to_onehot(np.array([output[-1]]), vocab_size), dtype = tf.float32)
        X = tf.reshape(X, [1, -1])
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(np.array(tf.argmax(Y[0], axis = 1))))
    # print(output)
    # print([idx_to_char[i] for i in output])
    re.append([idx_to_char[i] for i in output])
    return re


# print(predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, idx_to_char, char_to_idx))


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


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps, lr, clipping_theta, batch_size,
                          pred_period, pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = ConData.data_iter_random
    else:
        data_iter_fn = ConData.data_iter_consecutive
    params = get_params()
    # loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate = lr)
    predictdata = str()
    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps)
        for X, Y in data_iter:
            # print(X, Y)
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens)
            # else:  # 否则需要使用detach函数从计算图分离隐藏状态
            # for s in state:
            # s.detach()
            with tf.GradientTape(persistent = True) as tape:
                tape.watch(params)
                inputs = to_onehot(X, vocab_size)
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
                # 拼接之后形状为(num_steps * batch_size, vocab_size)
                outputs = tf.concat(outputs, 0)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                # batch * num_steps 的向量，这样跟输出的行一一对应
                y = Y.T.reshape((-1,))
                # print(Y,y)
                y = tf.convert_to_tensor(y, dtype = tf.float32)
                # 使用交叉熵损失计算平均分类误差
                l = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y, outputs, from_logits = True))
                # l = loss(y,outputs)
                # print("loss",np.array(l))

            grads = tape.gradient(l, params)
            grads = grad_clipping(grads, clipping_theta)  # 裁剪梯度
            optimizer.apply_gradients(zip(grads, params))
            # sgd(params, lr, 1 , grads)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += np.array(l).item() * len(y)
            n += len(y)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            # print(params)
            for prefix in prefixes:
                predictdata = predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size,
                                          idx_to_char,
                                          char_to_idx)
                print(' -> ', predictdata[0])
    # print("predictdata",predictdata)
    return predictdata[0]

"""
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 18, 16, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, [
    [0.9999996052920886, 0.9999996117741351, 0.9999996192583512, 0.9999996276703567, 0.9999996369362534,
     0.9999996469828518, 0.9999996577378863, 0.9999996691302143, 0.9999996810900043, 0.9999996935489076]]

PredictData = train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                                    vocab_size, corpus_indices, idx_to_char,
                                    char_to_idx, True, num_epochs, num_steps, lr,
                                    clipping_theta, batch_size, pred_period, pred_len,
                                    prefixes)
truedata = ConData.TureData()
# print(len(truedata), len(data))
# 600 652
x1, x2 = np.linspace(0, 6, 600), np.linspace(0, 6.6, 660)
parms = [(x1, truedata), (x2, truedata + PredictData)]
titles = ["true", "predict"]
import Plot

Plot.ShowImg(parms, titles)
"""
"""
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
"""
