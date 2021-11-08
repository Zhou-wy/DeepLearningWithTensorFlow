import tensorflow.keras as keras
from data import data
import KerasRNN as RNNModel
import os

# 数据集
Condata = data.ControlData()
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = Condata.load_data()

# 从零开始实现
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

"""
def get_params():
    def _one(shape):
        return tf.Variable(tf.random.normal(shape = shape, stddev = 0.01, dtype = tf.float32))

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                tf.Variable(tf.zeros(num_hiddens), dtype = tf.float32))

    W_xz, W_hz, b_z = _three()
    W_xr, W_hr, b_r = _three()
    W_xh, W_hh, b_h = _three()
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs), dtype = tf.float32)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    return params


# 下面的代码定义隐藏状态初始化函数init_gru_state。
# 同6.4节（循环神经网络的从零开始实现）中定义的init_rnn_state函数一样，
# 返回由一个形状为(批量大小, 隐藏单元个数)的值为0的Tensor组成的元组。
def init_gru_state(batch_size, num_hiddens):
    return (tf.zeros(shape = (batch_size, num_hiddens)),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X, [-1, W_xh.shape[0]])
        Z = tf.sigmoid(tf.matmul(X, W_xz) + tf.matmul(H, W_hz) + b_z)
        R = tf.sigmoid(tf.matmul(X, W_xh) + tf.matmul(H, W_hr) + b_r)
        H_tilda = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

RNN.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
"""
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 18, 16, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, [[0.9999997930212569, 0.9999997656385977, 0.9999997403897148,
                                            0.9999997172561449, 0.9999996962131402, 0.9999996772300609,
                                            0.9999996602707657, 0.9999996452940042, 0.9999996322538063,
                                            0.9999996210998685, 0.9999996117779381, 0.999999604230188,
                                            0.9999995983955896, 0.9999995942102761, 0.9999995916078974,
                                            0.9999995905199677, 0.9999995908762017, 0.9999995926048424,
                                            0.999999595632975, 0.9999995998868328, 0.9999996052920886,
                                            0.9999996117741351, 0.9999996192583512, 0.9999996276703567,
                                            0.9999996369362534, 0.9999996469828518, 0.9999996577378863,
                                            0.9999996691302143, 0.9999996810900043, 0.9999996935489076]]
gru_layer = keras.layers.GRU(num_hiddens, time_major = True, return_sequences = True, return_state = True)
logdir = './model/GRU'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, "GRU_Model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only = True),
    keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-3),
]
model = RNNModel.RNNModel(gru_layer, vocab_size)
RNNModel.train_and_predict_rnn_keras(model, num_hiddens, vocab_size, corpus_indices, idx_to_char, char_to_idx,
                                     num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len,
                                     prefixes)

