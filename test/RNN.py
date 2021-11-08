import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from data import data
from matplotlib import pyplot as plt


def ShowImg(parms, labels):
    color = ['b', 'r', 'g', 'y']
    linestyle = ['-', '--', '-.', ':']
    # 设置表格
    plt.grid(alpha = 0.4)
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.title("LSTM")
    for i, (parm_x, parm_y) in enumerate(parms):
        plt.plot(parm_x, parm_y, label = labels[i], color = color[i], linestyle = linestyle[i])
    # 添加图例
    plt.legend(loc = "upper right")
    plt.show()


# 数据集#
ConData = data.ControlData()
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = ConData.load_data()
TrueData = ConData.TureData()
"""
x1 = np.linspace(0, 6, 600)
params, labels = [(x1, TrueData)], ["TureData"]
ShowImg(params, labels)
"""
