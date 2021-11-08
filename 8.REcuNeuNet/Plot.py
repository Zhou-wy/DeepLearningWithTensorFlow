import matplotlib.pyplot as plt
import tensorflow.keras as keras

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




