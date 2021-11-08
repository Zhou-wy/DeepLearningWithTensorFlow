import numpy as np
import tensorflow as tf
import math
import sys
import matplotlib.pyplot as plt

sys.path.append("..")

"""
def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x  # f(x) = x ** 2 导数
        results.append(x)

    print("epoch 10, x:", x)
    return results


def show(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    plt.plot(f_line, [x * x for x in f_line])
    plt.plot(res, [x * x for x in res], "-o")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


show(gd(0.3))
"""


# 多维度下降
def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
        print("epoch %d, x1 %f, x2 %f" % (i + 1, x1, x2))
    return results


def show_trace_2d(f, results):
    plt.plot(*zip(*results), "-o", color="r")
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='b')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def f_2d(x1, x2):  # 目标函数
    return x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    eta = 0.1
    return x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0


# show_trace_2d(f_2d, train_2d(gd_2d))
def sgd_2d(x1, x2, s1, s2):
    eta = 0.1
    return (x1 - eta * (2 * x1 + np.random.normal(0.01)),
            x2 - eta * (4 * x2 + np.random.normal(0.01)), 0, 0)


#show_trace_2d(f_2d, train_2d(sgd_2d))
