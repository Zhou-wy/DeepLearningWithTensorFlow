import numpy as np
import time
import sys
import tensorflow as tf
import GradGown as GG

sys.path.append("..")



def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)


# GG.show_trace_2d(f_2d, GG.train_2d(gd_2d))
gamma = 0.5
eta = 0.6  # 学习率

def mommentum_2d(x1, x2, v1, v2):
    v1 = gamma * x1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

GG.show_trace_2d(f_2d, GG.train_2d(mommentum_2d))
