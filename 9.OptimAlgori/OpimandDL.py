import sys
import tensorflow as tf
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

sys.path.append("..")

# 假设给定函数 f(x) = x * cos(pai * x)
# 找出局部最小值和全局最小值
"""def f(x):
    return x * np.cos(np.pi * x)


x = np.arange(-1.0, 2.0, 0.05)
fig,  = plt.plot(x, f(x))
fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0),
                  arrowprops=dict(arrowstyle='->'))
fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8),
                  arrowprops=dict(arrowstyle='->'))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
"""
"""
# 鞍点
x = np.arange(-2., 2, 0.1)
fig, = plt.plot(x, x ** 3)
fig.axes.annotate("saddle point", xy=(0, -0.2), xytext=(-0.52, -5.2), arrowprops=dict(arrowstyle="->"))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
"""

# 再举一个三维空间的例子：例如：f(x,y) = x ** 2 + y ** 2
"""
x, y = np.mgrid[-1: 1: 31j, -1: 1: 31j]
z = x**2 - y**2

ax = plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
ax.plot([0], [0], [0], 'rx')
ticks = [-1,  0, 1]
plt.xticks(ticks)
plt.yticks(ticks)
ax.set_zticks(ticks)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""




