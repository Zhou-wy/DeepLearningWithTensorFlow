import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np
x = tf.constant([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

k = tf.constant([[1, 2],
                 [2, 3]])


# 自定义卷积运算#
def corr2d(X, K):
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))  # 输出矩阵的维度
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.cast(tf.reduce_sum(X[i:(i + h), j:(j + w)] * K), dtype = tf.float32))
            print("Y = ", Y, "\n")
    return Y


class Conv2D(keras.layers.Layer):
    def __init__(self, units):
        super(Conv2D, self).__init__()
        self.units = units

    def build(self, kernel_size):
        self.w = self.add_weight(name = "w",
                                 shape = kernel_size,
                                 initializer = tf.random_normal_initializer()
                                 )
        self.b = self.add_weight(name = "b",
                                 shape = (1,),
                                 initializer = tf.random_normal_initializer())

    def call(self, inputs):
        return corr2d(inputs, self.w) + self.b


img = cv2.imread("./Lenna.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print(X)


cv2.imshow("img",img)
#cv2.imshow("conv",np.array(Y_hat))
cv2.waitKey(0)
cv2.destroyAllWindows()


