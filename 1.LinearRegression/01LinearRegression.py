import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow import data

############制作数据集############
n_inputs = 2
n_example = 1000
true_w = [2. , -3.4]
true_b = 4.2
features = tf.random.normal(shape=(n_example, n_inputs), mean=0.0, stddev=1.0, dtype=tf.float32, seed=10)
labels =  true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += tf.random.normal(shape=(labels.shape), mean=0.0, stddev=0.01, dtype=tf.float32, seed=10)
plt.scatter(features[:,0], labels, color = "r", s = 5)
plt.scatter(features[:,1], labels, color = "b", s = 5)
plt.show()
#################################

##########读取数据################
batch_size = 10
# 将训练数据的特征和标签组合
dataset = data.Dataset.from_tensor_slices((features, labels))
# 随机读取小批量
dataset = dataset.shuffle(buffer_size=n_example)
dataset = dataset.batch(batch_size)
data_iter = iter(dataset)
###############################

######定义模型和初始化参数###########
model = keras.Sequential()
model = keras.Sequential()
model.add(keras.layers.Dense(1, kernel_initializer=tf.initializers.RandomNormal(stddev=0.01)))
#############################


######## 定义损失函数和定义优化算法########
loss = tf.losses.MeanSquaredError()
trainer = keras.optimizers.SGD(learning_rate=0.03)
###########################

epochs = 20
lr = 0.1

for epoch in range (1, epochs + 1):
    for (batch, (X, y)) in enumerate (dataset):
        with tf.GradientTape () as tape:
            l = loss (model (X, training=True), y)

        grads = tape.gradient (l, model.trainable_variables)
        trainer.apply_gradients (zip (grads, model.trainable_variables))

    l = loss (model (features), labels)
    print ('epoch %d, loss: %f' % (epoch, l))






