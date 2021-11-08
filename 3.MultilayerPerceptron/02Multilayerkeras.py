import tensorflow as tf
import pandas as pd
import os
from tensorflow import keras
from matplotlib import pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
"""train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(batch_size).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(batch_size).batch(batch_size)"""


# 获取标签 and 画图
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels, nrow, ncol):
    _, figs = plt.subplots(nrow, ncol, figsize = (nrow * 1.5, ncol * 1.3))
    for i in range(nrow):
        for j in range(ncol):
            index = i * ncol + j
            figs[i][j].imshow(tf.reshape(images[index], (28, 28)).numpy())
            figs[i][j].set_title(labels[index])
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


"""
X, y = iter(test_iter).next()
show_fashion_mnist(X[0:9], get_fashion_mnist_labels(y[0:9]), 3, 3)
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(256, activation = 'relu', ),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
# Tensorboard, earlystopping, ModelCheckpoint
logdir = './callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,
                                 "fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only = True),
    keras.callbacks.EarlyStopping(patience = 5, min_delta = 1e-3),
]

model.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.1),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
his = model.fit(x_train, y_train, epochs = 30,
                batch_size = 256,
                validation_data = (x_test, y_test),
                validation_freq = 1,
                callbacks = callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(his)
