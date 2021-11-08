import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import os
import pandas as pd

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.
x_test = x_test / 255.

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(10, activation = "softmax")
])

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

m_loss = 'sparse_categorical_crossentropy'
m_optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)

model.compile(optimizer = m_optimizer, loss = m_loss, metrics = ["accuracy"])
history = model.fit(x_train, y_train, epochs = 10, batch_size = 256, validation_data = (x_test, y_test),
                    callbacks = callbacks)
tf.print(history)


# 画accuracy曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize = (8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)
