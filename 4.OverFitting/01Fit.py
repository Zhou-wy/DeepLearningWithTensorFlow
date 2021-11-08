import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = tf.random.normal(shape=(n_train + n_test, 1))
poly_features = tf.concat([features, tf.pow(features, 2), tf.pow(features, 3)],1)
print(poly_features.shape)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]+ true_w[2] * poly_features[:, 2] + true_b)
print(tf.shape(labels))
labels += tf.random.normal(labels.shape,0,0.1)

print(features[:2], "\n", labels[:2])

num_epochs, loss = 100, tf.losses.MeanSquaredError()

def plot_show(x_vals, y_vals, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    l1 = ax.plot(x_vals, y_vals, 'ys-')  # solid line with yellow colour and square marker
    l2 = ax.plot(x_label, y_label, 'go--')  # dash line with green colour and circle marker
    ax.legend(labels = ('train loss', 'test loss'), loc = 'upper left')  # legend placed at lower right
    ax.set_title("Loss")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.show()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1))
    batch_size = min(10, train_labels.shape[0])
    train_iter = tf.data.Dataset.from_tensor_slices(
        (train_features, train_labels)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices(
        (test_features, test_labels)).batch(batch_size)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                l = loss(y, net(X))

            grads = tape.gradient(l, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

        train_ls.append(loss(train_labels, net(train_features)).numpy().mean())
        test_ls.append(loss(test_labels, net(test_features)).numpy().mean())


    plot_show(range(1,num_epochs+1),train_ls,range(1,num_epochs+1),test_ls)
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    print('weight:', net.get_weights()[0],'\nbias:', net.get_weights()[1])

fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
             labels[:n_train], labels[n_train:])