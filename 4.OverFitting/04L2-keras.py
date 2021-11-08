import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

n_train, n_test, num_inputs = 20, 100, 200
true_w, ture_b = tf.ones((num_inputs, 1)) * 0.01, 0.03

features = tf.random.normal(shape = (n_train + n_test, num_inputs))  # (120,200)
labels = keras.backend.dot(features, true_w) + ture_b
labels += tf.random.normal(mean = 0.01, shape = labels.shape)  # (120,1)
train_feature, test_feature = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

def plot_show(x_vals, y_vals, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    l2 = ax.plot(x_vals, y_vals, 'ys-')  # solid line with yellow colour and square marker
    l1 = ax.plot(x_label, y_label, 'go--')  # dash line with green colour and circle marker
    ax.legend(labels = ('train loss', 'test loss'), loc = 'upper left')  # legend placed at lower right
    ax.set_title("Loss")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.show()

def init_parms():
    w = tf.Variable(tf.random.normal(mean = 1, shape = (num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape = (1,)))
    return [w, b]