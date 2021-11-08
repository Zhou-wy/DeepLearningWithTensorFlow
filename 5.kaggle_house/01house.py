import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
batch_size = 256
# 导入数据
boston_housing = pd.read_csv("boston_house_prices.csv")
features = np.array(boston_housing.iloc[:,0:13])
labels = np.array((boston_housing['MEDV']))
titles = list(boston_housing.columns)
train_features, test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.2)
"""
train_iter = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size).shuffle(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batch_size).shuffle(batch_size)
"""


# 画图
def show_plot(x, y, nrows, ncols):
    assert x.shape[0] == y.shape[0], "输入和长度不一致"
    # 设置13个散点图对于的标题
    titles = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B-1000", "LSTAT",
              "MEDV"]
    _, figs = plt.subplots(nrows, ncols, figsize = (12, 8))
    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            if index < 13:
                figs[i][j].scatter(x[:, index], y[:, ], s = 3)
                figs[i][j].set_title(titles[index])
            else:
                break
    plt.tight_layout()  #
    plt.show()


def get_net():
    net = keras.models.Sequential()
    net.add(keras.layers.Dense(1))
    return net


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = tf.concat([X_train, X_part], axis = 0)
            y_train = tf.concat([y_train, y_part], axis = 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        # create model
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        # Compile model
        net.compile(loss = tf.keras.losses.mean_squared_logarithmic_error,
                    optimizer = tf.keras.optimizers.Adam(learning_rate))
        # Fit the model
        history = net.fit(data[0], data[1], validation_data = (data[2], data[3]), epochs = num_epochs,
                          batch_size = batch_size, validation_freq = 1, verbose = 0)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        print('fold %d, train rmse %f, valid rmse %f'
              % (i, loss[-1], val_loss[-1]))
    plt.subplot(1, 2, 2)
    plt.plot(loss, label = 'train')
    plt.plot(val_loss, label = 'valid')
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.show()


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.5, 0, 64
k_fold(k, train_features, train_labels, num_epochs, lr, batch_size)
