import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_images(imgs, num_rows, num_cols, scale = 5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize = figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()
    return axes

def apply(img, aug, num_rows = 3, num_cols = 3, scale = 2):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)

aug = tf.image.random_hue
"""
aug = tf.image.random_crop
aug = tf.image.random_flip_up_down
aug = tf.image.random_flip_left_right
aug = tf.image.random_brightness
aug = tf.image.random_contrast

"""
aug = tf.image.random_hue
num_rows = 3
num_cols = 3
scale = 1.5
max_delta = 0.5
img = cv2.imread("./data/test.jpeg")
Y = [aug(img, max_delta) for _ in range(num_rows * num_cols)]
show_images(Y, num_rows, num_cols, scale)
