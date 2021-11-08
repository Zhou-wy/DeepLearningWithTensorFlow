import tensorflow as tf
import numpy as np
import os
import pathlib
import tensorflow.keras as keras

for gpu in tf.config.experimental.list_physical_devices('gpu'):
    tf.config.experimental.set_memory_growth(gpu, True)

train_dir = "./hotdog/train"
test_dir = "./hotdog/test"

train_dir = pathlib.Path(train_dir)
train_count = len(list(train_dir.glob("*/*.jpg")))
test_dir = pathlib.Path(test_dir)
test_count = len(list(test_dir.glob("*/*.jpg")))

CLASS_NAMES = np.array(
        [item.name for item in train_dir.glob('*') if item.name != 'LICENSE.txt' and item.name[0] != '.'])
image_generator = keras.preprocessing.image.ImageDataGenerator(rescale = 1. / 255)
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_data_gen = image_generator.flow_from_directory(directory = str(train_dir),
                                                     batch_size = BATCH_SIZE,
                                                     target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                     shuffle = True,
                                                     classes = list(CLASS_NAMES))
test_data_gen = image_generator.flow_from_directory(directory = str(test_dir),
                                                    batch_size = BATCH_SIZE,
                                                    target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                    shuffle = True,
                                                    classes = list(CLASS_NAMES))
import matplotlib.pyplot as plt

def show_batch(image_batch, label_batch,batch):
    plt.figure(figsize=(10,10))
    for n in range(batch):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
    plt.show()

#image_batch, label_batch = next(train_data_gen)
#ishow_batch(image_batch, label_batch,batch = 25)


ResNet50 = keras.applications.resnet_v2.ResNet50V2(weights = "imagenet",input_shape = (224,224,3))
for layer in ResNet50.layers:
    layer.trainable = False
net = keras.models.Sequential()
net.add(ResNet50)
net.add(keras.layers.Flatten())
net.add(keras.layers.Dense(2,activation = "softmax"))

net.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

history = net.fit_generator(
                    train_data_gen,
                    steps_per_epoch=10,
                    epochs=10,
                    validation_data=test_data_gen,
                    validation_steps=10
                    )

