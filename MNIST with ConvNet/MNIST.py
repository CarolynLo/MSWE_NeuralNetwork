import keras
keras.__version__
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

model = models.Sequential()
## input size
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

## layers
# model.summary()
# The number of channels is controlled by the first argument passed to the Conv2D layers (e.g. 32 or 64)

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928     # output tensor
# =================================================================
# Total params: 55,744
# Trainable params: 55,744
# Non-trainable params: 0

## fully connected layer
## flatten 3D data into 1D
model.add(layers.Flatten())
## densely-connected classifier network
model.add(layers.Dense(64, activation='relu'))
## 10 way classification use 10 outputs and softmax (possibilities, summing to 1)
model.add(layers.Dense(10, activation='softmax'))

## all layers in model
# model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928     
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 576)               0         # 3*3*64 = 576
# _________________________________________________________________
# dense_1 (Dense)              (None, 64)                36928     
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                650       
# =================================================================
# Total params: 93,322
# Trainable params: 93,322
# Non-trainable params: 0

## train data
## download dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

## normalize dataset
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
## cataegorize
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss,",",test_acc)