import keras
keras.__version__
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

# download dataset

from keras.datasets import imdb

# num_words=10000 means that we will only keep the top 10,000 most frequently occurring words
# data: list of words, label: 0 for negative and 1 for positive
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# turn list of integers into tensor

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
## relu turns negative number into zero
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
## sigmoid turns number between 0 to 1, to get the possibility of positive(1)
model.add(layers.Dense(1, activation='sigmoid'))

# adjust epochs from 20 to 4


## set up a loss function and an optimizer
## can also use 'mean_squared_error', but 'binary_crossentropy' more proper to possibility and 'distance' from prediction to real

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 20 iterations, n mini-batches of 512 samples

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
## keys are different from examples

history_dict = history.history
history_dict.keys()

## plot the training and validation loss



acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss= history.history['loss']
val_loss=history.history['val_loss']

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

## plot the training and validation accuracy

plt.clf()   # clear figure
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show() # it shows overfitting

# ## previous one

# model = models.Sequential()
# ## relu turns negative number into zero
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# ## sigmoid turns number between 0 to 1, to get the possibility of positive(1)
# model.add(layers.Dense(1, activation='sigmoid'))

# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# adjust epochs from 20 to 4

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

print(results)