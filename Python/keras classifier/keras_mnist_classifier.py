### Keras mnist

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Dropout, Activation

# Because softmax wouldnt work.
import tensorflow as tf

# Importing split datasets from separate CSVs.
# Python script used for splitting can be found in the folder.
x_train = pd.read_csv('x_train.csv', header=None)
y_train = pd.read_csv('y_train.csv', header=None)

x_test = pd.read_csv('x_test.csv', header=None)
y_test = pd.read_csv('y_test.csv', header=None)

# Instread of this you can just download entire MNIST dataset
# with sklearn. I had one file with train set and one with test set
# so i decided to split it myself......
# That code is in 'splitting:mnist.py' script.

'''
    DATA PREPROCESSING
'''

print('Preprocessing...')

# Converting Pandas DataFrame objects to numpy arrays.
x_train = x_train.values
y_train = y_train.values

x_test = x_test.values
y_test = y_test.values


# Converting label (y) ndarrays to normal arrays.
y_train = y_train.flatten()
y_test = y_test.flatten()


# One hot encoding.
n_classes = 10

y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

# Dimensionality reduction with PCA.
pca = PCA(n_components = 154)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Normalizing data
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

'''
    BUILDING THE MODEL
'''

print('Building the model...')

model = Sequential()

# Number of hidden layers = 2, number of neurons in them = 101.
model.add(Dense(101, input_shape=(154,)))

# Using ReLU activation function on hidden layer.
model.add(Activation('relu'))

# Dropout - for overfitting prevention.
model.add(Dropout(0.2))

model.add(Dense(101))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Number of neurons in output layer = 10.
model.add(Dense(10))

# Using Softmax activation function on output layer.
model.add(Activation(tf.nn.softmax))


'''
    TRAINING
'''

print('Training...')

# MSE for loss function, sucess metrics = accuracy and optimizer = ADAM.
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

history = model.fit(x_train, y_train,
                batch_size=128, epochs=20,
                verbose=2,
                validation_data=(x_test,y_test))

# Saving results to file.
import os

model_name = 'keras_mnist1_pca.h5'
model_path = os.path.join(model_name)
model.save(model_path)

print('Model saved.', '\n')

'''
    MODEL EVALUATION
'''

print("Evaluating model...")

# Evaluating loss and accuracy.
loss_and_metrics = model.evaluate(x_test, y_test)

print('Test loss:', loss_and_metrics[0])
print('Test Accuracy:',loss_and_metrics[1])

# Predicting values
predict = model.predict_classes(x_test)

# Show number of fake positives and true positives.
true_positives = np.nonzero(predict == y_test)[0]
fake_positives = np.nonzero(predict != y_test)[0]

print('True positives:',len(true_positives))
print('Fake positives:',len(fake_positives))

print('done')

