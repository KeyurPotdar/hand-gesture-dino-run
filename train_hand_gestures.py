import numpy as np
import pandas as pd

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

NUM_CLASSES = 2
FRAC_TRAIN = .9
IMG_X = 50
IMG_Y = 50
SAVE_PATH = 'hand_gesture.h5'


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(IMG_X, IMG_Y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint1 = ModelCheckpoint(SAVE_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint2 = ModelCheckpoint(SAVE_PATH, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks = [checkpoint1, checkpoint2]

    return model, callbacks


def train():
    data = pd.read_csv('train.csv', header=None)
    total_samples = data.shape[0]
    num_train = int(total_samples * FRAC_TRAIN)
    data = np.array(data)
    np.random.shuffle(data)

    X = data[:, 1:] / 255.
    y = data[:, 0].reshape(-1, 1)
    X_train, X_test = X[:num_train, :], X[num_train:, :]
    y_train, y_test = y[:num_train, :], y[num_train:, :]
    X_train, X_test = X_train.reshape(-1, IMG_X, IMG_Y, 1), X_test.reshape(-1, IMG_X, IMG_Y, 1)

    model, callbacks = create_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, callbacks=callbacks)
    model.save(SAVE_PATH)


train()
