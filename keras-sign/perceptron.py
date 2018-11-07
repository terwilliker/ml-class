# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, Reshape, MaxPooling2D
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.loss = "categorical_crossentropy"
config.optimizer = "adam"
config.epochs = 15

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

print(X_train.shape)
#exit()

img_width = X_test.shape[1]
img_height = X_test.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(y_train.shape)

num_classes = y_train.shape[1]

# you may want to normalize the data here..

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# create model
model=Sequential()
#model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Reshape((28,28,1), input_shape=(img_width, img_height)))
model.add(Dropout(0.3))
model.add(Conv2D(8, (3,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(16, (3,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(200, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(300, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer=config.optimizer,
                metrics=['accuracy'])
#model.compile(loss=config.loss, optimizer=config.optimizer, metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test), callbacks=[WandbCallback(data_type="image", labels=signdata.letters)])

print(model.predict(X_train[:2]))