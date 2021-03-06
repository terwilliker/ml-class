from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import subprocess
import os

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config
config.img_size = 100
config.batch_size = 128
config.epochs = 50
config.team = "Make_Simpsons_Great_Again"

if config.team is None:
    raise ValueError("You must set config.team on line 16!")

# download the data if it doesn't exist
if not os.path.exists("simpsons"):
    print("Downloading Simpsons dataset...")
    subprocess.check_output("curl https://storage.googleapis.com/wandb-production.appspot.com/mlclass/simpsons.tar.gz | tar xvz", shell=True)

# this is the augmentation configuration we will use for training
# see: https://keras.io/preprocessing/image/#imagedatagenerator-class
train_datagen = ImageDataGenerator(
   rescale=1./255,
   rotation_range=40,
   width_shift_range=0.2,
   height_shift_range=0.2,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   fill_mode='nearest')

#print(train_datagen.type)

# only rescaling augmentation for testing:
test_datagen = ImageDataGenerator(rescale=1./255)

#print(test_datagen.type)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    'simpsons/train',  # this is the target directory
    target_size=(config.img_size,config.img_size),
    batch_size=config.batch_size)

print(train_generator[1])

# this is a similar generator, for validation data
test_generator = test_datagen.flow_from_directory(
        'simpsons/test',
        target_size=(config.img_size,config.img_size),
        batch_size=config.batch_size)

print(test_generator[1])

labels = list(test_generator.class_indices.keys())

#print(labels)
#exit()

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(config.img_size, config.img_size, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3,3), input_shape=(config.img_size, config.img_size, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3,3), input_shape=(config.img_size, config.img_size, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(196, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(13, activation="softmax"))
model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=config.epochs,
        workers=4,
        validation_data=test_generator,
        callbacks=[WandbCallback(data_type="image", labels=labels, generator=test_generator, save_model=False)],
        validation_steps=len(test_generator))

model.save("trainweights")