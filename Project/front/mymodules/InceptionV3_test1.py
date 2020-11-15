import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
import time
from .PlotLosses import PlotLosses

def Learn(augmentation, input_epochs, train_dir, val_dir, window):
    #path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_dir = os.path.join(BASE_DIR, train_dir)
    val_dir = os.path.join(BASE_DIR, val_dir)
    # Define hyperparameter
    # INPUT_SIZE = 299
    INPUT_SIZE = 200
    CHANNELS = 3
    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNELS)
    NUM_CLASSES = window.learn_num_data[0]
    NUM_TRAIN_IMGS = window.learn_num_data[1]
    NUM_VAL_IMGS = window.learn_num_data[2]
    BATCH_SIZE = 32


    HORIZONTAL_FLIP = augmentation[0]
    VERTICAL_FLIP = augmentation[1]
    BRIGHTNESS_RANGE = augmentation[2]
    ROTATION_RANGE = augmentation[3]

    EPOCHS = input_epochs
    train_steps_per_epoch = NUM_TRAIN_IMGS // BATCH_SIZE
    val_steps_per_epoch = NUM_VAL_IMGS // BATCH_SIZE

    # Data Preprocessing
    training_datagen = ImageDataGenerator(
                            rescale = 1./255,
                            horizontal_flip = HORIZONTAL_FLIP,
                            vertical_flip = VERTICAL_FLIP,
                            brightness_range = BRIGHTNESS_RANGE,
                            rotation_range = ROTATION_RANGE,
                            )
    validation_datagen = ImageDataGenerator(
                            rescale = 1./255
                            )


    train_generator = training_datagen.flow_from_directory(
        train_dir,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        class_mode='categorical',
        batch_size= BATCH_SIZE
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        class_mode='categorical',
        batch_size= BATCH_SIZE
    )


    # Load pre-trained model
    base_model = tf.keras.applications.InceptionV3(include_top=False, 
                                            weights='imagenet', 
                                            input_shape=INPUT_SHAPE,)

    # Freeze the pre-trained layers
    base_model.trainable = False

    # Add a fully connected layer
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    model.summary()

    # Compile
    model.compile(optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

    # Callbacks
    checkpoint_filepath = os.path.join(BASE_DIR, 'checkpoint', window.settingsData[3] + '.h5')
    
    plotLosses = PlotLosses(input_epochs, window)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss',
                                        #  restore_best_weights=True
                                        ),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True,
                                            # save_weights_only=True,
                                        ),
        plotLosses,

    ]


    # training model
    history = model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=train_steps_per_epoch, validation_data = validation_generator, validation_steps=val_steps_per_epoch, verbose = 1,  callbacks=callbacks)
    window.textBox_terminal.append("Training Done!")
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    
    Max_val_accuracy = round(np.max(val_accuracy), 4))
    Min_val_loss = round(np.min(val_loss), 4)
    message = "Epoch: "+ str(np.argmin(val_loss)+1)+ " , Min val_loss: "+ str(Min_val_loss)
    window.textBox_terminal.append(message)
    plt.close()