import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from .PlotLosses import PlotLosses

def Retrain(augmentation, input_epochs, train_path, val_path, window, trained_model_path):
    #path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(BASE_DIR, train_path)
    val_dir = os.path.join(BASE_DIR, val_path)
    MODEL_PATH = os.path.join(BASE_DIR, trained_model_path)
    # Define hyperparameter
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
    model = tf.keras.models.load_model(MODEL_PATH)
    model.pop()
    # Add last layer
    for layr in model.layers:
        if ('vgg16' in layr.name):
            layr.trainable = False
        elif ('resnet152' in layr.name):
            layr.trainable = False
        elif ('efficientnetb0' in layr.name):
            layr.trainable = False
        elif ('inception_v3' in layr.name):
            layr.trainable = False
            
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='predictions'+str(NUM_CLASSES)))
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

    max_val_accuracy = round(np.max(val_accuracy), 4)
    min_val_loss = round(np.min(val_loss), 4)
    message = "Epoch: "+ str(np.argmin(val_loss)+1)+ " , Min val_loss: "+ str(min_val_loss)
    window.textBox_terminal.append(message)
    window.settingsData.append(min_val_loss)
    window.settingsData.append(max_val_accuracy)
    plt.close()