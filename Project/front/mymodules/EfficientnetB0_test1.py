import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
import time
from .PlotLosses import PlotLosses

def cutout(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, pixel_level=False):
    def cutout_apply(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = 0
            if input_img.ndim == 2:
                c = 0
        else:
            c= 0

        input_img[top:top + h, left:left + w] = c

        return input_img

    return cutout_apply



def Learn(augmentation, input_epochs, train_path, val_path, window):
    #path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(BASE_DIR)
    train_dir = BASE_DIR + '\\' + train_path
    val_dir = BASE_DIR + '\\' + val_path
    # Define hyperparameter
    INPUT_SIZE = 200
    CHANNELS = 3
    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNELS)
    NUM_CLASSES = window.learn_num_data[0]
    NUM_TRAIN_IMGS = window.learn_num_data[1] * NUM_CLASSES
    NUM_VAL_IMGS = window.learn_num_data[2] * NUM_CLASSES
    BATCH_SIZE = 32

    HORIZONTAL_FLIP = augmentation[0]
    VERTICAL_FLIP = augmentation[1]
    BRIGHTNESS_RANGE = augmentation[2]
    ROTATION_RANGE = augmentation[3]
    if augmentation[4] == True:
        CUT_OUT = cutout(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, pixel_level=False)
    else:
        CUT_OUT = None

    EPOCHS = input_epochs
    train_steps_per_epoch = NUM_TRAIN_IMGS // BATCH_SIZE
    val_steps_per_epoch = NUM_VAL_IMGS // BATCH_SIZE

    # Data Preprocessing
    training_datagen = ImageDataGenerator(
                            # rescale = 1./255,
                            horizontal_flip = HORIZONTAL_FLIP,
                            vertical_flip = VERTICAL_FLIP,
                            brightness_range = BRIGHTNESS_RANGE,
                            rotation_range = ROTATION_RANGE,
                            preprocessing_function = CUT_OUT,
                            )
    validation_datagen = ImageDataGenerator(
                            # rescale = 1./255
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
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, 
                                                weights='imagenet', 
                                                input_shape=INPUT_SHAPE,)

    # Freeze the pre-trained layers
    base_model.trainable = False

    # Add a fully connected layer
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.GlobalMaxPooling2D(name="max_pooling2d"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_out"))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name="predictions"))


    model.summary()

    # Compile
    model.compile(optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

    # Callbacks
    checkpoint_filepath = os.path.join(BASE_DIR, 'checkpoint', window.settingsData[0] + '_' + window.settingsData[3] + '.h5')

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