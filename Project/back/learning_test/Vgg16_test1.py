import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
import time

# from livelossplot import PlotLossesKeras


def Learn(augmentation, input_epochs, train_path, val_path, tbt, fig, canvas):
    #path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(BASE_DIR, train_path)
    val_dir = os.path.join(BASE_DIR, val_path)
    print(train_dir)
    print(val_dir)
    # Define hyperparameter
    INPUT_SIZE = 200
    CHANNELS = 3
    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNELS)
    NUM_CLASSES = 10
    NUM_TRAIN_IMGS = 3000
    NUM_VAL_IMGS = 1000
    BATCH_SIZE = 32

    HORIZONTAL_FLIP = augmentation[0]
    VERTICAL_FLIP = augmentation[1]
    BRIGHTNESS_RANGE = None
    ROTATION_RANGE = augmentation[2]

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
    base_model = tf.keras.applications.VGG16(include_top=False, 
                                            weights='imagenet', 
                                            input_shape=INPUT_SHAPE ,)

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
    checkpoint_filepath = os.path.join(BASE_DIR, 'learning_test/checkpoint/VGG16_cifar10.h5')

    class PlotLosses(keras.callbacks.Callback):
        def __init__(self, tbt, figure, canvas):
            self.textBox_terminal = tbt
            self.fig = figure
            self.canvas = canvas

        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []
            self.acc = []
            self.val_acc = []
            # self.fig = plt.figure()

            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.acc.append(logs.get('accuracy'))
            self.val_acc.append(logs.get('val_accuracy'))
            self.i += 1

            # 터미널 출력
            self.textBox_terminal.append(
                "Epoch {}/{} : loss = {}, accuracy = {}, val_loss = {}, val_accuracy = {}".format(self.i, input_epochs, round(self.losses[-1], 4), round(self.acc[-1], 4), round(self.val_losses[-1], 4), round(self.val_acc[-1], 4)))

            self.fig.clear()
            ax1 = self.fig.add_subplot(121)
            ax1.plot(self.x, self.acc, label="train_accuracy")
            ax1.plot(self.x, self.val_acc, label="val_accuracy")
            ax1.legend()
            ax1.set_title("accuracy")
            ax2 = self.fig.add_subplot(122)
            ax2.plot(self.x, self.losses, label="train_loss")
            ax2.plot(self.x, self.val_losses, label="val_loss")
            ax2.legend()
            ax2.set_title("loss")

            if self.i == input_epochs:
                now = time.gmtime(time.time())
                file_name = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + \
                    str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
                self.fig.savefig('result_logs\\'+file_name)
                print("figure saved!")
                self.textBox_terminal.append("Training Done!")

            self.canvas.draw()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_accuracy',
                                        #  restore_best_weights=True
                                        ),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                            monitor='val_accuracy',
                                            mode='max',
                                            save_best_only=True,
                                            # save_weights_only=True,
                                        ),
        PlotLosses(tbt, fig, canvas),
    ]


    # training model
    history = model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=train_steps_per_epoch, validation_data = validation_generator, validation_steps=val_steps_per_epoch, verbose = 1,  callbacks=callbacks)
    plt.close()
