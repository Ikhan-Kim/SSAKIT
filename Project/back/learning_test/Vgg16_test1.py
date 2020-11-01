import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import keras

# from livelossplot import PlotLossesKeras



#path
TRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



# Define hyperparameter
INPUT_SIZE = 32
CHANNELS = 3
# INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNELS)
NUM_CLASSES = 10
NUM_TRAIN_IMGS = 50000
NUM_TEST_IMGS = 10000

BATCH_SIZE = 128
train_steps_per_epoch = NUM_TRAIN_IMGS // BATCH_SIZE
val_steps_per_epoch = NUM_TEST_IMGS // BATCH_SIZE

# Data Preprocessing
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train/255.0
X_test = X_test/255.0

# X_train = X_train.reshape(X_train.shape[0], *INPUT_SHAPE)
# X_test = X_test.reshape(X_test.shape[0], *INPUT_SHAPE)


# Load pre-trained model
base_model = tf.keras.applications.VGG16(include_top=False, 
                                        weights='imagenet', 
                                        input_shape=(INPUT_SIZE, INPUT_SIZE, CHANNELS),)

# Freeze the pre-trained layers
base_model.trainable = False

# Add a fully connected layer
model_input = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, CHANNELS))
model_output = tf.keras.layers.Flatten()(model_input)
model_output = tf.keras.layers.Dense(512, activation='relu')(model_output)
model_output = tf.keras.layers.Dropout(0.2)(model_output)
model_output = tf.keras.layers.Dense(256, activation='relu')(model_output)
model_output = tf.keras.layers.Dropout(0.2)(model_output)
model_output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(model_output)
model = tf.keras.Model(model_input, model_output)

model.summary()

# Compile
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Callbacks
checkpoint_filepath = os.path.join(TRAIN_DIR, 'learning_test/checkpoint/VGG16_cifar10.h5')


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
    # PlotLossesKeras(),
]


# training model
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=100, steps_per_epoch=train_steps_per_epoch, validation_data = (X_test, Y_test), validation_steps=val_steps_per_epoch, verbose = 1,  callbacks=callbacks)



# 정확도 그래프 (임시) 
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()