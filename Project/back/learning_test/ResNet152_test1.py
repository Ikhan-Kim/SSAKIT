import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define hyperparameter
INPUT_SIZE = 224
CHANNELS = 3
INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNELS)
NUM_CLASSES = 10
NUM_TRAIN_IMGS = 500
NUM_VAL_IMGS = 100
BATCH_SIZE = 64

HORIZONTAL_FLIP = False
VERTICAL_FLIP = False
BRIGHTNESS_RANGE = None
ROTATION_RANGE = 0

EPOCHS = 10
train_steps_per_epoch = NUM_TRAIN_IMGS // BATCH_SIZE
val_steps_per_epoch = NUM_VAL_IMGS // BATCH_SIZE

# Data Preprocessing
training_datagen = ImageDataGenerator(
                        rescale = 1./255,
                        horizontal_flip = HORIZONTAL_FLIP
                        vertical_flip = VERTICAL_FLIP
                        brightness_range = BRIGHTNESS_RANGE
                        rotation_range = ROTATION_RANGE
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
base_model = tf.keras.applications.ResNet152(include_top=False, 
                                            weights='imagenet', 
                                            input_shape=INPUT_SHAPE,)

# Freeze the pre-trained layers
base_model.trainable = False

# Add a fully connected layer
model = tf.keras.Sequential()
model.add(base_model)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model.summary()

# Compile
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

# Callbacks
checkpoint_filepath = os.path.join(BASE_DIR, 'learning_test/checkpoint/ResNet152_cifar10.h5')


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
]


# training model
history = model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=train_steps_per_epoch, validation_data = validation_generator, validation_steps=val_steps_per_epoch, verbose = 1,  callbacks=callbacks)



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

plt.show()