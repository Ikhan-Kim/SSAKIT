import os
import numpy as np
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image


def test():
    # path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_dir = os.path.join(BASE_DIR, 'final1/test')


    #path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'test_pjt3/checkpoint/VGG16_stl10_1109_epochs20_200.h5')
    test_dir = os.path.join(BASE_DIR, 'test_pjt3/dataset/test_testfunction')


    # Define hyperparameter
    INPUT_SIZE = 200
    CHANNELS = 3
    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNELS)
    NUM_CLASSES = 10
    NUM_TRAIN_IMGS = 3000
    NUM_VAL_IMGS = 1000
    BATCH_SIZE = 32


    # Data Preprocessing
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                                            rescale = 1./255
                                                            )


    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        class_mode='categorical',
        batch_size= BATCH_SIZE,
        shuffle =False,
    )

    true_labels = test_generator.classes[test_generator.index_array]


    # load model
    new_model = tf.keras.models.load_model(model_path)


    # testing files
    test_classifications = new_model.predict(test_generator)


    # making list of true_label and predicted_label
    predicted_labels = []
    result_labels = []

    for test_classification in test_classifications:
        predicted_labels.append(np.argmax(test_classification))

    for i in range(len(true_labels[0])):
        result_labels.append([true_labels[0][i], predicted_labels[i]])

    # print('55555555555555', result_labels)