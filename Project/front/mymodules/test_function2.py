import os
import numpy as np
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
import matplotlib.pyplot as plt
import itertools
from sklearn.utils.multiclass import unique_labels


from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def test():
    # path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_dir = os.path.join(BASE_DIR, 'final1/test')


    #path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'learning_test/checkpoint/VGG16_stl10.h5')
    # test_dir = os.path.join(BASE_DIR, 'test/test')
    test_dir = os.path.join(BASE_DIR, 'final1/test')

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
    # print(test_classifications[0])

    # making list of true_label and predicted_label
    predicted_labels = []
    result_labels = []

    for test_classification in test_classifications:
        predicted_labels.append(np.argmax(test_classification))

    for i in range(len(true_labels[0])):
        result_labels.append([true_labels[0][i], predicted_labels[i]])

    real = []
    for i in true_labels[0]:
        real.append(i)
    # print(real)


    class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    
    def plot_confusion_matrix(y_true, y_pred, classes,
                            normalize=False,
                            title=None,
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
        classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        # print('dsfgfdgfdgfdgfdsfasdfsfdsgredtygdrafsdfdsayhetrsdfdsf',classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        #자동 레이아웃 설정
        fig.tight_layout()
        return ax




    # Plot non-normalized confusion matrix
    plot_confusion_matrix(real, predicted_labels, classes=class_names,
                        title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(real, predicted_labels, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

    plt.show()