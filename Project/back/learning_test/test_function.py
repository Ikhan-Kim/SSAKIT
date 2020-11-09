import os
import numpy as np
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_dir = os.path.join(BASE_DIR, 'final1/test')


#path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'learning_test/checkpoint/VGG16_stl10.h5')
test_dir = os.path.join(BASE_DIR, 'test/test')


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
print(test_classifications[0])

# making list of true_label and predicted_label
predicted_labels = []
result_labels = []

for test_classification in test_classifications:
    predicted_labels.append(np.argmax(test_classification))

for i in range(len(true_labels[0])):
    result_labels.append([true_labels[0][i], predicted_labels[i]])

# print('55555555555555', result_labels)
cnt = 0
for i in result_labels:
    if i[0] == i[1]:
        cnt += 1
print(cnt)
class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
title = "stl-10 Normalized Confusion matrix, "
# disp = plot_confusion_matrix(new_model, true_labels, predicted_labels,
#                               display_labels=class_names,
#                               cmap=plt.cm.Blues,
#                               normalize=None)
# disp.ax_.set_title(title)
# print(title)
# print(disp.confusion_matrix)

# plt.show()
print(true_labels)
print(predicted_labels)

matrix = confusion_matrix(true_labels[0], predicted_labels)
np.set_printoptions(precision=2)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(matrix)
fig.colorbar(cax)

# plt.figure()
# disp = plot_confusion_matrix(test_classifications,true_labels[0], predicted_labels, display_labels=class_names, normalize=None)
# disp.ax_.set_title(title)
plt.show()