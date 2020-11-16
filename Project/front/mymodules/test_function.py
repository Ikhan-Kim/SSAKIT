import os
import numpy as np
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
import matplotlib.pyplot as plt
import itertools
# from sklearn.utils.multiclass import unique_labels


from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


import sklearn.metrics as metrics




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

# print('55555555555555', result_labels)
# cnt = 0
# for i in result_labels:
#     if i[0] == i[1]:
#         cnt += 1
# print(cnt)

real = []
for i in true_labels[0]:
    real.append(i)
# print(real)


class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
# title = "stl-10 Normalized Confusion matrix, "
# disp = plot_confusion_matrix(new_model, true_labels, predicted_labels,
#                               display_labels=class_names,
#                               cmap=plt.cm.Blues,
#                               normalize=None)
# disp.ax_.set_title(title)
# print(title)
# print(disp.confusion_matrix)

# plt.show()
# print(true_labels)
# print(predicted_labels)

# matrix = confusion_matrix(true_labels[0], predicted_labels)
# np.set_printoptions(precision=2)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(matrix)
# fig.colorbar(cax)

# plt.figure()
# disp = plot_confusion_matrix(test_classifications,true_labels[0], predicted_labels, display_labels=class_names, normalize=None)
# disp.ax_.set_title(title)
# plt.show()

# def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
#     accuracy = (np.trace(cm)*1000) / float(np.sum(cm))
#     print(accuracy)
#     print(cm.shape)
#     misclass = 1 - accuracy

#     if cmap is None:
#         cmap = plt.get_cmap('Blues')

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print(cm)
#         print(cm.astype('float'))
#         print(cm.sum(axis=1))
#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()

#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     print(thresh)
    
#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names)
#         plt.yticks(tick_marks, target_names)
    
#     if labels:
#         for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#             if normalize:
#                 # print(555555555555555555555555555555555, i, j)
#                 plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                          horizontalalignment="center",
#                          color="white" if cm[i, j] > thresh else "black")
#             else:
#                 plt.text(j, i, "{:,}".format(cm[i, j]),
#                          horizontalalignment="center",
#                          color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label \n accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
#     plt.show()
# plot_confusion_matrix(test_classifications, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix')

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



 
modelAClassValueNDArray       = np.array([0   , 0   , 0   , 0   , 0   , 0   , 1   , 0   , 1   , 1   , 0   , 0   , 0   , 1   , 1   ])
modelAClassProbabilityNDArray = np.array([0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.35, 0.35, 0.45, 0.45, 0.55, 0.55, 0.65, 0.85, 0.95])

modelBClassValueNDArray       = np.array([0   , 0   , 0   , 1   , 1   , 0   , 0   , 1   , 0   , 0   , 1   , 0   , 0   , 0   , 1   ])
modelBClassProbabilityNDArray = np.array([0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.25, 0.35, 0.35, 0.45, 0.55, 0.55, 0.65, 0.75, 0.95])

 

modelAPrecisionNDArray, modelARecallRateNDArray, _ = metrics.precision_recall_curve(modelAClassValueNDArray, modelAClassProbabilityNDArray)
modelBPrecisionNDArray, modelBRecallRateNDArray, _ = metrics.precision_recall_curve(modelBClassValueNDArray, modelBClassProbabilityNDArray)

 

modelAAP = metrics.average_precision_score(modelAClassValueNDArray, modelAClassProbabilityNDArray)
modelBAP = metrics.average_precision_score(modelBClassValueNDArray, modelBClassProbabilityNDArray)
print(modelAAP)
print(modelBAP)

plt.title("Precision-Recall Graph")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.plot(modelARecallRateNDArray, modelAPrecisionNDArray, "b", label = "Model A (AP = %0.2F)" % modelAAP)
plt.plot(modelBRecallRateNDArray, modelBPrecisionNDArray, "g", label = "Model B (AP = %0.2F)" % modelBAP)

plt.legend(loc = "upper right")
plt.show()



sens_F = np.array([1.0,  1.0, 1.0,  1.0, 0.75,  0.5,  0.5, 0.5, 0.5, 0.5, 0.0])
spec_F = np.array([0.0, 0.16, 0.5, 0.66, 0.66, 0.66, 0.83, 1.0, 1.0, 1.0, 1.0])

sens_G = np.array([1.0,  1.0, 0.75, 0.75, 0.5,  0.5,  0.5,  0.5, 0.25, 0.25, 0.0])
spec_G = np.array([0.0, 0.33, 0.33,  0.5, 0.5, 0.66, 0.66, 0.83, 0.83,  1.0, 1.0])

plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')

plt.plot(1-spec_F, sens_F, 'b', label = 'Model F')   
plt.plot(1-spec_G, sens_G, 'g', label = 'Model G') 
plt.plot([0,1],[1,1],'y--')
plt.plot([0,1],[0,1],'r--')

plt.legend(loc='lower right')
plt.show()