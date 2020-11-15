import os
import numpy as np
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
import matplotlib.pyplot as plt
import itertools
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from .Grad_cam import *

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


def test(model_name, window):
    #path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'checkpoint/'+model_name)
    test_dir = os.path.join(BASE_DIR, 'learnData/final1/test')

    # Define hyperparameter
    INPUT_SIZE = 200
    CHANNELS = 3
    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNELS)
    print(window.learn_num_data)
    NUM_CLASSES = window.learn_num_data[0]
    NUM_TRAIN_IMGS = window.learn_num_data[1]
    NUM_VAL_IMGS = window.learn_num_data[2]
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
    predictions = []

    for test_classification in test_classifications:
        predicted_labels.append(np.argmax(test_classification))
        predictions.append(int(round(np.max(test_classification), 2)*100))

    for i in range(len(true_labels[0])):
        result_labels.append([true_labels[0][i], predicted_labels[i], predictions[i]])
    real = []
    for i in true_labels[0]:
        real.append(i)


    # class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    
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
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        #대각선값
        diagonal = 0
        #전체값
        ssum = 0
        for i in range(len(cm)):
            diagonal += cm[i][i]
            for j in cm[i]:
                ssum += j
        acc = round(diagonal / ssum, 3) * 100

        precision = []
        recall = []
        for i in range(len(cm)):
            temp = 0
            tmp = 0
            for j in range(len(cm)):
                temp += cm[j][i]
                tmp += cm[i][j]
            temp = int(round(cm[i][i] / temp, 2) * 100)
            tmp = int(round(cm[i][i] / tmp, 2) * 100)
            precision.append(temp)
            recall.append(tmp)
        macro_precision = sum(precision) / len(precision)

        # classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

        # confusion matrix에 대응되는 이미지 파일 주소를 저장
        test_classes = os.listdir('./learnData/' + window.projectName + '/test/')
        img_info = [[[] for j in range(len(cm))] for i in range(len(cm))]
        for i in range(len(cm)):
            for j in range(len(result_labels)//len(cm)):
                img_info[i][result_labels[i * len(result_labels)//len(cm) + j][1]].append([os.listdir('./learndata/' + window.projectName + '/test/' + classes[i])[j], result_labels[i * len(cm) + j][2]])
        
        # click 함수가 없는 Widget들을 클릭 가능하게 해주는 함수
        def clickable(widget):
            class Filter(QObject):
                clicked = pyqtSignal()	#pyside2 사용자는 pyqtSignal() -> Signal()로 변경
                def eventFilter(self, obj, event):
                    if obj == widget:
                        if event.type() == QEvent.MouseButtonRelease:
                            if obj.rect().contains(event.pos()):
                                self.clicked.emit()
                                # The developer can opt for .emit(obj) to get the object within the slot.
                                return True
                    
                    return False
            
            filter = Filter(widget)
            widget.installEventFilter(filter)
            return filter.clicked
        

        def show_img(i, j):
            for x in reversed(range(window.testedImageLayout.count())): 
                window.testedImageLayout.itemAt(x).widget().setParent(None)
            img_path = './learnData/' + window.projectName + '/test/' + classes[i] + '/'
            scrollArea = QScrollArea()
            l = QVBoxLayout()
            for file in img_info[i][j]:
                pixmap = QPixmap(os.path.join(img_path, file[0]))
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(96, 96)
                    imgLabel = QLabel(pixmap=pixmap)
                    
                    def show_cam():
                        VGG16_Grad_cam(classes[i], file[0])
                    clickable(imgLabel).connect(show_cam)
                    # clickable(imgLabel).connect(show_cam(classes[i], file[0]))
                    imgLabelFileName = QLabel(file[0])
                    imgPrediction = QLabel(str(file[1]) + "%")
                    l.addWidget(imgLabel)
                    l.addWidget(imgLabelFileName)
                    l.addWidget(imgPrediction)
            w = QWidget()
            w.setLayout(l)
            scrollArea.setWidget(w)
            window.testedImageLayout.addWidget(scrollArea)

        # show confusion matrix
        window.confusionMatrixTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        window.confusionMatrixTable.setColumnCount(len(cm))
        window.confusionMatrixTable.setHorizontalHeaderLabels(classes)
        window.confusionMatrixTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        window.confusionMatrixTable.setRowCount(len(cm))
        window.confusionMatrixTable.setVerticalHeaderLabels(classes)
        for i in range(len(cm)):
            for j in range(len(cm)):
                window.confusionMatrixTable.setItem(i, j, QTableWidgetItem(str(cm[i][j])))
        
        # show confusion matrix image
        window.confusionMatrixTable.cellClicked.connect(show_img)


        # show precision, recall, accuracy
        window.precisionTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        window.precisionTable.setColumnCount(len(cm))
        window.precisionTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        window.recallTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        window.recallTable.setRowCount(len(cm))
        window.recallTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tmpPrecision = []
        tmpRecall = []
        for i in range(len(cm)):
            tmpPrecision.append(str(precision[i]) + "%")
            tmpRecall.append(str(recall[i]) + "%")
        window.precisionTable.setHorizontalHeaderLabels(tmpPrecision)
        window.recallTable.setVerticalHeaderLabels(tmpRecall)

        window.accuracyTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        window.accuracyTable.setColumnCount(1)
        window.accuracyTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        window.accuracyTable.setHorizontalHeaderLabels([str(acc) + "%"])
        
        window.macroPrecisionLabel.setText(str(macro_precision) + "%")

        



        # fig, ax = plt.subplots()
        # im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # ax.figure.colorbar(im, ax=ax)
        # # We want to show all ticks...
        # ax.set(xticks=np.arange(cm.shape[1]),
        #     yticks=np.arange(cm.shape[0]),
        #     # ... and label them with the respective list entries
        #     xticklabels=classes, yticklabels=classes,
        #     title=title,
        #     ylabel='True label',
        #     xlabel='Predicted label')

        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #         rotation_mode="anchor")

        # # Loop over data dimensions and create text annotations.
        # fmt = '.2f' if normalize else 'd'
        # thresh = cm.max() / 2.
        # for i in range(cm.shape[0]):
        #     for j in range(cm.shape[1]):
        #         ax.text(j, i, format(cm[i, j], fmt),
        #                 ha="center", va="center",
        #                 color="white" if cm[i, j] > thresh else "black")
        # #자동 레이아웃 설정
        # fig.tight_layout()
        # return ax




    # Plot non-normalized confusion matrix
    plot_confusion_matrix(real, predicted_labels, classes=window.class_names,
                        title='Confusion matrix, without normalization')

    # # Plot normalized confusion matrix
    # plot_confusion_matrix(real, predicted_labels, classes=class_names, normalize=True,
    #                     title='Normalized confusion matrix')