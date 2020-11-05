import sys
import os
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtCore, QtGui
from PIL import Image
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

from IPython.display import clear_output
import matplotlib.pyplot as plt
import pyqtgraph as pg

# 연결할 ui 파일의 경로 설정
UI_Path = './ui/NetworkSetting.ui'
form_class = uic.loadUiType(UI_Path)[0]


class AnotherFormLayout(QDialog):
    NumGridRows = 3
    NumButtons = 4

    def __init__(self):
        super().__init__()
        self.createFormGroupBox()

        buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formDataPreprocessing)
        mainLayout.addWidget(self.formNueralNetwork)
        mainLayout.addWidget(self.formLearn)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

        self.setWindowTitle("Nueral Network Settings")

    def createFormGroupBox(self):
        # data preprocessing
        self.formDataPreprocessing = QGroupBox("Data Preprocessing")
        layout = QFormLayout()
        self.lineTarget = QLineEdit()
        layout.addRow(QLabel("target size:"), self.lineTarget)
        layout.addRow(QLabel("class mode:"), QComboBox())
        self.lineBatch = QLineEdit()
        layout.addRow(QLabel("batch size:"), self.lineBatch)
        self.lineRgb = QLineEdit()
        layout.addRow(QLabel("rgb:"), self.lineRgb)
        self.formDataPreprocessing.setLayout(layout)
        # nn setting
        self.formNueralNetwork = QGroupBox("Nueral Network")
        layoutNN = QFormLayout()
        layoutNN.addRow(QLabel("select NN:"), QComboBox())
        self.formNueralNetwork.setLayout(layoutNN)
        # Learn Settings
        self.formLearn = QGroupBox("Learn Settings")

    def accept(self):
        print('hi')
        print(self.lineTarget.text(), self.lineBatch.text(), self.lineRgb.text())


class WindowClass(QMainWindow, form_class):
    mainImg = "C:/Users/multicampus/Desktop/s03p31c203/Project/front/test_img/test1.png"

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 기본 설정?>
        self.learnSettingDisplay = AnotherFormLayout()
        pixmap = QtGui.QPixmap(self.mainImg)
        self.imgLabel.setPixmap(pixmap)
        # 버튼별 함수 실행
        self.btnDataLoad.clicked.connect(self.dataLoadFn)
        self.btnLearnSettings.clicked.connect(self.learnSettingsFn)
        self.dirTreeView.doubleClicked.connect(self.fileViewFn)
        self.btnTraining.clicked.connect(self.training)
        self.textBox_terminal.setGeometry(QtCore.QRect(0, 510, 1200, 190))

    def dataLoadFn(self):
        self.dirName = QFileDialog.getExistingDirectory(self, self.tr("Open Data files"), "./",
                                                        QFileDialog.ShowDirsOnly)
        treeModel = QFileSystemModel()
        self.dirTreeView.setModel(treeModel)
        treeModel.setRootPath(QDir.rootPath())
        self.dirTreeView.setRootIndex(treeModel.index(self.dirName))

    def learnSettingsFn(self, checked):
        if self.learnSettingDisplay.isVisible():
            self.learnSettingDisplay.hide()
        else:
            self.learnSettingDisplay.show()

    def fileViewFn(self, index):
        self.mainImg = self.dirTreeView.model().filePath(index)
        print(self.mainImg)
        pixmap = QtGui.QPixmap(self.mainImg)
        self.imgLabel.setPixmap(pixmap)

    def training(self):
        # path
        TRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Define hyperparameter
        INPUT_SIZE = 32
        CHANNELS = 3
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

        # Load pre-trained model
        base_model = tf.keras.applications.VGG16(include_top=False,
                                                 weights='imagenet',
                                                 input_shape=(INPUT_SIZE, INPUT_SIZE, CHANNELS),)

        # Freeze the pre-trained layers
        base_model.trainable = False

        # Add a fully connected layer
        model_input = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, CHANNELS))
        model_output = tf.keras.layers.Flatten()(model_input)
        model_output = tf.keras.layers.Dense(
            512, activation='relu')(model_output)
        model_output = tf.keras.layers.Dropout(0.2)(model_output)
        model_output = tf.keras.layers.Dense(
            256, activation='relu')(model_output)
        model_output = tf.keras.layers.Dropout(0.2)(model_output)
        model_output = tf.keras.layers.Dense(
            NUM_CLASSES, activation='softmax')(model_output)
        model = tf.keras.Model(model_input, model_output)

        model.summary()

        # Compile
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Callbacks
        checkpoint_filepath = os.path.join(
            TRAIN_DIR, 'learning_test/checkpoint/VGG16_cifar10.h5')
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

        class PlotLosses(keras.callbacks.Callback):
            def __init__(self, tbt):
                self.textBox_terminal = tbt
                self.win = WindowClass()
                print("textBox copied")
                # plt.ion()

            def on_train_begin(self, logs={}):
                self.i = 0
                self.x = []
                self.losses = []
                self.val_losses = []
                self.acc = []
                self.val_acc = []
                self.fig = plt.figure()

                self.logs = []

            def on_epoch_end(self, epoch, logs={}):
                self.logs.append(logs)
                self.x.append(self.i)
                self.losses.append(logs.get('loss'))
                self.val_losses.append(logs.get('val_loss'))
                self.acc.append(logs.get('accuracy'))
                self.val_acc.append(logs.get('val_accuracy'))
                self.i += 1
                # f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

                # clear_output(wait=True)

                # ax1.set_yscale('log')
                # ax1.plot(self.x, self.losses, label="loss")
                # ax1.plot(self.x, self.val_losses, label="val_loss")
                # ax1.legend()

                # ax2.plot(self.x, self.acc, label="accuracy")
                # ax2.plot(self.x, self.val_acc, label="validation accuracy")
                # ax2.legend()

                # 터미널 출력
                self.textBox_terminal.append(
                    "Epoch {} : lose = {}".format(self.i, self.losses[-1]))

                plt.plot(self.losses)

                plt.draw()
                plt.pause(0.01)

                if self.i == 5:
                    now = time.gmtime(time.time())
                    file_name = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + \
                        str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
                    plt.savefig('result_logs\\'+file_name)
                plt.clf()

        plot_losses = PlotLosses(self.textBox_terminal)

        # training model
        history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=5, steps_per_epoch=train_steps_per_epoch, validation_data=(
            X_test, Y_test), validation_steps=val_steps_per_epoch, verbose=1,  callbacks=plot_losses)

        plt.close()


if __name__ == "__main__":
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    # WindowClass의 인스턴스 생성
    myWindow = WindowClass()

    # 프로그램 화면을 보여주는 코드
    myWindow.show()

    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
