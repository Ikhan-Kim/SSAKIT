import sys
import os
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtCore, QtGui
from PIL import Image
import time

from django.conf import settings
import numpy as np
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

# 연결할 ui 파일의 경로 설정
UI_Path = './ui/NetworkSetting.ui'
form_class = uic.loadUiType(UI_Path)[0]

class AnotherFormLayout(QDialog):
    NumGridRows = 3
    NumButtons = 4

    def __init__(self):
        super().__init__()
        self.createFormGroupBox()
        
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
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

class WindowClass(QMainWindow, form_class) :
    mainImg = "C:/Users/multicampus/Desktop/s03p31c203/Project/front/test_img/test1.png"
    def __init__(self) :
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
        # self.btnTraining.clicked.connect(self.training)

    def dataLoadFn(self):
        self.dirName = QFileDialog.getExistingDirectory(self, self.tr("Open Data files"), "./",
                                                  QFileDialog.ShowDirsOnly)
        treeModel = QFileSystemModel()
        self.dirTreeView.setModel(treeModel)
        treeModel.setRootPath(QDir.rootPath())
        self.dirTreeView.setRootIndex(treeModel.index(self.dirName))
        # if self.dirWidget.isVisible():
        #     self.dirWidget.hide()
        # else:
        #     self.dirWidget.show()
        
        self.btnTraining.clicked.connect(self.training)
        # 터미널
        self.textBox_terminal.setGeometry(QtCore.QRect(0, 510, 1280,190))
        self.textBox_terminal.setText("asdf")
        self.process = QtCore.QProcess()
        self.process.readyReadStandardError.connect(self.onReadyReadStandardError)
        self.process.readyReadStandardOutput.connect(self.onReadyReadStandardOutput)
        # 결과 로그
        pixmap = QtGui.QPixmap("result_logs\\20201117177.png")
        pixmap = pixmap.scaledToWidth(480)
        self.result_figure.setPixmap(pixmap)

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
        #path
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
        history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=5, steps_per_epoch=train_steps_per_epoch, validation_data = (X_test, Y_test), validation_steps=val_steps_per_epoch, verbose = 1,  callbacks=callbacks)

        # 터미널에 히스토리 출력
        loss_history = history.history["loss"] #type is list
        for i in range(len(loss_history)):
            self.textBox_terminal.append("Epoch {} : lose = {}".format(i, loss_history[i]))


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

        # plt.show()
        now = time.gmtime(time.time())
        file_name = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
        plt.savefig('result_logs\\'+file_name)


if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv) 

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass() 

    #프로그램 화면을 보여주는 코드
    myWindow.show()

    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()