import sys, os, traceback
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtCore
import time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from back import create_dir, set_directory
from back.learning_test import InceptionV3_test1, ResNet152_test1, Vgg16_test1

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# 연결할 ui 파일의 경로 설정
UI_Path = './ui/NetworkSetting.ui'
form_class = uic.loadUiType(UI_Path)[0]

# multiThread
class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data
    
    error
        `tuple` (exctype, value, traceback.format_exc() )
    
    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress 

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, tbt, fig, canvas, settingsData, learn_train_path, learn_val_path, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.textBox_terminal = tbt
        self.fig = fig
        self.canvas = canvas
        self.settingsData = settingsData
        self.learn_train_path = learn_train_path
        self.learn_val_path = learn_val_path
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()    

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress        

    @pyqtSlot()
    def run(self):
        if self.settingsData[0] == 'VGG':
            print('VGG')
            print(self.learn_train_path, self.learn_val_path)
            Vgg16_test1.Learn(
                self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, self.textBox_terminal, self.fig, self.canvas)
        elif self.settingsData[0] == 'InceptionV3':
            print('Inception')
            InceptionV3_test1.Learn(
                self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, self.textBox_terminal, self.fig, self.canvas)
        elif self.settingsData[0] == 'ResNet152':
            print('ResNet')
            ResNet152_test1.Learn(
                self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, self.textBox_terminal, self.fig, self.canvas)


# preprocess setting popup
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
        mainLayout.addWidget(self.formAugmentation)
        # mainLayout.addWidget(self.formDataPreprocessing)
        mainLayout.addWidget(self.formNueralNetwork)
        mainLayout.addWidget(self.formLearn)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

        self.setWindowTitle("Train Settings")

    def createFormGroupBox(self):
        # Augmentation
        self.formAugmentation = QGroupBox("Augmentation")
        layout = QFormLayout()
        self.checkBoxHorizantal = QCheckBox("Horizantal Flip", self)
        layout.addRow(self.checkBoxHorizantal)
        self.checkBoxVertical = QCheckBox("Vertical Flip", self)
        layout.addRow(self.checkBoxVertical)
        self.checkBoxRotation90 = QCheckBox("Rotation 90", self)
        layout.addRow(self.checkBoxRotation90)
        self.checkBoxRotation180 = QCheckBox("Rotation 180", self)
        layout.addRow(self.checkBoxRotation180)
        self.formAugmentation.setLayout(layout)
        # data preprocessing
        # self.formDataPreprocessing = QGroupBox("Data Preprocessing")
        # layout = QFormLayout()
        # self.lineTarget = QLineEdit()
        # layout.addRow(QLabel("target size:"), self.lineTarget)
        # layout.addRow(QLabel("class mode:"), QComboBox())
        # self.lineBatch = QLineEdit()
        # layout.addRow(QLabel("batch size:"), self.lineBatch)
        # self.lineRgb = QLineEdit()
        # layout.addRow(QLabel("rgb:"), self.lineRgb)
        # self.formDataPreprocessing.setLayout(layout)
        # nn setting
        self.formNueralNetwork = QGroupBox("Nueral Network")
        layoutNN = QFormLayout()
        self.comboBoxNN = QComboBox()
        self.comboBoxNN.addItems(["VGG", "InceptionV3", "ResNet152"])
        layoutNN.addRow(QLabel("select NN:"), self.comboBoxNN)
        self.formNueralNetwork.setLayout(layoutNN)
        # Learn Settings
        self.formLearn = QGroupBox("Learn Settings")
        layoutLS = QFormLayout()
        self.lineEpochs = QLineEdit()
        # onlyInt = QIntValidator()
        # self.lineEpochs.setValidator(onlyInt)
        layoutLS.addRow(QLabel("Epochs"), self.lineEpochs)
        self.formLearn.setLayout(layoutLS)

    def accept(self):
        settings_data = []
        settings_data.append(self.comboBoxNN.currentText())
        aug = [False, False, 0]
        if self.checkBoxHorizantal.isChecked() == True:
            aug[0] = True
        if self.checkBoxVertical.isChecked() == True:
            aug[1] = True
        if self.checkBoxRotation90.isChecked() == True:
            aug[2] = 90
        # if self.checkBoxRotation180.isChecked() == True:
        #     WindowClass.settingsData.append("Rotation 180")
        settings_data.append(aug)
        settings_data.append(int(self.lineEpochs.text()))
        WindowClass.settingsData = settings_data
        print(WindowClass.settingsData)
        self.hide()


class ProjectNameClass(QDialog):
    def __init__(self):
        super().__init__()
        self.lineName = QLineEdit()
        self.btnOk = QPushButton('OK')
        self.btnOk.clicked.connect(self.projectNameFn)
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.lineName)
        mainLayout.addWidget(self.btnOk)
        self.setLayout(mainLayout)

    def projectNameFn(self):
        WindowClass.projectName = self.lineName.text()
        self.hide()


# MainWindow
class WindowClass(QMainWindow, form_class):
    mainImg = "C:/Users/multicampus/Desktop/s03p31c203/Project/front/test_img/test1.png"
    settingsData = []
    projectName = ''
    # learn_train_path = ''
    # learn_val_path = ''

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 기본 설정?>
        self.learnSettingDisplay = AnotherFormLayout()
        self.projectNameDisplay = ProjectNameClass()
        pixmap = QtGui.QPixmap(self.mainImg)
        self.imgLabel.setPixmap(pixmap)
        # 버튼별 함수 실행
        self.btnCreateProject.clicked.connect(self.createProjectFn)
        self.btnDataLoad.clicked.connect(self.dataLoadFn)
        self.btnLearnSettings.clicked.connect(self.learnSettingsFn)
        self.dirTreeView.doubleClicked.connect(self.fileViewFn)
        self.btnTraining.clicked.connect(self.training)
        self.btnTest.clicked.connect(self.testing)
        # 터미널
        self.textBox_terminal.setGeometry(QtCore.QRect(0, 510, 1200, 190))
        # live loss plot
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.testLayout.addWidget(self.canvas)

        self.setWindowTitle('SSAKIT')

    def createProjectFn(self):
        if self.projectNameDisplay.isVisible():
            self.projectNameDisplay.hide()
        else:
            self.projectNameDisplay.show()

    def dataLoadFn(self):
        self.pathName = QFileDialog.getExistingDirectory(self, self.tr("Open Data files"), "./",
                                                         QFileDialog.ShowDirsOnly)
        self.dirName = self.pathName.split('/')[-1]
        self.testPath = '../back/' + self.projectName
        treeModel = QFileSystemModel()
        self.dirTreeView.setModel(treeModel)
        treeModel.setRootPath(QDir.rootPath())
        self.dirTreeView.setRootIndex(treeModel.index(self.testPath))
        create_dir.create_dir_flow(self.projectName)
        set_directory.set_directory(
            self.projectName, self.dirName, self.pathName)
        # self.setWindowTitle(self.projectName)

    def learnSettingsFn(self, checked):
        if self.learnSettingDisplay.isVisible():
            self.learnSettingDisplay.hide()
        else:
            self.learnSettingDisplay.show()
        self.learn_train_path = self.projectName + "/train"
        self.learn_val_path = self.projectName + "/validation"
        self.learn_test_path = self.projectName + "/test"

    def fileViewFn(self, index):
        self.mainImg = self.dirTreeView.model().filePath(index)
        self.dirTreeView.hideColumn(1)
        self.dirTreeView.hideColumn(2)
        self.dirTreeView.hideColumn(3)
        pixmap = QtGui.QPixmap(self.mainImg)
        self.imgLabel.setPixmap(pixmap)

    # ▼▼ codes for multiTrhead ▼▼
    def progress_fn(self, n):
        print("%d%% done" % n)
 
    def print_output(self, s):
        print(s)
        
    def thread_complete(self):
        print("THREAD COMPLETE!")
    # ▲▲ codes for multiTrhead ▲▲

    def training(self):
        print('train')
        # Pass the function to execute
        worker = Worker(self.textBox_terminal, self.fig, self.canvas, self.settingsData, self.learn_train_path, self.learn_val_path) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker) 
    
    def 대충_테스트_함수(self, test_path):
        # 실제론 임포트 할거임
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        predictions = model.predict(test_images)
        return test_loss, test_acc, predictions

    def testing(self):
        예측로스, 예측어큐러시, 예측리스트 = self.대충_테스트_함수(self.learn_test_path)


if __name__ == "__main__":
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    # WindowClass의 인스턴스 생성
    myWindow = WindowClass()

    # 프로그램 화면을 보여주는 코드
    myWindow.show()

    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
