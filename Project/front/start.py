import sys, os, traceback, shutil
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import time
from PIL import Image
from stat import *

from mymodules import create_dir, set_directory
from mymodules import InceptionV3_test1, ResNet50_test1, Vgg16_test1, EfficientnetB0_test1, test_function2, Retrain_model

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ClassEditWidget import ClassEditWidget

# DB 연동
import sqlite3

from datetime import datetime

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
 

# 연결할 ui 파일의 경로 설정
form = resource_path('./NetworkSetting.ui')
form_class = uic.loadUiType(form)[0]

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

    def __init__(self, order, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.order = order
        self.settingsData = myWindow.settingsData
        self.learn_train_path = myWindow.learn_train_path
        self.learn_val_path = myWindow.learn_val_path
        self.model_name = myWindow.test_model_name
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()    

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress        

    @pyqtSlot()
    def run(self):
        myWindow.btnDisable()
        if self.order == 'training':
            if self.settingsData[4] == 'new':
                if self.settingsData[0] == 'VGG':
                    print('VGG')
                    print(self.learn_train_path, self.learn_val_path)
                    Vgg16_test1.Learn(
                        self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, myWindow)
                elif self.settingsData[0] == 'InceptionV3':
                    print('Inception')
                    InceptionV3_test1.Learn(
                        self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, myWindow)
                elif self.settingsData[0] == 'ResNet50':
                    print('ResNet')
                    ResNet50_test1.Learn(
                        self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, myWindow)
                elif self.settingsData[0] == 'EfficientnetB0':
                    print('EfficientnetB0')
                    EfficientnetB0_test1.Learn(
                        self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, myWindow)
            elif self.settingsData[4] == 'continue':
                # 여기에 retrain
                Retrain_model.Retrain(
                    self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, myWindow, 'checkpoint/' + self.settingsData[0]
                )
            myWindow.textBox_terminal.moveCursor(QtGui.QTextCursor.End)
        elif self.order == 'test':
            test_function2.test(self.model_name, myWindow)
            myWindow.label_14.hide()
            myWindow.TestResultWidget.setStyleSheet("background-color : rgb(64, 64, 64);")
            myWindow.ResultSave.show()
            myWindow.ResultNo.show()

            myWindow.label_6.show()
            myWindow.label_8.show()
            myWindow.label_9.show()
            myWindow.confusionMatrixTable.show()
            myWindow.precisionTable.show()
            myWindow.recallTable.show()
            myWindow.label_10.show()
            myWindow.label_11.show()
            myWindow.accuracyTable.show()
            myWindow.label_12.show()
            myWindow.macroPrecisionLabel.show()
            myWindow.label_13.show()
        
            myWindow.TestResultWidget.show()
        elif self.order == 'dataLoad':
            for idx, dirName in enumerate(myWindow.class_names):
                set_directory.set_directory(
                    myWindow.projectName, dirName, myWindow.pathName + '/' + dirName, idx
                )

        myWindow.btnEnable()

# preprocess setting popup #train wizard
class AnotherFormLayout(QDialog):
    NumGridRows = 3
    NumButtons = 4
    sw_new_continue = 'new'
    colorSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setGeometry(460,200,900,600) #옆, 위,  width, height
        self.createFormGroupBox()
        self.setTLTables()

        self.setStyleSheet("background-color: #847f7f;")

        buttonBox = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        buttonBox.setStyleSheet("background-color: rgb(241, 127, 66); font: 12pt 'a디딤돌'; color: rgb(255, 255,255);")

        self.formContinueNetwork.hide()
        self.new_learn = QPushButton("New")
        self.new_learn.setStyleSheet("background-color: rgb(241, 127, 66); font: 12pt 'a디딤돌'; color: rgb(255, 255,255);")
        self.continue_learn = QPushButton("Continue")
        self.continue_learn.setStyleSheet("background-color: rgb(175, 171, 171); font: 12pt 'a디딤돌'; color: rgb(255, 255,255);")
        self.buttonsWidget = QWidget()
        self.buttonsWidgetLayout = QHBoxLayout(self.buttonsWidget)
        self.buttonsWidgetLayout.addWidget(self.new_learn)
        self.buttonsWidgetLayout.addWidget(self.continue_learn)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formModelName)
        mainLayout.addWidget(self.formAugmentation)
        mainLayout.addWidget(self.buttonsWidget)
        mainLayout.addWidget(self.formNeuralNetwork)
        mainLayout.addWidget(self.formContinueNetwork)
        mainLayout.addWidget(self.formLearn)
        mainLayout.addWidget(self.formTrainList)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)
        self.new_learn.clicked.connect(self.newFn)
        self.continue_learn.clicked.connect(self.continueFn)
        self.setWindowTitle("Train Wizard")
    
    def createFormGroupBox(self):
        # model name
        self.formModelName = QGroupBox("Set Model Name")
        self.formModelName.setStyleSheet("font: 12pt 'a디딤돌'; color: rgb(255, 255, 255); ")
        layoutMN = QFormLayout()
        self.setModelName = QLineEdit()
        self.setModelName.setPlaceholderText("모델 이름 설정")
        self.setModelName.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        layoutMN.addRow(QLabel("Model Name : "), self.setModelName)
        self.formModelName.setLayout(layoutMN)

        # Augmentation
        self.formAugmentation = QGroupBox("Augmentation")
        self.formAugmentation.setStyleSheet("font: 12pt 'a디딤돌'; color: rgb(255, 255, 255); ")
        layout = QFormLayout()
        self.checkBoxHorizantal = QCheckBox("[H] Horizantal Flip", self)
        layout.addRow(self.checkBoxHorizantal)
        self.checkBoxVertical = QCheckBox("[V] Vertical Flip", self)
        layout.addRow(self.checkBoxVertical)
        self.checkBoxBrightness = QCheckBox("[B] Brightness", self)
        layout.addRow(self.checkBoxBrightness)
        self.checkBoxCutout = QCheckBox("[C] Cutout", self)
        layout.addRow(self.checkBoxCutout)
        self.checkBoxRotation90 = QRadioButton("[R-90] Rotation 90", self)
        layout.addRow(self.checkBoxRotation90)
        self.checkBoxRotation180 = QRadioButton("[R-180] Rotation 180", self)
        layout.addRow(self.checkBoxRotation180)
        self.formAugmentation.setLayout(layout)
        # nn setting
        self.formNeuralNetwork = QGroupBox("New Neural Network")
        self.formNeuralNetwork.setStyleSheet("font: 12pt 'a디딤돌'; color: rgb(255, 255, 255); ")
        layoutNN = QFormLayout()
        self.comboBoxNN = QComboBox()
        self.comboBoxNN.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        self.comboBoxNN.addItems(["VGG", "ResNet50", "EfficientnetB0"])
        layoutNN.addRow(QLabel("select :"), self.comboBoxNN)
        self.formNeuralNetwork.setLayout(layoutNN)
        # continue nn setting
        self.formContinueNetwork = QGroupBox("Continue Neural Network")
        self.formContinueNetwork.setStyleSheet("font: 12pt 'a디딤돌'; color: rgb(255, 255, 255); ")
        layoutContinue = QFormLayout()
        self.comboBoxContinue = QComboBox()
        self.comboBoxContinue.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        continue_list = os.listdir('./checkpoint')
        self.comboBoxContinue.addItems(continue_list)
        layoutContinue.addRow(QLabel("select :"), self.comboBoxContinue)
        self.formContinueNetwork.setLayout(layoutContinue)
        # Learn Settings
        self.formLearn = QGroupBox("Learn Settings")
        self.formLearn.setStyleSheet("font: 12pt 'a디딤돌'; color: rgb(255, 255, 255); ")
        layoutLS = QFormLayout()
        self.lineEpochs = QSpinBox()
        self.lineEpochs.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        self.lineEpochs.setRange(1, 10000)
        # onlyInt = QIntValidator()
        # self.lineEpochs.setValidator(onlyInt)
        layoutLS.addRow(QLabel("Epochs"), self.lineEpochs)
        self.formLearn.setLayout(layoutLS)

        # Train List
        self.formTrainList = QGroupBox("previous train parameter")
        self.formTrainList.setStyleSheet("font: 12pt 'a디딤돌'; color: rgb(255, 255, 255); ")
        self.trainList = QTableWidget()
        self.trainList.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        tableLayout = QFormLayout()
        tableLayout.addRow(self.trainList)
        self.formTrainList.setLayout(tableLayout)

    def accept(self):
        checkString = '\\/:*?\"<>|'
        if self.setModelName.text() == "":
            myWindow.warningMSG("주의", "모델 이름을 입력해 주세요.")
        else:
            for c in checkString:
                if c in self.setModelName.text():
                    myWindow.warningMSG("주의", "모델 이름에 다음 문자들은 포함 될 수 없습니다. : \n \\ / : * ? \" < > |")
                    break
            else:
                if os.path.isfile('./checkpoint/' + self.comboBoxNN.currentText() + '_' + self.setModelName.text() + '.h5'):
                    reply = QMessageBox.question(self, 'Message',
                        "같은 이름이 모델이 있습니다. 덮어 씌우시겠습니까?", QMessageBox.Yes | 
                        QMessageBox.No, QMessageBox.No)

                    if reply == QMessageBox.Yes:
                        print("YES")
                        settings_data = []
                        if self.sw_new_continue == 'new':
                            settings_data.append(self.comboBoxNN.currentText())
                        else:
                            settings_data.append(self.comboBoxContinue.currentText())
                        aug = [False, False, None, False, False]
                        if self.checkBoxHorizantal.isChecked() == True:
                            aug[0] = True
                        if self.checkBoxVertical.isChecked() == True:
                            aug[1] = True
                        if self.checkBoxBrightness.isChecked() == True:
                            aug[2] = [0.2, 1.2]
                        if self.checkBoxRotation90.isChecked() == True:
                            aug[3] = 90
                        if self.checkBoxRotation180.isChecked() == True:
                            aug[3] = 180
                        if self.checkBoxCutout.isChecked() == True:
                            aug[4] = True
                        settings_data.append(aug)
                        settings_data.append(int(self.lineEpochs.text()))
                        settings_data.append(self.setModelName.text())
                        settings_data.append(self.sw_new_continue)
                        WindowClass.settingsData = settings_data
                        self.colorSignal.emit()
                        print(WindowClass.settingsData)
                        self.hide()
                        
                    else:
                        print("NO")
                else:
                    settings_data = []
                    if self.sw_new_continue == 'new':
                        settings_data.append(self.comboBoxNN.currentText())
                    else:
                        settings_data.append(self.comboBoxContinue.currentText())
                    aug = [False, False, None, False, False]
                    if self.checkBoxHorizantal.isChecked() == True:
                        aug[0] = True
                    if self.checkBoxVertical.isChecked() == True:
                        aug[1] = True
                    if self.checkBoxBrightness.isChecked() == True:
                        aug[2] = [0.2, 1.2]
                    if self.checkBoxRotation90.isChecked() == True:
                        aug[3] = 90
                    if self.checkBoxRotation180.isChecked() == True:
                        aug[3] = 180
                    if self.checkBoxCutout.isChecked() == True:
                        aug[4] = True
                    settings_data.append(aug)
                    settings_data.append(int(self.lineEpochs.text()))
                    settings_data.append(self.setModelName.text())
                    settings_data.append(self.sw_new_continue)
                    WindowClass.settingsData = settings_data
                    self.colorSignal.emit()
                    print(WindowClass.settingsData)
                    self.hide()

    def newFn(self):
        self.sw_new_continue = 'new'
        self.formNeuralNetwork.show()
        self.formContinueNetwork.hide()
        self.new_learn.setStyleSheet("background-color: rgb(241, 127, 66); font: 12pt 'a디딤돌'; color: rgb(255, 255,255);")
        self.continue_learn.setStyleSheet("background-color: rgb(175, 171, 171); font: 12pt 'a디딤돌'; color: rgb(255, 255,255);")

    def continueFn(self):
        self.sw_new_continue = 'continue'
        self.formNeuralNetwork.hide()

        continue_list = os.listdir('./checkpoint')
        self.comboBoxContinue.clear()
        self.comboBoxContinue.addItems(continue_list)

        self.formContinueNetwork.show()
        self.new_learn.setStyleSheet("background-color: rgb(175, 171, 171); font: 12pt 'a디딤돌'; color: rgb(255, 255,255);")
        self.continue_learn.setStyleSheet("background-color: rgb(241, 127, 66); font: 12pt 'a디딤돌'; color: rgb(255, 255,255);")
          
    # 불러온 데이터 table에 보여주기
    def setTLTables(self):
        self.trainList.setColumnCount(6)
        self.trainList.setHorizontalHeaderLabels(['Date', 'Model_Name', 'Augmentation', 'Network',  'Epochs' , 'Loss' , 'Accuracy'])
        self.trainList.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # print("setTL")
        rows = WindowClass.train_list_data
        # print("WindowClass.train_list_data",WindowClass.train_list_data)
        # Table column 수, header 설정+너비
        print("rows==========================", rows)

        cnt = len(rows)
        # print("~~~~~~~~~~~~~~~", cnt)
        self.trainList.setRowCount(cnt)

        for x in range(cnt):
            Date, Model_Name, Augmentation, Network,  Epochs , Loss , Accuracy = rows[x][1:]
            print(x, "***************************", Date, Model_Name, Augmentation, Network,  Epochs , Loss , Accuracy)
            self.trainList.setItem(x, 0, QTableWidgetItem(Date))
            self.trainList.setItem(x, 1, QTableWidgetItem(Model_Name))
            self.trainList.setItem(x, 2, QTableWidgetItem(Augmentation))
            self.trainList.setItem(x, 3, QTableWidgetItem(Network))
            self.trainList.setItem(x, 4, QTableWidgetItem(str(Epochs)))
            self.trainList.setItem(x, 5, QTableWidgetItem(str(Loss)))
            self.trainList.setItem(x, 6, QTableWidgetItem(str(Accuracy)))
    
# open project
class ProjectNameClass(QDialog):
    # Signal 선언부
    send_valve_popup_signal = pyqtSignal(bool, name='sendValvePopupSignal')

    nameSignal = pyqtSignal()
    # project_list = os.listdir('./learnData/')
    def __init__(self):
        super().__init__()

        self.clickedRow = ""
        self.setGeometry(660,200,600,600)

        # 기본 구조
        self.setStyleSheet("background-color: #847f7f;")
        self.project_list = os.listdir('./learnData/')
        self.setWindowTitle('Open Project')
        self.formLoadProject = QGroupBox("기존 프로젝트 불러오기")
        self.formLoadProject.setStyleSheet("font: 12pt 'a디딤돌'; color: rgb(255, 255, 255);")
        self.formNewProject = QGroupBox("새 프로젝트 생성하기")
        self.formNewProject.setStyleSheet("font: 12pt 'a디딤돌'; color: rgb(255, 255, 255);")
        loadlayout = QFormLayout()
        newlayout = QFormLayout() 
        
        # 기존 프로젝트 불러오기
        self.loadTable = QTableWidget()
        self.loadTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.loadTable.setStyleSheet("color: rgb(0, 0, 0); background-color: rgb(255, 255, 255);")
        self.createTable()
        self.loadTable.cellClicked.connect(self.cellClick)
        self.loadTable.cellDoubleClicked.connect(self.cellDoubleClick)
        buttonBox = QDialogButtonBox()
        buttonBox.addButton('삭제하기', QDialogButtonBox.AcceptRole)
        buttonBox.addButton('불러오기', QDialogButtonBox.RejectRole)
        buttonBox.setStyleSheet("background-color: rgb(241, 127, 66);")
        buttonBox.accepted.connect(self.pjtDelete)
        buttonBox.rejected.connect(self.pjtSelect)

        # 새 프로젝트 생성
        self.lineName = QLineEdit()
        self.lineName.setPlaceholderText("프로젝트 이름 입력")
        self.lineName.setFixedWidth(450)
        self.lineName.setStyleSheet("font: 10pt 'a스마일L'; background-color: rgb(255, 255, 255); color: black;")
        self.btnOk = QPushButton('생성하기')
        self.btnOk.setStyleSheet("font: 12pt 'a디딤돌'; background-color: rgb(241, 127, 66);")
        self.btnOk.clicked.connect(self.projectNameFn)
        loadlayout.addRow(self.loadTable)
        loadlayout.addRow(buttonBox)
        newlayout.addRow(self.lineName, self.btnOk)
        mainLayout = QVBoxLayout()
        self.formLoadProject.setLayout(loadlayout)
        self.formNewProject.setLayout(newlayout)
        mainLayout.addWidget(self.formLoadProject)
        mainLayout.addWidget(self.formNewProject)
        self.setLayout(mainLayout)

    def createTable(self):
        self.loadTable.setColumnCount(1)
        self.loadTable.setHorizontalHeaderLabels(['프로젝트 이름'])
        self.loadTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.loadTable.setRowCount(len(self.project_list))
        for x in range(len(self.project_list)):
            self.loadTable.setItem(x, 0, QTableWidgetItem(self.project_list[x]))
            # self.loadTable.setItem(x, 1, QTableWidgetItem("선택"))
            # self.loadTable.setItem(x, 2, QTableWidgetItem("❌"))

    def cellClick(self, row, column):
        self.clickedRow = row
        print(self.clickedRow)
    
    def cellDoubleClick(self, row, column):
        WindowClass.projectName = self.project_list[self.clickedRow]
        self.nameSignal.emit()
        self.hide()  

    def pjtSelect(self):
        if self.clickedRow =="":
            myWindow.warningMSG("알림", "프로젝트를 선택해 주세요")
        else:
            print('선택클릭')
            WindowClass.projectName = self.project_list[self.clickedRow]
            self.nameSignal.emit()
            self.hide()            
    
    def pjtDelete(self):
        if self.clickedRow =="":
            myWindow.warningMSG("알림", "프로젝트를 선택해 주세요")
        else:
            a = QMessageBox.question(self, "삭제 확인", "정말로 삭제 하시겠습니까?",
                                 QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
            if a == QMessageBox.Yes:
                del_path = './learnData/' + self.project_list[self.clickedRow]
                shutil.rmtree(del_path)
                self.project_list = os.listdir('./learnData/')
                self.clickedRow = ""
                self.createTable()

    def projectNameFn(self):
        checkString = '\\/:*?\"<>|'
        if self.lineName.text() == "":
            myWindow.warningMSG("알림", "프로젝트 이름을 입력해주세요")
        else:
            for c in checkString:
                if c in self.lineName.text():
                    myWindow.warningMSG("주의", "프로젝트 이름에 다음 문자들은 포함 될 수 없습니다. : \n \\ / : * ? \" < > |")
                    break
            else:
                WindowClass.projectName = self.lineName.text()
                self.nameSignal.emit()
                self.hide()


# Test Model Seclet #영환오빠
class TestModelSelect(QDialog):
    def __init__(self):
        super().__init__()
        self.clickedRow = ""
        self.setGeometry(760,300,400,400)
        self.setStyleSheet("background-color: #847f7f;")
        self.setWindowTitle("Test Model Select")

        self.label = QLabel()
        self.label = QLabel("Model Select", self)
        self.label.setStyleSheet("font: 18pt 'a로케트'; color: rgb(255, 238, 228);")

        self.listW = QListWidget()
        for i in range(len(os.listdir("./checkpoint"))):
            self.listW.addItem(os.listdir("./checkpoint")[i])
        self.listW.setStyleSheet("background-color: rgb(255, 255, 255); font: 14pt 'a디딤돌'; color: rgb(0, 0,0);")
        
        self.listW.itemClicked.connect(self.itemClick)
        self.listW.itemActivated.connect(self.itemActivated_event)


        # 확인버튼
        buttonBox = QDialogButtonBox()
        buttonBox.addButton('불러오기', QDialogButtonBox.AcceptRole)
        buttonBox.addButton('삭제하기', QDialogButtonBox.RejectRole)
        buttonBox.setStyleSheet("background-color: rgb(241, 127, 66); font: 12pt 'a디딤돌'; color: rgb(255, 255,255);")
        buttonBox.accepted.connect(self.load)
        buttonBox.rejected.connect(self.delete)

        # self.label_14.show()
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.listW)
        vbox.addWidget(buttonBox)
        self.setLayout(vbox)
        # self.setGeometry(300, 300, 300, 300)
        
    def itemClick(self, item):
        self.clickedRow = item.text()

    def load(self):
        if self.clickedRow =="":
            myWindow.warningMSG("알림", "모델을 선택해 주세요")
        else:
            self.hide()
            test_function2.test(self.clickedRow, myWindow)
            myWindow.label_14.hide()
            myWindow.TestResultWidget.setStyleSheet("background-color : rgb(64, 64, 64);")
            myWindow.ResultSave.show()
            myWindow.ResultNo.show()

            myWindow.label_6.show()
            myWindow.label_8.show()
            myWindow.label_9.show()
            myWindow.confusionMatrixTable.show()
            myWindow.precisionTable.show()
            myWindow.recallTable.show()
            myWindow.label_10.show()
            myWindow.label_11.show()
            myWindow.accuracyTable.show()
            myWindow.label_12.show()
            myWindow.macroPrecisionLabel.show()
            myWindow.label_13.show()
        
            myWindow.TestResultWidget.show()  

    def delete(self):
        if self.clickedRow =="":
            myWindow.warningMSG("알림", "모델을 선택해 주세요")
        else:
            a = QMessageBox.question(self, "삭제 확인", "정말로 삭제 하시겠습니까?",
                                 QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
            if a == QMessageBox.Yes:
                del_path = './checkpoint/' + self.clickedRow
                os.remove(del_path)
                self.clickedRow = ""
                self.listW.clear()
                for i in range(len(os.listdir("./checkpoint"))):
                    self.listW.addItem(os.listdir("./checkpoint")[i])

    def itemActivated_event(self, item):
        self.hide()

        # myWindow.test_model_name = item.text()
        # # Pass the function to execute
        # worker = Worker('test') # Any other args, kwargs are passed to the run function
        # worker.signals.result.connect(myWindow.print_output)
        # worker.signals.finished.connect(myWindow.thread_complete)
        # worker.signals.progress.connect(myWindow.progress_fn)

        # # Execute
        # myWindow.threadpool.start(worker)

        test_function2.test(item.text(), myWindow)
        myWindow.label_14.hide()
        myWindow.TestResultWidget.setStyleSheet("background-color : rgb(64, 64, 64);")
        myWindow.ResultSave.show()
        myWindow.ResultNo.show()

        myWindow.label_6.show()
        myWindow.label_8.show()
        myWindow.label_9.show()
        myWindow.confusionMatrixTable.show()
        myWindow.precisionTable.show()
        myWindow.recallTable.show()
        myWindow.label_10.show()
        myWindow.label_11.show()
        myWindow.accuracyTable.show()
        myWindow.label_12.show()
        myWindow.macroPrecisionLabel.show()
        myWindow.label_13.show()
    
        myWindow.TestResultWidget.show()
        

# MainWindow
class WindowClass(QMainWindow, form_class):
    mainImg = ""
    settingsData = []
    class_names = []
    class_data = []
    projectName = ''
    learnDataPath = ''
    learn_train_path = ''
    # class 갯수, train img 갯수, val img 갯수 순임.
    learn_num_data = []
    sIMG = ""
    train_list_data = []
    test_model_name = ''
    learn_val_path = ''
    send_valve_popup_signal = pyqtSignal(bool, name='sendValvePopupSignal')

    # colors 리스트
    colors = [
        "#EA341B", "#EADA1B", "#71EA1B", "#1BEAD4", "#1B41EA",
        "#E71BEA", "#EC9576", "#2A9614", "#144E96", "#521496",
        "#48C9B0", "#F1C40F", "#5B2C6F ", "#A2D9CE", "#EC7063",
        "#154360", "#F7DC6F", "#AED6F1", "#F09D28", "#E912C4",
        "#60E91A", "#9E314C", "#F39C12", "#10A69B", "#A6A110",
    ]

    # 임시 데이터
    # horizental__, vertical__, brightness__, rotation__, epochs__, model_name__, loss__, accuracy__ = True, False, True, 90, 100, "모델이름", 999, 9999
    # settingsData_DY = ['학습모델', [horizental__, vertical__, brightness__, rotation__], epochs__, model_name__, loss__, accuracy__]
    # Date created at
    
    def __init__(self) :     
        super().__init__()
        # design
        # changing the background color to yellow 
        self.setStyleSheet("background-color: #847f7f;")

        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0)
        self.pushButton_5.hide()
        self.label_4.hide()
        self.setWindowIcon(QtGui.QIcon('./assets/img/main_icon.jpg'))
        # 기본 설정?>
        
        self.projectNameDisplay = ProjectNameClass()
        # self.testModelSelectDisplay = TestModelSelect()
        pixmap = QtGui.QPixmap(self.mainImg)
        self.imgLabel.setPixmap(pixmap)
        self.btnHome.setIcon(QIcon('./assets/img/home.jpg'))
        self.btnHome.setIconSize(QSize(50, 50)) 
        self.btnOpenDir.setIcon(QIcon('./assets/img/folder.jpg'))
        self.btnOpenDir.setIconSize(QSize(23, 23)) 
        self.btnDataLoad.setIcon(QIcon('./assets/img/imageUpload.jpg'))
        self.btnDataLoad.setIconSize(QSize(25, 25))        
        self.megaphone_2.setIcon(QIcon('./assets/img/megaphone1.jpg'))
        self.megaphone_2.setIconSize(QSize(30, 30))
        self.f5Btn.setIcon(QIcon('./assets/img/sync.jpg'))
        self.f5Btn.setIconSize(QSize(20, 20))
        
        # 버튼별 함수 실행
        self.btnCreateProject.clicked.connect(self.createProjectFn)
        # self.btnDataLoad.setStyleSheet("background-image: url(front\assets\img\imageUpload.png);")
        self.btnDataLoad.clicked.connect(self.dataLoadFn)
        self.btnLearnSettings.clicked.connect(self.learnSettingsFn)
        self.dirTreeView.doubleClicked.connect(self.fileViewFn)
        self.btnTraining.clicked.connect(self.training)
        self.projectNameDisplay.nameSignal.connect(self.createNameFn)
        self.btnTest.clicked.connect(self.test)
        self.btnOpenDir.clicked.connect(self.openDirFn)
        self.btnHome.clicked.connect(self.moveHome)
        self.ResultNo.clicked.connect(self.rmh5file)
        self.TestResultWidget.hide()
        self.label_6.hide()
        self.label_8.hide()
        self.label_9.hide()
        self.confusionMatrixTable.hide()
        self.precisionTable.hide()
        self.recallTable.hide()
        self.label_10.hide()
        self.label_11.hide()
        self.accuracyTable.hide()
        self.label_12.hide()
        self.macroPrecisionLabel.hide()
        self.label_13.hide()

        # 터미널
        # live loss plot
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.plotLayout.addWidget(self.canvas)

        self.setWindowTitle('SSAKIT')

        # class Edit btn 클릭 => 위젯 열기
        self.classEditBtn.clicked.connect(self.ClassEditBtnFunc)
        # edit 금지 모드
        self.classType.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 새로고침 버튼 ()
        self.f5Btn.clicked.connect(self.f5BtnFunc)

        # Navigator
        self.loadNavi()

        # TL_insert()
        self.ResultSave.clicked.connect(self.TL_insert)
        self.ResultNo.clicked.connect(self.RMh5file)

    # 체크포인트 삭제_다영 임시
    def RMh5file(self):
        self.ResultSave.hide()
        self.ResultNo.hide()
        self.TestResultWidget.setStyleSheet("background-color: rgb(132, 127, 127);")
        self.resultWidgetMSG.setText("학습한 모델이 삭제 되었습니다 !")

    def btnColorChange(self, btn):
        # print(" btn change", btn)
        btns = [self.btnLearnSettings,  self.btnTraining, self.btnTest]
        btns.remove(btn)
        btn.setStyleSheet("background-color: rgb(241, 127, 66); font: 12pt 'a로케트'; color: rgb(255, 255, 255);")
        for b in btns:
            b.setStyleSheet("background-color: #ffeee4; font: 12pt 'a로케트'; color: rgb(0, 0, 0);")

    def createNameFn(self):
        # sql 연동
        self.sqlConnect()
        self.trainListSqlConnect()
        self.learnSettingDisplay = AnotherFormLayout()
        self.learnSettingDisplay.colorSignal.connect(self.changeColorFn)
        # # ClassEditWidget 불러오기
        # self.openClassEditWidget = ClassEditWidget(WindowClass.class_data, self.dbName)
        self.setWindowTitle('SSAKIT -' + self.projectName)
        self.learnDataPath = './learnData/' + self.projectName
        create_dir.create_dir_flow(self.projectName)
        treeModel = QFileSystemModel()
        self.dirTreeView.setModel(treeModel)
        treeModel.setRootPath(QDir.rootPath())
        self.dirTreeView.setRootIndex(treeModel.index(self.learnDataPath))
        self.dirTreeView.hideColumn(1)
        self.dirTreeView.hideColumn(2)
        self.dirTreeView.hideColumn(3)
        self.pjtTitle.setText(self.projectName)
        self.class_names = os.listdir(self.learnDataPath + '/train')
        self.mainWidget.hide()
        
    def changeColorFn(self):
        self.btnColorChange(self.btnTraining)
        self.infoMSG.setText("Training 버튼을 클릭해 주세요.")
        self.cnt_file()

    def createProjectFn(self):
        if self.projectNameDisplay.isVisible():
            self.projectNameDisplay.hide()
        else:
            self.projectNameDisplay.show()
    
    def moveHome(self):
        self.mainWidget.show()
        # self.createProjectFn()
        self.tabWidget.setCurrentIndex(0)
        self.btnColorChange(self.btnLearnSettings)

    def dataLoadFn(self):
        if self.projectName:
            self.pathName = QFileDialog.getExistingDirectory(self, self.tr("Open Data files"), "./",
                                                            QFileDialog.ShowDirsOnly)
            if self.pathName:
                self.class_names = os.listdir(self.pathName)
                print(self.pathName)
                
                if self.pathName:
                    # Pass the function to execute
                    worker = Worker('dataLoad') # Any other args, kwargs are passed to the run function
                    worker.signals.result.connect(myWindow.print_output)
                    worker.signals.finished.connect(myWindow.thread_complete)
                    worker.signals.progress.connect(myWindow.progress_fn)

                    # Execute
                    myWindow.threadpool.start(worker)
        else:
            self.warningMSG("주의", "프로젝트를 먼저 생성/선택 해주십시오.")

    def learnSettingsFn(self, checked):        
        # self.setTLTables(item_list)
        self.TL_select()
        self.learnSettingDisplay.setTLTables()
        # self.learnSettingDisplay = AnotherFormLayout()

        # ClassEditWidget 불러오기
        self.openClassEditWidget = ClassEditWidget(WindowClass.class_data, self.dbName)

        self.tabWidget.setCurrentIndex(0)
        self.class_names = os.listdir(self.learnDataPath + '/train')
        if self.projectName:
            if self.learnSettingDisplay.isVisible():
                self.learnSettingDisplay.hide()
            else:
                self.learnSettingDisplay.show()
            self.learn_train_path = 'learnData/' + self.projectName + "/train"
            self.learn_val_path = 'learnData/' + self.projectName + "/validation"
        else:
            self.warningMSG("주의", "프로젝트를 먼저 생성/선택 해주십시오.")

    def fileViewFn(self, index):
        self.mainImg = self.dirTreeView.model().filePath(index)
        pixmap = QtGui.QPixmap(self.mainImg)
        pixmap2 = pixmap.scaledToWidth(600)
        self.imgLabel.setPixmap(pixmap2)

        try:
            # do stuff
            img = Image.open(self.mainImg)
            st = os.stat(self.mainImg)
            self.fileName.setText(img.filename.split('/')[-1])
            self.fileSize.setText(str(st[ST_SIZE]))
            self.extension.setText(img.format)
            if img.mode == 'RGB':
                self.channel.setText("3")
            else:
                self.channel.setText("1")
            self.wValue.setText(str(img.width))
            self.hValue.setText(str(img.height))
            self.className.setText('_'.join(img.filename.split('/')[-2].split('_')[1:]))
        except IOError:
            # filename not an image file
            print("\'" + self.mainImg + "\' is not an image file")

       
    # ▼▼ codes for multiTrhead ▼▼
    def progress_fn(self, n):
        print("%d%% done" % n)
 
    def print_output(self, s):
        print(s)
        
    def thread_complete(self):
        print("THREAD COMPLETE!")
    # ▲▲ codes for multiTrhead ▲▲

    def warningMSG(self, title: str, content: str):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.send_valve_popup_signal.emit(True)
    
    def btnDisable(self):
        print("Buttons are disabled!")
        self.btnCreateProject.setEnabled(False)
        self.btnDataLoad.setEnabled(False)
        self.btnLearnSettings.setEnabled(False)
        self.btnTraining.setEnabled(False)
        self.btnTest.setEnabled(False)
        self.btnOpenDir.setEnabled(False)
        self.btnHome.setEnabled(False)
        self.classEditBtn.setEnabled(False)

    def btnEnable(self):
        self.btnCreateProject.setEnabled(True)
        self.btnDataLoad.setEnabled(True)
        self.btnLearnSettings.setEnabled(True)
        self.btnTraining.setEnabled(True)
        self.btnTest.setEnabled(True)
        self.btnOpenDir.setEnabled(True)
        self.btnHome.setEnabled(True)
        self.classEditBtn.setEnabled(True)

    def training(self):
        if self.settingsData:
            self.infoMSG.setText("training이 완료되면 Test 버튼을 클릭 해 주세요.")
            self.tabWidget.setCurrentIndex(1)
            self.btnColorChange(self.btnTraining)
            self.textBox_terminal.append('Ready for training...')
            # Pass the function to execute
            worker = Worker('training') # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self.print_output)
            worker.signals.finished.connect(self.thread_complete)
            worker.signals.progress.connect(self.progress_fn)

            # Execute
            self.threadpool.start(worker)
        else:
            self.warningMSG("주의", "데이터 로드 및 이미지 전처리를 먼저 실행해 주십시오.")
        # self.btnEnable()

    def test(self):
        self.infoMSG.setText("Image Viewer의 이미지를 클릭하면 CAM 을 확인 할 수 있습니다.")
        self.btnColorChange(self.btnTest)
        self.tabWidget.setCurrentIndex(2)
        self.testModelSelectDisplay = TestModelSelect()
        self.testModelSelectDisplay.show()
        self.btnColorChange(self.btnTest)
        # self.cnt_file()

    # ClassEditWidget띄우기
    def ClassEditBtnFunc(self):
        # ClassEditWidget 불러오기
        self.openClassEditWidget = ClassEditWidget(WindowClass.class_data, self.dbName)
        self.openClassEditWidget.show()

    def f5BtnFunc(self):
        print("새로고침")
        self.selectData()

    # DB) SQL 연결 및 테이블 생성
    def sqlConnect(self):
        self.dbName = WindowClass.projectName + ".db"
        try: 
            self.conn = sqlite3.connect(self.dbName, isolation_level=None)
        except:
            print("문제가 있네요!")
            exit(1)
        print("연결성공!")
        self.cur = self.conn.cursor()

        # 테이블 생성
        self.createSql = "CREATE TABLE IF NOT EXISTS classLabel (idx INTEGER PRIMARY KEY, color TEXT, label TEXT, train INTEGER, val INTEGER, test INTEGER)"
        self.cmd = self.createSql
        self.run()
 
        # 초기 데이터 삽입
        self.selectSql = "SELECT * FROM classLabel"
        self.cmd = self.selectSql
        self.run()

        item_list = [list(item[:]) for item in self.cur.fetchall()]

        if len(item_list) == 0:
            for class_data_idx, data in enumerate(self.class_data) :
                # print("class_data_idx, data : ", class_data_idx, data)
                # self.label_idx = class_data_idx
                self.color = self.colors[class_data_idx]
                self.label = data[0]
                self.train =  data[1]
                self.val =  data[2]
                self.test =  data[3]

                # print("===", self.color, self.label, self.train, self.val, self.test)

                self.cmd = "insert into classLabel(`color`, `label`, `train`, `val`, `test`) values('{}', '{}', {}, {}, {})"\
                    .format(self.color, self.label, self.train, self.val, self.test)
                self.cur.execute(self.cmd)
                self.conn.commit()

        self.selectData()

    # DB 데이터 불러오기
    def selectData(self):
        self.selectSql = "SELECT * FROM classLabel"
        self.cmd = self.selectSql
        self.run()

        item_list = [list(item[:]) for item in self.cur.fetchall()]
        self.setTables(item_list)
    
    # 불러온 데이터 table widget 에서 보여주기
    def setTables(self, rows):
        # Table column 수, header 설정+너비
        self.classType.setColumnCount(5)
        self.classType.setHorizontalHeaderLabels(['color', 'class', 'train', 'val', 'test', ""])
        self.classType.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        cnt = len(rows)
        self.classType.setRowCount(cnt)

        for x in range(cnt):
            # 리스트 내부의 column쌍은 튜플로 반환하므로 튜플의 각 값을 변수에 저장
            # print(rows)
            color, label, train, val, test = rows[x][1:]
            
            # print("rows[x]", rows[x][0], rows[x][1], rows[x][2])
            # 테이블의 각 셀에 값 입력
            # self.classType.setItem(x, 0, QTableWidgetItem(str(idx)))
            self.classType.setItem(x, 0, QTableWidgetItem(""))
            self.classType.item(x, 0).setBackground(QtGui.QColor(color))
            self.classType.setItem(x, 1, QTableWidgetItem('_'.join(label.split('_')[1:])))
            self.classType.setItem(x, 2, QTableWidgetItem(str(train)))
            self.classType.setItem(x, 3, QTableWidgetItem(str(val)))
            self.classType.setItem(x, 4, QTableWidgetItem(str(test)))

    # DB) sql문 실행 함수
    def run(self):
        self.cur.execute(self.cmd)
        self.conn.commit()

    # DB) 종료 함수
    def closeEvent(self, QCloseEvent):
        print("DB close!")
        self.conn.close()

    ########## TL DB
    # DB 연결, TL 테이블 생성
    def trainListSqlConnect(self):
        try: 
            self.conn = sqlite3.connect(self.dbName, isolation_level=None)
        except:
            print("TL 문제가 있네요!")
            exit(1)
        print("TL_연결성공!")
        self.cur = self.conn.cursor()

        # 테이블 생성
        self.createSql = "CREATE TABLE IF NOT EXISTS trainList (idx INTEGER PRIMARY KEY, Date TEXT, Model_Name TEXT, Augmentation TEXT, Network TEXT, Epochs INTEGER, Loss INTEGER, Accuracy INTEGER)"
        self.cmd = self.createSql
        self.cur.execute(self.cmd)
        self.conn.commit()

        self.TL_select()
        
        # 테스트용 코드 / 돌아가는 거 확인하면 save 버튼 눌렀을 때 DB에 저장될 수 있도록 바꾸기
        # self.TL_insert()

    def TL_select(self):
        # 보여주기
        # DB 연결
        try: 
            self.conn = sqlite3.connect(self.dbName, isolation_level=None)
        except:
            print("TL 문제가 있네요!2")
            exit(1)
        print("연결성공!")
        self.cur = self.conn.cursor()

        # Data 선택해서 setTables에 보내주기
        self.selectTLSql = "SELECT * FROM trainList"
        self.cmd = self.selectTLSql
        self.cur.execute(self.cmd)
        self.conn.commit()

        WindowClass.train_list_data = [list(item[:]) for item in self.cur.fetchall()]
        print(self.train_list_data)

    def TL_insert(self):
        self.ResultSave.hide()
        self.ResultNo.hide()
        self.TestResultWidget.setStyleSheet("background-color: rgb(241, 127, 66);")
        self.resultWidgetMSG.setText("학습한 모델이 저장 되었습니다 !")
        # save 버튼, No 버튼 hide

        # insert
        # settingsData_DY = [['학습모델', [horizental__, vertical__, brightness__, rotation__], epochs__, model_name__, loss__, accuracy__]
        print("settingsData : ", self.settingsData)
        present = datetime.now()
        month = str(present.month)
        day = str(present.day)
        hour = str(present.hour)
        minute = str(present.minute)
        # second = str(present.second)
        # (1) Data 받아오기
        
        if self.settingsData[:] == 0:
            pass
        else:
            Date = month +'/' + day +' ' + hour + ':' + minute
            print(Date)
            Network, Augmentation, Epochs, Model_Name, sibal, Loss , Accuracy = self.settingsData[:]
            # print("Network, Augmentation, Epochs, Model_Name, Loss , Accuracy", Network, Augmentation, Epochs, Model_Name, Loss , Accuracy)

            Aug_lst = []

            for idx, a in enumerate(Augmentation):
                if a != False:
                    if idx == 0:
                        Aug_lst.append('H')
                    elif idx == 1:
                        Aug_lst.append('V')
                    elif idx == 2:
                        if a != None:
                            Aug_lst.append('B')
                    elif idx == 3:
                        if a == 90:
                            Aug_lst.append('R-90')
                        elif a == 180:
                            Aug_lst.append('R-180')
                        else:
                            continue
                    elif idx == 4:
                        Aug_lst.append('C')
                else:
                    continue

            Aug_str = ', '.join(Aug_lst)
            print("==================== Aug_str ==================", Aug_str)

            # (2) insert
            self.insertSql = "INSERT INTO trainList (Date, Model_Name, Augmentation, Network, Epochs, Loss, Accuracy) VALUES (?,?,?,?,?,?,?)"
            self.cur.execute(self.insertSql, (Date, Model_Name, Aug_str, Network, Epochs, Loss, Accuracy))
            self.conn.commit()

        # 알림창

    # Navigator
    def loadNavi(self):
        if self.sIMG == "":
            self.detail_widget2.hide()
        else:
            self.detail_widget2.show()
        # dummydata
        # self.wValue.setText("너비")
        # self.hValue.setText("높이")

        # self.fileName.setText("파일명")
        # self.fileSize.setText("파일사이즈")
        # self.extension.setText("확장자")
        # self.channel.setText("채널")

    def cnt_file(self):
        self.learn_num_data = []
        self.learn_num_data.append(len(self.class_names))
        cnt_train, cnt_val, cnt_test = 0, 0, 0
        sum_train, sum_val, sum_test = 0, 0, 0
        file_path = self.learnDataPath + '/train/'
        for folder in self.class_names:
            cnt_train = len([name for name in os.listdir(self.learnDataPath + '/train/' + folder) if os.path.isfile(os.path.join(self.learnDataPath + '/train/' + folder, name))])
            cnt_val = len([name for name in os.listdir(self.learnDataPath + '/validation/' + folder) if os.path.isfile(os.path.join(self.learnDataPath + '/validation/' + folder, name))])
            cnt_test = len([name for name in os.listdir(self.learnDataPath + '/test/' + folder) if os.path.isfile(os.path.join(self.learnDataPath + '/test/' + folder, name))])
            sum_train += cnt_train
            sum_val += cnt_val
            sum_test += cnt_test
            self.class_data.append([folder, cnt_train, cnt_val, cnt_test])
        print(self.class_data)
        self.sqlConnect()
        self.learn_num_data.append(cnt_train)
        self.learn_num_data.append(cnt_val)

    def openDirFn(self):
        os.startfile(resource_path(self.learnDataPath))

    def rmh5file(self):
        print(myWindow.select_test_model)
        rm_path = './checkpoint/' + myWindow.select_test_model
        try:
            print(rm_path)
            os.remove(rm_path)
        except:
            print('remove file not found')

if __name__ == "__main__":
    try:
        os.chdir(sys._MEIPASS)
        print(sys._MEIPASS)
    except:
        os.chdir(os.getcwd())
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)
    os.makedirs("./checkpoint", exist_ok=True)
    os.makedirs("./learnData", exist_ok=True)
    os.makedirs("./result_logs", exist_ok=True)
    # WindowClass의 인스턴스 생성
    myWindow = WindowClass()

    # 프로그램 화면을 보여주는 코드
    myWindow.show()

    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
