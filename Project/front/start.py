import sys, os, traceback, shutil
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtGui import *
import time

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from back import create_dir, set_directory
# from back.learning_test import InceptionV3_test1, ResNet152_test1, Vgg16_test1, test_function2, EfficientnetB4_test1
from mymodules import create_dir, set_directory
from mymodules import InceptionV3_test1, ResNet152_test1, Vgg16_test1, EfficientnetB4_test1, test_function2

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ClassEditWidget import ClassEditWidget


# DB 연동
import sqlite3

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
 

# 연결할 ui 파일의 경로 설정
form = resource_path('./ui/NetworkSetting.ui')
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

    def __init__(self, settingsData, learn_train_path, learn_val_path, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
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
                self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, myWindow)
        elif self.settingsData[0] == 'InceptionV3':
            print('Inception')
            InceptionV3_test1.Learn(
                self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, myWindow)
        elif self.settingsData[0] == 'ResNet152':
            print('ResNet')
            ResNet152_test1.Learn(
                self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, myWindow)
        elif self.settingsData[0] == 'EfficientnetB4':
            print('EfficientnetB4')
            EfficientnetB4_test1.Learn(
                self.settingsData[1], self.settingsData[2], self.learn_train_path, self.learn_val_path, myWindow)
        myWindow.btnEnable()

# preprocess setting popup #train wizard
class AnotherFormLayout(QDialog):
    NumGridRows = 3
    NumButtons = 4
    colorSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setGeometry(800,100,600,600)
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
        self.trainList = QTableWidget()
        mainLayout.addWidget(self.trainList)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

        self.setWindowTitle("Train Settings")

        self.setTLTables()

    def createFormGroupBox(self):
        # Augmentation
        self.formAugmentation = QGroupBox("Augmentation")
        layout = QFormLayout()
        self.checkBoxHorizantal = QCheckBox("[H] Horizantal Flip", self)
        layout.addRow(self.checkBoxHorizantal)
        self.checkBoxVertical = QCheckBox("[V] Vertical Flip", self)
        layout.addRow(self.checkBoxVertical)
        self.checkBoxRotation90 = QCheckBox("[R-90] Rotation 90", self)
        layout.addRow(self.checkBoxRotation90)
        self.checkBoxRotation180 = QCheckBox("[R-180] Rotation 180", self)
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
        self.comboBoxNN.addItems(["VGG", "InceptionV3", "ResNet152", "EfficientnetB4"])
        layoutNN.addRow(QLabel("select NN:"), self.comboBoxNN)
        self.formNueralNetwork.setLayout(layoutNN)
        # Learn Settings
        self.formLearn = QGroupBox("Learn Settings")
        layoutLS = QFormLayout()
        self.lineEpochs = QSpinBox()
        self.lineEpochs.setRange(1, 10000)
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
        self.colorSignal.emit()
        print(WindowClass.settingsData)
        self.hide()

    # DB 연결, 테이블 생성
    def trainListSqlConnect(self):
        try: 
            self.conn = sqlite3.connect("test2.db", isolation_level=None)
        except:
            print("문제가 있네요!")
            exit(1)
        print("연결성공!")
        self.cur = self.conn.cursor()

        # 테이블 생성
        self.createSql = "CREATE TABLE IF NOT EXISTS trainList (idx INTEGER PRIMARY KEY, Date TEXT,  Model selection TEXT, Augmentation TEXT, Epochs INTEGER, Loss INTEGER, Accuracy INTEGER)"
        self.cmd = self.createSql
        self.cur.execute(self.cmd)
        self.conn.commit()
    
    def setTLTables(self):
        # Table column 수, header 설정+너비
        self.trainList.setColumnCount(6)
        self.trainList.setHorizontalHeaderLabels(['Date', 'Model selection', 'Augmentation',  'Epochs' , 'Loss' , 'Accuracy'])
        # accuracy, 
        # self.classTypeWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # # Table 너비 조절
        self.trainList.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

class ProjectNameClass(QDialog):
    nameSignal = pyqtSignal()
    class_names = []
    project_list = []
    def __init__(self):
        super().__init__()
        self.project_list = os.listdir('./learnData/')
        self.setWindowTitle('Open Project')
        self.formLoadProject = QGroupBox("프로젝트 불러오기")
        self.formNewProject = QGroupBox("프로젝트 생성하기")
        loadlayout = QFormLayout()
        newlayout = QFormLayout()
        self.loadTable = QTableWidget()
        self.createTable()
        self.loadTable.cellClicked.connect(self.cellClick)
        self.lineName = QLineEdit()
        self.btnOk = QPushButton('OK')
        self.btnOk.clicked.connect(self.projectNameFn)
        loadlayout.addRow(self.loadTable)
        newlayout.addRow(self.lineName)
        newlayout.addRow(self.btnOk)
        mainLayout = QVBoxLayout()
        self.formLoadProject.setLayout(loadlayout)
        self.formNewProject.setLayout(newlayout)
        mainLayout.addWidget(self.formLoadProject)
        mainLayout.addWidget(self.formNewProject)
        self.setLayout(mainLayout)

    def createTable(self):
        self.loadTable.setColumnCount(3)
        self.loadTable.setHorizontalHeaderLabels(['Name', '선택', '삭제'])
        self.loadTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.loadTable.setColumnWidth(1,50)
        self.loadTable.setColumnWidth(2,50)
        self.loadTable.setRowCount(len(self.project_list))
        for x in range(len(self.project_list)):
            self.loadTable.setItem(x, 0, QTableWidgetItem(self.project_list[x]))
            self.loadTable.setItem(x, 1, QTableWidgetItem("선택"))
            self.loadTable.setItem(x, 2, QTableWidgetItem("❌"))

    def cellClick(self, row, column):
        if column == 1:
            # 선택
            print('선택클릭')
            WindowClass.projectName = self.project_list[row]
            self.nameSignal.emit()
            self.hide()

        elif column == 2:
            # 삭제
            a = QMessageBox.question(self, "삭제 확인", "정말로 삭제 하시겠습니까?",
                                 QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
            if a == QMessageBox.Yes:
                del_path = './learnData/' + self.project_list[row]
                shutil.rmtree(del_path)
                self.project_list = os.listdir('./learnData/')
                self.hide()

    def projectNameFn(self):
        WindowClass.projectName = self.lineName.text()
        self.nameSignal.emit()
        self.hide()

class TestModelSelect(QDialog):
    def __init__(self):
        super().__init__()
        self.label = QLabel()
        # if len(os.listdir("../back/learning_test/checkpoint")) == 0:
        #     self.label = QLabel("학습된 모델이 없습니다.", self)
        # else:
        self.label = QLabel("모델을 선택해 주세요", self)
        self.listW = QListWidget()
        for i in range(len(os.listdir("../back/learning_test/checkpoint"))):
            self.listW.addItem(os.listdir("../back/learning_test/checkpoint")[i])
        self.listW.itemActivated.connect(self.itemActivated_event)
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.listW)
        self.setLayout(vbox)
        self.setGeometry(300, 300, 300, 300)

    def itemActivated_event(self, item):
        self.hide()
        test_function2.test(item.text())

# MainWindow
class WindowClass(QMainWindow, form_class):
    mainImg = "C:/Users/multicampus/Desktop/s03p31c203/Project/front/test_img/test1.png"
    settingsData = []
    projectName = ''
    learn_train_path = ''
    # learn_val_path = ''
    send_valve_popup_signal = pyqtSignal(bool, name='sendValvePopupSignal')

    # DB에 넣을 데이터 불러오기 => 불러온 이미지의 label 기반
    data = [
    {"color": "#FF5733", "label": "12R0", "train":50, "val":30, "test": 30},
    {"color": "#3372FF", "label": "4300", "train":50, "val":30, "test": 30},
    {"color": "#61FF33", "label": "4301", "train":50, "val":30, "test": 30},
    {"color": "#EA33FF", "label": "7501", "train":50, "val":30, "test": 30},
    ]

    def __init__(self) :     
        super().__init__()
        # design
        # changing the background color to yellow 
        self.setStyleSheet("background-color: #847f7f;")

        self.setupUi(self)
        # 기본 설정?>
        self.learnSettingDisplay = AnotherFormLayout()
        self.projectNameDisplay = ProjectNameClass()
        self.testModelSelectDisplay = TestModelSelect()
        pixmap = QtGui.QPixmap(self.mainImg)
        self.imgLabel.setPixmap(pixmap)
        # 버튼별 함수 실행
        self.btnCreateProject.clicked.connect(self.createProjectFn)
        # self.btnDataLoad.setStyleSheet("background-image: url(front\assets\img\imageUpload.png);")
        self.btnDataLoad.clicked.connect(self.dataLoadFn)
        self.btnLearnSettings.clicked.connect(self.learnSettingsFn)
        self.dirTreeView.doubleClicked.connect(self.fileViewFn)
        self.btnTraining.clicked.connect(self.training)
        self.projectNameDisplay.nameSignal.connect(self.createNameFn)
        self.learnSettingDisplay.colorSignal.connect(self.changeColorFn)
        self.btnTest.clicked.connect(self.test)
        # 터미널
        # self.textBox_terminal.setGeometry(QtCore.QRect(0, 0, 1200, 190))
        # live loss plot
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.plotLayout.addWidget(self.canvas)

        self.setWindowTitle('SSAKIT')

        # sql 연동
        self.sqlConnect()

        # ClassEditWidget 불러오기
        self.openClassEditWidget = ClassEditWidget(WindowClass.data)
        # class Edit btn 클릭 => 위젯 열기
        self.classEditBtn.clicked.connect(self.ClassEditBtnFunc)
        # edit 금지 모드
        self.classType.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 새로고침 버튼 ()
        self.f5Btn.clicked.connect(self.f5BtnFunc)

        # Navigator
        self.loadNavi()

    def btnColorChange(self, btn):
        # print(" btn change", btn)
        btns = [self.btnLearnSettings,  self.btnTraining, self.btnTest, self.pushButton_5]
        btns.remove(btn)
        btn.setStyleSheet("background-color: rgb(241, 127, 66); font: 12pt 'a로케트'; color: rgb(255, 255, 255);")
        for b in btns:
            b.setStyleSheet("background-color: #ffeee4; font: 12pt 'a로케트'; color: rgb(0, 0, 0);")

    def createNameFn(self):
        self.setWindowTitle('SSAKIT -' + self.projectName)
        self.testPath = './learnData/' + self.projectName
        create_dir.create_dir_flow(self.projectName)
        treeModel = QFileSystemModel()
        self.dirTreeView.setModel(treeModel)
        treeModel.setRootPath(QDir.rootPath())
        self.dirTreeView.setRootIndex(treeModel.index(self.testPath))
        self.pjtTitle.setText(self.projectName)
        self.mainWidget.hide()
        
    def changeColorFn(self):
        self.btnColorChange(self.btnTraining)

    def createProjectFn(self):
        if self.projectNameDisplay.isVisible():
            self.projectNameDisplay.hide()
        else:
            self.projectNameDisplay.show()

    def dataLoadFn(self):
        if self.projectName:
            self.pathName = QFileDialog.getExistingDirectory(self, self.tr("Open Data files"), "./",
                                                            QFileDialog.ShowDirsOnly)
            self.class_names = os.listdir(self.pathName)
            print(self.pathName)
            
            # if self.pathName:
            #     self.dirName = self.pathName.split('/')[-1]
            #     set_directory.set_directory(
            #         self.projectName, self.dirName, self.pathName)
            if self.pathName:
                for dirName in self.class_names:
                    set_directory.set_directory(
                        self.projectName, dirName, self.pathName + '/' + dirName
                    )
                # self.setWindowTitle(self.projectName)
        else:
            self.warningMSG("주의", "프로젝트를 먼저 생성/선택 해주십시오.")

    def learnSettingsFn(self, checked):
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
        self.dirTreeView.hideColumn(1)
        self.dirTreeView.hideColumn(2)
        self.dirTreeView.hideColumn(3)
        pixmap = QtGui.QPixmap(self.mainImg)
        pixmap2 = pixmap.scaledToWidth(430)
        self.imgLabel.setPixmap(pixmap2)

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

    def btnEnable(self):
        self.btnCreateProject.setEnabled(True)
        self.btnDataLoad.setEnabled(True)
        self.btnLearnSettings.setEnabled(True)
        self.btnTraining.setEnabled(True)
        self.btnTest.setEnabled(True)

    def training(self):
        self.btnColorChange(self.btnTraining)
        if self.learn_train_path:
            self.btnDisable()
            # Pass the function to execute
            worker = Worker(self.settingsData, self.learn_train_path, self.learn_val_path) # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self.print_output)
            worker.signals.finished.connect(self.thread_complete)
            worker.signals.progress.connect(self.progress_fn)

            # Execute
            self.threadpool.start(worker)
        else:
            self.warningMSG("주의", "데이터 로드 및 이미지 전처리를 먼저 실행해 주십시오.")
        # self.btnEnable()

    def test(self):
        self.testModelSelectDisplay.show()
        # test_function2.test()
        self.btnColorChange(self.btnTest)
        if self.isTrained:
            test_function2.test()
        else:
            self.warningMSG("주의", "모델 학습을 먼저 실행해 주십시오.")

    # ClassEditWidget띄우기
    def ClassEditBtnFunc(self):
        self.openClassEditWidget.show()

    def f5BtnFunc(self):
        print("새로고침")
        self.selectData()

    # DB) SQL 연결 및 테이블 생성
    def sqlConnect(self):
        try: 
            self.conn = sqlite3.connect("test2.db", isolation_level=None)
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
        # print(item_list, len(item_list))
        if len(item_list) == 0:
            for d in self.data:
                # print("d", d)
                self.color = d["color"]
                self.label = d["label"]
                self.train = d["train"]
                self.val = d["val"]
                self.test = d["test"]

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
        self.classType.setColumnCount(4)
        self.classType.setHorizontalHeaderLabels(['color', 'class', 'train', 'val', 'test'])
        
        # Table 너비 조절
        # self.classType.setColumnWidth(0,10)
        # self.classType.setColumnWidth(0,50)
        # self.classType.setColumnWidth(1,10)
        # self.classType.setColumnWidth(3,10)
        # self.classType.setColumnWidth(4,10)
        # self.classType.setColumnWidth(5,10)
        # self.classType.setColumnWidth(6,10)
        self.classType.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        cnt = len(rows)
        self.classType.setRowCount(cnt)

        for x in range(cnt):
            # 리스트 내부의 column쌍은 튜플로 반환하므로 튜플의 각 값을 변수에 저장
            # print(rows)
            color, label, train, val, test = rows[x][1:]
            
            # print("rows[x]", rows[x][0], rows[x][1], rows[x][2])
            # 테이블의 각 셀에 값 입력
            # self.classType.setItem(x, 0, QTableWidgetItem(""))
            # self.classType.item(x, 0).setBackground(QtGui.QColor(color))
            # self.classType.setItem(x, 1, QTableWidgetItem(label))
            # self.classType.setItem(x, 2, QTableWidgetItem(str(train)))
            # self.classType.setItem(x, 3, QTableWidgetItem(str(val)))
            # self.classType.setItem(x, 4, QTableWidgetItem(str(test)))

            # self.classType.setItem(x, 0, QTableWidgetItem(str(idx)))
            self.classType.setItem(x, 0, QTableWidgetItem(""))
            self.classType.item(x, 0).setBackground(QtGui.QColor(color))
            self.classType.setItem(x, 1, QTableWidgetItem(label))
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

    def ClassEditBtnFunc(self):
        # ClassEditWidget띄우기
        self.openClassEditWidget.show()

    # Navigator
    def loadNavi(self):
        # dummydata
        self.wValue.setText("너비")
        self.hValue.setText("높이")
        self.xValue.setText("0")
        self.yValue.setText("0")

        self.fileName.setText("파일명")
        self.fileSize.setText("파일사이즈")
        self.extension.setText("확장자")
        self.channel.setText("채널")
        self.bit.setText("비트")

if __name__ == "__main__":
    try:
        os.chdir(sys._MEIPASS)
        print(sys._MEIPASS)
    except:
        os.chdir(os.getcwd())
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    # WindowClass의 인스턴스 생성
    myWindow = WindowClass()

    # 프로그램 화면을 보여주는 코드
    myWindow.show()

    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
