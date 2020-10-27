import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtGui, QtCore
from ClassEdit import ClassEdit

# 연결할 ui 파일의 경로 설정
UI_Path = './test.ui'
form_class = uic.loadUiType(UI_Path)[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        # class 표 크기 조절
        # self.classType.setColumnWidth(0,10)
        self.classType.setColumnWidth(0,60)
        self.classType.setColumnWidth(1,50)
        self.classType.setColumnWidth(2,10)
        self.classType.setColumnWidth(3,10)
        self.classType.setColumnWidth(4,10)

        # load classes
        self.loadclass()

        # class Edit btn
        self.classEditBtu.clicked.connect(self.classEditBtuFunc)
        
        #########################################3
        # Navigator
        self.loadNavi()

        # 클릭했을경우 실행되는 함수
    #     self.btn_1.clicked.connect(self.btn1Function)
    #     self.btn_2.clicked.connect(self.btn2Function)

    # 업로드 버튼
        # self.FileUploadBtn.clicked.connect(self.FileUploadBtnFunc)
        # self.DUploadBtn.clicked.connect(self.DUploadBtnFunc)
        
    # def FileUploadBtnFunc(self):
    #     print('file upload btn')
    #     fileNames = QFileDialog.getOpenFileNames(self, self.tr("Open Data files"), "./", self.tr("Data Files (*.csv *.xls *.xlsx);; Images (*.png *.xpm *.jpg *.gif);; All Files(*.*)"))
    #     print(fileNames)

    # def DUploadBtnFunc(self):
    #     print('directory upload btn')
    #     dirName = QFileDialog.getExistingDirectory(self, self.tr("Open Data files"), "./", QFileDialog.ShowDirsOnly)
    #     print(dirName)

    def classEditBtuFunc(self):
        print('class Edit bnt')
        win = ClassEdit()
        r = win.showModal()

        if r:
            text = win.edit.text()
            self.labelTest.setText(text)
            
        
        # color picker widget
        # color = QtGui.QColor(0, 0, 0)

        # fontColor = QtGui.QAction('Font bg Color', self)
        # fontColor.triggered.connect(self.color_picker)

        # self.toolBar.addAction(fontColor)

    # def color_picker(self):
    #     color = QtGui.QcolorDialog.getColor()
    #     self.styleChoice.setStyleSheet("Qwidget { background-color: %s}" % color.name() )
    
    def loadclass(self):
        #dummy data
        classes = [
            {"color": "red", "label": "12R0", "train":50, "val":30, "test": 30},
            {"color": "pink", "label": "4300", "train":50, "val":30, "test": 30},
            {"color": "blue", "label": "4301", "train":50, "val":30, "test": 30},
             {"color": "green", "label": "7501", "train":50, "val":30, "test": 30},
        ]
        row = 0
        self.classType.setRowCount(len(classes))

        for cType in classes:
            self.classType.setItem(row, 0, QtWidgets.QTableWidgetItem(cType["color"]))
            self.classType.setItem(row, 1, QtWidgets.QTableWidgetItem(cType["label"]))
            self.classType.setItem(row, 2, QtWidgets.QTableWidgetItem(str(cType["train"])))
            self.classType.setItem(row, 3, QtWidgets.QTableWidgetItem(str(cType["val"])))
            self.classType.setItem(row, 4, QtWidgets.QTableWidgetItem(str(cType["test"])))
            row += 1

    def loadNavi(self):
        # dummydata
        self.wValue.setText("0w")
        self.hValue.setText("0h")
        self.xValue.setText("0x")
        self.yValue.setText("0y")

        self.fileName.setText("파일명")
        self.fileSize.setText("파일사이즈")
        self.extension.setText("확장자")
        self.channel.setText("채널")
        self.bit.setText("비트")

if __name__ == "__main__" :
    app = QApplication(sys.argv) 
    myWindow = WindowClass() 
    myWindow.show()
    app.exec_()
