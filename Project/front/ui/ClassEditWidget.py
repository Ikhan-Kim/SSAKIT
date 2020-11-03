import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QStandardItemModel

# DB 연동
import sqlite3

# 연결할 ui 파일의 경로 설정
UI_Path = './ClassEdit.ui'
form_class = uic.loadUiType(UI_Path)[0]

class ClassEditWidget(QMainWindow, form_class) :
    def __init__(self, data) :
        super().__init__()
        self.setupUi(self)
        # print(data)

        self.setWindowTitle("Class Edit Widget")
        # Save 버튼 클릭
        ## => 바뀐 부분 save 되고 창 닫기
        self.saveBtn.clicked.connect(self.saveBtnFunc)

        # Cancel 버튼 클릭
        # => 바뀐 부분 저장 하지 않고, 창 닫기
        self.cancelBtn.clicked.connect(QCoreApplication.instance().quit)

        # class 불러오기 (메인창에 보인 클래스들 불러오기)
        data = data
        row = 0
        self.classType.setRowCount(len(data))

        colors = [
            [255, 0, 0],
            [255,192,203],
            [0, 0, 255],
            [0, 255, 0],
             ]
        # for cType in data:
        #     red = colors[row][0]
        #     green = colors[row][1]
        #     blue = colors[row][2]

        #     self.classType.setItem(row, 0, QtWidgets.QTableWidgetItem(cType["color"]))
        #     self.classType.item(row, 0).setBackground(QtGui.QColor(red, green, blue))
        #     self.classType.setItem(row, 1, QtWidgets.QTableWidgetItem(cType["label"]))
        #     row += 1

        # 수정하기


    def saveBtnFunc(self):
        pass

if __name__ == "__main__" :
    app = QApplication(sys.argv) 
    myWindow = ClassEditWidget() 
    myWindow.show()
    app.exec_()
