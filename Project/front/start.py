import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PIL import Image

import tkinter
from  tkinter import filedialog
import tkinter as tk
from tkinter import ttk

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
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.learnSettingDisplay = AnotherFormLayout()
        self.pushButton.clicked.connect(self.pushButton1)
        self.btnLearnSettings.clicked.connect(self.learnSettingsFn)
        
    # 더블클릭했을경우 실행되는 함수
    def OnDoubleClick(self, event):
        item = self.treeview.selection()[0]
        # print("you clicked on", self.treeview.item(item,"text"))
        # with open(self.fullpath + "\\" + self.treeview.item(item,"text"), "r") as f:
        #     for line in f:
        #         print(line)
        im = Image.open(self.fullpath + "\\" + self.treeview.item(item,"text"))
        print(im.format, im.size, im.mode)
        im.show()

    def pushButton1(self):
        def fill_tree(treeview, node):
            if treeview.set(node, "type") != 'directory':
                return

            path = treeview.set(node, "fullpath")
            # Delete the possibly 'dummy' node present.
            treeview.delete(*treeview.get_children(node))

            parent = treeview.parent(node)
            for p in os.listdir(path):
                self.fullpath = path
                p = os.path.join(path, p)
                # print(p)
                treeview.bind("<Double-1>", self.OnDoubleClick)
                ptype = None
                if os.path.isdir(p):
                    ptype = 'directory'

                fname = os.path.split(p)[1]
                oid = treeview.insert(node, 'end', text=fname, values=[p, ptype])
                if ptype == 'directory':
                    treeview.insert(oid, 0, text='dummy')

        def update_tree(event):
            treeview = event.widget
            fill_tree(treeview, treeview.focus())

        def create_root(treeview, startpath):
            dfpath = os.path.abspath(startpath)
            node = treeview.insert('', 'end', text=dfpath,
                    values=[dfpath, "directory"], open=True)
            fill_tree(treeview, node)
            
        # 폴더를 선택하고 폴더 주소를 저장
        root = tkinter.Tk()
        root.title("selected directory")
        root.geometry("400x800")
        root.filename =  filedialog.askdirectory()
        path = os.path.realpath(root.filename)

        self.treeview = ttk.Treeview(columns=("fullpath", "type"), displaycolumns='')
        self.treeview.pack(fill='both', expand=True)
        create_root(self.treeview, path)
        self.treeview.bind('<<TreeviewOpen>>', update_tree)

        root.mainloop()

    def learnSettingsFn(self, checked):
        if self.learnSettingDisplay.isVisible():
            self.learnSettingDisplay.hide()
        else:
            self.learnSettingDisplay.show()

if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv) 

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass() 

    #프로그램 화면을 보여주는 코드
    myWindow.show()

    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()