import sys
from PyQt5 import uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *


# 연결할 ui 파일의 경로 설정
UI_Path = './ClassEdit.ui'
form_class = uic.loadUiType(UI_Path)[0]

class ClassEdit(QDialog):
    def __init__(self):
        super().__init__()
        self.editUI()

    def editUI(self):
        self.setWindowTitle('Class Edit')
        # self.setGeometry(100, 100, 200, 100)

        layout = QVBoxLayout()
        layout.addStretch(1)

        edit = QLineEdit()
        font = edit.font()
        font.setPointSize(20)
        edit.setFont(font)
        self.edit = edit
        
        subLayout = QHBoxLayout()
        
        btnOK = QPushButton("확인")
        btnOK.clicked.connect(self.onOKButtonClicked)
        
        btnCancel = QPushButton("취소")
        btnCancel.clicked.connect(self.onCancelButtonClicked)
        
        layout.addWidget(edit)
        
        subLayout.addWidget(btnOK)
        subLayout.addWidget(btnCancel)
        layout.addLayout(subLayout)
        
        layout.addStretch(1)
        
        self.setLayout(layout)

    def onOKButtonClicked(self):
        self.accept()

    def onCancelButtonClicked(self):
        self.reject()

    def showModal(self):
        return super().exec_()

    
# if __name__ == "__main__" :
#     app = QApplication(sys.argv) 
#     myWindow = WindowClass()
#     myWindow.show()
#     app.exec_()