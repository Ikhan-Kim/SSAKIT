import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

# 연결할 ui 파일의 경로 설정
UI_Path = './ui/NetworkSetting.ui'
form_class = uic.loadUiType(UI_Path)[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        # 클릭했을경우 실행되는 함수
    #     self.btn_1.clicked.connect(self.btn1Function)
    #     self.btn_2.clicked.connect(self.btn2Function)

    # def btn1Function(self):
    #     print('down bt1')

    # def btn2Function(self):
    #     print('down btn2')


if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv) 

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass() 

    #프로그램 화면을 보여주는 코드
    myWindow.show()

    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()