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
UI_Path = './ui/ClassEdit.ui'
form_class = uic.loadUiType(UI_Path)[0]

class ClassEditWidget(QMainWindow, form_class) :
    def __init__(self, data) :
        super().__init__()
        self.setupUi(self)

        self.setWindowTitle("Class Edit Widget")

        # sql 연동
        self.sqlConnect()

        # edit 금지 모드
        self.classTypeWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        #버튼 클릭시 데이터 입력
        self.pushButton.clicked.connect(self.insertData)

        #Table의 내부 셀을 클릭할 때 연결할 함수
        #셀을 클릭하여 연결한 함수에는 기본적으로 셀의 Row, Column 두개의 인자를 넘겨준다.
        self.classTypeWidget.cellClicked.connect(self.deleteData)

        # Save 버튼 클릭
        ## => 바뀐 부분 save 되고 창 닫기
        # self.saveBtn.clicked.connect(self.saveBtnFunc)

        # Cancel 버튼 클릭
        # => 바뀐 부분 저장 하지 않고, 창 닫기
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        # self.buttonBox.QDialogButtonBox.Ok.accepted()
        # QDialogButtonBox.accepted()
        # self.buttonBox.accepted.connect(self.accept)
        # self.buttonBox.accepted.connect(self.accept)
        # self.buttonBox.rejected.connect(self.reject)

        # self.retranslateUi(Dialog)
        # self.buttonBox.accepted.connect(accept)
        # self.buttonBox.rejected.connect(reject)
        # QtCore.QMetaObject.connectSlotsByName(Dialog)
        
        # self.cancelBtn.clicked.connect(QCoreApplication.instance().quit)
        # print(QCoreApplication.instance())

        # class 불러오기 (메인창에 보인 클래스들 불러오기)
        # data = data
        # row = 0
        # self.classType.setRowCount(len(data))

        # self.btnOK = QPushButton("Save")
        # self.btnOK.clicked.connect(self.onOKButtonClicked)
        
        # self.btnCancel = QPushButton("Cancel")
    #     self.btnCancel.clicked.connect(self.onCancelButtonClicked)

    # def onOKButtonClicked(self):
    #     print("okbtn")
    #     self.accept()

    # def onCancelButtonClicked(self):
    #     print("cancel")
    #     self.reject()

    # 수정하기
##########################################################

########################################################

    # DB) SQL 연결 및 테이블 생성
    def sqlConnect(self):
        try: 
            self.conn = sqlite3.connect("test2.db", isolation_level=None)
        except:
            print("문제가 있네요!")
            exit(1)
        print("연결성공!")
        self.cur = self.conn.cursor()

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
        self.classTypeWidget.setColumnCount(4)
        self.classTypeWidget.setHorizontalHeaderLabels(['idx', 'color', 'label', '삭제'])
        # self.classTypeWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # # Table 너비 조절
        self.classTypeWidget.setColumnWidth(0,10)
        self.classTypeWidget.setColumnWidth(3,50)
        self.classTypeWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.classTypeWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        
        cnt = len(rows)
        self.classTypeWidget.setRowCount(cnt)

        for x in range(cnt):
            # 리스트 내부의 column쌍은 튜플로 반환하므로 튜플의 각 값을 변수에 저장
            # print(rows)
            idx, color, label = rows[x][0], rows[x][1], rows[x][2]
            print("idx, color, label : ", idx, color, label)
            
            # print("rows[x]", rows[x][0], rows[x][1], rows[x][2])
            # 테이블의 각 셀에 값 입력
            self.classTypeWidget.setItem(x, 0, QTableWidgetItem(str(idx)))
            self.classTypeWidget.setItem(x, 1, QTableWidgetItem(""))
            self.classTypeWidget.item(x, 1).setBackground(QtGui.QColor(color))
            self.classTypeWidget.setItem(x, 2, QTableWidgetItem(label))
            self.classTypeWidget.setItem(x, 3, QTableWidgetItem("❌"))

    # 데이터 삽입하기 (Insert)
    # def InsertData(self):
    #     self.insertSql = "INSERT INTO classLabel (color, label) VALUES (?, ?)"

    #     cur.execute(sql, (color, label))
    #     conn.commit()
    #     conn.close()

    #     self.selectData()

    # DB) sql문 실행 함수
    def run(self):
        self.cur.execute(self.cmd)
        self.conn.commit()

    # DB) 종료 함수
    def closeEvent(self, QCloseEvent):
        print("DB close!")
        self.conn.close()

    # 저장하기
    def saveBtnFunc(self):
        self.closeEvent()

    # insert
    def insertData(self):
        #두개의 lineEdit에서 각각 색과 className를 받아온다.  
        color = self.lineEdit.text()
        label = self.lineEdit_2.text()
        
        conn = sqlite3.connect("test2.db")
        cur = conn.cursor()
        
        insertSql = "INSERT INTO classLabel (color, label) VALUES (?,?)"
        cur.execute(insertSql, (color, label))
        conn.commit()

        #데이터 입력 후 DB의 내용 불러와서 TableWidget에 넣기 위한 함수 호출
        self.selectData()

    # delete
    def deleteData(self, row, column):
        # 테이블 내부의 셀 클릭과 연결된 이벤트는 기본적으로 셀의 Row, Column을 인자로써 전달받는다.
        print("r,c", row, column)
        # 삭제 셀이 눌렸을 때, 삭제 셀은 4번째 셀이므로 column 값이 3일 경우만 작동한다.
        if column == 3: 
            conn = sqlite3.connect("test2.db")
            cur = conn.cursor()
            
            #DB의 데이터 idx는 선택한 Row의 첫번째 셀(0번 Column)의 값에 해당한다.
            idx = self.classTypeWidget.item(row, 0).text()
            print("idx", idx)
            sql = "DELETE FROM classLabel WHERE idx =?"
            cur.execute(sql, (idx,))
            conn.commit()
            
            # conn.close()
            
            #데이터 삭제 후 DB의 내용 불러와서 TableWidget에 넣기 위한 함수 호출
            self.selectData()

if __name__ == "__main__" :
    app = QApplication(sys.argv) 
    myWindow = ClassEditWidget(form_class) 
    myWindow.show()
    app.exec_()
