import sys
from PyQt5 import uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from ClassEditWidget import ClassEditWidget
from PyQt5.QtGui import QStandardItemModel

# DB 연동
import sqlite3

# 연결할 ui 파일의 경로 설정
UI_Path = './test.ui'
form_class = uic.loadUiType(UI_Path)[0]

class WindowClass(QMainWindow, form_class) :
    # 지금은 더미 데이터 => 이미지 불러올 때 데이터 불러와서 저장해야함.
    data = [
    {"color": "#FF5733", "label": "12R0", "train":50, "val":30, "test": 30},
    {"color": "#3372FF", "label": "4300", "train":50, "val":30, "test": 30},
    {"color": "#61FF33", "label": "4301", "train":50, "val":30, "test": 30},
    {"color": "#EA33FF", "label": "7501", "train":50, "val":30, "test": 30},
    ]

    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        # ClassEditWidget 불러오기
        self.openClassEditWidget = ClassEditWidget(WindowClass.data)

        #sql 연동
        self.sqlConnect()
        # self.insertData()
        # self.run()

        # class Edit btn 클릭 => 위젯 열기
        self.classEditBtn.clicked.connect(self.ClassEditBtnFunc)

        ###########################################

        # load data - 테이블 생성 및 데이터 불러오기
        self.setTable()

        # edit 금지 모드
        self.classType.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Navigator
        self.loadNavi()

    # DB)-1. SQL 연결 및 테이블 생성
    def sqlConnect(self):
        try: 
            self.conn = sqlite3.connect("dbName", isolation_level=None)
        except:
            print("문제가 있네요!")
            exit(1)
        print("연결성공!")
        self.cur = self.conn.cursor()

        self.createSql = "CREATE TABLE IF NOT EXISTS classLabel (idx INTEGER PRIMARY KEY, color TEXT, label TEXT, train INTEGER, val INTEGER, test INTEGER)"
        self.cmd = self.createSql
        self.run()

        # 초기 데이터 삽입
        # self.dataInputSql = ""
        # self.cmd = self.dataInputSql
        # self.run()

        for d in self.data:
            # print("d", d)
            self.color = d["color"]
            self.label = d["label"]
            self.train = d["train"]
            self.val = d["val"]
            self.test = d["test"]

            self.cmd = "insert into classLabel(`color`, `label`, `train`, `val`, `test`) values('{}', '{}', {}, {}, {})"\
                .format(self.color, self.label, self.train, self.val, self.test)
            # print("cmd : ", self.cmd)
            self.cur.execute(self.cmd)
            self.conn.commit()
        
        # 데이터베이스 내부 테이블의 내용을 모두 추출
        self.selectSql = "SELECT * FROM classLabel"
        self.cmd = self.selectSql
        self.run()

        rows = self.cur.fetchall()
        print(rows)

        # self.setTables(rows)
        cnt = len(rows)
        # print(rows)
        self.classType.setRowCount(cnt)

        for x in range(cnt):
            # 리스트 내부의 column쌍은 튜플로 반환하므로 튜플의 각 값을 변수에 저장
            color, label, train, test, val = rows[x]
            # print("⭐⭐⭐⭐⭐⭐", x)
            # print(rows[x])

            # 테이블의 각 셀에 값 입력
            # self.classType.setItem(x, 0, QTableWidgetItem(str(idx)))
            self.classType.setItem(x, 0, QTableWidgetItem(""))
            self.classType.item(x, 0).setBackground(QtGui.QColor(color))
            self.classType.setItem(x, 1, QTableWidgetItem(label))
            self.classType.setItem(x, 2, QTableWidgetItem(str(train)))
            self.classType.setItem(x, 3, QTableWidgetItem(str(val)))
            self.classType.setItem(x, 4, QTableWidgetItem(str(test)))

    def selectData():
        # # self.selectSql = "SELECT * FROM classLabel"
        # # self.cmd = self.selectSql
        # # self.run()

        # # rows = self.cur.fetchall()
        # # self.setTables(rows)

        # #데이터베이스 내부 테이블의 내용을 모두 추출
        # self.conn = sqlite3.connect("dbName")
        # self.cur = conn.cursor()
        
        # self.sql = "SELECT * FROM classLabel"
        # self.cur.execute(self.sql)
        # self.rows = cur.fetchall()        
        # print(self.rows)
        # self.conn.close()
        
        # #DB의 내용을 불러와서 TableWidget에 넣기 위한 함수 호출
        # self.setTables(self.rows)
        pass

    def insertData():
        pass
        # for d in self.data:
        #     self.color = d["color"]
        #     self.label = d["label"]
        #     self.train = d["train"]
        #     self.val = d["val"]
        #     self.test = d["test"]

        #     self.insertSql = "INSERT INTO classLabel (color, label, train, val, test) VALUES (?, ?, ?, ?, ?)"
        #     self.cmd = self.insertSql
            
        #     self.cur.execute(self.insertSql, (color, label, train, test, val))
        #     self.conn.commit()

        # self.selectData()
    
    def setTables(rows):
        # cnt = len(rows)
        # print(rows)
        # self.classType.setRowCount(cnt)

        # for x in range(cnt):
        #     # 리스트 내부의 column쌍은 튜플로 반환하므로 튜플의 각 값을 변수에 저장
        #     self.color, self.label, self.train, self.test, self.val = rows[x]
        #     print(rows)

        #     # 테이블의 각 셀에 값 입력
        #     # self.classType.setItem(x, 0, QTableWidgetItem(str(idx)))
        #     self.classType.setItem(x, 0, QTableWidgetItem(""))
        #     self.classType.item(x, 0).setBackground(QtGui.QColor(color))
        #     self.classType.setItem(x, 1, QTableWidgetItem(label))
        #     self.classType.setItem(x, 2, QTableWidgetItem(str(train)))
        #     self.classType.setItem(x, 3, QTableWidgetItem(str(val)))
        #     self.classType.setItem(x, 4, QTableWidgetItem(str(test)))

        pass

    # DB) sql문 실행 함수
    def run(self):
        self.cur.execute(self.cmd)
        self.conn.commit()
        print(self.cur.fetchall())

    # DB) 종료 함수
    def closeEvent(self, QCloseEvent):
        print("DB close!")
        self.conn.close()

    def ClassEditBtnFunc(self):
        # ClassEditWidget띄우기
        self.openClassEditWidget.show()

#########################################################        
        
        # color picker widget
        # color = QtGui.QColor(0, 0, 0)

        # fontColor = QtGui.QAction('Font bg Color', self)
        # fontColor.triggered.connect(self.color_picker)

        # self.toolBar.addAction(fontColor)

    # def color_picker(self):
    #     color = QtGui.QcolorDialog.getColor()
    #     self.styleChoice.setStyleSheet("Qwidget { background-color: %s}" % color.name() )
    
    
    def setTable(self):
        # Table column 수, header 설정+너비
        self.classType.setColumnCount(5)
        self.classType.setHorizontalHeaderLabels(['color', 'class', 'train', 'val', 'test'])
        # 너비 조절
        self.classType.setColumnWidth(0,60)
        self.classType.setColumnWidth(1,50)
        self.classType.setColumnWidth(2,10)
        self.classType.setColumnWidth(3,10)
        self.classType.setColumnWidth(4,10)
        self.classType.setColumnWidth(5,10)

        # self.classType.setRowCount(len(self.data))

        # row = 0
        # for cType in self.data:

        #     self.classType.setItem(row, 0, QtWidgets.QTableWidgetItem(" "))
        #     self.classType.item(row, 0).setBackground(QtGui.QColor(cType["color"]))
        #     self.classType.setItem(row, 1, QtWidgets.QTableWidgetItem(cType["label"]))
        #     self.classType.setItem(row, 2, QtWidgets.QTableWidgetItem(str(cType["train"])))
        #     self.classType.setItem(row, 3, QtWidgets.QTableWidgetItem(str(cType["val"])))
        #     self.classType.setItem(row, 4, QtWidgets.QTableWidgetItem(str(cType["test"])))
        #     row += 1

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

if __name__ == "__main__" :
    app = QApplication(sys.argv) 
    myWindow = WindowClass() 
    myWindow.show()
    app.exec_()
