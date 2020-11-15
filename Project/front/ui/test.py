import sys
from PyQt5 import uic

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from ClassEditWidget import ClassEditWidget
# from PyQt5.QtGui import QStandardItemModel

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
        # self.setTable()
        self.sqlConnect()
        # self.insertData()
        # self.run()

        # class Edit btn 클릭 => 위젯 열기
        self.classEditBtn.clicked.connect(self.ClassEditBtnFunc)

        ###########################################

        # load data - 테이블 생성 및 데이터 불러오기

        # edit 금지 모드
        self.classType.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Navigator
        self.loadNavi()

    # DB) SQL 연결 및 테이블 생성
    def sqlConnect(self):
        try: 
            self.conn = sqlite3.connect(dbName, isolation_level=None)
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
        # self.dataInputSql = ""
        # self.cmd = self.dataInputSql
        # self.run()

        for d in self.data:
            print("d", d)
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

        self.selectData()

    def selectData(self):
        self.selectSql = "SELECT * FROM classLabel"
        self.cmd = self.selectSql
        self.run()

        # rows = self.cur.fetchall()
        rowss = [list(item[1:]) for item in self.cur.fetchall()]
        # print("rowss",rowss)
        self.setTables(rowss)

    def insertData():
        pass
        # data = [
        #     {"color": "#FF5733", "label": "12R0", "train":50, "val":30, "test": 30},
        #     {"color": "#3372FF", "label": "4300", "train":50, "val":30, "test": 30},
        #     {"color": "#61FF33", "label": "4301", "train":50, "val":30, "test": 30},
        #     {"color": "#EA33FF", "label": "7501", "train":50, "val":30, "test": 30},
        #     ]

        # for d in data:
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
    
    def setTables(self, rows):
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
        cnt = len(rows)
        # cnt = 5
        self.classType.setRowCount(cnt)
        # rows_list = 

        for x in range(cnt):
            # 리스트 내부의 column쌍은 튜플로 반환하므로 튜플의 각 값을 변수에 저장
            print(rows)
            color, label, train, val, test = rows[x]
            print(color, type(color))

            
            print("rows[x]", rows[x][0], rows[x][1], rows[x][2])
            # 테이블의 각 셀에 값 입력
            # self.classType.setItem(x, 0, QTableWidgetItem(str(idx)))
            self.classType.setItem(x, 0, QTableWidgetItem(""))
            self.classType.item(x, 0).setBackground(QtGui.QColor(color))
            self.classType.setItem(x, 1, QTableWidgetItem(label))
            self.classType.setItem(x, 2, QTableWidgetItem(str(train)))
            self.classType.setItem(x, 3, QTableWidgetItem(str(val)))
            self.classType.setItem(x, 4, QTableWidgetItem(str(test)))

            # self.classType.setItem(x, 0, QTableWidgetItem(rows[x][0]))
            # # self.classType.item(x, 0).setBackground(QtGui.QColor(color))
            # self.classType.setItem(x, 1, QTableWidgetItem(rows[x][1]))
            # self.classType.setItem(x, 2, QTableWidgetItem(str(rows[x][2])))
            # self.classType.setItem(x, 3, QTableWidgetItem(str(rows[x][3])))
            # self.classType.setItem(x, 4, QTableWidgetItem(str(rows[x][4])))

            # self.classType.setItem(x, 0, QtWidgets.QTableWidgetItem("?"))
            # # self.classType.item(x, 0).setBackground(QtGui.QColor(red, green, blue))
            # self.classType.setItem(x, 1, QtWidgets.QTableWidgetItem("label"))
            # self.classType.setItem(x, 2, QtWidgets.QTableWidgetItem("train"))
            # self.classType.setItem(x, 3, QtWidgets.QTableWidgetItem("val"))
            # self.classType.setItem(x, 4, QtWidgets.QTableWidgetItem("test"))

            # self.classType.setItem(x, 0, QTableWidgetItem("?"))
            # # self.classType.item(x, 0).setBackground(QtGui.QColor(red, green, blue))
            # self.classType.setItem(x, 1, QTableWidgetItem("label"))
            # self.classType.setItem(x, 2, QTableWidgetItem("train"))
            # self.classType.setItem(x, 3, QTableWidgetItem("val"))
            # self.classType.setItem(x, 4, QTableWidgetItem("test"))

    # DB) sql문 실행 함수
    def run(self):
        self.cur.execute(self.cmd)
        self.conn.commit()
        # print(self.cur.fetchall())

    # DB) 종료 함수
    def closeEvent(self, QCloseEvent):
        print("DB close!")
        self.conn.close()

    def ClassEditBtnFunc(self):
        # ClassEditWidget띄우기
        self.openClassEditWidget.show()

        ########### init UI 테스트 ########

    # def initUI(self):
    #     self.LIST.setRootIsDecorated(False)
    #     self.LIST.setAlternatingRowColors(True)

    #     self.내용 = QStandardItemModel(0,3, self)
    #     self.내용.setHeaderData(0, Qt.Horizontal, "번호")
    #     self.내용.setHeaderData(1, Qt.Horizontal, "이름")
    #     self.내용.setHeaderData(2, Qt.Horizontal, "주소")
    #     # self.LIST.clicked.connect(self.slt)  # slt 클릭 이벤트 연결

    #     self.LIST.setModel(self.내용)
    #     self.LIST.setColumnWidth(0, 40)
    #     self.LIST.setColumnWidth(1, 80)

    # def showEvent(self, QShowEvent):
    #     pass

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
        pass
        # # Table column 수, header 설정+너비
        # self.classType.setColumnCount(5)
        # self.classType.setHorizontalHeaderLabels(['color', 'class', 'train', 'val', 'test'])
        # # 너비 조절
        # self.classType.setColumnWidth(0,60)
        # self.classType.setColumnWidth(1,50)
        # self.classType.setColumnWidth(2,10)
        # self.classType.setColumnWidth(3,10)
        # self.classType.setColumnWidth(4,10)
        # self.classType.setColumnWidth(5,10)

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

# ####### 외부 함수 ##########
# def CreateTable():
#     # sqlite3 db 파일 접속, 없으면 생성
#     conn = sq.connect(Daatabasename)
#     cur = conn.cursor()

#     # db에 classLabel 테이블이 있는지, sqlite3의 마스터 테이블에서 정보를 받아온다.
#     sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='classLabel'"
#     cur.execute(sql)
#     rows = cur.fetchall()

#     # classLabel 테이블이 없으면 새로 생성하고 있으면 통과
#     if not rows:
#         sql = "CREATE TABLE classLabel (idx INTEGER PRIMARY KEY, color TEXT, label TEXT, train INTEGER, test INTEGER, val INTEGER)"
#         cur.execute(sql)
#         conn.commit()
    
#     conn.close()




# def SelectData():
#     # db 내부 테이블의 내용을 모두 추출
#     conn = sq.connect(Daatabasename)
#     cur = conn.cursor()

#     sql = "SELECT * FROM classLabel"
#     cur.execute(sql)
#     rows = cur.fetchall()

#     conn.close()

#     # db 내용을 불러와서 TableWidget에 넣기 위한 함수 호출
#     setTables(rows)

#      # for cType in self.data:
#         #     red = colors[row][0]
#         #     green = colors[row][1]
#         #     blue = colors[row][2]

#         #     self.classType.setItem(row, 0, QtWidgets.QTableWidgetItem(cType["color"]))
#         #     self.classType.item(row, 0).setBackground(QtGui.QColor(red, green, blue))
#         #     self.classType.setItem(row, 1, QtWidgets.QTableWidgetItem(cType["label"]))
#         #     self.classType.setItem(row, 2, QtWidgets.QTableWidgetItem(str(cType["train"])))
#         #     self.classType.setItem(row, 3, QtWidgets.QTableWidgetItem(str(cType["val"])))
#         #     self.classType.setItem(row, 4, QtWidgets.QTableWidgetItem(str(cType["test"])))
#         #     row += 1

# def setTables(row):
#     # db 내부에 저장된 결과물의 개수를 저장한다.
#     count = len(row)

#     # 개수만큼 테이블의 row를 생성한다.
#     UI_set.classType.setRowCount(count)

#     # row 리스트 만큼 반복하여 Table에 DB 값을 넣는다.
#     for x in range(count):
#         # 리스트 내부의 column쌍은 튜플로 반환하므로 튜플의 각 값을 변수에 저장
#         idx, color, label, train, test, val = row[x]

#         # 테이블의 각 셀에 값 입력
#         UI_set.classType.setItem(x, 0, QTableWidgetItem(str(idx)))
#         UI_set.classType.setItem(x, 1, QTableWidgetItem(color))
#         UI_set.classType.setItem(x, 2, QTableWidgetItem(label))
#         UI_set.classType.setItem(x, 3, QTableWidgetItem(str(train)))
#         UI_set.classType.setItem(x, 4, QTableWidgetItem(str(test)))
#         UI_set.classType.setItem(x, 4, QTableWidgetItem(str(val)))

#         


if __name__ == "__main__" :
    app = QApplication(sys.argv) 
    myWindow = WindowClass() 
    myWindow.show()
    app.exec_()
