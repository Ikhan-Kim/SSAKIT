import sys
from PyQt5 import uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from ClassEditWidget import ClassEditWidget
from PyQt5.QtGui import QStandardItemModel

# DB 연동
import sqlite3 as sq

# DB 이름 설정
Daatabasename = "classtable.db"

# 연결할 ui 파일의 경로 설정
UI_Path = './test.ui'
form_class = uic.loadUiType(UI_Path)[0]

class WindowClass(QMainWindow, form_class) :

    data = [
    {"color": "", "label": "12R0", "train":50, "val":30, "test": 30},
    {"color": "", "label": "4300", "train":50, "val":30, "test": 30},
    {"color": "", "label": "4301", "train":50, "val":30, "test": 30},
    {"color": "", "label": "7501", "train":50, "val":30, "test": 30},
    ]

    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.setupUI()

        # ClassEditWidget 불러오기
        self.openClassEditWidget = ClassEditWidget(WindowClass.data)

        # load data
        self.loadclass()

        # edit 금지 모드
        self.classType.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        # Navigator
        self.loadNavi()
    
    def setupUI(self):
        global UI_set 

        # UI 파일 로딩
        UI_set = form_class

        # 테이블 세팅 (내부 메서드 호출)
        self.setTable()

        # DB 세팅을 위해 외부 함수 호출
        CreateTable()

        # DB 세팅 후, DB 값 불러오는 외부 함수 호출
        SelectData()

        InsertData()

        # class Edit btn 클릭 => 위젯 열기
        self.classEditBtn.clicked.connect(self.ClassEditBtnFunc)

    def setTable(self):
        # Table column 갯수
        # self.classType.setColumnCount(6)
        UI_set.tableWidget.setColumnCount(6)

        # Table 칼럼 헤더 라벨
        self.classType.setHorizontalHeaderLabels(['No','color', 'class', 'train', 'test', 'val'])


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
    
    
    def loadclass(self):
        # data = data
        row = 0
        self.classType.setRowCount(len(self.data))

        colors = [
            [255, 0, 0],
            [255,192,203],
            [0, 0, 255],
            [0, 255, 0],
             ]

        # for cType in self.data:
        #     red = colors[row][0]
        #     green = colors[row][1]
        #     blue = colors[row][2]

        #     self.classType.setItem(row, 0, QtWidgets.QTableWidgetItem(cType["color"]))
        #     self.classType.item(row, 0).setBackground(QtGui.QColor(red, green, blue))
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

####### 외부 함수 ##########
def CreateTable():
    # sqlite3 db 파일 접속, 없으면 생성
    conn = sq.connect(Daatabasename)
    cur = conn.cursor()

    # db에 classLabel 테이블이 있는지, sqlite3의 마스터 테이블에서 정보를 받아온다.
    sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='classLabel'"
    cur.execute(sql)
    rows = cur.fetchall()

    # classLabel 테이블이 없으면 새로 생성하고 있으면 통과
    if not rows:
        sql = "CREATE TABLE classLabel (idx INTEGER PRIMARY KEY, color TEXT, label TEXT, train INTEGER, test INTEGER, val INTEGER)"
        cur.execute(sql)
        conn.commit()
    
    conn.close()

def InsertData():
    # dummy data
    data = [
    {"color": "#FF5733", "label": "12R0", "train":50, "val":30, "test": 30},
    {"color": "#3372FF", "label": "4300", "train":50, "val":30, "test": 30},
    {"color": "#61FF33", "label": "4301", "train":50, "val":30, "test": 30},
    {"color": "#EA33FF", "label": "7501", "train":50, "val":30, "test": 30},
    ]

    color = "#FF5733"
    label = "12R0"
    train = 50
    test = 30
    val = 20

    conn = sq.connect(Daatabasename)
    cur.conn.cursor()

    sql = "INSERT INTO classLabel (color, label, train, test, val) VALUES (?, ?, ?, ?, ?)"

    cur.execute(sql, (color, label, train, test, val))
    conn.commit()

    conn.close()

    SelectData()


def SelectData():
    # db 내부 테이블의 내용을 모두 추출
    conn = sq.connect(Daatabasename)
    cur = conn.cursor()

    sql = "SELECT * FROM classLabel"
    cur.execute(sql)
    rows = cur.fetchall()

    conn.close()

    # db 내용을 불러와서 TableWidget에 넣기 위한 함수 호출
    setTables(rows)



    

     # for cType in self.data:
        #     red = colors[row][0]
        #     green = colors[row][1]
        #     blue = colors[row][2]

        #     self.classType.setItem(row, 0, QtWidgets.QTableWidgetItem(cType["color"]))
        #     self.classType.item(row, 0).setBackground(QtGui.QColor(red, green, blue))
        #     self.classType.setItem(row, 1, QtWidgets.QTableWidgetItem(cType["label"]))
        #     self.classType.setItem(row, 2, QtWidgets.QTableWidgetItem(str(cType["train"])))
        #     self.classType.setItem(row, 3, QtWidgets.QTableWidgetItem(str(cType["val"])))
        #     self.classType.setItem(row, 4, QtWidgets.QTableWidgetItem(str(cType["test"])))
        #     row += 1

def setTables(row):
    # db 내부에 저장된 결과물의 개수를 저장한다.
    count = len(row)
    print

    # 개수만큼 테이블의 row를 생성한다.
    UI_set.classType.setRowCount(count)

    # row 리스트 만큼 반복하여 Table에 DB 값을 넣는다.
    for x in range(count):
        # 리스트 내부의 column쌍은 튜플로 반환하므로 튜플의 각 값을 변수에 저장
        idx, color, label, train, test, val = row[x]

        # 테이블의 각 셀에 값 입력
        UI_set.classType.setItem(x, 0, QTableWidgetItem(str(idx)))
        UI_set.classType.setItem(x, 1, QTableWidgetItem(color))
        UI_set.classType.setItem(x, 2, QTableWidgetItem(label))
        UI_set.classType.setItem(x, 3, QTableWidgetItem(str(train)))
        UI_set.classType.setItem(x, 4, QTableWidgetItem(str(test)))
        UI_set.classType.setItem(x, 4, QTableWidgetItem(str(val)))



if __name__ == "__main__" :
    app = QApplication(sys.argv) 
    myWindow = WindowClass() 
    myWindow.show()
    app.exec_()
