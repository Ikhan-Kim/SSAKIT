import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtCore import pyqtSignal

# DB 연동
import sqlite3

# 연결할 ui 파일의 경로 설정
UI_Path = './ui/ClassEdit.ui'
form_class = uic.loadUiType(UI_Path)[0]

class ClassEditWidget(QMainWindow, form_class) :
    # Signal 선언부
    send_valve_popup_signal = pyqtSignal(bool, name='sendValvePopupSignal')

    def __init__(self, data) :
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Class Edit Widget")

        self.lineEdit.setPlaceholderText("색상 코드 입력")
        self.lineEdit_2.setPlaceholderText("label 입력")

        # sql 연동
        self.sqlConnect()

        # edit 금지 모드
        # self.classTypeWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # 버튼 클릭시 데이터 입력
        self.pushButton.clicked.connect(self.insertData)

        self.classTypeWidget.cellClicked.connect(self.cellClick)

        # Table의 내부 셀을 클릭할 때 연결할 함수
        # 셀을 클릭하여 연결한 함수에는 기본적으로 셀의 Row, Column 두개의 인자를 넘겨준다.
        # self.classTypeWidget.cellClicked.connect(self.deleteData)
        
        self.okBtn.clicked.connect(self.hideFunc)

        # self.classTypeWidget.clicked.connect(self.checkCurrentIndex)

    ########################################################

    def hideFunc(self) :
        self.hide()

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
        self.classTypeWidget.setColumnCount(5)
        self.classTypeWidget.setHorizontalHeaderLabels(['idx', 'color', 'label', '수정', '삭제'])
        # self.classTypeWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # # Table 너비 조절
        self.classTypeWidget.setColumnWidth(0,40)
        self.classTypeWidget.setColumnWidth(3,50)
        self.classTypeWidget.setColumnWidth(4,50)
        self.classTypeWidget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.classTypeWidget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        
        cnt = len(rows)
        self.classTypeWidget.setRowCount(cnt)

        for x in range(cnt):
            # 리스트 내부의 column쌍은 튜플로 반환하므로 튜플의 각 값을 변수에 저장
            # print(rows)
            idx, color, label = rows[x][0], rows[x][1], rows[x][2]
            # print("idx, color, label : ", idx, color, label)
            
            # print("rows[x]", rows[x][0], rows[x][1], rows[x][2])
            # 테이블의 각 셀에 값 입력
            self.classTypeWidget.setItem(x, 0, QTableWidgetItem(str(idx)))
            self.classTypeWidget.setItem(x, 1, QTableWidgetItem(""))
            self.classTypeWidget.item(x, 1).setBackground(QtGui.QColor(color))
            self.classTypeWidget.setItem(x, 2, QTableWidgetItem(label))
            self.classTypeWidget.setItem(x, 3, QTableWidgetItem("수정"))
            self.classTypeWidget.setItem(x, 4, QTableWidgetItem("❌"))

    # DB) sql문 실행 함수
    def run(self):
        self.cur.execute(self.cmd)
        self.conn.commit()

    # DB) 종료 함수
    def closeEvent(self, QCloseEvent):
        print("DB close!")
        self.conn.close()

    def warningMSG(self, title: str, content: str):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.setStandardButtons(QMessageBox.Ok)
        result = msg.exec_()
        if result == QMessageBox.Ok:
            self.send_valve_popup_signal.emit(True)

    # insert
    def insertData(self):
        #두개의 lineEdit에서 각각 색과 className를 받아온다.

        color = self.lineEdit.text()
        label = self.lineEdit_2.text()
        if color == "" and label == "":
            self.warningMSG("주의", "color 및 label을 입력해주세요")
        elif color == "" :
            self.warningMSG("주의", "color를 입력해 주세요.")
        elif label == "":
            self.warningMSG("주의", "label을 입력해 주세요.")
        else:        
            conn = sqlite3.connect("test2.db")
            cur = conn.cursor()
            
            insertSql = "INSERT INTO classLabel (color, label) VALUES (?,?)"
            cur.execute(insertSql, (color, label))
            conn.commit()

        #데이터 입력 후 DB의 내용 불러와서 TableWidget에 넣기 위한 함수 호출
        self.selectData()

    # update & delete
    def cellClick(self, row, column):
        # update
        if column == 3:
            # print("수정버튼 클릭됨")
            conn = sqlite3.connect("test2.db")
            cur = conn.cursor()
            a = QMessageBox.question(self, "수정 확인", "정말로 수정 하시겠습니까?",
                                 QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
            if a == QMessageBox.Yes:

                idx = self.classTypeWidget.item(row, 0).text()
                # color_ = self.classTypeWidget.item(row, 1).
                # color는 DB에서 값 불러와서 DB 값 수정 => bg color 재설정
                # self.classTypeWidget.setItem(x, 1, QTableWidgetItem(""))
                # self.classTypeWidget.item(x, 1).setBackground(QtGui.QColor(color))
                color_ = "#1BE6EA" # 컬러는 임시 데이터 ㅜㅜ
                label_ = self.classTypeWidget.item(row, 2).text()
                # print("idx - ",idx, "label : ", label_)

                #DB의 데이터 idx는 선택한 Row의 첫번째 셀(0번 Column)의 값에 해당한다.
                updateSql = "UPDATE classLabel SET `color` = '{}', `label` = '{}' WHERE idx=?".format(color_, label_)
                cur.execute(updateSql, (idx,))
                conn.commit()        
                conn.close()

                # self.cmd = "insert into classLabel(`color`, `label`, `train`, `val`, `test`) values('{}', '{}', {}, {}, {})"\
                # .format(self.color, self.label, self.train, self.val, self.test)

                # self.cmd = "update test2 set `name` = '{}', `addr` = '{}' where `no` = {}"  \
                #     .format(self.txt이름.text(), self.txt주소.text(), self.txt번호.text())
                
            #데이터 삭제 후 DB의 내용 불러와서 TableWidget에 넣기 위한 함수 호출
            self.selectData()


            
        
        # delete
        ## 테이블 내부의 셀 클릭과 연결된 이벤트는 기본적으로 셀의 Row, Column을 인자로써 전달받는다.
        ## 삭제 셀이 눌렸을 때, 삭제 셀은 5번째 셀이므로 column 값이 4일 경우만 작동한다.
        elif column == 4:
            print(row, column, "셀클릭-로우4")
            conn = sqlite3.connect("test2.db")
            cur = conn.cursor()
            a = QMessageBox.question(self, "삭제 확인", "정말로 삭제 하시겠습니까?",
                                 QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
            if a == QMessageBox.Yes:
                #DB의 데이터 idx는 선택한 Row의 첫번째 셀(0번 Column)의 값에 해당한다.
                idx = self.classTypeWidget.item(row, 0).text()
                sql = "DELETE FROM classLabel WHERE idx =?"
                cur.execute(sql, (idx,))
                conn.commit()        
                conn.close()
            
            #데이터 삭제 후 DB의 내용 불러와서 TableWidget에 넣기 위한 함수 호출
            self.selectData()

    def updateData(self, row, column):
        if column == 3:
            print("수정버튼 클릭됨")
            idx = self.classTypeWidget.item(row, 0).text()
            color_ = self.classTypeWidget.item(row, 1).text()
            lable_ = self.classTypeWidget.item(row, 2).text()
            print(idx, color_, label_)
            

            # conn = sqlite3.connect("test2.db")
            # cur = conn.cursor()
            # a = QMessageBox.question(self, "수정 확인", "정말로 수정 하시겠습니까?",
            #                      QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
            # if a == QMessageBox.Yes:
            #     #DB의 데이터 idx는 선택한 Row의 첫번째 셀(0번 Column)의 값에 해당한다.
            #     idx = self.classTypeWidget.item(row, 0).text()
            #     sql = "DELETE FROM classLabel WHERE idx =?"
            #     updateSql = "UPDATE classLabel SET _____ WHERE idx=?".format()
            #     cur.execute(updateSql, (idx,))
            #     conn.commit()        
            #     conn.close()
            
            # #데이터 삭제 후 DB의 내용 불러와서 TableWidget에 넣기 위한 함수 호출
            # self.selectData()
            
    
    def edit(self):
        if (self.txt이름.text() != "") and (self.txt주소.text() != ""):
            try:
                self.cmd = "update test2 set `name` = '{}', `addr` = '{}' where `no` = {}"  \
                    .format(self.txt이름.text(), self.txt주소.text(), self.txt번호.text())
                print(self.cmd)
                self.cur.execute(self.cmd)
                self.conn.commit()
            except:
                QMessageBox.information(self, "삽입 오류", "올바른 형식으로 입력하세요.",
                                        QMessageBox.Yes, QMessageBox.Yes)
                return
        else:
            QMessageBox.information(self, "입력 오류", "빈칸 없이 입력하세요.",
                                    QMessageBox.Yes, QMessageBox.Yes)
            return
        QMessageBox.information(self, "수정 성공", "수정되었습니다.",
                                QMessageBox.Yes, QMessageBox.Yes)

    # # delete
    # def deleteData(self, row, column):
    #     # 테이블 내부의 셀 클릭과 연결된 이벤트는 기본적으로 셀의 Row, Column을 인자로써 전달받는다.
    #     # 삭제 셀이 눌렸을 때, 삭제 셀은 4번째 셀이므로 column 값이 3일 경우만 작동한다.
    #     if column == 4: 
    #         conn = sqlite3.connect("test2.db")
    #         cur = conn.cursor()
    #         a = QMessageBox.question(self, "삭제 확인", "정말로 삭제 하시겠습니까?",
    #                              QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
    #         if a == QMessageBox.Yes:
    #             #DB의 데이터 idx는 선택한 Row의 첫번째 셀(0번 Column)의 값에 해당한다.
    #             idx = self.classTypeWidget.item(row, 0).text()
    #             sql = "DELETE FROM classLabel WHERE idx =?"
    #             cur.execute(sql, (idx,))
    #             conn.commit()        
    #             conn.close()
            
    #         #데이터 삭제 후 DB의 내용 불러와서 TableWidget에 넣기 위한 함수 호출
    #         self.selectData()

if __name__ == "__main__" :
    app = QApplication(sys.argv) 
    myWindow = ClassEditWidget(form_class) 
    myWindow.show()
    app.exec_()
