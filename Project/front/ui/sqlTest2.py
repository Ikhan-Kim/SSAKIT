import sqlite3
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtCore import Qt

class sql(QWidget):
    def __init__(self):
        super().__init__()
        self.sqlConnect()
        self.initUI()
        self.run()

    def sqlConnect(self):
        # sql 연결
        try: 
            self.conn = sqlite3.connect("dbName", isolation_level=None)
        except:
            print("문제가 있네요!")
            exit(1)
        print("연결성공!")
        self.cur = self.conn.cursor()

    def initUI(self):
        # UI => ClassEditWidget 화면
        # self.setGeometry(300, 300, 500, 520)
        # self.setWindowTitle("DB 활용 예제")

        self.w = 400
        self.h = 420
        self.btnSize = 40

        self.cmd이전 = QPushButton("이전", self)
        self.cmd이전.resize(self.btnSize, self.btn)


        self.show()

    def run(self):
        # sql 명령어 입력
        self.cmd = "CREATE TABLE IF NOT EXISTS table1 \
    (id integer PRIMARY KEY, color text, label text)"
        # desc table1
        self.cur.execute(self.cmd)
        self.conn.commit()
        print(self.cur.fetchall())

    def closeEvent(self, QCloseEvent):
        # sql 종료
        print("DB close!")
        self.conn.close()


app = QApplication(sys.argv)
w = sql()
sys.exit(app.exec_())