import pyqtgraph as pg

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, Qt, QThread, QTimer

import time
import random


class ExMain(QWidget):
    def __init__(self):
        super().__init__()

        hbox = QHBoxLayout()
        self.pw1 = pg.PlotWidget(title="line chart")

        hbox.addWidget(self.pw1)
        self.setLayout(hbox)

        # self.setGeometry(300, 100, 800, 500)  # x, y, width, height
        self.setWindowTitle("pyqtgraph 예제 - realtime")

        self.x = [1, 2, 3]
        self.y = [4, 5, 6]

        # self.pw2.enableAutoRange()
        self.pl = self.pw1.plot(pen='g')

        self.mytimer = QTimer()
        self.mytimer.start(100)  # 1초마다 차트 갱신 위함...
        self.mytimer.timeout.connect(self.get_data)

        self.draw_chart(self.x, self.y)
        self.show()

    def draw_chart(self, x, y):
        self.pl.setData(x=x, y=y)  # line chart 그리기

        cnt = len(y)
        new_y = []
        for i in range(cnt):
            new_y.append(random.random()*60)  # 0 이상 ~ 60 미만 random 숫자 만들기

    @pyqtSlot()
    def get_data(self):
        # print(time.localtime())
        # print(time.strftime("%H%M%S", time.localtime()))
        data = time.strftime("%S", time.localtime())  # 초 단위만 구함.

        last_x = self.x[-1]
        self.x.append(last_x + 1)

        self.y.append(int(data))
        self.draw_chart(self.x, self.y)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    ex = ExMain()

    sys.exit(app.exec_())
