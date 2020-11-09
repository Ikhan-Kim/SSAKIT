import pyqtgraph as pg
# import pyqtgraph.exporters
# import numpy as np

from PyQt5.QtWidgets import *


class MyMainWindow(QMainWindow):
    """
    QMainWindow 의 central widget 으로 pyqtgraph의 PlotWidget() 사용함.
    """

    def __init__(self):
        super().__init__()
        # pg.setConfigOption('background', 'y')  # global configuration options
        # pg.setConfigOptions(background='w')  # global configuration options
        # pg.setConfigOptions(background='w', foreground='b')  # key-value 형태로 여러개 인자 사용 가능.

        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        x = range(0, 10)

        # pw = pg.PlotWidget(background='w', title="aaaa")
        # title 매개변수는 내부적으로 PlotItem() 에서 사용됨.
        pw = pg.PlotWidget(title="기본예제")
        # pw = pg.PlotWidget()

        """
        ** 실제 차트 그리는 명령어는 아래 2가지 중 하나 사용하면됨.
        
        pw.plot(x, y, pen='r') 
        
        또는 
        
        pdi = pw.plot()  # PlotDataItem obj 반환
        pdi.setData(x, y, pen='g')
        """
        # pw.plot(x, y)
        # pw.plot(x, y, pen='r')  # plot() 메소드는 내부적으로 PlotItem 의 plot() 을 사용함.

        pdi = pw.plot()  # PlotDataItem obj 반환
        pdi.setData(x, y, pen='g')

        self.setCentralWidget(pw)  # pyqt5 와 pyqtgraph 연결.

        # self.setGeometry(300, 700, 350, 500)
        self.show()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    mywin = MyMainWindow()

    sys.exit(app.exec_())
