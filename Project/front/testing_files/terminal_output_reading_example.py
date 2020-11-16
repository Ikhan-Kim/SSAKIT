# ref to https://www.saltycrane.com/blog/2007/12/pyqt-example-how-to-run-command-and/
import os
import sys
from PyQt5 import QtWidgets, QtCore
import subprocess


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())


class MyWindow(QtWidgets.QWidget):
    errorSignal = QtCore.pyqtSignal(str) 
    outputSignal = QtCore.pyqtSignal(str)
    def __init__(self, *args):
        QtWidgets.QWidget.__init__(self, *args)

        # create objects
        label = QtWidgets.QLabel(self.tr("Enter command and press Return"))
        self.le = QtWidgets.QLineEdit()
        self.te = QtWidgets.QTextEdit()
        self.process = QtCore.QProcess()
        self.process.readyReadStandardError.connect(self.onReadyReadStandardError)
        self.process.readyReadStandardOutput.connect(self.onReadyReadStandardOutput)

        # layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.le)
        layout.addWidget(self.te)
        self.setLayout(layout)

        # create connection
        self.le.returnPressed.connect(self.run_command)

    def run_command(self):
        cmd = str(self.le.text())
        # stdouterr = subprocess.Popen(cmd)
        self.process.start(cmd)
    
    def onReadyReadStandardError(self):
        error = self.process.readAllStandardError().data().decode()
        self.te.setText(error)
        self.errorSignal.emit(error)

    def onReadyReadStandardOutput(self):
        result = self.process.readAllStandardOutput().data().decode()
        self.te.setText(result)
        self.outputSignal.emit(result)

if __name__ == "__main__":
    main()