import os

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QProcess, Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout


def has_bash():
    process = QProcess()
    process.start("which bash")
    process.waitForStarted()
    process.waitForFinished()
    if process.exitStatus() == QProcess.NormalExit:
        return bool(process.readAll())
    return False


class PipManager(QObject):
    started = pyqtSignal()
    finished = pyqtSignal()
    textChanged = pyqtSignal(str)

    def __init__(self, venv_dir, venv_name, parent=None):
        super().__init__(parent)

        self._venv_dir = venv_dir
        self._venv_name = venv_name

        self._process = QProcess(self)
        self._process.readyReadStandardError.connect(self.onReadyReadStandardError)
        self._process.readyReadStandardOutput.connect(self.onReadyReadStandardOutput)
        self._process.stateChanged.connect(self.onStateChanged)
        self._process.started.connect(self.started)
        self._process.finished.connect(self.finished)
        self._process.finished.connect(self.onFinished)
        self._process.setWorkingDirectory(venv_dir)

    def run_command(self, command="", options=None):
        if has_bash():
            if options is None:
                options = []
            # script = f"""source {self._venv_name}/bin/activate; pip {command} {" ".join(options)}; deactivate;"""
            script = "source {}/bin/activate; pip {} {}; deactivate;".format(self._venv_name, command, " ".join(options))
            self._process.start("bash", ["-c", script])

    @pyqtSlot(QProcess.ProcessState)
    def onStateChanged(self, state):
        if state == QProcess.NotRunning:
            print("not running")
        elif state == QProcess.Starting:
            print("starting")
        elif state == QProcess.Running:
            print("running")

    @pyqtSlot(int, QProcess.ExitStatus)
    def onFinished(self, exitCode, exitStatus):
        print(exitCode, exitStatus)

    @pyqtSlot()
    def onReadyReadStandardError(self):
        message = self._process.readAllStandardError().data().decode().strip()
        print("error:", message)
        self.finished.emit()
        self._process.kill()
        """self.textChanged.emit(message)"""

    @pyqtSlot()
    def onReadyReadStandardOutput(self):
        message = self._process.readAllStandardOutput().data().decode().strip()
        self.textChanged.emit(message)


class ProgBarDialog(QDialog):
    """
    Dialog showing output and a progress bar during the installation process.
    """

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setFixedWidth(400)
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.setWindowFlags(Qt.WindowMinimizeButtonHint)

        self.statusLabel = QLabel()
        self.placeHolder = QLabel()

        self.progressBar = QProgressBar()
        self.progressBar.setFixedHeight(23)
        self.progressBar.setRange(0, 0)

        v_Layout = QVBoxLayout(self)
        v_Layout.addWidget(self.statusLabel)
        v_Layout.addWidget(self.progressBar)
        v_Layout.addWidget(self.placeHolder)

    @pyqtSlot(str)
    def update_status(self, status):
        metrix = QFontMetrics(self.statusLabel.font())
        clippedText = metrix.elidedText(status, Qt.ElideRight, self.statusLabel.width())
        self.statusLabel.setText(clippedText)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    progBar = ProgBarDialog()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    venv_name = "testenv"

    manager = PipManager(current_dir, venv_name)
    manager.textChanged.connect(progBar.update_status)
    manager.started.connect(progBar.show)
    manager.finished.connect(progBar.close)

    manager.run_command("install", ["--upgrade", "pylint"])

    sys.exit(app.exec_())