from tensorflow import keras
import time

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, input_epochs, window):
        self.input_epochs = input_epochs
        self.textBox_terminal = window.textBox_terminal
        self.fig = window.fig
        self.canvas = window.canvas

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        # self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.i += 1

        # 터미널 출력
        self.textBox_terminal.append(
            "Epoch {}/{} : loss = {}, accuracy = {}, val_loss = {}, val_accuracy = {}".format(self.i, self.input_epochs, round(self.losses[-1], 4), round(self.acc[-1], 4), round(self.val_losses[-1], 4), round(self.val_acc[-1], 4)))

        self.fig.clear()
        ax1 = self.fig.add_subplot(121)
        ax1.plot(self.x, self.acc, label="train_accuracy")
        ax1.plot(self.x, self.val_acc, label="val_accuracy")
        ax1.legend()
        ax1.set_title("accuracy")
        ax2 = self.fig.add_subplot(122)
        ax2.plot(self.x, self.losses, label="train_loss")
        ax2.plot(self.x, self.val_losses, label="val_loss")
        ax2.legend()
        ax2.set_title("loss")

        if self.i == self.input_epochs:
            now = time.gmtime(time.time())
            file_name = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + \
                str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
            self.fig.savefig('result_logs\\'+file_name)
            print("figure saved!")

        self.canvas.draw()