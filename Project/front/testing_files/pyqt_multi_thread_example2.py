from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time
import traceback, sys, os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data
    
    error
        `tuple` (exctype, value, traceback.format_exc() )
    
    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress 

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, fig, canvas, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.fig = fig
        self.canvas = canvas
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()    

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress        

    @pyqtSlot()
    def run(self):
        TRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Define hyperparameter
        INPUT_SIZE = 32
        CHANNELS = 3
        NUM_CLASSES = 10
        NUM_TRAIN_IMGS = 50000
        NUM_TEST_IMGS = 10000

        BATCH_SIZE = 128
        train_steps_per_epoch = NUM_TRAIN_IMGS // BATCH_SIZE
        val_steps_per_epoch = NUM_TEST_IMGS // BATCH_SIZE

        # Data Preprocessing
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = X_train/255.0
        X_test = X_test/255.0

        # Load pre-trained model
        base_model = tf.keras.applications.VGG16(include_top=False,
                                                 weights='imagenet',
                                                 input_shape=(INPUT_SIZE, INPUT_SIZE, CHANNELS),)

        # Freeze the pre-trained layers
        base_model.trainable = False

        # Add a fully connected layer
        model_input = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, CHANNELS))
        model_output = tf.keras.layers.Flatten()(model_input)
        model_output = tf.keras.layers.Dense(
            512, activation='relu')(model_output)
        model_output = tf.keras.layers.Dropout(0.2)(model_output)
        model_output = tf.keras.layers.Dense(
            256, activation='relu')(model_output)
        model_output = tf.keras.layers.Dropout(0.2)(model_output)
        model_output = tf.keras.layers.Dense(
            NUM_CLASSES, activation='softmax')(model_output)
        model = tf.keras.Model(model_input, model_output)

        model.summary()

        # Compile
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Callbacks
        checkpoint_filepath = os.path.join(
            TRAIN_DIR, 'learning_test/checkpoint/VGG16_cifar10.h5')
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_accuracy',
                                             #  restore_best_weights=True
                                             ),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                               monitor='val_accuracy',
                                               mode='max',
                                               save_best_only=True,
                                               # save_weights_only=True,
                                               ),
            # PlotLossesKeras(),
        ]

        class PlotLosses(keras.callbacks.Callback):
            def __init__(self, figure, canvas):
                self.fig = figure
                self.canvas = canvas

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

                # plt.plot(self.losses, label="loss")
                # plt.legend()

                # plt.draw()
                # plt.pause(0.01)
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                ax.plot(self.x, self.losses, label="losses")
                ax.set_title("loss plot")
                self.canvas.draw()

                if self.i == 5:
                    now = time.gmtime(time.time())
                    file_name = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + \
                        str(now.tm_hour) + str(now.tm_min) + str(now.tm_sec)
                    plt.savefig('result_logs\\'+file_name)
                # plt.clf()

        plot_losses = PlotLosses(self.fig, self.canvas)

        # training model
        history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=5, steps_per_epoch=train_steps_per_epoch, validation_data=(
            X_test, Y_test), validation_steps=val_steps_per_epoch, verbose=1,  callbacks=plot_losses)

        plt.close()
        


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
    
        self.counter = 0
    
        layout = QVBoxLayout()
        
        self.l = QLabel("Start")
        b = QPushButton("DANGER!")
        b.pressed.connect(self.oh_no)
    
        layout.addWidget(self.l)
        layout.addWidget(b)
    
        #
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        #

        w = QWidget()
        w.setLayout(layout)
    
        self.setCentralWidget(w)
        self.show()

        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()
    
    def progress_fn(self, n):
        print("%d%% done" % n)

    def execute_this_fn(self, progress_callback):
        for n in range(0, 5):
            time.sleep(1)
            progress_callback.emit(n*100/4)
            
        return "Done."
 
    def print_output(self, s):
        print(s)
        
    def thread_complete(self):
        print("THREAD COMPLETE!")
 
    def oh_no(self):
        # Pass the function to execute
        worker = Worker(self.execute_this_fn, self.fig, self.canvas) # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)
        
        # Execute
        self.threadpool.start(worker) 

        
    def recurring_timer(self):
        self.counter +=1
        self.l.setText("Counter: %d" % self.counter)
    
    
app = QApplication([])
window = MainWindow()
app.exec_()