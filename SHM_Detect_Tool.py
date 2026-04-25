# -*- coding: utf-8 -*-
###################################################
# SHM Detect Tool                                 #
# Version: Beta 0.6                               #
#                                                 #
# Sep. 02, 2020                                   #
# A Tool to Detect the Result of SHM              #
###################################################
import torch
import re, sys, json
import matplotlib.pyplot as plt

sys.path.append("..")
from Source import src_pytorch_public as src
# import SHM_keywords_setting as setting
from PyQt5.QtWidgets import *
import qtawesome as qta
import shm_backend

__version__ = 'SHM Detect Tool Beta V0.8.0'
__author__ = 'zhouchao486@gmail.com'


class Application(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUI()
        self.filename = ''

    def setupUI(self):
        # Title and window size
        self.setWindowTitle(__version__)
        self.resize(500, 200)

        # Buttons
        self.status_text = QLabel()
        self.status_text.setText('Welcome!')

        # Load SHM
        self.load_shm_button = QPushButton(qta.icon('mdi.folder-open', color='blue'), 'Convert Shmoo to CSV')
        self.load_shm_button.setToolTip('Convert Shmoo log to CSV')
        self.load_shm_button.clicked.connect(self.load_shm)

        # Train net
        self.train_net_button = QPushButton(qta.icon('mdi.brain', color='blue'), 'Training CNN')
        self.train_net_button.setToolTip('Train a new cnn network')
        self.train_net_button.clicked.connect(lambda: self.cnn_net('training'))

        # Analyse
        self.analyse_shm_path_let = QLineEdit()
        self.analyse_shm_path_btn = QPushButton('Select CSV Shmoo File...')
        self.analyse_shm_path_btn.clicked.connect(self.get_csv_shm_path)
        self.analyse_shm_button = QPushButton(qta.icon('mdi.test-tube', color='blue'), 'Analyse Shmoo Log')
        self.analyse_shm_button.setToolTip('Analyse Shmoo Log')
        self.analyse_shm_button.clicked.connect(lambda: self.cnn_net('test'))

        # Exam CNN
        self.exam_cnn_button = QPushButton(qta.icon('mdi.kubernetes', color='blue'), 'Exam CNN')
        self.exam_cnn_button.setToolTip('Look into CNN')
        self.exam_cnn_button.clicked.connect(lambda: self.cnn_net('exam'))

        self.exam_cnn_label = QLabel()
        self.exam_cnn_label.setText('CNN Layer Index')
        self.qLineEdit = QLineEdit()

        # Layout
        layout = QGridLayout()
        self.setLayout(layout)

        # Tabs
        tabs = QTabWidget(self)
        self.train_cnn = QWidget()
        self.convt_shm = QWidget()
        self.detect_shm = QWidget()
        self.tab_train_cnn()
        self.tab_convt_shm()
        self.tab_detect_shm()
        tabs.addTab(self.train_cnn, 'Train CNN')
        tabs.addTab(self.convt_shm, 'Convert Shmoo')
        tabs.addTab(self.detect_shm, 'Detect Shmoo')

        layout.addWidget(tabs, 0, 0)
        layout.addWidget(self.status_text, 1, 0)

    def tab_train_cnn(self):
        layout = QGridLayout()
        layout.addWidget(self.train_net_button, 0, 0)
        self.train_cnn.setLayout(layout)

    def tab_convt_shm(self):
        layout = QGridLayout()

        layout.addWidget(self.load_shm_button, 0, 0)
        self.convt_shm.setLayout(layout)

    def tab_detect_shm(self):
        layout = QGridLayout()
        layout.addWidget(self.analyse_shm_path_let, 0, 0, 1, 6)
        layout.addWidget(self.analyse_shm_path_btn, 0, 6, 1, 2)
        layout.addWidget(self.analyse_shm_button, 0, 8, 1, 2)

        layout.addWidget(self.exam_cnn_label, 2, 0, 1, 4)
        layout.addWidget(self.qLineEdit, 2, 4, 1, 4)
        layout.addWidget(self.exam_cnn_button, 2, 8, 1, 2)
        self.detect_shm.setLayout(layout)

    def get_csv_shm_path(self):
        filterboi = 'csv File (*.csv)'
        filepath = QFileDialog.getOpenFileName(caption='Select CSV Shmoo File', filter=filterboi)
        if filepath[0] == '':
            self.status_text.setText('Please select a file')
        else:
            self.status_text.update()
            self.status_text.setText('Converting...')
            filename = filepath[0]
            self.filename = filename
            self.analyse_shm_path_let.setText(self.filename)

    def load_shm(self):
        filterboi = 'TXT log (*.txt)'
        filepath = QFileDialog.getOpenFileName(
            caption='Open SHM Log File', filter=filterboi)
        # Open std file/s
        if len(filepath[0]) == 0:
            pass
        else:
            if len(filepath[0]) > 1:
                filename = filepath[0]
                self.filename = filename
                self.read_shm_log(filename)
            else:
                pass

    def read_shm_log(self, filename):
        """Parse shmoo log → CSV. Delegates to backend."""
        try:
            with open(r'SHM_keywords_setting.json') as config_file:
                config_details = json.load(config_file)

            csv_path, _ = shm_backend.read_shm_log(filename, config_details)
            print(f"CSV written: {csv_path}")

        except Exception as e:
            QMessageBox.information(
                self, 'Error', e.__str__(),
                QMessageBox.Ok)

    def cnn_net(self, mode='training'):
        if mode == 'training':
            # Delegate training to backend
            net, test_iter = shm_backend.train_model()

            # GUI-specific: show validation plots
            net.eval()
            X, y = next(iter(test_iter))
            true_labels = src.get_custom_shm_labels(y.numpy(), 'E')
            y_hat = net(X)
            y_hat = src.reformat_output(y_hat)
            pred_labels = src.get_custom_shm_labels(
                y_hat.detach().numpy(), 'A')
            titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
            src.show_shm_fig(X[0:100], titles[0:100])

        elif mode == 'test':
            # Load model via backend helpers
            net = src.AlexNet()
            net.load_state_dict(torch.load('./state_dict.pth', weights_only=True))
            net.eval()

            filename = self.analyse_shm_path_let.text()
            titles = []
            titles_plot = []
            shmoo_body, shmoo_title, shmoo_dict = shm_backend.read_shmoo_csv(filename)
            _, figs = plt.subplots(5, 10, figsize=(12, 8))
            plt.tight_layout()
            figs = figs.flatten()
            for i in range(len(shmoo_title)):
                test_iter, tmp_raw_dict = shm_backend.convert_shm_to_tensor(
                    filename, -1, shmoo_body[i], shmoo_title[i], 'S')
                X, y = next(iter(test_iter))
                y_hat = net(X)
                y_hat = src.reformat_output(y_hat)
                true_labels = y[0]
                pred_labels = src.get_custom_shm_labels(y_hat.detach().numpy(), 'A')
                titles_plot.append(true_labels + '\n' + pred_labels[0])
                titles.append(true_labels + ':' + pred_labels[0])
                if i < 50:
                    figs[i].imshow(X[0].view((X[0].shape[1], X[0].shape[2])).numpy(), cmap='RdYlGn')
                    figs[i].set_title(titles_plot[-1])
                    figs[i].axes.get_xaxis().set_visible(False)
                    figs[i].axes.get_yaxis().set_visible(False)
            if titles:
                shm_backend.generate_shm_report_xlsx(titles, shmoo_dict, filename)
            plt.show()

        else:
            # Eval mode — delegate to backend, visualize activations in GUI
            filename = self.analyse_shm_path_let.text()
            text = self.qLineEdit.text()
            channel_index = int(text)

            act, channel_count_list = shm_backend.eval_model(
                csv_file=filename,
                model_path='./state_dict.pth',
                shmoo_index=5,
                channel_index=channel_index,
            )

            # GUI-specific: Visualization output
            fig = plt.figure()
            total_count = channel_count_list[channel_index]
            for i in range(total_count):
                ax = fig.add_subplot(int(total_count / 4), 4, i + 1, xticks=[], yticks=[])
                ax.imshow(act[0][i].detach().numpy(), cmap="gray")
            plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Application()
    viewer.show()
    sys.exit(app.exec_())
