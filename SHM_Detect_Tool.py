# -*- coding: utf-8 -*-
###################################################
# SHM Detect Tool                                 #
# Version: Beta 0.1                               #
#                                                 #
# Sep. 02, 2020                                   #
# A Tool to Detect the Result of SHM              #
###################################################
import torch
import sys
import re
import xlsxwriter

sys.path.append("..")
import d2lzh_pytorch as d2l

from PyQt5.QtWidgets import *
import qtawesome as qta

__version__ = 'SHM Detect Tool Beta V0.4.6'
__author__ = 'zhouchao486@gmail.com'


class Application(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUI()
        self.result_dict = {}

    def setupUI(self):
        # Title and window size
        self.setWindowTitle(__version__)
        self.resize(200, 100)
        # Load SHM
        self.load_shm_button = QPushButton(qta.icon('mdi.folder-open', color='blue'), 'Load SHM Log')
        self.load_shm_button.setToolTip('Load SHM log')
        self.load_shm_button.clicked.connect(self.load_shm)

        # Train net
        self.train_net_button = QPushButton(qta.icon('mdi.brain', color='blue'), 'Training CNN Net')
        self.train_net_button.setToolTip('Train a new cnn net')
        self.train_net_button.clicked.connect(lambda: self.cnn_net('training'))

        # Analyse
        self.analyse_shm_button = QPushButton(qta.icon('mdi.test-tube', color='blue'), 'Analyse SHM Log')
        self.analyse_shm_button.setToolTip('Train a new cnn net')
        self.analyse_shm_button.clicked.connect(lambda: self.cnn_net('test'))

        # Config layout
        layout = QGridLayout()
        layout.addWidget(self.load_shm_button, 0, 1)
        layout.addWidget(self.train_net_button, 0, 0)
        layout.addWidget(self.analyse_shm_button, 0, 2)
        self.setLayout(layout)

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
                self.read_shm_log(filename)
            else:
                pass

    def read_shm_log(self, filename):
        new_shm_flag = False
        shm_start_flag = False
        with open(filename, 'r') as buffer:
            while True:
                line = buffer.readline()
                if line.startswith('FC_') and line.endswith('_SHM:\n'):
                    cur_instance = line[0:-1]
                    line = buffer.readline()
                    if line.startswith('Site:'):
                        res = re.search('\d+', line)
                        cur_site_index = res.group() + ',' * 10
                        self.result_dict[cur_instance + cur_site_index] = []
                        new_shm_flag = True
                        shm_start_flag = False

                else:
                    if new_shm_flag:
                        res = re.search('(\t(P|\*|\.|#))+', line)
                        if (res is not None) and shm_start_flag == False:
                            shm_start_flag = True
                        elif (res is None) and shm_start_flag == True:
                            new_shm_flag = False
                            shm_start_flag = False

                        if shm_start_flag:
                            tmp = res.group().split('\t')[1:]
                            self.result_dict[cur_instance + cur_site_index].append(tmp)

                if len(line) == 0:
                    break

        with open('my_file.csv', 'w') as f:
            for key, values in self.result_dict.items():
                f.write('{0}\n'.format(key))
                for val in values:
                    f.write(','.join(i for i in val))
                    f.write('\n')
            # [f.write('{0}\n{1}\n'.format(key, value)) for key, value in self.result_dict.items()]
        # self.cnn_net('test')

    def cnn_net(self, mode='training'):
        # %% define&initial module
        # num_inputs, num_outputs, num_hiddens = 121, 2, 64  # 784, 10, 256
        # net = nn.Sequential(
        #     d2l.FlattenLayer(),
        #     nn.Linear(num_inputs, num_hiddens),
        #     nn.ReLU(),
        #     nn.Linear(num_hiddens, num_outputs),
        # )
        # for params in net.parameters():
        #     init.normal_(params, mean=0, std=0.01)

        # net = d2l.LeNet()
        net = d2l.AlexNet()

        if mode == 'training':
            net.train()
            # %% load data
            batch_size = 100  # 256
            filename = r'custom_SHM_data.csv'
            train_iter, test_iter = d2l.load_custom_shm_data(batch_size,
                                                             filename)  # d2l.load_data_fashion_mnist(batch_size)

            # %% define loss function
            # loss = torch.nn.CrossEntropyLoss()
            # nn.BCEWithLogitsLoss takes the raw logits of your model (without any non-linearity) and applies the sigmoid internally
            loss = torch.nn.BCEWithLogitsLoss()  # BCELoss() #MultiLabelSoftMarginLoss() #BCELoss()

            # %% optimise function
            lr = 0.01
            # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            # %% run training
            num_epochs = 150  # 320
            d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, lr, optimizer)

            # %% save the state
            torch.save(net.state_dict(), './stat_dict.pth')

            # %% show result
            net.eval()
            # print(net.training)
            X, y = iter(test_iter).next()
            true_labels = d2l.get_custom_shm_labels(y.numpy(), 'E')  # d2l.get_fashion_mnist_labels(y.numpy())
            y_hat = net(X)
            y_hat[y_hat > 0.5] = 1
            y_hat[y_hat <= 0.5] = 0
            pred_labels = d2l.get_custom_shm_labels(
                y_hat.detach().numpy(),
                'A')  # d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy()) net(X).detach().numpy()
            titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
            d2l.show_fashion_mnist(X[0:45], titles[0:45])
        else:
            # %% load state dict
            net.load_state_dict(torch.load('./stat_dict.pth'))
            # %% show result
            net.eval()
            test_iter, raw_dict = self.convert_shm_to_tensor()
            X, y = iter(test_iter).next()
            y_hat = net(X)
            y_hat[y_hat > 0.5] = 1
            y_hat[y_hat <= 0.5] = 0
            true_labels = y[0]
            pred_labels = d2l.get_custom_shm_labels(y_hat.detach().numpy(), 'A')
            titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
            d2l.show_fashion_mnist(X[0:45], titles[0:45])
            titles = [true + ':' + pred for true, pred in zip(true_labels, pred_labels)]
            self.generate_shm_report_xlsx(titles, raw_dict)

    def convert_shm_to_tensor(self):
        if sys.platform.startswith('win'):
            num_workers = 0  # 0
        else:
            num_workers = 4

        dataset = d2l.CsvDataset_Test('my_file.csv')
        batch_size = dataset.__len__()  # len(self.result_dict)
        test_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return test_iter, dataset.raw_dict

    def generate_shm_report_xlsx(self, titles, shms):
        report_name = 'report.xlsx'
        # In case someone has the file open
        try:
            # Create a workbook and add a  worksheet.
            workbook = xlsxwriter.Workbook(report_name)
            worksheet = workbook.add_worksheet('SHM Result')

            # Light red
            format_2XXX = workbook.add_format({'bg_color': '#FF0000'})
            # Dark green
            format_7XXX = workbook.add_format({'bg_color': '#008000'})

            # Optimise xlsx output format
            worksheet.outline_settings(True, False, True, False)
            worksheet.write_row(0, 0, ['Instance', '', '', '', 'Site Index', 'Result Symbol', 'Result'])
            row = 1
            for title, shm in zip(titles, shms):
                info_line = title.split(':')
                info_line[1:1] = [''] * 3
                worksheet.write_row(row, 0, info_line)
                worksheet.set_row(row, None, None, {'collapsed': True})
                row += 1
                for i in range(len(shms[shm])):
                    worksheet.write_row(row, 0, shms[shm][i])
                    worksheet.set_row(row, None, None, {'level': 1, 'hidden': True})
                    row += 1

                # row += shm.size(1)
            col = len(shms[shm][i])
            worksheet.conditional_format(0, 0, row, col,
                                         {'type': 'cell', 'criteria': 'equal to',
                                          'value': '"."', 'format': format_2XXX})
            worksheet.conditional_format(0, 0, row, col,
                                         {'type': 'cell', 'criteria': 'equal to',
                                          'value': '"P"', 'format': format_7XXX})
            workbook.close()
            print('Xlsx file is written!')

        except xlsxwriter.exceptions.FileCreateError:  # PermissionError:
            print("Please close " + report_name.split('/')[-1])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Application()
    viewer.show()
    sys.exit(app.exec_())
