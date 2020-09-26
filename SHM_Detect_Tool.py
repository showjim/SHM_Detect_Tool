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
import matplotlib.pyplot as plt

sys.path.append("..")
import src_pytorch as src

from PyQt5.QtWidgets import *
import qtawesome as qta

__version__ = 'SHM Detect Tool Beta V0.5.2'
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
        self.train_net_button = QPushButton(qta.icon('mdi.brain', color='blue'), 'Training CNN')
        self.train_net_button.setToolTip('Train a new cnn network')
        self.train_net_button.clicked.connect(lambda: self.cnn_net('training'))

        # Analyse
        self.analyse_shm_button = QPushButton(qta.icon('mdi.test-tube', color='blue'), 'Analyse SHM Log')
        self.analyse_shm_button.setToolTip('Analyse SHM Log')
        self.analyse_shm_button.clicked.connect(lambda: self.cnn_net('test'))

        # Exam CNN
        self.exam_cnn_button = QPushButton(qta.icon('mdi.kubernetes', color='blue'), 'Exam CNN')
        self.exam_cnn_button.setToolTip('Look into CNN')
        self.exam_cnn_button.clicked.connect(lambda: self.cnn_net('exam'))

        self.exam_cnn_label = QLabel()
        self.exam_cnn_label.setText('CNN Layer Index')
        self.qLineEdit = QLineEdit()

        # Config layout
        layout = QGridLayout()
        layout.addWidget(self.train_net_button, 0, 0)
        layout.addWidget(self.load_shm_button, 0, 1)
        layout.addWidget(self.analyse_shm_button, 0, 2)

        layout.addWidget(self.exam_cnn_label, 1, 0)
        layout.addWidget(self.qLineEdit, 1, 1)
        layout.addWidget(self.exam_cnn_button, 1, 2)
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
        net = src.AlexNet()

        # print parameters count
        pytorch_total_params = sum(p.numel() for p in net.parameters())
        print('neural network architecture has ', pytorch_total_params, ' parameters.')
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('neural network architecture has ', pytorch_total_params, ' trainable parameters.')

        if mode == 'training':
            net.train()
            # %% load data
            batch_size = 100  # 256
            filename = r'custom_SHM_data.csv'
            train_iter, test_iter = src.load_custom_shm_data(batch_size,
                                                             filename)  # d2l.load_data_fashion_mnist(batch_size)

            # %% define loss function
            # loss = torch.nn.CrossEntropyLoss()
            # nn.BCEWithLogitsLoss takes the raw logits of your model (without any non-linearity) and applies the sigmoid internally
            loss = torch.nn.MultiLabelSoftMarginLoss() #BCEWithLogitsLoss()  # BCELoss() #MultiLabelSoftMarginLoss() #BCELoss()

            # %% optimise function
            lr = 0.001
            # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)

            # %% run training
            num_epochs = 100  # 320
            src.train_network(net, train_iter, test_iter, loss, num_epochs, batch_size, None, lr, optimizer)

            # %% save the state
            torch.save(net.state_dict(), './stat_dict.pth')

            # %% show result
            net.eval()
            # print(net.training)
            X, y = iter(test_iter).next()
            true_labels = src.get_custom_shm_labels(y.numpy(), 'E')  # d2l.get_fashion_mnist_labels(y.numpy())
            y_hat = net(X)
            y_hat[y_hat > 0.5] = 1
            y_hat[y_hat <= 0.5] = 0
            pred_labels = src.get_custom_shm_labels(
                y_hat.detach().numpy(),
                'A')  # d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy()) net(X).detach().numpy()
            titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
            src.show_shm_fig(X[0:45], titles[0:45])

        elif mode == 'test':
            # %% load state dict
            net.load_state_dict(torch.load('./stat_dict.pth'))
            # %% show result
            net.eval()
            # net.train()
            test_iter, raw_dict = self.convert_shm_to_tensor(-1)
            X, y = iter(test_iter).next()
            y_hat = net(X)
            y_hat[y_hat > 0.5] = 1
            y_hat[y_hat <= 0.5] = 0
            true_labels = y[0]
            pred_labels = src.get_custom_shm_labels(y_hat.detach().numpy(), 'A')
            titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
            src.show_shm_fig(X[0:45], titles[0:45])
            titles = [true + ':' + pred for true, pred in zip(true_labels, pred_labels)]
            self.generate_shm_report_xlsx(titles, raw_dict)

        else:
            # %% load state dict
            net.load_state_dict(torch.load('./stat_dict.pth'))
            # %% show result
            net.eval()
            test_iter, raw_dict = self.convert_shm_to_tensor(-1)
            # X, y = iter(test_iter).next()
            # y_hat = net(X)
            # y_hat[y_hat > 0.5] = 1
            # y_hat[y_hat <= 0.5] = 0
            # true_labels = y[0]
            channel_count_list = [32, 16, 8, 4]
            text = self.qLineEdit.text()
            channel_index = int(text)
            # Extract the layer
            conv_out = LayerActivations(list(net._modules.items()), channel_index)
            # [3:4] is to choose Index 4 shm
            img = next(iter(test_iter))[0][3:4]

            # imshow(img)
            o = net(img)
            conv_out.remove()  #
            act = conv_out.features  # act is the feature of current layer

            # 可视化 输出
            fig = plt.figure()
            # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
            total_count = channel_count_list[channel_index]
            for i in range(total_count):
                ax = fig.add_subplot(int(total_count / 4), 4, i + 1, xticks=[], yticks=[])
                ax.imshow(act[0][i].detach().numpy(), cmap="gray")

            plt.show()

    def convert_shm_to_tensor(self, batch_cnt):
        if sys.platform.startswith('win'):
            num_workers = 0  # 0
        else:
            num_workers = 4

        dataset = src.CsvDataset_Test('my_file.csv')
        if batch_cnt < 0:
            batch_size = dataset.__len__()  # len(self.result_dict)
        else:
            batch_size = batch_cnt
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


# Hook class, to extract output of each layer
class LayerActivations:
    features = None

    def __init__(self, model, layer_num, backward=False):
        if not backward:
            self.hook = model[layer_num][1].register_forward_hook(self.hook_fn)
        else:
            self.hook = model[layer_num][1].register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Application()
    viewer.show()
    sys.exit(app.exec_())
