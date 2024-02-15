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
import xlsxwriter
import matplotlib.pyplot as plt

sys.path.append("..")
import src_pytorch_public as src
# import SHM_keywords_setting as setting
from PyQt5.QtWidgets import *
import qtawesome as qta
import pandas as pd
import numpy as np

__version__ = 'SHM Detect Tool Beta V0.7.1'
__author__ = 'zhouchao486@gmail.com'


class Application(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUI()
        self.result_dict = {}
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

        self.result_dict = {}
        # Load config values
        with open(r'SHM_keywords_setting.json') as config_file:
            config_details = json.load(config_file)
        keyword_site = config_details["keyword_site"] #'Site' #'DEVICE_NUMBER:' #'Site:' #'DEVICE_NUMBER:'
        keyword_item = config_details["keyword_item"] #'Test Name' #'Test Name' #'_SHM:' #'TestSuite = '
        keyword_start = config_details["keyword_start"] #'Tcoef(AC Spec)' #"Tcoef(AC Spec)" #'Tcoef(%)'
        keyword_end = config_details["keyword_end"] #'Tcoef(%)'
        keyword_pass = config_details["keyword_pass"] #'\+'#'P|\*' #'\+'
        keyword_fail = config_details["keyword_fail"] #'\-|E'#'\.|#' #'\-'
        keyword_y_axis_pos = config_details["keyword_y_axis_pos"] #"right" #"left"

        # # py setting file, cannot modify setting realtime
        # import SHM_keywords_setting as setting
        # keyword_site = setting.keyword_site # 'Site' #'DEVICE_NUMBER:' #'Site:' #'DEVICE_NUMBER:'
        # keyword_item = setting.keyword_item  # 'Test Name' #'Test Name' #'_SHM:' #'TestSuite = '
        # keyword_start = setting.keyword_start  # 'Tcoef(AC Spec)' #"Tcoef(AC Spec)" #'Tcoef(%)'
        # keyword_end = setting.keyword_end  # 'Tcoef(%)'
        # keyword_pass = setting.keyword_pass  # '\+'#'P|\*' #'\+'
        # keyword_fail = setting.keyword_fail  # '\-|E'#'\.|#' #'\-'
        # keyword_y_axis_pos = setting.keyword_y_axis_pos  # "right" #"left"

        new_shm_flag = False
        new_site_flag = False
        shm_start_flag = False
        shm_body_found_flag = False
        shm_end_flag = False
        try:
            with open(filename, 'r') as buffer:
                while True:
                    line = buffer.readline()
                    if keyword_item in line:#line.startswith('FC_') and line.endswith('_SHM:\n'):
                        cur_instance = line[0:-1] + ":"
                        new_shm_flag = True
                        new_site_flag = False
                        continue
                    if keyword_site in line and new_shm_flag == True:#('Site:'):
                        res = re.search('\d+', line)
                        if res:
                            cur_site_index = res.group() + ',' * 12
                        else:
                            # assume all right side y_axis is from CHAR studio default output
                            """
                            Site           Pattern(s)          X Pin(s)       Slow Axis Value     
                            0              top_AA_Scan_stuck_s *              N/A                 
                                 
                                              Y Axis: Tcoef(AC Spec)
                                 +++++++++++  1.500  
                                 ++++++-++++  1.400  
                                 +++++++++++  1.300  
                                 +++++++++-+  1.200  
                                 ++++++++++-  1.100  
                                 +++++++----  1.000  
                                 -++++++++++  900.000 m
                                 --+++++++++  800.000 m
                                 ----++++++-  700.000 m
                                 ------++-++  600.000 m
                                 -----------  500.000 m
                                 88899000000
                                 04826000000
                                 00000111111
                                 ...........
                                 00000000112
                                 00000048260
                                 00000000000
                                 mmmmm      
                                 X Axis: Vcoef(DC Spec)
                            """
                            if keyword_y_axis_pos == "right":
                                line = buffer.readline()
                                res = re.search('\d+', line)
                            if res:
                                cur_site_index = res.group() + ',' * 12
                            else:
                                cur_site_index = '' + ',' * 12
                                print("Warning: no site index found!")
                        new_site_flag = True
                        continue
                    if keyword_start in line and new_shm_flag == True and new_site_flag == True:
                        shm_start_flag = True
                        self.result_dict[cur_instance + cur_site_index] = []
                        continue

                    if new_shm_flag and new_site_flag and shm_start_flag:
                        res = re.search('(\s*([P*.#+\-]))+', line)
                        res_axis = re.findall('\d+\.\d+', line)
                        if (res is not None) and shm_body_found_flag == False:
                            shm_body_found_flag = True
                        elif (res is not None) and (res_axis is not None):
                            if len(res_axis) > 1:
                                new_shm_flag = False
                                new_site_flag = False
                                shm_start_flag = False
                                shm_body_found_flag = False
                                # need add Y-axis here
                                x_list = line.split()
                                self.result_dict[cur_instance + cur_site_index].append([keyword_start] + x_list)
                        elif (res is None) and shm_body_found_flag:
                            new_shm_flag = False
                            new_site_flag = False
                            shm_start_flag = False
                            shm_body_found_flag = False
                            # need add X-axis here
                            x_list = line.split()
                            self.result_dict[cur_instance + cur_site_index].append([keyword_start] + x_list)

                        if shm_body_found_flag:
                            tmp = res.string.split()
                            if keyword_y_axis_pos == "right":
                                if (re.search(keyword_pass, tmp[0]) is not None) or (
                                        re.search(keyword_fail, tmp[0]) is not None):
                                    tmp[0] = re.sub(keyword_pass, "P", tmp[0])
                                    tmp[0] = re.sub(keyword_fail, ".", tmp[0])
                                    if len(tmp) > 1:
                                        if len(tmp[0]) > 1:
                                            tmp = [''.join(tmp[1:])] + list(tmp[0])
                            else:
                                if (re.search(keyword_pass, tmp[-1]) is not None) or (
                                        re.search(keyword_fail, tmp[-1]) is not None):
                                    tmp[-1] = re.sub(keyword_pass, "P", tmp[-1])
                                    tmp[-1] = re.sub(keyword_fail, ".", tmp[-1])
                                    if len(tmp) > 1:
                                        if len(tmp[-1]) > 1:
                                            tmp = [''.join(tmp[0:-1])] + list(tmp[-1])
                            self.result_dict[cur_instance + cur_site_index].append(tmp)#[1:])

                    if len(line) == 0:
                        break
        except Exception as e:
            QMessageBox.information(
                self, 'Error', e.__str__(),
                QMessageBox.Ok)

        with open(self.filename + '_tmp_file.csv', 'w') as f:
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

        # Check that MPS is available
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        # net = d2l.LeNet()
        net = src.AlexNet()
        # net.to(device)

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
            loss = torch.nn.MultiLabelSoftMarginLoss() #MultiLabelSoftMarginLoss()  # BCEWithLogitsLoss()  # BCELoss() #MultiLabelSoftMarginLoss() #BCELoss()

            # %% optimise function
            lr = 0.0014
            # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0004)

            # %% run training
            num_epochs = 60  # 320
            src.train_network(net, train_iter, test_iter, loss, num_epochs, batch_size, None, lr, optimizer)

            # %% save the state
            torch.save(net.state_dict(), './state_dict.pth')

            # %% show result
            net.eval()
            # print(net.training)
            X, y = next(iter(test_iter)) #.__next__()# .next()
            # X = X.to(device)
            true_labels = src.get_custom_shm_labels(y.numpy(), 'E')  # d2l.get_fashion_mnist_labels(y.numpy())
            y_hat = net(X)
            y_hat = src.reformat_output(y_hat)
            # y_hat[y_hat >= 0.5] = 1
            # y_hat[y_hat < 0.5] = 0
            pred_labels = src.get_custom_shm_labels(
                y_hat.detach().numpy(),
                'A')  # d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy()) net(X).detach().numpy()
            titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
            src.show_shm_fig(X[0:100], titles[0:100])

        elif mode == 'test':
            # %% load state dict
            net.load_state_dict(torch.load('./state_dict.pth'))
            # %% show result
            net.eval()
            # net.train()
            filename = self.analyse_shm_path_let.text()#self.filename + '_tmp_file.csv'
            titles = []
            titles_plot = []
            raw_dict = {}
            shmoo_body, shmoo_title, shmoo_dict = self.read_shmoo_csv(filename) # #'my_file.csv')
            _, figs = plt.subplots(5, 10, figsize=(12, 8))
            plt.tight_layout()
            figs = figs.flatten()
            for i in range(len(shmoo_title)):
                test_iter, tmp_raw_dict = self.convert_shm_to_tensor(-1, shmoo_body[i], shmoo_title[i], 'S')
                # test_iter, raw_dict = self.convert_shm_to_tensor(-1, shmoo_body[0], shmoo_title[0], 'P')
                X, y = next(iter(test_iter)) #.next()
                y_hat = net(X)
                y_hat = src.reformat_output(y_hat)
                # y_hat[y_hat >= 0.5] = 1
                # y_hat[y_hat < 0.5] = 0
                true_labels = y[0]
                pred_labels = src.get_custom_shm_labels(y_hat.detach().numpy(), 'A')
                titles_plot.append(true_labels + '\n' + pred_labels[0]) # = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
                # src.show_shm_fig(X[0:50], titles[0:50])
                titles.append(true_labels + ':' + pred_labels[0]) #([true + ':' + pred for true, pred in zip(true_labels, pred_labels)])
                raw_dict.update(tmp_raw_dict)
                if i < 50:
                    figs[i].imshow(X[0].view((X[0].shape[1], X[0].shape[2])).numpy(), cmap='RdYlGn')
                    figs[i].set_title(titles_plot)
                    figs[i].axes.get_xaxis().set_visible(False)
                    figs[i].axes.get_yaxis().set_visible(False)
            self.generate_shm_report_xlsx(titles, shmoo_dict, filename)
            plt.show()

        else:
            # %% load state dict
            net.load_state_dict(torch.load('./state_dict.pth'))
            # %% show result
            net.eval()

            i = 5
            filename = self.analyse_shm_path_let.text()
            shmoo_body, shmoo_title, shmoo_dict = self.read_shmoo_csv(filename)#'my_file.csv')
            # test_iter, raw_dict = self.convert_shm_to_tensor(-1, mode='P')
            test_iter, raw_dict = self.convert_shm_to_tensor(-1, shmoo_body[i], shmoo_title[i], 'S')
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
            # img = next(iter(test_iter))[0][0:1]
            img, y = next(iter(test_iter))# next()

            # imshow(img)
            o = net(img)
            conv_out.remove()  #
            act = conv_out.features  # act is the feature of current layer

            # Visualization output
            fig = plt.figure()
            # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
            total_count = channel_count_list[channel_index]
            for i in range(total_count):
                ax = fig.add_subplot(int(total_count / 4), 4, i + 1, xticks=[], yticks=[])
                ax.imshow(act[0][i].detach().numpy(), cmap="gray")

            plt.show()

    def convert_shm_to_tensor(self, batch_cnt, shmoo_body=[], shmoo_title=[], mode='P'):
        if sys.platform.startswith('win'):
            num_workers = 0  # 0
        else:
            num_workers = 0 #4
        if mode == 'P':
            dataset = src.CsvDataset_Test(self.filename + '_tmp_file.csv')#'my_file.csv')
        else:
            dataset = src.CsvDataset_Test_Serial(shmoo_body, shmoo_title)
        if batch_cnt < 0:
            batch_size = dataset.__len__()  # len(self.result_dict)
        else:
            batch_size = batch_cnt
        test_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return test_iter, dataset.raw_dict

    def read_shmoo_csv(self, csv_file):
        tmpX = []
        tmpY = []
        tmpZ = []
        X = []
        Y = []
        Z = {}
        self.csv_df = pd.read_csv(csv_file, header=None)
        for index, row in self.csv_df.iterrows():
            if ':' in row[0]:
                if len(tmpX) > 0:
                    X.append(tmpX)
                    Y.append(tmpY)
                    Z[tmpY] = tmpZ
                tmpY = row[0] #self.csv_df.iloc[0].dropna().to_list()
                tmpX = []
                tmpZ = []
            elif not("P" in row.dropna().to_list()) and not("." in row.dropna().to_list()):
                #skip
                tmpZ.append(row.dropna().to_list())
                continue
            else:
                tmpX.append(row.dropna().to_list()[1:])
                tmpZ.append(row.dropna().to_list())
        X.append(tmpX)
        Y.append(tmpY)
        Z[tmpY] = tmpZ
        return X, Y, Z

    def generate_shm_report_xlsx(self, titles, shms, filename):
        report_name = filename + '_report.xlsx'
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
            worksheet.write_row(0, 0, ['Instance', '', '', '','', 'Site Index', 'Result Symbol', 'Result'])
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
            col = len(shms[shm][i-1])
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
