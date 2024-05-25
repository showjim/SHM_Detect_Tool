# -*- coding: utf-8 -*-
import torch, xlsxwriter
import os, json, re, sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
from SHM_Detect_Tool import __version__
from Source import src_pytorch_public as src
from Source.CharDataCorrelation import (
    getKeyWordFromSettingFile,
    getDatalogInfo,
)


class Application():
    def __init__(self):
        self.result_dict = {}

    def read_shm_log(self, filename, config_details, send_log):
        """
        filename: Shmoo log file
        config_details: config details
        """

        self.result_dict = {}
        # Load config values
        keyword_site = config_details["keyword_site"]  # 'Site' #'DEVICE_NUMBER:' #'Site:' #'DEVICE_NUMBER:'
        keyword_item = config_details["keyword_item"]  # 'Test Name' #'Test Name' #'_SHM:' #'TestSuite = '
        keyword_start = config_details["keyword_start"]  # 'Tcoef(AC Spec)' #"Tcoef(AC Spec)" #'Tcoef(%)'
        keyword_end = config_details["keyword_end"]  # 'Tcoef(%)'
        keyword_pass = config_details["keyword_pass"]  # '\+'#'P|\*' #'\+'
        keyword_fail = config_details["keyword_fail"]  # '\-|E'#'\.|#' #'\-'
        keyword_y_axis_pos = config_details["keyword_y_axis_pos"]  # "right" #"left"

        new_shm_flag = False
        new_site_flag = False
        shm_start_flag = False
        shm_body_found_flag = False
        shm_end_flag = False
        try:
            with open(filename, 'r') as buffer:
                while True:
                    line = buffer.readline()
                    if keyword_item in line:  # line.startswith('FC_') and line.endswith('_SHM:\n'):
                        cur_instance = line[0:-1] + ":"
                        new_shm_flag = True
                        new_site_flag = False
                        continue
                    if keyword_site in line and new_shm_flag == True:  # ('Site:'):
                        res = re.search('\d+', line)
                        if res:
                            cur_site_index = res.group() + ',' * 100 #make this head larger than the body
                        else:
                            # assume all right side y_axis is from CHAR studio default output
                            # """
                            # Site           Pattern(s)          X Pin(s)       Slow Axis Value
                            # 0              top_AA_Scan_stuck_s *              N/A
                            #
                            #                   Y Axis: Tcoef(AC Spec)
                            #      +++++++++++  1.500
                            #      ++++++-++++  1.400
                            #      +++++++++++  1.300
                            #      +++++++++-+  1.200
                            #      ++++++++++-  1.100
                            #      +++++++----  1.000
                            #      -++++++++++  900.000 m
                            #      --+++++++++  800.000 m
                            #      ----++++++-  700.000 m
                            #      ------++-++  600.000 m
                            #      -----------  500.000 m
                            #      88899000000
                            #      04826000000
                            #      00000111111
                            #      ...........
                            #      00000000112
                            #      00000048260
                            #      00000000000
                            #      mmmmm
                            #      X Axis: Vcoef(DC Spec)
                            # """
                            if keyword_y_axis_pos == "right":
                                line = buffer.readline()
                                res = re.search('\d+', line)
                            if res:
                                cur_site_index = res.group() + ',' * 100 #make this head larger than the body
                            else:
                                cur_site_index = '' + ',' * 100 #make this head larger than the body
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
                            self.result_dict[cur_instance + cur_site_index].append(tmp)  # [1:])

                    if len(line) == 0:
                        break
        except Exception as e:
            send_log('Error: ' + e.__str__())

        convt_shm_csv = filename + '_tmp_file.csv'
        with open(convt_shm_csv, 'w') as f:
            for key, values in self.result_dict.items():
                f.write('{0}\n'.format(key))
                for val in values:
                    f.write(','.join(i for i in val))
                    f.write('\n')
        return convt_shm_csv

    def read_shmoo_csv(self, csv_file):
        tmpX = []
        tmpY = []
        tmpZ = []
        X = []
        Y = []
        Z = {}
        self.csv_df = pd.read_csv(csv_file, header=None, engine="python")
        for index, row in self.csv_df.iterrows():
            if ':' in row[0]:
                if len(tmpX) > 0:
                    X.append(tmpX)
                    Y.append(tmpY)
                    Z[tmpY] = tmpZ
                tmpY = row[0]  # self.csv_df.iloc[0].dropna().to_list()
                tmpX = []
                tmpZ = []
            elif not ("P" in row.dropna().to_list()) and not ("." in row.dropna().to_list()):
                # skip
                tmpZ.append(row.dropna().to_list())
                continue
            else:
                tmpX.append(row.dropna().to_list()[1:])
                tmpZ.append(row.dropna().to_list())
        X.append(tmpX)
        Y.append(tmpY)
        Z[tmpY] = tmpZ
        return X, Y, Z

    def generate_shm_report_xlsx(self, titles, shms, filename, send_log):
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
            worksheet.write_row(0, 0, ['Instance', '', '', '', '', 'Site Index', 'Result Symbol', 'Result'])
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
            col = len(shms[shm][i - 1])
            worksheet.conditional_format(0, 0, row, col,
                                         {'type': 'cell', 'criteria': 'equal to',
                                          'value': '"."', 'format': format_2XXX})
            worksheet.conditional_format(0, 0, row, col,
                                         {'type': 'cell', 'criteria': 'equal to',
                                          'value': '"P"', 'format': format_7XXX})
            workbook.close()
            send_log('Xlsx file is written!')

        except xlsxwriter.exceptions.FileCreateError:  # PermissionError:
            send_log("Please close " + report_name.split('/')[-1])
        return report_name

    def convert_shm_to_tensor(self, batch_cnt, shmoo_body=[], shmoo_title=[], mode='P'):
        if sys.platform.startswith('win'):
            num_workers = 0  # 0
        else:
            num_workers = 0  # 4
        if mode == 'P':
            dataset = src.CsvDataset_Test(self.filename + '_tmp_file.csv')  # 'my_file.csv')
        else:
            dataset = src.CsvDataset_Test_Serial(shmoo_body, shmoo_title)
        if batch_cnt < 0:
            batch_size = dataset.__len__()  # len(self.result_dict)
        else:
            batch_size = batch_cnt
        test_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return test_iter, dataset.raw_dict

    def cnn_net(self, shm_csv_log_path, send_log, mode='test'):
        # Check that MPS is available
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        send_log(f"Using device: {device}")

        # net = d2l.LeNet()
        net = src.AlexNet()
        # net.to(device)

        # print parameters count
        pytorch_total_params = sum(p.numel() for p in net.parameters())
        send_log('neural network architecture has ' + str(pytorch_total_params) + ' parameters.')
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        send_log('neural network architecture has ' + str(pytorch_total_params) + ' trainable parameters.')

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
            loss = torch.nn.MultiLabelSoftMarginLoss()  # MultiLabelSoftMarginLoss()  # BCEWithLogitsLoss()  # BCELoss() #MultiLabelSoftMarginLoss() #BCELoss()

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
            X, y = next(iter(test_iter))  # .__next__()# .next()
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
            filename = shm_csv_log_path
            titles = []
            titles_plot = []
            # raw_dict = {}
            shmoo_body, shmoo_title, shmoo_dict = self.read_shmoo_csv(filename)
            # _, figs = plt.subplots(5, 10, figsize=(12, 8))
            # plt.tight_layout()
            # figs = figs.flatten()
            for i in range(len(shmoo_title)):
                test_iter, tmp_raw_dict = self.convert_shm_to_tensor(-1, shmoo_body[i], shmoo_title[i], 'S')
                # test_iter, raw_dict = self.convert_shm_to_tensor(-1, shmoo_body[0], shmoo_title[0], 'P')
                X, y = next(iter(test_iter))  # .next()
                y_hat = net(X)
                y_hat = src.reformat_output(y_hat)
                y_hat[y_hat >= 0.5] = 1
                y_hat[y_hat < 0.5] = 0
                true_labels = y[0]
                pred_labels = src.get_custom_shm_labels(y_hat.detach().numpy(), 'A')
                #     titles_plot.append(true_labels + '\n' + pred_labels[0]) # = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
                #     # src.show_shm_fig(X[0:50], titles[0:50])
                titles.append(true_labels + ':' + pred_labels[
                    0])  # ([true + ':' + pred for true, pred in zip(true_labels, pred_labels)])
            #     raw_dict.update(tmp_raw_dict)
            #     if i < 50:
            #         figs[i].imshow(X[0].view((X[0].shape[1], X[0].shape[2])).numpy(), cmap='RdYlGn')
            #         figs[i].set_title(titles_plot)
            #         figs[i].axes.get_xaxis().set_visible(False)
            #         figs[i].axes.get_yaxis().set_visible(False)
            report_name = self.generate_shm_report_xlsx(titles, shmoo_dict, filename, send_log)
            return report_name
            # plt.show()

        else:
            """Evaluation part, ignored in webapp"""


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False) or st.secrets["password"] == "":
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

def main(app=Application()):
    st.title(f"{__version__}")
    st.caption('Powered by Streamlit, written by Chao Zhou')
    st.subheader("", divider='rainbow')

    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    work_path = os.path.abspath('.')
    WorkPath = os.path.join(work_path, "workDir")
    if not os.path.exists(WorkPath):  # check the directory is existed or not
        os.mkdir(WorkPath)
    # OutputPath = os.path.join(work_path, "Output")
    # if not os.path.exists(OutputPath):  # check the directory is existed or not
    #     os.mkdir(OutputPath)

    if "FilePath" not in st.session_state:
        st.session_state["FilePath"] = ""
    if "JsonConfig" not in st.session_state:
        st.session_state["JsonConfig"] = ""
    if "shm_analyse_result" not in st.session_state:
        st.session_state["shm_analyse_result"] = ""
    if "shm_detect_logprint" not in st.session_state:
        st.session_state["shm_detect_logprint"] = ""
    if "shm_corr_result" not in st.session_state:
        st.session_state["shm_corr_result"] = ""

    # Sidebar for menu options
    with st.sidebar:
        st.header("Other Tools")
        st.page_link("http://taishanstone:8501", label="Check INFO Tool", icon="1️⃣")
        st.page_link("http://taishanstone:8502", label="Pattern Auto Edit Tool", icon="2️⃣")
        st.header("Help")
        if st.button("About"):
            st.info(
                "Thank you for using!\nCreated by Chao Zhou.\nAny suggestions please mail zhouchao486@gmail.com]")

    # Main UI Components
    st.subheader('Step 1. Upload Config Setting')
    json_file = st.file_uploader("Upload JSON", type=["json"], )
    if json_file:
        # Load config values
        config_details = json.load(json_file)
        st.session_state["JsonConfig"] = config_details
        st.json(config_details)

    st.subheader('Step 2. Pre-process Shmoo to CSV format')
    file_path = st.file_uploader("Upload Shmoo Log", type=["txt"], )
    if st.button("Upload Shmoo Log"):
        if file_path is not None:
            # save file
            with st.spinner('Reading file'):
                uploaded_path = os.path.join(WorkPath, file_path.name)
                with open(uploaded_path, mode="wb") as f:
                    f.write(file_path.getbuffer())
                if os.path.exists(uploaded_path) == True:
                    st.session_state["FilePath"] = uploaded_path
                    st.write(f"✅ {Path(uploaded_path).name} uploaed")

    with st.expander("Run Logs"):
        log_text_area = st.empty()  # text_area("", key="logs", height=300)

    def send_log(data_log):
        st.session_state["shm_detect_logprint"] += f'{datetime.now()} - {data_log}\n'
        log_text_area.code(st.session_state["shm_detect_logprint"])

    st.subheader('Step 3. Run to analyse CHAR log', divider="violet")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### Step 3A. Analyse Shmoo result')
        # if st.button('Convert Shmoo log to CSV'):
        #     # """Convert Shmoo log to CSV"""
        #     convt_csv_shm_file = app.read_shm_log(st.session_state["FilePath"], st.session_state["JsonConfig"], send_log)
        #     st.session_state["csv_shm_file"] = convt_csv_shm_file
        #     send_log(f"Convert Shmoo log to CSV format completed.")

        if st.button('Analyse Shmoo Result'):
            # """Convert Shmoo log to CSV"""
            convt_csv_shm_file = app.read_shm_log(st.session_state["FilePath"], st.session_state["JsonConfig"],
                                                  send_log)
            st.session_state["csv_shm_file"] = convt_csv_shm_file
            send_log(f"Convert Shmoo log to CSV format completed.")

            # """run analyse Shmoo log action"""
            report_name = app.cnn_net(st.session_state["csv_shm_file"], send_log, "test")
            st.session_state["shm_analyse_result"] = report_name
            send_log(f"Finish analysis!")

        if len(st.session_state["shm_analyse_result"]) > 0:
            result_file_path = st.session_state["shm_analyse_result"]
            result_file_name = os.path.basename(result_file_path)
            with open(result_file_path, "rb") as file:
                btn = st.download_button(
                    label="Download Result XLSX File",
                    data=file,
                    file_name=result_file_name,
                    mime="application/octet-stream"
                )

    with col2:
        st.markdown('#### Step 3B. Compare CHAR log')
        site_lbl = st.text_input("Specify the site to process for each file", placeholder="Like 0,1;0,2; Or leave blank to process all sites...")
        if st.button('Generate CHAR correlation report'):
            file_paths = st.session_state["FilePath"]
            config_details = st.session_state["JsonConfig"]
            TER_keyword = getKeyWordFromSettingFile(config_details)
            char_corr_report_name = getDatalogInfo(TER_keyword, file_paths, site_lbl)
            st.session_state["shm_corr_result"] = char_corr_report_name

        if len(st.session_state["shm_corr_result"]) > 0:
            corr_result_file_path = st.session_state["shm_corr_result"]
            corr_result_file_name = os.path.basename(corr_result_file_path)
            with open(corr_result_file_path, "rb") as file:
                char_corr_btn = st.download_button(
                    label="Download CHAR Corr XLSX File",
                    data=file,
                    file_name=corr_result_file_name,
                    mime="application/octet-stream"
                )


# Run the main function
if __name__ == "__main__":
    app = Application()
    main(app)
