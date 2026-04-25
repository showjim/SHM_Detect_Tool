# -*- coding: utf-8 -*-
import hmac

import os, json, re, sys
from pathlib import Path
from datetime import datetime
import streamlit as st
from SHM_Detect_Tool import __version__
import shm_backend
from Source.CharDataCorrelation import (
    getKeyWordFromSettingFile,
    getDatalogInfo,
)


class Application():
    def __init__(self):
        self.sites_in_log_list = []

    def read_shm_log(self, filename, config_details, send_log):
        """Parse shmoo log → intermediate CSV. Delegates to backend."""
        csv_path, sites_list = shm_backend.read_shm_log(
            filename, config_details, send_log)
        self.sites_in_log_list = sites_list
        return csv_path

    def cnn_net(self, shm_csv_log_path, send_log, mode='test',
                isParallelPlot='Disable', model_path='./state_dict.pth'):
        """Run shmoo analysis. Delegates to backend for inference + report."""
        if mode == 'test':
            # Read CSV, run inference, generate report
            shmoo_body, shmoo_title, shmoo_dict = shm_backend.read_shmoo_csv(
                shm_csv_log_path)

            import torch
            from Source import src_pytorch_public as src
            net = src.AlexNet()
            net.load_state_dict(torch.load(model_path, weights_only=True))
            net.eval()

            pytorch_total_params = sum(p.numel() for p in net.parameters())
            send_log(f'neural network architecture has {pytorch_total_params} parameters.')
            pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            send_log(f'neural network architecture has {pytorch_total_params} trainable parameters.')

            titles = []
            for i in range(len(shmoo_title)):
                test_iter, _ = shm_backend.convert_shm_to_tensor(
                    shm_csv_log_path, -1, shmoo_body[i], shmoo_title[i], 'S')
                X, y = next(iter(test_iter))
                y_hat = net(X)
                y_hat = src.reformat_output(y_hat)
                y_hat[y_hat >= 0.5] = 1
                y_hat[y_hat < 0.5] = 0
                true_labels = y[0]
                pred_labels = src.get_custom_shm_labels(
                    y_hat.detach().numpy(), 'A')
                titles.append(true_labels + ':' + pred_labels[0])

            report_name = shm_backend.generate_shm_report_xlsx(
                titles, shmoo_dict, shm_csv_log_path, isParallelPlot,
                self.sites_in_log_list, send_log)
            return report_name

        elif mode == 'training':
            net, test_iter = shm_backend.train_model(logger=send_log)
            send_log("Training complete.")


def main(app):
    st.set_page_config(page_title="SHM Detect Tool",  # page_icon=":material/robot_2:",
                       layout='wide', initial_sidebar_state='auto')
    st.title('SHM Detect Tool Webapp V' + __version__)

    # Session state initialization
    if "shm_detect_logprint" not in st.session_state:
        st.session_state["shm_detect_logprint"] = ''
    if "shm_analyse_result" not in st.session_state:
        st.session_state["shm_analyse_result"] = ''
    if "shm_corr_result" not in st.session_state:
        st.session_state["shm_corr_result"] = ''
    if "JsonConfig" not in st.session_state:
        st.session_state["JsonConfig"] = {}
    if "FilePaths" not in st.session_state:
        st.session_state["FilePaths"] = []

    st.subheader('Step 1. Upload Config file', divider="violet")
    uploaed_config = st.file_uploader("Upload config JSON file", type="json",
                                       accept_multiple_files=False,
                                       key="config_uploader")
    if uploaed_config:
        config_details = json.load(uploaed_config)
        st.session_state["JsonConfig"] = config_details
        st.write(f"✅ Config loaded: {uploaed_config.name}")
        st.json(config_details)

    st.subheader('Step 2. Upload Shmoo log file(s)', divider="violet")
    uploaded_files = st.file_uploader("Upload Shmoo log file(s)", type=["txt", "log"],
                                      accept_multiple_files=True,
                                      key="file_uploader")
    if uploaded_files:
        uploaded_paths = []
        for uploaded_file in uploaded_files:
            # Save uploaded file to local temp path
            uploaded_path = os.path.join(os.getcwd(), uploaded_file.name)
            with open(uploaded_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            uploaded_paths.append(uploaded_path)
        if uploaded_paths:
            st.session_state.FilePaths = uploaded_paths
            for p in uploaded_paths:
                st.write(f"✅ {Path(p).name} uploaded")

    with st.expander("Run Logs"):
        log_text_area = st.empty()

    def send_log(data_log):
        st.session_state["shm_detect_logprint"] += f'{datetime.now()} - {data_log}\n'
        log_text_area.code(st.session_state["shm_detect_logprint"])

    st.subheader('Step 3. Run to analyse CHAR log', divider="violet")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### Step 3A. Analyse Shmoo result')
        isParallelPlot = st.select_slider(label="`Enable Parallel Plot`",
                                         options=["Disable", "15", "25"],
                                         value="Disable"
                                         )
        if st.button('Analyse Shmoo Result'):
            # """Convert Shmoo log to CSV"""
            convt_csv_shm_file = app.read_shm_log(st.session_state["FilePaths"][-1], st.session_state["JsonConfig"],
                                                  send_log)
            st.session_state["csv_shm_file"] = convt_csv_shm_file
            send_log(f"Convert Shmoo log to CSV format completed.")

            # """run analyse Shmoo log action"""
            report_name = app.cnn_net(st.session_state["csv_shm_file"], send_log, "test", isParallelPlot)
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
        site_lbl = st.text_input("`Specify the site to process for each file`",
                                 placeholder="Like 0,1;0,2; Or leave blank to process all sites...")
        interval_columns = st.text_input("`Specify the gap between sites`", placeholder="25", value="25")
        if st.button('Generate CHAR correlation report'):
            file_paths = ";" .join(st.session_state.FilePaths)
            config_details = st.session_state.JsonConfig
            TER_keyword = getKeyWordFromSettingFile(config_details)
            char_corr_report_name = getDatalogInfo(TER_keyword, file_paths, site_lbl, int(interval_columns))
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
