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


def main(app):
    st.set_page_config(page_title="SHM Detect Tool",  # page_icon=":material/robot_2:",
                       layout='wide', initial_sidebar_state='auto')
    st.title(f'{__version__}')
    st.caption('Powered by Streamlit, written by Chao Zhou')
    st.subheader("", divider='rainbow')

    with st.expander("Disclaimer", True):
        st.warning("""The developer of this efficiency tool has taken all reasonable measures to ensure its quality and functionality. However, it is provided "as is" and the developer makes no representations or warranties of any kind, express or implied, as to its accuracy, reliability, or suitability for a particular purpose.

The user assumes all risks associated with the use of this tool, and the developer will not be liable for any damages, including but not limited to direct, indirect, special, incidental, or consequential damages, arising out of the use or inability to use this tool.

The developer welcomes feedback and bug reports from users. If you encounter any issues or have any suggestions, please contact me at Teams. Your input will help us improve the tool and provide a better user experience.

By using this tool, you acknowledge that you have read and understood this disclaimer and agree to be bound by its terms.""",
                   icon="⚠️")

    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    work_path = os.path.abspath('.')
    WorkPath = os.path.join(work_path, "workDir")
    if not os.path.exists(WorkPath):
        os.mkdir(WorkPath)

    # Sidebar for menu options
    with st.sidebar:
        st.header("Other Tools")
        st.page_link("http://taishanstone:8501", label="Check INFO Tool", icon="1️⃣")
        st.page_link("http://taishanstone:8502", label="Pattern Auto Edit Tool", icon="2️⃣")
        st.header("Help")
        if st.button("About"):
            st.info(
                "Thank you for using!\nCreated by Chao Zhou.\nAny suggestions please mail zhouchao486@gmail.com]")

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

    # Main UI Components
    st.subheader('Step 1. Upload Config Setting')
    json_file = st.file_uploader("Upload JSON", type=["json"], )
    if json_file:
        # Load config values
        config_details = json.load(json_file)
        st.session_state["JsonConfig"] = config_details
        st.json(config_details)

    st.subheader('Step 2. Pre-process Shmoo to CSV format')
    file_paths = st.file_uploader("Upload Shmoo Log", type=["txt"], accept_multiple_files=True)
    if st.button("Upload Shmoo Log"):
        if file_paths is not None and len(file_paths) > 0:
            # save file
            with st.spinner('Reading file'):
                uploaded_paths = []
                for file_path in file_paths:
                    uploaded_path = os.path.join(WorkPath, file_path.name)
                    uploaded_paths.append(uploaded_path)
                    with open(uploaded_path, mode="wb") as f:
                        f.write(file_path.getbuffer())
                if os.path.exists(uploaded_path) == True:
                    st.session_state.FilePaths = uploaded_paths
                    st.write(f"✅ {Path(uploaded_path).name} uploaded")

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
