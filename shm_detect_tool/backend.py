# -*- coding: utf-8 -*-
"""
SHM Detect Tool — Shared Backend Module

Consolidates deep learning, log parsing, and report generation logic
that was previously duplicated between the GUI and webapp interfaces.
All three interfaces (GUI, webapp, CLI) delegate to this module.
"""

import os
import re
import sys
import json
import torch
import xlsxwriter
import pandas as pd
from typing import List, Tuple, Optional, Callable

from shm_detect_tool.Source import src_pytorch_public as src
from shm_detect_tool.Source.CharDataCorrelation import (
    getKeyWordFromSettingFile,
    getDatalogInfo,
)


# ---------------------------------------------------------------------------
# Log Parsing
# ---------------------------------------------------------------------------

def get_all_site_nums(each_file: str, site_keyword: str) -> List[str]:
    """Extract all unique site numbers from a shmoo log file.

    Args:
        each_file: Path to the shmoo log file.
        site_keyword: Keyword that precedes site numbers (e.g. 'Site: ').

    Returns:
        List of site number strings, e.g. ['0', '1', '2'].
    """
    site_info = []
    with open(each_file, 'r', encoding='utf-8') as f:
        for each_line in f.readlines():
            if site_keyword != "" and site_keyword in each_line:
                # Extract number after keyword position (robust for all formats)
                idx = each_line.find(site_keyword)
                after = each_line[idx + len(site_keyword):]
                match = re.search(r'\d+', after)
                if match:
                    site_info.append(int(match.group()))

    unique_site_info = list(set(site_info))
    str_site = ''
    for x in unique_site_info:
        if str_site == '':
            str_site = str(x)
        else:
            str_site = str_site + ',' + str(x)

    return str_site.split(',')


def read_shm_log(filename: str, config: dict,
                 logger: Callable = print) -> Tuple[str, List[str]]:
    """Parse a shmoo log (.txt) with config (.json) → intermediate CSV.

    Args:
        filename: Path to the shmoo log file.
        config: Parsed JSON config dictionary.
        logger: Logging function (default: print).

    Returns:
        Tuple of (csv_path, sites_list).

    Raises:
        Exception on parse errors (caller handles UI).
    """
    result_dict = {}

    # Load config values
    keyword_site = config["keyword_site"]
    keyword_item = config["keyword_item"]
    keyword_start = config["keyword_start"]
    keyword_end = config["keyword_end"]
    keyword_pass = config["keyword_pass"]
    keyword_fail = config["keyword_fail"]
    keyword_y_axis_pos = config["keyword_y_axis_pos"]

    new_shm_flag = False
    new_site_flag = False
    shm_start_flag = False
    shm_body_found_flag = False
    cur_instance = ""
    cur_site_index = ""

    with open(filename, 'r') as buffer:
        while True:
            line = buffer.readline()
            if keyword_item in line:
                cur_instance = line[0:-1] + ":"
                new_shm_flag = True
                new_site_flag = False
                continue
            if keyword_site in line and new_shm_flag == True:
                res = re.search(r'\d+', line)
                if res:
                    cur_site_index = res.group() + ',' * 100
                else:
                    if keyword_y_axis_pos == "right":
                        line = buffer.readline()
                        res = re.search(r'\d+', line)
                    if res:
                        cur_site_index = res.group() + ',' * 100
                    else:
                        cur_site_index = '' + ',' * 100
                        logger("Warning: no site index found!")
                new_site_flag = True
                continue
            if keyword_start in line and new_shm_flag == True and new_site_flag == True:
                shm_start_flag = True
                result_dict[cur_instance + cur_site_index] = []
                continue

            if new_shm_flag and new_site_flag and shm_start_flag:
                res = re.search(r'(\s*([P*.#+\-]))+', line)
                res_axis = re.findall(r'\d+\.\d+', line)
                if (res is not None) and shm_body_found_flag == False:
                    shm_body_found_flag = True
                elif (res is not None) and len(res_axis) > 0:
                    if len(res_axis) > 1:
                        new_shm_flag = False
                        new_site_flag = False
                        shm_start_flag = False
                        shm_body_found_flag = False
                        # Left Y-axis: X-axis values on a single horizontal line
                        x_list = line.split()
                        result_dict[cur_instance + cur_site_index].append(
                            [keyword_start] + x_list)
                elif (res is None) and shm_body_found_flag:
                    new_shm_flag = False
                    new_site_flag = False
                    shm_start_flag = False
                    shm_body_found_flag = False
                    if keyword_y_axis_pos == "right":
                        # Vertical X-axis: collect all digit/dot/unit rows and transpose
                        axis_rows = [line.strip()]
                        while True:
                            next_line = buffer.readline()
                            if len(next_line) == 0:
                                break
                            stripped = next_line.strip()
                            if not stripped or 'Axis' in stripped:
                                break
                            axis_rows.append(stripped)
                        # Transpose columns to build X-axis values
                        num_cols = max(len(r) for r in axis_rows)
                        x_values = []
                        for col_idx in range(num_cols):
                            value = ''
                            for row in axis_rows:
                                if col_idx < len(row) and row[col_idx] != ' ':
                                    value += row[col_idx]
                            x_values.append(value)
                        result_dict[cur_instance + cur_site_index].append(
                            [keyword_start] + x_values)
                    else:
                        x_list = line.split()
                        result_dict[cur_instance + cur_site_index].append(
                            [keyword_start] + x_list)

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
                    result_dict[cur_instance + cur_site_index].append(tmp)

            if len(line) == 0:
                break

    # Write intermediate CSV
    csv_path = filename + '_tmp_file.csv'
    with open(csv_path, 'w') as f:
        for key, values in result_dict.items():
            f.write('{0}\n'.format(key))
            for val in values:
                f.write(','.join(i for i in val))
                f.write('\n')

    # Extract site list
    sites_list = get_all_site_nums(filename, keyword_site)

    return csv_path, sites_list


# ---------------------------------------------------------------------------
# CSV Parsing
# ---------------------------------------------------------------------------

def read_shmoo_csv(csv_file: str) -> Tuple[list, list, dict]:
    """Parse an intermediate shmoo CSV file.

    Returns:
        Tuple of (shmoo_body_list, shmoo_title_list, shmoo_dict).
    """
    tmpX = []
    tmpY = []
    tmpZ = []
    X = []
    Y = []
    Z = {}
    csv_df = pd.read_csv(csv_file, header=None, engine='python')
    for index, row in csv_df.iterrows():
        if ':' in row[0]:
            if len(tmpX) > 0:
                X.append(tmpX)
                Y.append(tmpY)
                Z[tmpY] = tmpZ
            tmpY = row[0]
            tmpX = []
            tmpZ = []
        elif not ("P" in row.dropna().to_list()) and not ("." in row.dropna().to_list()):
            # axis data row
            tmpZ.append(row.dropna().to_list())
            continue
        else:
            tmpX.append(row.dropna().to_list()[1:])
            tmpZ.append(row.dropna().to_list())
    X.append(tmpX)
    Y.append(tmpY)
    Z[tmpY] = tmpZ
    return X, Y, Z


# ---------------------------------------------------------------------------
# Tensor Conversion
# ---------------------------------------------------------------------------

def convert_shm_to_tensor(csv_file: str, batch_cnt: int,
                          shmoo_body=None, shmoo_title=None,
                          mode: str = 'P'):
    """Convert shmoo data to PyTorch tensors for inference.

    Args:
        csv_file: Path to intermediate CSV (used when mode='P').
        batch_cnt: Batch size. Use -1 for full dataset.
        shmoo_body: Shmoo body data (used when mode='S').
        shmoo_title: Shmoo title data (used when mode='S').
        mode: 'P' for file-based, 'S' for serial/in-memory.

    Returns:
        Tuple of (test_iter, raw_dict).
    """
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 0
    if shmoo_body is None:
        shmoo_body = []
    if shmoo_title is None:
        shmoo_title = []
    if mode == 'P':
        dataset = src.CsvDataset_Test(csv_file)
    else:
        dataset = src.CsvDataset_Test_Serial(shmoo_body, shmoo_title)
    if batch_cnt < 0:
        batch_size = len(dataset)
    else:
        batch_size = batch_cnt
    test_iter = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_iter, dataset.raw_dict


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_shm_report_xlsx(titles: list, shms: dict, filename: str,
                             parallel_gap: str = 'Disable',
                             sites_list: Optional[List[str]] = None,
                             logger: Callable = print) -> str:
    """Generate an XLSX report from shmoo analysis results.

    Args:
        titles: List of title strings (instance:site:result).
        shms: Dict mapping title → list of shmoo data rows.
        filename: Base filename for the report.
        parallel_gap: 'Disable', '15', or '25' — column gap for parallel plot.
        sites_list: List of site numbers (required when parallel_gap is enabled).
        logger: Logging function.

    Returns:
        Path to the generated XLSX report.
    """
    report_name = filename + '_report.xlsx'
    try:
        workbook = xlsxwriter.Workbook(report_name)
        worksheet = workbook.add_worksheet('SHM Result')

        format_2XXX = workbook.add_format({'bg_color': '#FF0000'})
        format_7XXX = workbook.add_format({'bg_color': '#008000'})

        worksheet.outline_settings(True, False, True, False)

        if parallel_gap != "Disable" and sites_list is not None:
            interval_columns = int(parallel_gap)
            siteCnt = 0
            for selected_site in sites_list:
                siteCnt = siteCnt + 1
                iColumn = (siteCnt - 1) * interval_columns
                worksheet.write_row(0, iColumn,
                                    ['Instance', '', '', '', '', 'Site Index',
                                     'Result Symbol', 'Result'])
                row = 1
                col = 0
                for title, shm in zip(titles, shms):
                    info_line = title.split(':')
                    if info_line[2] != selected_site:
                        continue
                    info_line[1:1] = [''] * 3
                    worksheet.write_row(row, iColumn, info_line)
                    worksheet.set_row(row, None, None, {'collapsed': True})
                    row += 1
                    for i in range(len(shms[shm])):
                        worksheet.write_row(row, iColumn, shms[shm][i])
                        worksheet.set_row(row, None, None,
                                          {'level': 1, 'hidden': True})
                        row += 1
                        col = max(col, len(shms[shm][i]))

                if row > 1:
                    worksheet.conditional_format(
                        0, iColumn, row, col + iColumn,
                        {'type': 'cell', 'criteria': 'equal to',
                         'value': '"."', 'format': format_2XXX})
                    worksheet.conditional_format(
                        0, iColumn, row, col + iColumn,
                        {'type': 'cell', 'criteria': 'equal to',
                         'value': '"P"', 'format': format_7XXX})
        else:
            worksheet.write_row(0, 0,
                                ['Instance', '', '', '', '', 'Site Index',
                                 'Result Symbol', 'Result'])
            row = 1
            col = 0
            for title, shm in zip(titles, shms):
                info_line = title.split(':')
                info_line[1:1] = [''] * 3
                worksheet.write_row(row, 0, info_line)
                worksheet.set_row(row, None, None, {'collapsed': True})
                row += 1
                for i in range(len(shms[shm])):
                    worksheet.write_row(row, 0, shms[shm][i])
                    worksheet.set_row(row, None, None,
                                      {'level': 1, 'hidden': True})
                    row += 1
                    col = max(col, len(shms[shm][i]))

            worksheet.conditional_format(
                0, 0, row, col,
                {'type': 'cell', 'criteria': 'equal to',
                 'value': '"."', 'format': format_2XXX})
            worksheet.conditional_format(
                0, 0, row, col,
                {'type': 'cell', 'criteria': 'equal to',
                 'value': '"P"', 'format': format_7XXX})

        workbook.close()
        logger('Xlsx file is written!')

    except xlsxwriter.exceptions.FileCreateError:
        logger("Please close " + report_name.split('/')[-1])

    return report_name


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(dataset_path: str = 'custom_SHM_data.csv',
                output_path: str = './state_dict.pth',
                epochs: int = 60, lr: float = 0.0014,
                batch_size: int = 100,
                logger: Callable = print):
    """Train AlexNet on a labeled shmoo CSV dataset.

    Args:
        dataset_path: Path to the training CSV file.
        output_path: Path to save the trained model weights.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        logger: Logging function.

    Returns:
        Tuple of (trained_net, test_iter) for optional visualization.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger(f"Using device: {device}")

    net = src.AlexNet()

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    logger(f'neural network architecture has {pytorch_total_params} parameters.')
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger(f'neural network architecture has {pytorch_total_params} trainable parameters.')

    net.train()
    train_iter, test_iter = src.load_custom_shm_data(batch_size, dataset_path)

    loss = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0004)

    src.train_network(net, train_iter, test_iter, loss, epochs,
                      batch_size, None, lr, optimizer)

    torch.save(net.state_dict(), output_path)
    logger(f"Model saved to {output_path}")

    return net, test_iter


# ---------------------------------------------------------------------------
# Inference (Analyse Shmoo)
# ---------------------------------------------------------------------------

def analyse_shmoo(log_path: str, config: dict,
                  model_path: str = './state_dict.pth',
                  parallel_gap: str = 'Disable',
                  logger: Callable = print,
                  cleanup_csv: bool = False) -> str:
    """Full shmoo analysis pipeline: txt+json → xlsx report.

    Args:
        log_path: Path to the shmoo log file (.txt).
        config: Parsed JSON config dictionary.
        model_path: Path to the trained model weights.
        parallel_gap: 'Disable', '15', or '25'.
        logger: Logging function.
        cleanup_csv: If True, delete intermediate CSV after report generation.

    Returns:
        Path to the generated XLSX report.
    """
    # Step 1: Parse shmoo log → CSV
    csv_path, sites_list = read_shm_log(log_path, config, logger)
    logger("Convert Shmoo log to CSV format completed.")

    # Step 2: Load model
    net = src.AlexNet()
    net.load_state_dict(torch.load(model_path, weights_only=True))
    net.eval()

    # Step 3: Run inference on each shmoo
    shmoo_body, shmoo_title, shmoo_dict = read_shmoo_csv(csv_path)
    titles = []
    for i in range(len(shmoo_title)):
        test_iter, _ = convert_shm_to_tensor(
            csv_path, -1, shmoo_body[i], shmoo_title[i], 'S')
        X, y = next(iter(test_iter))
        y_hat = net(X)
        y_hat = src.reformat_output(y_hat)
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        true_labels = y[0]
        pred_labels = src.get_custom_shm_labels(y_hat.detach().numpy(), 'A')
        titles.append(true_labels + ':' + pred_labels[0])

    # Step 4: Generate XLSX report
    report_path = generate_shm_report_xlsx(
        titles, shmoo_dict, log_path, parallel_gap, sites_list, logger)

    # Step 5: Cleanup intermediate CSV
    if cleanup_csv and os.path.exists(csv_path):
        os.remove(csv_path)
        logger(f"Cleaned up intermediate file: {csv_path}")

    logger("Finish analysis!")
    return report_path


# ---------------------------------------------------------------------------
# Eval (GUI-only caller)
# ---------------------------------------------------------------------------

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


def eval_model(csv_file: str, model_path: str = './state_dict.pth',
               shmoo_index: int = 5, channel_index: int = 0,
               logger: Callable = print):
    """Load model, run inference on one shmoo, extract layer activations.

    Used by GUI for CNN layer feature visualization.

    Args:
        csv_file: Path to intermediate shmoo CSV.
        model_path: Path to model weights.
        shmoo_index: Index of the shmoo to evaluate.
        channel_index: CNN layer index to extract activations from.
        logger: Logging function.

    Returns:
        Tuple of (activations_tensor, channel_count_list).
    """
    net = src.AlexNet()
    net.load_state_dict(torch.load(model_path, weights_only=True))
    net.eval()

    shmoo_body, shmoo_title, shmoo_dict = read_shmoo_csv(csv_file)
    test_iter, raw_dict = convert_shm_to_tensor(
        csv_file, -1, shmoo_body[shmoo_index],
        shmoo_title[shmoo_index], 'S')

    channel_count_list = [32, 16, 8, 4]

    conv_out = LayerActivations(list(net._modules.items()), channel_index)
    img, y = next(iter(test_iter))
    o = net(img)
    conv_out.remove()
    act = conv_out.features

    return act, channel_count_list


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlate_char_logs(file_paths: List[str], config: dict,
                        site_labels: str = '', gap: int = 25,
                        logger: Callable = print) -> str:
    """Generate site-to-site CHAR log comparison XLSX report.

    Args:
        file_paths: List of shmoo log file paths.
        config: Parsed JSON config dictionary.
        site_labels: Semicolon-separated site labels per file (e.g. '0,1;0,2').
        gap: Column gap between sites in XLSX.
        logger: Logging function.

    Returns:
        Path to the generated correlation XLSX report.
    """
    TER_keyword = getKeyWordFromSettingFile(config)
    joined_paths = ";".join(file_paths)
    report_path = getDatalogInfo(TER_keyword, joined_paths, site_labels, gap)
    logger(f"Correlation report generated: {report_path}")
    return report_path
