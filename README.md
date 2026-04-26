# SHM Detect Tool — Shmoo Log Analysis with CNN

**Automatically classify shmoo plot pass/fail patterns using a PyTorch CNN**

[![PyPI Version](https://img.shields.io/pypi/v/shm-detect-tool.svg)](https://pypi.org/project/shm-detect-tool/)
[![Python Versions](https://img.shields.io/pypi/pyversions/shm-detect-tool.svg)](https://pypi.org/project/shm-detect-tool/)
[![License](https://img.shields.io/pypi/l/shm-detect-tool.svg)](https://github.com/showjim/SHM_Detect_Tool/blob/master/LICENSE)
[![PyPI Downloads](https://img.shields.io/pypi/dm/shm-detect-tool.svg)](https://pypi.org/project/shm-detect-tool/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI Usage](#cli-usage)
  - [GUI Usage](#gui-usage)
  - [Web App Usage](#web-app-usage)
- [JSON Configuration](#json-configuration)
- [Screenshots](#screenshots)
- [License](#license)

---

## Overview

The **SHM Detect Tool** uses a CNN (Convolutional Neural Network) built on PyTorch to automatically analyse shmoo plot datalogs from ATE (Automated Test Equipment). It parses raw shmoo text logs, converts them into image-like tensors, and classifies each plot's pass/fail pattern — replacing tedious manual inspection with fast, repeatable AI-based detection.

### Background

- **Challenge**: Engineers spend significant time visually inspecting shmoo plots to identify pass/fail patterns, margins, and anomalies across multiple test sites and conditions.
- **Solution**: SHM Detect Tool automates this process by training an AlexNet-based CNN to classify shmoo patterns, generating structured XLSX reports with pass/fail highlighting and site-to-site correlation.

---

## Features

- **CNN-Based Shmoo Classification** — AlexNet architecture trained on labeled shmoo datasets
- **Shmoo Log Parsing** — convert raw text shmoo logs to structured CSV with configurable keyword patterns
- **XLSX Report Generation** — automated reports with pass/fail highlighting per shmoo plot
- **Site-to-Site Correlation** — compare shmoo results across multiple test sites with overlay reports
- **Parallel Plot Support** — side-by-side multi-site shmoo comparison in XLSX
- **CLI Interface** — scriptable commands for training, analysis, and correlation
- **GUI Interface** — PyQt5-based desktop application for interactive workflows
- **Web App Interface** — Streamlit-based web UI for browser-based analysis
- **Configurable Parsing** — JSON config files to support different shmoo log formats (S2S, CHAR Studio, etc.)

---

## Installation

### Requirements

- Python 3.9 – 3.14
- Windows, macOS, or Linux

### Install from PyPI (Recommended)

```bash
# Basic installation (CLI + core analysis)
pip install shm-detect-tool

# With GUI support (PyQt5)
pip install shm-detect-tool[gui]

# With web app support (Streamlit)
pip install shm-detect-tool[webapp]
```

After installation, the `shm-detect` command is immediately available:

```bash
shm-detect --version
shm-detect --help
```

### Install from Source

```bash
git clone https://github.com/showjim/SHM_Detect_Tool.git
cd SHM_Detect_Tool
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -e .[gui,webapp]
```

---

## Usage

### CLI Usage

```bash
# Analyse a shmoo log and generate an XLSX report
shm-detect analyse --log shmoo.txt --config SHM_keywords_setting.json

# Train a new CNN model
shm-detect train --dataset custom_SHM_data.csv

# Compare shmoo results across multiple log files
shm-detect correlate --files file1.txt file2.txt --config SHM_keywords_setting.json

# Show version
shm-detect -V
```

See [doc/CLI_USAGE.md](doc/CLI_USAGE.md) for the full CLI reference including all options and examples.

### GUI Usage

Launch the desktop application:

```bash
python SHM_Detect_Tool.py
```

#### Workflow

1. **Convert Shmoo** — load a raw shmoo `.txt` log and convert it to CSV
2. **Detect Shmoo** — select the CSV file, run CNN inference, and review the generated XLSX report
3. **Train CNN** — (optional) train a new model from a labeled dataset

### Web App Usage

Launch the Streamlit-based web interface:

```bash
streamlit run SHM_Detect_Tool_webapp.py
```

---

## JSON Configuration

The tool uses a JSON config file to define parsing rules for different shmoo log formats (CHAR Studio, customised, etc.). Each format has unique keywords for identifying site info, test items, axis boundaries, and pass/fail symbols.

### Config Fields

| Field | Description |
|-------|-------------|
| `keyword_site` | Keyword that identifies site information lines in the log |
| `keyword_item` | Keyword that identifies test instance or item lines |
| `keyword_start` | Keyword marking the Y-axis label start (shmoo body begins after this line) |
| `keyword_end` | Keyword marking the X-axis end boundary |
| `keyword_pass` | Regex pattern matching pass symbols (e.g., `P`, `*`, `+`) |
| `keyword_fail` | Regex pattern matching fail symbols (e.g., `.`, `#`, `-`, `E`) |
| `keyword_y_axis_pos` | Y-axis label position: `"left"` (customised) or `"right"` (CHAR Studio) |

The package ships with a default CHAR Studio config:
```json
{
  "keyword_site": "Site",
  "keyword_item": "<",
  "keyword_start": "Tcoef(AC Spec)",
  "keyword_end": "X Axis: Vcoef(DC Spec)",
  "keyword_pass": "\\+",
  "keyword_fail": "-|E",
  "keyword_y_axis_pos": "right"
}
```

See [doc/CLI_USAGE.md](doc/CLI_USAGE.md) for complete examples, regex notes, and format-specific details.

---

## Screenshots

![SHM Detect Tool](/img/Picture1.png)

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

<div align="center">

**Built for the semiconductor ATE community**

</div>