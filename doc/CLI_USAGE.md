# SHM Detect Tool — CLI Usage Guide

## Overview

`shm-detect` is a command-line interface for the SHM Detect Tool.  
It provides three subcommands for shmoo log analysis workflows:

| Command | Description |
|---------|-------------|
| `train` | Train the CNN model from a labeled dataset |
| `analyse` | Analyse shmoo logs and generate XLSX reports |
| `correlate` | Compare CHAR logs across sites with overlay reports |

---

## Prerequisites

- Python 3.10+
- Activate the virtual environment:
  ```bash
  source .venv/bin/activate
  ```
- A trained model file (`state_dict.pth`) is required for the `analyse` command. Use the `train` command to generate one, or use the provided default.
- A JSON config file (`SHM_keywords_setting.json`) is required for `analyse` and `correlate` commands.

---

## Quick Start

```bash
# Show top-level help
shm-detect --help

# Analyse a shmoo log (most common use case)
shm-detect analyse --log shmoo.txt --config SHM_keywords_setting.json

# Train a new model
shm-detect train --dataset custom_SHM_data.csv

# Compare multiple CHAR logs
shm-detect correlate --files file1.txt file2.txt --config SHM_keywords_setting.json
```

---

## Commands

### 1. `train` — Train the CNN Model

Train a new AlexNet-based CNN model using a labeled shmoo dataset.

```bash
shm-detect train [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `custom_SHM_data.csv` | Path to the training CSV dataset |
| `--output` | `./state_dict.pth` | Output path for the trained model weights |
| `--epochs` | `60` | Number of training epochs |
| `--lr` | `0.0014` | Learning rate |
| `--batch-size` | `100` | Training batch size |

**Examples:**

```bash
# Train with defaults (60 epochs)
shm-detect train

# Quick smoke test (1 epoch)
shm-detect train --epochs 1 --output ./test_model.pth

# Full training with custom parameters
shm-detect train --dataset my_data.csv --epochs 100 --lr 0.001 --batch-size 64
```

**Output:** A `.pth` model file saved to the specified `--output` path.

---

### 2. `analyse` — Analyse Shmoo Log

Parse a shmoo log file (`.txt`) with a JSON config, run CNN inference, and generate an XLSX report with pass/fail highlighting.

```bash
shm-detect analyse --log <LOG_FILE> --config <CONFIG_FILE> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--log` | *(required)* | Path to the shmoo log file (`.txt`) |
| `--config` | *(required)* | Path to JSON config file |
| `--model` | `./state_dict.pth` | Path to the trained model weights |
| `--gap` | `Disable` | Parallel plot column gap: `Disable`, `15`, or `25` |

**Examples:**

```bash
# Basic analysis
shm-detect analyse --log sample/S2S_Shmoo_site0.txt --config SHM_keywords_setting.json

# Use a custom model
shm-detect analyse --log shmoo.txt --config config.json --model custom_model.pth

# Enable parallel plot (multi-site side-by-side with 25-column gap)
shm-detect analyse --log shmoo.txt --config config.json --gap 25
```

**Output:** An XLSX report named `<log_file>_report.xlsx` (e.g., `shmoo.txt_report.xlsx`).  
The intermediate CSV file is automatically cleaned up after report generation.

**About Parallel Plot (`--gap`):**  
When the shmoo log contains multiple test sites, the `--gap` option arranges results side-by-side in the XLSX with the specified column gap between sites. This makes it easy to visually compare pass/fail patterns across sites.

- `Disable` — default, sites are listed sequentially  
- `15` — 15 columns gap between sites  
- `25` — 25 columns gap between sites  

---

### 3. `correlate` — Compare CHAR Logs

Generate a site-to-site comparison and overlay report from multiple CHAR log files.

```bash
shm-detect correlate --files <FILE1> <FILE2> ... --config <CONFIG_FILE> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--files` | *(required)* | One or more shmoo log files to compare |
| `--config` | *(required)* | Path to JSON config file |
| `--sites` | `""` (all sites) | Site labels per file, semicolon-separated |
| `--gap` | `25` | Column gap between sites in the XLSX |

**Examples:**

```bash
# Compare two log files (all sites)
shm-detect correlate --files file1.txt file2.txt --config SHM_keywords_setting.json

# Specify which sites to compare
shm-detect correlate --files file1.txt file2.txt --config config.json --sites "0,1;0,2"

# Custom column gap
shm-detect correlate --files f1.txt f2.txt f3.txt --config config.json --gap 30
```

**Output:** A correlation XLSX report with site-by-site comparison.

---

## JSON Config File

The JSON config file defines how to parse the shmoo log. Both `analyse` and `correlate` commands require this file.

### Example: S2S Format (Left Y-Axis)

```json
{
  "keyword_site": "Site: ",
  "keyword_item": "_SHM:",
  "keyword_start": "Tcoef(%)",
  "keyword_end": "Vcoef(%)",
  "keyword_pass": "P|\\*",
  "keyword_fail": "\\.|#",
  "keyword_y_axis_pos": "left"
}
```

### Example: CHAR Studio Format (Right Y-Axis)

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

### Config Fields

| Field | Description |
|-------|-------------|
| `keyword_site` | Keyword to identify site information lines |
| `keyword_item` | Keyword to identify test instance/item lines |
| `keyword_start` | Keyword to identify the Y-axis start (shmoo body begins after this) |
| `keyword_end` | Keyword for X-axis end marker |
| `keyword_pass` | Regex pattern for pass symbols (e.g., `\\+`, `P\|\\*`) |
| `keyword_fail` | Regex pattern for fail symbols (e.g., `\\-`, `\\.\|#`) |
| `keyword_y_axis_pos` | Y-axis position: `"left"` or `"right"` |

> **Note:** For CHAR studio logs with vertical X-axis labels (right Y-axis format), the tool automatically transposes the vertical digits into proper axis values.

---

## Typical Workflow

```
1. Prepare config     →  Create or select a JSON config file matching your log format
2. Analyse shmoo      →  shm-detect analyse --log shmoo.txt --config config.json
3. Review report      →  Open the generated _report.xlsx in Excel
4. (Optional) Compare →  shm-detect correlate --files f1.txt f2.txt --config config.json
```
