# -*- coding: utf-8 -*-
"""
SHM Detect Tool — Command Line Interface

Usage:
    python shm_cli.py train    --dataset custom_SHM_data.csv --output state_dict.pth
    python shm_cli.py analyse  --log shmoo.txt --config config.json
    python shm_cli.py correlate --files f1.txt f2.txt --config config.json
"""

import argparse
import json
import sys

import shm_backend


def cmd_train(args):
    """Handle the 'train' subcommand."""
    print(f"Training with dataset: {args.dataset}")
    net, test_iter = shm_backend.train_model(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    print(f"Training complete. Model saved to: {args.output}")


def cmd_analyse(args):
    """Handle the 'analyse' subcommand."""
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    print(f"Analysing shmoo log: {args.log}")
    print(f"Using config: {args.config}")
    print(f"Using model: {args.model}")
    if args.gap != 'Disable':
        print(f"Parallel plot gap: {args.gap}")

    report_path = shm_backend.analyse_shmoo(
        log_path=args.log,
        config=config,
        model_path=args.model,
        parallel_gap=args.gap,
        cleanup_csv=True,
    )
    print(f"Report generated: {report_path}")


def cmd_correlate(args):
    """Handle the 'correlate' subcommand."""
    with open(args.config, 'r') as f:
        config = json.load(f)

    print(f"Correlating files: {args.files}")
    print(f"Sites: {args.sites if args.sites else '(all)'}")
    print(f"Column gap: {args.gap}")

    report_path = shm_backend.correlate_char_logs(
        file_paths=args.files,
        config=config,
        site_labels=args.sites,
        gap=args.gap,
    )
    print(f"Correlation report generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        prog='shm_cli',
        description='SHM Detect Tool — Shmoo log analysis CLI',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True

    # --- train ---
    p_train = subparsers.add_parser('train', help='Train the CNN model')
    p_train.add_argument('--dataset', default='custom_SHM_data.csv',
                         help='Path to training CSV dataset (default: custom_SHM_data.csv)')
    p_train.add_argument('--output', default='./state_dict.pth',
                         help='Output model path (default: ./state_dict.pth)')
    p_train.add_argument('--epochs', type=int, default=60,
                         help='Number of training epochs (default: 60)')
    p_train.add_argument('--lr', type=float, default=0.0014,
                         help='Learning rate (default: 0.0014)')
    p_train.add_argument('--batch-size', type=int, default=100,
                         help='Batch size (default: 100)')
    p_train.set_defaults(func=cmd_train)

    # --- analyse ---
    p_analyse = subparsers.add_parser('analyse', help='Analyse shmoo log and generate XLSX report')
    p_analyse.add_argument('--log', required=True,
                           help='Path to shmoo log file (.txt)')
    p_analyse.add_argument('--config', required=True,
                           help='Path to JSON config file')
    p_analyse.add_argument('--model', default='./state_dict.pth',
                           help='Path to model weights (default: ./state_dict.pth)')
    p_analyse.add_argument('--gap', default='Disable',
                           choices=['Disable', '15', '25'],
                           help='Parallel plot column gap (default: Disable)')
    p_analyse.set_defaults(func=cmd_analyse)

    # --- correlate ---
    p_corr = subparsers.add_parser('correlate', help='Generate CHAR log correlation report')
    p_corr.add_argument('--files', nargs='+', required=True,
                        help='Shmoo log files to correlate')
    p_corr.add_argument('--config', required=True,
                        help='Path to JSON config file')
    p_corr.add_argument('--sites', default='',
                        help='Site labels per file, semicolon-separated (e.g. "0,1;0,2")')
    p_corr.add_argument('--gap', type=int, default=25,
                        help='Column gap between sites in XLSX (default: 25)')
    p_corr.set_defaults(func=cmd_correlate)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
