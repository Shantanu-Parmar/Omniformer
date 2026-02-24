# main.py
"""
Entry point for the Omniformer continual learning pipeline.

Usage:
    python main.py --parquet-path data/merged.parquet --log-level WORK --help

Runs the infinite streaming loop:
reservoir → iterator → epoch manager → reward manager
"""

import argparse
import torch
import signal
import sys
import time
from pathlib import Path

from config import Config
from logging.structured_logger import StructuredLogger
from preprocessing.reservoir_buffer import ReservoirBuffer
from operators.iterator import Iterator
from operators.epoch_manager import EpochManager
from operators.reward_manager import RewardManager
from operators.run_loop import run_loop
from model import IterationLSTM


def parse_args():
    parser = argparse.ArgumentParser(description="Continual GW glitch attribution pipeline")
    parser.add_argument("--parquet-path", type=str, default="data/merged.parquet", help="Path to input Parquet file")
    parser.add_argument("--log-level", type=str, default="WORK", choices=["FULL", "WORK", "MINIMAL"], help="Logging verbosity")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--initial-lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--no-console", action="store_true", help="Disable console logging")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config with overrides from CLI
    config = Config(
        parquet_path=args.parquet_path,
        log_level=args.log_level,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        initial_lr=args.initial_lr,
    )

    # Logger
    logger = StructuredLogger(
        log_level=config.log_level,
        log_dir=config.log_dir,
        console=not args.no_console,
    )

    logger.work("startup", {
        "config": config.__dict__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Create checkpoint dir
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Run the main loop
    try:
        run_loop(config, logger)
    except KeyboardInterrupt:
        logger.work("shutdown", {"reason": "KeyboardInterrupt"})
    except Exception as e:
        logger.error("main", {"error": str(e), "traceback": str(sys.exc_info())})
        raise
    finally:
        logger.work("shutdown", {"reason": "completed", "end_time": time.strftime("%Y-%m-%d %H:%M:%S")})
        logger.close()


if __name__ == "__main__":
    main()