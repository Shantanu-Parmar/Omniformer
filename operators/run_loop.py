# operators/run_loop.py
"""
The main linear streaming loop:
- Pulls rows from reservoir
- Feeds to iterator
- Aggregates in epoch manager
- Triggers reward & adjustments
- Handles checkpoints & logging
"""

import torch
from typing import Optional
from pathlib import Path
import time

from config import Config
from logging.structured_logger import StructuredLogger
from preprocessing.reservoir_buffer import ReservoirBuffer
from operators.iterator import Iterator
from operators.epoch_manager import EpochManager
from operators.reward_manager import RewardManager
from model import IterationLSTM
import torch.optim as optim


def run_loop(config: Config, logger: StructuredLogger):
    """
    Infinite (or until EOF) processing loop.
    """

    # Initialize model
    model = IterationLSTM(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_lstm_layers=config.num_lstm_layers,
        num_aux_families=config.num_aux_families,
        dropout=config.dropout,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)

    # Managers
    epoch_manager = EpochManager(config, logger)
    reward_manager = RewardManager(config, logger)
    reward_manager.register_optimizer(optimizer)

    # Reservoir
    buffer = ReservoirBuffer(
        parquet_path=config.parquet_path,
        chunk_size=config.chunk_size,
        target_buffer_mb=config.target_buffer_mb,
        low_water_threshold=config.low_water_threshold,
        log_dir=config.log_dir,
    )
    buffer.start()

    # Iterator
    iterator = Iterator(model, config, logger)

    current_hidden = None
    epoch_idx = 0
    iteration_in_epoch = 0

    try:
        for row in buffer:
            # Feed to iterator
            iterator.step(row, epoch_manager)

            # Check if iteration complete (inside iterator already signals)
            # Epoch manager checks if epoch complete
            if epoch_manager.is_complete():
                poll_score, next_pool, randomness_p = epoch_manager.finalize_and_prepare_next(
                    reward_signal={}  # will be updated after reward
                )

                # Reward check
                if reward_manager.should_compute_reward(epoch_idx):
                    signal = reward_manager.compute_reward()
                    reward_manager.apply_adjustment()

                    # Pass ramp boost to epoch manager if needed
                    if signal.get("ramp_boost", False):
                        # Here we could override epoch_manager.current_randomness_p
                        # But since epoch_manager already has its own ramp logic, we can just log
                        logger.work("ramp", {"boost_from_reward": True})

                epoch_idx += 1
                iteration_in_epoch = 0

                # Checkpoint (optional)
                if epoch_idx % config.checkpoint_every_n_epochs == 0:
                    ckpt_path = Path(config.checkpoint_dir) / f"epoch_{epoch_idx}.pt"
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch_idx": epoch_idx,
                        "poll_score": poll_score,
                    }, ckpt_path)
                    logger.work("checkpoint", {"path": str(ckpt_path), "poll": poll_score})

                # Prepare for next epoch
                # Iterator gets new hidden pool (weighted random)
                # But since iterator holds current_hidden, we can set it here if needed
                # For now, iterator manages its own hidden persistence

    except EOFError:
        logger.info("Reached end of data stream")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        buffer.stop()
        logger.info("Run loop ended")