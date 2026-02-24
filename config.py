# config.py
"""
Central configuration for the entire pipeline.
All hyperparameters are defined here with defaults.
Can be overridden via CLI, env vars, or config file later.
"""

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class Config:
    # General
    parquet_path: str = "data/merged_labeled.parquet"  # single Parquet file
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    log_level: Literal["FULL", "WORK", "MINIMAL"] = "WORK"

    # Model (IterationLSTM)
    input_dim: int = 9                                # frequency, tstart, tend, fstart, fend, snr, q, amplitude, phase
    hidden_dim: int = 128
    num_lstm_layers: int = 3
    num_aux_families: int = 8                         # from your static list
    dropout: float = 0.1

    # Channel mapping (static aux list)
    aux_channel_names: List[str] = field(default_factory=lambda: [
        "LSC-REFL_A_LF_OUT_DQ",
        "LSC-REFL_A_RF9_Q_ERR_DQ",
        "PEM-EX_ADC_0_18_OUT_DQ",
        "PEM-VAULT_MAG_1030X195Y_COIL_X_DQ",
        "PEM-VAULT_MAG_1030X195Y_COIL_Y_DQ",
        "CAL-PCALY_RX_PD_OUT_DQ",
        "OMC-PZT1_MON_AC_OUT_DQ",
        "PSL-FSS_FAST_MON_OUT_DQ",
    ])

    # Strain row identification
    strain_channel_substring: str = "STRAIN"          # case-insensitive contains check

    # Feature columns from strain row (must match Parquet columns)
    strain_feature_columns: List[str] = field(default_factory=lambda: [
        "frequency",
        "tstart",
        "tend",
        "fstart",
        "fend",
        "snr",
        "q",
        "amplitude",
        "phase",
    ])

    # Time boundaries
    iteration_timestamps: int = 80000                 # ~5 s at 16 kHz (unique timestamps per iteration)
    epoch_iterations: int = 10                        # ~50 s total (N iterations per epoch)

    # Scoring / success
    success_threshold: float = 0.70                   # used for success_rate in summaries

    # Reservoir / preprocessing
    chunk_size: int = 200000                          # rows per read chunk
    target_buffer_mb: float = 500.0                   # soft target buffer size
    low_water_threshold: float = 0.70                 # refill when below this fill fraction
    max_ram_fraction: float = 0.60                    # safety cap (fraction of available RAM)

    # Ramp / exploration
    initial_randomness_p: float = 0.5
    ramp_decay_rate: float = 0.92                     # p *= rate per epoch
    min_randomness_p: float = 0.05
    ramp_boost_on_negative_reward: float = 0.4        # min p when reward negative

    # Reward / optimization
    initial_lr: float = 3e-4
    reward_every_n_epochs: int = 5                    # compute reward every M epochs
    trend_window: int = 5
    min_improvement: float = 0.01
    patience: int = 8
    big_plateau_threshold: int = 5                    # consecutive reward cycles with no improvement → recreate optimizer

    # Optimizer recreation
    optimizer_type: str = "Adam"                      # can be Adam, AdamW, RMSprop, etc.
    optimizer_recreate_on_plateau: bool = True

    # Checkpointing
    checkpoint_every_n_epochs: int = 10
    checkpoint_on_improvement: bool = True
    checkpoint_best_only: bool = True