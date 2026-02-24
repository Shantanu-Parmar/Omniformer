# operators/iterator.py
"""
Iterator: consumes rows from reservoir, groups by timestamp, runs model on strain row features,
computes multi-class prediction vs truth (OR-style for multiple aux), scores accuracy.
"""

import torch
from typing import Dict, Any, List, Tuple, Optional
from model import IterationLSTM
from logging.structured_logger import StructuredLogger
from config import Config


class Iterator:
    def __init__(
        self,
        model: IterationLSTM,
        config: Config,
        logger: StructuredLogger,
    ):
        self.model = model
        self.config = config
        self.logger = logger

        self.current_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Static channel mapping (class index)
        self.aux_channels = [
            "LSC-REFL_A_LF_OUT_DQ",
            "LSC-REFL_A_RF9_Q_ERR_DQ",
            "PEM-EX_ADC_0_18_OUT_DQ",
            "PEM-VAULT_MAG_1030X195Y_COIL_X_DQ",
            "PEM-VAULT_MAG_1030X195Y_COIL_Y_DQ",
            "CAL-PCALY_RX_PD_OUT_DQ",
            "OMC-PZT1_MON_AC_OUT_DQ",
            "PSL-FSS_FAST_MON_OUT_DQ",
        ]
        self.class_map = {name: idx + 1 for idx, name in enumerate(self.aux_channels)}  # 1 to 8
        self.background_class = 0

        # Feature columns from strain row
        self.strain_feature_columns = [
            "frequency",
            "tstart",
            "tend",
            "fstart",
            "fend",
            "snr",
            "q",
            "amplitude",
            "phase",
        ]

        self.timestamp_buffer: List[Dict[str, Any]] = []
        self.current_timestamp: Optional[float] = None
        self.timestamp_count = 0  # for iteration boundary

    def step(self, row: Dict[str, Any], epoch_manager):
        ts = row["timestamp"]

        # New timestamp → process previous segment if any
        if ts != self.current_timestamp and self.current_timestamp is not None:
            acc = self._process_current_segment()
            epoch_manager.add_timestamp_result(acc, self.current_hidden)
            self.timestamp_buffer = []
            self.timestamp_count += 1

            # Check if iteration complete (~5 s worth of timestamps)
            if self.timestamp_count >= self.config.iteration_timestamps:
                # Signal epoch manager that iteration ended
                # (epoch_manager can decide if epoch is done)
                self.timestamp_count = 0

        self.timestamp_buffer.append(row)
        self.current_timestamp = ts

    def _process_current_segment(self) -> float:
        if not self.timestamp_buffer:
            return 0.0

        # Find strain row
        strain_rows = [r for r in self.timestamp_buffer if "STRAIN" in r["chan_name"].upper()]
        if not strain_rows:
            self.logger.work("iterator", {"warning": "no strain row in segment", "ts": self.current_timestamp})
            return 0.0

        strain_row = strain_rows[0]  # only one

        # Extract features
        try:
            features = [float(strain_row[col]) for col in self.strain_feature_columns]
            x_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, input_dim)
        except (KeyError, ValueError) as e:
            self.logger.work("iterator", {"error": f"feature extraction failed: {e}", "ts": self.current_timestamp})
            return 0.0

        # Model forward
        probs, self.current_hidden = self.model(x_t, self.current_hidden)

        # Ground truth class
        aux_present = [r["chan_name"] for r in self.timestamp_buffer if "STRAIN" not in r["chan_name"].upper()]
        if not aux_present:
            truth_class = self.background_class
        else:
            # Pick first aux as truth class (OR style: any present counts)
            first_aux = aux_present[0]
            truth_class = self.class_map.get(first_aux, self.background_class)  # fallback to background if unknown

        # Predicted class
        pred_class = probs.argmax(dim=-1).item()

        # Score (exact match, but forgiving if multiple aux)
        acc = 1.0 if pred_class == truth_class else 0.0

        # Logging
        if self.logger.log_level == "FULL":
            self.logger.full("iterator", {
                "ts": self.current_timestamp,
                "pred_class": pred_class,
                "truth_class": truth_class,
                "acc": acc,
                "probs": probs.tolist(),
                "hidden_norm": self.current_hidden[0].norm().item() if self.current_hidden else None,
            })

        return acc

    def reset(self):
        self.current_hidden = None
        self.timestamp_buffer = []
        self.current_timestamp = None
        self.timestamp_count = 0