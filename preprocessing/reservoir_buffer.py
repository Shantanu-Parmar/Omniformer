# preprocessing/reservoir_buffer.py
"""
Simple, memory-aware reservoir buffer for sequential Parquet reading.

- Opens Parquet file once.
- Reads in chunks of configurable size (default 200_000 rows).
- Fills buffer only when fill level < 70%.
- Hard cap on buffer size (estimated from target MB).
- Producer thread throttles when full or consumer is slow.
- Yields full row dicts (all columns preserved).
- Diagnostic logging (console + file) for progress and pressure.
"""

import pyarrow.parquet as pq
import threading
import time
import logging
from collections import deque
from pathlib import Path
from typing import Iterator, Dict, Any
import psutil


# Diagnostic logger for preprocessing (separate from model training logs)
logger = logging.getLogger("preprocessing")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(console_handler)

def setup_file_handler(log_dir: str = "logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"{log_dir}/preproc_diagnostics_{timestamp}.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)


class ReservoirBuffer:
    def __init__(
        self,
        parquet_path: str,
        chunk_size: int = 200_000,
        target_buffer_mb: float = 500.0,
        low_water_threshold: float = 0.70,
        log_dir: str = "logs",
    ):
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.is_file():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self.chunk_size = chunk_size
        self.target_buffer_mb = target_buffer_mb
        self.low_water_threshold = low_water_threshold

        # Estimate bytes per row (rough, will refine after first chunk)
        self.bytes_per_row_estimate = 200  # initial guess
        self.max_buffer_rows = self._calculate_max_rows()

        self.buffer = deque(maxlen=self.max_buffer_rows)
        self.lock = threading.Lock()
        self.running = False
        self.producer_thread = None
        self.low_water_event = threading.Event()

        # pyarrow setup - open once
        self.parquet_file = pq.ParquetFile(str(self.parquet_path))
        self.row_group_idx = 0
        self.current_batch_iter = None

        setup_file_handler(log_dir)
        logger.info(f"Reservoir initialized | file={self.parquet_path} | chunk_size={chunk_size} | target_mb={target_buffer_mb} | max_rows≈{self.max_buffer_rows}")

    def _calculate_max_rows(self) -> int:
        avail_bytes = psutil.virtual_memory().available
        safe_bytes = int(avail_bytes * 0.6)  # conservative
        target_bytes = int(self.target_buffer_mb * 1024 * 1024)
        max_bytes = min(safe_bytes, target_bytes)
        return max(10_000, max_bytes // self.bytes_per_row_estimate)

    def start(self):
        self.running = True
        self.producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.producer_thread.start()
        logger.info("Producer thread started")

    def stop(self):
        self.running = False
        if self.producer_thread and self.producer_thread.is_alive():
            self.producer_thread.join(timeout=5.0)
        logger.info("Reservoir stopped")

    def _producer_loop(self):
        while self.running:
            with self.lock:
                current_fill = len(self.buffer) / self.buffer.maxlen if self.buffer.maxlen else 0
                ram_percent = psutil.virtual_memory().percent

            if current_fill >= 0.95 or ram_percent > 85:
                time.sleep(1.0)  # deep backoff
                continue

            if current_fill >= self.low_water_threshold:
                time.sleep(0.2)  # light backoff
                continue

            # Read next chunk
            try:
                if self.current_batch_iter is None:
                    self.current_batch_iter = self.parquet_file.iter_batches(batch_size=self.chunk_size)

                batch = next(self.current_batch_iter)
                rows = batch.to_pylist()  # list of dicts

                # Update row size estimate (after first batch)
                if self.row_group_idx == 0 and rows:
                    sample_row = rows[0]
                    self.bytes_per_row_estimate = sum(len(str(v).encode()) for v in sample_row.values()) + 100  # rough
                    self.max_buffer_rows = self._calculate_max_rows()
                    logger.info(f"Updated row size estimate: ~{self.bytes_per_row_estimate} bytes | new max_rows={self.max_buffer_rows}")

                with self.lock:
                    for row in rows:
                        self.buffer.append(row)

                self.row_group_idx += 1
                logger.info(f"Read chunk {self.row_group_idx} | rows_added={len(rows)} | buffer_fill={len(self.buffer)}/{self.buffer.maxlen} | ram={ram_percent:.1f}%")

            except StopIteration:
                logger.info("Reached end of Parquet file")
                break
            except Exception as e:
                logger.error(f"Error reading chunk: {e}")
                time.sleep(5.0)

    def get_next_row(self) -> Dict[str, Any]:
        """
        Consumer-facing: get one full row dict (all columns).
        Blocks if buffer empty.
        """
        while self.running:
            with self.lock:
                if self.buffer:
                    row = self.buffer.popleft()
                    # Wake producer if we dropped below threshold
                    if len(self.buffer) / self.buffer.maxlen < self.low_water_threshold:
                        self.low_water_event.set()
                    return row

            # Buffer empty → wait for producer
            logger.warning("Buffer empty - waiting for next chunk")
            self.low_water_event.wait(timeout=5.0)
            self.low_water_event.clear()

        raise EOFError("Reservoir closed - end of data")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Make buffer iterable directly"""
        return self

    def __next__(self) -> Dict[str, Any]:
        return self.get_next_row()