# utils/memory.py
"""
Memory monitoring and estimation utilities.

Helps prevent OOM by:
- Checking available RAM
- Estimating safe buffer sizes
- Warning on high pressure
"""

import psutil
from typing import Dict, Tuple
import logging

logger = logging.getLogger("memory_utils")


def get_memory_stats() -> Dict[str, float]:
    """
    Returns current memory stats in GB.
    """
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024 ** 3),
        "available_gb": mem.available / (1024 ** 3),
        "used_gb": mem.used / (1024 ** 3),
        "percent_used": mem.percent,
        "free_gb": mem.free / (1024 ** 3),
    }


def is_high_memory_pressure(threshold_percent: float = 85.0) -> bool:
    """
    Returns True if RAM usage is above threshold.
    """
    stats = get_memory_stats()
    if stats["percent_used"] > threshold_percent:
        logger.warning(f"High memory pressure: {stats['percent_used']:.1f}% used, available={stats['available_gb']:.2f} GB")
        return True
    return False


def estimate_safe_buffer_rows(
    target_mb: float,
    bytes_per_row_estimate: int = 200,
    max_ram_fraction: float = 0.60,
    safety_margin_gb: float = 2.0,
) -> int:
    """
    Calculates a safe number of rows for buffer based on available RAM.
    """
    stats = get_memory_stats()
    avail_gb = stats["available_gb"]

    # Apply safety margin and fraction limit
    safe_gb = avail_gb * max_ram_fraction - safety_margin_gb
    safe_gb = max(0.5, safe_gb)  # never go below 500 MB

    target_bytes = min(target_mb * 1024 * 1024, safe_gb * 1024 * 1024)
    max_rows = int(target_bytes // bytes_per_row_estimate)

    logger.info(f"Memory estimate | available={avail_gb:.2f} GB | safe={safe_gb:.2f} GB | max_rows≈{max_rows} "
                f"(target_mb={target_mb}, bytes_per_row≈{bytes_per_row_estimate})")

    return max(10_000, max_rows)  # minimum 10k rows


def log_memory_summary(prefix: str = ""):
    """
    Logs a human-readable memory summary (for WORK/FULL logging).
    """
    stats = get_memory_stats()
    msg = (f"{prefix}Memory | total={stats['total_gb']:.1f} GB | "
           f"used={stats['used_gb']:.1f} GB ({stats['percent_used']:.1f}%) | "
           f"available={stats['available_gb']:.1f} GB")
    logger.info(msg)


def monitor_and_throttle(
    current_rows: int,
    max_rows: int,
    sleep_base_sec: float = 0.5,
    high_pressure_threshold: float = 85.0,
) -> float:
    """
    Returns sleep time for producer thread.
    Increases sleep if buffer full or RAM high.
    """
    fill_ratio = current_rows / max_rows if max_rows else 0
    stats = get_memory_stats()

    sleep_sec = sleep_base_sec

    if fill_ratio > 0.95:
        sleep_sec *= 3.0  # near full → longer pause
    elif fill_ratio > 0.85:
        sleep_sec *= 1.5

    if stats["percent_used"] > high_pressure_threshold:
        sleep_sec *= 4.0  # RAM pressure → aggressive backoff
        logger.warning(f"High RAM pressure ({stats['percent_used']:.1f}%) → throttling {sleep_sec:.2f}s")

    return sleep_sec