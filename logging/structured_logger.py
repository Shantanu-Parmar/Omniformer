# logging/structured_logger.py
"""
Structured JSONL logger with three levels: FULL, WORK, MINIMAL.

- Writes one JSON object per line (JSONL format)
- Optional console output (colored or plain)
- Level filtering: only logs entries at or below the configured level
- Easy to grep, parse, or load into pandas/wandb later
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Literal

import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)  # for colored console


class StructuredLogger:
    LEVEL_PRIORITY = {
        "MINIMAL": 0,
        "WORK": 1,
        "FULL": 2,
    }

    COLOR_MAP = {
        "MINIMAL": Fore.GREEN,
        "WORK": Fore.CYAN,
        "FULL": Fore.WHITE,
        "ERROR": Fore.RED,
        "WARNING": Fore.YELLOW,
    }

    def __init__(
        self,
        log_level: Literal["FULL", "WORK", "MINIMAL"] = "WORK",
        log_dir: str = "logs",
        console: bool = True,
        enable_colors: bool = True,
    ):
        self.log_level = log_level.upper()
        self.console = console
        self.enable_colors = enable_colors

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"run_{timestamp}.jsonl"

        self._open_file()

        # Initial log entry
        self.log("startup", {
            "log_level": self.log_level,
            "log_file": str(self.log_file),
            "console": self.console,
            "colors": self.enable_colors,
        }, level="WORK")

    def _open_file(self):
        self.file_handle = open(self.log_file, "a", encoding="utf-8")

    def close(self):
        if hasattr(self, "file_handle"):
            self.file_handle.close()

    def _should_log(self, level: str) -> bool:
        requested = self.LEVEL_PRIORITY.get(level.upper(), 0)
        configured = self.LEVEL_PRIORITY.get(self.log_level, 0)
        return requested <= configured

    def log(
        self,
        category: str,
        data: Dict[str, Any],
        level: Literal["FULL", "WORK", "MINIMAL", "ERROR", "WARNING"] = "WORK",
    ):
        if not self._should_log(level):
            return

        entry = {
            "ts_unix": time.time(),
            "ts_iso": datetime.utcnow().isoformat(),
            "category": category,
            "level": level,
            **data,
        }

        # Write to JSONL file
        json_line = json.dumps(entry, sort_keys=True, default=str)  # str fallback for tensors etc.
        self.file_handle.write(json_line + "\n")
        self.file_handle.flush()

        # Console output (optional)
        if self.console:
            color = self.COLOR_MAP.get(level, Fore.WHITE)
            msg = f"{color}[{entry['ts_iso']}] {category} ({level}): {json.dumps(data, sort_keys=True, default=str)}{Style.RESET_ALL}"
            print(msg)

    # Convenience shortcuts
    def minimal(self, category: str, data: Dict[str, Any]):
        self.log(category, data, "MINIMAL")

    def work(self, category: str, data: Dict[str, Any]):
        self.log(category, data, "WORK")

    def full(self, category: str, data: Dict[str, Any]):
        self.log(category, data, "FULL")

    def warning(self, category: str, data: Dict[str, Any]):
        self.log(category, data, "WARNING")

    def error(self, category: str, data: Dict[str, Any]):
        self.log(category, data, "ERROR")


# Example usage (for testing)
if __name__ == "__main__":
    logger = StructuredLogger(log_level="FULL", console=True)
    logger.full("test", {"key": "value", "number": 42})
    logger.work("epoch", {"poll_score": 0.75})
    logger.minimal("reward", {"action": "increase_lr"})
    logger.error("critical", {"msg": "buffer empty"})