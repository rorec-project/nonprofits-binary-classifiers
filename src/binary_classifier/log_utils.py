"""Reusable logging setup that writes to both stdout and a timestamped file."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(stem: str = "pipeline", log_dir: str | Path = "logs") -> Path:
    """Configure dual logging: INFO+ to stdout, INFO+ to a dated file.

    Parameters
    ----------
    stem : str
        Log file name stem (e.g. ``"pipeline"`` → ``logs/pipeline_20260413_091527.log``)
    log_dir : str | Path
        Directory for log files (created if missing).

    Returns
    -------
    Path
        Absolute path to the created log file.

    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (log_dir / f"{stem}_{timestamp}.log").resolve()

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt_console = logging.Formatter("%(message)s")
    fmt_file = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt_console)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt_file)

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    logging.info("Logging to %s", log_path)
    return log_path
