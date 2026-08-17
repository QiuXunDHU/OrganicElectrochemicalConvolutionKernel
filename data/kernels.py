import csv
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = ("Voltage", *(f"P{i}" for i in range(1, 10)))


def load_conv_kernels(csv_path):
    """Load and validate voltage-indexed 3x3 convolution kernels from CSV."""
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Kernel CSV does not exist: {path}")

    kernels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or ())]
        if missing_columns:
            raise ValueError(
                f"Kernel CSV {path} is missing columns: {', '.join(missing_columns)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                voltage = float(row["Voltage"])
                kernel = np.asarray(
                    [float(row[f"P{i}"]) for i in range(1, 10)],
                    dtype=np.float32,
                ).reshape(3, 3)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value in {path} at CSV line {line_number}: {error}"
                ) from error

            if not np.isfinite(kernel).all() or not np.isfinite(voltage):
                raise ValueError(f"Non-finite kernel value in {path} at CSV line {line_number}")
            if voltage in kernels:
                raise ValueError(f"Duplicate voltage {voltage} in {path} at CSV line {line_number}")
            kernels[voltage] = kernel

    if not kernels:
        raise ValueError(f"Kernel CSV contains no data rows: {path}")
    return kernels
