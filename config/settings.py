from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_conv_kernels


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_ROOT = DATA_DIR / "raw" / "UCMerced_LandUse" / "Images"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OECT_KERNEL_CSV = DATA_DIR / "Phototransistor.csv"
OECT_GATE_VOLTAGE = 0.05
OECT_RESPONSE_SOURCE = OECT_KERNEL_CSV.relative_to(PROJECT_ROOT).as_posix()

# Backward-compatible names retained for external scripts and old metadata readers.
DEVICE_KERNEL_CSV = OECT_KERNEL_CSV
DEVICE_KERNEL_VOLTAGE = OECT_GATE_VOLTAGE

_oect_kernels = load_conv_kernels(OECT_KERNEL_CSV)
if OECT_GATE_VOLTAGE not in _oect_kernels:
    available = ", ".join(str(value) for value in sorted(_oect_kernels))
    raise KeyError(
        f"OECT gate voltage {OECT_GATE_VOLTAGE} V was not found in "
        f"{OECT_KERNEL_CSV}. Available voltages: {available}"
    )

KERNEL_MAP = {
    # "device" remains the public CLI name for checkpoint and experiment compatibility.
    "device": _oect_kernels[OECT_GATE_VOLTAGE],
    "laplacian": np.array(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
        dtype=np.float32,
    ),
}
SUPPORTED_KERNEL_NAMES = (*KERNEL_MAP.keys(), "learnable", "none")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = (
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark",
    "overpass", "parkinglot", "river", "runway", "sparseresidential",
    "storagetanks", "tenniscourt",
)


def configure_plot_style():
    """Apply the optional SciencePlots theme without making imports fragile."""
    try:
        import scienceplots  # noqa: F401  Registers matplotlib styles.

        plt.style.use(["science", "no-latex", "high-vis", "nature"])
    except (ImportError, OSError):
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": "Times New Roman",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 300,
        }
    )


configure_plot_style()
