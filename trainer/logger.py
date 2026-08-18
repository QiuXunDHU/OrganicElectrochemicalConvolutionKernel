import json
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import EXPERIMENTS_DIR


class ExperimentLogger:
    """Synchronous experiment logger with deterministic flush semantics."""

    def __init__(self, exp_name, output_root=None):
        self.exp_name = exp_name
        root = Path(output_root or EXPERIMENTS_DIR).expanduser().resolve()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = root / f"{exp_name}_{timestamp}"
        self._lock = threading.Lock()
        self._closed = False
        self._setup_directories()

    def _setup_directories(self):
        for name in ("models", "logs", "figures", "data"):
            (self.base_dir / name).mkdir(parents=True, exist_ok=True)

    def _append_csv(self, relative_path, row):
        if self._closed:
            raise RuntimeError("Cannot write to a closed ExperimentLogger")
        path = self.base_dir / relative_path
        with self._lock:
            pd.DataFrame([row]).to_csv(
                path,
                mode="a",
                header=not path.exists(),
                index=False,
            )

    def log_metrics(
        self,
        backbone,
        kernel_name,
        epoch,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        learning_rate=None,
    ):
        self._append_csv(
            "data/training_metrics.csv",
            {
                "Backbone": backbone,
                "Kernel": kernel_name,
                "Epoch": epoch,
                "Train_Loss": train_loss,
                "Val_Loss": val_loss,
                "Train_Acc": train_acc,
                "Val_Acc": val_acc,
                "Learning_Rate": learning_rate,
            },
        )

    def log_confusion_matrix(self, backbone, kernel_name, matrix):
        self._append_csv(
            "data/confusion_matrices.csv",
            {
                "Backbone": backbone,
                "Kernel": kernel_name,
                "Matrix": json.dumps(matrix.tolist()),
            },
        )

    def save_config(self, config, filename="experiment_config.json"):
        path = self.base_dir / "data" / filename
        with path.open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2, default=str)
        return path

    def flush(self):
        """Writes are synchronous; the method documents the caller contract."""
        return None

    def close(self):
        self.flush()
        self._closed = True
