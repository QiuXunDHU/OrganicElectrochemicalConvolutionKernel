import pandas as pd
import os
from datetime import datetime
import multiprocessing as mp
from pathlib import Path



class ExperimentLogger:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.base_dir = Path(f"experiments/{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self._setup_directories()

        self.log_queue = mp.Queue()
        self.writer_process = mp.Process(target=self._log_writer)
        self.writer_process.start()

    def _setup_directories(self):
        (self.base_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'logs').mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'data').mkdir(parents=True, exist_ok=True)

    def _log_writer(self):
        while True:
            item = self.log_queue.get()
            if item is None:
                break
            log_type, data = item
            if log_type == 'metrics':
                pd.DataFrame(data).to_csv(
                    self.base_dir / 'data' / 'training_metrics.csv',
                    mode='a',
                    header=not os.path.exists(self.base_dir / 'data' / 'training_metrics.csv'),
                    index=False
                )
            elif log_type == 'confusion_matrix':
                pd.DataFrame(data).to_csv(
                    self.base_dir / 'data' / 'confusion_matrices.csv',
                    mode='a',
                    header=False,
                    index=False
                )

    def log_metrics(self, backbone, kernel_name, epoch, train_loss, val_loss, train_acc, val_acc):
        self.log_queue.put((
            'metrics',
            {
                'Backbone': [backbone],
                'Kernel': [kernel_name],  # 改为记录名称
                'Epoch': [epoch],
                'Train_Loss': [train_loss],
                'Val_Loss': [val_loss],
                'Train_Acc': [train_acc],
                'Val_Acc': [val_acc]
            }
        ))

    def log_confusion_matrix(self, backbone, kernel_name, cm):
        self.log_queue.put((
            'confusion_matrix',
            {
                'Backbone': [backbone],
                'Kernel': [kernel_name],  # 改为记录名称
                'Matrix': [cm.tolist()]
            }
        ))

    def close(self):
        self.log_queue.put(None)
        self.writer_process.join()
