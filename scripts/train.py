import torch
import pandas as pd
import os
import logging
import traceback
from datetime import datetime

from config import KERNEL_MAP
from data import prepare_data_loaders
from models import CustomCNN
from trainer import ExperimentLogger, AdvancedTrainer
from visusalizatio import ResultVisualizer


def main():
    # backbones = ['resnet18', 'mobilenet_v2', 'densenet121', 'vit', 'swin']
    backbones = [ 'vit', 'swin']

    kernel_names = list(KERNEL_MAP.keys())


    exp_logger = ExperimentLogger("LandUse_Classification")

    logging.basicConfig(
        filename=str(exp_logger.base_dir / 'logs' / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

    results = []

    for backbone in backbones:
        for kernel_name in kernel_names:
            kernel = KERNEL_MAP[kernel_name]
            try:
                logging.info(f"Start training: Backbone={backbone}, Kernel={kernel_name}")
                loaders = prepare_data_loaders(backbone)

                model = CustomCNN(
                    conv_kernel=kernel,
                    backbone_name=backbone,
                    pretrained=True
                )

                trainer = AdvancedTrainer(model, logger=exp_logger)
                trainer.train(
                    train_loader=loaders['train'],
                    val_loader=loaders['val'],
                    backbone=backbone,
                    kernel_name=kernel_name,  # 传递名称
                    epochs=150
                )

                # 生成安全模型路径
                model_name = f"best_{backbone}_{kernel_name}.pth"
                best_model_path = exp_logger.base_dir / 'models' / model_name
                torch.save(model.state_dict(), best_model_path)

                # 加载并评估模型
                model.load_state_dict(torch.load(best_model_path))
                cm_path = exp_logger.base_dir / 'figures' / f'confusion_matrix_{backbone}_{kernel_name}.png'
                metrics = ResultVisualizer.analyze(model, loaders['test'], save_path=cm_path)
                ResultVisualizer.visualize_curves(exp_logger.base_dir)
                ResultVisualizer.visualize_bar_charts(exp_logger.base_dir)
                # 记录可读结果
                results.append({
                    'Backbone': backbone,
                    'ConvKernel': kernel_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1'],
                })

                # 实时保存结果
                pd.DataFrame(results).to_csv(
                    exp_logger.base_dir / 'data' / 'experiment_results.csv',
                    index=False
                )
            except Exception as e:
                logging.error(f"Error in {backbone} with kernel {kernel}: {str(e)}")
                traceback.print_exc()
                pd.DataFrame(results).to_csv(
                    exp_logger.base_dir / 'data' / 'interrupted_results.csv',
                    index=False
                )
                continue

    exp_logger.close()
    ResultVisualizer.visualize_curves(exp_logger.base_dir)
    ResultVisualizer.visualize_bar_charts(exp_logger.base_dir)
if __name__ == '__main__':
    main()