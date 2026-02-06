import torch
from pathlib import Path
from config import KERNEL_MAP, DEVICE
from data import prepare_data_loaders
from models import CustomCNN
from visusalizatio import VisualizationHelper


def visualization_main():
    # === 配置参数 ===
    EXP_DIR = Path("../scripts/experiments/LandUse_Classification_20250512_222511")
    BACKBONE = 'resnet18'
    KERNEL_NAME = 'laplacian'  # 必须与训练时使用的kernel名称一致

    # === 加载模型 ===
    def load_trained_model():
        # 从KERNEL_MAP获取对应的kernel参数
        kernel = KERNEL_MAP[KERNEL_NAME]

        # 初始化模型结构时必须包含相同的conv_kernel参数
        model = CustomCNN(
            conv_kernel=kernel,  # 关键修改：必须传递训练时使用的kernel
            backbone_name=BACKBONE,
            pretrained=True
        )

        # 加载权重（严格模式）
        state_dict = torch.load(EXP_DIR / 'models' / f'best_{BACKBONE}_{KERNEL_NAME}.pth',
                                map_location=DEVICE)

        # 检查并过滤不需要的键
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=True)

        model.to(DEVICE)
        model.eval()
        return model

    # === 准备数据 ===
    def prepare_test_loader():
        loaders = prepare_data_loaders(BACKBONE)
        return loaders['test']

    # === 执行可视化 ===
    print("Loading model...")
    model = load_trained_model()

    print("Preparing data...")
    test_loader = prepare_test_loader()

    print("Generating visualizations...")

    VIS_PARAMS = {
        'batch_size': 64,
        'max_samples': 1000,  # 示例值，设置为None处理所有样本
        'device': DEVICE
    }
    VisualizationHelper.visualize_all_results(
        exp_dir=EXP_DIR,
        model=model,
        test_loader=test_loader,
        **VIS_PARAMS  # 传递所有参数
    )
if __name__ == '__main__':
    visualization_main()