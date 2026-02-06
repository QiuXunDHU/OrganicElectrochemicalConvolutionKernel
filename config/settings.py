import matplotlib.pyplot as plt
import torch
from data import load_conv_kernels

# === 全局配置部分添加 ===
KERNEL_MAP = {
    'device': load_conv_kernels("../data/Phototransistor.csv")[0.05],
    'laplacian': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
}

# Configuration
plt.style.use(['science', 'no-latex', 'high-vis','nature'])
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300
})
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = (
    'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
    'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
    'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
    'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
    'storagetanks', 'tenniscourt'
)
