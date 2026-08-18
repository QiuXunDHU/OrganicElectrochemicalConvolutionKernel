# PhotosistorsNetwork

<p align="right"><strong>中文</strong> | <a href="README_EN.md">English</a></p>

用于运行可复现图像分类流程的软件。

> 本页仅说明公开的软件接口和运行方法。

## 功能

- 四种前端模式：`device`、`laplacian`、`learnable`、`none`
- 五种分类骨干：`resnet18`、`mobilenet_v2`、`densenet121`、`vit`、`swin`
- 固定 seed 的分层训练/验证/测试划分
- 最佳验证 checkpoint、早停和配置记录
- 分类指标、混淆矩阵、训练曲线和 Grad-CAM 可视化
- 当前格式与历史 `state_dict` checkpoint 兼容

## 软件流程

<p align="center">
  <img src="docs/assets/system-overview.svg" alt="图像分类软件流程" width="100%">
</p>

## 算子接口

对于固定的 3×3 矩阵 \(K\) 和灰度输入 \(X\)，前端计算

$$
Y_{b,1,u,v}=\sum_{i=0}^{2}\sum_{j=0}^{2}
K_{ij}X_{b,1,su+i,sv+j},
\qquad s=3.
$$

卷积前端均为 1→1 通道、kernel size 3、stride 3、无 padding、无 bias。`device` 是固定预设的兼容 CLI 名称；固定预设不参与训练。

| 模式 | 前端 | 参数状态 |
|---|---|---|
| `device` | 预设固定 3×3 核 | 冻结 |
| `laplacian` | 固定 Laplacian 3×3 核 | 冻结 |
| `learnable` | 随机初始化 3×3 卷积 | 可训练 |
| `none` | 无前端 | 不适用 |

前端作为额外模块置于分类骨干之前；骨干输入层另行适配为单通道输入。

## 安装

推荐 Python 3.10 或 3.11。CPU 环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
python -m pip install -r requirements.txt
```

CUDA 用户应根据本机环境选择兼容的 PyTorch 2.7.1 和 torchvision 0.22.1；其余依赖版本见 [`requirements.txt`](requirements.txt)。`--pretrained` 可能下载外部骨干权重，默认关闭。

## 数据

默认分类数据目录：

```text
data/raw/UCMerced_LandUse/Images/
├─ <class-name>/
├─ ...
└─ <class-name>/
```

加载器会检查类别目录与软件配置是否一致；其他位置可通过 `--data-root` 指定。第三方数据来源与使用条款见 [`data/raw/UCMerced_LandUse/readme.txt`](data/raw/UCMerced_LandUse/readme.txt)。

## 使用

四模式 CPU smoke test：

```powershell
python -m scripts.train `
  --backbones resnet18 `
  --kernels device laplacian learnable none `
  --batch-size 2 `
  --num-workers 0 `
  --device cpu `
  --smoke-test
```

最小训练示例：

```powershell
python -m scripts.train `
  --backbones resnet18 `
  --kernels device learnable `
  --epochs 100 `
  --patience 10 `
  --seed 42
```

训练后可视化：

```powershell
python -m scripts.visualize `
  --exp-dir experiments/LandUse_Classification_YYYYMMDD_HHMMSS `
  --backbone resnet18 `
  --kernel device `
  --max-samples 100
```

完整参数见 `python -m scripts.train --help` 和 `python -m scripts.visualize --help`。

## 输出与复现

运行结果写入 `experiments/<运行名称_时间戳>/`，包括配置 JSON、逐 epoch 指标、测试指标、混淆矩阵、图表、日志和最佳模型。

默认数据划分为 60%/20%/20%。测试前会恢复验证集 Accuracy 最佳的 checkpoint。重复运行时应保留 seed、配置和对应结果。

## 许可

- 依赖版本固定在 [`requirements.txt`](requirements.txt)。
- 项目原创代码采用 [MIT License](LICENSE)。第三方数据和外部权重遵循各自来源条款。
