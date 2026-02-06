import datetime
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from config import CLASS_NAMES, DEVICE
import numpy as np

class ResultVisualizer:
    @staticmethod
    def analyze(model, loader, class_names=CLASS_NAMES, save_path=None):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())

        report = classification_report(all_labels, all_preds,
                                       target_names=class_names, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(12, 10), dpi=300)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(
            xticks_rotation=45,
            values_format='d',
            colorbar=True,
            cmap='Blues',
            ax=plt.gca()
        )
        plt.title('Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            # ========== CSV保存部分 ==========
            csv_dir = save_path.parent.parent / 'data'
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_filename = save_path.stem + '.csv'
            csv_path = csv_dir / csv_filename
            cm_df = pd.DataFrame(
                cm,
                index=pd.Index(class_names, name='Actual'),
                columns=pd.Index(class_names, name='Predicted')
            )
            cm_df.to_csv(csv_path)
        plt.close()

        return {
            'accuracy': report['accuracy'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1': report['macro avg']['f1-score'],
            'cm': cm
        }

    @staticmethod
    def visualize_curves(exp_dir):
        df = pd.read_csv(exp_dir / 'data' / 'training_metrics.csv')

        # 绘制损失曲线
        for backbone in df['Backbone'].unique():
            plt.figure(figsize=(10, 6))
            backbone_df = df[df['Backbone'] == backbone]
            for kernel in backbone_df['Kernel'].unique():
                kernel_df = backbone_df[backbone_df['Kernel'] == kernel]
                plt.plot(kernel_df['Epoch'], kernel_df['Train_Loss'],
                         label=f'{kernel} (Train)')
                plt.plot(kernel_df['Epoch'], kernel_df['Val_Loss'],
                         '--', label=f'{kernel} (Val)')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'{backbone} Training and Validation Loss')
            plt.legend()
            plt.savefig(exp_dir / 'figures' / f'{backbone}_loss_curve.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        # 绘制准确率曲线
        for backbone in df['Backbone'].unique():
            plt.figure(figsize=(10, 6))
            backbone_df = df[df['Backbone'] == backbone]
            for kernel in backbone_df['Kernel'].unique():
                kernel_df = backbone_df[backbone_df['Kernel'] == kernel]
                plt.plot(kernel_df['Epoch'], kernel_df['Train_Acc'],
                         label=f'{kernel} (Train)')
                plt.plot(kernel_df['Epoch'], kernel_df['Val_Acc'],
                         '--', label=f'{kernel} (Val)')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy (%)', fontsize=12)
            plt.title(f'{backbone} Training and Validation Accuracy')
            plt.legend()
            plt.savefig(exp_dir / 'figures' / f'{backbone}_acc_curve.png',
                        bbox_inches='tight', dpi=300)
            plt.close()
    @staticmethod
    def visualize_bar_charts(exp_dir):
        results_path = exp_dir / 'data' / 'experiment_results.csv'
        if not results_path.exists():
            return

        df = pd.read_csv(results_path)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            pivot_df = df.pivot(index='Backbone', columns='ConvKernel', values=metric)
            pivot_df.plot(kind='bar', ax=ax, rot=0, width=0.8)
            ax.set_title(f'{metric} Comparison', fontsize=14)
            ax.set_xlabel('Backbone', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(title='Kernel Type', bbox_to_anchor=(1.05, 1),
                      loc='upper left')

        plt.tight_layout()
        plt.savefig(exp_dir / 'figures' / 'performance_bar_charts.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


class VisualizationHelper:
    def __init__(self, model, test_loader, class_names, exp_dir,
                 device=DEVICE, batch_size=64, max_samples=None):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.exp_dir = Path(exp_dir)
        self.device = device
        self.batch_size = batch_size  # 新增参数定义
        self.max_samples = max_samples  # 新增参数定义
        self.sample_counter = 0

        # 创建保存目录
        self.save_dir = self.exp_dir / 'visualizations'
        (self.save_dir).mkdir(parents=True, exist_ok=True)

        # 获取原始数据集样本路径
        self.test_samples = self._get_test_samples()
        self._register_hooks()

    def _get_test_samples(self):
        """更健壮的路径解析方法"""
        current_dataset = self.test_loader.dataset
        full_dataset = None
        indices = None

        # 递归查找原始数据集
        while True:
            # 处理ApplyTransformDataset的情况（检查subset属性）
            if hasattr(current_dataset, 'subset'):
                current_dataset = current_dataset.subset
            # 处理Subset的情况（检查indices属性）
            elif hasattr(current_dataset, 'indices'):
                indices = current_dataset.indices
                current_dataset = current_dataset.dataset
            else:
                break

        # 最终必须是ImageFolder类型
        if not isinstance(current_dataset, datasets.ImageFolder):
            raise ValueError("Cannot locate original ImageFolder dataset")

        # 如果没有indices则是完整数据集
        if indices is None:
            indices = range(len(current_dataset))

        return [
            (os.path.normpath(current_dataset.samples[i][0]), current_dataset.samples[i][1])
            for i in indices
        ]
    def _register_hooks(self):
        """注册钩子捕获中间结果"""
        self.activations = {}
        self.gradients = {}

        def forward_hook(module, input, output):
            self.activations['initial_conv'] = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients['initial_conv'] = grad_output[0]

        if hasattr(self.model, 'initial_conv') and self.model.initial_conv is not None:
            self.model.initial_conv.register_forward_hook(forward_hook)
            self.model.initial_conv.register_full_backward_hook(backward_hook)

    def _get_target_layer(self):
        """递归查找最后一个Conv2d层"""
        def find_last_conv(module):
            last_conv = None
            for child in module.children():
                # 优先查找子模块中的卷积层
                if isinstance(child, nn.Conv2d):
                    last_conv = child
                # 递归搜索子模块
                child_conv = find_last_conv(child)
                if child_conv is not None:
                    last_conv = child_conv
            return last_conv

        target_layer = find_last_conv(self.model.backbone)
        if target_layer is None:
            raise ValueError(f"No Conv2d layer found in {self.model.backbone_name} backbone")
        return target_layer

    def _apply_transforms(self, image):
        """改进的图像转换方法"""
        try:
            # 如果已经是PIL图像则直接使用
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image.astype('uint8'))

            # 增加尺寸校验
            if image.size[0] < 32 or image.size[1] < 32:
                image = image.resize((256, 256))

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1)
            ])
            return transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Transform error: {str(e)}")
            return torch.zeros(1, 1, 256, 256).to(self.device)

    def _save_image(self, tensor, filename, colormap='viridis'):
        """保存张量为图像"""
        tensor = tensor.cpu().detach().squeeze()
        if tensor.dim() == 3:
            tensor = tensor.mean(dim=0)

        plt.figure(figsize=(6, 6), dpi=150)
        plt.imshow(tensor.numpy(), cmap=colormap)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _compute_gradcam(self, input_tensor, class_idx=None):
        """修正维度处理的Grad-CAM计算"""
        target_layer = self._get_target_layer()
        features = []
        gradients = []

        def forward_hook(module, input, output):
            features.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_full_backward_hook(backward_hook)

        try:
            self.model.zero_grad()
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = output.argmax().item()

            # 创建onehot编码
            one_hot = torch.zeros_like(output)
            one_hot[0][class_idx] = 1

            # 反向传播
            output.backward(gradient=one_hot, retain_graph=True)

            # 计算权重
            pooled_gradients = torch.mean(gradients[-1], dim=[0, 2, 3])

            # 获取激活图并计算CAM
            activations = features[-1]
            cam = torch.einsum('ijkl,j->ikl', activations, pooled_gradients)  # shape: [B,H,W]
            cam = F.relu(cam)

            # 规范化和维度调整
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cam.unsqueeze(1)  # 添加通道维度 [B,1,H,W]

            # 双三次插值到输入尺寸
            target_size = (input_tensor.shape[2], input_tensor.shape[3])
            cam = F.interpolate(cam,
                                size=target_size,
                                mode='bicubic',
                                align_corners=False)

            return cam.squeeze().cpu().numpy(), class_idx  # 移除批次和通道维度

        except Exception as e:
            print(f"Grad-CAM计算错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
        finally:
            handle_forward.remove()
            handle_backward.remove()

    def visualize(self):
        """处理所有测试样本的可视化"""
        error_log = []
        metadata = []

        for idx, (img_path, label) in enumerate(self.test_samples):
            if self.max_samples and idx >= self.max_samples:
                break

            try:
                file_stem = Path(img_path).stem
                sample_dir = self.save_dir / file_stem
                sample_dir.mkdir(parents=True, exist_ok=True)

                with Image.open(img_path) as img:
                    # 处理基础图像
                    rgb_img = img.convert('RGB')
                    input_tensor = self._apply_transforms(rgb_img)
                    target_size = (input_tensor.shape[3], input_tensor.shape[2])

                    # 保存基础图像
                    self._save_basic_images(rgb_img, target_size, sample_dir)

                    # 边缘增强可视化
                    if hasattr(self.model, 'initial_conv'):
                        self._visualize_initial_conv_edges(
                            input_tensor,
                            sample_dir / 'initial_conv_edge.png'
                        )

                    # Grad-CAM可视化
                    cam, pred_class = self._compute_gradcam(input_tensor, label)
                    if cam is not None:
                        self._save_gradcam_results(
                            cam,
                            rgb_img.resize(target_size),
                            sample_dir
                        )

                # 记录元数据
                metadata.append(self._create_metadata_entry(
                    img_path, sample_dir, label, pred_class, input_tensor
                ))

                self._print_progress(idx)

            except Exception as e:
                error_log.append(self._format_error(img_path, e))
                continue

        self._save_metadata_and_errors(metadata, error_log)

    def _visualize_initial_conv_edges(self, input_tensor, save_path):
        """优化后的Initial Conv边缘增强可视化"""
        try:
            with torch.no_grad(), torch.cuda.amp.autocast():
                # 前向传播获取激活
                activation = self.model.initial_conv(input_tensor)

                # 使用PyTorch计算梯度（GPU加速）
                grad_x = torch.gradient(activation, dim=3)[0]
                grad_y = torch.gradient(activation, dim=2)[0]
                edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2)

                # 转换为NumPy并归一化
                edge_np = edge_strength.squeeze().cpu().numpy()
                edge_np = (edge_np - edge_np.min()) / (edge_np.max() - edge_np.min() + 1e-8)

                # 优化可视化参数
                plt.figure(figsize=(6, 6), dpi=150, facecolor='white')
                plt.imshow(edge_np,
                           cmap=LinearSegmentedColormap.from_list(
                               'edge_cmap',
                               [(1, 1, 1), (0.12, 0.56, 1.0), (0.0, 0.15, 0.55)],
                               N=256
                           ),
                           interpolation='none',  # 原始分辨率不插值
                           vmin=0,
                           vmax=1)
                plt.axis('off')
                # 优化保存参数
                plt.savefig(
                save_path,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none'
                )
                plt.close()

        except RuntimeError as e:
            print(f"Edge visualization failed: {str(e)}")
            traceback.print_exc()

    def _save_basic_images(self, rgb_img, target_size, sample_dir):
        """保存基础图像（RGB/灰度）"""
        try:
            # 保存原始尺寸
            rgb_img.save(sample_dir / 'original_rgb.jpg', quality=95)
            rgb_img.convert('L').save(sample_dir / 'original_gray.jpg', quality=95)

            # 保存调整尺寸
            resized_rgb = rgb_img.resize(target_size, Image.LANCZOS)
            resized_rgb.save(sample_dir / 'resized_rgb.jpg', optimize=True, quality=85)
            resized_rgb.convert('L').save(sample_dir / 'resized_gray.jpg', quality=85)

        except OSError as e:
            print(f"Image saving error: {str(e)}")

    def _save_gradcam_results(self, cam, rgb_img, sample_dir):
        """保存Grad-CAM结果"""
        try:
            # 转换灰度图
            gray_img = rgb_img.convert('L')

            # 生成叠加结果
            rgb_overlay = self._apply_heatmap(cam, rgb_img)
            gray_overlay = self._apply_heatmap(cam, gray_img)

            # 保存结果
            Image.fromarray(rgb_overlay).save(
                sample_dir / 'gradcam_rgb.jpg',
                quality=85,
                subsampling=0
            )
            Image.fromarray(gray_overlay).save(
                sample_dir / 'gradcam_gray.jpg',
                quality=85,
                subsampling=0
            )
        except Exception as e:
            print(f"Grad-CAM saving failed: {str(e)}")

    def _create_metadata_entry(self, img_path, sample_dir, label, pred_class, input_tensor):
        return {
            'original_path': str(img_path),
            'sample_dir': str(sample_dir.relative_to(self.save_dir)),
            'true_label': self.class_names[label],
            'predicted_label': self.class_names[pred_class] if pred_class is not None else None,
            'tensor_shape': list(input_tensor.shape),
            'processing_time': datetime.now().isoformat()
        }

    def _print_progress(self, idx):
        if (idx + 1) % 50 == 0:
            total = self.max_samples if self.max_samples else len(self.test_samples)
            print(f"Processed {idx + 1}/{total} samples")

    def _format_error(self, img_path, error):
        return f"Error processing {img_path}:\n{str(error)}\n{traceback.format_exc()}"

    def _save_metadata_and_errors(self, metadata, error_log):
        if metadata:
            pd.DataFrame(metadata).to_csv(self.save_dir / 'processing_metadata.csv', index=False)
        if error_log:
            with open(self.save_dir / 'processing_errors.log', 'w') as f:
                f.write('\n'.join(error_log))

    def _apply_heatmap(self, cam, pil_image):
        """应用热力图到PIL图像"""
        # 调整热力图尺寸匹配图像
        cam_resized = np.array(Image.fromarray(cam).resize(
            pil_image.size,
            resample=Image.Resampling.BICUBIC
        ))

        # 生成热力图
        heatmap = (plt.cm.jet(cam_resized)[..., :3] * 255).astype(np.uint8)

        # 叠加到原图
        original_array = np.array(pil_image)
        if original_array.ndim == 2:  # 灰度图转RGB
            original_array = np.stack([original_array] * 3, axis=-1)

        return (original_array * 0.6 + heatmap * 0.4).astype(np.uint8)

    @staticmethod
    def visualize_all_results(exp_dir, model, test_loader, **kwargs):
        """确保传递所有参数"""
        visualizer = VisualizationHelper(
            model=model,
            test_loader=test_loader,
            class_names=CLASS_NAMES,
            exp_dir=exp_dir,
            **kwargs  # 关键修复：传递所有参数
        )
        visualizer.visualize()
