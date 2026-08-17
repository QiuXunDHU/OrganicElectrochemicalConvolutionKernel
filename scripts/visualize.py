import argparse
from pathlib import Path

import torch

from config import (
    CLASS_NAMES,
    DEFAULT_DATASET_ROOT,
    DEVICE,
    KERNEL_MAP,
    OECT_GATE_VOLTAGE,
    OECT_RESPONSE_SOURCE,
    SUPPORTED_KERNEL_NAMES,
)
from data import prepare_data_loaders
from models import SUPPORTED_BACKBONES, CustomCNN, load_model_checkpoint
from visusalizatio import VisualizationHelper


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate sample and Grad-CAM visualizations.")
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, required=True)
    parser.add_argument("--kernel", choices=SUPPORTED_KERNEL_NAMES, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def resolve_device(value):
    device = DEVICE if value == "auto" else torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{device}' was requested, but CUDA is unavailable")
    return torch.device(device)


def _model_arguments(kernel_name):
    if kernel_name in KERNEL_MAP:
        fixed_front_end_kind = "oect" if kernel_name == "device" else "generic"
        return KERNEL_MAP[kernel_name], "fixed", fixed_front_end_kind
    if kernel_name == "learnable":
        return None, "learnable", "generic"
    return None, "none", "generic"


def load_trained_model(exp_dir, backbone, kernel_name, device):
    kernel, mode, fixed_front_end_kind = _model_arguments(kernel_name)
    model = CustomCNN(
        conv_kernel=kernel,
        backbone_name=backbone,
        pretrained=False,
        initial_conv_mode=mode,
        fixed_front_end_kind=fixed_front_end_kind,
        oect_gate_voltage=OECT_GATE_VOLTAGE if fixed_front_end_kind == "oect" else None,
        oect_response_source=OECT_RESPONSE_SOURCE if fixed_front_end_kind == "oect" else None,
    )
    checkpoint_path = Path(exp_dir).expanduser().resolve() / "models" / f"best_{backbone}_{kernel_name}.pth"
    payload = load_model_checkpoint(model, checkpoint_path, map_location=device, strict=True)

    if not payload.get("legacy_checkpoint"):
        saved_backbone = payload.get("backbone")
        saved_kernel = payload.get("kernel_name")
        if saved_backbone and saved_backbone != backbone:
            raise ValueError(
                f"Checkpoint backbone is '{saved_backbone}', but --backbone is '{backbone}'"
            )
        if saved_kernel and saved_kernel != kernel_name:
            raise ValueError(
                f"Checkpoint kernel is '{saved_kernel}', but --kernel is '{kernel_name}'"
            )

    model.to(device).eval()
    return model


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    exp_dir = args.exp_dir.expanduser().resolve()
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")

    print("Loading model...")
    model = load_trained_model(exp_dir, args.backbone, args.kernel, device)
    print("Preparing data...")
    test_loader = prepare_data_loaders(
        args.backbone,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        expected_classes=CLASS_NAMES,
    )["test"]
    print("Generating visualizations...")
    VisualizationHelper.visualize_all_results(
        exp_dir=exp_dir,
        model=model,
        test_loader=test_loader,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
