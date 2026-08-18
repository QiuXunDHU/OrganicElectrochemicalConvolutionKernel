import argparse
import logging
import random
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    CLASS_NAMES,
    DEFAULT_DATASET_ROOT,
    DEVICE,
    DEVICE_KERNEL_VOLTAGE,
    EXPERIMENTS_DIR,
    KERNEL_MAP,
    OECT_GATE_VOLTAGE,
    OECT_RESPONSE_SOURCE,
    SUPPORTED_KERNEL_NAMES,
)
from data import prepare_data_loaders
from models import SUPPORTED_BACKBONES, CustomCNN, load_model_checkpoint
from trainer import AdvancedTrainer, ExperimentLogger
from visusalizatio import ResultVisualizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible UC Merced classification experiments with a measured "
            "photoresponsive OECT convolution or comparison front ends."
        )
    )
    parser.add_argument("--backbones", nargs="+", choices=SUPPORTED_BACKBONES, default=["resnet18"])
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=SUPPORTED_KERNEL_NAMES,
        default=["device", "learnable"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--experiment-name", default="LandUse_Classification")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a concrete device such as cuda:0")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained backbone weights (may download files). Disabled by default.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate data/model combinations with one forward pass and skip training.",
    )
    return parser.parse_args(argv)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(value):
    device = DEVICE if value == "auto" else torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{device}' was requested, but CUDA is unavailable")
    return torch.device(device)


def build_model(backbone, kernel_name, pretrained=False):
    if kernel_name in KERNEL_MAP:
        kernel = KERNEL_MAP[kernel_name]
        mode = "fixed"
        fixed_front_end_kind = "oect" if kernel_name == "device" else "generic"
    elif kernel_name == "learnable":
        kernel = None
        mode = "learnable"
        fixed_front_end_kind = "generic"
    elif kernel_name == "none":
        kernel = None
        mode = "none"
        fixed_front_end_kind = "generic"
    else:
        raise ValueError(
            f"Unsupported kernel '{kernel_name}'. Choose one of: {', '.join(SUPPORTED_KERNEL_NAMES)}"
        )
    return CustomCNN(
        conv_kernel=kernel,
        backbone_name=backbone,
        pretrained=pretrained,
        initial_conv_mode=mode,
        fixed_front_end_kind=fixed_front_end_kind,
        oect_gate_voltage=OECT_GATE_VOLTAGE if fixed_front_end_kind == "oect" else None,
        oect_response_source=OECT_RESPONSE_SOURCE if fixed_front_end_kind == "oect" else None,
    )


def experiment_config(args, device):
    return {
        "backbones": list(args.backbones),
        "kernels": list(args.kernels),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "output_root": str(Path(args.output_root).expanduser().resolve()),
        "device": str(device),
        "pretrained": args.pretrained,
        "class_names": list(CLASS_NAMES),
        # Retained so consumers of earlier configuration files continue to work.
        "device_kernel_voltage": DEVICE_KERNEL_VOLTAGE,
        "oect_gate_voltage": OECT_GATE_VOLTAGE,
        "oect_response_source": OECT_RESPONSE_SOURCE,
        "oect_front_end": {
            "cli_kernel_name": "device",
            "photoresponsive": True,
            "measurement": "3x3 response matrix",
            "kernel_size": [3, 3],
            "stride": [3, 3],
            "bias": False,
            "kernel_normalization": "none",
            "trainable": False,
            "pytorch_operation": "cross_correlation",
            "formula": (
                "Y[b,1,u,v] = sum_(i=0)^2 sum_(j=0)^2 "
                "K_OECT[i,j](V_G) * X[b,1,3u+i,3v+j]"
            ),
        },
        "fixed_kernels": {name: kernel.tolist() for name, kernel in KERNEL_MAP.items()},
    }


def run_smoke_test(args, device):
    smoke_batch_size = min(args.batch_size, 2)
    for backbone in args.backbones:
        for kernel_name in args.kernels:
            set_global_seed(args.seed)
            loaders = prepare_data_loaders(
                backbone,
                data_root=args.data_root,
                batch_size=smoke_batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                expected_classes=CLASS_NAMES,
            )
            model = build_model(backbone, kernel_name, pretrained=False).to(device).eval()
            inputs, _ = next(iter(loaders["train"]))
            with torch.no_grad():
                outputs = model(inputs.to(device))
            expected_shape = (inputs.shape[0], len(CLASS_NAMES))
            if tuple(outputs.shape) != expected_shape:
                raise RuntimeError(
                    f"Smoke test shape mismatch for {backbone}/{kernel_name}: "
                    f"expected {expected_shape}, got {tuple(outputs.shape)}"
                )
            print(
                f"[OK] {backbone}/{kernel_name}: input={tuple(inputs.shape)} "
                f"output={tuple(outputs.shape)}"
            )
    return 0


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    set_global_seed(args.seed)

    if args.smoke_test:
        return run_smoke_test(args, device)

    logger = ExperimentLogger(args.experiment_name, output_root=args.output_root)
    config = experiment_config(args, device)
    logger.save_config(config)
    logging.basicConfig(
        filename=str(logger.base_dir / "logs" / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
        force=True,
    )

    results = []
    failures = 0
    try:
        for backbone in args.backbones:
            for kernel_name in args.kernels:
                try:
                    set_global_seed(args.seed)
                    logging.info("Start training: backbone=%s kernel=%s", backbone, kernel_name)
                    loaders = prepare_data_loaders(
                        backbone,
                        data_root=args.data_root,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        seed=args.seed,
                        expected_classes=CLASS_NAMES,
                    )
                    model = build_model(backbone, kernel_name, pretrained=args.pretrained)
                    trainer = AdvancedTrainer(model, device=device, logger=logger)
                    checkpoint_path = logger.base_dir / "models" / f"best_{backbone}_{kernel_name}.pth"
                    run_config = {
                        **config,
                        "backbone": backbone,
                        "kernel_name": kernel_name,
                        "initial_conv_mode": model.initial_conv_mode,
                        "front_end_kind": model.front_end_kind,
                        "kernel_values": KERNEL_MAP[kernel_name].tolist()
                        if kernel_name in KERNEL_MAP
                        else None,
                    }
                    training_result = trainer.train(
                        train_loader=loaders["train"],
                        val_loader=loaders["val"],
                        backbone=backbone,
                        kernel_name=kernel_name,
                        epochs=args.epochs,
                        patience=args.patience,
                        checkpoint_path=checkpoint_path,
                        metadata=run_config,
                    )
                    logger.flush()

                    load_model_checkpoint(model, checkpoint_path, map_location=device, strict=True)
                    model.to(device).eval()
                    confusion_path = (
                        logger.base_dir / "figures" / f"confusion_matrix_{backbone}_{kernel_name}.png"
                    )
                    metrics = ResultVisualizer.analyze(
                        model,
                        loaders["test"],
                        class_names=CLASS_NAMES,
                        save_path=confusion_path,
                        device=device,
                    )
                    results.append(
                        {
                            "Backbone": backbone,
                            "ConvKernel": kernel_name,
                            "Seed": args.seed,
                            "Best_Epoch": training_result["best_epoch"],
                            "Best_Val_Accuracy": training_result["best_val_acc"] / 100.0,
                            "Accuracy": metrics["accuracy"],
                            "Precision": metrics["precision"],
                            "Recall": metrics["recall"],
                            "F1": metrics["f1"],
                        }
                    )
                    pd.DataFrame(results).to_csv(
                        logger.base_dir / "data" / "experiment_results.csv",
                        index=False,
                    )
                    ResultVisualizer.visualize_curves(logger.base_dir)
                    ResultVisualizer.visualize_bar_charts(logger.base_dir)
                except Exception as error:
                    failures += 1
                    logging.exception(
                        "Experiment failed: backbone=%s kernel=%s", backbone, kernel_name
                    )
                    print(f"[ERROR] {backbone}/{kernel_name}: {error}")
                    traceback.print_exc()
                    pd.DataFrame(results).to_csv(
                        logger.base_dir / "data" / "interrupted_results.csv",
                        index=False,
                    )
    finally:
        logger.close()

    if not results:
        raise RuntimeError(f"All {failures} experiment combinations failed; see {logger.base_dir / 'logs'}")
    if failures:
        print(f"Completed {len(results)} combinations with {failures} failure(s).")
    print(f"Experiment outputs: {logger.base_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
