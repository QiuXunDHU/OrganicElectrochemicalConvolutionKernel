"""Photoresponsive OECT measured-response convolution front end."""

import math

import torch
import torch.nn as nn


class OECTFrontEnd(nn.Conv2d):
    """Apply a measured 3x3 photoresponsive OECT response as a fixed operator.

    PyTorch ``Conv2d`` evaluates the discrete cross-correlation

        Y[b, 1, u, v] = sum_i sum_j K_OECT[i, j](V_G)
                         * X[b, 1, s*u + i, s*v + j],

    for ``i, j in {0, 1, 2}`` and ``s = 3``. The measured response is used
    directly: no bias or kernel normalization is applied, and its weight is
    frozen. Subclassing ``nn.Conv2d`` deliberately preserves the checkpoint
    key ``initial_conv.weight`` used by earlier project versions.
    """

    def __init__(self, response_kernel, gate_voltage, response_source):
        try:
            gate_voltage = float(gate_voltage)
        except (TypeError, ValueError) as exc:
            raise ValueError("gate_voltage must be a finite number") from exc
        if not math.isfinite(gate_voltage):
            raise ValueError("gate_voltage must be a finite number")
        if response_source is None or not str(response_source).strip():
            raise ValueError("response_source must identify the measured OECT response")

        kernel_tensor = torch.as_tensor(response_kernel, dtype=torch.float32).detach()
        if kernel_tensor.shape != (3, 3):
            raise ValueError(
                "OECT response kernel must have shape (3, 3), "
                f"got {tuple(kernel_tensor.shape)}"
            )
        if not torch.isfinite(kernel_tensor).all():
            raise ValueError("OECT response kernel contains non-finite values")

        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=3,
            padding=0,
            bias=False,
        )
        with torch.no_grad():
            self.weight.copy_(kernel_tensor.reshape(1, 1, 3, 3).to(self.weight.device))
        self.weight.requires_grad_(False)

        self.gate_voltage = gate_voltage
        self.response_source = str(response_source)
        self.photoresponsive = True
        self.kernel_normalization = "none"
        self.operation = "cross_correlation"
