"""The residual policy/value convolutional network."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from chessy.encoding import ACTION_SIZE
from chessy.model.config import ModelConfig


class PolicyValueOutput(NamedTuple):
    """Raw policy and loss/draw/win logits returned by :class:`ChessyModel`."""

    policy_logits: torch.Tensor
    value_logits: torch.Tensor


class ResidualBlock(nn.Module):
    """Two normalized 3x3 convolutions with an identity skip connection."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.relu1(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return self.relu2(outputs + residual)


class ChessyModel(nn.Module):
    """``residual-cnn-v1`` producing unmasked policy and WDL value logits."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.stem_conv = nn.Conv2d(
            config.input_planes, config.channels, kernel_size=3, padding=1, bias=False
        )
        self.stem_norm = nn.GroupNorm(config.group_norm_groups, config.channels)
        self.stem_relu = nn.ReLU(inplace=False)
        self.residual_blocks = nn.ModuleList(
            ResidualBlock(config.channels, config.group_norm_groups)
            for _ in range(config.residual_blocks)
        )
        self.policy_conv = nn.Conv2d(
            config.channels, config.action_planes, kernel_size=1, bias=True
        )
        self.value_conv = nn.Conv2d(config.channels, config.value_channels, kernel_size=1, bias=False)
        self.value_norm = nn.GroupNorm(config.group_norm_groups, config.value_channels)
        self.value_relu1 = nn.ReLU(inplace=False)
        self.value_linear1 = nn.Linear(
            config.value_channels * config.board_size * config.board_size, config.value_hidden
        )
        self.value_relu2 = nn.ReLU(inplace=False)
        self.value_linear2 = nn.Linear(config.value_hidden, config.value_classes)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, boards: torch.Tensor) -> PolicyValueOutput:
        """Evaluate a float32 ``[batch, 119, 8, 8]`` board tensor."""
        if not isinstance(boards, torch.Tensor):
            raise ValueError("boards must be a torch.Tensor")
        expected_shape = (self.config.input_planes, self.config.board_size, self.config.board_size)
        if boards.ndim != 4:
            raise ValueError(f"boards must have shape [B, {expected_shape[0]}, 8, 8]")
        if tuple(boards.shape[1:]) != expected_shape:
            raise ValueError(
                f"boards must have shape [B, {expected_shape[0]}, "
                f"{expected_shape[1]}, {expected_shape[2]}], got {tuple(boards.shape)}"
            )
        if boards.dtype != torch.float32:
            raise ValueError(f"boards must have dtype torch.float32, got {boards.dtype}")

        features = self.stem_relu(self.stem_norm(self.stem_conv(boards)))
        for block in self.residual_blocks:
            features = block(features)
        policy_logits = self.policy_conv(features).flatten(start_dim=1)
        if policy_logits.shape[1] != ACTION_SIZE:
            raise RuntimeError("policy head output is incompatible with az73-v1")
        value_features = self.value_relu1(self.value_norm(self.value_conv(features)))
        value_features = value_features.flatten(start_dim=1)
        value_logits = self.value_linear2(self.value_relu2(self.value_linear1(value_features)))
        return PolicyValueOutput(policy_logits=policy_logits, value_logits=value_logits)
