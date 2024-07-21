# `.\pytorch\torch\onnx\symbolic_opset20.py`

```py
"""
This file exports ONNX ops for opset 20.

Note [ONNX Operators that are added/updated in opset 20]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-20-of-the-default-onnx-operator-set
New operators:
    AffineGrid
    ConstantOfShape
    DFT
    Gelu
    GridSample
    ImageDecoder
    IsInf
    IsNaN
    ReduceMax
    ReduceMin
    RegexFullMatch
    StringConcat
    StringSplit
"""

import functools  # 导入 functools 模块

import torch.nn.functional as F  # 导入 PyTorch 的 functional 模块

from torch import _C  # 导入 PyTorch 核心模块 _C
from torch.onnx import symbolic_helper  # 导入 PyTorch 的 ONNX 符号帮助模块
from torch.onnx._internal import _beartype, jit_utils, registration  # 导入 PyTorch 的内部模块

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = ["_grid_sampler", "_affine_grid_generator", "gelu"]  # 导出的符号列表


def convert_grid_sample_mode(mode_s):
    return (
        "linear" if mode_s == "bilinear" else "cubic" if mode_s == "bicubic" else mode_s
    )  # 根据输入的 mode_s 转换为对应的字符串，用于 grid sample 操作的插值模式


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=20)  # 使用 opset=20 部分应用注册 ONNX 符号函数


@_onnx_symbolic("aten::grid_sampler")
@symbolic_helper.parse_args("v", "v", "i", "i", "b")
@_beartype.beartype
def _grid_sampler(
    g: jit_utils.GraphContext,
    input: _C.Value,
    grid: _C.Value,
    mode_enum: int,
    padding_mode_enum: int,
    align_corners: bool,
):
    mode_s = {v: k for k, v in F.GRID_SAMPLE_INTERPOLATION_MODES.items()}[mode_enum]  # type: ignore[call-arg, index]
    # 根据 mode_enum 解析出对应的插值模式字符串，参考 GRID_SAMPLE_INTERPOLATION_MODES
    # 更改的模式字符串详见 https://onnx.ai/onnx/operators/text_diff_GridSample_16_20.html
    mode_s = convert_grid_sample_mode(mode_s)  # 转换为 ONNX 支持的插值模式字符串
    padding_mode_s = {v: k for k, v in F.GRID_SAMPLE_PADDING_MODES.items()}[padding_mode_enum]  # type: ignore[call-arg, index]
    # 根据 padding_mode_enum 解析出对应的填充模式字符串，参考 GRID_SAMPLE_PADDING_MODES
    return g.op(
        "GridSample",
        input,
        grid,
        align_corners_i=int(align_corners),
        mode_s=mode_s,
        padding_mode_s=padding_mode_s,
    )  # 在图中添加 GridSample 操作节点


@_onnx_symbolic("aten::affine_grid_generator")
@symbolic_helper.parse_args("v", "v", "b")
@_beartype.beartype
def _affine_grid_generator(
    g: jit_utils.GraphContext,
    theta: _C.Value,
    size: _C.Value,
    align_corners: bool,
):
    return g.op(
        "AffineGrid",
        theta,
        size,
        align_corners_i=int(align_corners),
    )  # 在图中添加 AffineGrid 操作节点


@_onnx_symbolic("aten::gelu")
@symbolic_helper.parse_args("v", "s")
@_beartype.beartype
def gelu(g: jit_utils.GraphContext, self: _C.Value, approximate: str = "none"):
    return g.op("Gelu", self, approximate_s=approximate)  # 在图中添加 Gelu 操作节点
```