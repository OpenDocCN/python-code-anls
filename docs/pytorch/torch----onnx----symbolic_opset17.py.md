# `.\pytorch\torch\onnx\symbolic_opset17.py`

```py
# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 17.

Note [ONNX Operators that are added/updated in opset 17]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-17-of-the-default-onnx-operator-set
New operators:
    BlackmanWindow
    DFT
    HammingWindow
    HannWindow
    LayerNormalization
    MelWeightMatrix
    STFT
    SequenceMap
"""

import functools
from typing import Optional, Sequence

import torch
from torch import _C
from torch.onnx import _type_utils, errors, symbolic_helper
from torch.onnx._internal import _beartype, jit_utils, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = ["layer_norm", "stft", "quantized_layer_norm"]

# Partial function application to set opset=17 for ONNX symbolic functions
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=17)


# ONNX symbolic function for `aten::layer_norm`
@_onnx_symbolic("aten::layer_norm")
@symbolic_helper.parse_args("v", "is", "v", "v", "f", "none")
def layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
    cudnn_enable: bool,
):
    # Calculate the axis for normalization based on the last D dimensions
    axis = -len(normalized_shape)
    # Determine the scalar type of the input tensor
    scalar_type = _type_utils.JitScalarType.from_value(
        input, _type_utils.JitScalarType.FLOAT
    )
    # Get the dtype from the scalar type
    dtype = scalar_type.dtype()
    
    # If weight is not provided, create a tensor of ones
    if symbolic_helper._is_none(weight):
        weight_value = torch.ones(normalized_shape, dtype=dtype)
        weight = g.op("Constant", value_t=weight_value)
    
    # If bias is not provided, create a tensor of zeros
    if symbolic_helper._is_none(bias):
        bias_value = torch.zeros(normalized_shape, dtype=dtype)
        bias = g.op("Constant", value_t=bias_value)
    
    # Generate the ONNX operation for LayerNormalization
    return g.op(
        "LayerNormalization",
        input,
        weight,
        bias,
        epsilon_f=eps,
        axis_i=axis,
    )


# ONNX symbolic function for `quantized::layer_norm`
@_onnx_symbolic("quantized::layer_norm")
def quantized_layer_norm(
    g: jit_utils.GraphContext,
    x,
    normalized_shape,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    # Dequantize the input tensor `x` using helper function
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    
    # Apply layer normalization using the previously defined function
    output = layer_norm(g, x, normalized_shape, weight, bias, eps, False)
    
    # Quantize the output tensor using helper function
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# Helper function to compute the sizes of left and right edges of a window
def _compute_edge_sizes(n_fft, window_size):
    """Helper function to compute the sizes of the edges (left and right)
    of a given window centered within an FFT size."""
    left = (n_fft - window_size) // 2
    right = n_fft - left - window_size
    return left, right


# ONNX symbolic function for `aten::stft`
@_onnx_symbolic("aten::stft")
@symbolic_helper.parse_args("v", "i", "i", "i", "v", "b", "b", "b")
@_beartype.beartype
def stft(
    g: jit_utils.GraphContext,
    input: _C.Value,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
):
    # Placeholder for STFT symbolic function implementation
    pass
    window: Optional[_C.Value] = None,
    # 定义一个可选类型的变量 window，初始赋值为 None，类型为 _C.Value

    normalized: bool = False,
    # 定义一个布尔类型的变量 normalized，初始赋值为 False

    onesided: Optional[bool] = True,
    # 定义一个可选类型的布尔变量 onesided，初始赋值为 True

    return_complex: Optional[bool] = False,
    # 定义一个可选类型的布尔变量 return_complex，初始赋值为 False
# 定义一个函数，将 `torch.stft` 关联到 `STFT` 的 ONNX 操作符
# 注意，torch.stft 调用 _VF.stft，不包含居中或填充选项
# 因此，此函数不包含这两个参数。详见 torch.stft 源代码了解更多信息。
def associate_stft_op(g: Graph, input: _C.Value, n_fft: int, hop_length: Optional[int] = None,
                      win_length: Optional[int] = None, window: Optional[_C.Value] = None,
                      normalized: bool = False, onesided: bool = True, return_complex: Optional[bool] = False) -> _C.Value:
    """
    Associates `torch.stft` with the `STFT` ONNX operator.
    Note that torch.stft calls _VF.stft, without centering or padding options.
    Hence, this function does not contain these two arguments.
    See torch.stft source code for more info.

    Args:
        g: Graph to write the ONNX representation into
        input: Input tensor for the transformation
        n_fft: FFT size
        hop_length: Size of the hop. Defaults to `floot(n_fft // 4)`
        win_length: Size of the analysis window. Defaults to `n_fft`
        window: Analysis window. Defaults to a window of all ones
        normalized: Whether to return a normalized STFT
        onesided: Whether to return only half (+1) of the results, given the
            symmetry of the STFT
        return_complex: Whether to return the complex value (Note: Must be
            `False` or `None`)

    Returns:
        op: Operator for torch.stft associated with STFT (ONNX)
    """

    # 检查是否需要返回复数类型的结果，如果是则抛出错误
    if return_complex:
        raise errors.SymbolicValueError(
            msg="STFT does not currently support complex types", value=input
        )

    # 获取 STFT 的尺寸参数
    frame_step_value = hop_length if hop_length is not None else n_fft // 4
    frame_step_const = g.op(
        "Constant", value_t=torch.tensor(frame_step_value, dtype=torch.int64)
    )
    frame_length_const = g.op(
        "Constant", value_t=torch.tensor(n_fft, dtype=torch.int64)
    )

    # 如果输入信号是一维的，则添加一个批处理维度
    signal = input
    signal_rank = symbolic_helper._get_tensor_rank(signal)
    if signal_rank == 1:
        signal = g.op(
            "Unsqueeze",
            signal,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )
    elif signal_rank > 2:
        # 如果信号的维度大于 2，则抛出错误
        raise errors.SymbolicValueError(
            msg="STFT can only take inputs of 1 [signal] or 2 [batch, signal] dimensions. "
            f"Current rank of signal is {signal_rank}, please reduce it.",
            value=input,
        )

    # 获取分析窗口 `window`，并确保其大小与 `win_length` 或 `n_fft` 相同
    n_win = symbolic_helper._get_tensor_dim_size(window, dim=0)
    if n_win is not None:
        win_length_default = win_length if win_length else n_fft
        assert n_win == win_length_default, (
            "Analysis window size must equal `win_length` or `n_fft`. "
            f"Please, set `win_length` or `n_fft` to match `window` size ({n_win})",
        )

        # 如果窗口大小小于 FFT 大小，则在左右填充零（ONNX 的 STFT 要求）
        if n_win < n_fft:
            left, right = _compute_edge_sizes(n_fft, n_win)
            left_win = g.op("Constant", value_t=torch.zeros(left))
            right_win = g.op("Constant", value_t=torch.zeros(right))
            window = g.op("Concat", left_win, window, right_win, axis_i=0)

    # 如果需要的话，创建窗口
    # 如果窗口为 None，则进行处理
    if symbolic_helper._is_none(window):
        # 如果指定了 win_length
        if win_length:
            # 如果 win_length 大于 n_fft，则抛出错误
            if win_length > n_fft:
                raise errors.SymbolicValueError(
                    msg="The analysis window can't be longer than the size of the FFT. "
                    f"Please set `win_length` ({win_length}) to `n_fft` ({n_fft}) or less.",
                    value=input,
                )

            # 计算左右边界大小以进行居中处理
            left, right = _compute_edge_sizes(n_fft, win_length)
            # 创建一个居中的 torch 窗口
            torch_window = torch.hstack(
                (torch.zeros(left), torch.ones(win_length), torch.zeros(right))
            )
        else:
            # 如果未指定 win_length，则使用矩形窗口（全为1）
            torch_window = torch.ones(n_fft)
        
        # 确保 torch_window 的长度与 n_fft 相符
        assert torch_window.shape[0] == n_fft
        
        # 将 torch_window 转换为 ONNX 张量常量，用于后续的图操作
        window = g.op("Constant", value_t=torch_window)
    
    # 将 window 转换为指定类型的 ONNX 张量
    window = g.op(
        "Cast", window, to_i=_type_utils.JitScalarType.from_value(signal).onnx_type()
    )

    # 运行 STFT
    result = g.op(
        "STFT",
        signal,
        frame_step_const,
        window,
        frame_length_const,
        onesided_i=1 if onesided is None or onesided else 0,
    )

    # 转置以模仿 torch.stft 的行为
    result = g.op("Transpose", result, perm_i=[0, 2, 1, 3])

    # 如果信号的秩为1，则移除批次维度
    if signal_rank == 1:
        result = g.op(
            "Squeeze",
            result,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    # 如果需要归一化
    if normalized:
        # 计算归一化系数 sqrt(n_fft)
        sqrt_nfft = torch.sqrt(torch.tensor(n_fft, dtype=signal.type().dtype()))
        # 对结果进行归一化操作
        result = g.op("Div", result, g.op("Constant", value_t=sqrt_nfft))

    # 返回最终结果
    return result
```