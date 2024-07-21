# `.\pytorch\torch\onnx\symbolic_opset18.py`

```
# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 18.

Note [ONNX Operators that are added/updated in opset 18]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-18-of-the-default-onnx-operator-set
New operators:
    BitwiseAnd
    CenterCropPad
    Col2Im
    Mish
    OptionalGetElement
    OptionalHasElement
    Pad
    Resize
    ScatterElements
    ScatterND
    Split
"""

import functools
from typing import List, Optional, Sequence, Tuple

import torch
from torch import _C
from torch.onnx import _type_utils, symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import _beartype, jit_utils, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

__all__ = [
    "col2im",
]

# Partial function to register ONNX symbolic functions with opset version 18
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=18)


# Define ONNX symbolic function for bitwise AND operation
@_onnx_symbolic("aten::__and_")
@_onnx_symbolic("aten::bitwise_and")
@_beartype.beartype
def __and_(g: jit_utils.GraphContext, self, other):
    # Perform type promotion based on input types
    args = [self, other]
    
    # Collect arguments requiring type promotion based on tensor rank
    prom_args = [arg for arg in args if symbolic_helper._get_tensor_rank(arg)]
    if len(prom_args) == 0:
        prom_args = args
    
    # Determine the promoted type
    promotion_jit_type = symbolic_helper._type_promote_from_values(*prom_args)
    
    # Cast inputs to the promoted type if necessary
    self = symbolic_helper._maybe_cast_to_type(g, self, promotion_jit_type)
    other = symbolic_helper._maybe_cast_to_type(g, other, promotion_jit_type)
    
    # Select ONNX operation based on the promoted type
    if promotion_jit_type == _type_utils.JitScalarType.BOOL:
        return g.op("And", self, other)
    return g.op("BitwiseAnd", self, other)


# Define ONNX symbolic function for col2im operation
@_onnx_symbolic("aten::col2im")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is")
@_beartype.beartype
def col2im(
    g,
    input: _C.Value,
    output_size: _C.Value,
    kernel_size: _C.Value,
    dilation: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
):
    # Adjust padding format from [p0, p1, ..., pn] to [p0, p0, p1, p1, ..., pn, pn]
    adjusted_padding = []
    for pad in padding:
        for _ in range(2):
            adjusted_padding.append(pad)
    
    # Determine the number of dimensions in the output size tensor
    num_dimensional_axis = symbolic_helper._get_tensor_sizes(output_size)[0]
    
    # Ensure adjusted_padding is populated if empty
    if not adjusted_padding:
        adjusted_padding = [0, 0] * num_dimensional_axis
    
    # Ensure dilation is populated if empty
    if not dilation:
        dilation = [1] * num_dimensional_axis
    
    # Ensure stride is populated if empty
    if not stride:
        stride = [1] * num_dimensional_axis
    
    # Generate ONNX operation for col2im with adjusted parameters
    return g.op(
        "Col2Im",
        input,
        output_size,
        kernel_size,
        dilations_i=dilation,
        pads_i=adjusted_padding,
        strides_i=stride,
    )
# 定义一个函数 `_reduce_with_dtype`，用于辅助特定的降维操作，根据数据类型进行处理
def _reduce_with_dtype(onnx_op: str, name: str, allow_multi_dim_support: bool = True):
    # 调用 symbolic_helper 模块中的 _reduce_with_dtype_helper 函数来完成具体的降维操作
    return symbolic_helper._reduce_with_dtype_helper(
        onnx_op, name, allow_multi_dim_support
    )


# 标记为 ONNX 符号化操作 "_onnx_symbolic"，表示下面的函数与 ONNX 模型的符号化相关联
# 标记为 quantized_args 表示这个函数支持量化参数
# 标记为 parse_args 表示这个函数需要解析一些参数
# 使用 beartype 进行函数参数类型的验证
# 定义了一个函数 "_native_layer_norm"，用于执行原生的 layer normalization 操作
def _native_layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
) -> Tuple[_C.Value, _C.Value, _C.Value]:
    # 调用 opset9 模块中的 native_layer_norm 函数，执行具体的 layer normalization 操作
    return opset9.native_layer_norm(g, input, normalized_shape, weight, bias, eps)


# 标记为 ONNX 符号化操作 "_onnx_symbolic"，表示下面的函数与 ONNX 模型的符号化相关联
# 使用 beartype 进行函数参数类型的验证
# 定义了一个函数 "_glu"，用于执行 gated linear unit (GLU) 操作
def _glu(g: jit_utils.GraphContext, input, dim):
    # 调用 symbolic_helper 模块中的 _get_tensor_dim_size 函数，获取指定维度的张量大小
    dim_size = symbolic_helper._get_tensor_dim_size(input, dim)
    # 如果 dim_size 不为空，则断言其为偶数
    if dim_size is not None:
        assert dim_size % 2 == 0

    # 使用图操作 "Split" 对输入张量进行分割，得到两个部分
    first, second = g.op("Split", input, axis_i=dim, num_outputs_i=2, outputs=2)
    # 返回使用图操作 "Mul" 和 "Sigmoid" 处理后的结果
    return g.op("Mul", first, g.op("Sigmoid", second))


# 标记为 ONNX 符号化操作 "_onnx_symbolic"，表示下面的函数与 ONNX 模型的符号化相关联
# 使用 beartype 进行函数参数类型的验证
# 定义了一个函数 "max"，用于执行最大值操作
def max(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # 调用 symbolic_helper 模块中的 _max_helper 函数，执行具体的最大值操作
    return symbolic_helper._max_helper(g, self, dim_or_y, keepdim)


# 标记为 ONNX 符号化操作 "_onnx_symbolic"，表示下面的函数与 ONNX 模型的符号化相关联
# 使用 quantized_args 标记这个函数支持量化参数
# 使用 beartype 进行函数参数类型的验证
# 定义了一个函数 "maximum"，用于执行元素级的最大值操作
def maximum(g: jit_utils.GraphContext, input, other):
    # 调用 max 函数，传递参数 other 作为 dim_or_y 参数，执行具体的最大值操作
    return max(g, input, dim_or_y=other)


# 标记为 ONNX 符号化操作 "_onnx_symbolic"，表示下面的函数与 ONNX 模型的符号化相关联
# 使用 beartype 进行函数参数类型的验证
# 定义了一个函数 "min"，用于执行最小值操作
def min(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # 调用 symbolic_helper 模块中的 _min_helper 函数，执行具体的最小值操作
    return symbolic_helper._min_helper(g, self, dim_or_y, keepdim)


# 标记为 ONNX 符号化操作 "_onnx_symbolic"，表示下面的函数与 ONNX 模型的符号化相关联
# 使用 quantized_args 标记这个函数支持量化参数
# 使用 beartype 进行函数参数类型的验证
# 定义了一个函数 "minimum"，用于执行元素级的最小值操作
def minimum(g: jit_utils.GraphContext, input, other):
    # 调用 min 函数，传递参数 other 作为 dim_or_y 参数，执行具体的最小值操作
    return min(g, input, dim_or_y=other)


# 标记为 ONNX 符号化操作 "_onnx_symbolic"，表示下面的函数与 ONNX 模型的符号化相关联
# 使用 quantized_args 标记这个函数支持量化参数
# 使用 parse_args 标记这个函数需要解析一些参数
# 使用 beartype 进行函数参数类型的验证
# 定义了一个函数 "amax"，用于执行元素级的最大值操作（张量沿指定轴的最大值）
def amax(g: jit_utils.GraphContext, self, dim, keepdim):
    # 创建一个表示轴的常量张量
    axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
    # 使用图操作 "ReduceMax" 对输入张量进行沿指定轴的最大值操作
    return g.op("ReduceMax", self, axes, keepdims_i=keepdim)


# 标记为 ONNX 符号化操作 "_onnx_symbolic"，表示下面的函数与 ONNX 模型的符号化相关联
# 使用 quantized_args 标记这个函数支持量化参数
# 使用 parse_args 标记这个函数需要解析一些参数
# 使用 beartype 进行函数参数类型的验证
# 定义了一个函数 "amin"，用于执行元素级的最小值操作（张量沿指定轴的最小值）
def amin(g: jit_utils.GraphContext, self, dim, keepdim):
    # 创建一个表示轴的常量张量
    axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
    # 使用图操作 "ReduceMin" 对输入张量进行沿指定轴的最小值操作
    return g.op("ReduceMin", self, axes, keepdims_i=keepdim)


# 标记为 ONNX 符号化操作 "_onnx_symbolic"，表示下面的函数与 ONNX 模型的符号化相关联
# 使用 quantized_args 标记这个函数支持量化参数
# 使用 parse_args 标记这个函数需要解析一些参数
# 使用 beartype 进行函数参数类型的验证
# 定义了一个函数 "aminmax"，用于执行元素级的最小和最大值操作（张量沿指定轴的最小和最大值）
def aminmax(g: jit_utils.GraphContext, self, dim, keepdim):
    # 如果维度参数不为None，则将其转换为常量，并创建一个包含该维度的张量常量
    if not symbolic_helper._is_none(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")
        axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
        # 返回张量self在指定维度上的最小值和最大值的操作
        return g.op("ReduceMin", self, axes, keepdims_i=keepdim), g.op(
            "ReduceMax", self, axes, keepdims_i=keepdim
        )
    else:
        # 如果维度参数为None，则返回张量self在所有维度上的最小值和最大值的操作
        return g.op("ReduceMin", self, keepdims_i=keepdim), g.op(
            "ReduceMax", self, keepdims_i=keepdim
        )
# 注册 ONNX 符号化处理函数，处理 "aten::var_mean" 操作
# 使用 Beartype 进行类型检查和注解
@_onnx_symbolic("aten::var_mean")
@_beartype.beartype
def _var_mean(g: jit_utils.GraphContext, input, *args):
    # 如果参数个数为1，则调用帮助函数处理 var_mean 操作
    if len(args) == 1:
        return symbolic_helper._var_mean_helper(g, input, None, args[0], None)
    else:
        # 否则，调用帮助函数处理 var_mean 操作，传递所有参数
        return symbolic_helper._var_mean_helper(g, input, *args)


# 注册 ONNX 符号化处理函数，处理 "aten::logsumexp" 操作
# 使用 parse_args 解析参数列表的类型信息
# 使用 Beartype 进行类型检查和注解
@_onnx_symbolic("aten::logsumexp")
@symbolic_helper.parse_args("v", "is", "i")
@_beartype.beartype
def _logsumexp(g: jit_utils.GraphContext, input, dim, keepdim):
    # 如果 dim 为 None，则使用 ReduceLogSumExp 操作对 input 进行降维求和
    if dim is None:
        return g.op("ReduceLogSumExp", input, keepdims_i=0)
    else:
        # 否则，创建包含维度信息的常量张量 axes，并使用 ReduceLogSumExp 操作
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
        return g.op("ReduceLogSumExp", input, axes, keepdims_i=keepdim)


# 注册 ONNX 符号化处理函数，处理 "aten::linalg_matrix_norm" 操作
# 使用 parse_args 解析参数列表的类型信息
# 使用 Beartype 进行类型检查和注解
def _linalg_matrix_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: List[int],
    keepdim: bool,
    dtype: torch._C.Value,
):
    # 调用 opset9 中的 linalg_matrix_norm 函数处理矩阵范数计算
    return opset9.linalg_matrix_norm(g, self, ord, dim, keepdim, dtype)


# 注册 ONNX 符号化处理函数，处理 "aten::embedding_bag" 操作
# 使用 parse_args 解析参数列表的类型信息
# 使用 Beartype 进行类型检查和注解
def embedding_bag(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    # 调用帮助函数处理 embedding_bag 操作
    return symbolic_helper._embedding_bag_helper(
        g,
        embedding_matrix,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )


# 注册 ONNX 符号化处理函数，处理 "aten::linalg_vector_norm" 操作
# 使用 parse_args 解析参数列表的类型信息
# 使用 Beartype 进行类型检查和注解
def linalg_vector_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: float,
    dim: Optional[Sequence[int]],
    keepdim: bool,
    dtype: torch._C.Value,
):
    # 调用帮助函数处理 linalg_vector_norm 操作
    return symbolic_helper._linalg_vector_norm_helper(g, self, ord, dim, keepdim, dtype)
```