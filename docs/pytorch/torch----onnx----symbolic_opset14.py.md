# `.\pytorch\torch\onnx\symbolic_opset14.py`

```
"""
This file exports ONNX ops for opset 14.

Note [ONNX operators that are added/updated in opset 14]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
    HardSwish, Trilu

Updated operators:
    Reshape
    Add, Sub, Mul, Div
    GRU, LSTM, RNN
    BatchNorm, Cumsum, Relu
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

# 导入必要的库和模块
from __future__ import annotations  # 允许使用未标记类型的函数

import functools
from typing import Optional

import torch
from torch.onnx import _constants, _type_utils, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration

__all__ = [
    "hardswish",
    "tril",
    "triu",
    "reshape",
    "batch_norm",
    "quantized_hardswish",
    "scaled_dot_product_attention",
]

# 部分函数装饰器，设置默认的 opset 版本为 14
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=14)


@_onnx_symbolic("aten::hardswish")
@symbolic_helper.parse_args("v")
@_beartype.beartype
def hardswish(g: jit_utils.GraphContext, self):
    # 将 torch 操作 "aten::hardswish" 映射为 ONNX 操作 "HardSwish"
    return g.op("HardSwish", self)


@_onnx_symbolic("aten::tril")
@_beartype.beartype
def tril(g: jit_utils.GraphContext, self, diagonal, out=None):
    # 将 torch 操作 "aten::tril" 映射为 ONNX 操作 "Trilu"，确保 lower_i=0
    return g.op("Trilu", self, diagonal, upper_i=0)


@_onnx_symbolic("aten::triu")
@_beartype.beartype
def triu(g: jit_utils.GraphContext, self, diagonal, out=None):
    # 将 torch 操作 "aten::triu" 映射为 ONNX 操作 "Trilu"，确保 upper_i=1
    return g.op("Trilu", self, diagonal, upper_i=1)


@_onnx_symbolic("aten::reshape")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "v")
@_beartype.beartype
def reshape(g: jit_utils.GraphContext, self, shape):
    # 由于 ORT 中的一个 bug https://github.com/microsoft/onnxruntime/issues/10664
    # reshape 导出时无法使用 opset 14 中引入的 allowzero 属性，因此设定 allowzero=0
    return symbolic_helper._reshape_helper(g, self, shape, allowzero=0)


@_onnx_symbolic("aten::batch_norm")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
@_beartype.beartype
def batch_norm(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    cudnn_enabled,
):
    # 如果启用自动混合精度并且输入张量的 dtype 不同，且当前 opset 版本小于 15，则报错
    if (
        torch.is_autocast_enabled()
        and not symbolic_helper.args_have_same_dtype(
            [input, weight, bias, running_mean, running_var]
        )
        and GLOBALS.export_onnx_opset_version < 15
    ):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "BatchNormalization",
            14,
            15,
            "All input tensors must have the same `dtype`."
            " Turn off Autocast or export using opset version 15.",
            input,
        )

    # 检查 batch normalization 的训练模式
    symbolic_helper.check_training_mode(training, "batch_norm")
    # 调用 batch normalization 的辅助函数，处理权重、偏置、均值、方差
    weight, bias, running_mean, running_var = symbolic_helper._batchnorm_helper(
        g, input, weight, bias, running_mean, running_var
    )
    # 调用深度学习框架的 BatchNormalization 操作
    out = g.op(
        "BatchNormalization",  # 调用 BatchNormalization 操作
        input,                  # 输入张量
        weight,                 # 权重张量
        bias,                   # 偏置张量
        running_mean,           # 当前运行时均值张量
        running_var,            # 当前运行时方差张量
        epsilon_f=eps,          # epsilon 参数，用于数值稳定性
        momentum_f=1 - momentum,  # 动量参数，用于更新运行时均值和方差
        training_mode_i=0 if not training else 1,  # 训练模式标志
        outputs=1 if not training else 3,  # 输出张量的数量，根据训练模式确定
    )
    
    # 如果不是训练模式，直接返回 BatchNormalization 的输出张量
    if not training:
        return out
    else:
        # 如果是训练模式，解包 BatchNormalization 的输出
        res, new_running_mean, new_running_var = out
        # 将新计算的运行时均值和方差张量的类型设定为与原始类型相同
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        # 返回 BatchNormalization 的输出张量
        return res
# 标注函数为quantized_hardswish，符号化为"quantized::hardswish"
# 使用Beartype装饰器对函数进行类型检查和验证
@_onnx_symbolic("quantized::hardswish")
@_beartype.beartype
def quantized_hardswish(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    # 调用dequantize_helper函数，从x中去量化信息
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    # 调用hardswish函数对x进行操作，存储结果在output中
    output = hardswish(g, x)

    # 调用quantize_helper函数，将output量化，并使用op_scale和op_zero_point进行量化参数设置
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 从https://github.com/microsoft/onnxscript/blob/6b1b81700b4523f31d8c6d3321e5d8ef5d42b764/onnxscript/function_libs/torch_aten/ops/nn.py#L1504中移植而来
# 符号化为"aten::scaled_dot_product_attention"
# 对输入参数进行解析和转换，定义了函数scaled_dot_product_attention
@_onnx_symbolic("aten::scaled_dot_product_attention")
@symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v")
@_beartype.beartype
def scaled_dot_product_attention(
    g: jit_utils.GraphContext,
    query: torch._C.Value,
    key: torch._C.Value,
    value: torch._C.Value,
    attn_mask: Optional[torch._C.Value] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[torch._C.Value] = None,
):
    # 断言确保is_causal和attn_mask不能同时设置
    assert (not is_causal) or (
        is_causal and symbolic_helper._is_none(attn_mask)
    ), "is_causal and attn_mask cannot be set at the same time"

    # 尝试获取常数scale，若不是常数则计算_attention_scale
    scale = symbolic_helper._maybe_get_const(scale, "f")
    if symbolic_helper._is_none(scale):
        scale = _attention_scale(g, query)

    # 若is_causal为True，则计算_causal_attention_mask
    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key)

    # 交换key张量的最后两个维度
    key_shape_builtin = symbolic_helper._get_tensor_rank(key)
    key_transposed_axes = list(range(key_shape_builtin))
    key_transposed_axes[-1], key_transposed_axes[-2] = (
        key_transposed_axes[-2],
        key_transposed_axes[-1],
    )
    key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)

    # 缩放q和k以便在矩阵乘法前提升稳定性，参考https://tinyurl.com/sudb9s96
    query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
    key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
    mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)

    # 若attn_mask为None，则将mul_qk_add设置为mul_qk
    if symbolic_helper._is_none(attn_mask):
        mul_qk_add = mul_qk
    # 若attn_mask为布尔类型，则将布尔掩码转换为浮点数，实现mask填充
    elif (
        _type_utils.JitScalarType.from_value(attn_mask)
        == _type_utils.JitScalarType.BOOL
    ):
        const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
        const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
        attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    # 若attn_mask为浮点数类型或者半精度类型，则直接进行Add操作
    elif _type_utils.JitScalarType.from_value(attn_mask) in (
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.BFLOAT16,
    ):
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    else:
        raise ValueError(
            f"Unsupported type for attn_mask: {_type_utils.JitScalarType.from_value(attn_mask)}"
        )


    # 如果存在注意力掩码，则将注意力矩阵和掩码相加
    mul_qk_add = g.op("Add", mul_qk, attn_mask)



    attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)


    # 对相加后的注意力矩阵执行 softmax 操作，以获取注意力权重
    attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)



    if dropout_p != 0:
        attn_weight = g.op(
            "Dropout",
            attn_weight,
            g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
        )


    # 如果设置了非零的 dropout 概率，则对注意力权重应用 dropout
    attn_weight = g.op(
        "Dropout",
        attn_weight,
        g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
    )



    return g.op("MatMul", attn_weight, value)


    # 返回注意力权重矩阵与值矩阵的乘积结果
    return g.op("MatMul", attn_weight, value)
@_beartype.beartype
def _attention_scale(
    g: jit_utils.GraphContext, query: torch._C.Value
) -> torch._C.Value:
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    # 获取 query 张量的形状
    query_shape = g.op("Shape", query)
    # 获取 query 张量的最后一个维度长度
    query_shape_last = g.op(
        "Slice",
        query_shape,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
        g.op(
            "Constant", value_t=torch.tensor([_constants.INT64_MAX], dtype=torch.int64)
        ),
    )
    # 获取 query 张量的最后一个维度长度，并转换为与 query 相同的数据类型
    embedding_size = g.op(
        "Cast",
        query_shape_last,
        to_i=_type_utils.JitScalarType.from_value(query).onnx_type(),
    )
    # 创建常数张量值为 1.0 的标量
    const_one = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float))
    # 计算 scale = 1 / math.sqrt(query.size(-1))
    scale = g.op("Div", const_one, g.op("Sqrt", embedding_size))
    # 将 scale 转换回原始类型
    scale = g.op(
        "Cast",
        scale,
        to_i=_type_utils.JitScalarType.from_value(query).onnx_type(),
    )
    return scale


@_beartype.beartype
def _causal_attention_mask(
    g: jit_utils.GraphContext, query: torch._C.Value, key: torch._C.Value
) -> torch._C.Value:
    """Create a causal mask for the given query and key tensors.

    Equivalent to::
        mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = torch.zeros(L, S, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(not mask, -float('inf'))

    Args:
        query: Tensor of shape [..., L, E]
        key: Tensor of shape [..., S, E]

    Returns:
        Tensor of shape [L, S]
    """

    # 获取 query 和 key 张量的形状
    query_shape = g.op("Shape", query)
    key_shape = g.op("Shape", key)

    # 获取最后两个维度的长度
    last_idx = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    second_last_idx = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.int64))
    target_length = g.op("Slice", query_shape, second_last_idx, last_idx)
    source_length = g.op("Slice", key_shape, second_last_idx, last_idx)
    
    # 构建 mask = torch.ones(L, S) 的形状
    size = g.op("Concat", target_length, source_length, axis_i=0)
    const_one = g.op("Constant", value_t=torch.tensor([1.0]))
    attn_mask = g.op("Expand", const_one, size)
    
    # 将 mask 转换为上三角矩阵
    attn_mask = g.op("Trilu", attn_mask, upper_i=0)
    
    # 构建 causal mask，下三角为 0，上三角为 -inf
    const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
    const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
    attn_mask = g.op(
        "Where", g.op("Equal", attn_mask, const_zero), const_neg_inf, const_zero
    )
    
    return attn_mask
```