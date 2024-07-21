# `.\pytorch\torch\onnx\symbolic_opset12.py`

```
# mypy: allow-untyped-defs
from __future__ import annotations

# 导入必要的模块和库
import functools
import sys
from typing import Optional, Tuple

import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import (
    _type_utils,
    errors,
    symbolic_helper,
    symbolic_opset9 as opset9,
    utils,
)
from torch.onnx._internal import _beartype, jit_utils, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

# This file exports ONNX ops for opset 12

__all__ = [
    "argmax",
    "argmin",
    "binary_cross_entropy_with_logits",
    "celu",
    "cross_entropy_loss",
    "dropout",
    "einsum",
    "ge",
    "le",
    "native_dropout",
    "nll_loss",
    "nll_loss2d",
    "nll_loss_nd",
    "outer",
    "pow",
    "tensordot",
    "unfold",
]

# Partial function for registering ONNX symbolic functions for opset 12
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=12)


# Helper function for handling einsum operations
@_beartype.beartype
def _einsum_helper(g: jit_utils.GraphContext, equation, tensors):
    # Raise error if einsum inputs are empty
    if not tensors:
        raise RuntimeError("Einsum inputs are empty.")
    # ONNX does not support bool for Einsum inputs, so cast them to INT64 if necessary
    if symbolic_helper._is_bool(tensors[0]):
        tensors = [
            g.op("Cast", tensor, to_i=_C_onnx.TensorProtoDataType.INT64)
            for tensor in tensors
        ]
        return g.op(
            "Cast",
            g.op("Einsum", *tensors, equation_s=equation),
            to_i=_C_onnx.TensorProtoDataType.BOOL,
        )
    else:
        return g.op("Einsum", *tensors, equation_s=equation)


# Symbolic function decorator for einsum operation
@_onnx_symbolic("aten::einsum")
@symbolic_helper.parse_args("s", "v", "is")
@_beartype.beartype
def einsum(g: jit_utils.GraphContext, equation, tensor_list, path=None):
    tensors = symbolic_helper._unpack_list(tensor_list)
    return _einsum_helper(g, equation, tensors)


# Symbolic function decorator for outer product operation
@_onnx_symbolic("aten::outer")
@symbolic_helper.parse_args("v", "v")
@_beartype.beartype
def outer(g: jit_utils.GraphContext, input, other):
    # Ensure other is cast to the same type as input
    if _type_utils.JitScalarType.from_value(
        other, _type_utils.JitScalarType.UNDEFINED
    ) != _type_utils.JitScalarType.from_value(input):
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_value(input).onnx_type(),
        )
    return _einsum_helper(g, "i,j->ij", [input, other])


# Helper function for dropout operation that returns both masked input and mask
@_beartype.beartype
def _dropout_returns_masked_input_and_mask(
    g: jit_utils.GraphContext, input: torch._C.Value, p: float, train: bool
) -> Tuple[torch._C.Value, Optional[torch._C.Value]]:
    symbolic_helper.check_training_mode(train, "dropout")
    # In eval mode, dropout is a no-op, returning input and None for mask
    if not train:
        return input, None
    p = g.op("Constant", value_t=torch.tensor(p))
    t = g.op("Constant", value_t=torch.tensor(train, dtype=torch.bool))
    r, mask = g.op("Dropout", input, p, t, outputs=2)
    return r, mask


# Symbolic function decorator for dropout operation
@_onnx_symbolic("aten::dropout")
# 使用装饰器将函数标记为需要解析参数的辅助函数，参数包括 "v", "f", "b"
@symbolic_helper.parse_args("v", "f", "b")
# 使用装饰器将函数标记为需要类型检查的函数
@_beartype.beartype
# 定义dropout函数，对输入数据进行dropout操作
def dropout(g: jit_utils.GraphContext, input, p, train):
    # 调用_dropout_returns_masked_input_and_mask函数获取经过掩码处理的输入数据
    masked, _ = _dropout_returns_masked_input_and_mask(g, input, p, train)
    # 返回掩码处理后的输入数据
    return masked


# 使用ONNX符号化装饰器，指定对应的ATen操作为native_dropout
@_onnx_symbolic("aten::native_dropout")
# 使用装饰器将函数标记为需要解析参数的辅助函数，参数包括 "v", "f", "b"
@symbolic_helper.parse_args("v", "f", "b")
# 使用装饰器将函数标记为需要类型检查的函数
@_beartype.beartype
# 定义native_dropout函数，对输入数据进行native_dropout操作
def native_dropout(g: jit_utils.GraphContext, input, p, train):
    # 直接调用_dropout_returns_masked_input_and_mask函数返回其结果
    return _dropout_returns_masked_input_and_mask(g, input, p, train)


# 使用ONNX符号化装饰器，指定对应的ATen操作为nll_loss
@_onnx_symbolic("aten::nll_loss")
# 使用装饰器将函数标记为需要类型检查的函数
@_beartype.beartype
# 定义nll_loss函数，计算负对数似然损失
def nll_loss(g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index):
    # 获取减少方式的常量值
    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    reduction_vals = ["none", "mean", "sum"]
    reduction = reduction_vals[reduction]

    # 获取忽略索引的常量值
    ignore_index = symbolic_helper._maybe_get_const(ignore_index, "i")
    # 如果weight为None，则使用NegativeLogLikelihoodLoss操作计算损失
    if weight.node().mustBeNone():
        nllloss = g.op(
            "NegativeLogLikelihoodLoss",
            self,
            target,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )
    else:
        # 否则，使用带权重的NegativeLogLikelihoodLoss操作计算损失
        nllloss = g.op(
            "NegativeLogLikelihoodLoss",
            self,
            target,
            weight,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )

    # 返回负对数似然损失操作的结果
    return nllloss


# 使用ONNX符号化装饰器，指定对应的ATen操作为nll_loss2d
@_onnx_symbolic("aten::nll_loss2d")
# 使用装饰器将函数标记为需要类型检查的函数
@_beartype.beartype
# 定义nll_loss2d函数，调用nll_loss函数计算二维负对数似然损失
def nll_loss2d(
    g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index
):
    return nll_loss(g, self, target, weight, reduction, ignore_index)


# 使用ONNX符号化装饰器，指定对应的ATen操作为nll_loss_nd
@_onnx_symbolic("aten::nll_loss_nd")
# 使用装饰器将函数标记为需要类型检查的函数
@_beartype.beartype
# 定义nll_loss_nd函数，调用nll_loss函数计算多维负对数似然损失
def nll_loss_nd(
    g: jit_utils.GraphContext, self, target, weight, reduction, ignore_index
):
    return nll_loss(g, self, target, weight, reduction, ignore_index)


# 使用ONNX符号化装饰器，指定对应的ATen操作为cross_entropy_loss
@_onnx_symbolic("aten::cross_entropy_loss")
# 使用装饰器将函数标记为需要类型检查的函数
@_beartype.beartype
# 定义cross_entropy_loss函数，计算交叉熵损失
def cross_entropy_loss(
    g: jit_utils.GraphContext,
    self,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
):
    # 获取减少方式的常量值
    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    reduction_vals = ["none", "mean", "sum"]
    reduction = reduction_vals[reduction]

    # 获取标签平滑的常量值
    label_smoothing = symbolic_helper._maybe_get_const(label_smoothing, "f")
    # 如果标签平滑不为None且大于0.0，则抛出错误，因为ONNX不支持标签平滑
    if label_smoothing is not None and label_smoothing > 0.0:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX does not support label_smoothing", self
        )

    # in onnx SoftmaxCrossEntropyLoss specification, ignore_index is optional without default value.
    # 获取 ignore_index 参数的常量值，如果未指定则使用默认值 -100
    ignore_index = symbolic_helper._maybe_get_const(ignore_index, "i")
    # 检查 weight 节点是否为 None，若为 None，则使用没有 weight 参数的 SoftmaxCrossEntropyLoss 操作
    if weight.node().mustBeNone():
        celoss = g.op(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )
    else:
        # 使用带有 weight 参数的 SoftmaxCrossEntropyLoss 操作
        celoss = g.op(
            "SoftmaxCrossEntropyLoss",
            self,
            target,
            weight,
            reduction_s=reduction,
            ignore_index_i=ignore_index,
        )

    # 返回计算得到的交叉熵损失值
    return celoss
# 定义一个装饰器，将函数注册为处理 ONNX 符号的方法，处理 binary_cross_entropy_with_logits 符号
@_onnx_symbolic("aten::binary_cross_entropy_with_logits")
# 解析参数，指定参数的类型和数量
@symbolic_helper.parse_args("v", "v", "v", "v", "i")
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# 定义 binary_cross_entropy_with_logits 函数，用于计算带 logits 的二元交叉熵损失
def binary_cross_entropy_with_logits(
    g: jit_utils.GraphContext,  # 参数 g 表示图上下文
    input,  # 输入数据
    target,  # 目标数据
    weight,  # 权重
    pos_weight,  # 正权重
    reduction  # 减少类型（0: none, 1: mean, 2: sum）
):
    # 创建一个常量节点，值为 tensor([1])
    p = g.op("Constant", value_t=torch.tensor([1]))
    # 对输入数据应用 sigmoid 操作
    sig_x = opset9.sigmoid(g, input)
    # 对 sigmoid 结果应用 log 操作
    log_sig_x = opset9.log(g, sig_x)
    # 计算 1 - sigmoid(input)
    sub_1_x = opset9.sub(g, p, sig_x)
    # 计算 1 - target
    sub_1_y = opset9.sub(g, p, target)
    # 计算 log(1 - sigmoid(input))
    log_1_x = opset9.log(g, sub_1_x)
    
    # 如果没有正权重或正权重为 None，则计算不带正权重的交叉熵损失
    if pos_weight is None or symbolic_helper._is_none(pos_weight):
        output = opset9.neg(
            g,
            opset9.add(
                g, opset9.mul(g, target, log_sig_x), opset9.mul(g, sub_1_y, log_1_x)
            ),
        )
    else:
        # 否则，计算带正权重的交叉熵损失
        output = opset9.neg(
            g,
            opset9.add(
                g,
                opset9.mul(g, opset9.mul(g, target, log_sig_x), pos_weight),
                opset9.mul(g, sub_1_y, log_1_x),
            ),
        )

    # 如果有权重且权重不为 None，则将输出乘以权重
    if weight is not None and not symbolic_helper._is_none(weight):
        output = opset9.mul(g, weight, output)

    # 解析 reduction 参数，获取其常量值
    reduction = symbolic_helper._maybe_get_const(reduction, "i")
    # 根据 reduction 的不同类型进行相应的输出处理
    if reduction == 0:
        return output  # 返回未减少的输出
    elif reduction == 1:
        return g.op("ReduceMean", output, keepdims_i=0)  # 返回均值减少的输出
    elif reduction == 2:
        return g.op("ReduceSum", output, keepdims_i=0)  # 返回求和减少的输出
    else:
        # 如果 reduction 不是 0, 1, 2 中的任何一个，返回不支持的提示信息
        return symbolic_helper._onnx_unsupported(
            "binary_cross_entropy_with_logits with reduction other than none, mean, or sum",
            input,
        )


# 定义一个装饰器，将函数注册为处理 ONNX 符号的方法，处理 celu 符号
@_onnx_symbolic("aten::celu")
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# 定义 celu 函数，用于计算 celu 激活函数
def celu(g: jit_utils.GraphContext, self, alpha):
    # 解析 alpha 参数，获取其常量值
    alpha = symbolic_helper._maybe_get_const(alpha, "f")
    # 如果输入数据的类型是 double，则将其转换为 float
    if (
        _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.DOUBLE
    ):
        self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        # 应用 Celu 激活函数
        out = g.op("Celu", self, alpha_f=alpha)
        # 将输出再次转换为 double 类型
        return g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.DOUBLE)

    # 如果输入数据类型不是 double，则直接应用 Celu 激活函数
    return g.op("Celu", self, alpha_f=alpha)


# 定义一个装饰器，将函数注册为处理 ONNX 符号的方法，处理 argmax 符号
@_onnx_symbolic("aten::argmax")
# 解析参数，指定参数的类型和数量
@symbolic_helper.parse_args("v", "v", "b")
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# 定义 argmax 函数，用于计算输入数据在指定维度上的最大值索引
def argmax(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
):
    # 调用 _argmin_argmax_helper 辅助函数，实现 ArgMax 功能
    return symbolic_helper._argmin_argmax_helper(g, input, dim, keepdim, "ArgMax")


# 定义一个装饰器，将函数注册为处理 ONNX 符号的方法，处理 argmin 符号
@_onnx_symbolic("aten::argmin")
# 解析参数，指定参数的类型和数量
@symbolic_helper.parse_args("v", "v", "b")
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# 定义 argmin 函数，用于计算输入数据在指定维度上的最小值索引
def argmin(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
):
    # 调用 _argmin_argmax_helper 辅助函数，实现 ArgMin 功能
    return symbolic_helper._argmin_argmax_helper(g, input, dim, keepdim, "ArgMin")


# 定义一个装饰器，将函数注册为处理 ONNX 符号的方法，处理 pow 符号
@_onnx_symbolic("aten::pow")
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# 定义 pow 函数，用于计算输入数据的指数幂
def pow(g: jit_utils.GraphContext, self, exponent):
    # 应用 Pow 操作
    return g.op("Pow", self, exponent)


# 定义一个装饰器，将函数注册为处理 ONNX 符号的方法，处理 ge 符号
@_onnx_symbolic("aten::ge")
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# 定义 ge 函数，用于计算输入数据是否大于或等于另一数据
def ge(g: jit_utils.GraphContext, input, other):
    # 比较输入数据和其他数据的大小关系
    return g.op("GreaterOrEqual", input, other)
# 使用装饰器定义 ONNX 符号化操作 "aten::le"，将其转换为 LessOrEqual 操作
@_onnx_symbolic("aten::le")
# 应用 beartype 装饰器，用于类型检查和验证
@_beartype.beartype
# 定义 unfold 函数，用于在图中展开张量的特定维度
def le(g: jit_utils.GraphContext, input, other):
    # 在图上执行 LessOrEqual 操作，比较 input 和 other
    return g.op("LessOrEqual", input, other)


# 使用装饰器定义 ONNX 符号化操作 "aten::unfold"
@_onnx_symbolic("aten::unfold")
# 使用 parse_args 装饰器解析参数，指定参数类型为 (value, int, value, value)
@symbolic_helper.parse_args("v", "i", "v", "v")
# 应用 beartype 装饰器，用于类型检查和验证
@_beartype.beartype
# 定义 unfold 函数，用于在图中展开张量的特定维度
def unfold(g: jit_utils.GraphContext, input, dimension, size, step):
    # 尝试获取 size 和 step 的常量值，类型为整数
    const_size = symbolic_helper._maybe_get_const(size, "i")
    const_step = symbolic_helper._maybe_get_const(step, "i")
    # 如果 size 和 step 都不是常量值
    if not symbolic_helper._is_value(const_size) and not symbolic_helper._is_value(
        const_step
    ):
        # 使用 opset9 模块中的 unfold 函数在图中执行展开操作
        return opset9.unfold(g, input, dimension, const_size, const_step)

    # 获取输入张量在指定维度上的大小
    sizedim = symbolic_helper._get_tensor_dim_size(input, dimension)
    # 如果 sizedim 不为 None，则执行以下操作
    low_start = g.op("Constant", value_t=torch.tensor(0))
    # 创建一个常量节点，值为 0
    low_end = g.op("Constant", value_t=torch.tensor(sizedim))
    # 创建一个常量节点，值为 sizedim
    hi_end = g.op("Constant", value_t=torch.tensor(sizedim + 1))
    # 创建一个常量节点，值为 sizedim + 1
    low_indices = g.op("Range", low_start, low_end, step)
    # 使用起始值、结束值和步长创建一个范围节点
    hi_indices = g.op("Range", size, hi_end, step)
    # 使用起始值 size、结束值 hi_end 和步长创建一个范围节点

    # 计算低维度和高维度的尺寸
    low_size = symbolic_helper._size_helper(
        g, low_indices, g.op("Constant", value_t=torch.tensor(0))
    )
    # 使用 helper 函数计算低维度的尺寸
    hi_size = symbolic_helper._size_helper(
        g, hi_indices, g.op("Constant", value_t=torch.tensor(0))
    )
    # 使用 helper 函数计算高维度的尺寸

    # 获取输入张量的维度数
    ndim = symbolic_helper._get_tensor_rank(input)
    assert ndim is not None
    # 确保 ndim 不为 None
    perm = list(range(0, ndim))
    perm.append(perm.pop(dimension))
    # 重新排列维度，将指定维度移到最后

    # 初始化一个未展开的列表
    unsqueeze_list = []
    # 创建一个常量节点，值为 1
    loop_condition = g.op("Constant", value_t=torch.tensor(1))
    # 将 loop_condition 节点转换为 BOOL 类型
    loop_condition = g.op(
        "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
    )
    # 计算循环的长度，取低维度和高维度尺寸的最小值
    loop_len = g.op("Min", low_size, hi_size)

    # 向图中添加带有块的循环操作
    loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
        g, "Loop", loop_len, loop_condition, n_blocks=1
    )
    loop_block = loop_context.block
    # 向循环块中添加输入迭代器
    block_input_iter = utils._add_input_to_block(loop_block)

    # FIXME(justinchuby): cond is unused?
    cond = utils._add_input_to_block(loop_block)

    # 使用 Gather 操作从 low_indices 中收集数据
    starts = loop_context.op("Gather", low_indices, block_input_iter)
    # 使用 Gather 操作从 hi_indices 中收集数据
    ends = loop_context.op("Gather", hi_indices, block_input_iter)
    # 创建一个常量节点，值为 [2]
    axes = loop_context.op("Constant", value_t=torch.tensor([2]))
    # 使用 unsqueeze_helper 函数在 loop_context 中展开 starts
    starts = symbolic_helper._unsqueeze_helper(loop_context, starts, [0])
    # 使用 unsqueeze_helper 函数在 loop_context 中展开 ends
    ends = symbolic_helper._unsqueeze_helper(loop_context, ends, [0])
    # 使用 Slice 操作在输入张量上切片
    stack = loop_context.op("Slice", input, starts, ends, axes)

    # 使用 unsqueeze_helper 函数在 loop_context 中展开 Transpose 操作的 stack
    unsqueeze = symbolic_helper._unsqueeze_helper(
        loop_context, loop_context.op("Transpose", stack, perm_i=perm), [dimension]
    )
    # 将 unsqueeze 结果添加到 unsqueeze_list 中
    unsqueeze_list.append(unsqueeze)
    # 使用 Concat 操作连接 unsqueeze_list 中的所有张量
    concat = loop_context.op("Concat", *unsqueeze_list, axis_i=0)

    # 将 loop_condition 节点转换为 BOOL 类型，并输出到循环块
    cond_out = loop_context.op(
        "Cast", loop_condition, _C_onnx.TensorProtoDataType.BOOL
    )
    utils._add_output_to_block(loop_block, cond_out)
    # 将 concat 输出添加到循环块
    utils._add_output_to_block(loop_block, concat)

    # 获取循环节点的输出，并重新排列维度
    loop_output = loop.node().output()
    perm = [0, 1, 2, 3, 4]
    perm[0], perm[dimension + 1] = perm[dimension + 1], perm[0]
    # 使用 Transpose 操作重新排列维度
    transpose = g.op("Transpose", loop_output, perm_i=perm)
    # 使用 squeeze_helper 函数在 g 中展开 transpose
    squeeze = symbolic_helper._squeeze_helper(g, transpose, [0])

    # 返回 squeeze 结果作为函数的输出
    return squeeze

# 如果 sizedim 为 None，则输出未实现的提示信息
return symbolic_helper._unimplemented("Unfold", "input size not accessible")
# 声明一个装饰器函数，用于ONNX符号化处理，处理"aten::tensordot"操作符
# 装饰器函数将在tensordot函数定义时调用，接收函数作为参数
@_onnx_symbolic("aten::tensordot")
# 解析函数参数的装饰器，期望输入和输出的张量、维度等参数
@symbolic_helper.parse_args("v", "v", "is", "is", "v")
# 使用beartype库的装饰器，用于类型检查
@_beartype.beartype
# 定义tensordot函数，接收图形上下文g、输入张量input_a和input_b、维度dims_a和dims_b，以及可选的输出张量out
def tensordot(g: jit_utils.GraphContext, input_a, input_b, dims_a, dims_b, out=None):
    # 如果提供了输出张量out，则抛出未实现异常，因为不支持tensordot的out参数
    if out is not None:
        symbolic_helper._unimplemented(
            "Tensordot", "Out parameter is not supported for tensordot."
        )

    # 获取输入张量input_a的维度数
    dim_count_a = symbolic_helper._get_tensor_rank(input_a)
    # 如果无法获取维度数，则抛出符号值错误异常
    if dim_count_a is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of tensordot for tensor(input_a) of unknown rank.",
            input_a,
        )

    # 获取输入张量input_b的维度数
    dim_count_b = symbolic_helper._get_tensor_rank(input_b)
    # 如果无法获取维度数，则抛出符号值错误异常
    if dim_count_b is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of tensordot for tensor(input_b) of unknown rank.",
            input_b,
        )

    # 将负数维度转换为有效维度索引，dims_a和dims_b是维度描述列表
    dims_a = [
        (dims_a[i] + dim_count_a) if (dims_a[i] < 0) else dims_a[i]
        for i in range(len(dims_a))
    ]
    dims_b = [
        (dims_b[i] + dim_count_b) if (dims_b[i] < 0) else dims_b[i]
        for i in range(len(dims_b))
    ]

    # 计算未在dims_a中的左侧维度索引
    left_dims_a = [i for i in range(dim_count_a) if (i not in dims_a)]
    # 计算未在dims_b中的左侧维度索引
    left_dims_b = [i for i in range(dim_count_b) if (i not in dims_b)]

    # 使用opset9.permute对输入张量input_a进行维度置换，将dims_a放置在left_dims_a之后
    new_input_a = opset9.permute(g, input_a, left_dims_a + dims_a)
    # 使用opset9.permute对输入张量input_b进行维度置换，将dims_b放置在left_dims_b之前
    new_input_b = opset9.permute(g, input_b, dims_b + left_dims_b)

    # 获取new_input_a的形状
    input_shape = g.op("Shape", new_input_a)
    # 获取new_input_a的左侧维度大小，即left_dims_a对应的形状
    left_sizes_a = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[0], ends=[len(left_dims_a)]
    )
    # 构造重塑操作的目标形状sizes
    shape_sizes = [
        left_sizes_a,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
    ]
    # 执行opset9._reshape_from_tensor函数，根据sizes重塑new_input_a
    output_a = opset9._reshape_from_tensor(g, new_input_a, shape_sizes)

    # 获取output_a的形状
    input_shape = g.op("Shape", output_a)
    # 使用slice_helper获取output_a的维度形状
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[-1], ends=[sys.maxsize]
    )
    # 构造重塑操作的目标形状sizes
    shape_sizes = [
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
        slices,
    ]
    # 执行opset9._reshape_from_tensor函数，根据sizes重塑new_input_a
    output_a = opset9._reshape_from_tensor(g, new_input_a, shape_sizes)

    # 获取new_input_b的形状
    input_shape = g.op("Shape", new_input_b)
    # 获取new_input_b的左侧维度大小，即dims_b对应的形状
    left_sizes_b = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[len(dims_b)], ends=[sys.maxsize]
    )
    # 使用slice_helper获取new_input_b的维度形状
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[0], ends=[len(dims_b)]
    )
    # 构造重塑操作的目标形状sizes
    shape_sizes = [
        slices,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
    ]
    # 执行opset9._reshape_from_tensor函数，根据sizes重塑new_input_b
    output_b = opset9._reshape_from_tensor(g, new_input_b, shape_sizes)

    # 获取output_b的形状
    input_shape = g.op("Shape", output_b)
    # 使用slice_helper获取output_b的维度形状
    slices = symbolic_helper._slice_helper(
        g, input_shape, axes=[0], starts=[-1], ends=[sys.maxsize]
    )
    # 构造重塑操作的目标形状sizes
    shape_sizes = [
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
        slices,
    ]
    # 执行opset9._reshape_from_tensor函数，根据sizes重塑new_input_b
    output_b = opset9._reshape_from_tensor(g, new_input_b, shape_sizes)

    # 使用einsum函数计算张量乘积，指定输入为"ij,jk->ik"，输出为output
    output = einsum(g, "ij,jk->ik", g.op("prim::ListConstruct", *[output_a, output_b]))
    # 定义一个包含两个列表的列表，用于存储两个不同的大小信息
    shape_sizes = [left_sizes_a, left_sizes_b]
    # 调用 opset9 模块中的 _reshape_from_tensor 函数，重新塑形张量
    # g 是图结构，output 是输出张量，shape_sizes 是输入张量的大小信息列表
    return opset9._reshape_from_tensor(g, output, shape_sizes)
```