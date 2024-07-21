# `.\pytorch\torch\onnx\symbolic_opset9.py`

```py
# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 9.

Opset 9 is supported by ONNX release 1.4.1
release on 01/23/19
"""

# 导入必要的模块
from __future__ import annotations

import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch._C._onnx as _C_onnx  # 导入 torch._C._onnx 模块并命名为 _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C

# Monkey-patch graph manipulation methods on Graph, used for the ONNX symbolics
# 导入 ONNX 符号操作所需的功能模块
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

# 定义导出的 ONNX 操作名称列表
__all__ = [
    "abs",
    "acos",
    "add",
    "addcmul",
    "addmm",
    "alias",
    "amax",
    "amin",
    "aminmax",
    "arange",
    "argmax",
    "argmin",
    "as_strided",
    "as_tensor",
    "asin",
    "atan",
    "atan2",
    "baddbmm",
    "batch_norm",
    "bernoulli",
    "bitwise_not",
    "bitwise_or",
    "bmm",
    "broadcast_tensors",
    "broadcast_to",
    "bucketize",
    "cat",
    "cdist",
    "ceil",
    "clamp_max",
    "clamp_min",
    "clamp",
    "clone",
    "constant_pad_nd",
    "contiguous",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "conv1d",
    "conv2d",
    "conv3d",
    "convert_element_type",
    "convolution",
    "cos",
    "cosine_similarity",
    "cross",
    "cumsum",
    "detach",
    "dim",
    "div",
    "dot",
    "dropout",
    "elu",
    "embedding_bag",
    "embedding",
    "empty_like",
    "empty",
    "eq",
    "erf",
    "exp",
    "expand_as",
    "expand",
    "eye",
    "fill",
    "flatten",
    "floor_divide",
    "floor",
    "floordiv",
    "frobenius_norm",
    "full_like",
    "full",
    "gather",
    "ge",
    "gelu",
    "get_pool_ceil_padding",
    "glu",
    "group_norm",
    "gt",
    "hann_window",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "index_add",
    "index_copy",
    "index_fill",
    "index_put",
    "index_select",
    "index",
    "instance_norm",
    "is_floating_point",
    "is_pinned",
    "isnan",
    "item",
    "kl_div",
    "layer_norm",
    "le",
    "leaky_relu",
    "lerp",
    "lift",
    "linalg_cross",
    "linalg_matrix_norm",
    "linalg_norm",
    "linalg_vector_norm",
    "linear",
    "linspace",
    "log_sigmoid",
    "log_softmax",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logit",
    "logsumexp",
    "lstm_cell",
    "lstm",
    "lt",
    "masked_fill",
    "masked_fill_",
    "matmul",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
    "max",
    "maximum",
    "meshgrid",
    "min",
    "minimum",
    "mish",
    "mm",
    # 定义字符串列表，包含了一系列函数名
    [
        "movedim",  # 移动维度
        "mse_loss",  # 均方误差损失
        "mul",  # 乘法
        "multinomial",  # 多项式分布采样
        "mv",  # 矩阵向量乘法
        "narrow",  # 窄化张量维度
        "native_layer_norm",  # 本地层归一化
        "ne",  # 不等于
        "neg",  # 取负
        "new_empty",  # 创建空张量
        "new_full",  # 创建指定填充值的张量
        "new_ones",  # 创建全为1的张量
        "new_zeros",  # 创建全为0的张量
        "nonzero_numpy",  # 非零元素索引 (numpy)
        "nonzero",  # 非零元素索引
        "norm",  # 张量范数计算
        "numel",  # 张量元素总数
        "numpy_T",  # 转置 (numpy)
        "one_hot",  # 独热编码
        "ones_like",  # 创建形状相同的全1张量
        "ones",  # 创建全为1的张量
        "onnx_placeholder",  # ONNX 占位符
        "pad",  # 填充
        "pairwise_distance",  # 两两间距离计算
        "permute",  # 维度置换
        "pixel_shuffle",  # 像素重排
        "pixel_unshuffle",  # 像素反重排
        "pow",  # 幂运算
        "prelu",  # Parametric ReLU 激活函数
        "prim_constant_chunk",  # 常量分块
        "prim_constant_split",  # 常量分割
        "prim_constant",  # 常量
        "prim_data",  # 数据
        "prim_device",  # 设备
        "prim_dtype",  # 数据类型
        "prim_if",  # 条件语句
        "prim_layout",  # 布局
        "prim_list_construct",  # 列表构造
        "prim_list_unpack",  # 列表解包
        "prim_loop",  # 循环
        "prim_max",  # 最大值
        "prim_min",  # 最小值
        "prim_shape",  # 形状
        "prim_tolist",  # 转为列表
        "prim_tuple_construct",  # 元组构造
        "prim_type",  # 类型
        "prim_unchecked_cast",  # 未检查的类型转换
        "prim_uninitialized",  # 未初始化的值
        "rand_like",  # 随机数生成，形状与输入相同
        "rand",  # 随机数生成
        "randint_like",  # 随机整数生成，形状与输入相同
        "randint",  # 随机整数生成
        "randn_like",  # 标准正态分布随机数生成，形状与输入相同
        "randn",  # 标准正态分布随机数生成
        "reciprocal",  # 倒数
        "reflection_pad",  # 反射填充
        "relu",  # ReLU 激活函数
        "relu6",  # ReLU6 激活函数
        "remainder",  # 取余
        "repeat_interleave",  # 重复插入
        "repeat",  # 重复
        "replication_pad",  # 复制填充
        "reshape_as",  # 根据另一张量的形状重塑
        "reshape",  # 重塑形状
        "roll",  # 循环移动
        "rrelu",  # 随机 ReLU 激活函数
        "rsqrt",  # 平方根倒数
        "rsub",  # 右侧减法
        "scalar_tensor",  # 标量张量
        "scatter_add",  # 散布累加
        "scatter",  # 散布
        "select",  # 选择
        "selu",  # SELU 激活函数
        "sigmoid",  # Sigmoid 激活函数
        "sign",  # 符号函数
        "silu",  # SiLU 激活函数
        "sin",  # 正弦函数
        "size",  # 尺寸
        "slice",  # 切片
        "softmax",  # Softmax 激活函数
        "softplus",  # Softplus 激活函数
        "softshrink",  # Softshrink 激活函数
        "sort",  # 排序
        "split_with_sizes",  # 按大小拆分
        "split",  # 拆分
        "sqrt",  # 平方根
        "square",  # 平方
        "squeeze",  # 压缩张量维度
        "stack",  # 堆叠张量
        "std_mean",  # 标准差和均值
        "std",  # 标准差
        "sub",  # 减法
        "t",  # 转置
        "take",  # 取出
        "tan",  # 正切函数
        "tanh",  # 双曲正切函数
        "tanhshrink",  # 双曲正切缩减函数
        "tensor",  # 张量创建
        "threshold",  # 阈值函数
        "to",  # 张量类型转换
        "topk",  # Top-K 元素
        "transpose",  # 转置
        "true_divide",  # 真除
        "type_as",  # 类型匹配
        "unbind",  # 解绑
        "unfold",  # 展开
        "unsafe_chunk",  # 不安全的分块
        "unsafe_split_with_sizes",  # 不安全的按大小拆分
        "unsafe_split",  # 不安全的拆分
        "unsqueeze",  # 增加张量维度
        "unsupported_complex_operators",  # 不支持的复杂操作符
        "noop_complex_operators",  # 空操作复杂操作符
        "unused",  # 未使用
        "var_mean",  # 方差和均值
        "var",  # 方差
        "view_as",  # 类似视图
        "view",  # 视图
        "where",  # 条件选择
        "wrap_logical_op_with_cast_to",  # 包装逻辑运算以转换到
        "wrap_logical_op_with_negation",  # 包装逻辑运算以否定
        "zeros_like",  # 创建形状相同的全0张量
        "zeros",  # 创建全为0的张量
        "zero",  # 零
    ]
# 创建一个偏函数，用于将函数注册为符号操作，并指定使用的 ONNX 版本为 9
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=9)


def _export(name: str):
    """将函数导出到当前全局命名空间中。

    Args:
        name (str): 要导出的函数名称。

    Returns:
        callable: 装饰器函数。
    """
    
    def wrapper(func):
        # 将函数添加到全局命名空间中
        globals()[name] = func
        # 将函数名称添加到 __all__ 列表中，以便在模块导入时能够被访问到
        __all__.append(name)
        return func

    return wrapper


@_beartype.beartype
def unused(g):
    """表示“缺失”的可选输入。

    Args:
        g: 图形操作对象。

    Returns:
        torch.Tensor: 表示缺失输入的常量张量。
    """
    n = g.op("prim::Constant")
    n.setType(_C.OptionalType.ofTensor())
    return n


@_onnx_symbolic("aten::_shape_as_tensor")
@_beartype.beartype
def _shape_as_tensor(g: jit_utils.GraphContext, input):
    """将输入张量转换为形状张量的符号操作。

    Args:
        g (GraphContext): 图形上下文。
        input (Tensor): 输入张量。

    Returns:
        torch.Tensor: 表示输入张量形状的张量。
    """
    return g.op("Shape", input)


@_onnx_symbolic("aten::_reshape_from_tensor")
@_beartype.beartype
def _reshape_from_tensor(g: jit_utils.GraphContext, input, shape):
    """从形状张量重新调整输入张量的符号操作。

    Args:
        g (GraphContext): 图形上下文。
        input (Tensor): 输入张量。
        shape (Tensor or List[Tensor]): 目标形状张量或张量列表。

    Returns:
        torch.Tensor: 调整形状后的张量。
    """
    if isinstance(shape, list):
        shape = g.op("Concat", *shape, axis_i=0)
    return reshape(g, input, shape)


@_onnx_symbolic("aten::reshape")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def reshape(g: jit_utils.GraphContext, self, shape):
    """执行张量的重塑符号操作。

    Args:
        g (GraphContext): 图形上下文。
        self (Tensor): 输入张量。
        shape (Tensor): 目标形状张量。

    Returns:
        torch.Tensor: 重塑后的张量。
    """
    return symbolic_helper._reshape_helper(g, self, shape)


@_onnx_symbolic("aten::reshape_as")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def reshape_as(g: jit_utils.GraphContext, self, other):
    """根据另一个张量的形状执行重塑符号操作。

    Args:
        g (GraphContext): 图形上下文。
        self (Tensor): 输入张量。
        other (Tensor): 参考形状的张量。

    Returns:
        torch.Tensor: 重塑后的张量。
    """
    shape = g.op("Shape", other)
    return reshape(g, self, shape)


@_onnx_symbolic("aten::add")
@_beartype.beartype
def add(g: jit_utils.GraphContext, self, other, alpha=None):
    """将 add 函数转换为相应的 ONNX 操作符。

    这个函数不应该由用户直接调用。

    Args:
        g (GraphContext): 图形上下文。
        self (Tensor): 第一个操作数。
        other (Tensor): 第二个操作数。
        alpha (float, optional): 第二个操作数的缩放因子。默认为 None。

    Returns:
        torch.Tensor: ONNX 操作符。
    """
    if symbolic_helper._is_value(self) and symbolic_helper._is_tensor_list(self):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "Add", 9, 11, "Add between list of tensors not supported", self
        )
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        other = g.op("Mul", other, alpha)
    return g.op("Add", self, other)


@_onnx_symbolic("aten::sub")
@_beartype.beartype
def sub(g: jit_utils.GraphContext, self, other, alpha=None):
    """将 sub 函数转换为相应的 ONNX 操作符。

    这个函数不应该由用户直接调用。

    Args:
        g (GraphContext): 图形上下文。
        self (Tensor): 第一个操作数。
        other (Tensor): 第二个操作数。
        alpha (Optional[Tensor]): 第二个操作数的缩放因子。如果未提供，将默认为 1。

    Returns:
        torch.Tensor: ONNX 操作符。
    """
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        other = g.op("Mul", other, alpha)
    return g.op("Sub", self, other)
# 对应于 ONNX 符号化的 "aten::rsub" 操作
@_onnx_symbolic("aten::rsub")
# 使用 beartype 进行类型检查和注解
@_beartype.beartype
def rsub(g: jit_utils.GraphContext, self, other, alpha=None):
    # 调用 sub 函数，但是交换了 self 和 other 的顺序
    return sub(g, other, self, alpha=alpha)


# 对应于 ONNX 符号化的 "aten::mul" 操作
@_onnx_symbolic("aten::mul")
# 使用 beartype 进行类型检查和注解
@_beartype.beartype
def mul(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_bool(self) and symbolic_helper._is_bool(other):
        # 当 self 和 other 都是布尔类型时，使用 And 作为等效操作符
        return g.op("And", self, other)
    else:
        # 否则使用乘法操作符
        return g.op("Mul", self, other)


# 对应于 ONNX 符号化的 "aten::div" 操作
@_onnx_symbolic("aten::div")
# 使用 beartype 进行类型检查和注解
@_beartype.beartype
def div(g: jit_utils.GraphContext, self, other, *args):
    if len(args) == 0:
        # 如果没有额外参数，则调用 true_divide 函数
        return true_divide(g, self, other)
    else:
        # 否则调用 _div_rounding_mode 函数
        return _div_rounding_mode(g, self, other, *args)


# 对应于 ONNX 符号化的 "aten::addcmul" 操作
@_onnx_symbolic("aten::addcmul")
# 解析参数 "v", "v", "v", "f" 并进行类型检查和注解
@symbolic_helper.parse_args("v", "v", "v", "f")
@_beartype.beartype
def addcmul(g: jit_utils.GraphContext, self, tensor1, tensor2, value=1.0):
    # 创建一个常量张量，值为 value
    value_tens = g.op("Constant", value=torch.tensor([value]))
    # 调用 add 函数，计算 self + (tensor1 * tensor2 * value_tens)
    return add(g, self, mul(g, mul(g, tensor1, tensor2), value_tens))


# 解析参数 "v", "v", "s" 并进行类型检查和注解
@symbolic_helper.parse_args("v", "v", "s")
@_beartype.beartype
def _div_rounding_mode(g: jit_utils.GraphContext, self, other, rounding_mode):
    if rounding_mode is None:
        # 如果 rounding_mode 是 None，则调用 true_divide 函数
        return true_divide(g, self, other)
    elif rounding_mode == "floor":
        # 如果 rounding_mode 是 "floor"，则调用 _floor_divide 函数
        return _floor_divide(g, self, other)
    elif rounding_mode == "trunc":
        # 如果 rounding_mode 是 "trunc"，则调用 _trunc_divide 函数
        return _trunc_divide(g, self, other)
    else:
        # 否则抛出错误，表示不支持的 rounding_mode
        raise errors.SymbolicValueError(
            f'Unsupported rounding mode: "{rounding_mode}". Expected None, "floor" or "trunc"',
            self,
        )


# 对应于 ONNX 符号化的 "_trunc_divide" 函数
@_beartype.beartype
def _trunc_divide(g: jit_utils.GraphContext, self, other):
    # 使用 "Div" 操作符计算除法
    out = g.op("Div", self, other)
    # 对于截断操作，ONNX 不支持，因此需要转换成 INT64 类型
    out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.INT64)

    # 匹配 PyTorch 的行为：
    # - 如果 self 是浮点数，则输出类型为 self 的类型
    # - 如果 self 不是浮点数且 other 是浮点数，则输出类型为 JitScalarType.FLOAT
    # - 如果 self 和 other 都不是浮点数，则输出类型为 self 的输出类型
    # - 输出类型默认为 Float
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        if not symbolic_helper._is_fp(self) and symbolic_helper._is_fp(other):
            out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        else:
            out = g.op(
                "Cast",
                out,
                to_i=scalar_type.onnx_type(),
            )
    else:
        out = g.op("Cast", out, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    return out


# 使用 beartype 进行类型检查和注解
@_beartype.beartype
# 定义函数 _floor_divide，执行整数除法操作
def _floor_divide(g: jit_utils.GraphContext, self, other):
    # 如果 self 或 other 是浮点数类型，则调用 true_divide 函数执行真实除法
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        out = true_divide(g, self, other)
        # 返回对 out 应用 Floor 操作后的结果
        return g.op("Floor", out)
    else:
        # 如果 self 和 other 都是整数类型，则执行截断整数除法
        div = g.op("Div", self, other)
        
        # 计算 self < 0 且 other < 0 的逻辑异或结果
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        negative = g.op(
            "Xor",
            symbolic_helper._lt_helper(g, self, zero),
            symbolic_helper._lt_helper(g, other, zero),
        )

        # 计算 self % other != 0 的情况下，通过减去 1 实现向下舍入
        mod = g.op("Sub", self, g.op("Mul", div, other))
        fixup_mask = g.op("And", negative, g.op("Not", g.op("Equal", mod, zero)))

        # 创建值为 1 的常量张量
        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        # 计算 fixup_mask 乘以 1 的结果，即减去 1 的修正值
        fixup = g.op("Mul", fixup_mask, one)
        # 返回 div 减去 fixup 的结果，实现向下舍入的整数除法操作
        return g.op("Sub", div, fixup)


# 定义函数 floor_divide，为符号操作 "aten::floor_divide" 提供具体实现
@_onnx_symbolic("aten::floor_divide")
@_beartype.beartype
def floor_divide(g: jit_utils.GraphContext, self, other):
    # 废弃的行为，floor_divide 实际上执行截断除法
    return _trunc_divide(g, self, other)


# 定义函数 floordiv，为符号操作 "aten::floordiv" 提供具体实现
@_onnx_symbolic("aten::floordiv")
@_beartype.beartype
def floordiv(g: jit_utils.GraphContext, self, other):
    # 调用 floor_divide 函数执行整数除法
    return floor_divide(g, self, other)


# 定义函数 true_divide，为符号操作 "aten::true_divide" 提供具体实现
@_onnx_symbolic("aten::true_divide")
@_beartype.beartype
def true_divide(g: jit_utils.GraphContext, self, other):
    """Division where both inputs are cast to floating types

    If both inputs are floating, performs div as usual
    If only one input is a floating type, the other input is cast to its type
    If neither input is a floating type, both inputs are cast to the default scalar type
    """

    # Case 1: either values are floating
    # 如果其中一个值是浮点数类型，则按照通常的除法执行
    # 标量类型分析将处理隐式转换
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        return g.op("Div", self, other)

    # Case 2: neither is floating
    # 如果两个输入值都不是浮点数类型，则将两个输入值转换为默认的标量类型
    scalar_type = torch.get_default_dtype()
    onnx_scalar_type = _C_onnx.TensorProtoDataType.FLOAT
    assert scalar_type is torch.float or scalar_type is torch.double
    if torch.get_default_dtype() is torch.double:
        onnx_scalar_type = _C_onnx.TensorProtoDataType.DOUBLE

    self = g.op("Cast", self, to_i=onnx_scalar_type)
    other = g.op("Cast", other, to_i=onnx_scalar_type)
    # 返回将转换后的 self 和 other 执行除法的结果
    return g.op("Div", self, other)


# 定义函数 reciprocal，为符号操作 "aten::reciprocal" 提供具体实现
@_onnx_symbolic("aten::reciprocal")
@_beartype.beartype
def reciprocal(g: jit_utils.GraphContext, self):
    # torch.reciprocal 隐式转换为浮点数类型，因此我们也执行相同操作
    if not symbolic_helper._is_fp(self):
        self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    # 返回对 self 应用 Reciprocal 操作的结果
    return g.op("Reciprocal", self)


# 定义函数 cat，为符号操作 "aten::cat" 提供具体实现
@_onnx_symbolic("aten::cat")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
# 定义一个函数 cat，用于在指定维度上对 pytorch 张量进行 ONNX 格式的拼接
def cat(g: jit_utils.GraphContext, tensor_list, dim):
    """Implement concatenation of pytorch tensors in ONNX along the specified `dim` dimension.

    Parameters:
        g (jit_utils.GraphContext): Graph context.
        tensor_list (List[torch.Tensor]): List of tensors to concatenate.
        dim (int): Dimension along which to concatenate the tensors.

    Returns:
        ONNX graph node representing the concatenated tensor.
    """
    # 将 tensor_list 中的元素解包并存储在 tensors 中
    tensors = symbolic_helper._unpack_list(tensor_list)
    
    # 过滤掉空张量，因为 torch.cat 会忽略空张量如 `torch.Tensor([])`
    # 这些空张量需要从 ONNX 的拼接输入中移除，否则由于输入的秩数不同，形状推断可能会失败（空张量秩数为 0，非空张量秩数大于 0）
    nonempty_tensors = []
    for t in tensors:
        if symbolic_helper._is_constant(t) and not symbolic_helper._get_tensor_dim_size(
            t, 0
        ):
            continue
        nonempty_tensors.append(t)
    
    # 断言至少存在一个非空张量
    assert len(nonempty_tensors) > 0
    
    # 断言所有非空张量的秩数要么为 None，要么与第一个非空张量的秩数相同
    assert all(
        symbolic_helper._get_tensor_rank(nonempty_tensors[0]) is None
        or symbolic_helper._get_tensor_rank(t) is None
        or symbolic_helper._get_tensor_rank(t)
        == symbolic_helper._get_tensor_rank(nonempty_tensors[0])
        for t in nonempty_tensors
    )
    
    # 清空 tensor_list 的所有输入
    tensor_list.node().removeAllInputs()
    # 将所有非空张量添加到 tensor_list 的输入中
    for t in nonempty_tensors:
        tensor_list.node().addInput(t)

    # 重新解包 tensor_list 中的元素，并在指定维度上进行 ONNX 的 Concat 操作
    tensors = symbolic_helper._unpack_list(tensor_list)
    return g.op("Concat", *tensors, axis_i=dim)


# 对应的其他函数只返回 ONNX 的操作节点，不需要做过多的解释
@_onnx_symbolic("aten::stack")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def stack(g: jit_utils.GraphContext, tensor_list, dim):
    unsqueezed = [
        symbolic_helper._unsqueeze_helper(g, t, [dim])
        for t in symbolic_helper._unpack_list(tensor_list)
    ]
    return g.op("Concat", *unsqueezed, axis_i=dim)


@_onnx_symbolic("aten::list")
@_beartype.beartype
def _list(g: jit_utils.GraphContext, self):
    return self


@_onnx_symbolic("aten::mm")
@_beartype.beartype
def mm(g: jit_utils.GraphContext, self, other):
    # 创建一个虚拟的 C 张量。仅用于 API 目的，值为 1
    C = g.op("Constant", value_t=torch.tensor([1]))
    return g.op("Gemm", self, other, C, beta_f=0.0, alpha_f=1.0)


@_onnx_symbolic("aten::bmm")
@_beartype.beartype
def bmm(g: jit_utils.GraphContext, self, other):
    return g.op("MatMul", self, other)


@_onnx_symbolic("aten::matmul")
@_beartype.beartype
def matmul(g: jit_utils.GraphContext, self, other):
    return g.op("MatMul", self, other)


@_onnx_symbolic("aten::addmm")
@symbolic_helper.parse_args("v", "v", "v", "t", "t")
@_beartype.beartype
def addmm(g: jit_utils.GraphContext, self, mat1, mat2, beta, alpha):
    scalar_type = None
    self_scalar_type = symbolic_helper._try_get_scalar_type(self)
    mat1_scalar_type = symbolic_helper._try_get_scalar_type(mat1)
    mat2_scalar_type = symbolic_helper._try_get_scalar_type(mat2)
    # 如果 self_scalar_type 不为空，则使用 self_scalar_type 作为 scalar_type
    if self_scalar_type is not None:
        scalar_type = self_scalar_type
    # 否则，如果 mat1_scalar_type 不为空，则使用 mat1_scalar_type 作为 scalar_type
    elif mat1_scalar_type is not None:
        scalar_type = mat1_scalar_type
    # 否则，如果 mat2_scalar_type 不为空，则使用 mat2_scalar_type 作为 scalar_type
    elif mat2_scalar_type is not None:
        scalar_type = mat2_scalar_type

    # 获取 mat1 和 mat2 的张量秩
    mat1_rank = symbolic_helper._get_tensor_rank(mat1)
    mat2_rank = symbolic_helper._get_tensor_rank(mat2)

    # 定义一个函数，用于检查输入是否不为 None 且不等于指定值
    def is_not_none_nor(v, u):
        return v is not None and v != u

    # 如果 scalar_type 不为空，并且 mat1 或 mat2 的秩不等于 2
    if scalar_type is not None and (
        is_not_none_nor(mat1_rank, 2) or is_not_none_nor(mat2_rank, 2)
    ):
        # 执行矩阵乘法操作，并赋值给 res1
        res1 = g.op("MatMul", mat1, mat2)
        # 将 self 赋值给 res2
        res2 = self

        # 将 alpha 和 beta 转换为标量值
        alpha = symbolic_helper._scalar(alpha)
        beta = symbolic_helper._scalar(beta)

        # 如果 alpha 不等于 1，则创建一个表示 alpha 常量的节点，并与 res1 相乘
        if alpha != 1:
            alpha = g.op(
                "Constant", value_t=torch.tensor(alpha, dtype=scalar_type.dtype())
            )
            res1 = g.op("Mul", res1, alpha)
        
        # 如果 beta 不等于 1，则创建一个表示 beta 常量的节点，并与 res2 相乘
        if beta != 1:
            beta = g.op(
                "Constant",
                value_t=torch.tensor(
                    symbolic_helper._scalar(beta), dtype=scalar_type.dtype()
                ),
            )
            res2 = g.op("Mul", res2, beta)

        # 返回 res1 和 res2 相加的结果
        return g.op("Add", res1, res2)

    # 如果不满足上述条件，则执行通用矩阵乘法操作 Gemm
    return g.op(
        "Gemm",
        mat1,
        mat2,
        self,
        beta_f=symbolic_helper._scalar(beta),
        alpha_f=symbolic_helper._scalar(alpha),
    )
# 注册 ONNX 符号化函数，将 aten::neg 映射到负数操作
# 添加类型检查装饰器 beartype
@_onnx_symbolic("aten::neg")
@_beartype.beartype
def neg(g: jit_utils.GraphContext, self):
    # 在图上执行负数操作 Neg
    return g.op("Neg", self)


# 注册 ONNX 符号化函数，将 aten::sqrt 映射到平方根操作
# 添加类型检查装饰器 beartype
@_onnx_symbolic("aten::sqrt")
@_beartype.beartype
def sqrt(g: jit_utils.GraphContext, self):
    # 如果 self 是以下整数类型之一，则将其转换为 FLOAT 类型
    if _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.UINT8,
        _type_utils.JitScalarType.INT8,
        _type_utils.JitScalarType.INT16,
        _type_utils.JitScalarType.INT,
        _type_utils.JitScalarType.INT64,
    }:
        # 使用 Cast 操作将 self 转换为 FLOAT
        self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    # 在图上执行平方根操作 Sqrt
    return g.op("Sqrt", self)


# 注册 ONNX 符号化函数，将 aten::rsqrt 映射到倒数平方根操作
# 添加类型检查装饰器 beartype
def rsqrt(g: jit_utils.GraphContext, self):
    # 使用 torch.ones(1) 创建张量，将其类型转换为 self 的类型，并计算其平方根
    return g.op(
        "Div", symbolic_helper._if_scalar_type_as(torch.ones(1), self), sqrt(g, self)
    )


# 注册 ONNX 符号化函数，将 aten::tanh 映射到双曲正切操作
# 添加量化参数装饰器，固定 scale 和 zero_point
@_onnx_symbolic("aten::tanh")
@symbolic_helper.quantized_args(True, scale=2.0 / 256.0, zero_point=128)
@_beartype.beartype
def tanh(g: jit_utils.GraphContext, self):
    # 在图上执行双曲正切操作 Tanh
    return g.op("Tanh", self)


# 注册 ONNX 符号化函数，将 aten::sin 映射到正弦操作
# 添加类型检查装饰器 beartype
@_onnx_symbolic("aten::sin")
@_beartype.beartype
def sin(g: jit_utils.GraphContext, self):
    # 在图上执行正弦操作 Sin
    return g.op("Sin", self)


# 注册 ONNX 符号化函数，将 aten::cos 映射到余弦操作
# 添加类型检查装饰器 beartype
@_onnx_symbolic("aten::cos")
@_beartype.beartype
def cos(g: jit_utils.GraphContext, self):
    # 在图上执行余弦操作 Cos
    return g.op("Cos", self)


# 注册 ONNX 符号化函数，将 aten::tan 映射到正切操作
# 添加类型检查装饰器 beartype
@_onnx_symbolic("aten::tan")
@_beartype.beartype
def tan(g: jit_utils.GraphContext, self):
    # 在图上执行正切操作 Tan
    return g.op("Tan", self)


# 注册 ONNX 符号化函数，将 aten::asin 映射到反正弦操作
# 添加类型检查装饰器 beartype
@_onnx_symbolic("aten::asin")
@_beartype.beartype
def asin(g: jit_utils.GraphContext, self):
    # 在图上执行反正弦操作 Asin
    return g.op("Asin", self)


# 注册 ONNX 符号化函数，将 aten::acos 映射到反余弦操作
# 添加类型检查装饰器 beartype
@_onnx_symbolic("aten::acos")
@_beartype.beartype
def acos(g: jit_utils.GraphContext, self):
    # 在图上执行反余弦操作 Acos
    return g.op("Acos", self)


# 注册 ONNX 符号化函数，将 aten::atan 映射到反正切操作
# 添加类型检查装饰器 beartype
@_onnx_symbolic("aten::atan")
@_beartype.beartype
def atan(g: jit_utils.GraphContext, self):
    # 在图上执行反正切操作 Atan
    return g.op("Atan", self)


# 注册 ONNX 符号化函数，将 aten::atan2 映射到反正切操作
# 添加类型检查装饰器 beartype
@_onnx_symbolic("aten::atan2")
@_beartype.beartype
def atan2(g: jit_utils.GraphContext, self, other):
    # 计算斜率 slope = self / other
    slope = g.op("Div", self, other)
    # 计算反正切 atan
    atan = g.op("Atan", slope)
    # 创建常量节点
    const_zero = g.op("Constant", value_t=torch.tensor(0))
    const_pi = g.op("Constant", value_t=torch.tensor(math.pi))

    # 条件：第二或第三象限
    condition_second_or_third_quadrant = g.op("Greater", self, const_zero)
    # 第二或第三象限的反正切值
    second_third_quadrant = g.op(
        "Where",
        condition_second_or_third_quadrant,
        g.op("Add", atan, const_pi),
        g.op("Sub", atan, const_pi),
    )

    # 条件：第一或第四象限
    condition_14_or_23_quadrant = g.op("Less", other, const_zero)
    # 根据条件选择结果
    result = g.op("Where", condition_14_or_23_quadrant, second_third_quadrant, atan)

    return result


# 注册 ONNX 符号化函数，将 aten::sigmoid 映射到 Sigmoid 操作
# 添加量化参数装饰器，固定 scale 和 zero_point
@_onnx_symbolic("aten::sigmoid")
@symbolic_helper.quantized_args(True, scale=1.0 / 256.0, zero_point=0)
@_beartype.beartype
def sigmoid(g: jit_utils.GraphContext, self):
    """
    将对应的 PyTorch 函数转换为 ONNX 操作符。

    此函数不应由用户直接调用。

    Args:
        g (jit_utils.GraphContext): 图形上下文。
        self (Tensor): 输入张量。
    Returns:
        ONNX 操作符，代表 Sigmoid 函数
    """
    # 使用图形上下文 g 的 op 方法，创建一个 Sigmoid 的 ONNX 操作符，作用于输入张量 self
    return g.op("Sigmoid", self)
# 将函数标记为ONNX符号化处理"aten::sign"，并使用Beartype进行类型检查
@_onnx_symbolic("aten::sign")
@_beartype.beartype
def sign(g: jit_utils.GraphContext, self):
    # 使用ONNX操作"Sign"实现对self的符号函数操作
    return g.op("Sign", self)


# 标记函数为量化参数的符号化辅助函数，使用Beartype进行类型检查
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def _slice(g: jit_utils.GraphContext, input, axes, starts, ends):
    # 断言starts和ends列表长度相等
    assert len(starts) == len(ends)
    # 如果starts长度为1且starts[0]为0，ends[0]为_INT64_MAX，则返回input
    if len(starts) == 1 and starts[0] == 0 and ends[0] == _constants.INT64_MAX:
        return input
    # 使用ONNX操作"Slice"实现对input的切片操作，指定轴、起始和结束位置
    return g.op("Slice", input, axes_i=axes, starts_i=starts, ends_i=ends)


# 标记函数为"aten::sum"的ONNX符号化处理，装饰器指定使用"ReduceSum"名称
@_onnx_symbolic(
    "aten::sum", decorate=[symbolic_helper._apply_params("ReduceSum", "sum")]
)
# 标记函数为"aten::mean"的ONNX符号化处理，装饰器指定使用"ReduceMean"名称
@_onnx_symbolic(
    "aten::mean", decorate=[symbolic_helper._apply_params("ReduceMean", "mean")]
)
# 标记函数为"aten::prod"的ONNX符号化处理，装饰器指定使用"ReduceProd"名称，不支持多维度
@_onnx_symbolic(
    "aten::prod",
    decorate=[
        symbolic_helper._apply_params(
            "ReduceProd", "prod", allow_multi_dim_support=False
        )
    ],
)
@_beartype.beartype
def _reduce_with_dtype(onnx_op: str, name: str, allow_multi_dim_support: bool = True):
    # 调用符号化辅助函数，实现基于数据类型的降维操作，传递ONNX操作名称、名称和是否支持多维度
    return symbolic_helper._reduce_with_dtype_helper(
        onnx_op, name, allow_multi_dim_support
    )


# 标记函数为"aten::cumsum"的ONNX符号化处理
@_onnx_symbolic("aten::cumsum")
# 解析参数为向量、整数、空值
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype
def cumsum(g: jit_utils.GraphContext, input, dim, dtype):
    # 报告不支持的ONNX操作"cumsum"，传递ONNX操作名称和输入
    symbolic_helper._onnx_opset_unsupported("cumsum", 9, 11, input)


# 标记函数为"aten::_sample_dirichlet"的ONNX符号化处理，使用Beartype进行类型检查
@_onnx_symbolic("aten::_sample_dirichlet")
@_beartype.beartype
def _sample_dirichlet(g: jit_utils.GraphContext, self, generator):
    # 报告不支持的ONNX操作"_sample_dirichlet"，传递ONNX操作名称和self
    return symbolic_helper._onnx_unsupported("_sample_dirichlet", self)


# 标记函数为"aten::_standard_gamma"的ONNX符号化处理，使用Beartype进行类型检查
@_onnx_symbolic("aten::_standard_gamma")
@_beartype.beartype
def _standard_gamma(g: jit_utils.GraphContext, self, generator):
    # 报告不支持的ONNX操作"_standard_gamma"，传递ONNX操作名称和self
    return symbolic_helper._onnx_unsupported("_standard_gamma", self)


# 标记函数为"aten::t"的ONNX符号化处理，使用Beartype进行类型检查
@_onnx_symbolic("aten::t")
@_beartype.beartype
def t(g: jit_utils.GraphContext, self):
    # 获取张量的秩
    rank = symbolic_helper._get_tensor_rank(self)
    # 如果秩为None或小于2，添加Identity节点以模仿eager模式的行为
    if rank is None or rank < 2:
        return g.op("Identity", self)
    # 使用ONNX操作"Transpose"实现对self的转置操作，指定轴的排列顺序
    return g.op("Transpose", self, perm_i=(1, 0))


# 标记函数为"aten::numpy_T"的ONNX符号化处理，标记函数为量化参数
@_onnx_symbolic("aten::numpy_T")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def numpy_T(g: jit_utils.GraphContext, input):
    # 获取张量的秩
    ndim = symbolic_helper._get_tensor_rank(input)
    # 断言秩不为None
    assert ndim is not None
    # 计算反转的轴顺序
    perm = list(reversed(range(0, ndim)))
    # 使用ONNX操作"Transpose"实现对input的转置操作，指定轴的排列顺序
    return g.op("Transpose", input, perm_i=perm)


# 标记函数为"aten::expand"的ONNX符号化处理，标记函数为量化参数
@_onnx_symbolic("aten::expand")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def expand(g: jit_utils.GraphContext, self, size, implicit):
    """实现扩展pytorch张量的ONNX函数，根据指定的`size`"""
    # 可能获取常数的size
    size = symbolic_helper._maybe_get_const(size, "is")
    # 如果size不是值类型，则创建一个包含size的常数张量
    if not symbolic_helper._is_value(size):
        size = g.op("Constant", value_t=torch.LongTensor(size))
    # 如果 size 是一个 packed list（压缩列表），则执行以下操作
    elif symbolic_helper._is_packed_list(size):
        # 在展开时，-1 表示维度保持不变。
        # 因为 onnx::expand 支持两向广播，-1 可以导出到 onnx 中的 1
        # 使用 _reshape_helper 函数重塑 size，将其作为 g.op 的参数
        size = symbolic_helper._reshape_helper(
            g, stack(g, size, 0), g.op("Constant", value_t=torch.tensor([-1]))
        )
    # 将数据类型设为 INT64
    dtype = _type_utils.JitScalarType.INT64
    # 创建一个与 size 相同形状和数据类型的全为 1 的张量
    ones = ones_like(g, size, dtype)
    # 创建一个与 size 相同形状和数据类型的全为 -1 的张量，并与 ones 相乘
    neg_ones = mul(g, ones, g.op("Constant", value_t=torch.tensor(-1)))
    # 如果 size 等于 neg_ones，则将 size 中相应的位置改为 ones 中的值，否则保持不变
    size = where(g, g.op("Equal", size, neg_ones), ones, size)
    # 返回一个 Expand 操作，将当前对象 self 按照 size 扩展
    return g.op("Expand", self, size)
# 使用装饰器指定符号化处理函数的名称为 "aten::broadcast_to"
# 使用装饰器标识符号化辅助函数要处理量化参数
# 使用装饰器应用 Beartype 检查
def broadcast_to(g: jit_utils.GraphContext, self, size):
    # 尝试获取 size 的常量值，如果不是常量则不处理
    size = symbolic_helper._maybe_get_const(size, "is")
    if not symbolic_helper._is_value(size):
        # 如果 size 不是常量，则将其封装为 torch.LongTensor 类型的常量
        size = g.op("Constant", value_t=torch.LongTensor(size))
    elif symbolic_helper._is_packed_list(size):
        # 如果 size 是打包列表，则进行形状重塑处理
        # -1 维度值表示维度保持不变
        # 由于 onnx::expand 支持双向广播，-1 维度值可以导出为 onnx 中的 1
        size = symbolic_helper._reshape_helper(
            g, stack(g, size, 0), g.op("Constant", value_t=torch.tensor([-1]))
        )
    # 数据类型设置为 INT64
    dtype = _type_utils.JitScalarType.INT64
    # 创建与 size 形状相同的全为 1 的张量
    ones = ones_like(g, size, dtype)
    # 创建与 size 形状相同的全为 -1 的张量
    neg_ones = mul(g, ones, g.op("Constant", value_t=torch.tensor(-1)))
    # 根据条件进行选择，如果 size 等于 neg_ones，则使用 ones 替换 size
    size = where(g, g.op("Equal", size, neg_ones), ones, size)
    # 执行操作，将 self 根据 size 扩展
    return g.op("Expand", self, size)


# 使用装饰器指定符号化处理函数的名称为 "aten::expand_as"
# 使用装饰器标识符号化辅助函数要处理量化参数，并且要求两个参数都是 torch.Tensor
# 使用装饰器应用 Beartype 检查
def expand_as(g: jit_utils.GraphContext, self, other):
    # 尝试获取 self 的常量值
    self_t = symbolic_helper._maybe_get_const(self, "t")
    if isinstance(self_t, torch.Tensor):
        # 如果 self_t 是 torch.Tensor，则保存原始数据类型
        orig_type = self_t.dtype
        # 将 self_t 转换为 torch.double 类型的张量
        self_t = self_t.to(torch.double)
        # 创建空列表保存维度
        dims = []
        # 遍历 self_t 的每个维度
        for d in range(self_t.dim()):
            # 如果 self_t 在当前维度上的均值的扩展与 self_t 相等，则加入 dims 列表
            if torch.equal(self_t.mean(d).unsqueeze(d).expand_as(self_t), self_t):
                dims.append(d)
                # 使用 self_t 的均值来创建 Constant 操作的常量张量，保持维度不变
                self = g.op(
                    "Constant", value_t=self_t.mean(dims, keepdim=True).to(orig_type)
                )
    # 获取 other 的形状
    shape = g.op("Shape", other)
    # 执行操作，将 self 根据 other 的形状扩展
    return g.op("Expand", self, shape)


# 使用装饰器指定符号化处理函数的名称为 "aten::embedding"
# 使用装饰器标识符号化辅助函数要处理量化参数
# 使用装饰器指定参数解析方式，参数依次为张量、张量、整数、布尔值、张量
# 使用装饰器应用 Beartype 检查
def embedding(
    g: jit_utils.GraphContext,
    weight,
    indices,
    padding_idx,
    scale_grad_by_freq,
    sparse,
):
    # 如果 scale_grad_by_freq 为真且处于全局训练模式下，则引发异常
    if scale_grad_by_freq and GLOBALS.export_training:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of embedding with scale_grad_by_freq=True "
            "for training mode. ONNX does not support scaling the gradients.",
            weight,
        )
    # 如果 padding_idx 大于等于 0 且处于全局训练模式下，则发出警告
    if padding_idx >= 0 and GLOBALS.export_training:
        warnings.warn(
            "Warning: ONNX export of embedding with padding_idx >= 0 "
            "for training mode. "
            "ONNX does not support not updating the embedding vector at padding_idx during training."
        )
    # 执行 Gather 操作，从 weight 中根据 indices 收集数据
    return g.op("Gather", weight, indices)


# 使用装饰器指定符号化处理函数的名称为 "aten::embedding_bag"
# 使用装饰器标识符号化辅助函数要处理量化参数
# 使用装饰器指定参数解析方式，参数依次为张量、张量、张量、整数、整数、整数、张量、整数、整数
# 使用装饰器应用 Beartype 检查
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
    # 如果 per_sample_weights 不是 None，则调用 _onnx_unsupported 函数，指示不支持带 per_sample_weights 的 embedding_bag
    if not symbolic_helper._is_none(per_sample_weights):
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with per_sample_weights"
        )

    # 调用 _onnx_unsupported 函数，指示不支持简单的 embedding_bag 操作，带有 embedding_matrix 参数
    return symbolic_helper._onnx_unsupported("embedding_bag", embedding_matrix)
# 使用装饰器定义 ONNX 符号化函数，指定对应的 PyTorch ATen 操作 "aten::size"
# 同时处理量化参数，不对输出进行量化
@_onnx_symbolic("aten::size")
@symbolic_helper.quantized_args(True, quantize_output=False)
@_beartype.beartype
def size(g: jit_utils.GraphContext, self, dim=None):
    # 如果维度参数 dim 为 None，则使用 ONNX 操作 "Shape" 返回张量的形状
    if dim is None:
        return g.op("Shape", self)
    
    # 如果 dim 可能为负数，则通过计算张量的秩来将其转换为正数
    if symbolic_helper._maybe_get_const(dim, "i") < 0:
        rank = symbolic_helper._get_tensor_rank(self)
        if rank is not None:
            dim = symbolic_helper._maybe_get_const(dim, "i") + rank
            dim = g.op("Constant", value_t=torch.tensor(dim))
    
    # 调用辅助函数 _size_helper 处理获取指定维度的大小信息
    return symbolic_helper._size_helper(g, self, dim)


# 使用装饰器定义 ONNX 符号化函数，指定对应的 PyTorch ATen 操作 "aten::transpose"
# 处理量化参数，无需量化输出
@_onnx_symbolic("aten::transpose")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def transpose(g: jit_utils.GraphContext, self, dim0, dim1):
    # 如果 dim0 等于 dim1，直接返回自身，这是一个微优化
    if dim0 == dim1:  # micro-optimization
        return self
    
    # 获取张量的秩信息
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is not None:
        # 创建一个包含所有维度索引的列表
        axes = list(range(rank))
        # 交换 dim0 和 dim1 在列表中的位置，模拟转置操作
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        # 使用 ONNX 操作 "Transpose" 执行转置操作，指定轴的排列顺序
        return g.op("Transpose", self, perm_i=axes)
    else:
        # 如果无法获取张量秩信息，则抛出符号化数值错误
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of transpose for tensor of unknown rank.",
            self,
        )


# 使用装饰器定义 ONNX 符号化函数，指定对应的 PyTorch ATen 操作 "aten::permute"
@_onnx_symbolic("aten::permute")
@symbolic_helper.parse_args("v", "is")
@_beartype.beartype
def permute(g: jit_utils.GraphContext, self, dims):
    # 如果 dims 是顺序排列的维度列表，则直接返回自身
    if dims == list(range(0, len(dims))):
        return self
    
    # 使用 ONNX 操作 "Transpose" 执行维度置换操作，指定维度的排列顺序
    return g.op("Transpose", self, perm_i=dims)


# 使用装饰器定义 ONNX 符号化函数，指定对应的 PyTorch ATen 操作 "aten::view"
# 处理量化参数
@_onnx_symbolic("aten::view")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def view(g: jit_utils.GraphContext, self, size):
    # 调用 reshape 函数执行形状重塑操作
    return reshape(g, self, size)


# 使用装饰器定义 ONNX 符号化函数，指定对应的 PyTorch ATen 操作 "aten::view_as"
@_onnx_symbolic("aten::view_as")
@_beartype.beartype
def view_as(g: jit_utils.GraphContext, self, other):
    # 获取 other 张量的形状
    shape = g.op("Shape", other)
    # 调用 reshape 函数执行形状重塑操作
    return reshape(g, self, shape)


# 使用装饰器定义 ONNX 符号化函数，指定对应的 PyTorch ATen 操作 "aten::unsafe_chunk"
@_onnx_symbolic("aten::unsafe_chunk")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def unsafe_chunk(g: jit_utils.GraphContext, self, chunks, dim, _outputs=None):
    # 如果未指定 _outputs 参数，则返回不支持动态输出数量的详细信息
    if _outputs is None:
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "unsafe_chunk", 9, 11, "Dynamic number of outputs not supported", self
        )
    
    # 获取指定维度上的张量尺寸
    size = symbolic_helper._get_tensor_dim_size(self, dim)
    if size is None:
        # 如果无法获取维度大小信息，则返回未实现的详细信息
        return symbolic_helper._unimplemented(
            "unsafe_chunk", "unknown dimension size", self
        )
    
    # 计算每个分块的大小
    split_size = (size + chunks - 1) // chunks
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    
    # 使用 ONNX 操作 "Split" 执行分块操作，指定分块大小、轴和输出数
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=_outputs)


# 使用装饰器定义 ONNX 符号化函数，指定对应的 PyTorch ATen 操作 "aten::split"
@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
@_beartype.beartype
def split(g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None):
    # 检查是否不支持分割大小的符号助手方法
    if not symbolic_helper._is_split_static(split_size_or_sizes, _outputs):
        # 如果不支持动态输出数量，则返回详细的不支持信息
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "split", 9, 11, "Dynamic number of outputs not supported", self
        )
    
    # 从分割大小或尺寸中获取分割值
    split_val = symbolic_helper._node_get(split_size_or_sizes.node(), "value")
    
    # 如果分割值的维度大于0，则调用使用指定尺寸分割的方法
    if split_val.dim() > 0:
        return split_with_sizes(g, self, split_size_or_sizes, dim, _outputs)
    
    # 否则，获取常量分割大小
    split_size = symbolic_helper._get_const(split_size_or_sizes, "i", "split_size")

    # 获取指定维度上的张量大小
    size = symbolic_helper._get_tensor_dim_size(self, dim)
    
    # 如果大小未知
    if size is None:
        # 如果有输出数量，则计算总大小
        if _outputs is not None:
            size = split_size * _outputs
        else:
            # 否则返回详细的未知尺寸不支持信息
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "split", 9, 11, "Unknown dimension size not supported", self
            )
    
    # 根据分割大小计算分割数组
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    # 如果有余数，将余数添加到分割数组中
    if leftover:
        splits.append(leftover)
    
    # 返回使用指定参数调用的操作节点
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=_outputs)
# 使用装饰器指定该函数对应的ONNX符号化操作"aten::unsafe_split"
# 应用类型检查装饰器
@_onnx_symbolic("aten::unsafe_split")
@_beartype.beartype
def unsafe_split(
    g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None
):
    # 调用split函数来执行拆分操作
    return split(g, self, split_size_or_sizes, dim, _outputs)


# 使用装饰器指定该函数对应的ONNX符号化操作"aten::split_with_sizes"
# 解析函数参数，并指定参数类型
@_onnx_symbolic("aten::split_with_sizes")
@symbolic_helper.parse_args("v", "is", "i", "i")
@_beartype.beartype
def split_with_sizes(g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None):
    # 如果拆分尺寸不是静态确定的，则返回不支持动态输出的错误信息
    if not symbolic_helper._is_split_static(split_sizes, _outputs):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "split_with_sizes", 9, 11, "Dynamic number of outputs not supported", self
        )
    # 使用ONNX操作"Split"来执行拆分操作
    return g.op("Split", self, split_i=split_sizes, axis_i=dim, outputs=_outputs)


# 使用装饰器指定该函数对应的ONNX符号化操作"aten::unsafe_split_with_sizes"
@_onnx_symbolic("aten::unsafe_split_with_sizes")
@_beartype.beartype
def unsafe_split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None
):
    # 调用split_with_sizes函数来执行拆分操作
    return split_with_sizes(g, self, split_sizes, dim, _outputs)


# 使用装饰器指定该函数对应的ONNX符号化操作"aten::unbind"
# 解析函数参数，并指定参数类型
@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
def unbind(g: jit_utils.GraphContext, self, dim=0, _outputs=None):
    # 如果没有指定输出数量，则返回不支持动态输出的错误信息
    if _outputs is None:
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "unbind", 9, 11, "Dynamic number of outputs not supported", self
        )

    # 使用ONNX操作"Split"来执行拆分操作，生成指定数量的输出
    outputs = g.op("Split", self, split_i=[1] * _outputs, axis_i=dim, outputs=_outputs)
    # 如果输出数量为1，则转换为列表形式
    outputs = [outputs] if _outputs == 1 else outputs
    # 对每个输出执行挤压操作，去除指定维度的尺寸为1的维度
    squeezed_outputs = [
        symbolic_helper._squeeze_helper(g, out, [dim]) for out in outputs
    ]
    return squeezed_outputs


# 使用装饰器指定该函数对应的ONNX符号化操作"aten::select"
# 处理量化参数
@_onnx_symbolic("aten::select")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "v")
@_beartype.beartype
def select(g: jit_utils.GraphContext, self, dim, index):
    """Implement the select functionality for a pytorch tensor in ONNX.

    Selects elements from the input tensor along the specified `dim` dimension based on the `index` tensor.
    """
    # 尝试将index转换为标量（scalar）
    index = symbolic_helper._maybe_get_scalar(index)
    # 如果index不是值，并且小于0，则处理负数索引情况
    if (not symbolic_helper._is_value(index)) and (index < 0):
        if index == -1:
            end_index = _constants.INT64_MAX
        else:
            end_index = index + 1
        # 使用切片辅助函数来处理负数索引情况
        slice_node = symbolic_helper._slice_helper(
            g, self, axes=[dim], starts=[index], ends=[end_index]
        )
        # 对切片结果进行挤压操作，去除指定维度的尺寸为1的维度
        return symbolic_helper._squeeze_helper(g, slice_node, [dim])
    else:
        # 否则，使用ONNX操作"Gather"来收集指定索引的元素
        return g.op("Gather", self, index, axis_i=dim)


# 使用装饰器指定该函数对应的ONNX符号化操作"aten::square"
@_onnx_symbolic("aten::square")
@_beartype.beartype
def square(g: jit_utils.GraphContext, self):
    # 使用ONNX操作"Mul"来执行平方操作
    return g.op("Mul", self, self)


# 使用装饰器指定该函数对应的ONNX符号化操作"aten::squeeze"
@_onnx_symbolic("aten::squeeze")
@_beartype.beartype
def squeeze(g: jit_utils.GraphContext, self, dim=None):
    # 如果未指定维度，则使用ONNX操作"Squeeze"来执行挤压操作
    if dim is None:
        return g.op("Squeeze", self)

    # 否则，获取维度的常量值
    squeeze_dim = symbolic_helper._get_const(dim, "i", "dim")
    # 处理负数维度的情况
    # Handle negative dims
    # 如果要进行的挤压操作的维度小于 0，则需要根据输入张量的秩进行调整
    if squeeze_dim < 0:
        # 获取张量的秩（如果已知）
        rank = symbolic_helper._get_tensor_rank(self)
        # 如果秩已知，则发出警告并调整挤压维度以适应 ONNX 规范
        if rank is not None:
            warnings.warn(
                "ONNX export squeeze with negative axis "
                + str(squeeze_dim)
                + " might cause the onnx model to be incorrect. "
                + "Negative axis is not supported in ONNX. "
                + "Axis is converted to "
                + str(squeeze_dim + rank)
                + " based on input shape at export time. "
                + "Passing an tensor of different rank in execution will be incorrect."
            )
            squeeze_dim += rank
        else:
            # 如果输入秩未知，则标记为未实现，并返回
            return symbolic_helper._unimplemented(
                "squeeze", "negative axis with unknown input rank", self
            )

    # 获取指定维度的大小
    dim_size = symbolic_helper._get_tensor_dim_size(self, squeeze_dim)
    # 如果无法获取维度大小，则发出警告并返回适用的 squeeze 操作
    if dim_size is None:
        warnings.warn(
            "This model contains a squeeze operation on dimension "
            + str(squeeze_dim)
            + " on an input "
            + "with unknown shape. Note that if the size of dimension "
            + str(squeeze_dim)
            + " of the input "
            + "is not 1, the ONNX model will return an error. Opset version 11 supports squeezing on "
            + "non-singleton dimensions, it is recommended to export this model using opset "
            + "version 11 or higher."
        )
        return symbolic_helper._squeeze_helper(g, self, axes_i=[squeeze_dim])

    # 如果维度大小大于 1，则发出警告并返回原张量，不执行 squeeze 操作
    if dim_size > 1:
        warnings.warn(
            "This model contains a squeeze operation on dimension "
            + str(squeeze_dim)
            + ". The size of "
            + "this dimension in the given input is "
            + str(dim_size)
            + ". The model will "
            + "be exported without the squeeze node. If the model is intended to be used with dynamic "
            + "input shapes, please use opset version 11 to "
            + "export the model."
        )
        return self

    # 如果维度大小为 1，则发出警告，建议使用 opset version 11 导出模型以支持动态输入形状
    warnings.warn(
        "This model contains a squeeze operation on dimension "
        + str(squeeze_dim)
        + ". If the model is "
        + "intended to be used with dynamic input shapes, please use opset version 11 to export the model."
    )
    return symbolic_helper._squeeze_helper(g, self, axes_i=[squeeze_dim])
# 根据 ONNX 符号注册的装饰器，将该函数注册为对应的 ONNX 符号"aten::prelu"
# 使用 beartype 装饰器，对函数参数进行类型检查
def prelu(g: jit_utils.GraphContext, self, weight):
    # 获取 self 张量的维度数
    self_rank = symbolic_helper._get_tensor_rank(self)
    # 获取 weight 张量的尺寸
    weight_sizes = symbolic_helper._get_tensor_sizes(weight)
    # 计算 weight 张量的维度数
    weight_rank = len(weight_sizes)
    
    # 如果 self 张量的维度数不为空
    if self_rank is not None:
        # 如果 self 张量的维度数大于2
        if self_rank > 2:
            # 使 weight 张量在指定维度上实现单向广播
            weight = symbolic_helper._unsqueeze_helper(
                g, weight, list(range(1, self_rank - 1))
            )
        # 如果 self 张量的维度数为0，并且 weight 张量的尺寸为[1]
        elif self_rank == 0 and weight_sizes == [1]:
            # 对 weight 张量进行挤压操作
            weight = symbolic_helper._squeeze_helper(g, weight, [0])
            # 将 weight 张量的维度数设置为0
            weight_rank = 0

    # 如果 self 张量的维度数和 weight 张量的维度数均不为空
    if self_rank is not None and weight_rank is not None:
        # 断言 self 张量的维度数应大于等于 weight 张量的维度数
        assert (
            self_rank >= weight_rank
        ), f"rank(x) should be >= rank(slope) but got {self_rank} < {weight_rank}"
    
    # 返回通过 ONNX 运算符"PRelu"实现的操作
    return g.op("PRelu", self, weight)
    # 使用 g 对象的 op 方法调用 LeakyRelu 操作，对输入 input 进行操作
    # 使用 alpha_f 参数设置 LeakyRelu 的负斜率（即斜率小于零时的系数）
    return g.op("LeakyRelu", input, alpha_f=negative_slope)
# 使用装饰器定义 ONNX 符号化函数 "aten::glu"
# parse_args("v", "i") 解析函数参数，v 表示向量，i 表示整数
# 使用 beartype 进行类型检查和注解
@_onnx_symbolic("aten::glu")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def glu(g: jit_utils.GraphContext, input, dim):
    # 获取输入张量在指定维度上的大小
    dim_size = symbolic_helper._get_tensor_dim_size(input, dim)
    if dim_size is not None:
        # 断言维度大小为偶数
        assert dim_size % 2 == 0

    # 使用 ONNX 符号化函数 op("Split")，将输入张量按指定维度分割成两部分
    first, second = g.op("Split", input, axis_i=dim, outputs=2)
    # 返回通过乘法和 Sigmoid 函数处理后的结果
    return g.op("Mul", first, g.op("Sigmoid", second))


# 使用装饰器定义 ONNX 符号化函数 "aten::softmax"
# parse_args("v", "i", "none") 解析函数参数，v 表示向量，i 表示整数，none 表示可选参数
# 使用 beartype 进行类型检查和注解
@_onnx_symbolic("aten::softmax")
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype
def softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    # Softmax 在向量级别进行归一化
    # PyTorch 和 ONNX 在将输入张量拆分为向量时使用不同的策略
    # 因此 dim 和 axis 有不同的含义
    # PyTorch 沿着 dim 维度将输入张量切片成向量
    # ONNX 将输入张量重塑为二维张量，axis 表示输入被强制转换的位置
    # 如果输入是一个 2 x 3 的张量：
    # input = [[1.0, 1.0, 1.0],
    #          [1.0, 1.0, 1.0]]
    # 当 dim = 0 时，结果为：
    # result = [[0.5, 0.5, 0.5],
    #           [0.5, 0.5, 0.5]]
    # 当 axis = 0 时，结果为：
    # result = [[0.167, 0.167, 0.167],
    #           [0.167, 0.167, 0.167]]
    # 只有当 dim 和 axis 都等于 ndim - 1（最后一个维度）时，它们的语义才是等价的
    # 因此在 dim 和 axis 都等于 ndim - 1 时使用 softmax
    # 否则将输入转置，将要归一化的向量放到最后一个维度
    # 当导出时无法确定输入的秩时，使用子图计算 softmax
    input_dim = symbolic_helper._get_tensor_rank(input)
    if input_dim is not None:
        # TODO: remove this as onnx opset 11 spec allows negative axes
        # 如果 dim 是负数，将其转换为正数索引
        if dim < 0:
            dim = input_dim + dim

        # 判断是否需要进行输入的转置操作
        is_transpose_required = input_dim != dim + 1

        if is_transpose_required:
            # 构建转置轴列表
            axes = list(range(input_dim))
            axes[dim], axes[-1] = axes[-1], axes[dim]
            # 对输入进行转置操作
            input = g.op("Transpose", input, perm_i=axes)
            dim = input_dim - 1

        # 应用 softmax 操作，并指定归一化的维度 axis_i=dim
        softmax = g.op("Softmax", input, axis_i=dim)

        # 如果指定了 dtype 并且不是常量节点，则将 softmax 结果转换为指定类型
        if dtype and dtype.node().kind() != "prim::Constant":
            parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
            softmax = g.op(
                "Cast",
                softmax,
                to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type(),
            )

        if is_transpose_required:
            # 如果进行了转置操作，将 softmax 结果再次转置回原始轴顺序
            softmax = g.op("Transpose", softmax, perm_i=axes)  # type: ignore[possibly-undefined]
        return softmax

    # 如果无法获取输入的秩信息，在导出时使用 max 归一化
    # 计算输入张量在指定维度上的最大值并进行减法操作
    input = g.op("Sub", input, g.op("ReduceMax", input, axes_i=[dim], keepdims_i=1))

    # 对输入张量进行指数函数操作
    exp = g.op("Exp", input)
    # 计算指定维度上的求和并返回
    sum = symbolic_helper._reducesum_helper(g, exp, axes_i=[dim])
    # 应用 softmax 操作并返回结果
    softmax = g.op("Div", exp, sum)
    # 检查 dtype 是否存在并且不是常量节点
    if dtype and dtype.node().kind() != "prim::Constant":
        # 从 symbolic_helper 模块获取常量 dtype 的解析结果
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        # 将 softmax 张量转换为指定的 ONNX 类型，使用 Cast 操作符
        softmax = g.op(
            "Cast", softmax, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    # 返回转换后的 softmax 张量
    return softmax
# 注册 ONNX 符号化函数，将 "aten::softplus" 映射到此函数
# 应用 beartype 装饰器，用于类型检查
@_onnx_symbolic("aten::softplus")
@_beartype.beartype
def softplus(g: jit_utils.GraphContext, self, beta, threshold):
    # 获取 beta 是否为常数，如果不是则返回按照 Softplus 的定义计算结果
    beta_const = symbolic_helper._maybe_get_const(beta, "f")
    if beta_const != 1:
        return g.op("Div", g.op("Softplus", g.op("Mul", self, beta)), beta)
    # 返回按照 Softplus 的定义计算结果
    return g.op("Softplus", self)


# 注册 ONNX 符号化函数，将 "aten::get_pool_ceil_padding" 映射到此函数
# 应用 beartype 装饰器，用于类型检查
def get_pool_ceil_padding(input, kernel_size, stride, padding):
    # TODO(justinchuby): 看起来这个操作在 torch 中已经不推荐使用
    # 获取输入张量的尺寸
    sizes = symbolic_helper._get_tensor_sizes(input)
    # 获取维度信息，若尺寸不可访问则返回未实现的操作
    dim = sizes[-len(padding) :] if sizes is not None else None
    if dim is None or any(i is None for i in dim):
        return symbolic_helper._unimplemented(
            "get_pool_ceil_padding", "input size not accessible", input
        )
    # 计算池化层的输出尺寸，使用向上取整确保边界内开始
    ceiled_output_dim = [
        int(math.ceil((dim[i] + 2 * padding[i] - kernel_size[i]) / float(stride[i])))
        + 1
        for i in range(0, len(padding))
    ]
    # 确保最后一个池化从内部开始
    ceiled_output_dim = [
        (
            ceiled_output_dim[i] - 1
            if (((ceiled_output_dim[i] - 1) * stride[i]) >= (dim[i] + padding[i]))
            else ceiled_output_dim[i]
        )
        for i in range(0, len(ceiled_output_dim))
    ]
    # 计算池化的填充量，确保不超过核的大小
    padding_ceil = [
        (
            0
            if (stride[i] == 1)
            else (
                kernel_size[i]
                - (
                    dim[i]
                    + 2 * padding[i]
                    - ((ceiled_output_dim[i] - 1) * stride[i] + 1)
                )
            )
        )
        for i in range(0, len(padding))
    ]
    # 确保填充不大于核的大小
    padding_ceil = [
        (
            (
                int(padding_ceil[i])
                if padding_ceil[i] < kernel_size[i] - 1
                else int(kernel_size[i] - 1)
            )
            if ((padding_ceil[i] + 2 * padding[i]) >= (kernel_size[i]))
            else int(padding_ceil[i])
        )
        for i in range(0, len(padding_ceil))
    ]
    # 返回计算出的填充量
    return padding_ceil


# 注册 ONNX 符号化函数，将 "aten::max_pool1d"、"aten::max_pool2d"、"aten::max_pool3d" 映射到此函数
# 通过装饰器应用参数和导出函数
@_onnx_symbolic(
    "aten::max_pool1d",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool1d", torch.nn.modules.utils._single, 1, return_indices=False
        ),
        _export("max_pool1d"),
    ],
)
@_onnx_symbolic(
    "aten::max_pool2d",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool2d", torch.nn.modules.utils._pair, 2, return_indices=False
        ),
        _export("max_pool2d"),
    ],
)
@_onnx_symbolic(
    "aten::max_pool3d",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool3d", torch.nn.modules.utils._triple, 3, return_indices=False
        ),
        _export("max_pool3d"),
    ],
)
# 应用 beartype 装饰器，用于类型检查
@_beartype.beartype
def _max_pool(name, tuple_fn, ndims, return_indices):
    # 应用量化参数解析器和参数解析
    @symbolic_helper.quantized_args(True, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
    # 使用装饰器 @_beartype.beartype 对 symbolic_fn 函数进行类型检查和验证
    @_beartype.beartype
    # 定义 symbolic_fn 函数，接受多个参数：g, input, kernel_size, stride, padding, dilation, ceil_mode
    def symbolic_fn(g, input, kernel_size, stride, padding, dilation, ceil_mode):
        # 如果 dilation 不全为1，则返回未实现的操作
        if set(tuple_fn(dilation)) != {1}:
            return symbolic_helper._unimplemented(name, "dilation", input)
        # 如果 stride 为假值（例如0或None），则将其设为 kernel_size
        if not stride:
            stride = kernel_size
        # 将 padding 转换为元组形式
        padding = tuple(tuple_fn(padding))
        # 如果 ceil_mode 为真，则计算额外的 padding 以保证输出尺寸向上取整
        if ceil_mode:
            # 调用 get_pool_ceil_padding 函数计算额外的 padding
            padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
            # 将计算得到的额外 padding 加到原始 padding 中
            padding = padding + tuple(a + b for (a, b) in zip(padding_ceil, padding))
        else:
            # 如果 ceil_mode 为假，则将原始 padding 扩展成两倍
            padding = padding * 2
        # 构建 kwargs 字典，包含了 kernel_shape_i, pads_i, strides_i 等参数
        kwargs = {
            "kernel_shape_i": tuple_fn(kernel_size),
            "pads_i": padding,
            "strides_i": tuple_fn(stride),
        }
        # 如果 return_indices 为真，则进行以下操作
        if return_indices:
            # 执行 g.op("MaxPool", ...) 操作，并返回结果 r 和 indices
            r, indices = g.op("MaxPool", input, outputs=2, **kwargs)
            # 再次执行 g.op("MaxPool", ...) 操作，获得 flattened_indices
            _, flattened_indices = g.op(
                "MaxPool",
                input,
                outputs=2,
                kernel_shape_i=[1 for _ in range(ndims)],
                strides_i=[1 for _ in range(ndims)],
            )
            # 使用 symbolic_helper._slice_helper 函数处理 indices，使其具有非扁平化的索引值
            s = symbolic_helper._slice_helper(
                g,
                flattened_indices,
                axes=[2 + i for i in range(ndims)],
                starts=list(tuple_fn(0)),
                ends=list(tuple_fn(1)),
            )
            # 对 indices 进行修正，使其具有非扁平化的索引值
            indices = sub(g, indices, s)
            # 返回结果 r 和修正后的 indices
            return r, indices
        else:
            # 如果 return_indices 为假，则只执行 g.op("MaxPool", ...) 操作，返回结果 r
            r = g.op("MaxPool", input, outputs=1, **kwargs)
            # 返回结果 r
            return r
    
    # 返回 symbolic_fn 函数作为最终结果
    return symbolic_fn
# 定义符号化的函数 max_pool1d_with_indices，使用 _onnx_symbolic 函数装饰
max_pool1d_with_indices = _onnx_symbolic("aten::max_pool1d_with_indices")(
    _max_pool(
        "max_pool1d_with_indices",
        torch.nn.modules.utils._single,  # 使用单元素元组作为参数的函数
        1,  # 池化操作维度为 1 维
        return_indices=True,  # 返回池化操作的索引
    )
)

# 定义符号化的函数 max_pool2d_with_indices，使用 _onnx_symbolic 函数装饰
max_pool2d_with_indices = _onnx_symbolic("aten::max_pool2d_with_indices")(
    _max_pool(
        "max_pool2d_with_indices",
        torch.nn.modules.utils._pair,  # 使用元组作为参数的函数
        2,  # 池化操作维度为 2 维
        return_indices=True,  # 返回池化操作的索引
    )
)

# 定义符号化的函数 max_pool3d_with_indices，使用 _onnx_symbolic 函数装饰
max_pool3d_with_indices = _onnx_symbolic("aten::max_pool3d_with_indices")(
    _max_pool(
        "max_pool3d_with_indices",
        torch.nn.modules.utils._triple,  # 使用三元组作为参数的函数
        3,  # 池化操作维度为 3 维
        return_indices=True,  # 返回池化操作的索引
    )
)

# 定义符号化的函数 _avg_pool，使用装饰器 @_onnx_symbolic 装饰，处理 avg_pool1d 操作
@_onnx_symbolic(
    "aten::avg_pool1d",
    decorate=[
        symbolic_helper._apply_params("avg_pool1d", torch.nn.modules.utils._single),  # 应用单元素元组参数的帮助函数
        _export("avg_pool1d"),  # 导出 avg_pool1d 操作
    ],
)
# 使用装饰器 @_onnx_symbolic 装饰，处理 avg_pool2d 操作
@_onnx_symbolic(
    "aten::avg_pool2d",
    decorate=[
        symbolic_helper._apply_params("avg_pool2d", torch.nn.modules.utils._pair),  # 应用元组参数的帮助函数
        _export("avg_pool2d"),  # 导出 avg_pool2d 操作
    ],
)
# 使用装饰器 @_onnx_symbolic 装饰，处理 avg_pool3d 操作
@_onnx_symbolic(
    "aten::avg_pool3d",
    decorate=[
        symbolic_helper._apply_params("avg_pool3d", torch.nn.modules.utils._triple),  # 应用三元组参数的帮助函数
        _export("avg_pool3d"),  # 导出 avg_pool3d 操作
    ],
)
# 定义私有函数 _avg_pool，用于池化操作，带有装饰器 @_beartype.beartype
@_beartype.beartype
def _avg_pool(name, tuple_fn):
    # 定义符号化的函数 symbolic_fn，用于计算平均池化操作的符号化表达
    @symbolic_helper.quantized_args(True)
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    @_beartype.beartype
    def symbolic_fn(
        g,  # 图形对象，用于构建符号化操作
        input: _C.Value,  # 输入数据的符号化值
        kernel_size: Sequence[int],  # 卷积核大小的序列
        stride: Sequence[int],  # 步幅大小的序列
        padding: Union[int, Sequence[int]],  # 填充大小或填充序列
        ceil_mode: int,  # 是否使用 ceil 模式的标志
        count_include_pad: int,  # 是否计算填充在内的标志
        divisor_override=None,  # 覆盖除数的值（可选）
        ):
            # 如果未指定步幅，则默认步幅为核大小
            if not stride:
                stride = kernel_size
            # 调用符号辅助函数，计算平均池化的填充值
            padding = symbolic_helper._avgpool_helper(
                tuple_fn, padding, kernel_size, stride, divisor_override, name
            )
            # 断言填充值为元组类型
            assert isinstance(padding, tuple)
            adjusted_padding = padding
            # 虽然 onnx::AvgPool 提供了 count_include_pad 参数，
            # 但 PyTorch 中带有 ceil_mode 的平均池化存在滑动窗口越界的情况，
            # 因此需要进行相应的调整
            # 更多细节参见 https://github.com/pytorch/pytorch/issues/57178
            if count_include_pad:
                # 如果需要包含填充的计数，则在输入周围进行填充
                input = symbolic_helper._op_with_optional_float_cast(
                    g,
                    "Pad",
                    input,
                    pads_i=((0,) * 2 + padding) * 2,
                    mode_s="constant",
                    value_f=0.0,
                    opset_before=11,
                )
                adjusted_padding = (0,) * len(padding)
            if ceil_mode:
                # 如果启用了 ceil_mode，则计算需要的填充值
                padding_ceil = get_pool_ceil_padding(input, kernel_size, stride, padding)
                adjusted_padding = adjusted_padding + tuple(
                    a + b for (a, b) in zip(padding_ceil, adjusted_padding)
                )
            else:
                # 否则直接将填充值加倍
                adjusted_padding = adjusted_padding * 2
            # 使用符号函数库的操作方法执行平均池化
            output = g.op(
                "AveragePool",
                input,
                kernel_shape_i=tuple_fn(kernel_size),
                strides_i=tuple_fn(stride),
                pads_i=adjusted_padding,
            )
            return output

        # 返回符号化函数
        return symbolic_fn
# 使用装饰器 @_onnx_symbolic 将函数注册为 ONNX 的符号化函数，处理 adaptive_avg_pool1d 操作
# decorate 列表包含两个元素，第一个元素调用 symbolic_helper._apply_params 函数设置参数，
# 第二个元素调用 _export 函数导出操作
@_onnx_symbolic(
    "aten::adaptive_avg_pool1d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_avg_pool1d", "AveragePool", torch.nn.modules.utils._single
        ),
        _export("adaptive_avg_pool1d"),
    ],
)

# 同上，注册 adaptive_avg_pool2d 操作的符号化函数
@_onnx_symbolic(
    "aten::adaptive_avg_pool2d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_avg_pool2d", "AveragePool", torch.nn.modules.utils._pair
        ),
        _export("adaptive_avg_pool2d"),
    ],
)

# 同上，注册 adaptive_avg_pool3d 操作的符号化函数
@_onnx_symbolic(
    "aten::adaptive_avg_pool3d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_avg_pool3d", "AveragePool", torch.nn.modules.utils._triple
        ),
        _export("adaptive_avg_pool3d"),
    ],
)

# 同上，注册 adaptive_max_pool1d 操作的符号化函数，需要提供 max_pool1d_with_indices 函数
@_onnx_symbolic(
    "aten::adaptive_max_pool1d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_max_pool1d",
            "MaxPool",
            torch.nn.modules.utils._single,
            max_pool1d_with_indices,
        ),
        _export("adaptive_max_pool1d"),
    ],
)

# 同上，注册 adaptive_max_pool2d 操作的符号化函数，需要提供 max_pool2d_with_indices 函数
@_onnx_symbolic(
    "aten::adaptive_max_pool2d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_max_pool2d",
            "MaxPool",
            torch.nn.modules.utils._pair,
            max_pool2d_with_indices,
        ),
        _export("adaptive_max_pool2d"),
    ],
)

# 同上，注册 adaptive_max_pool3d 操作的符号化函数，需要提供 max_pool3d_with_indices 函数
@_onnx_symbolic(
    "aten::adaptive_max_pool3d",
    decorate=[
        symbolic_helper._apply_params(
            "adaptive_max_pool3d",
            "MaxPool",
            torch.nn.modules.utils._triple,
            max_pool3d_with_indices,
        ),
        _export("adaptive_max_pool3d"),
    ],
)

# 使用装饰器 @_beartype.beartype 和 @_beartype.beartype 对 _adaptive_pool 函数进行修饰
def _adaptive_pool(name, type, tuple_fn, fn=None):
    # 使用装饰器 @symbolic_helper.quantized_args(True, False) 对函数进行修饰
    @symbolic_helper.quantized_args(True, False)
    # 返回修饰后的函数对象
    @_beartype.beartype
    def symbolic_fn(g, input, output_size):
        # 定义一个符号函数，用于符号化操作
        # _adaptive_pool 用于当输出大小在所有维度上都是1时执行全局池化。
        # 它还支持输出大小是输入大小的因子的情况。
        # 对于这些情况，步长和卷积核大小在同一维度的所有索引上都是统一的，
        # 这使得可以将其导出到 ONNX。
        # 对于 MaxPool，GlobalMaxPool 不返回索引，
        # 所以我们尝试使用 max_poolxd_with_indices，如果不可能
        # （输入不是完整的张量或输出大小不是输入大小的因子）
        # 然后我们调用 GlobalAveragePool，并且对于索引返回 None。
        
        # 将 output_size_value 初始化为 output_size 的值
        output_size_value = output_size
        try:
            # 尝试解析 output_size，期望其是一个列表
            output_size = symbolic_helper._parse_arg(output_size, "is")
        except Exception:
            # FIXME(justinchuby): 避免捕获 Exception。
            # 应该捕获一个更具体的异常。
            return symbolic_helper._onnx_unsupported(
                "adaptive pooling, since output_size is not constant.", input
            )
        
        # 如果 output_size 的每个维度都是 1，并且操作类型是 "AveragePool"
        if output_size == [1] * len(output_size) and type == "AveragePool":
            return g.op("GlobalAveragePool", input)
        
        # 获取输入张量的尺寸
        sizes = symbolic_helper._get_tensor_sizes(input)
        
        try:
            # 提取张量的维度信息（从第三维开始）
            dim = sizes[2:]
        except Exception:
            # FIXME(justinchuby): 避免捕获 Exception。
            # 应该捕获一个更具体的异常。
            dim = None
        
        # 如果 dim 是 None 或者任何一个维度信息是 None
        if dim is None or any(i is None for i in dim):
            if output_size == [1] * len(output_size):
                # 如果输出大小的每个维度都是 1，则执行全局最大池化，并返回 None 作为索引
                return g.op("GlobalMaxPool", input), None
            return symbolic_helper._unimplemented(
                name, "input size not accessible", input
            )
        
        # 验证输出大小是否能够整除输入大小的每个维度
        mod = [dim[i] % output_size[i] for i in range(0, len(dim))]
        if mod != [0] * len(mod):
            if output_size == [1] * len(output_size):
                # 如果输出大小的每个维度都是 1，则执行全局最大池化，并返回 None 作为索引
                return g.op("GlobalMaxPool", input), None
            return symbolic_helper._unimplemented(
                name, "output size that are not factor of input size", output_size_value
            )
        
        # 计算每个维度的 kernel size
        k = [int(dim[i] / output_size[i]) for i in range(0, len(dim))]
        
        # 如果操作类型是 "MaxPool"，则调用 fn 函数以获取输出中的索引
        if type == "MaxPool":
            return fn(g, input, k, k, (0,) * len(dim), (1,) * len(dim), False)
        
        # 否则，执行相应的池化操作，并返回输出结果
        output = g.op(type, input, kernel_shape_i=tuple_fn(k), strides_i=tuple_fn(k))
        return output

    return symbolic_fn
# 对输入参数进行类型检查和装饰，确保函数输入符合预期类型
@_beartype.beartype
# 准备 ONNX 格式的填充数据，根据 PyTorch 的填充格式
def _prepare_onnx_paddings(dim: int, pad):
    """Generate paddings in ONNX order based on pad in pytorch.
    Args:
        dim: the dimension of the tensor.
        pad: the paddings in pytorch.
            The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ...
    """
    # 扩展填充数据以适应 ONNX 的格式，如果填充数据不足则用0补充
    paddings = list(pad[:]) + [0] * (dim * 2 - len(pad))
    # 将填充数据按照 ONNX 的顺序重新排序，首先是开始填充，然后是结束填充
    paddings = paddings[-2::-2] + paddings[-1::-2]
    return paddings


# 对输入参数进行类型检查和装饰，确保函数输入符合预期类型
@_beartype.beartype
# 转换填充节点，从符号帮助器中获取常量填充值
def _convert_padding_node(input):
    padding = symbolic_helper._maybe_get_const(input, "is")
    # 如果填充是值且是打包列表类型，则解包列表
    if symbolic_helper._is_value(padding) and symbolic_helper._is_packed_list(padding):
        input_list = symbolic_helper._unpack_list(padding)
        try:
            # 获取每个填充值的常量值
            padding = [
                symbolic_helper._get_const(v, "i", "padding") for v in input_list
            ]
        except Exception:
            # FIXME(justinchuby): 避免捕获所有异常，应该捕获更具体的异常
            # 如果出现异常，返回详细的不支持的 ONNX 操作信息
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "Pad", 9, 11, "The sizes of the padding must be constant", input
            )
    return padding


# 对输入参数进行类型检查和装饰，确保函数输入符合预期类型
@_onnx_symbolic("aten::constant_pad_nd")
@_beartype.beartype
# 实现常量填充的 ONNX 符号化操作
def constant_pad_nd(g: jit_utils.GraphContext, input, padding, value):
    mode = "constant"
    try:
        # 获取填充值的常量值
        value = symbolic_helper._get_const(value, "f", "value")
    except Exception:
        # FIXME(justinchuby): 避免捕获所有异常，应该捕获更具体的异常
        # 如果出现异常，返回详细的不支持的 ONNX 操作信息
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "Pad", 9, 11, "The value for the padding must be constant", value
        )

    # 转换填充节点为 ONNX 格式的填充
    padding = _convert_padding_node(padding)
    # 准备 ONNX 格式的填充数据
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    # 执行带有可选浮点数转换的 ONNX 操作，返回符号化帮助器的操作结果
    return symbolic_helper._op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, value_f=value, opset_before=11
    )


# 对输入参数进行类型检查和装饰，确保函数输入符合预期类型
@_beartype.beartype
# 实现环形填充的 ONNX 符号化操作
def _pad_circular(g: jit_utils.GraphContext, input: _C.Value, pad: _C.Value):
    # 转换填充节点为 ONNX 格式的填充
    padding = _convert_padding_node(pad)
    # 断言填充数据长度必须是偶数
    assert len(padding) % 2 == 0
    # 计算输入数据的维度
    ndim = len(padding) // 2

    cur = input
    # 遍历每个维度的索引，进行填充操作
    for idx in range(ndim):
        # 获取当前维度上的右侧填充量
        pad_r = padding[-(2 * idx + 1)]
        # 获取当前维度上的左侧填充量
        pad_l = padding[-(2 * idx + 2)]
        # 初始化一个空列表用于存放切片后的张量
        tensors = []

        # 如果左侧填充量大于0，则进行左侧切片操作
        if pad_l > 0:
            left = symbolic_helper._slice_helper(
                g, cur, axes=[2 + idx], starts=[-(pad_l)], ends=[_constants.INT64_MAX]
            )
            tensors.append(left)

        # 如果左侧或右侧填充量小于0，则进行中间切片操作
        if pad_l < 0 or pad_r < 0:
            # 计算开始位置，确保不超出张量边界
            start = builtins.max(0, -pad_l)
            # 计算结束位置，确保不超出张量边界
            end = -(builtins.max(0, -pad_r))
            # 执行中间部分的切片操作
            middle = symbolic_helper._slice_helper(
                g,
                cur,
                axes=[2 + idx],
                starts=[start],
                ends=[end],
            )
            tensors.append(middle)
        else:
            # 如果不需要填充，则直接将当前张量加入列表
            tensors.append(cur)

        # 如果右侧填充量大于0，则进行右侧切片操作
        if pad_r > 0:
            right = symbolic_helper._slice_helper(
                g, cur, axes=[2 + idx], starts=[0], ends=[pad_r]
            )
            tensors.append(right)

        # 将切片后的张量按指定轴进行拼接
        cur = g.op("Concat", *tensors, axis_i=(2 + idx))

    # 返回最终拼接后的张量
    return cur
# 使用装饰器将函数注册为对应的 ONNX 符号化函数，处理 reflection_pad1d, reflection_pad2d, reflection_pad3d 操作
# 使用装饰器将函数注册为对应的 ONNX 符号化函数，处理 replication_pad1d, replication_pad2d, replication_pad3d 操作
@_onnx_symbolic("aten::reflection_pad1d")
@_onnx_symbolic("aten::reflection_pad2d")
@_onnx_symbolic("aten::reflection_pad3d")
@_beartype.beartype
def reflection_pad(g: jit_utils.GraphContext, input, padding):
    # 设定填充模式为 "reflect"
    mode = "reflect"
    # 将 padding 参数转换为标准格式
    padding = _convert_padding_node(padding)
    # 根据输入的张量和 padding 获取符号化的填充方式
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    # 调用符号化辅助函数执行带有可选浮点数转换的操作
    return symbolic_helper._op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, opset_before=11
    )


# 使用装饰器将函数注册为对应的 ONNX 符号化函数，处理 replication_pad1d, replication_pad2d, replication_pad3d 操作
@_onnx_symbolic("aten::replication_pad1d")
@_onnx_symbolic("aten::replication_pad2d")
@_onnx_symbolic("aten::replication_pad3d")
@_beartype.beartype
def replication_pad(g: jit_utils.GraphContext, input, padding):
    # 设定填充模式为 "edge"
    mode = "edge"
    # 将 padding 参数转换为标准格式
    padding = _convert_padding_node(padding)
    # 根据输入的张量和 padding 获取符号化的填充方式
    paddings = _prepare_onnx_paddings(symbolic_helper._get_tensor_rank(input), padding)
    # 调用符号化辅助函数执行带有可选浮点数转换的操作
    return symbolic_helper._op_with_optional_float_cast(
        g, "Pad", input, pads_i=paddings, mode_s=mode, opset_before=11
    )


# 使用装饰器将函数注册为对应的 ONNX 符号化函数，处理 aten::pad 操作
@_onnx_symbolic("aten::pad")
@_beartype.beartype
def pad(
    g: jit_utils.GraphContext,
    input: _C.Value,
    pad: _C.Value,
    mode: _C.Value,
    value: _C.Value,
):
    # 解析填充模式为字符串类型
    mode = symbolic_helper._parse_arg(mode, "s")
    # 根据填充模式选择相应的填充操作
    if mode == "replicate":
        return replication_pad(g, input, pad)
    elif mode == "reflect":
        return reflection_pad(g, input, pad)
    elif mode == "constant":
        return constant_pad_nd(g, input, pad, value)
    elif mode == "circular":
        return _pad_circular(g, input, pad)
    else:
        # 抛出填充模式错误异常
        raise errors.SymbolicValueError(f"Unrecognized padding mode {mode}", input)


# 使用装饰器将函数注册为对应的 ONNX 符号化函数，处理多种上采样操作
@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[
        symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest"),
        _export("upsample_nearest1d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[
        symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest"),
        _export("upsample_nearest2d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[
        symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest"),
        _export("upsample_nearest3d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[
        symbolic_helper._apply_params("upsample_linear1d", 3, "linear"),
        _export("upsample_linear1d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[
        symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear"),
        _export("upsample_bilinear2d"),
    ],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[
        symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear"),
        _export("upsample_trilinear3d"),
    ],
)
@_beartype.beartype
def _interpolate(name: str, dim: int, interpolate_mode: str):
    # 在 ONNX 符号化函数中，处理插值操作的名称、维度和插值模式
    # 定义一个符号函数 symbolic_fn，接受参数 g, input, output_size 和任意其他参数
    def symbolic_fn(g, input, output_size, *args):
        # 调用 symbolic_helper 模块的 _get_interpolate_attributes 方法，
        # 获取插值模式 scales 和 align_corners
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        # 调用 symbolic_helper 模块的 _interpolate_warning 方法，
        # 执行插值警告操作
        symbolic_helper._interpolate_warning(interpolate_mode)
        # 调用 symbolic_helper 模块的 _maybe_get_scalar 方法，
        # 尝试从 align_corners 获取一个标量值
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        # 如果 align_corners 为真，则调用 symbolic_helper 模块的 _unimplemented 方法，
        # 报告名称为 name 的操作未实现，因为 align_corners == True
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True", input)
        # 如果 scales 为 None，则调用 symbolic_helper 模块的 _interpolate_size_to_scales 方法，
        # 将插值大小转换为比例 scales
        if scales is None:
            scales = symbolic_helper._interpolate_size_to_scales(
                g, input, output_size, dim
            )
        # 返回一个使用 g.op 方法创建的 Upsample 操作，
        # 参数为 input, scales, mode_s=interpolate_mode
        return g.op("Upsample", input, scales, mode_s=interpolate_mode)
    
    # 返回定义的符号函数 symbolic_fn
    return symbolic_fn
# 使用装饰器声明函数为 ONNX 符号化函数，处理 "aten::__interpolate" 操作符
@_onnx_symbolic("aten::__interpolate")
# 使用 beartype 装饰器对函数进行类型检查和类型提示
@_beartype.beartype
# 定义 __interpolate 函数，接受多个参数，其中 g 是图上下文对象
def __interpolate(
    g: jit_utils.GraphContext,
    input,               # 输入参数，可能是张量或其他数据类型
    size,                # 插值操作的目标大小
    scale_factor,        # 插值尺度因子
    mode,                # 插值模式，如 'nearest' 或 'linear'
    align_corners,       # 是否在插值中保持角点对齐
    recompute_scale_factor,  # 是否重新计算尺度因子
    antialias,           # 是否进行抗锯齿处理
):
    # 调用 _interpolate_get_scales_and_mode 函数获取插值的尺度和模式
    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    # 使用 g.op 方法创建 Upsample 操作，对输入进行上采样，返回结果张量
    return g.op("Upsample", input, scales, mode_s=mode)


# 使用装饰器声明函数为 ONNX 符号化函数，处理 "aten::bitwise_not" 操作符
@_onnx_symbolic("aten::bitwise_not")
# 使用 beartype 装饰器对函数进行类型检查和类型提示
@_beartype.beartype
# 定义 bitwise_not 函数，接受图上下文对象 g 和输入参数 input
def bitwise_not(g: jit_utils.GraphContext, input):
    # 如果输入不是布尔类型，抛出 SymbolicValueError 异常
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise Not "
            "for non-boolean input values",
            input,
        )
    # 使用 g.op 方法创建 Not 操作，对布尔输入进行按位取反操作，返回结果张量
    return g.op("Not", input)


# 使用装饰器声明函数为 ONNX 符号化函数，处理 "aten::bitwise_or" 操作符
@_onnx_symbolic("aten::bitwise_or")
# 使用 beartype 装饰器对函数进行类型检查和类型提示
@_beartype.beartype
# 定义 bitwise_or 函数，接受图上下文对象 g、输入参数 self 和 other
def bitwise_or(g, self, other):
    # 如果 self 不是布尔类型，抛出 SymbolicValueError 异常
    if not symbolic_helper._is_bool(self):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values. self: ",
            self,
        )
    # 如果 other 不是布尔类型，抛出 SymbolicValueError 异常
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values. other: ",
            other,
        )
    # 使用 g.op 方法创建 Or 操作，对两个布尔输入进行按位或操作，返回结果张量
    return g.op("Or", self, other)


# 定义 wrap_logical_op_with_cast_to 函数，用于将操作装饰为将结果强制转换为指定类型的操作
@_beartype.beartype
def wrap_logical_op_with_cast_to(to_type):
    # 定义 decorator 函数作为 wrap_logical_op_with_cast_to 的内部函数
    def decorator(fn):
        # 定义 wrap_with_cast 函数作为 decorator 的内部函数
        @functools.wraps(fn)
        def wrap_with_cast(g, input, other):
            # 获取全局中名为 "_cast_{to_type}" 的函数，执行强制类型转换
            to_cast_func = globals()[f"_cast_{to_type}"]
            # 调用 fn 函数，对输入进行操作，并将结果强制转换为指定类型
            return fn(g, to_cast_func(g, input, False), to_cast_func(g, other, False))

        return wrap_with_cast

    return decorator


# 定义 wrap_logical_op_with_negation 函数，用于对逻辑操作进行包装，添加逻辑非操作
@_beartype.beartype
def wrap_logical_op_with_negation(func: Callable) -> Callable:
    # 定义 wrap_with_not 函数作为 wrap_logical_op_with_negation 的内部函数
    @functools.wraps(func)
    def wrap_with_not(g, input, other):
        # 对 func 函数进行操作，并对结果进行逻辑非操作
        return g.op("Not", func(g, input, other))

    return wrap_with_not


# 使用装饰器声明函数为 ONNX 符号化函数，处理 "aten::__not_" 操作符
@_onnx_symbolic("aten::__not_")
# 使用 beartype 装饰器对函数进行类型检查和类型提示
@_beartype.beartype
# 定义 __not_ 函数，接受图上下文对象 g 和输入参数 self
def __not_(g: jit_utils.GraphContext, self):
    # 如果输入不是布尔类型，抛出 SymbolicValueError 异常
    if not symbolic_helper._is_bool(self):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise Not "
            "for non-boolean input values",
            self,
        )
    # 使用 g.op 方法创建 Not 操作，对布尔输入进行按位取反操作，返回结果张量
    return g.op("Not", self)


# 使用装饰器声明函数为 ONNX 符号化函数，处理 "aten::eq" 操作符
@_onnx_symbolic("aten::eq")
# 装饰符号化助手函数，处理量化参数
@symbolic_helper.quantized_args(True, True)
# 使用 beartype 装饰器对函数进行类型检查和类型提示
@_beartype.beartype
# 定义 eq 函数，接受图上下文对象 g、输入参数 self 和 other
def eq(g: jit_utils.GraphContext, self, other):
    # 如果 self 和 other 均为 _C.DeviceObjType 类型，则认为它们是相等的
    if isinstance(self.type(), _C.DeviceObjType) and isinstance(
        other.type(), _C.DeviceObjType
    ):
        # ONNX 不支持设备类型，所以认为它们全部相等，常量折叠检查
        return g.op("Constant", value_t=torch.tensor(True, dtype=torch.bool))
    # 获取 self 和 other 的节点
    self_node = self.node()
    other_node = other.node()
    # 检查两个节点是否都是类型为 "onnx::Constant"
    if self_node.kind() == other_node.kind() == "onnx::Constant":
        # 检查两个节点的 "value" 属性是否都是字符串类型
        if self_node.kindOf("value") == other_node.kindOf("value") == "s":
            # 导出字符串到 ONNX 不被支持
            # 如果两个字符串都是常量，可以直接比较它们
            # 这个等式比较操作会被常量折叠优化
            return g.op(
                "Constant",
                # 创建一个包含布尔值的张量，表示两个字符串常量的相等性
                value_t=torch.tensor(
                    self_node.s("value") == other_node.s("value"),
                    dtype=torch.bool,
                ),
            )

    # 如果条件不符合上述要求，则返回将 self 和 other 比较的 Equal 操作节点
    return g.op("Equal", self, other)
# 定义一个符号化函数，处理 torch 中的 'aten::ne' 操作
@_onnx_symbolic("aten::ne")
# 处理量化参数为 True 的符号化辅助函数
@symbolic_helper.quantized_args(True, True)
# 将逻辑操作包装成带否定的形式
@wrap_logical_op_with_negation
# 对函数进行类型检查和参数验证的装饰器
@_beartype.beartype
# 定义 ne 函数，接收图上下文 g，以及 self 和 other 两个输入参数
def ne(g: jit_utils.GraphContext, self, other):
    # 调用 eq 函数，实现 self != other 的逻辑
    return eq(g, self, other)


# 定义一个符号化函数，处理 torch 中的 'aten::gt' 操作
@_onnx_symbolic("aten::gt")
# 处理量化参数为 True 的符号化辅助函数
@symbolic_helper.quantized_args(True, True)
# 对函数进行类型检查和参数验证的装饰器
@_beartype.beartype
# 定义 gt 函数，接收图上下文 g，以及 input 和 other 两个输入参数
def gt(g: jit_utils.GraphContext, input, other):
    # 调用 _gt_impl 函数，实现 input > other 的逻辑
    return _gt_impl(g, input, other)


# 定义一个内部函数，实现数值比较操作的具体逻辑
@_beartype.beartype
# 定义 _gt_impl 函数，接收图上下文 g，以及 input 和 other 两个输入参数
def _gt_impl(g: jit_utils.GraphContext, input, other):
    # 如果 input 和 other 都是布尔值
    if symbolic_helper._is_bool(input) and symbolic_helper._is_bool(other):
        # 将 input 和 other 分别转换为 INT32 类型
        input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT32)
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.INT32)
    # 返回比较结果，调用 Greater 操作
    return g.op("Greater", input, other)


# 定义一个符号化函数，处理 torch 中的 'aten::lt' 操作
@_onnx_symbolic("aten::lt")
# 处理量化参数为 True 的符号化辅助函数
@symbolic_helper.quantized_args(True, True)
# 对函数进行类型检查和参数验证的装饰器
@_beartype.beartype
# 定义 lt 函数，接收图上下文 g，以及 input 和 other 两个输入参数
def lt(g: jit_utils.GraphContext, input, other):
    # 调用 _lt_impl 函数，实现 input < other 的逻辑
    return _lt_impl(g, input, other)


# 定义一个内部函数，实现数值比较操作的具体逻辑
@_beartype.beartype
# 定义 _lt_impl 函数，接收图上下文 g，以及 input 和 other 两个输入参数
def _lt_impl(g: jit_utils.GraphContext, input, other):
    # 如果 input 和 other 都是布尔值
    if symbolic_helper._is_bool(input) and symbolic_helper._is_bool(other):
        # 将 input 和 other 分别转换为 INT32 类型
        input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT32)
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.INT32)
    # 返回比较结果，调用 Less 操作
    return g.op("Less", input, other)


# 定义一个符号化函数，处理 torch 中的 'aten::ge' 操作
@_onnx_symbolic("aten::ge")
# 处理量化参数为 True 的符号化辅助函数
@symbolic_helper.quantized_args(True, True)
# 将逻辑操作包装成带否定的形式
@wrap_logical_op_with_negation
# 对函数进行类型检查和参数验证的装饰器
@_beartype.beartype
# 定义 ge 函数，接收图上下文 g，以及 input 和 other 两个输入参数
def ge(g: jit_utils.GraphContext, input, other):
    # 调用 _lt_impl 函数，实现 input >= other 的逻辑
    return _lt_impl(g, input, other)


# 定义一个符号化函数，处理 torch 中的 'aten::le' 操作
@_onnx_symbolic("aten::le")
# 处理量化参数为 True 的符号化辅助函数
@symbolic_helper.quantized_args(True, True)
# 将逻辑操作包装成带否定的形式
@wrap_logical_op_with_negation
# 对函数进行类型检查和参数验证的装饰器
@_beartype.beartype
# 定义 le 函数，接收图上下文 g，以及 input 和 other 两个输入参数
def le(g: jit_utils.GraphContext, input, other):
    # 调用 _gt_impl 函数，实现 input <= other 的逻辑
    return _gt_impl(g, input, other)


# 定义一个符号化函数，处理 torch 中的 'aten::__and_' 操作
@_onnx_symbolic("aten::__and_")
# 对函数进行类型检查和参数验证的装饰器
@_beartype.beartype
# 定义 __and_ 函数，接收图上下文 g，以及 input 和 other 两个输入参数
def __and_(g: jit_utils.GraphContext, input, other):
    # 如果 input 不是布尔值，抛出异常
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise AND "
            "for non-boolean input values",
            input,
        )
    # 如果 other 不是布尔值，抛出异常
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise AND "
            "for non-boolean input values",
            other,
        )
    # 返回按位与操作的结果，调用 And 操作
    return g.op("And", input, other)


# 定义一个符号化函数，处理 torch 中的 'aten::__or_' 操作
@_onnx_symbolic("aten::__or_")
# 对函数进行类型检查和参数验证的装饰器
@_beartype.beartype
# 定义 __or_ 函数，接收图上下文 g，以及 input 和 other 两个输入参数
def __or_(g: jit_utils.GraphContext, input, other):
    # 如果 input 不是布尔值，抛出异常
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values",
            input,
        )
    # 如果 other 不是布尔值，抛出异常
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise OR "
            "for non-boolean input values",
            other,
        )
    # 返回按位或操作的结果，调用 Or 操作
    return g.op("Or", input, other)


# 定义一个符号化函数，处理 torch 中的 'aten::__xor_' 操作
@_onnx_symbolic("aten::__xor_")
# 对函数进行类型检查和参数验证的装饰器
@_beartype.beartype
# 定义 __xor_ 函数，接收图上下文 g，以及 input 和 other 两个输入参数
def __xor_(g: jit_utils.GraphContext, input, other):
    # 检查 input 是否为布尔类型，若不是则抛出异常
    if not symbolic_helper._is_bool(input):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise XOR "
            "for non-boolean input values",
            input,
        )
    # 检查 other 是否为布尔类型，若不是则抛出异常
    if not symbolic_helper._is_bool(other):
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting bitwise XOR "
            "for non-boolean input values",
            other,
        )
    # 使用 ONNX 操作符 "Xor" 对 input 和 other 执行按位异或操作，并返回结果
    return g.op("Xor", input, other)
# 定义一个对 ONNX 符号化的装饰器，处理逻辑与操作
@_onnx_symbolic("aten::logical_and")
# 使用装饰器将逻辑与操作包装，确保输入被强制转换为布尔类型
@wrap_logical_op_with_cast_to("Bool")
# 应用 Beartype 装饰器，用于类型检查
@_beartype.beartype
# 定义逻辑与操作的函数，返回对应的 ONNX 操作节点
def logical_and(g: jit_utils.GraphContext, input, other):
    return g.op("And", input, other)


# 定义一个对 ONNX 符号化的装饰器，处理逻辑或操作
@_onnx_symbolic("aten::logical_or")
# 使用装饰器将逻辑或操作包装，确保输入被强制转换为布尔类型
@wrap_logical_op_with_cast_to("Bool")
# 应用 Beartype 装饰器，用于类型检查
@_beartype.beartype
# 定义逻辑或操作的函数，返回对应的 ONNX 操作节点
def logical_or(g: jit_utils.GraphContext, input, other):
    return g.op("Or", input, other)


# 定义一个对 ONNX 符号化的装饰器，处理逻辑异或操作
@_onnx_symbolic("aten::logical_xor")
# 使用装饰器将逻辑异或操作包装，确保输入被强制转换为布尔类型
@wrap_logical_op_with_cast_to("Bool")
# 应用 Beartype 装饰器，用于类型检查
@_beartype.beartype
# 定义逻辑异或操作的函数，返回对应的 ONNX 操作节点
def logical_xor(g: jit_utils.GraphContext, input, other):
    return g.op("Xor", input, other)


# 定义一个对 ONNX 符号化的装饰器，处理逻辑非操作
@_onnx_symbolic("aten::logical_not")
# 应用 Beartype 装饰器，用于类型检查
@_beartype.beartype
# 定义逻辑非操作的函数，返回对应的 ONNX 操作节点
def logical_not(g: jit_utils.GraphContext, input):
    # 在 ONNX 中执行 Cast 操作将输入转换为布尔类型
    return g.op("Not", g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.BOOL))


# 定义一个对 ONNX 符号化的装饰器，处理右移操作
@_onnx_symbolic("aten::__rshift_")
# 应用 Beartype 装饰器，用于类型检查
@_beartype.beartype
# 定义右移操作的函数，返回对应的 ONNX 操作节点
def __rshift_(g: jit_utils.GraphContext, self, other):
    # 确保将 other 强制转换为 self 的类型
    self_scalar_type = _type_utils.JitScalarType.from_value(self)
    if (
        _type_utils.JitScalarType.from_value(other, _type_utils.JitScalarType.UNDEFINED)
        != self_scalar_type
    ):
        other = g.op(
            "Cast",
            other,
            to_i=self_scalar_type.onnx_type(),
        )

    # 创建一个常量节点，值为 2，类型为 torch.float32
    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # 如果 self 不是浮点数类型，将 other 转换为浮点数类型
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    # 计算 2 的 other 次幂
    two_pow = g.op("Pow", two, other)
    # 将结果转换为 self 的类型
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=self_scalar_type.onnx_type(),
    )
    # 执行除法操作，计算右移结果
    rshift = g.op("Div", self, two_pow)
    return rshift


# 定义一个对 ONNX 符号化的装饰器，处理左移操作
@_onnx_symbolic("aten::__lshift_")
# 应用 Beartype 装饰器，用于类型检查
@_beartype.beartype
# 定义左移操作的函数，返回对应的 ONNX 操作节点
def __lshift_(g: jit_utils.GraphContext, self, other):
    # 确保将 other 强制转换为 self 的类型
    self_scalar_type = _type_utils.JitScalarType.from_value(self)
    if (
        _type_utils.JitScalarType.from_value(other, _type_utils.JitScalarType.UNDEFINED)
        != self_scalar_type
    ):
        other = g.op(
            "Cast",
            other,
            to_i=self_scalar_type.onnx_type(),
        )

    # 创建一个常量节点，值为 2，类型为 torch.float32
    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # 如果 self 不是浮点数类型，将 other 转换为浮点数类型
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    # 计算 2 的 other 次幂
    two_pow = g.op("Pow", two, other)
    # 将结果转换为 self 的类型
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=self_scalar_type.onnx_type(),
    )
    # 执行乘法操作，计算左移结果
    lshift = g.op("Mul", self, two_pow)
    return lshift


# 定义一个对 ONNX 符号化的装饰器，处理条件选择操作
@_onnx_symbolic("aten::where")
# 使用装饰器解析参数，并指定参数类型和数量
@symbolic_helper.parse_args("v", "v", "v", "i")
# 应用 Beartype 装饰器，用于类型检查
@_beartype.beartype
# 定义条件选择操作的函数，返回对应的 ONNX 操作节点
def where(g: jit_utils.GraphContext, condition, self=None, other=None, _outputs=None):
    # 如果条件不是布尔类型或字节类型张量，则进行类型转换为布尔类型
    if not symbolic_helper._is_bool(condition):
        condition = g.op("Cast", condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    # 如果当前对象为None，则对条件进行非零元素索引操作
    if self is None:
        condition = nonzero(g, condition)
        # 调用符号助手的解绑辅助函数，返回解绑后的结果
        return symbolic_helper._unbind_helper(
            g, condition, g.op("Constant", value_t=torch.tensor(1)), _outputs
        )
    # 返回条件为真时返回self，条件为假时返回other的操作
    return g.op("Where", condition, self, other)
# 使用装饰器定义 ONNX 符号化函数，处理 log_softmax 操作
@_onnx_symbolic("aten::log_softmax")
# 解析参数装饰器，指定参数类型和是否可选的情况
@symbolic_helper.parse_args("v", "i", "none")
# 添加类型检查装饰器
@_beartype.beartype
# 定义 log_softmax 函数，接收图上下文 g、输入 input、维度 dim 和数据类型 dtype（可选）
def log_softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    # PyTorch 的 dim 和 ONNX 的 axis 具有不同的含义。
    # 参见 Softmax 的注释了解详细信息。
    # TODO: 随着 ONNX opset 11 规范的变化，应该移除这段代码
    # 获取输入张量的维度
    input_dim = symbolic_helper._get_tensor_rank(input)
    if input_dim is None:
        # 如果无法获取输入维度，返回未实现的错误信息
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX 和 PyTorch 在分割输入时采用不同的策略。"
            "导出时必须知道输入的秩。",
        )
    # 如果 dim 是负数，转换为对应的非负数索引
    if dim < 0:
        dim = input_dim + dim
    # 检查是否需要进行转置操作
    is_transpose_required = input_dim != dim + 1
    # 如果需要转置，则重新排列轴以支持 ONNX 中除了 dim = -1 外的其他情况
    if is_transpose_required:
        axes = list(range(input_dim))
        axes[dim], axes[-1] = axes[-1], axes[dim]
        input = g.op("Transpose", input, perm_i=axes)
        dim = input_dim - 1
    # 使用 ONNX 操作符 LogSoftmax 计算对数 softmax
    return_op = g.op("LogSoftmax", input, axis_i=dim)
    # 如果指定了 dtype 并且不是常量节点，则进行类型转换
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        return_op = g.op(
            "Cast", return_op, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    # 如果之前进行了转置操作，则在返回前再次进行逆转置
    if is_transpose_required:
        return_op = g.op("Transpose", return_op, perm_i=axes)  # type: ignore[possibly-undefined]
    # 返回 LogSoftmax 操作的结果
    return return_op


# 使用装饰器定义 ONNX 符号化函数，处理 _log_softmax 操作
@_onnx_symbolic("aten::_log_softmax")
# 解析参数装饰器，指定参数类型
@symbolic_helper.parse_args("v", "i", "i")
# 添加类型检查装饰器
@_beartype.beartype
# 定义 _log_softmax 函数，接收图上下文 g、输入 input、维度 dim 和 half_to_float 标志
def _log_softmax(g: jit_utils.GraphContext, input, dim, half_to_float):
    # 如果 half_to_float 为真，并且输入是半精度浮点数，则转换为单精度浮点数
    if (
        half_to_float
        and _type_utils.JitScalarType.from_value(
            input, _type_utils.JitScalarType.UNDEFINED
        )
        == _type_utils.JitScalarType.HALF
    ):
        input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    # 调用 log_softmax 函数计算对数 softmax
    return log_softmax(g, input, dim)


# 使用装饰器定义 ONNX 符号化函数，处理 _convolution 操作
@_onnx_symbolic("aten::_convolution")
# 解析参数装饰器，指定所有参数类型
@symbolic_helper.parse_args(
    "v", "v", "v", "is", "is", "is", "i", "is", "i", "i", "i", "i", "i"
)
# 添加类型检查装饰器
@_beartype.beartype
# 定义 _convolution 函数，接收图上下文 g、输入 input、权重 weight、偏置 bias、步幅 stride、填充 padding、
# 膨胀 dilation、是否转置 transposed、输出填充 output_padding、组数 groups、基准 benchmark、确定性 deterministic、
# 是否启用 cuDNN cudnn_enabled、是否允许 TF32 模式 allow_tf32（可选）
def _convolution(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32=None,
):
    # 获取权重张量的尺寸
    weight_size = symbolic_helper._get_tensor_sizes(weight)
    try:
        # 尝试获取卷积核形状
        kernel_shape = weight_size[2:]
    except Exception:
        # 如果出现异常，抛出符号化值错误
        kernel_shape = None

    if kernel_shape is None or any(i is None for i in kernel_shape):
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of convolution for kernel of unknown shape.",
            input,
        )

    # 构造参数列表
    args = [input, weight]
    # ONNX 只支持 1D 偏置
    # 如果偏置项不为None，并且偏置项是一维张量
    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) == 1
    ):
        # 将偏置项添加到参数列表中
        args.append(bias)

    # 构建关键字参数字典
    kwargs = {
        "kernel_shape_i": weight_size[2:],  # 卷积核形状，从第2维开始到最后
        "strides_i": stride,  # 卷积步长
        # 注意：ONNX支持不对称填充，而PyTorch仅支持对称填充
        "pads_i": padding + padding,  # 填充
        "dilations_i": dilation,  # 卷积扩展
        "group_i": groups,  # 分组卷积数
    }

    # 如果输出填充中有任何非零元素
    if any(o != 0 for o in output_padding):
        # 对于转置卷积操作，ONNX支持output_shape和output_padding两种方式，它们在表达上是等效的
        # output_padding更加直观，因此在这里使用它
        assert transposed  # 断言是否为转置卷积操作
        assert len(stride) == len(output_padding)  # 断言步长和输出填充长度相同
        kwargs["output_padding_i"] = output_padding  # 输出填充参数

    # 创建ONNX图操作节点，根据是否为转置卷积选择ConvTranspose或Conv
    n = g.op("ConvTranspose" if transposed else "Conv", *args, **kwargs)

    # 如果偏置项不为None，并且偏置项不是一维张量
    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) != 1
    ):
        # 返回加法操作节点，将卷积结果和偏置项相加
        return g.op("Add", n, bias)
    else:
        # 否则，直接返回卷积操作节点
        return n
# 将函数注册为 ONNX 符号化函数，处理 aten::_convolution_mode 操作
@_onnx_symbolic("aten::_convolution_mode")
# 使用装饰器 parse_args 解析函数参数，期望的参数类型依次为：
# "v" 表示任意类型，"is" 表示整数或字符串，"s" 表示字符串
@symbolic_helper.parse_args(
    "v",
    "v",
    "v",
    "is",
    "s",
    "is",
    "i",
)
# 使用装饰器 beartype，进行参数类型检查
@_beartype.beartype
# 定义 _convolution_mode 函数，接收以下参数：
# g: jit_utils.GraphContext，表示计算图上下文
# input: 输入数据
# weight: 卷积核权重
# bias: 偏置项
# stride: 步长
# padding: 填充方式
# dilation: 膨胀率
# groups: 分组卷积的组数
def _convolution_mode(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
):
    # 获取权重的尺寸信息
    weight_size = symbolic_helper._get_tensor_sizes(weight)
    try:
        # 尝试获取卷积核形状
        kernel_shape = weight_size[2:]
    except Exception:
        # 捕获所有异常并记录警告，应避免使用通用的 Exception 类
        # 应使用更具体的异常类型
        kernel_shape = None

    # 如果卷积核形状未知或任何维度为 None，则抛出符号化数值错误
    if kernel_shape is None or any(i is None for i in kernel_shape):
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of convolution for kernel of unknown shape.",
            input,
        )

    # 初始化参数列表为输入数据和权重
    args = [input, weight]
    # 如果偏置项不为 None 并且偏置项是一维的，则添加到参数列表中
    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) == 1
    ):
        args.append(bias)

    # 将填充方式转换为 ONNX 支持的命名
    if padding == "valid":
        padding = "VALID"
    elif padding == "same":
        padding = "SAME_UPPER"
    
    # 定义关键字参数字典，包括卷积核形状、步长、填充、膨胀率和分组数
    kwargs = {
        "kernel_shape_i": weight_size[2:],
        "strides_i": stride,
        "auto_pad_s": padding,
        "dilations_i": dilation,
        "group_i": groups,
    }

    # 在计算图上执行卷积操作 "Conv"，传入参数列表和关键字参数字典
    n = g.op("Conv", *args, **kwargs)

    # 如果偏置项不为 None 并且偏置项不是一维的，则执行加法操作 "Add"
    if (
        not symbolic_helper._is_none(bias)
        and symbolic_helper._get_tensor_rank(bias) != 1
    ):
        return g.op("Add", n, bias)
    else:
        return n


# 将函数注册为 ONNX 符号化函数，处理 aten::convolution 操作
@_onnx_symbolic("aten::convolution")
# 使用装饰器 parse_args 解析函数参数，期望的参数类型依次为：
# "v" 表示任意类型，"is" 表示整数或字符串，"i" 表示整数
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is", "i")
# 使用装饰器 beartype，进行参数类型检查
@_beartype.beartype
# 定义 convolution 函数，接收以下参数：
# g: jit_utils.GraphContext，表示计算图上下文
# input: 输入数据
# weight: 卷积核权重
# bias: 偏置项
# stride: 步长
# padding: 填充方式
# dilation: 膨胀率
# transposed: 是否是反卷积
# output_padding: 输出填充
# groups: 分组卷积的组数
def convolution(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
):
    # 调用 _convolution 函数进行卷积操作
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


# 将函数注册为 ONNX 符号化函数，处理 aten::conv1d 操作
@_onnx_symbolic("aten::conv1d")
# 使用装饰器 parse_args 解析函数参数，期望的参数类型依次为：
# "v" 表示任意类型，"is" 表示整数或字符串，"i" 表示整数
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
# 使用装饰器 beartype，进行参数类型检查
@_beartype.beartype
# 定义 conv1d 函数，接收以下参数：
# g: jit_utils.GraphContext，表示计算图上下文
# input: 输入数据
# weight: 卷积核权重
# bias: 偏置项
# stride: 步长
# padding: 填充方式
# dilation: 膨胀率
# groups: 分组卷积的组数
def conv1d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    # 将填充方式解析为字符串
    str_padding = symbolic_helper._parse_arg(padding, "s")
    # 如果填充方式为 "valid" 或 "same"，则调用 _convolution_mode 进行卷积模式操作
    if str_padding in ["valid", "same"]:
        return _convolution_mode(
            g,
            input,
            weight,
            bias,
            stride,
            str_padding,
            dilation,
            groups,
        )
    else:
        # 否则，将填充方式解析为整数并调用 _convolution 函数进行一般卷积操作
        padding = symbolic_helper._parse_arg(padding, "is")
        return _convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            (),
            groups,
            None,
            None,
            None,
            None,
        )
# 注册一个名为 conv2d 的函数，用于执行二维卷积操作
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
@_beartype.beartype
def conv2d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    # 将 padding 参数解析为字符串类型
    str_padding = symbolic_helper._parse_arg(padding, "s")
    # 如果解析后的字符串 padding 在 ["valid", "same"] 中，则执行特定模式的卷积操作
    if str_padding in ["valid", "same"]:
        return _convolution_mode(
            g,
            input,
            weight,
            bias,
            stride,
            str_padding,
            dilation,
            groups,
        )
    else:
        # 否则，将 padding 参数解析为整数数组
        padding = symbolic_helper._parse_arg(padding, "is")
        # 执行普通的卷积操作
        return _convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            (),
            groups,
            None,
            None,
            None,
            None,
        )


# 注册一个名为 conv3d 的函数，用于执行三维卷积操作
@_onnx_symbolic("aten::conv3d")
@symbolic_helper.parse_args("v", "v", "v", "is", "v", "is", "i")
@_beartype.beartype
def conv3d(
    g: jit_utils.GraphContext, input, weight, bias, stride, padding, dilation, groups
):
    # 将 padding 参数解析为字符串类型
    str_padding = symbolic_helper._parse_arg(padding, "s")
    # 如果解析后的字符串 padding 在 ["valid", "same"] 中，则执行特定模式的卷积操作
    if str_padding in ["valid", "same"]:
        return _convolution_mode(
            g,
            input,
            weight,
            bias,
            stride,
            str_padding,
            dilation,
            groups,
        )
    else:
        # 否则，将 padding 参数解析为整数数组
        padding = symbolic_helper._parse_arg(padding, "is")
        # 执行普通的卷积操作
        return _convolution(
            g,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            (),
            groups,
            None,
            None,
            None,
            None,
        )


# 注册一个名为 conv_transpose1d 的函数，用于执行一维转置卷积操作
@_onnx_symbolic("aten::conv_transpose1d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
@_beartype.beartype
def conv_transpose1d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    # 执行转置卷积操作
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


# 注册一个名为 conv_transpose2d 的函数，用于执行二维转置卷积操作
@_onnx_symbolic("aten::conv_transpose2d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
@_beartype.beartype
def conv_transpose2d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    # 执行转置卷积操作
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )


# 注册一个名为 conv_transpose3d 的函数，用于执行三维转置卷积操作
@_onnx_symbolic("aten::conv_transpose3d")
@symbolic_helper.parse_args("v", "v", "v", "is", "is", "is", "i", "is")
@_beartype.beartype
def conv_transpose3d(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    stride,
    padding,
    output_padding,
    groups,
    dilation,
):
    # 执行转置卷积操作
    return _convolution(
        g,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        True,
        output_padding,
        groups,
        None,
        None,
        None,
        None,
    )
    g: jit_utils.GraphContext,  # 定义变量 g，类型为 jit_utils.GraphContext，表示图形上下文对象
    input,                     # 输入张量，通常是网络层的输入数据
    weight,                    # 权重张量，用于卷积操作
    bias,                      # 偏置张量，用于卷积操作中的偏置加法
    stride,                    # 步长参数，用于卷积操作指定滑动窗口的步长大小
    padding,                   # 填充参数，用于卷积操作中控制输入张量的填充大小
    output_padding,            # 输出填充参数，用于转置卷积操作控制输出张量的填充大小
    groups,                    # 分组参数，用于深度卷积中将输入和输出通道分成多个组进行计算
    dilation,                  # 膨胀参数，用于卷积操作中指定卷积核元素之间的间隔大小
# 定义批量归一化操作函数，用于 ONNX 符号化
@_onnx_symbolic("aten::batch_norm")
# 解析参数注解，指定每个参数的类型和标记
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
# 使用 Beartype 进行类型检查和注解
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
    # 检查训练模式是否正确
    symbolic_helper.check_training_mode(training, "batch_norm")

    # 如果启用了自动混合精度并且输入的张量类型不同，并且当前 ONNX opset 版本低于 15
    if (
        torch.is_autocast_enabled()
        and not symbolic_helper.args_have_same_dtype(
            [input, weight, bias, running_mean, running_var]
        )
        and GLOBALS.export_onnx_opset_version < 15
    ):
        # 返回 ONNX 操作不支持的详细信息，要求所有输入张量必须具有相同的 dtype
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "BatchNormalization",
            9,
            15,
            "All input tensors must have the same `dtype`. "
            "Turn off Autocast or export using opset version 15.",
            input,
        )

    # 使用符号化助手函数处理权重、偏置、运行均值和方差
    weight, bias, running_mean, running_var = symbolic_helper._batchnorm_helper(
        g, input, weight, bias, running_mean, running_var
    )
    # 在图中添加 BatchNormalization 操作节点
    out = g.op(
        "BatchNormalization",
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon_f=eps,
        momentum_f=1 - momentum,
        outputs=1 if not training else 5,
    )
    # 如果不是训练模式，则直接返回输出
    if not training:
        return out
    else:
        # 否则，解构输出结果
        res, new_running_mean, new_running_var, saved_mean, saved_var = out
        # 设置新的运行均值和方差的类型与原始运行均值和方差相同
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        # 设置保存的均值和方差的调试名称，用于调试目的
        saved_mean.setDebugName("batch_norm_dead_output-" + saved_mean.debugName())
        saved_var.setDebugName("batch_norm_dead_output-" + saved_var.debugName())
        # 返回计算结果 res
        return res
    # 如果是半精度类型，则根据 eps_cst 创建一个对应的 JitScalarType
    eps_dtype = _type_utils.JitScalarType.from_value(eps_cst)
    # 使用 Cast 操作将 numerator 转换为 eps_dtype 对应的 ONNX 类型
    numerator = g.op(
        "Cast", numerator, to_i=_type_utils.JitScalarType(eps_dtype).onnx_type()
    )

# variance = e((x - e(x))^2), 而 (x - e(x)) 是 layer_norm 公式中的 numerator
if g.opset < 18:
    # 如果 opset 小于 18，使用 ReduceMean 操作计算方差
    variance = g.op("ReduceMean", pow(g, numerator, two_cst), axes_i=axes)
else:
    # 如果 opset 大于等于 18，使用 ReduceMean 和 Constant 操作计算方差
    variance = g.op(
        "ReduceMean",
        pow(g, numerator, two_cst),
        g.op("Constant", value_t=torch.tensor(axes, dtype=torch.long)),
    )

# 计算 denominator = sqrt(variance + eps_cst)
denominator = sqrt(g, g.op("Add", variance, eps_cst))
# 计算 normalized = numerator / denominator
normalized = g.op("Div", numerator, denominator)

# 如果是半精度类型，则将 normalized 转换回输入类型
if is_type_half:
    input_dtype = _type_utils.JitScalarType.from_value(input)
    normalized = g.op(
        "Cast", normalized, to_i=_type_utils.JitScalarType(input_dtype).onnx_type()
    )

# 如果 weight 不为 None，则将 normalized 乘以 weight
if not (weight is None or symbolic_helper._is_none(weight)):
    normalized = mul(g, normalized, weight)
# 如果 bias 不为 None，则将 normalized 加上 bias
if not (bias is None or symbolic_helper._is_none(bias)):
    normalized = add(g, normalized, bias)

# rdenominator := 1 / sqrt(variance + eps)
# 根据 aten::native_layer_norm，rdenominator 的 dtype 应与 input、mean 和 normalized 相同，因此需要将其转换回相同的类型
if is_type_half:
    # 如果是半精度类型，则将 denominator 转换为 input_dtype 对应的 ONNX 类型
    denominator = g.op(
        "Cast", denominator, to_i=_type_utils.JitScalarType(input_dtype).onnx_type()  # type: ignore[possibly-undefined]
    )
    # 计算 rdenominator = 1 / denominator
    rdenominator = g.op("Reciprocal", denominator)
else:
    # 如果不是半精度类型，则直接计算 rdenominator = 1 / denominator
    rdenominator = reciprocal(g, denominator)

# 返回计算得到的 normalized、mean 和 rdenominator
return normalized, mean, rdenominator
# 使用装饰器将函数注册为 ONNX 符号化函数，处理 "aten::layer_norm" 操作
# 同时为函数添加量化参数，指定了参数的布尔值
# 解析函数参数类型为 "v", "is", "v", "v", "f", "b"
# 应用 Beartype 运行时类型检查装饰器
@_onnx_symbolic("aten::layer_norm")
@symbolic_helper.quantized_args(True, False, False, False)
@symbolic_helper.parse_args("v", "is", "v", "v", "f", "b")
@_beartype.beartype
def layer_norm(
    g: jit_utils.GraphContext,
    input: _C.Value,
    normalized_shape: Sequence[int],
    weight: _C.Value,
    bias: _C.Value,
    eps: float,
    cudnn_enable: bool,
) -> _C.Value:
    # 调用原生的层归一化函数 native_layer_norm 处理归一化操作
    normalized, _, _ = native_layer_norm(g, input, normalized_shape, weight, bias, eps)
    # 返回归一化结果
    return normalized


# 使用装饰器将函数注册为 ONNX 符号化函数，处理 "aten::instance_norm" 操作
# 解析函数参数类型为 "v", "v", "v", "v", "v", "b", "f", "f", "b"
# 应用 Beartype 运行时类型检查装饰器
@_onnx_symbolic("aten::instance_norm")
@symbolic_helper.parse_args("v", "v", "v", "v", "v", "b", "f", "f", "b")
@_beartype.beartype
def instance_norm(
    g: jit_utils.GraphContext,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    use_input_stats: bool,
    momentum: Number,
    eps: Number,
    cudnn_enabled: bool,
):
    # 检查训练模式是否开启，用于实例归一化
    symbolic_helper.check_training_mode(use_input_stats, "instance_norm")
    # 获取输入张量的通道大小
    channel_size = symbolic_helper._get_tensor_dim_size(input, 1)
    
    # 如果权重为 None 或者未定义，则创建默认值权重张量
    if weight is None or symbolic_helper._is_none(weight):
        if channel_size is None:
            # 如果通道大小未知，则抛出符号化值错误
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of instance_norm for unknown channel size.",
                input,
            )
        # 创建权重张量，默认为全1
        weight_value = torch.tensor(
            [1.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )
        weight = g.op("Constant", value_t=weight_value)
    
    # 如果偏置为 None 或者未定义，则创建默认值偏置张量
    if bias is None or symbolic_helper._is_none(bias):
        if channel_size is None:
            # 如果通道大小未知，则抛出符号化值错误
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of instance_norm for unknown channel size.",
                input,
            )
        # 创建偏置张量，默认为全0
        bias_value = torch.tensor(
            [0.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )
        bias = g.op("Constant", value_t=bias_value)
    
    # 如果运行时均值或方差为 None 或未定义，则调用实例归一化操作
    if (
        running_mean is None
        or symbolic_helper._is_none(running_mean)
        or running_var is None
        or symbolic_helper._is_none(running_var)
    ):
        return g.op("InstanceNormalization", input, weight, bias, epsilon_f=eps)
    else:
        # 获取输入张量的符号化尺寸
        input_size = symbolic_helper._get_tensor_sizes(input)
        
        # 如果输入形状是 [N, C, H, W]，则重塑为 [1, N * C, H, W] 并调用 batch_norm。
        # 更多关于 instance_norm() 的信息：
        # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L542
        
        # 复制输入尺寸以便修改
        input_size_reshape = input_size.copy()
        n = input_size[0]
        
        # 如果批大小为未知，则抛出符号值错误
        if n is None:
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of instance_norm training for unknown "
                "batch size.",
                input,
            )
        
        c = input_size[1]
        
        # 修改重塑后的输入尺寸
        input_size_reshape[0] = 1
        input_size_reshape[1] = n * c
        
        # 使用 repeat 函数重复权重、偏置、运行均值和运行方差
        weight_ = repeat(
            g, weight, g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64))
        )
        bias_ = repeat(
            g, bias, g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64))
        )
        running_mean_ = repeat(
            g,
            running_mean,
            g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64)),
        )
        running_var_ = repeat(
            g,
            running_var,
            g.op("Constant", value_t=torch.tensor([n], dtype=torch.int64)),
        )
        
        # 使用 g.op() 函数执行张量重塑
        input_reshaped = g.op(
            "Reshape",
            input,
            g.op("Constant", value_t=torch.LongTensor(input_size_reshape)),
        )
        
        # 调用 batch_norm 函数进行归一化
        out = batch_norm(
            g,
            input_reshaped,
            weight_,
            bias_,
            running_mean_,
            running_var_,
            use_input_stats,
            momentum,
            eps,
            cudnn_enabled,
        )
        
        # 使用 view 函数重塑输出张量的形状
        return view(g, out, g.op("Constant", value_t=torch.tensor(input_size)))
# 注解解析ONNX符号函数"aten::unfold"
@_onnx_symbolic("aten::unfold")
# 使用装饰器解析参数"v", "i", "i", "i"
@symbolic_helper.parse_args("v", "i", "i", "i")
# 使用beartype装饰器进行类型检查
@_beartype.beartype
# 定义unfold函数，接受图上下文g，输入input，维度dimension，大小size，步长step
def unfold(g: jit_utils.GraphContext, input, dimension, size, step):
    # 获取输入张量的大小信息
    sizes = symbolic_helper._get_tensor_sizes(input)
    
    # 尝试获取指定维度的大小，如果失败则将sizedim设为None
    try:
        sizedim = sizes[dimension]
    # 捕获所有异常，这里应该避免捕获通用的Exception异常
    except Exception:
        # FIXME(justinchuby): 避免捕获通用的异常类型，应捕获更具体的异常
        sizedim = None
    
    # 如果成功获取到了sizedim大小
    if sizedim is not None:
        # 计算低索引和高索引的范围
        low_indices = range(0, sizedim, step)
        hi_indices = range(size, sizedim + 1, step)
        
        # 使用列表推导式生成切片操作的堆栈
        stack = [
            symbolic_helper._slice_helper(
                g, input, axes=[dimension], starts=[low], ends=[hi]
            )
            for low, hi in zip(low_indices, hi_indices)
        ]
        
        # 获取张量的维度数量
        ndim = len(sizes)
        # 创建一个维度的排列列表
        perm = list(range(0, ndim))
        perm.append(perm.pop(dimension))
        
        # 使用列表推导式生成unsqueeze操作的列表
        unsqueeze = [
            symbolic_helper._unsqueeze_helper(
                g, g.op("Transpose", t, perm_i=perm), [dimension]
            )
            for t in stack
        ]
        
        # 返回合并后的张量，沿指定轴维度
        return g.op("Concat", *unsqueeze, axis_i=dimension)
    else:
        # 如果无法获取到sizedim大小，则返回未实现的操作
        return symbolic_helper._unimplemented(
            "Unfold", "input size not accessible", input
        )


# 注解解析ONNX符号函数"aten::elu"
@_onnx_symbolic("aten::elu")
# 使用quantized_args装饰器进行量化参数处理
@symbolic_helper.quantized_args(True)
# 使用解析参数装饰器解析参数"v", "t", "t", "t"
@symbolic_helper.parse_args("v", "t", "t", "t")
# 使用beartype装饰器进行类型检查
@_beartype.beartype
# 定义elu函数，接受图上下文g，输入input，alpha，scale，input_scale
def elu(g: jit_utils.GraphContext, input, alpha, scale, input_scale):
    # 如果scale存在且不等于1.0，则返回未实现的操作
    if scale and scale != 1.0:
        return symbolic_helper._unimplemented(
            "scale", "does not support scale in Elu", scale
        )
    # 如果input_scale存在且不等于1.0，则返回未实现的操作
    if input_scale and input_scale != 1.0:
        return symbolic_helper._unimplemented(
            "input_scale", "does not support input_scale in Elu", input_scale
        )
    
    # 返回Elu操作，应用于输入input，alpha_f使用标量alpha
    # See Note [Export inplace]
    return g.op("Elu", input, alpha_f=symbolic_helper._scalar(alpha))


# 注解解析ONNX符号函数"aten::selu"
@_onnx_symbolic("aten::selu")
# 使用quantized_args装饰器进行量化参数处理
@symbolic_helper.quantized_args(True)
# 使用beartype装饰器进行类型检查
@_beartype.beartype
# 定义selu函数，接受图上下文g，输入input
def selu(g: jit_utils.GraphContext, input):
    # 返回Selu操作，应用于输入input
    return g.op("Selu", input)


# 注解解析ONNX符号函数"aten::index_select"
@_onnx_symbolic("aten::index_select")
# 使用解析参数装饰器解析参数"v", "i", "v"
@symbolic_helper.parse_args("v", "i", "v")
# 使用beartype装饰器进行类型检查
@_beartype.beartype
# 定义index_select函数，接受图上下文g，输入self，维度dim，索引index
def index_select(g: jit_utils.GraphContext, self, dim, index):
    # 如果索引为标量，则将index转换为1D张量以保持与输入张量相同的秩
    # 在ONNX中，为了保持与输入张量相同的秩，我们将索引转换为1D张量
    return symbolic_helper._select_helper(g, self, dim, index)


# 注解解析ONNX符号函数"aten::index_put"
# 使用beartype装饰器进行类型检查
@_beartype.beartype
# 定义index_put函数，接受图上下文g，输入self，indices_list_value，值values，是否累加accumulate
def index_put(g: jit_utils.GraphContext, self, indices_list_value, values, accumulate):
    # 如果indices_list_value是打包列表，则解包indices_list_value
    if symbolic_helper._is_packed_list(indices_list_value):
        indices_list = symbolic_helper._unpack_list(indices_list_value)
    else:
        indices_list = [indices_list_value]
    
    # 解析是否累加参数accumulate
    accumulate = symbolic_helper._parse_arg(accumulate, "b")
    # 如果 indices_list 的长度为 0，则执行以下操作
    if len(indices_list) == 0:
        # 如果 accumulate 参数为真，则调用 add 函数并返回其结果
        if accumulate:
            return add(g, self, values)
        # 如果 accumulate 参数为假，则直接返回 values
        return values
    
    # 如果 indices_list 的长度不为 0，则调用 symbolic_helper 模块的 _onnx_opset_unsupported 函数
    # 参数是字符串 "index_put"，以及版本号 9 和 11，以及 self 对象作为参数
    symbolic_helper._onnx_opset_unsupported("index_put", 9, 11, self)
# 注册 ONNX 符号化处理函数，处理 "aten::index_fill" 操作
# 使用 beartype 对函数参数进行类型检查
@_onnx_symbolic("aten::index_fill")
@_beartype.beartype
def index_fill(g: jit_utils.GraphContext, self, dim, index, value):
    # 解析 dim 参数为整数
    dim_value = symbolic_helper._parse_arg(dim, "i")
    # 根据 index 帮助函数展开索引的形状并获取展开后的索引
    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    # 将 value 转换为标量（如果可能）
    value = symbolic_helper._maybe_get_scalar(value)
    # 根据 self 的类型，将 value 转换为相同类型
    value = symbolic_helper._if_scalar_type_as(value, self)
    # 使用 expand 函数将 value 扩展到 expanded_index_shape 的形状
    expanded_value = expand(g, value, expanded_index_shape, None)

    # 调用 scatter 函数，返回结果
    return scatter(g, self, dim, expanded_index, expanded_value)


# 注册 ONNX 符号化处理函数，处理 "aten::index_copy" 操作
# 使用 beartype 对函数参数进行类型检查
@_onnx_symbolic("aten::index_copy")
@_beartype.beartype
def index_copy(g: jit_utils.GraphContext, self, dim, index, source):
    # 解析 dim 参数为整数
    dim_value = symbolic_helper._parse_arg(dim, "i")
    # 根据 index 帮助函数展开索引的形状并获取展开后的索引
    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    # 调用 scatter 函数，返回结果
    return scatter(g, self, dim, expanded_index, source)


# 注册 ONNX 符号化处理函数，处理 "aten::bucketize" 操作
# 解析函数参数 "v", "v", "b", "b"，使用 beartype 对参数进行类型检查
@_onnx_symbolic("aten::bucketize")
@symbolic_helper.parse_args("v", "v", "b", "b")
@_beartype.beartype
def bucketize(
    g: jit_utils.GraphContext, self, boundaries, out_int32=False, right=False
):
    # 输出类型默认为 INT64
    out_type = _C_onnx.TensorProtoDataType.INT64
    # 如果指定 out_int32 为 True，则修改输出类型为 INT32
    if out_int32:
        out_type = _C_onnx.TensorProtoDataType.INT32
    # 创建新形状，将 boundaries 复制给 self 的每个元素
    new_shape = g.op("Concat", g.op("Shape", boundaries), g.op("Shape", self), axis_i=0)
    # 根据 ONNX 的 numpy 风格广播，使用 unsqueeze 操作进行展开
    tensor_rank = symbolic_helper._get_tensor_rank(self)
    assert tensor_rank is not None
    unsqueeze_axes = list(range(1, tensor_rank + 1))
    expanded_boundaries = expand(
        g,
        symbolic_helper._unsqueeze_helper(g, boundaries, unsqueeze_axes),
        new_shape,
        None,
    )
    # 比较 self 的每个元素和 boundaries，获取结果张量
    # 包含前导 1 和尾部 0
    if right:
        cond = ge(g, self, expanded_boundaries)
    else:
        cond = gt(g, self, expanded_boundaries)
    # 将条件张量转换为指定的输出类型
    cond_out = g.op("Cast", cond, to_i=out_type)
    # 求和以获取每个元素对应的 1 的数量，即桶索引
    return symbolic_helper._reducesum_helper(g, cond_out, axes_i=[0], keepdims_i=0)


# 注册 ONNX 符号化处理函数，处理 "aten::type_as" 操作
# 使用 beartype 对函数参数进行类型检查
@_onnx_symbolic("aten::type_as")
@_beartype.beartype
def type_as(g: jit_utils.GraphContext, self, other):
    # 尝试获取 self 和 other 的标量类型
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    other_dtype = symbolic_helper._try_get_scalar_type(other)
    # 如果 self 和 other 的标量类型相同且不为 None，则直接返回 self
    if self_dtype == other_dtype and self_dtype is not None:
        return self
    # 如果 other 的标量类型不为 None，则进行类型转换操作
    if other_dtype is not None:
        return g.op(
            "Cast",
            self,
            to_i=other_dtype.onnx_type(),
        )
    # 抛出 SymbolicValueError 异常，指示 ONNX 导出不支持对未知数据类型的张量执行 type_as 操作
    raise errors.SymbolicValueError(
        "Unsupported: ONNX export of type_as for tensor "
        "of unknown dtype. Please check if the dtype of the "
        "parameter passed to the type_as function is correct.",
        other,
    )
    # 异常信息包含对于不支持操作的说明以及建议用户检查传递给 type_as 函数的参数数据类型是否正确
# 将函数注册为 ONNX 符号操作 "aten::cosine_similarity" 的符号操作函数
# 解析函数参数，参数类型为向量、向量、整数、浮点数
# 应用 Beartype 装饰器，用于参数类型检查
def cosine_similarity(g: jit_utils.GraphContext, x1, x2, dim, eps):
    # 计算 x1 和 x2 的逐元素乘积，并在指定维度上求和，得到交叉项
    cross = symbolic_helper._reducesum_helper(
        g, mul(g, x1, x2), axes_i=[dim], keepdims_i=0
    )
    # 计算 x1 的 L2 范数的平方，并在指定维度上求和
    x1_l2 = symbolic_helper._reducesum_helper(
        g, mul(g, x1, x1), axes_i=[dim], keepdims_i=0
    )
    # 计算 x2 的 L2 范数的平方，并在指定维度上求和
    x2_l2 = symbolic_helper._reducesum_helper(
        g, mul(g, x2, x2), axes_i=[dim], keepdims_i=0
    )
    # 计算除数，为 max(sqrt(x1_l2 * x2_l2), eps)，其中 eps 为常数
    div_tens = max(
        g, sqrt(g, mul(g, x1_l2, x2_l2)), g.op("Constant", value_t=torch.tensor([eps]))
    )
    # 返回余弦相似度
    return div(g, cross, div_tens)


# 将函数注册为 ONNX 符号操作 "aten::pairwise_distance" 的符号操作函数
# 应用 Beartype 装饰器，用于参数类型检查
def pairwise_distance(g: jit_utils.GraphContext, input1, input2, p, eps, keepdim):
    # 如果 eps 不是一个值，则将其设置为常数张量
    if not symbolic_helper._is_value(eps):
        eps = g.op("Constant", value_t=torch.tensor([eps]))
    # 计算 p 的倒数
    inv_p = div(
        g,
        g.op("Constant", value_t=torch.tensor([1], dtype=torch.float)),
        add(g, p, eps),
    )
    # 计算 input1 和 input2 之间的 pairwise distance，根据参数 p 和 keepdim
    summation = symbolic_helper._reducesum_helper(
        g,
        pow(g, sub(g, input1, input2), p),
        axes_i=[-1],
        keepdims_i=symbolic_helper._parse_arg(keepdim, "i"),
    )
    # 返回 pairwise distance 的结果
    return pow(g, summation, inv_p)


# 将函数注册为 ONNX 符号操作 "aten::clone" 的符号操作函数
# ignore clone operators that are inserted by PyTorch autograd
# 应用 Beartype 装饰器，用于参数类型检查
def clone(g: jit_utils.GraphContext, input, unused_memory_format):
    # 返回输入对象 input，用于 ONNX 的 clone 操作
    return input


# 将函数注册为 ONNX 符号操作 "aten::abs" 的符号操作函数
# 应用 Beartype 装饰器，用于参数类型检查
def abs(g: jit_utils.GraphContext, self):
    # 返回输入张量 self 的绝对值
    return g.op("Abs", self)


# 将函数注册为 ONNX 符号操作 "aten::log" 的符号操作函数
# 应用 Beartype 装饰器，用于参数类型检查
def log(g: jit_utils.GraphContext, self):
    # 返回输入张量 self 的自然对数
    return g.op("Log", self)


# 将函数注册为 ONNX 符号操作 "aten::log1p" 的符号操作函数
# 应用 Beartype 装饰器，用于参数类型检查
def log1p(g: jit_utils.GraphContext, self):
    # 计算 log(1 + self)，用于 ONNX 的 log1p 操作
    return log(g, add(g, symbolic_helper._if_scalar_type_as(torch.ones(1), self), self))


# 将函数注册为 ONNX 符号操作 "aten::log10" 的符号操作函数
# 应用 Beartype 装饰器，用于参数类型检查
def log10(g: jit_utils.GraphContext, self):
    _ln10 = 2.30258509299404568401
    # 计算输入张量 self 的以 10 为底的对数
    return g.op("Div", log(g, self), g.op("Constant", value_t=torch.tensor([_ln10])))


# 将函数注册为 ONNX 符号操作 "aten::pow" 的符号操作函数
# 应用 Beartype 装饰器，用于参数类型检查
def pow(g: jit_utils.GraphContext, self, exponent):
    # 获取输入张量 self 的数据类型
    f_dtype = _type_utils.JitScalarType.from_value(self)
    # 如果 self 不是浮点数，转换为浮点数类型
    if not symbolic_helper._is_fp(self):
        f_dtype = _type_utils.JitScalarType.FLOAT
        self = g.op("Cast", self, to_i=f_dtype.onnx_type())
    # 如果 exponent 不是浮点数，转换为与 self 相同的数据类型
    if not symbolic_helper._is_fp(exponent):
        exponent = g.op(
            "Cast",
            exponent,
            to_i=f_dtype.onnx_type(),
        )
    # 计算 self 的 exponent 次方
    pow = g.op("Pow", self, exponent)
    # 返回结果
    return pow


# 将函数注册为 ONNX 符号操作 "aten::clamp" 的符号操作函数
# 应用 Beartype 装饰器，用于参数类型检查
def clamp(g: jit_utils.GraphContext, self, min, max):
    # min 或 max 可能为 None，需要分别处理成 Clip 操作，因为 ONNX 不支持 None 语法
    if symbolic_helper._is_none(min):
        return clamp_max(g, self, max)
    # 如果 max 是 None，则调用 symbolic_helper._is_none 函数判断是否为 None
    elif symbolic_helper._is_none(max):
        # 如果 max 是 None，则返回 self 和 min 之间的最小值，使用 clamp_min 函数
        return clamp_min(g, self, min)
    # 如果 max 不是 None，则执行下面的代码块
    else:
        # 判断 min 和 max 是否都是常量，并且都调用 symbolic_helper._is_constant 函数来判断
        if symbolic_helper._is_constant(min) and symbolic_helper._is_constant(max):
            # 如果 min 和 max 都是常量，则调用 symbolic_helper._op_with_optional_float_cast 函数进行操作
            return symbolic_helper._op_with_optional_float_cast(
                g,
                "Clip",
                self,
                # 解析 min 和 max 参数为浮点数类型
                min_f=symbolic_helper._parse_arg(min, "f"),
                max_f=symbolic_helper._parse_arg(max, "f"),
                opset_before=12,
            )
        else:
            # 如果 min 或 max 不是常量，则先对 self 和 min 进行最小值截断，再对结果和 max 进行最大值截断
            return clamp_max(g, clamp_min(g, self, min), max)
# 注释：
# 定义一个带有装饰器的函数，用于在 ONNX 中符号化 "aten::clamp_min" 操作
@_onnx_symbolic("aten::clamp_min")
# 使用 parse_args 装饰器解析函数参数，期望两个向量参数
@symbolic_helper.parse_args("v", "v")
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# clamp_min 函数定义，接收图形上下文 g，self 和最小值 min 作为参数
def clamp_min(g: jit_utils.GraphContext, self, min):
    # 如果最小值 min 是常量
    if symbolic_helper._is_constant(min):
        # 调用 op_with_optional_float_cast 函数，执行 Clip 操作，可能包含浮点数转换
        return symbolic_helper._op_with_optional_float_cast(
            g, "Clip", self, min_f=symbolic_helper._parse_arg(min, "f"), opset_before=12
        )
    else:
        # 获取 self 的数据类型
        dtype = _type_utils.JitScalarType.from_value(self)
        # 将 min 转换为 dtype 指定的整数类型
        min = g.op("Cast", min, to_i=dtype.onnx_type())
        # 调用 op_with_optional_float_cast 函数，执行 Max 操作，可能包含浮点数转换
        return symbolic_helper._op_with_optional_float_cast(
            g, "Max", self, min, opset_before=12
        )


# 注释：
# 定义一个带有装饰器的函数，用于在 ONNX 中符号化 "aten::clamp_max" 操作
@_onnx_symbolic("aten::clamp_max")
# 使用 parse_args 装饰器解析函数参数，期望两个向量参数
@symbolic_helper.parse_args("v", "v")
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# clamp_max 函数定义，接收图形上下文 g，self 和最大值 max 作为参数
def clamp_max(g: jit_utils.GraphContext, self, max):
    # 如果最大值 max 是常量
    if symbolic_helper._is_constant(max):
        # 调用 op_with_optional_float_cast 函数，执行 Clip 操作，可能包含浮点数转换
        return symbolic_helper._op_with_optional_float_cast(
            g, "Clip", self, max_f=symbolic_helper._parse_arg(max, "f"), opset_before=12
        )
    else:
        # 获取 self 的数据类型
        dtype = _type_utils.JitScalarType.from_value(self)
        # 将 max 转换为 dtype 指定的整数类型
        max = g.op("Cast", max, to_i=dtype.onnx_type())
        # 调用 op_with_optional_float_cast 函数，执行 Min 操作，可能包含浮点数转换
        return symbolic_helper._op_with_optional_float_cast(
            g, "Min", self, max, opset_before=12
        )


# 注释：
# 定义一个带有装饰器的函数，用于在 ONNX 中符号化 "aten::max" 操作
@_onnx_symbolic("aten::max")
# max 函数的 torch.max 有两个接口：torch.max(x, dim, keepdim) 和 torch.max(x, y)
# TODO(justinchuby): 支持输出中的多个量化参数
@_beartype.beartype
# max 函数定义，接收图形上下文 g，self，dim_or_y 和 keepdim 作为参数
def max(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # 调用 _max_helper 函数，执行最大值辅助操作
    return symbolic_helper._max_helper(g, self, dim_or_y, keepdim)


# 注释：
# 定义一个带有装饰器的函数，用于在 ONNX 中符号化 "aten::maximum" 操作
@_onnx_symbolic("aten::maximum")
# 使用 quantized_args 装饰器，配置为 True，True
@symbolic_helper.quantized_args(True, True)
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# maximum 函数定义，接收图形上下文 g，input 和 other 作为参数
def maximum(g: jit_utils.GraphContext, input, other):
    # 调用 max 函数，执行最大值操作，将 other 作为 dim_or_y 参数传递
    return max(g, input, dim_or_y=other)


# 注释：
# 定义一个带有装饰器的函数，用于在 ONNX 中符号化 "aten::min" 操作
@_onnx_symbolic("aten::min")
# TODO(justinchuby): 支持输出中的多个量化参数
@_beartype.beartype
# min 函数定义，接收图形上下文 g，self，dim_or_y 和 keepdim 作为参数
def min(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # 调用 _min_helper 函数，执行最小值辅助操作
    return symbolic_helper._min_helper(g, self, dim_or_y, keepdim)


# 注释：
# 定义一个带有装饰器的函数，用于在 ONNX 中符号化 "aten::minimum" 操作
@_onnx_symbolic("aten::minimum")
# 使用 quantized_args 装饰器，配置为 True，True
@symbolic_helper.quantized_args(True, True)
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# minimum 函数定义，接收图形上下文 g，input 和 other 作为参数
def minimum(g: jit_utils.GraphContext, input, other):
    # 调用 min 函数，执行最小值操作，将 other 作为 dim_or_y 参数传递
    return min(g, input, dim_or_y=other)


# 注释：
# 定义一个带有装饰器的函数，用于在 ONNX 中符号化 "aten::amax" 操作
@_onnx_symbolic("aten::amax")
# 使用 quantized_args 装饰器，配置为 True
@symbolic_helper.quantized_args(True)
# 使用 parse_args 装饰器解析函数参数，期望向量和整数作为参数
@symbolic_helper.parse_args("v", "is", "i")
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# amax 函数定义，接收图形上下文 g，self，dim 和 keepdim 作为参数
def amax(g: jit_utils.GraphContext, self, dim, keepdim):
    # 调用 op 函数，执行 ReduceMax 操作，指定轴和是否保持维度
    return g.op("ReduceMax", self, axes_i=dim, keepdims_i=keepdim)


# 注释：
# 定义一个带有装饰器的函数，用于在 ONNX 中符号化 "aten::amin" 操作
@_onnx_symbolic("aten::amin")
# 使用 quantized_args 装饰器，配置为 True
@symbolic_helper.quantized_args(True)
# 使用 parse_args 装饰器解析函数参数，期望向量和整数作为参数
@symbolic_helper.parse_args("v", "is", "i")
# 应用 beartype 装饰器，用于运行时类型检查
@_beartype.beartype
# amin 函数定义，接收图形上下文 g，self，dim 和 keepdim 作为参数
def amin(g: jit_utils.GraphContext, self, dim, keepdim):
    # 调用 op 函数，执行 ReduceMin 操作，指定轴和是否保持维度
    return g.op("ReduceMin", self, axes_i=dim, keepdims_i=keepdim)


# 注释：
# 定义一个带有装饰器的函数，用于在 ONNX 中符号化 "aten::aminmax" 操作
@_onnx_symbolic("aten::aminmax")
# 使用 quantized_args 装饰器，配置为 True
@symbolic_helper.quantized_args(True)
    # 如果 dim 不是 None，则调用 symbolic_helper._get_const() 获取其常量值，标识为整数，命名为 "dim"
    if not symbolic_helper._is_none(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")
        # 将 dim 添加到 reduce_kwargs 字典的 "axes_i" 键中
        reduce_kwargs["axes_i"] = [dim]

    # 返回两个操作节点，分别是对当前对象 self 进行 ReduceMin 和 ReduceMax 操作，使用 reduce_kwargs 作为额外参数
    return g.op("ReduceMin", self, **reduce_kwargs), g.op(
        "ReduceMax", self, **reduce_kwargs
    )
# 将函数注册为 ONNX 符号函数 "aten::exp"
@_onnx_symbolic("aten::exp")
# 使用 beartype 进行类型检查和注解
@_beartype.beartype
def exp(g: jit_utils.GraphContext, self):
    # 在图中添加指数运算的节点 "Exp"
    return g.op("Exp", self)


# 将函数注册为 ONNX 符号函数 "aten::dropout_" 和 "aten::dropout"
@_onnx_symbolic("aten::dropout_")
@_onnx_symbolic("aten::dropout")
# 解析参数 "v", "f", "i" 并使用 beartype 进行类型检查和注解
@symbolic_helper.parse_args("v", "f", "i")
@_beartype.beartype
def dropout(g: jit_utils.GraphContext, input, p, train):
    # 检查训练模式，如果 train 为 False，则 dropout 不起作用，直接返回 input
    symbolic_helper.check_training_mode(train, "dropout")
    if not train:
        return input
    # 在图中添加 dropout 操作的节点 "Dropout"，将 dropout 比率设置为 p
    r, _ = g.op("Dropout", input, ratio_f=p, outputs=2)
    return r


# 将函数注册为多个不支持的 dropout 类型的 ONNX 符号函数
@_onnx_symbolic(
    "aten::alpha_dropout_",
    decorate=[symbolic_helper._apply_params("aten::alpha_dropout_")],
)  # 查看注释 [Export inplace]
@_onnx_symbolic(
    "aten::feature_alpha_dropout_",
    decorate=[symbolic_helper._apply_params("aten::feature_alpha_dropout_")],
)
@_onnx_symbolic(
    "aten::feature_dropout_",
    decorate=[symbolic_helper._apply_params("aten::feature_dropout_")],
)
@_onnx_symbolic(
    "aten::feature_alpha_dropout",
    decorate=[symbolic_helper._apply_params("aten::feature_alpha_dropout")],
)
@_onnx_symbolic(
    "aten::alpha_dropout",
    decorate=[symbolic_helper._apply_params("aten::alpha_dropout")],
)
@_onnx_symbolic(
    "aten::feature_dropout",
    decorate=[symbolic_helper._apply_params("aten::feature_dropout")],
)
@_beartype.beartype
def _unsupported_dropout(name: str):
    # 解析参数 "v", "none", "b" 并使用 beartype 进行类型检查和注解
    @symbolic_helper.parse_args("v", "none", "b")
    @_beartype.beartype
    def feature_dropout(g, input, p, train):
        # 注意：在推理模式下，FeatureDropout 被导出为一个恒等操作
        if train:
            return symbolic_helper._unimplemented(name, "training mode", input)
        # 在训练模式下，不支持的 dropout 操作直接返回 input
        return input

    return feature_dropout


# 将函数注册为 ONNX 符号函数 "aten::norm"
@_onnx_symbolic("aten::norm")
# 解析参数 "v", "t", "is", "i", "v" 并使用 beartype 进行类型检查和注解
@symbolic_helper.parse_args("v", "t", "is", "i", "v")
@_beartype.beartype
def norm(g: jit_utils.GraphContext, self, p, dim, keepdim, dtype=None):
    if p == 1:
        # 使用 ReduceL1 辅助函数来创建对应的 ONNX 操作节点
        f = symbolic_helper._reduce_op_symbolic_helper("ReduceL1")
    elif p == 2:
        # 使用 ReduceL2 辅助函数来创建对应的 ONNX 操作节点
        f = symbolic_helper._reduce_op_symbolic_helper("ReduceL2")
    else:
        # 如果 p 不是 1 或 2，抛出符号值错误
        raise errors.SymbolicValueError(
            "ONNX export only p-norms with p of 1 or 2", self
        )
    # 使用选定的 Reduce 函数 f 对输入进行操作，指定维度和保持维度信息
    result = f(g, self, dim=dim, keepdim=keepdim)
    if dtype is not None:
        # 如果指定了 dtype，则将结果转换为指定的数据类型
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        result = g.op("Cast", result, to_i=_type_utils.JitScalarType(dtype).onnx_type())
    return result


# 将函数注册为 ONNX 符号函数 "aten::conv_tbc"
@_onnx_symbolic("aten::conv_tbc")
# 解析参数 "v", "v", "v", "i" 并使用 beartype 进行类型检查和注解
@symbolic_helper.parse_args("v", "v", "v", "i")
@_beartype.beartype
def conv_tbc(g: jit_utils.GraphContext, input, weight, bias, pad):
    # 输入 input 必须有 3 维，参见链接中的说明
    # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ConvolutionTBC.cpp#L8-L10
    # input = (time, batch, in_channels)
    # weight = (kernel_width, in_channels, out_channels)
    # bias = (out_channels,)
    # 在图中添加转置操作 "Transpose"，调整输入和权重的维度顺序
    input = g.op("Transpose", input, perm_i=[1, 2, 0])
    weight = g.op("Transpose", weight, perm_i=[2, 1, 0])
    # 使用 conv1d 函数对输入 input 进行一维卷积操作，使用给定的权重 weight 和偏置 bias，
    # [1] 指定步长为1，
    # [pad] 指定填充的大小，
    # [1] 指定 dilation 的大小为1。
    conv = conv1d(g, input, weight, bias, [1], [pad], [1], 1)
    # 使用 g 对象调用 op 方法，创建一个 "Transpose" 操作，
    # 将 conv 的维度按照 perm_i 指定的顺序进行转置，顺序为 [2, 0, 1]。
    return g.op("Transpose", conv, perm_i=[2, 0, 1])
# 定义 _unique 函数，用于在 ONNX 符号化环境中处理 aten::_unique 操作
@_onnx_symbolic("aten::_unique")
# 使用 symbolic_helper.parse_args 装饰器，解析函数参数，包括图形上下文 g、输入 input、排序标志 sorted、返回反向索引标志 return_inverse
@symbolic_helper.parse_args("v", "i", "i")
# 应用 beartype 装饰器，进行输入参数类型检查和强制转换
@_beartype.beartype
# 定义 _unique 函数，返回 _onnx_unsupported 函数处理后的结果
def _unique(g: jit_utils.GraphContext, input, sorted, return_inverse):
    return symbolic_helper._onnx_unsupported("_unique", input)


# 定义 _unique2 函数，用于在 ONNX 符号化环境中处理 aten::_unique2 操作
@_onnx_symbolic("aten::_unique2")
# 使用 symbolic_helper.parse_args 装饰器，解析函数参数，包括图形上下文 g、输入 input、排序标志 sorted、返回反向索引标志 return_inverse、返回计数标志 return_counts
@symbolic_helper.parse_args("v", "i", "i", "i")
# 应用 beartype 装饰器，进行输入参数类型检查和强制转换
@_beartype.beartype
# 定义 _unique2 函数，调用 _onnx_opset_unsupported 函数处理 _unique2 操作不支持的情况
def _unique2(g: jit_utils.GraphContext, input, sorted, return_inverse, return_counts):
    symbolic_helper._onnx_opset_unsupported("_unique2", 9, 11, input)


# 定义 _cast_Byte 函数，用于在 ONNX 符号化环境中处理 aten::_cast_Byte 操作
@_onnx_symbolic("aten::_cast_Byte")
# 使用 deprecated 装饰器，标记函数为废弃，提供相关信息：废弃版本 "2.0"，替代建议 "Avoid using this function and create a Cast node instead"，未来版本 "the future"
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
# 应用 beartype 装饰器，进行输入参数类型检查和强制转换
@_beartype.beartype
# 定义 _cast_Byte 函数，返回 Cast 操作结果，将输入 input 转换为 UINT8 类型
def _cast_Byte(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.UINT8)


# 定义 _cast_Char 函数，用于在 ONNX 符号化环境中处理 aten::_cast_Char 操作
@_onnx_symbolic("aten::_cast_Char")
# 使用 deprecated 装饰器，标记函数为废弃，提供相关信息：废弃版本 "2.0"，替代建议 "Avoid using this function and create a Cast node instead"，未来版本 "the future"
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
# 应用 beartype 装饰器，进行输入参数类型检查和强制转换
@_beartype.beartype
# 定义 _cast_Char 函数，返回 Cast 操作结果，将输入 input 转换为 INT8 类型
def _cast_Char(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT8)


# 定义 _cast_Short 函数，用于在 ONNX 符号化环境中处理 aten::_cast_Short 操作
@_onnx_symbolic("aten::_cast_Short")
# 使用 deprecated 装饰器，标记函数为废弃，提供相关信息：废弃版本 "2.0"，替代建议 "Avoid using this function and create a Cast node instead"，未来版本 "the future"
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
# 应用 beartype 装饰器，进行输入参数类型检查和强制转换
@_beartype.beartype
# 定义 _cast_Short 函数，返回 Cast 操作结果，将输入 input 转换为 INT16 类型
def _cast_Short(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT16)


# 定义 _cast_Int 函数，用于在 ONNX 符号化环境中处理 aten::_cast_Int 操作
@_onnx_symbolic("aten::_cast_Int")
# 使用 deprecated 装饰器，标记函数为废弃，提供相关信息：废弃版本 "2.0"，替代建议 "Avoid using this function and create a Cast node instead"，未来版本 "the future"
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
# 应用 beartype 装饰器，进行输入参数类型检查和强制转换
@_beartype.beartype
# 定义 _cast_Int 函数，返回 Cast 操作结果，将输入 input 转换为 INT32 类型
def _cast_Int(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT32)


# 定义 _cast_Long 函数，用于在 ONNX 符号化环境中处理 aten::_cast_Long 操作
@_onnx_symbolic("aten::_cast_Long")
# 使用 deprecated 装饰器，标记函数为废弃，提供相关信息：废弃版本 "2.0"，替代建议 "Avoid using this function and create a Cast node instead"，未来版本 "the future"
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
# 应用 beartype 装饰器，进行输入参数类型检查和强制转换
@_beartype.beartype
# 定义 _cast_Long 函数，返回 Cast 操作结果，将输入 input 转换为 INT64 类型
def _cast_Long(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT64)


# 定义 _cast_Half 函数，用于在 ONNX 符号化环境中处理 aten::_cast_Half 操作
@_onnx_symbolic("aten::_cast_Half")
# 使用 deprecated 装饰器，标记函数为废弃，提供相关信息：废弃版本 "2.0"，替代建议 "Avoid using this function and create a Cast node instead"，未来版本 "the future"
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
# 应用 beartype 装饰器，进行输入参数类型检查和强制转换
@_beartype.beartype
# 定义 _cast_Half 函数，返回 Cast 操作结果，将输入 input 转换为 FLOAT16 类型
def _cast_Half(g: jit_utils.GraphContext, input, non_blocking):
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.FLOAT16)


# 定义 _cast_Float 函数，用于在 ONNX 符号化环境中处理 aten::_cast_Float 操作
@_onnx_symbolic("aten::_cast_Float")
# 使用 deprecated 装饰器，标记函数为废弃，提供相关信息：废弃版本 "2.0"，替代建议 "Avoid using this function and create a Cast node instead"，未来版本 "the future"
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
# 应用 beartype 装饰器，进行输入参数类型检查和强制转换
    # 使用ONNX运算符创建一个“Cast”操作，将输入张量转换为双精度数据类型。
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.DOUBLE)
# 将函数标记为ONNX符号化的函数，对应的运算是aten::_cast_Bool
# 使用beartype进行类型检查和注解
@_onnx_symbolic("aten::_cast_Bool")
@_deprecation.deprecated(
    "2.0",
    "the future",
    "Avoid using this function and create a Cast node instead",
)
@_beartype.beartype
def _cast_Bool(g: jit_utils.GraphContext, input, non_blocking):
    # 在图上创建一个Cast操作，将输入input转换为_BOOL类型
    return g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.BOOL)


# 将函数标记为ONNX符号化的函数，对应的运算是aten::empty
# 使用symbolic_helper.parse_args解析参数
# 使用beartype进行类型检查和注解
@_onnx_symbolic("aten::empty")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
@_beartype.beartype
def empty(
    g: jit_utils.GraphContext,
    sizes,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    # 调用zeros函数，返回相同参数的结果
    return zeros(g, sizes, dtype, layout, device, pin_memory)


# 将函数标记为ONNX符号化的函数，对应的运算是aten::empty_like
# 使用symbolic_helper.parse_args解析参数
# 使用beartype进行类型检查和注解
@_onnx_symbolic("aten::empty_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
@_beartype.beartype
def empty_like(
    g: jit_utils.GraphContext,
    input,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    # 调用zeros_like函数，返回相同参数的结果
    return zeros_like(g, input, dtype, layout, device, pin_memory)


# 将函数标记为ONNX符号化的函数，对应的运算是aten::new_empty
# 使用beartype进行类型检查和注解
def new_empty(
    g: jit_utils.GraphContext, self, sizes, dtype, layout, device, pin_memory=False
):
    # 尝试从self中获取标量类型
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    # 如果dtype为空且self_dtype不为空，则将dtype设置为self_dtype
    if symbolic_helper._is_none(dtype) and self_dtype is not None:
        dtype = self_dtype
    # 调用empty函数，返回相同参数的结果
    return empty(g, sizes, dtype, layout, device, pin_memory)


# 将函数标记为ONNX符号化的函数，对应的运算是aten::scalar_tensor
# 使用beartype进行类型检查和注解
def scalar_tensor(g: jit_utils.GraphContext, scalar, dtype, *options):
    # 获取常量dtype的值，如果为None则设置为默认值JitScalarType.FLOAT
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    if dtype is None:
        dtype = _type_utils.JitScalarType.FLOAT
    # 在图上创建一个Cast操作，将scalar转换为指定的dtype类型
    scalar = g.op("Cast", scalar, to_i=_type_utils.JitScalarType(dtype).onnx_type())
    return scalar


# 将函数标记为ONNX符号化的函数，对应的运算是aten::tensor
# 使用beartype进行类型检查和注解
def tensor(
    g: jit_utils.GraphContext, data, dtype=None, device=None, requires_grad=False
):
    # 获取常量dtype的值，如果为None则根据data的类型设置dtype
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # 如果data是打包的列表，则根据第一个元素的值设置dtype
    if symbolic_helper._is_packed_list(data):
        if dtype is None:
            dtype = _type_utils.JitScalarType.from_value(
                symbolic_helper._unpack_list(data)[0]
            )
        input_list = list()
        # 遍历打包的列表，将每个元素转换为指定的dtype类型并添加到input_list中
        for t in symbolic_helper._unpack_list(data):
            shape_reference = g.op("Constant", value_t=torch.LongTensor([1]))
            t = symbolic_helper._reshape_helper(g, t, shape_reference)
            t = g.op("Cast", t, to_i=_type_utils.JitScalarType(dtype).onnx_type())
            input_list.append(t)
        # 在轴0上连接所有元素
        return g.op("Concat", *input_list, axis_i=0)
    else:
        # 如果data不是打包的列表，则根据data的类型设置dtype
        if dtype is None:
            dtype = _type_utils.JitScalarType.from_value(data)
        # 如果data是列表且为张量列表或标量列表，则使用ConcatFromSequence将data连接起来
        if symbolic_helper._is_list(data) and (
            symbolic_helper._is_tensor_list(data)
            or symbolic_helper._is_scalar_list(data)
        ):
            data = g.op("ConcatFromSequence", data, axis_i=0, new_axis_i=1)
    # 在图上创建一个Cast操作，将data转换为指定的dtype类型
    return g.op("Cast", data, to_i=_type_utils.JitScalarType(dtype).onnx_type())


# 将函数标记为ONNX符号化的函数，对应的运算是aten::as_tensor
# 使用beartype进行类型检查和注解
@_onnx_symbolic("aten::as_tensor")
@_beartype.beartype
# 定义一个函数 as_tensor，将数据转换为张量表示
def as_tensor(g: jit_utils.GraphContext, data, dtype=None, device=None):
    return tensor(g, data, dtype, device)


# 使用 ONNX 符号绑定函数 "@_onnx_symbolic" 将函数绑定到 "aten::zeros" 操作
# 解析参数使用装饰器 "@symbolic_helper.parse_args"
# 使用装饰器 "@_beartype.beartype" 进行类型检查
def zeros(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=False):
    # 如果未指定 dtype，则默认为浮点类型
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    
    # 获取 sizes 的常量值，如果是空列表，则创建一个空的 Constant 操作
    sizes_ = symbolic_helper._maybe_get_const(sizes, "is")
    if isinstance(sizes_, list) and len(sizes_) == 0:
        sizes = g.op("Constant", value_t=torch.tensor([]).to(torch.int64))
    
    # 返回一个 ConstantOfShape 操作，创建一个全零张量
    return g.op(
        "ConstantOfShape",
        sizes,
        value_t=torch.tensor([0], dtype=scalar_type.dtype()),
    )


# 使用 ONNX 符号绑定函数 "@_onnx_symbolic" 将函数绑定到 "aten::zeros_like" 操作
# 解析参数使用装饰器 "@symbolic_helper.parse_args"
# 使用装饰器 "@_beartype.beartype" 进行类型检查
def zeros_like(
    g: jit_utils.GraphContext,
    input,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    # 获取输入张量的形状信息
    shape = g.op("Shape", input)
    
    # 如果未指定 dtype，则从输入张量中推断类型
    if symbolic_helper._is_none(dtype):
        scalar_type = _type_utils.JitScalarType.from_value(
            input, _type_utils.JitScalarType.FLOAT
        )
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    
    # 返回一个 ConstantOfShape 操作，创建一个形状相同的全零张量
    return g.op(
        "ConstantOfShape",
        shape,
        value_t=torch.tensor([0], dtype=scalar_type.dtype()),
    )


# 使用 ONNX 符号绑定函数 "@_onnx_symbolic" 将函数绑定到 "aten::new_zeros" 操作
# 使用装饰器 "@_beartype.beartype" 进行类型检查
def new_zeros(
    g: jit_utils.GraphContext, self, sizes, dtype, layout, device, pin_memory=False
):
    # 尝试获取 self 张量的数据类型
    self_dtype = symbolic_helper._try_get_scalar_type(self)

    # 如果未指定 dtype 并且 self 张量类型不为空，则使用 self 张量的数据类型
    if symbolic_helper._is_none(dtype) and self_dtype is not None:
        dtype = self_dtype
    
    # 调用 zeros 函数创建一个全零张量
    return zeros(g, sizes, dtype, layout, device, pin_memory)


# 使用 ONNX 符号绑定函数 "@_onnx_symbolic" 将函数绑定到 "aten::zero" 操作
# 使用装饰器 "@_beartype.beartype" 进行类型检查
def zero(g: jit_utils.GraphContext, self):
    # 尝试获取 self 张量的数据类型
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    
    # 调用 zeros_like 函数创建一个形状相同的全零张量
    return zeros_like(g, self, self_dtype)


# 使用 ONNX 符号绑定函数 "@_onnx_symbolic" 将函数绑定到 "aten::ones" 操作
# 解析参数使用装饰器 "@symbolic_helper.parse_args"
# 使用装饰器 "@_beartype.beartype" 进行类型检查
def ones(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=False):
    # 如果未指定 dtype，则默认为浮点类型
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    
    # 获取 sizes 的常量值，如果是空列表，则创建一个空的 Constant 操作
    sizes_ = symbolic_helper._maybe_get_const(sizes, "is")
    if isinstance(sizes_, list) and len(sizes_) == 0:
        sizes = g.op("Constant", value_t=torch.tensor([]).to(torch.int64))
    
    # 返回一个 ConstantOfShape 操作，创建一个全一张量
    return g.op(
        "ConstantOfShape",
        sizes,
        value_t=torch.tensor([1], dtype=scalar_type.dtype()),
    )


# 使用 ONNX 符号绑定函数 "@_onnx_symbolic" 将函数绑定到 "aten::ones_like" 操作
# 解析参数使用装饰器 "@symbolic_helper.parse_args"
# 使用装饰器 "@_beartype.beartype" 进行类型检查
def ones_like(
    g: jit_utils.GraphContext,
    input,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    # 使用ONNX图的操作节点"Shape"，获取输入张量的形状信息
    shape = g.op("Shape", input)

    # 检查数据类型是否为符号类型，如果是则根据输入值创建浮点数标量类型
    if symbolic_helper._is_none(dtype):
        scalar_type = _type_utils.JitScalarType.from_value(
            input, _type_utils.JitScalarType.FLOAT
        )
    else:
        # 否则，根据给定的dtype创建标量类型
        scalar_type = _type_utils.JitScalarType(dtype)

    # 使用ONNX图的操作节点"ConstantOfShape"，创建一个常数张量，形状由上一步获取的shape确定，值为[1]
    return g.op(
        "ConstantOfShape",
        shape,
        value_t=torch.tensor([1], dtype=scalar_type.dtype()),
    )
# 注解一个ONNX符号化函数，将其与"aten::new_ones"关联
# 使用Beartype库装饰器进行类型检查和类型提示
@_onnx_symbolic("aten::new_ones")
@_beartype.beartype
def new_ones(
    g: jit_utils.GraphContext, self, sizes, dtype, layout, device, pin_memory=False
):
    # 获取self张量的数据类型
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    # 如果dtype是None并且self_dtype不为None，则使用self的数据类型作为dtype
    if symbolic_helper._is_none(dtype) and self_dtype is not None:
        dtype = self_dtype
    # 调用ones函数生成全1张量，并返回结果
    return ones(g, sizes, dtype, layout, device, pin_memory)


# 注解一个ONNX符号化函数，将其与"aten::full"关联
# 使用Beartype库装饰器进行类型检查和类型提示
@_onnx_symbolic("aten::full")
@_beartype.beartype
def full(
    g: jit_utils.GraphContext, sizes, value, dtype, layout, device, pin_memory=False
):
    # 尝试获取value的常量值
    const_value = symbolic_helper._maybe_get_const(value, "t")
    # 如果const_value是值类型
    if symbolic_helper._is_value(const_value):
        # 如果dtype为None，则设为JitScalarType.FLOAT
        dtype = _type_utils.JitScalarType.FLOAT if dtype is None else dtype
        # 创建全0张量
        tmp = zeros(g, sizes, dtype, layout, device)
        # 将value添加到tmp上，并返回结果
        return add(g, tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        # 获取dtype的常量值
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        # 如果dtype为None，则设为JitScalarType.FLOAT
        if dtype is None:
            scalar_type = _type_utils.JitScalarType.FLOAT
        else:
            scalar_type = _type_utils.JitScalarType(dtype)
        # 尝试获取sizes的常量值
        sizes_ = symbolic_helper._maybe_get_const(sizes, "is")
        # 如果sizes是列表且长度为0，则创建一个空的shape常量张量
        if isinstance(sizes_, list) and len(sizes_) == 0:
            sizes = g.op("Constant", value_t=torch.tensor([]).to(torch.int64))
        # 创建一个常量形状的张量，并返回结果
        return g.op(
            "ConstantOfShape",
            sizes,
            value_t=const_value.view(1).to(scalar_type.dtype()),
        )


# 注解一个ONNX符号化函数，将其与"aten::full_like"关联
# 使用Beartype库装饰器进行类型检查和类型提示
def full_like(
    g: jit_utils.GraphContext,
    input,
    fill_value,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    # 尝试获取fill_value的常量值
    fill_value = symbolic_helper._maybe_get_const(fill_value, "f")
    # 获取dtype的常量值
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # 如果dtype为None，则根据input的值类型选择JitScalarType.FLOAT作为scalar_type
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.from_value(
            input, _type_utils.JitScalarType.FLOAT
        )
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    # 如果fill_value是值类型
    if symbolic_helper._is_value(fill_value):
        # 创建一个与input相同形状的全0张量
        tmp = zeros_like(g, input, dtype, layout, device)
        # 将fill_value转换为scalar_type的ONNX类型，并添加到tmp上，并返回结果
        fill_value = g.op("Cast", fill_value, to_i=scalar_type.onnx_type())
        return add(g, tmp, fill_value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        # 获取input的形状
        shape = g.op("Shape", input)
        # 创建一个常量形状的张量，并返回结果
        return g.op(
            "ConstantOfShape",
            shape,
            value_t=torch.tensor([fill_value], dtype=scalar_type.dtype()),
        )


# 注解一个ONNX符号化函数，将其与"aten::new_full"关联
# 使用Beartype库装饰器进行类型检查和类型提示
@_onnx_symbolic("aten::new_full")
@_beartype.beartype
def new_full(
    g: jit_utils.GraphContext,
    self,
    size,
    fill_value,
    dtype,
    layout,
    device,
    pin_memory=False,
):
    # 获取self张量的数据类型
    self_dtype = symbolic_helper._try_get_scalar_type(self)
    # 如果dtype是None并且self_dtype不为None，则使用self的数据类型作为dtype
    if symbolic_helper._is_none(dtype) and self_dtype is not None:
        dtype = self_dtype
    # 调用full函数生成填充的张量，并返回结果
    return full(g, size, fill_value, dtype, layout, device, pin_memory)


# 注解一个ONNX符号化函数，将其与"aten::eye"关联
# 使用Beartype库装饰器进行类型检查和类型提示
@_onnx_symbolic("aten::eye")
@_beartype.beartype
def eye(g: jit_utils.GraphContext, *args):
    # 如果参数个数为5，执行以下逻辑
    if len(args) == 5:
        # 解构参数并赋值给对应变量
        n, dtype, layout, device, pin_memory = args
        # 使用符号辅助函数处理维度信息，增加一个维度
        dim_size = symbolic_helper._unsqueeze_helper(g, n, [0])
        # 构造张量形状，连接两个dim_size，按照 axis_i=0 连接
        shape = g.op("Concat", dim_size, dim_size, axis_i=0)
        # 创建一个全零张量
        tensor = zeros(g, shape, dtype, layout, device)
        # 返回一个操作节点，生成类似单位矩阵的张量
        return g.op("EyeLike", tensor)
    
    # 如果参数个数为6，执行以下逻辑
    if len(args) == 6:
        # 解构参数并赋值给对应变量
        n, m, dtype, layout, device, pin_memory = args
        # 使用符号辅助函数处理维度信息，增加一个维度
        shape = g.op(
            "Concat",
            symbolic_helper._unsqueeze_helper(g, n, [0]),
            symbolic_helper._unsqueeze_helper(g, m, [0]),
            axis_i=0,
        )
        # 创建一个全零张量
        tensor = zeros(g, shape, dtype, layout, device)
        # 返回一个操作节点，生成类似单位矩阵的张量
        return g.op("EyeLike", tensor)

    # 如果参数个数不为5或6，返回未实现的错误信息
    return symbolic_helper._unimplemented("aten::eye", f"with {len(args)} arguments")
# 使用装饰器将函数注册为处理 "aten::slice" 的符号函数
# 使用 beartype 进行类型检查和注解
@_onnx_symbolic("aten::slice")
@_beartype.beartype
def slice(g: jit_utils.GraphContext, self, *args):
    # 检查参数个数是否为4，表示执行 aten::slice 的操作
    if len(args) == 4:
        # 解析参数为 dim, start, end, step
        dim, start, end, step = args
        
        # 将 step 解析为整数
        step = symbolic_helper._parse_arg(step, "i")
        
        # 如果 step 不等于1，抛出错误，因为当前不支持 step 不为1的情况
        if step != 1:
            raise errors.SymbolicValueError("step!=1 is currently not supported", self)
        
        # 检查 start 和 end 是否为常量或者 NoneType
        is_start_none = start.node().kind() == "prim::Constant" and isinstance(
            start.type(), _C.NoneType
        )
        is_end_none = end.node().kind() == "prim::Constant" and isinstance(
            end.type(), _C.NoneType
        )
        
        # 检查 start 和 end 是否为 ONNX 的常量
        is_start_onnx_const = start.node().kind() == "onnx::Constant"
        is_end_onnx_const = end.node().kind() == "onnx::Constant"
        
        # 如果 start 或 end 不是常量或 NoneType，或者 dim 不是 ONNX 的常量，
        # 并且全局变量 GLOBALS.operator_export_type 为 ONNX，抛出错误
        if (
            ((not is_start_none) and (not is_start_onnx_const))
            or ((not is_end_none) and (not is_end_onnx_const))
            or dim.node().kind() != "onnx::Constant"
        ):
            if GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX:
                raise errors.SymbolicValueError(
                    "Unsupported: ONNX export of Slice with dynamic inputs. DynamicSlice "
                    "is a deprecated experimental op. Please use statically allocated "
                    "variables or export to a higher opset version.",
                    self,
                )
            else:
                # 对 start、end 和 dim 使用 _unsqueeze_helper 在指定维度上进行扩展
                start_unsqueezed = symbolic_helper._unsqueeze_helper(g, start, [0])
                end_unsqueezed = symbolic_helper._unsqueeze_helper(g, end, [0])
                dim_unsqueezed = symbolic_helper._unsqueeze_helper(g, dim, [0])
                
                # 返回一个 DynamicSlice 操作，用于动态切片
                return g.op(
                    "DynamicSlice",
                    self,
                    start_unsqueezed,
                    end_unsqueezed,
                    dim_unsqueezed,
                )
        else:
            # 解析 start、end 和 dim 为整数
            start = 0 if is_start_none else symbolic_helper._parse_arg(start, "i")
            end = (
                _constants.INT64_MAX
                if is_end_none
                else symbolic_helper._parse_arg(end, "i")
            )
            dim = symbolic_helper._parse_arg(dim, "i")
            
            # 使用 _slice_helper 执行切片操作
            return symbolic_helper._slice_helper(
                g, self, axes=[dim], starts=[start], ends=[end]
            )
    elif len(args) == 3:
        # 如果参数个数为3，则表示进行切片操作
        start, end, step = args
        # 切片操作默认在第一维度上进行
        dim = 0
        # 检查起始位置是否为 None
        is_start_none = start.node().kind() == "prim::Constant" and isinstance(
            start.type(), _C.NoneType
        )
        # 检查结束位置是否为 None
        is_end_none = end.node().kind() == "prim::Constant" and isinstance(
            end.type(), _C.NoneType
        )
        # 如果起始位置是 None，则设定为0，否则解析起始位置参数为整数
        start = 0 if is_start_none else symbolic_helper._parse_arg(start, "i")
        # 如果结束位置是 None，则设定为 INT64_MAX，否则解析结束位置参数为整数
        end = (
            _constants.INT64_MAX
            if is_end_none
            else symbolic_helper._parse_arg(end, "i")
        )
        # 调用辅助函数进行切片操作
        return symbolic_helper._slice_helper(
            g, self, axes=[dim], starts=[start], ends=[end]
        )

    # 如果参数个数不为3，则返回未实现的错误信息
    return symbolic_helper._unimplemented("aten::slice", f"with {len(args)} arguments")
# 将函数标记为对应 ONNX 符号 "aten::hardtanh" 的符号化函数
@_onnx_symbolic("aten::hardtanh")
# 应用量化参数装饰器，设置函数为量化版本
@symbolic_helper.quantized_args(True)
# 解析参数注解为 "v", "f", "f"，指定函数的参数类型和数量
@symbolic_helper.parse_args("v", "f", "f")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 hardtanh，接收 JIT 图上下文 g，输入 self 和两个浮点数参数 min_val 和 max_val
def hardtanh(g: jit_utils.GraphContext, self: _C.Value, min_val: float, max_val: float):
    # 调用符号化助手函数 _op_with_optional_float_cast，传入操作图 g、操作名 "Clip"、自身值 self、最小浮点数 min_val 和最大浮点数 max_val，以及操作集版本号 opset_before
    return symbolic_helper._op_with_optional_float_cast(
        g, "Clip", self, min_f=min_val, max_f=max_val, opset_before=12
    )


# 将函数标记为对应 ONNX 符号 "aten::hardswish" 的符号化函数
@_onnx_symbolic("aten::hardswish")
# 应用量化参数装饰器，设置函数为量化版本
@symbolic_helper.quantized_args(True)
# 解析参数注解为 "v"，指定函数的参数类型和数量
@symbolic_helper.parse_args("v")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 hardswish，接收 JIT 图上下文 g 和输入 self
def hardswish(g: jit_utils.GraphContext, self):
    # 调用 hardsigmoid 函数，传入 g 和 self，并将结果存储在 hs 中
    hs = hardsigmoid(g, self)
    # 返回 g 上的乘法操作 "Mul"，传入 self 和 hs 作为操作数
    return g.op("Mul", self, hs)


# 将函数标记为对应 ONNX 符号 "aten::hardsigmoid" 的符号化函数
@_onnx_symbolic("aten::hardsigmoid")
# 使用固定的缩放因子和零点值对函数进行量化参数设置
@symbolic_helper.quantized_args(True, scale=1.0 / 256.0, zero_point=0)
# 解析参数注解为 "v"，指定函数的参数类型和数量
@symbolic_helper.parse_args("v")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 hardsigmoid，接收 JIT 图上下文 g 和输入 self
def hardsigmoid(g: jit_utils.GraphContext, self):
    # 将 alpha_f 设置为 1 / 6，使操作等效于 PyTorch 的 Hardsigmoid 定义
    # 参考：https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
    return g.op("HardSigmoid", self, alpha_f=1 / 6)


# 将函数标记为对应 ONNX 符号 "aten::tanhshrink" 的符号化函数
@_onnx_symbolic("aten::tanhshrink")
# 解析参数注解为 "v"，指定函数的参数类型和数量
@symbolic_helper.parse_args("v")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 tanhshrink，接收 JIT 图上下文 g 和输入 self
def tanhshrink(g: jit_utils.GraphContext, self):
    # 返回 g 上的减法操作 "Sub"，传入 self 和 tanh(g, self) 作为操作数
    return g.op("Sub", self, tanh(g, self))


# 将函数标记为对应 ONNX 符号 "aten::hardshrink" 的符号化函数
@_onnx_symbolic("aten::hardshrink")
# 解析参数注解为 "v", "f"，指定函数的参数类型和数量
@symbolic_helper.parse_args("v", "f")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 hardshrink，接收 JIT 图上下文 g、输入 self 和参数 lambd
def hardshrink(g: jit_utils.GraphContext, self, lambd):
    # 从 self 中获取标量类型
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.FLOAT
    )
    # 创建一个常量操作 "Constant"，值为 torch.tensor(lambd, dtype=scalar_type.dtype())
    lambd_op = g.op(
        "Constant",
        value_t=torch.tensor(lambd, dtype=scalar_type.dtype()),
    )
    # 创建逻辑或条件，比较 self 是否大于 lambd_op 或小于负 lambd_op
    cond = logical_or(g, gt(g, self, lambd_op), lt(g, self, neg(g, lambd_op)))
    # 返回条件选择操作 "Where"，根据 cond 选择 self 或者常量 0
    return g.op(
        "Where",
        cond,
        self,
        g.op(
            "Constant",
            value_t=torch.tensor(0, dtype=scalar_type.dtype()),
        ),
    )


# 将函数标记为对应 ONNX 符号 "aten::softshrink" 的符号化函数
@_onnx_symbolic("aten::softshrink")
# 解析参数注解为 "v", "f"，指定函数的参数类型和数量
@symbolic_helper.parse_args("v", "f")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 softshrink，接收 JIT 图上下文 g、输入 self 和参数 lambd
def softshrink(g: jit_utils.GraphContext, self, lambd):
    # 从 self 中获取标量类型
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.FLOAT
    )
    # 创建一个常量操作 "Constant"，值为 torch.tensor(lambd, dtype=scalar_type.dtype())
    lambd_op = g.op(
        "Constant",
        value_t=torch.tensor(lambd, dtype=scalar_type.dtype()),
    )
    # 创建大于条件，比较 self 是否大于 lambd_op
    gt_cond = gt(g, self, lambd_op)
    # 根据 gt_cond 使用条件选择操作 "Where"
    gt_out = g.op(
        "Where",
        gt_cond,
        sub(g, self, lambd_op),
        g.op(
            "Constant",
            value_t=torch.tensor(0, dtype=scalar_type.dtype()),
        ),
    )
    # 创建小于条件，比较 self 是否小于负 lambd_op
    lt_cond = lt(g, self, neg(g, lambd_op))
    # 根据 lt_cond 使用条件选择操作 "Where"
    lt_out = g.op(
        "Where",
        lt_cond,
        add(g, self, lambd_op),
        g.op(
            "Constant",
            value_t=torch.tensor(0, dtype=scalar_type.dtype()),
        ),
    )
    # 返回 g 上的加法操作 "Add"，将 gt_out 和 lt_out 相加
    return add(g, gt_out, lt_out)


# 将函数标记为对应 ONNX 符号 "aten::alias" 的符号化函数
@_onnx_symbolic("aten::alias")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 alias，接收 JIT 图上下文 g 和输入 self
def alias(g: jit_utils.GraphContext, self):
    # 直接返回输入 self
    return self


# 将函数标记为对应 ONNX 符号 "aten::unsqueeze" 的符号化函数
@_onnx_symbolic("aten::unsqueeze")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 unsqueeze，接收 JIT 图上下文 g 和输入 self
def unsqueeze(g: jit_utils.GraphContext, self):
    # 函数体未提供，此处略去
    pass
# 使用 `symbolic_helper.parse_args` 装饰器解析参数，指定了函数的输入参数类型为 "v", "i"
# 使用 `_beartype.beartype` 装饰器确保函数输入参数类型的正确性
def unsqueeze(g: jit_utils.GraphContext, self, dim):
    """Implement unsqueezing a pytorch tensor in ONNX by inserting a new dimension at the specified `dim`"""
    
    # 处理负数的 dim 参数
    if dim < 0:
        # 获取张量 self 的秩（rank）
        rank = symbolic_helper._get_tensor_rank(self)
        if rank is not None:
            # 如果秩存在，发出警告并调整 dim 参数
            warnings.warn(
                "ONNX export unsqueeze with negative axis " + str(dim)
                + " might cause the onnx model to be incorrect. "
                + "Negative axis is not supported in ONNX. "
                + "Axis is converted to " + str(dim + rank + 1)
                + " based on input shape at export time. "
                + "Passing an tensor of different rank in execution will be incorrect."
            )
            dim = dim + rank + 1
        else:
            # 如果秩不存在，则调用未实现的处理方法
            return symbolic_helper._unimplemented(
                "unsqueeze", "negative axis with unknown input rank", self
            )

    # 调用 _unsqueeze_helper 方法执行张量的 unsqueeze 操作
    return symbolic_helper._unsqueeze_helper(g, self, axes_i=[dim])


# 使用 `_onnx_symbolic` 装饰器指定函数对应的 ONNX 符号
# 使用 `symbolic_helper.parse_args` 装饰器解析参数，指定了函数的输入参数类型为 "v", "i", "i", "none"
# 使用 `_beartype.beartype` 装饰器确保函数输入参数类型的正确性
def sort(g: jit_utils.GraphContext, self, dim, descending, out=None):
    if out is not None:
        # 如果存在输出参数 out，则调用未实现的处理方法
        symbolic_helper._unimplemented(
            "Sort", "Out parameter is not supported for sort", self
        )
    
    # 获取张量 self 的尺寸
    self_sizes = symbolic_helper._get_tensor_sizes(self)
    try:
        # 尝试获取指定维度 dim 的尺寸
        dim_size = self_sizes[dim]
    except Exception:
        # 如果获取尺寸失败，调用未实现的处理方法
        dim_size = None
        return symbolic_helper._unimplemented("Sort", "input size not accessible", self)

    # 使用 ONNX 操作 "TopK" 执行张量的排序操作，返回前 k 个元素及其索引
    return g.op("TopK", self, k_i=dim_size, axis_i=dim, outputs=2)


# 使用 `_onnx_symbolic` 装饰器指定函数对应的 ONNX 符号
# 使用 `_beartype.beartype` 装饰器确保函数输入参数类型的正确性
def numel(g: jit_utils.GraphContext, self):
    # 调用 _numel_helper 方法获取张量 self 的元素数量
    return symbolic_helper._numel_helper(g, self)


# 使用 `_onnx_symbolic` 装饰器指定函数对应的 ONNX 符号
# 使用 `symbolic_helper.parse_args` 装饰器解析参数，指定了函数的输入参数类型为 "v", "i", "i", "i", "i", "none"
# 使用 `_beartype.beartype` 装饰器确保函数输入参数类型的正确性
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=None):
    if out is not None:
        # 如果存在输出参数 out，则调用未实现的处理方法
        symbolic_helper._unimplemented(
            "TopK", "Out parameter is not supported for topk", self
        )
    if not largest:
        # 如果 largest 参数为 False，则调用未实现的处理方法
        symbolic_helper._unimplemented("TopK", "Ascending TopK is not supported", self)

    # 使用 ONNX 操作 "TopK" 执行张量的 Top-K 操作，返回前 k 个元素及其索引
    return g.op("TopK", self, k_i=k, axis_i=dim, outputs=2)


# 使用 `_onnx_symbolic` 装饰器指定函数对应的 ONNX 符号
# 使用 `_beartype.beartype` 装饰器确保函数输入参数类型的正确性
def convert_element_type(g: jit_utils.GraphContext, self, *args):
    # 获取转换的目标数据类型
    dtype = symbolic_helper._get_const(args[0], "i", "dtype")
    # 使用 ONNX 操作 "Cast" 执行张量元素类型的转换
    return g.op("Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type())


# 使用 `_onnx_symbolic` 装饰器指定函数对应的 ONNX 符号
# 使用 `_beartype.beartype` 装饰器确保函数输入参数类型的正确性
def to(g: jit_utils.GraphContext, self, *args):
    # TODO: Add implementation for ONNX symbolic operation "aten::to"
    # 使用装饰器 @_beartype.beartype 对函数进行类型检查
    @_beartype.beartype
    def is_aten_to_device_only(args):
        # 如果参数个数为4
        if len(args) == 4:
            # 检查是否是 aten::to(Tensor, Device, bool, bool, memory_format) 类型的操作
            return (
                args[0].node().kind() == "prim::device"
                or args[0].type().isSubtypeOf(_C.ListType.ofInts())
                or isinstance(args[0].type(), _C.DeviceObjType)
            )
        # 如果参数个数为5
        elif len(args) == 5:
            # 检查是否是 aten::to(Tensor, Device, ScalarType, bool, bool, memory_format) 类型的操作
            # 当 dtype 为 None 时，表示这是一个 aten::to(device) 的调用
            dtype = symbolic_helper._get_const(args[1], "i", "dtype")
            return dtype is None
        # 如果参数个数为6或7
        elif len(args) in (6, 7):
            # 检查是否是 aten::to(Tensor, ScalarType, Layout, Device, bool, bool, memory_format) 的操作
            # 或者 aten::to(Tensor, ScalarType, Layout, Device, bool, bool, bool, memory_format) 的操作
            # 当 dtype 为 None 时，表示这是一个 aten::to(device) 的调用
            dtype = symbolic_helper._get_const(args[0], "i", "dtype")
            return dtype is None
        # 默认情况，返回 False
        return False

    # 如果是 aten::to(Tensor, Device, ...) 的操作，不处理，直接返回 self
    if is_aten_to_device_only(args):
        return self

    # 如果参数个数为4
    if len(args) == 4:
        # 测试情况下，args[0] 可能是 onnx::Constant[value=<Tensor>]()
        # 在这种情况下，常量值是一个张量而不是整数，
        # 因此 symbolic_helper._maybe_get_const(args[0], 'i') 将不起作用。
        dtype = args[0]
        # 如果 args[0] 是一个值并且其节点类型为 "onnx::Constant"
        if (
            symbolic_helper._is_value(args[0])
            and args[0].node().kind() == "onnx::Constant"
        ):
            # 获取常量节点的值
            tval = symbolic_helper._node_get(args[0].node(), "value")
            # 如果 tval 是一个 torch.Tensor，并且其形状为0
            if isinstance(tval, torch.Tensor) and len(tval.shape) == 0:
                # 将 tval 转换为 Python 中的整数
                tval = tval.item()
                dtype = int(tval)
            else:
                dtype = tval

        # 如果 dtype 是一个值或者是一个 torch.Tensor
        if symbolic_helper._is_value(dtype) or isinstance(dtype, torch.Tensor):
            # aten::to(Tensor, Tensor, bool, bool, memory_format)
            # 获取 args[0] 的数据类型，并创建一个 Cast 操作
            dtype = _type_utils.JitScalarType.from_value(args[0])
            return g.op(
                "Cast",
                self,
                to_i=dtype.onnx_type(),
            )
        else:
            # aten::to(Tensor, ScalarType, bool, bool, memory_format)
            # 忽略 memory_format 参数，创建一个 Cast 操作
            return g.op("Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type())
    
    # 如果参数个数为5
    elif len(args) == 5:
        # aten::to(Tensor, Device, ScalarType, bool, bool, memory_format)
        # 获取 dtype，并忽略 memory_format 参数，创建一个 Cast 操作
        dtype = symbolic_helper._get_const(args[1], "i", "dtype")
        return g.op("Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type())
    elif len(args) == 6:
        # 如果参数个数为6，表示调用的是 aten::to(Tensor, ScalarType, Layout, Device, bool, bool, memory_format) -> Tensor
        # 获取第一个参数作为数据类型（dtype）
        dtype = symbolic_helper._get_const(args[0], "i", "dtype")
        # Layout、device和memory_format参数被忽略
        # 使用 Cast 操作将当前对象 self 转换为指定的 dtype，并返回结果
        return g.op("Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type())
    elif len(args) == 7:
        # 如果参数个数为7，表示调用的是 aten::to(Tensor, ScalarType, Layout, Device, bool, bool, bool, memory_format) -> Tensor
        # 获取第一个参数作为数据类型（dtype）
        dtype = symbolic_helper._get_const(args[0], "i", "dtype")
        # Layout、device和memory_format参数被忽略
        # 使用 Cast 操作将当前对象 self 转换为指定的 dtype，并返回结果
        return g.op("Cast", self, to_i=_type_utils.JitScalarType(dtype).onnx_type())

    # 如果以上两个条件都不满足，则输出不支持的签名信息并返回
    return symbolic_helper._onnx_unsupported("Unknown aten::to signature", self)
# 注解函数，使用 onnx_symbolic 装饰器，指定 aten::repeat 作为对应的 ONNX 符号操作
# 使用 beartype 装饰器确保类型检查
@_onnx_symbolic("aten::repeat")
@_beartype.beartype
def repeat(g: jit_utils.GraphContext, self, repeats):
    # 定义数据类型为 INT64
    dtype = _type_utils.JitScalarType.INT64
    # 根据 repeats 的形状生成全为 1 的张量，数据类型为 dtype
    shape_ = ones_like(g, repeats, dtype)
    # 使用 "Expand" 操作对 self 进行扩展，形状为 shape_
    self = g.op("Expand", self, shape_)
    # 返回通过 "Tile" 操作对 self 进行瓦片化得到的结果
    return g.op("Tile", self, repeats)


# 注解函数，使用 onnx_symbolic 装饰器，指定 aten::repeat_interleave 作为对应的 ONNX 符号操作
# 使用 beartype 装饰器确保类型检查
def repeat_interleave(
    g: jit_utils.GraphContext, self, repeats, dim=None, output_size=None
):
    # 获取 repeats 的维度和大小
    repeats_dim = symbolic_helper._get_tensor_rank(repeats)
    repeats_sizes = symbolic_helper._get_tensor_sizes(repeats)
    # 获取 self 的大小
    input_sizes = symbolic_helper._get_tensor_sizes(self)
    
    # 如果 repeats 的维度或大小未知，则抛出异常
    if repeats_dim is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats rank.",
            self,
        )
    if repeats_sizes is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats size.",
            self,
        )
    if input_sizes is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown input size.",
            self,
        )

    # 如果 dim 为 None，则将 self 展平为一维数组，并将 dim 设置为 0
    # 否则，尝试获取 dim 的标量值
    if symbolic_helper._is_none(dim):
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([-1]))
        )
        dim = torch.tensor(0, dtype=torch.int64)
    else:
        dim = symbolic_helper._maybe_get_scalar(dim)

    # 处理 dim 为负数的情况，转换为正数索引
    if dim < 0:
        dim += len(input_sizes)

    # 复制 input_sizes 列表，并将其中为 None 的元素替换为 0 和 -1
    input_sizes_temp = input_sizes.copy()
    for idx, input_size in enumerate(input_sizes):
        if input_size is None:
            input_sizes[idx], input_sizes_temp[idx] = 0, -1

    # 处理 repeats 是 int 或单值张量的情况
    if repeats_dim == 0 or (repeats_dim == 1 and repeats_sizes[0] == 1):
        # 如果 input_sizes[dim] 为 0，则返回不支持的详细信息
        if input_sizes[dim] == 0:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported along dimension with unknown input size",
                self,
            )
        # 调用 _repeat_interleave_single_value_repeat_helper 处理单值重复的情况
        return symbolic_helper._repeat_interleave_single_value_repeat_helper(
            g, self, repeats, dim
        )

    # 处理 repeats 是一维张量的情况
    # 如果重复维度为1，则处理以下逻辑
    elif repeats_dim == 1:
        # 如果输入大小为0，则返回不支持的详细信息
        if input_sizes[dim] == 0:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported along dimension with unknown input size",
                self,
            )
        # 如果重复数大小为None，则返回不支持动态重复的详细信息
        if repeats_sizes[0] is None:
            return symbolic_helper._onnx_opset_unsupported_detailed(
                "repeat_interleave",
                9,
                13,
                "Unsupported for cases with dynamic repeats",
                self,
            )
        # 断言重复数大小与输入大小一致
        assert (
            repeats_sizes[0] == input_sizes[dim]
        ), "repeats must have the same size as input along dim"
        # 将重复数设置为重复数大小
        reps = repeats_sizes[0]
    else:
        # 抛出符号值错误，说明重复数必须是0维或1维张量
        raise errors.SymbolicValueError("repeats must be 0-dim or 1-dim tensor", self)

    # 最终拆分的列表
    final_splits = list()
    
    # 使用符号帮助函数进行重复插入分裂的帮助函数
    r_splits = symbolic_helper._repeat_interleave_split_helper(g, repeats, reps, 0)
    # 使用符号帮助函数进行重复插入分裂的帮助函数，用于自身对象和维度
    i_splits = symbolic_helper._repeat_interleave_split_helper(g, self, reps, dim)
    
    # 将输入大小设置为-1和1
    input_sizes[dim], input_sizes_temp[dim] = -1, 1
    
    # 对r_splits中的每个r_split进行迭代
    for idx, r_split in enumerate(r_splits):
        # 在维度+1处对i_splits[idx]进行展开
        i_split = unsqueeze(g, i_splits[idx], dim + 1)
        
        # 创建r_concat，包括张量和轴
        r_concat = [
            g.op("Constant", value_t=torch.LongTensor(input_sizes_temp[: dim + 1])),
            r_split,
            g.op("Constant", value_t=torch.LongTensor(input_sizes_temp[dim + 1 :])),
        ]
        # 在轴i=0上进行连接
        r_concat = g.op("Concat", *r_concat, axis_i=0)
        
        # 使用expand函数展开i_split和r_concat
        i_split = expand(g, i_split, r_concat, None)
        
        # 使用reshape_helper函数对i_split进行重塑
        i_split = symbolic_helper._reshape_helper(
            g,
            i_split,
            g.op("Constant", value_t=torch.LongTensor(input_sizes)),
            allowzero=0,
        )
        
        # 将i_split添加到final_splits列表中
        final_splits.append(i_split)
    
    # 返回在维度dim上连接final_splits的操作
    return g.op("Concat", *final_splits, axis_i=dim)
# 用于处理 `aten::pixel_shuffle` 的符号化函数装饰器
@_onnx_symbolic("aten::pixel_shuffle")
# 解析参数的辅助装饰器，期望一个张量和一个整数作为参数
@symbolic_helper.parse_args("v", "i")
# 使用 beartype 进行类型检查的装饰器
@_beartype.beartype
# 像素重排操作的符号化函数定义，接受一个图形上下文对象 g、self 张量和放大因子 upscale_factor
def pixel_shuffle(g: jit_utils.GraphContext, self, upscale_factor):
    # 获取 self 张量的维度信息
    dims = symbolic_helper._get_tensor_sizes(self)
    # 如果维度不是 4，说明不支持的输入维度，返回未实现提示
    if len(dims) != 4:
        return symbolic_helper._unimplemented(
            "pixel_shuffle", "only support 4d input", self
        )
    # 如果任何维度为 None，表明是动态输入形状，需要进行两次重塑操作
    if any(i is None for i in dims[1:]):
        # 执行视图操作，在指定维度上添加新的维度
        after_view = symbolic_helper._reshape_helper(
            g,
            symbolic_helper._unsqueeze_helper(g, self, [2, 3]),
            g.op(
                "Constant",
                value_t=torch.tensor([0, -1, upscale_factor, upscale_factor, 0, 0]),
            ),
            allowzero=0,
        )
        # 执行转置操作，调整张量的维度顺序
        after_transpose = g.op("Transpose", after_view, perm_i=[0, 1, 4, 2, 5, 3])
        # 对于动态输入形状，执行两次重塑操作
        reshape_h = symbolic_helper._reshape_helper(
            g,
            after_transpose,
            g.op("Constant", value_t=torch.tensor([0, 0, -1, 1, 0, 0])),
            allowzero=0,
        )
        reshape_w = symbolic_helper._reshape_helper(
            g,
            reshape_h,
            g.op("Constant", value_t=torch.tensor([0, 0, 0, 0, -1, 1])),
            allowzero=0,
        )
        # 执行挤压操作，去除指定维度
        return symbolic_helper._squeeze_helper(g, reshape_w, [3, 5])
    else:
        # 计算输出通道数
        output_channel = dims[1] // upscale_factor // upscale_factor
        # 执行视图操作，在指定维度上添加新的维度
        after_view = symbolic_helper._reshape_helper(
            g,
            self,
            g.op(
                "Constant",
                value_t=torch.tensor(
                    [
                        -1,
                        output_channel,
                        upscale_factor,
                        upscale_factor,
                        dims[2],
                        dims[3],
                    ]
                ),
            ),
            allowzero=0,
        )
        # 执行转置操作，调整张量的维度顺序
        after_transpose = g.op("Transpose", after_view, perm_i=[0, 1, 4, 2, 5, 3])
        # 执行重塑操作，调整张量的形状
        return symbolic_helper._reshape_helper(
            g,
            after_transpose,
            g.op(
                "Constant",
                value_t=torch.tensor(
                    [
                        -1,
                        output_channel,
                        dims[2] * upscale_factor,
                        dims[3] * upscale_factor,
                    ]
                ),
            ),
            allowzero=0,
        )


# 用于处理 `aten::pixel_unshuffle` 的符号化函数装饰器
@_onnx_symbolic("aten::pixel_unshuffle")
# 解析参数的辅助装饰器，期望一个张量和一个整数作为参数
@symbolic_helper.parse_args("v", "i")
# 使用 beartype 进行类型检查的装饰器
@_beartype.beartype
# 像素反重排操作的符号化函数定义，接受一个图形上下文对象 g、self 张量和缩小因子 downscale_factor
def pixel_unshuffle(g: jit_utils.GraphContext, self, downscale_factor):
    # 获取 self 张量的维度信息
    dims = symbolic_helper._get_tensor_sizes(self)
    # 如果维度不是 4，说明不支持的输入维度，返回未实现提示
    if len(dims) != 4:
        return symbolic_helper._unimplemented(
            "pixel_shuffle", "only support 4d input", self
        )
    if any(i is None for i in dims[1:]):
        # 如果输入维度中有任何一个是 None，表示输入形状是动态的，需要进行两次重塑操作

        # 第一次重塑操作，调用 _unsqueeze_helper 和 _reshape_helper 辅助方法
        reshape_h = symbolic_helper._reshape_helper(
            g,
            symbolic_helper._unsqueeze_helper(g, self, [3]),  # 在第3维度上增加一个维度
            g.op("Constant", value_t=torch.tensor([0, 0, -1, downscale_factor, 0])),  # 定义重塑的形状
            allowzero=0,  # 不允许零值
        )

        # 第二次重塑操作，继续使用 _reshape_helper 辅助方法
        reshape_w = symbolic_helper._reshape_helper(
            g,
            reshape_h,
            g.op("Constant", value_t=torch.tensor([0, 0, 0, 0, -1, downscale_factor])),  # 定义重塑的形状
            allowzero=0,  # 不允许零值
        )

        # 转置操作，交换张量的维度顺序
        after_transpose = g.op("Transpose", reshape_w, perm_i=[0, 1, 3, 5, 2, 4])

        # 最后一次重塑操作，定义最终的输出形状
        final_reshape = symbolic_helper._reshape_helper(
            g,
            after_transpose,
            g.op("Constant", value_t=torch.tensor([0, -1, 1, 1, 0, 0])),  # 定义重塑的形状
            allowzero=0,  # 不允许零值
        )

        # 使用 _squeeze_helper 辅助方法去除指定维度上的尺寸为1的维度
        return symbolic_helper._squeeze_helper(g, final_reshape, [2, 3])

    else:
        # 如果所有输入维度都已知，则按照静态输入形状处理

        # 计算输出通道数
        output_channel = dims[1] * downscale_factor * downscale_factor

        # 将输入张量重塑为指定形状，使用 _reshape_helper 辅助方法
        after_view = symbolic_helper._reshape_helper(
            g,
            self,
            g.op(
                "Constant",
                value_t=torch.tensor(
                    [
                        -1,
                        dims[1],
                        dims[2] // downscale_factor,
                        downscale_factor,
                        dims[3] // downscale_factor,
                        downscale_factor,
                    ]
                ),
            ),
            allowzero=0,  # 不允许零值
        )

        # 转置操作，交换张量的维度顺序
        after_transpose = g.op("Transpose", after_view, perm_i=[0, 1, 3, 5, 2, 4])

        # 最后一次重塑操作，定义最终的输出形状
        return symbolic_helper._reshape_helper(
            g,
            after_transpose,
            g.op(
                "Constant",
                value_t=torch.tensor(
                    [
                        -1,
                        output_channel,
                        dims[2] // downscale_factor,
                        dims[3] // downscale_factor,
                    ]
                ),
            ),
            allowzero=0,  # 不允许零值
        )
# 使用 @_beartype 装饰器对函数进行类型检查和注解
@_beartype.beartype
# 定义一个通用的循环神经网络函数，用于导出模型到ONNX格式
def _generic_rnn(
    g: jit_utils.GraphContext,  # 图形上下文对象，用于构建计算图
    variant,  # 模型变体，可以是RNN、GRU或LSTM
    input,  # 输入张量
    initial_states,  # 初始状态张量（可以是单个张量或元组，对于LSTM是(h0, c0)）
    all_weights,  # 所有权重参数的列表
    has_biases,  # 是否包含偏置
    num_layers,  # 网络层数
    dropout,  # 是否使用dropout
    train,  # 是否处于训练模式
    bidirectional,  # 是否是双向RNN
    batch_first=None,  # 是否批次优先
    batch_sizes=None,  # 批次大小列表（用于不同大小的批次序列）
):
    # 发出警告，提醒用户在使用ONNX格式导出时要注意批次大小为1的问题
    warnings.warn(
        "Exporting a model to ONNX with a batch_size other than 1, "
        + "with a variable length with "
        + variant
        + " can cause an error "
        + "when running the ONNX model with a different batch size. "
        + "Make sure to save the model with a batch size of 1, "
        + "or define the initial states (h0/c0) as inputs of the model. "
    )

    # ONNX支持的激活函数列表
    onnxActivations = [
        "Relu",
        "Tanh",
        "Sigmoid",
        "Affine",
        "LeakyRelu",
        "ThresholdedRelu",
        "ScaledTanh",
        "HardSigmoid",
        "Elu",
        "Softsign",
        "Softplus",
    ]
    # 将激活函数转换为小写形式，并创建对应的字典映射
    variantToOnnxActivationMap = dict(
        zip([act_fun.lower() for act_fun in onnxActivations], onnxActivations)
    )
    # 每个层次的权重数目（包括偏置）
    weights_per_layer = 4 if has_biases else 2

    # 如果是LSTM并且权重列表长度与期望长度不符，则返回未实现错误信息
    if variant == "LSTM" and len(all_weights) != num_layers * weights_per_layer * (
        1 + bidirectional
    ):
        return symbolic_helper._unimplemented("LSTM", "LSTMs with projections", input)
    
    # 断言权重列表长度符合预期的数量
    assert len(all_weights) == num_layers * weights_per_layer * (1 + bidirectional)

    # 将权重按层次划分，每个层次的权重保存在layer_weights列表中
    layer_weights = [
        all_weights[i : i + weights_per_layer]
        for i in range(0, len(all_weights), weights_per_layer)
    ]

    # 如果设置了batch_first，则将输入转置为(seq, batch, feat)形式
    if batch_first:
        input = g.op("Transpose", input, perm_i=[1, 0, 2])

    # 如果启用了dropout并且处于训练模式，则返回未实现错误信息
    if dropout and train:
        return symbolic_helper._unimplemented(
            "RNN/GRU/LSTM", "dropout in training mode", input
        )

    # 如果模型变体以"RNN"开头，则选择对应的ONNX激活函数
    if variant.startswith("RNN"):
        nonlinearity = variantToOnnxActivationMap[variant[4:].lower()]
        variant = "RNN"

    # 获取隐藏层的大小
    w_hh = all_weights[1]
    hidden_size = symbolic_helper._get_tensor_dim_size(w_hh, 1)

    # 如果隐藏层大小未知，则返回未实现错误信息
    if hidden_size is None:
        return symbolic_helper._unimplemented(
            "RNN/GRU/LSTM", "unknown hidden size", input
        )

    # 判断是否是单向RNN
    unidirectional = not bidirectional

    # 初始化上一个输出为输入张量
    prev_output = input

    # 初始化隐藏状态输出列表
    h_outs = []

    # 如果是RNN或GRU，则初始化初始隐藏状态h0
    if variant == "RNN" or variant == "GRU":
        h0 = initial_states
    
    # 如果是LSTM，则初始化初始隐藏状态h0和细胞状态c0
    elif variant == "LSTM":
        h0, c0 = initial_states
        # 初始化细胞状态输出列表
        c_outs = []

    # 如果未提供批次大小信息，则使用unused函数（该函数未提供具体实现）
    sequence_lens = unused(g) if batch_sizes is None else batch_sizes

    # 根据不同的RNN变体，重新排列输入数据以适应ONNX格式
    if variant == "GRU":
        # pytorch格式是重置门、输入门、隐藏状态
        # ONNX格式是输入、重置门、隐藏状态
        reform_permutation = [(1, 2), (0, 1), (2, 3)]
    
    elif variant == "LSTM":
        # pytorch格式是输入、遗忘门、细胞状态、输出
        # ONNX格式是输入、输出、遗忘门、细胞状态
        reform_permutation = [(0, 1), (3, 4), (1, 3)]

    # 返回带有类型检查的函数定义
    @_beartype.beartype
    @_beartype.beartype
    # 使用装饰器对函数进行类型检查和类型提示
    def transform_weights_no_bias(layer_index):
        # 获取指定层的权重数据
        weights = layer_weights[layer_index]
        # 根据变体类型选择权重数据的组成方式
        if variant == "RNN":
            weight_ih, weight_hh = weights
        elif variant == "GRU" or variant == "LSTM":
            # 对权重数据进行重组，以符合指定的隐藏单元大小和重组顺序
            weight_ih, weight_hh = (
                reform_weights(g, w, hidden_size, reform_permutation) for w in weights
            )
        # 对权重数据进行扩展，以匹配特定的操作要求
        return tuple(
            symbolic_helper._unsqueeze_helper(g, x, [0]) for x in (weight_ih, weight_hh)  # type: ignore[possibly-undefined]
        )

    @_beartype.beartype
    # 使用装饰器对函数进行类型检查和类型提示
    def transform_weights(layer_index):
        # 获取指定层的权重数据
        weights = layer_weights[layer_index]
        # 根据变体类型选择权重数据的组成方式
        if variant == "RNN":
            weight_ih, weight_hh, bias_ih, bias_hh = weights
        elif variant == "GRU" or variant == "LSTM":
            # 对权重数据进行重组，以符合指定的隐藏单元大小和重组顺序
            weight_ih, weight_hh, bias_ih, bias_hh = (
                reform_weights(g, w, hidden_size, reform_permutation) for w in weights
            )
        # 对偏置数据进行连接，以匹配特定的操作要求
        bias_concat = g.op("Concat", bias_ih, bias_hh, axis_i=0)  # type: ignore[possibly-undefined]
        # 对权重和偏置数据进行扩展，以匹配特定的操作要求
        return tuple(
            symbolic_helper._unsqueeze_helper(g, x, [0])
            for x in (weight_ih, weight_hh, bias_concat)  # type: ignore[possibly-undefined]
        )

    @_beartype.beartype
    # 使用装饰器对函数进行类型检查和类型提示
    def retrieve_state(x, start, end):
        # 如果层数为1，则直接返回输入 x；否则，对输入 x 进行切片操作
        return (
            x
            if num_layers == 1
            else symbolic_helper._slice_helper(
                g, x, axes=[0], starts=[start], ends=[end]
            )
        )

    if batch_first:
        # 如果 batch_first 为 True，则对 prev_output 进行转置，调整维度顺序
        prev_output = g.op("Transpose", prev_output, perm_i=[1, 0, 2])

    # 如果有多层输出，则对多个输出进行连接操作
    h_outs = h_out if num_layers == 1 else g.op("Concat", *h_outs, axis_i=0)  # type: ignore[possibly-undefined]

    # 根据变体类型选择返回的输出内容
    if variant == "RNN" or variant == "GRU":
        return prev_output, h_outs
    elif variant == "LSTM":
        # 如果是 LSTM 变体，则对 c_outs 进行连接操作，并返回多个输出内容
        c_outs = c_out if num_layers == 1 else g.op("Concat", *c_outs, axis_i=0)  # type: ignore[possibly-undefined]
        return prev_output, h_outs, c_outs
# 使用装饰器将函数注册为 ONNX 符号化函数，处理 LSTM 类型的操作
@_onnx_symbolic("aten::lstm")
# 使用 Beartype 进行类型检查和验证参数
@_beartype.beartype
# 定义 LSTM 操作的符号化函数，接收图上下文和动态参数列表
def lstm(g: jit_utils.GraphContext, *args):
    # 如果第四个参数是张量列表，则调用 _lstm_packed 函数处理压缩输入序列的 LSTM 操作
    if symbolic_helper._is_tensor_list(args[3]):
        return _lstm_packed(g, *args)
    else:
        # 否则调用 _lstm_full 函数处理完整输入序列的 LSTM 操作
        return _lstm_full(g, *args)


# 使用装饰器将函数注册为 ONNX 符号化函数，处理 LSTM 单元格操作
@_onnx_symbolic("aten::lstm_cell")
# 使用 Beartype 进行类型检查和验证参数
@_beartype.beartype
# 定义 LSTM 单元格操作的符号化函数，接收图上下文和多个参数
def lstm_cell(g: jit_utils.GraphContext, self, hidden, w_ih, w_hh, b_ih, b_hh):
    # 对输入张量进行增维处理
    input = symbolic_helper._unsqueeze_helper(g, self, [0])
    # 将隐藏状态列表展开成单独的张量列表
    hidden = symbolic_helper._unpack_list(hidden)
    # 对每个隐藏状态张量进行增维处理
    hidden = [symbolic_helper._unsqueeze_helper(g, x, [0]) for x in hidden]
    # 根据是否存在偏置张量，确定权重的组合方式
    weight = (
        (w_ih, w_hh, b_ih, b_hh) if symbolic_helper._is_tensor(b_ih) else (w_ih, w_hh)
    )
    # 判断是否存在偏置张量
    has_biases = True if symbolic_helper._is_tensor(b_ih) else False
    # 调用通用 RNN 函数处理 LSTM 单元格操作，获取输出状态
    _, h_outs, c_outs = _generic_rnn(
        g,
        "LSTM",
        input,
        hidden,
        weight,
        has_biases,
        num_layers=1,
        dropout=0,
        train=0,
        bidirectional=False,
        batch_first=False,
    )
    # 对输出状态进行降维处理
    return symbolic_helper._squeeze_helper(g, h_outs, [0]), symbolic_helper._squeeze_helper(g, c_outs, [0])


# 使用装饰器将函数注册为 ONNX 符号化函数，处理 GRU 和一层隐藏 RNN 类型的操作
@_onnx_symbolic("aten::gru", decorate=[symbolic_helper._apply_params("GRU"), _export("gru")])
@_onnx_symbolic("aten::rnn_tanh", decorate=[symbolic_helper._apply_params("RNN_TANH"), _export("rnn_tanh")])
@_onnx_symbolic("aten::rnn_relu", decorate=[symbolic_helper._apply_params("RNN_RELU"), _export("rnn_relu")])
# 定义一层隐藏 RNN 操作的符号化函数，接收操作类型参数
def _one_hidden_rnn(kind: str):
    # 使用装饰器解析函数参数并进行 Beartype 类型检查
    @symbolic_helper.parse_args("v", "v", "v", "i", "i", "f", "i", "i", "i")
    @_beartype.beartype
    # 定义一个函数用于执行完整的 RNN 操作，包括未打包和打包输入的情况
    def _rnn_full(
        g,
        input,
        hidden,
        weight_v,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
    ):
        # 将权重列表解包成权重张量
        weight = symbolic_helper._unpack_list(weight_v)
        # 调用通用的 RNN 函数，执行 RNN 计算
        return _generic_rnn(
            g,
            kind,  # kind 变量未定义，可能是全局变量或者需要从外部引入
            input,
            hidden,
            weight,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )
    
    # 使用装饰器将函数声明为符号化函数，并解析相应的参数类型
    @symbolic_helper.parse_args("v", "v", "v", "v", "i", "i", "f", "i", "i")
    def _rnn_packed(
        g,
        input,
        batch_sizes,
        hidden,
        weight_v,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
    ):
        # 将权重列表解包成权重张量
        weight = symbolic_helper._unpack_list(weight_v)
        # 调用通用的 RNN 函数，执行 RNN 计算，支持输入是打包形式
        return _generic_rnn(
            g,
            kind,  # kind 变量未定义，可能是全局变量或者需要从外部引入
            input,
            hidden,
            weight,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_sizes=batch_sizes,  # 如果输入是打包形式，则传递 batch_sizes 参数
        )
    
    # 定义一个符号化函数，根据输入参数的类型选择调用 _rnn_full 或 _rnn_packed 函数
    def symbolic(g, *args):
        # 如果第四个参数是张量列表，则调用 _rnn_packed 函数
        if symbolic_helper._is_tensor_list(args[3]):
            return _rnn_packed(g, *args)
        else:
            # 否则调用 _rnn_full 函数
            return _rnn_full(g, *args)
    
    # 返回符号化函数
    return symbolic
# 为 ONNX 符号化注册函数 "_dim_arange"
@_onnx_symbolic("aten::_dim_arange")
# 解析函数参数为 "v", "i"
@symbolic_helper.parse_args("v", "i")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 "_dim_arange"，接受 GraphContext 对象 g，以及 like 和 dim 两个参数
def _dim_arange(g: jit_utils.GraphContext, like, dim):
    # 使用输入对象 "like" 的形状创建一个 Shape 运算
    like_shape = g.op("Shape", like)
    # 从 like_shape 中根据给定维度 dim 提取索引，创建 Gather 运算
    stop = g.op(
        "Gather", like_shape, g.op("Constant", value_t=torch.tensor(dim)), axis_i=0
    )
    # 调用 arange 函数生成一个 arange 运算，返回结果
    # arange 函数的具体定义在函数 arange 中实现
    return arange(g, stop, 4, None, None, None)


# 为 ONNX 符号化注册函数 "aten::detach"
@_onnx_symbolic("aten::detach")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 "detach"，接受 GraphContext 对象 g 和 input 参数
def detach(g: jit_utils.GraphContext, input):
    # 删除 aten::detach 节点，因为 ONNX 仅支持推理
    return input


# 为 ONNX 符号化注册函数 "aten::contiguous"
@_onnx_symbolic("aten::contiguous")
# 解析函数参数为 "v", "i"
@symbolic_helper.parse_args("v", "i")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 "contiguous"，接受 GraphContext 对象 g、input 和 memory_format 三个参数
def contiguous(g: jit_utils.GraphContext, input, memory_format):
    # 如果 memory_format 大于 2，则抛出错误，因为 ONNX 不支持这种格式
    if memory_format > 2:  # 允许的值为 any, preserve 和 contiguous_format
        raise errors.SymbolicValueError(
            "onnx memory_format support is not implemented", input
        )
    # 返回输入对象 input，因为没有改变
    return input


# 为 ONNX 符号化注册函数 "aten::_pack_padded_sequence"
@_onnx_symbolic("aten::_pack_padded_sequence")
# 解析函数参数为 "v", "v", "i"
@symbolic_helper.parse_args("v", "v", "i")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 "_pack_padded_sequence"，接受 GraphContext 对象 g 和 input、lengths、batch_first 三个参数
def _pack_padded_sequence(g: jit_utils.GraphContext, input, lengths, batch_first):
    # 当前 ONNX 中没有 PackPadded 运算符，我们依赖优化步骤稍后移除此运算符
    # 如果所有的 PackPadded 运算符都无法优化，则会报错
    if batch_first:
        # 如果 batch_first 为 True，则使用 Transpose 运算符交换 input 的第一和第二维度
        input = g.op("Transpose", input, perm_i=[1, 0, 2])
    # 检查 lengths 是否为 Tensor 类型，如果不是，则抛出错误
    if not lengths.type().isSubtypeOf(torch._C.TensorType.get()):
        raise errors.SymbolicValueError(
            "'lengths' must be a Tensor for ONNX export", input
        )
    # 将 lengths 强制转换为 INT32 类型的 TensorProtoDataType
    lengths = g.op("Cast", lengths, to_i=_C_onnx.TensorProtoDataType.INT32)
    # 返回一个 prim::PackPadded 运算符，输入为 input 和 lengths，输出为 2 个结果
    return g.op("prim::PackPadded", input, lengths, outputs=2)


# 为 ONNX 符号化注册函数 "aten::_pad_packed_sequence"
@_onnx_symbolic("aten::_pad_packed_sequence")
# 解析函数参数为 "v", "v", "i", "t", "v"
@symbolic_helper.parse_args("v", "v", "i", "t", "v")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 "_pad_packed_sequence"，接受 GraphContext 对象 g 和 data、batch_sizes、batch_first、padding_value、total_length 五个参数
def _pad_packed_sequence(
    g: jit_utils.GraphContext,
    data,
    batch_sizes,
    batch_first,
    padding_value,
    total_length,
):
    # 忽略 total_length，因为 _symbolic_pad_packed_sequence 不支持它
    # 在使用 data_parallel 模型训练时才有用/使用，所以对于 ONNX 来说它无关紧要
    # 使用 prim::PadPacked 运算符对 data 和 batch_sizes 进行填充，输出为 2 个结果
    data, lengths = g.op("prim::PadPacked", data, batch_sizes, outputs=2)
    # 如果 batch_first 为 True，则使用 Transpose 运算符交换 data 的第一和第二维度
    if batch_first:
        data = g.op("Transpose", data, perm_i=[1, 0, 2])
    # 返回 data 和 lengths 两个结果
    return data, lengths


# 为 ONNX 符号化注册函数 "aten::randint"
@_onnx_symbolic("aten::randint")
# 应用 Beartype 类型检查装饰器
@_beartype.beartype
# 定义函数 "randint"，接受 GraphContext 对象 g 和 low、high、shapes、dtype、*options 参数
def randint(g: jit_utils.GraphContext, low, high, shapes, dtype, *options):
    # 从 dtype 中获取常量值，预期其为整数类型，如果不是则抛出异常
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # 使用 symbolic_helper._get_const 函数获取 low 的常量值，如果获取失败则引发异常
    low_i = symbolic_helper._get_const(low, "i", "low")
    # 使用 symbolic_helper._get_const 函数获取 high 的常量值，如果获取失败则引发异常
    high_i = symbolic_helper._get_const(high, "i", "high")
    # 如果未提供 dtype，则默认使用 INT64 类型
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.INT64
    else:
        # 否则，根据提供的 dtype 创建相应的 JitScalarType
        scalar_type = _type_utils.JitScalarType(dtype)
    # 如果 low_i 为 None，则说明 low 不是常量，无法处理，抛出不支持的异常
    if low_i is None:
        raise symbolic_helper._onnx_unsupported("randint", low)
    # 如果 high_i 为 None，则说明 high 不是常量，无法处理，抛出不支持的异常
    if high_i is None:
        raise symbolic_helper._onnx_unsupported("randint", high)

    # 获取 shapes 的常量值，如果 shapes 不是常量，则 shape 为 None
    shape = symbolic_helper._maybe_get_const(shapes, "is")
    # 如果 shape 是符号值，则创建一个形状为 shape 的常数张量，值为 0
    if symbolic_helper._is_value(shape):
        shape_const = g.op(
            "ConstantOfShape",
            shapes,
            value_t=torch.tensor([0], dtype=torch.float),
        )
        # 使用 RandomUniformLike 操作生成一个符合 shape_const 形状的均匀分布随机数张量
        randn = g.op(
            "RandomUniformLike",
            shape_const,
            low_f=low_i,
            high_f=high_i,
        )
    else:
        # 否则，使用 RandomUniform 操作生成一个符合 shape 形状的均匀分布随机数张量
        randn = g.op(
            "RandomUniform",
            shape_i=shape,
            low_f=low_i,
            high_f=high_i,
        )

    # 将生成的随机数张量转换为整数类型
    int_dtype = _type_utils.JitScalarType.INT64
    randint = g.op("Cast", randn, to_i=int_dtype.onnx_type())
    # 如果生成的随机数类型与 scalar_type 不同，则再次进行类型转换
    if int_dtype != scalar_type:
        randint = g.op("Cast", randint, to_i=scalar_type.onnx_type())
    # 返回最终生成的随机整数张量
    return randint
# 声明一个函数randint_like，用于处理aten::randint_like操作的符号化函数
@_onnx_symbolic("aten::randint_like")
@_beartype.beartype
def randint_like(g: jit_utils.GraphContext, self, low, high, dtype, *options):
    # 从dtype参数获取整数常量，如果无法获取，则抛出异常
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # 从low参数获取整数常量，如果无法获取，则抛出异常
    low_i = symbolic_helper._get_const(low, "i", "low")
    # 从high参数获取整数常量，如果无法获取，则抛出异常
    high_i = symbolic_helper._get_const(high, "i", "high")
    # 如果dtype为空，则设置scalar_type为INT64类型
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.INT64
    else:
        # 否则根据dtype设置scalar_type为相应的JitScalarType类型
        scalar_type = _type_utils.JitScalarType(dtype)
    # 如果low_i为空，则抛出不支持的异常
    if low_i is None:
        raise symbolic_helper._onnx_unsupported("randint", low)
    # 如果high_i为空，则抛出不支持的异常
    if high_i is None:
        raise symbolic_helper._onnx_unsupported("randint", high)

    # 使用g.op函数创建一个"RandomUniformLike"操作节点，生成一个介于low_i和high_i之间的随机数
    randn = g.op(
        "RandomUniformLike",
        self,
        low_f=low_i,
        high_f=high_i,
    )

    # 将randn强制转换为整数类型
    int_dtype = _type_utils.JitScalarType.INT64
    randint = g.op("Cast", randn, to_i=int_dtype.onnx_type())
    # 如果int_dtype不等于scalar_type，则将randint再次强制转换为scalar_type类型
    if int_dtype != scalar_type:
        randint = g.op("Cast", randint, to_i=scalar_type.onnx_type())
    # 返回最终的整数随机数结果
    return randint


# 声明一个函数randn，用于处理aten::randn操作的符号化函数
@_onnx_symbolic("aten::randn")
@_beartype.beartype
def randn(g: jit_utils.GraphContext, shapes, dtype, *options):
    # 从dtype参数获取常量整数，如果无法获取，则设置默认为FLOAT类型
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # 如果dtype为空，则设置scalar_type为FLOAT类型，否则根据dtype设置scalar_type为相应的JitScalarType类型
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    # 从shapes参数获取常量整数，如果无法获取，则返回一个ConstantOfShape节点
    shape = symbolic_helper._maybe_get_const(shapes, "is")
    # 如果shape是一个值节点，则创建一个ConstantOfShape节点，用0填充，并返回一个"RandomNormalLike"操作节点
    if symbolic_helper._is_value(shape):
        shape_const = g.op(
            "ConstantOfShape",
            shapes,
            value_t=torch.tensor([0], dtype=torch.float),
        )
        return g.op(
            "RandomNormalLike",
            shape_const,
            dtype_i=scalar_type.onnx_type(),
        )
    # 否则返回一个"RandomNormal"操作节点，生成一个形状为shape的随机正态分布数据
    return g.op(
        "RandomNormal",
        shape_i=shape,
        dtype_i=scalar_type.onnx_type(),
    )


# 声明一个函数rand，用于处理aten::rand操作的符号化函数
@_onnx_symbolic("aten::rand")
@_beartype.beartype
def rand(g: jit_utils.GraphContext, shapes, dtype, *options):
    # 从dtype参数获取常量整数，如果无法获取，则设置默认为FLOAT类型
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # 如果dtype为空，则设置scalar_type为FLOAT类型，否则根据dtype设置scalar_type为相应的JitScalarType类型
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    # 从shapes参数获取常量整数，如果无法获取，则返回一个ConstantOfShape节点
    shape = symbolic_helper._maybe_get_const(shapes, "is")
    # 如果shape是一个值节点，则创建一个ConstantOfShape节点，用0填充，并返回一个"RandomUniformLike"操作节点
    if symbolic_helper._is_value(shape):
        shape_const = g.op(
            "ConstantOfShape",
            shapes,
            value_t=torch.tensor([0], dtype=torch.float),
        )
        return g.op(
            "RandomUniformLike",
            shape_const,
            dtype_i=scalar_type.onnx_type(),
        )
    # 否则返回一个"RandomUniform"操作节点，生成一个形状为shape的均匀分布随机数
    return g.op(
        "RandomUniform",
        shape_i=shape,
        dtype_i=scalar_type.onnx_type(),
    )


# 声明一个函数randn_like，用于处理aten::randn_like操作的符号化函数
@_onnx_symbolic("aten::randn_like")
@_beartype.beartype
def randn_like(
    g: jit_utils.GraphContext,
    self,
    dtype,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    # 从dtype参数获取常量整数，如果无法获取，则抛出异常
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # 如果未提供数据类型参数，则使用浮点型作为默认类型
    if dtype is None:
        # 使用_type_utils模块中的JitScalarType类，根据指定的值（FLOAT）创建一个标量类型对象
        scalar_type = _type_utils.JitScalarType.from_value(
            self, _type_utils.JitScalarType.FLOAT
        )
    else:
        # 使用_type_utils模块中的JitScalarType类，根据给定的数据类型创建一个标量类型对象
        scalar_type = _type_utils.JitScalarType(dtype)
    # 调用ONNX图操作（Operation），生成一个“RandomNormalLike”操作节点，使用当前对象作为输入，并指定数据类型
    return g.op("RandomNormalLike", self, dtype_i=scalar_type.onnx_type())
# 定义一个在ONNX符号化环境中的函数装饰器，用于将aten::rand_like操作转换为对应的ONNX操作
# 使用beartype装饰器对函数进行类型检查和修饰
@_onnx_symbolic("aten::rand_like")
@_beartype.beartype
def rand_like(
    g: jit_utils.GraphContext,
    self,
    dtype,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=None,
):
    # 将dtype转换为常量，并确保其为整数类型
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    # 如果dtype为None，则设定为浮点数类型
    if dtype is None:
        dtype = _type_utils.JitScalarType.from_value(
            self, _type_utils.JitScalarType.FLOAT
        )
    # 调用ONNX操作，在图中插入RandomUniformLike节点，模拟PyTorch的rand_like行为
    return g.op(
        "RandomUniformLike", self, dtype_i=_type_utils.JitScalarType(dtype).onnx_type()
    )


# 定义一个在ONNX符号化环境中的函数装饰器，用于将aten::rrelu操作转换为对应的ONNX操作
# 使用parse_args装饰器解析函数参数，确保参数的类型和格式正确
@_onnx_symbolic("aten::rrelu")
@symbolic_helper.parse_args("v", "f", "f", "i", "none")
@_beartype.beartype
def rrelu(g: jit_utils.GraphContext, input, lower, upper, training, generator):
    # 如果不是训练模式，则计算斜率并返回LeakyRelu节点
    if not training:
        slope = (upper + lower) / 2.0
        return g.op("LeakyRelu", input, alpha_f=slope)
    # 在图中插入RandomUniformLike节点，模拟PyTorch的rrelu行为
    p = g.op("RandomUniformLike", input, high_f=upper, low_f=lower)
    return g.op("PRelu", input, p)


# 定义一个在ONNX符号化环境中的函数装饰器，用于将aten::bernoulli操作转换为对应的ONNX操作
# 使用beartype装饰器对函数进行类型检查和修饰
@_onnx_symbolic("aten::bernoulli")
@_beartype.beartype
def bernoulli(g: jit_utils.GraphContext, input, p=None, generator=None, out=None):
    # 如果指定了out参数，则抛出未实现的异常
    if out is not None and not symbolic_helper._is_none(out):
        symbolic_helper._unimplemented(
            "Bernoulli", "out parameter is not supported for bernoulli", input
        )
    # 如果指定了generator参数，则抛出未实现的异常
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Bernoulli", "generator is not supported for bernoulli", input
        )

    # 获取输入的数据类型，并确保其可访问
    dtype = _type_utils.JitScalarType.from_value(
        input, _type_utils.JitScalarType.UNDEFINED
    )
    # 如果dtype未定义，则抛出未实现的异常
    if dtype == _type_utils.JitScalarType.UNDEFINED:
        return symbolic_helper._unimplemented(
            "Bernoulli", "input dtype not accessible", input
        )

    # 在图中插入RandomUniformLike节点，生成介于0到1之间的随机数
    rands = g.op(
        "RandomUniformLike",
        input,
        high_f=1.0,
        low_f=0.0,
        dtype_i=dtype.onnx_type(),
    )
    # 如果指定了p参数，则使用该参数；否则使用输入作为概率值
    prob = p if p is not None and not symbolic_helper._is_none(p) else input
    # 在图中插入Less节点，比较随机数和概率值，并返回布尔结果
    output = g.op("Less", rands, prob)
    # 在图中插入Cast节点，将布尔结果转换为指定的数据类型
    return g.op("Cast", output, to_i=dtype.onnx_type())


# 定义一个在ONNX符号化环境中的函数装饰器，用于将aten::log_sigmoid操作转换为对应的ONNX操作
# 使用parse_args装饰器解析函数参数，确保参数的类型和格式正确
@_onnx_symbolic("aten::log_sigmoid")
@symbolic_helper.parse_args("v")
@_beartype.beartype
def log_sigmoid(g: jit_utils.GraphContext, input):
    # 在图中插入Sigmoid节点，计算输入数据的Sigmoid函数值
    p = g.op("Sigmoid", input)
    # 在图中插入Log节点，计算Sigmoid函数值的自然对数
    return g.op("Log", p)


# 定义一个在ONNX符号化环境中的函数装饰器，用于将aten::erf操作转换为对应的ONNX操作
# 使用parse_args装饰器解析函数参数，确保参数的类型和格式正确
@_onnx_symbolic("aten::erf")
@symbolic_helper.parse_args("v")
@_beartype.beartype
def erf(g: jit_utils.GraphContext, input):
    # 在图中插入Erf节点，计算输入数据的误差函数值
    return g.op("Erf", input)


# 定义一个在ONNX符号化环境中的函数装饰器，用于将aten::flatten操作转换为对应的ONNX操作
# 使用quantized_args装饰器解析函数参数，并确保参数的类型和格式正确
# 使用parse_args装饰器解析函数参数，确保参数的类型和格式正确
@_onnx_symbolic("aten::flatten")
@symbolic_helper.quantized_args(True, False, False)
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
def flatten(g: jit_utils.GraphContext, input, start_dim, end_dim):
    # 获取输入张量的维度
    dim = symbolic_helper._get_tensor_rank(input)
    # 如果无法确定维度，则抛出未实现的异常
    if dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
            input,
        )

    # 如果维度为0，则调用辅助函数对输入张量进行形状重塑
    if dim == 0:
        return symbolic_helper._reshape_helper(g, input, [1])
    # 如果输入张量的维度为1，则返回一个“Identity”操作，表示输入张量本身
    if dim == 1:
        return g.op("Identity", input)
    
    # TODO: 当ONNX操作集版本为11时，可以允许负数作为axes参数，因此这里需要移除这个条件判断
    if end_dim < 0:
        end_dim = dim + end_dim
    
    # 当输出张量的形状是2维时，使用ONNX的Flatten操作
    if start_dim == 1 and end_dim == dim - 1:
        return g.op("Flatten", input, axis_i=start_dim)
    
    # 当输出张量的形状是2维时，使用ONNX的Flatten操作
    if start_dim == 0 and end_dim == dim - 2:
        return g.op("Flatten", input, axis_i=end_dim + 1)

    # 使用symbolic_helper模块中的_flatten_helper函数来执行扁平化操作
    return symbolic_helper._flatten_helper(g, input, start_dim, end_dim, dim)
# 将函数标记为在ONNX中符号化处理"aten::nonzero"
# 使用装饰器解析参数"v"，并应用Beartype类型检查
@_onnx_symbolic("aten::nonzero")
@symbolic_helper.parse_args("v")
@_beartype.beartype
def nonzero(g: jit_utils.GraphContext, input):
    """Emitted from `torch.nonzero(x, as_tuple=False)`"""
    # 调用t函数，生成ONNX操作"NonZero"，并返回结果
    return t(g, g.op("NonZero", input))


# 将函数标记为在ONNX中符号化处理"aten::nonzero_numpy"
# 使用装饰器应用Beartype类型检查
@_onnx_symbolic("aten::nonzero_numpy")
# Emitted from `torch.nonzero(x, as_tuple=True)`
def nonzero_numpy(g: jit_utils.GraphContext, input, _outputs=None):
    # 调用unbind函数，返回非零元素索引的结果
    return unbind(g, nonzero(g, input), 1, _outputs=_outputs)


# 将函数标记为在ONNX中符号化处理"aten::isnan"
# 使用装饰器解析参数"v"，并应用Beartype类型检查
@_onnx_symbolic("aten::isnan")
@symbolic_helper.parse_args("v")
@_beartype.beartype
def isnan(g: jit_utils.GraphContext, input):
    # 生成ONNX操作"IsNaN"，并返回结果
    output = g.op("IsNaN", input)
    return output


# 将函数标记为在ONNX中符号化处理"aten::any"
# 使用装饰器应用Beartype类型检查
@_onnx_symbolic("aten::any")
@_beartype.beartype
def _any(g: jit_utils.GraphContext, *args):
    # aten::any(Tensor self)
    # 如果参数长度为1，将输入作为参数进行处理
    if len(args) == 1:
        input = args[0]
        dim, keepdim = None, 0
    # aten::any(Tensor self, int[]? dim, bool keepdim)
    else:
        # 否则，从参数中解析dim和keepdim
        input, dim, keepdim = args
        # 可能是整数列表或单个整数，解析dim为整数列表
        dim = symbolic_helper._parse_arg(dim, "t")
        dim = [int(d) for d in dim.view(-1)]
        keepdim = symbolic_helper._parse_arg(keepdim, "i")
    
    # 将输入转换为INT64类型的ONNX张量
    input = g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.INT64)
    # 使用_reducesum_helper函数进行求和操作，按照dim指定的轴求和并保持维度
    input_sum = symbolic_helper._reducesum_helper(
        g, input, axes_i=dim, keepdims_i=keepdim
    )
    # 返回input_sum是否大于0的比较结果
    return gt(g, input_sum, g.op("Constant", value_t=torch.tensor(0, dtype=torch.long)))


# 将函数标记为在ONNX中符号化处理"aten::all"
# 使用装饰器应用Beartype类型检查
@_onnx_symbolic("aten::all")
@_beartype.beartype
def _all(g: jit_utils.GraphContext, *args):
    # 对输入进行逻辑非操作
    input = g.op("Not", args[0])
    # aten::all(Tensor self)
    # 如果参数长度为1，对输入进行逻辑非操作后返回_not函数的结果
    if len(args) == 1:
        return g.op("Not", _any(g, input))
    # aten::all(Tensor self, int[]? dim, bool keepdim)
    # 否则，对输入进行逻辑非操作后调用_any函数，传递dim和keepdim参数
    else:
        return g.op("Not", _any(g, input, args[1], args[2]))


# 将函数标记为在ONNX中符号化处理"aten::narrow"
# 使用装饰器解析参数"v", "i", "i", "i"，并应用Beartype类型检查
@_onnx_symbolic("aten::narrow")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def narrow(g: jit_utils.GraphContext, input, dim, start, length):
    # 使用_slice_helper函数进行切片操作，指定轴、起始点和结束点
    return symbolic_helper._slice_helper(
        g, input, axes=[dim], starts=[start], ends=[start + length]
    )


# 将函数标记为在ONNX中符号化处理"aten::argmax"
# 使用装饰器解析参数"v", "v", "b"，并应用Beartype类型检查
@_onnx_symbolic("aten::argmax")
@symbolic_helper.parse_args("v", "v", "b")
@_beartype.beartype
def argmax(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
):
    # 使用_argmin_argmax_helper函数进行ArgMax操作
    return symbolic_helper._argmin_argmax_helper(g, input, dim, keepdim, "ArgMax")


# 将函数标记为在ONNX中符号化处理"aten::argmin"
# 使用装饰器解析参数"v", "v", "b"，并应用Beartype类型检查
@_onnx_symbolic("aten::argmin")
@symbolic_helper.parse_args("v", "v", "b")
@_beartype.beartype
def argmin(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
):
    # 使用_argmin_argmax_helper函数进行ArgMin操作
    return symbolic_helper._argmin_argmax_helper(g, input, dim, keepdim, "ArgMin")


# 将函数标记为在ONNX中符号化处理"aten::scatter"
# 使用装饰器解析参数"v", "i", "v", "v"，并应用Beartype类型检查
@_onnx_symbolic("aten::scatter")
@symbolic_helper.parse_args("v", "i", "v", "v")
@_beartype.beartype
def scatter(g: jit_utils.GraphContext, self, dim, index, src):
    # 获取src的数据类型
    src_type = _type_utils.JitScalarType.from_value(
        src, _type_utils.JitScalarType.UNDEFINED
    )
    # 可能获取标量src
    src = symbolic_helper._maybe_get_scalar(src)
    # 如果 src 是符号助手的值
    if symbolic_helper._is_value(src):
        # 返回一个 Scatter 操作，将 src 散列到 self 上的 index 处，指定轴向为 dim
        return g.op("Scatter", self, index, src, axis_i=dim)
    else:
        # 如果 src 不是符号助手的值，则进行类型检查
        # 检查标量 "src" 是否与 self 的类型相同（PyTorch 允许标量 src 的类型与 self 不同，但 src 是张量时不允许）
        self_scalar_type = _type_utils.JitScalarType.from_value(self)
        # 如果 self 的标量类型不等于 src 的类型
        if self_scalar_type != src_type:
            # 插入一个 Cast 节点，将 src 转换为 self_scalar_type 指定的类型
            src = g.op("Cast", src, to_i=self_scalar_type.onnx_type())
        # 返回一个 Scatter 操作，将 expand_as(g, src, index) 散列到 self 上的 index 处，指定轴向为 dim
        return g.op("Scatter", self, index, expand_as(g, src, index), axis_i=dim)
# 注释：定义一个函数，用于在ONNX符号化中处理torch.scatter_add操作
@_onnx_symbolic("aten::scatter_add")
# 注释：解析函数参数，指定参数的类型和数量
@symbolic_helper.parse_args("v", "i", "v", "v")
# 注释：使用Beartype库对函数进行类型检查和修饰
@_beartype.beartype
def scatter_add(g: jit_utils.GraphContext, self, dim, index, src):
    # 注释：尝试获取self张量的标量类型
    scalar_type = symbolic_helper._try_get_scalar_type(self)
    # 注释：如果无法获取标量类型，则返回未实现的消息
    if scalar_type is None:
        return symbolic_helper._unimplemented(
            "scatter_add", "input dtype not accessible", self
        )
    # 注释：获取self张量的大小
    sizes = symbolic_helper._get_tensor_sizes(self, allow_nonstatic=False)
    # 注释：如果sizes不为空，则创建一个与sizes大小相同、类型为标量类型的零张量
    if sizes:
        to_add = g.op("Constant", value_t=torch.zeros(sizes, dtype=scalar_type.dtype()))
    else:
        # 注释：如果sizes为空，则调用zeros_like函数创建一个与self相同大小和类型的零张量
        to_add = zeros_like(g, self, scalar_type)
    # 注释：调用_scatter_helper函数，将src张量的数据根据dim和index参数添加到to_add张量中
    to_add = symbolic_helper._scatter_helper(g, to_add, dim, index, src)
    # 注释：调用add函数，将self张量和to_add张量相加并返回结果
    return add(g, self, to_add)


# 注释：定义一个函数，用于在ONNX符号化中处理torch.log2操作
@_onnx_symbolic("aten::log2")
# 注释：使用Beartype库对函数进行类型检查和修饰
@_beartype.beartype
def log2(g: jit_utils.GraphContext, self):
    # 注释：定义ln(2)的常量值
    _ln2 = 0.693147180559945309
    # 注释：计算self张量的对数并除以ln(2)，返回结果
    return g.op("Div", log(g, self), g.op("Constant", value_t=torch.tensor(_ln2)))


# 注释：定义一个函数，用于在ONNX符号化中处理torch.is_floating_point操作
@_onnx_symbolic("aten::is_floating_point")
# 注释：使用Beartype库对函数进行类型检查和修饰
@_beartype.beartype
def is_floating_point(g: jit_utils.GraphContext, self):
    # 注释：如果self张量是浮点数类型，则返回一个布尔类型的True常量张量
    if symbolic_helper._is_fp(self):
        return g.op("Constant", value_t=torch.BoolTensor([1]))
    # 注释：如果self张量不是浮点数类型，则返回一个布尔类型的False常量张量
    return g.op("Constant", value_t=torch.BoolTensor([0]))


# 注释：定义一个函数，用于在ONNX符号化中处理torch.__is__操作
@_onnx_symbolic("aten::__is_")
# 注释：使用Beartype库对函数进行类型检查和修饰
@_beartype.beartype
def __is_(g: jit_utils.GraphContext, self, other):
    # 注释：如果other张量为None，并且self张量也为None，则返回一个布尔类型的True常量张量
    if symbolic_helper._is_none(other):
        if symbolic_helper._is_none(self):
            return g.op("Constant", value_t=torch.BoolTensor([1]))
        # 注释：如果other张量为None，但是self张量不是None，则返回一个布尔类型的False常量张量
        return g.op("Constant", value_t=torch.BoolTensor([0]))
    # 注释：如果other张量不为None，则调用eq函数比较self张量和other张量，返回比较结果
    return eq(g, self, other)


# 注释：定义一个函数，用于在ONNX符号化中处理torch.__isnot__操作
@_onnx_symbolic("aten::__isnot_")
# 注释：使用wrap_logical_op_with_negation修饰函数，实现逻辑操作的否定
@wrap_logical_op_with_negation
# 注释：使用Beartype库对函数进行类型检查和修饰
@_beartype.beartype
def __isnot_(g: jit_utils.GraphContext, self, other):
    # 注释：调用__is__函数实现torch.__isnot__的逻辑
    return __is_(g, self, other)


# 注释：定义一个函数，用于在ONNX符号化中处理torch.one_hot操作
@_onnx_symbolic("aten::one_hot")
# 注释：使用Beartype库对函数进行类型检查和修饰
@_beartype.beartype
def one_hot(g: jit_utils.GraphContext, self, num_classes):
    # 注释：创建一个包含[0, 1]的长整型常量张量，用作one_hot操作的值
    values = g.op("Constant", value_t=torch.LongTensor([0, 1]))
    # 注释：由于onnxruntime对于OneHot操作有类型限制，需要根据num_classes参数的类型进行类型转换
    if _type_utils.JitScalarType.from_value(
        num_classes, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.UINT8,
        _type_utils.JitScalarType.INT8,
        _type_utils.JitScalarType.INT,
        _type_utils.JitScalarType.INT16,
    }:
        # 注释：将num_classes参数转换为int64类型以确保兼容性
        num_classes = g.op("Cast", num_classes, to_i=_C_onnx.TensorProtoDataType.INT64)
    # 注释：调用OneHot操作，根据self张量和num_classes参数进行one_hot编码，并返回结果
    return g.op("OneHot", self, num_classes, values, axis_i=-1)


# 注释：定义一个函数，用于在ONNX符号化中处理torch.gather操作
@_onnx_symbolic("aten::gather")
# 注释：解析函数参数，指定参数的类型和数量
@symbolic_helper.parse_args("v", "i", "v", "v")
# 注释：使用Beartype库对函数进行类型检查和修饰
@_beartype.beartype
def gather(g: jit_utils.GraphContext, self, dim, index, sparse_grad=False):
    # 注释：如果sparse_grad参数可能为常量且为True，则返回未实现的消息
    if symbolic_helper._maybe_get_const(sparse_grad, "i"):
        return symbolic_helper._unimplemented("gather", "sparse_grad == True", self)
    # 注释：此处是一个必要的解决方案，因为GatherElement仅在opset 11及以上版本支持，
    #       而ONNX中的Gather操作与torch.gather不同。
    scalar_type = _type_utils.JitScalarType.from_value(self)
    # 创建一个名为 'values' 的常量节点，其值为 [0, 1] 的长整型张量
    values = g.op("Constant", value_t=torch.LongTensor([0, 1]))
    # 使用 size 函数获取 self 张量在维度 dim 上的大小，返回一个常量节点
    depth = size(g, self, g.op("Constant", value_t=torch.LongTensor([dim])))
    # 创建一个 Cast 节点，将 OneHot 操作的结果转换为指定的数据类型
    index = g.op(
        "Cast",
        # 执行 OneHot 操作，生成一个独热编码的张量
        g.op("OneHot", index, depth, values, axis_i=dim),
        # 将 Cast 操作转换为指定的数据类型，使用 scalar_type 的 ONNX 类型
        to_i=scalar_type.onnx_type(),
    )
    # 创建一个 Mul 节点，对 self 张量在 dim+1 维度上进行扩展，并与 index 进行逐元素乘法
    mul = g.op("Mul", symbolic_helper._unsqueeze_helper(g, self, [dim + 1]), index)
    # 调用 _reducesum_helper 函数，对 mul 张量在 dim 维度上进行求和操作
    return symbolic_helper._reducesum_helper(g, mul, axes_i=[dim], keepdims_i=0)
# 将函数标记为接受和解析特定参数的帮助函数，使用装饰器进行修饰
@symbolic_helper.parse_args("v", "is", "i", "i")
# 使用 Beartype 装饰器对函数进行类型检查和类型注解
@_beartype.beartype
# 定义一个函数，计算输入张量的方差均值
def _var_mean(g: jit_utils.GraphContext, input, dim, correction, keepdim):
    return symbolic_helper._var_mean_helper(g, input, dim, correction, keepdim)


# 将函数标记为对应于 ONNX 符号 "aten::std" 的符号函数，使用装饰器进行修饰
@_onnx_symbolic("aten::std")
# 使用 Beartype 装饰器对函数进行类型检查和类型注解
@_beartype.beartype
# 定义一个函数，计算输入张量的标准差
def std(g: jit_utils.GraphContext, input, *args):
    # 调用 _var_mean 函数计算方差和均值
    var, _ = var_mean(g, input, *args)
    # 返回方差的平方根作为标准差
    return g.op("Sqrt", var)


# 将函数标记为对应于 ONNX 符号 "aten::var" 的符号函数，使用装饰器进行修饰
@_onnx_symbolic("aten::var")
# 使用 Beartype 装饰器对函数进行类型检查和类型注解
@_beartype.beartype
# 定义一个函数，计算输入张量的方差
def var(g: jit_utils.GraphContext, input, *args):
    # 调用 _var_mean 函数计算方差和均值
    var, _ = var_mean(g, input, *args)
    # 返回计算得到的方差
    return var


# 将函数标记为对应于 ONNX 符号 "aten::var_mean" 的符号函数，使用装饰器进行修饰
@_onnx_symbolic("aten::var_mean")
# 使用 Beartype 装饰器对函数进行类型检查和类型注解
@_beartype.beartype
# 定义一个函数，根据参数长度调用 _var_mean 函数计算方差或方差均值
def var_mean(g: jit_utils.GraphContext, input, *args):
    if len(args) == 1:
        return _var_mean(g, input, None, args[0], None)
    else:
        return _var_mean(g, input, *args)


# 将函数标记为对应于 ONNX 符号 "aten::std_mean" 的符号函数，使用装饰器进行修饰
@_onnx_symbolic("aten::std_mean")
# 使用 Beartype 装饰器对函数进行类型检查和类型注解
@_beartype.beartype
# 定义一个函数，计算输入张量的标准差和均值
def std_mean(g: jit_utils.GraphContext, input, *args):
    # 调用 var_mean 函数计算方差和均值
    var, mean = var_mean(g, input, *args)
    # 返回方差的平方根作为标准差，以及计算得到的均值
    return g.op("Sqrt", var), mean


# 将函数标记为对应于 ONNX 符号 "aten::logsumexp" 的符号函数，使用装饰器进行修饰
@_onnx_symbolic("aten::logsumexp")
# 使用 parse_args 装饰器对函数参数进行解析和类型检查
@symbolic_helper.parse_args("v", "is", "i")
# 使用 Beartype 装饰器对函数进行类型检查和类型注解
@_beartype.beartype
# 定义一个函数，计算输入张量在指定维度上的 logsumexp
def logsumexp(g: jit_utils.GraphContext, input, dim, keepdim):
    # 使用 ONNX 运算符 "ReduceLogSumExp" 对输入张量进行操作，指定轴和是否保持维度
    return g.op("ReduceLogSumExp", input, axes_i=dim, keepdims_i=keepdim)


# 将函数标记为对应于 ONNX 符号 "aten::arange" 的符号函数，使用装饰器进行修饰
@_onnx_symbolic("aten::arange")
# 使用 Beartype 装饰器对函数进行类型检查和类型注解
@_beartype.beartype
# 定义一个函数，生成指定范围内的等差数列张量
def arange(g: jit_utils.GraphContext, *args):
    # 定义一个内部函数，根据参数 dtype 获取对应的数据类型
    @_beartype.beartype
    def _get_arange_dtype(dtype):
        dtype = symbolic_helper._maybe_get_const(dtype, "i")
        return dtype

    # 定义一个内部函数，将浮点类型的步长转换为整型
    @_beartype.beartype
    def _float_step_convert(range_tensor):
        if symbolic_helper._is_fp(range_tensor):
            range_tensor = g.op(
                "Cast",
                g.op("Ceil", range_tensor),
                to_i=_type_utils.JitScalarType.INT64.onnx_type(),
            )
        return range_tensor

    if len(args) == 2 or len(args) == 5:
        if len(args) == 2:
            # 当参数个数为 2 时，执行适合的 arange 操作
            # aten::arange(Scalar end, Tensor out)
            dtype = None
        else:
            # 当参数个数为 5 时，执行适合的 arange 操作
            # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
            dtype = _get_arange_dtype(args[1])
        
        # 调用 _arange_cast_helper 函数，根据参数生成起始值、结束值、步长
        dtype, end, start, step = symbolic_helper._arange_cast_helper(
            g, end=args[0], dtype=dtype
        )
        # 将结束值升维，以便后续操作
        end = symbolic_helper._unsqueeze_helper(g, end, [0])
        # 转换浮点步长为整型
        range_tensor = _float_step_convert(end)
        # 调用 _squeeze_helper 函数，处理生成的 arange 张量
        arange_tensor = symbolic_helper._squeeze_helper(
            g, nonzero(g, ones(g, range_tensor, dtype, None, None)), [1]
        )
        # 返回类型转换后的 arange 张量
        return g.op(
            "Cast", arange_tensor, to_i=_type_utils.JitScalarType(dtype).onnx_type()
        )
    # 如果参数个数为4或7，则进入条件判断
    elif len(args) == 4 or len(args) == 7:
        if len(args) == 4:
            # 当参数个数为4时，表示调用的是 aten::arange(Scalar start, Scalar end, Scalar step, Tensor out)
            dtype = None
        else:
            # 当参数个数为7时，表示调用的是 aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
            # 获取参数 args[3] 对应的数据类型
            dtype = _get_arange_dtype(args[3])
        # 使用 symbolic_helper._arange_cast_helper 处理起始值、结束值、步长及数据类型
        dtype, end, start, step = symbolic_helper._arange_cast_helper(
            g, start=args[0], end=args[1], step=args[2], dtype=dtype
        )
        # 对 step 进行维度调整，确保符合操作要求
        step = symbolic_helper._unsqueeze_helper(g, step, [0])
        # 对 end 进行维度调整，确保符合操作要求
        end = symbolic_helper._unsqueeze_helper(g, end, [0])
        # 对 start 进行维度调整，确保符合操作要求
        start = symbolic_helper._unsqueeze_helper(g, start, [0])
        # 计算 range_tensor，这是 end 和 start 之差除以 step 的结果
        range_tensor = _float_step_convert(g.op("Div", g.op("Sub", end, start), step))
        # 计算 arange_tensor，将 range_tensor 转换为非零索引后乘以 step，并加上 start
        arange_tensor = symbolic_helper._squeeze_helper(
            g, nonzero(g, ones(g, range_tensor, None, None, None)), [1]
        )
        arange_tensor = g.op("Add", g.op("Mul", arange_tensor, step), start)
        # 将 arange_tensor 转换为指定的数据类型 dtype
        return g.op(
            "Cast", arange_tensor, to_i=_type_utils.JitScalarType(dtype).onnx_type()
        )
    elif len(args) == 6:
        # 当参数个数为6时，表示调用的是 aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        # 获取参数 args[2] 对应的数据类型
        dtype = _get_arange_dtype(args[2])
        # 使用 symbolic_helper._arange_cast_helper 处理起始值、结束值及数据类型
        dtype, end, start, step = symbolic_helper._arange_cast_helper(
            g, start=args[0], end=args[1], dtype=dtype
        )
        # 对 end 进行维度调整，确保符合操作要求
        end = symbolic_helper._unsqueeze_helper(g, end, [0])
        # 对 start 进行维度调整，确保符合操作要求
        start = symbolic_helper._unsqueeze_helper(g, start, [0])
        # 计算 range_tensor，这是 end 和 start 之差的结果
        range_tensor = _float_step_convert(g.op("Sub", end, start))
        # 计算 arange_tensor，将 range_tensor 转换为非零索引后加上 start
        arange_tensor = g.op(
            "Add",
            symbolic_helper._squeeze_helper(
                g, nonzero(g, ones(g, range_tensor, dtype, *(args[3:]))), [1]
            ),
            start,
        )
        # 将 arange_tensor 转换为指定的数据类型 dtype
        return g.op(
            "Cast", arange_tensor, to_i=_type_utils.JitScalarType(dtype).onnx_type()
        )

    # 如果参数个数不是4、6或7，则返回未实现的提示信息
    return symbolic_helper._unimplemented("aten::arange", f"with {len(args)} arguments")
# 注册为ONNX符号化操作"aten::linspace"，并使用Beartype对函数参数进行类型检查
@_onnx_symbolic("aten::linspace")
@_beartype.beartype
def linspace(
    g: jit_utils.GraphContext, start, end, steps, dtype, layout, device, pin_memory
):
    # 使用_arange_helper函数生成一个包含步长的张量
    range_tensor = symbolic_helper._arange_helper(g, steps, None)
    # 计算步长值
    step = div(
        g,
        sub(g, end, start),  # 计算(end - start)
        sub(g, steps, g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))),  # 计算(steps - 1)
    )
    # 返回生成的等间距张量
    return add(g, mul(g, range_tensor, step), start)


# 注册为ONNX符号化操作"aten::lift"，该函数在ONNX跟踪中不执行任何操作
@_onnx_symbolic("aten::lift")
@_beartype.beartype
def lift(g: jit_utils.GraphContext, self):
    # at::lift()在ONNX跟踪中是一个空操作
    return self


# 注册为ONNX符号化操作"aten::masked_fill"，实现对PyTorch张量进行掩码填充的功能
@_onnx_symbolic("aten::masked_fill")
@_beartype.beartype
def masked_fill(g: jit_utils.GraphContext, self, mask, value):
    """Implement the masked_fill functionality available for a pytorch tensor in ONNX.

    Fills elements of the input tensor with `value` where `mask` is True.
    """
    # 将mask张量转换为布尔类型
    mask = g.op("Cast", mask, to_i=_C_onnx.TensorProtoDataType.BOOL)
    # 获取value的标量值
    value = symbolic_helper._maybe_get_scalar(value)
    # 使用Where操作符根据mask值进行条件填充
    return g.op("Where", mask, symbolic_helper._if_scalar_type_as(value, self), self)


# 注册为ONNX符号化操作"aten::masked_fill_"，调用masked_fill函数实现相同的功能
@_onnx_symbolic("aten::masked_fill_")
@_beartype.beartype
def masked_fill_(g: jit_utils.GraphContext, self, mask, value):
    return masked_fill(g, self, mask, value)


# 注册为ONNX符号化操作"aten::index"，实现索引操作
@_onnx_symbolic("aten::index")
@_beartype.beartype
def index(g: jit_utils.GraphContext, self, index):
    # 如果index是打包列表，则解包
    if symbolic_helper._is_packed_list(index):
        indices = symbolic_helper._unpack_list(index)
    else:
        indices = [index]

    # 内部函数，尝试将mask转换为索引
    @_beartype.beartype
    def try_mask_to_index(index):
        if not symbolic_helper._is_none(index) and (
            _type_utils.JitScalarType.from_value(
                index, _type_utils.JitScalarType.UNDEFINED
            )
            == _type_utils.JitScalarType.UINT8
            or symbolic_helper._is_bool(index)
        ):
            # 如果ONNX版本低于9，则抛出错误
            if g.opset < 9:
                raise errors.SymbolicValueError(
                    "Exporting masked indices are only supported after ONNX opset 9.",
                    self,
                )
            # 提示警告信息，Byte类型的索引在ONNX图中仅支持1维索引
            warnings.warn(
                "Exporting aten::index operator with indices of type Byte. "
                "Only 1-D indices are supported. In any other case, "
                "this will produce an incorrect ONNX graph."
            )
            # 使用_squeeze_helper函数将非零索引压缩为1维
            index = symbolic_helper._squeeze_helper(g, nonzero(g, index), [1])
        return index

    # 对所有索引应用try_mask_to_index函数
    indices = [try_mask_to_index(idx) for idx in indices]
    # 如果索引只有一个，则使用_select_helper函数进行选择操作
    if len(indices) == 1:
        return symbolic_helper._select_helper(
            g, self, 0, indices[0], apply_reshape=False
        )


# 注册为ONNX符号化操作"aten::linalg_norm"
# 使用parse_args装饰器解析参数，实现torch.linalg.norm函数的符号化
@_onnx_symbolic("aten::linalg_norm")
@symbolic_helper.parse_args("v", "v", "is", "b", "v")
@_beartype.beartype
def linalg_norm(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    ord: torch._C.Value,
    dim: Optional[Sequence[int]],
    keepdim: bool,
    dtype: torch._C.Value,
):
    # 根据条件生成ONNX图的实现
    # 初始化变量 ord_value，设为 None
    ord_value = None
    
    # 检查是否未指定 dim
    if dim is None:
        # 如果 ord 是 None，则通过 symbolic_helper._reshape_helper 对 self 进行形状重塑为一维
        if symbolic_helper._is_none(ord):
            self = symbolic_helper._reshape_helper(g, self, [-1])
            # 将 ord 设置为一个常量张量，其值为 [2]
            ord = g.op("Constant", value_t=torch.LongTensor([2]))
        
        # 获取 self 的维度信息
        self_dim = symbolic_helper._get_tensor_rank(self)
        
        # 如果无法获取 self 的维度信息，则返回未实现的功能，提示在导出时必须知道输入的维度
        if self_dim is None:
            return symbolic_helper._unimplemented(
                "dim", "Input rank must be known at export time.", self
            )
        
        # 如果 self 的维度为 1
        if self_dim == 1:
            # 将 ord 解析为一个浮点数，并赋给 ord_value
            ord_value = symbolic_helper._parse_arg(ord, "f")
        
        # 否则，设置 dim 为 [0, 1]
        else:
            dim = [0, 1]
    
    # 如果 dim 已经指定
    else:
        # 如果 dim 的长度为 1
        if len(dim) == 1:
            # 如果 ord 是 None，则将 ord 设置为一个常量张量，其值为 [2]
            if symbolic_helper._is_none(ord):
                ord = g.op("Constant", value_t=torch.LongTensor([2]))
            
            # 将 ord 解析为一个浮点数，并赋给 ord_value
            ord_value = symbolic_helper._parse_arg(ord, "f")
    
    # 如果 ord_value 存在
    if ord_value:
        # 调用 linalg_vector_norm 函数，计算向量的范数，并返回结果
        return linalg_vector_norm(g, self, ord_value, dim, keepdim, dtype)
    
    # 否则，调用 linalg_matrix_norm 函数，计算矩阵的范数，并返回结果
    return linalg_matrix_norm(g, self, ord, dim, keepdim, dtype)
# 注册 ONNX 符号化函数 "aten::linalg_vector_norm"，对 linalg_vector_norm 函数进行装饰
# 解析函数参数 "v", "f", "is", "b", "v"
# beartype 装饰器，用于运行时类型检查
@_onnx_symbolic("aten::linalg_vector_norm")
@symbolic_helper.parse_args("v", "f", "is", "b", "v")
@_beartype.beartype
def linalg_vector_norm(
    g: jit_utils.GraphContext,      # 参数 g，表示图上下文
    self: torch._C.Value,           # 参数 self，表示 Torch 值
    ord: float,                     # 参数 ord，表示范数的阶数
    dim: Optional[Sequence[int]],   # 参数 dim，表示维度序列的可选类型
    keepdim: bool,                  # 参数 keepdim，表示是否保持维度
    dtype: torch._C.Value,          # 参数 dtype，表示 Torch 值的数据类型
):
    return symbolic_helper._linalg_vector_norm_helper(g, self, ord, dim, keepdim, dtype)

# 注册 ONNX 符号化函数 "aten::linalg_matrix_norm"，对 linalg_matrix_norm 函数进行装饰
# 解析函数参数 "v", "v", "is", "b", "v"
# beartype 装饰器，用于运行时类型检查
@_onnx_symbolic("aten::linalg_matrix_norm")
@symbolic_helper.parse_args("v", "v", "is", "b", "v")
@_beartype.beartype
def linalg_matrix_norm(
    g: jit_utils.GraphContext,      # 参数 g，表示图上下文
    self: torch._C.Value,           # 参数 self，表示 Torch 值
    ord: torch._C.Value,            # 参数 ord，表示范数的阶数
    dim: List[int],                 # 参数 dim，表示整数列表的维度
    keepdim: bool,                  # 参数 keepdim，表示是否保持维度
    dtype: torch._C.Value,          # 参数 dtype，表示 Torch 值的数据类型
):
    # 根据 https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html 的条件进行判断
    ord_value = symbolic_helper._parse_arg(ord, "s")
    if ord_value == "fro":  # 如果 ord_value 是 "fro"
        return frobenius_norm(g, self, dim, keepdim)  # 返回 Frobenius 范数的计算结果
    elif ord_value == "nuc":  # 如果 ord_value 是 "nuc"
        return symbolic_helper._unimplemented("linalg.matrix_norm", "ord==nuc", self)  # 返回未实现的提示信息
    else:
        ord_value = symbolic_helper._parse_arg(ord, "f")  # 解析 ord_value 为 float 类型
        if ord_value is None:
            return frobenius_norm(g, self, dim, keepdim)  # 返回 Frobenius 范数的计算结果
        if ord_value == 2 or ord_value == -2:
            # 对于 ord = 2/-2，由于缺少运算符用于计算奇异值，返回未实现的提示信息
            return symbolic_helper._unimplemented("linalg.matrix_norm", "ord==2", self)
        # 将 dim 向量包装以处理负 dim 值
        self_dim = symbolic_helper._get_tensor_rank(self)  # 获取张量的秩信息
        if self_dim is None:
            # 在导出时必须知道输入秩，否则返回未实现的提示信息
            return symbolic_helper._unimplemented(
                "linalg.matrix_norm", "Input rank must be known at export time.", self
            )
        # 对于 ord = 1/-1 和 ord = inf/-inf 的常见实现情况
        if dim[0] < 0:
            dim[0] += self_dim
        if dim[1] < 0:
            dim[1] += self_dim
        if ord_value == math.inf or ord_value == -math.inf:
            dim[0], dim[1] = dim[1], dim[0]
        if dim[1] > dim[0] and not keepdim:
            dim[1] -= 1
        sum = symbolic_helper._reducesum_helper(
            g, g.op("Abs", self), axes_i=[dim[0]], keepdims_i=keepdim
        )
        if ord_value > 0:
            result, indices = max(
                g,
                sum,
                dim_or_y=g.op("Constant", value_t=torch.LongTensor([dim[1]])),
                keepdim=keepdim,
            )
        else:
            result, indices = min(
                g,
                sum,
                dim_or_y=g.op("Constant", value_t=torch.LongTensor([dim[1]])),
                keepdim=keepdim,
            )
        return result

# 注册 ONNX 符号化函数 "aten::linalg_cross"，对 linalg_cross 函数进行装饰
# 解析函数参数 "v", "v", "i"
# beartype 装饰器，用于运行时类型检查
@_onnx_symbolic("aten::linalg_cross")
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def linalg_cross(g: jit_utils.GraphContext, input, other, dim=-1):
    return cross(g, input, other, dim)

# 注册 ONNX 符号化函数 "aten::frobenius_norm"，对 frobenius_norm 函数进行装饰
# 解析函数参数 "v", "is", "b"
@_onnx_symbolic("aten::frobenius_norm")
@symbolic_helper.parse_args("v", "is", "b")
# 使用装饰器进行类型检查和符号处理
@_beartype.beartype
# 计算给定图上的 Frobenius 范数
def frobenius_norm(g: jit_utils.GraphContext, self, dim=None, keepdim=False):
    # 计算 self 与自身的元素级平方
    sqr = g.op("Mul", self, self)
    # 对 sqr 沿指定维度进行求和，保持维度信息
    sumsqr = symbolic_helper._reducesum_helper(g, sqr, axes_i=dim, keepdims_i=keepdim)
    # 返回 sumsqr 的平方根，即 Frobenius 范数
    return g.op("Sqrt", sumsqr)


# 使用装饰器指定 ONNX 符号化函数，处理多项式分布抽样
@_onnx_symbolic("aten::multinomial")
# 解析参数，进行符号化处理
@symbolic_helper.parse_args("v", "i", "b", "v")
# 使用装饰器进行类型检查
@_beartype.beartype
def multinomial(
    g: jit_utils.GraphContext, input, num_samples, replacement=False, generator=None
):
    # 如果提供了 generator 参数，报未实现错误，因为不支持 generator
    if generator is not None and not symbolic_helper._is_none(generator):
        symbolic_helper._unimplemented(
            "Multinomial", "generator is not supported for multinomial", input
        )
    # 如果 replacement=False 且 num_samples > 1，报未实现错误，因为不支持这种组合
    if not replacement and num_samples > 1:
        symbolic_helper._unimplemented(
            "Multinomial",
            "replacement=False when num_samples > 1 is not supported for multinomial",
            input,
        )

    # 对 input 取对数
    log_input = log(g, input)
    # 返回 Multinomial 操作的 ONNX 表示
    return g.op(
        "Multinomial",
        log_input,
        dtype_i=_C_onnx.TensorProtoDataType.INT64,
        sample_size_i=num_samples,
    )


# 使用装饰器指定 ONNX 符号化函数，处理批次矩阵乘加运算
@_onnx_symbolic("aten::baddbmm")
# 使用装饰器进行类型检查
@_beartype.beartype
def baddbmm(g: jit_utils.GraphContext, self, batch1, batch2, beta, alpha):
    # 确定 self 的标量类型
    scalar_type = _type_utils.JitScalarType.from_value(self)
    # 计算 batch1 与 batch2 的矩阵乘法
    batch_mul = matmul(g, batch1, batch2)
    # 将 alpha 转换为与 scalar_type 对应的 ONNX 类型后，与 batch_mul 相乘
    mul_a = mul(
        g,
        batch_mul,
        g.op("Cast", alpha, to_i=scalar_type.onnx_type()),
    )
    # 将 self 乘以 beta 后，与 mul_a 相加
    mul_b = mul(
        g,
        self,
        g.op("Cast", beta, to_i=scalar_type.onnx_type()),
    )
    # 返回最终的加法结果
    return add(g, mul_a, mul_b)


# 使用装饰器指定 ONNX 符号化函数，处理多维网格生成
@_onnx_symbolic("aten::meshgrid")
# 解析参数，进行符号化处理
@symbolic_helper.parse_args("v", "s")
# 使用装饰器进行类型检查
@_beartype.beartype
def meshgrid(g: jit_utils.GraphContext, tensor_list, indexing: Optional[str] = None):
    # 如果 indexing 未指定，默认为 "ij"
    if indexing is None:
        indexing = "ij"
    # 如果 indexing 不在 {"ij", "xy"} 中，报错
    elif indexing not in {"ij", "xy"}:
        raise errors.SymbolicValueError(
            f"Unsupported indexing: {indexing}", tensor_list
        )
    # 解包 tensor_list
    unpacked_tensor_list = symbolic_helper._unpack_list(tensor_list)
    # 如果 indexing 是 "xy"，调整前两个张量的顺序
    if indexing == "xy":
        unpacked_tensor_list[:2] = unpacked_tensor_list[1::-1]
    # 对解包后的张量列表进行重塑，使其维度为 [-1]
    tensors = [
        symbolic_helper._reshape_helper(
            g, t, g.op("Constant", value_t=torch.LongTensor([-1]))
        )
        for t in unpacked_tensor_list
    ]
    # 获取每个张量的形状
    tensors_shape = [g.op("Shape", t) for t in tensors]
    # 将所有张量的形状在指定轴上进行拼接
    out_shape = g.op("Concat", *tensors_shape, axis_i=0)
    # 对每个张量进行扩展，使其形状与 out_shape 一致
    out = []
    for i, t in enumerate(tensors):
        shape_i = [g.op("Constant", value_t=torch.ones(1, dtype=torch.int64))] * len(
            tensors
        )
        shape_i[i] = tensors_shape[i]
        t_reshaped = _reshape_from_tensor(g, t, g.op("Concat", *shape_i, axis_i=0))
        out.append(g.op("Expand", t_reshaped, out_shape))
    # 如果 indexing 是 "xy"，交换输出列表中的前两个张量
    if indexing == "xy":
        out[0], out[1] = out[1], out[0]
    # 返回由 out 组成的列表构造函数的 ONNX 表示
    return g.op("prim::ListConstruct", *out)


# 使用装饰器指定 ONNX 符号化函数，处理取余运算
@_onnx_symbolic("aten::remainder")
# 使用装饰器进行类型检查
@_beartype.beartype
def remainder(g: jit_utils.GraphContext, input, other):
    # 使用 _floor_divide 函数计算 input 除以 other 的地板除法
    div = _floor_divide(g, input, other)
    # 使用图 g 中的 op 方法创建一个乘法操作节点，将 div 和 other 作为参数
    quo = g.op("Mul", div, other)
    
    # 使用图 g 中的 op 方法创建一个减法操作节点，将 input 和 quo 作为参数
    return g.op("Sub", input, quo)
# 使用装饰器将函数注册为ONNX符号函数，处理aten::gelu操作
@_onnx_symbolic("aten::gelu")
# 解析参数v和s，作为装饰器的一部分
@symbolic_helper.parse_args("v", "s")
# 使用beartype装饰器，确保函数输入参数类型正确
@_beartype.beartype
# 定义gelu函数，处理GELU激活函数
def gelu(g: jit_utils.GraphContext, self: torch._C.Value, approximate: str = "none"):
    # 如果approximate参数为"tanh"，执行以下操作
    if approximate == "tanh":
        # 定义常量参数
        kBeta = math.sqrt(2 / math.pi)
        kKappa = 0.044715

        # 创建torch张量，用于数学运算
        beta = torch.tensor(kBeta, dtype=torch.double)
        kappa = torch.tensor(kKappa, dtype=torch.double)
        one = torch.tensor(1.0, dtype=torch.double)
        half = torch.tensor(0.5, dtype=torch.double)

        # 计算self的立方，并进行相关数学运算
        self_cube = mul(g, self, mul(g, self, self))
        inner = mul(g, beta, add(g, self, mul(g, kappa, self_cube)))

        # 返回GELU激活函数的计算结果
        return mul(g, half, mul(g, self, add(g, one, g.op("Tanh", inner))))
    else:
        # 如果approximate参数不是"tanh"，执行以下操作

        # 定义常量_sqrt2并计算Erf函数
        _sqrt2 = 1.4142135623730951
        erf = g.op("Erf", g.op("Div", self, torch.tensor(_sqrt2, dtype=torch.double)))
        erf_plusone = add(
            g, erf, g.op("Constant", value_t=torch.tensor(1, dtype=torch.double))
        )

        # 返回计算结果
        return mul(
            g,
            mul(g, self, erf_plusone),
            g.op("Constant", value_t=torch.tensor(0.5, dtype=torch.double)),
        )


# 使用装饰器将函数注册为ONNX符号函数，处理aten::group_norm操作
@_onnx_symbolic("aten::group_norm")
# 处理量化参数，并作为装饰器的一部分
@symbolic_helper.quantized_args(True, False, False, False)
# 解析参数v、i、v、v、f、i，作为装饰器的一部分
@symbolic_helper.parse_args("v", "i", "v", "v", "f", "i")
# 使用beartype装饰器，确保函数输入参数类型正确
@_beartype.beartype
# 定义group_norm函数，处理分组归一化操作
def group_norm(
    g: jit_utils.GraphContext, input, num_groups, weight, bias, eps, cudnn_enabled
):
    # 获取输入张量的通道大小
    channel_size = symbolic_helper._get_tensor_dim_size(input, 1)
    if channel_size is not None:
        assert channel_size % num_groups == 0

    # 获取输入张量的秩
    input_rank = symbolic_helper._get_tensor_rank(input)
    if input_rank is None:
        # 如果无法获取输入张量的秩，返回未实现错误信息
        return symbolic_helper._unimplemented("group_norm", "unknown input rank", input)

    # 定义变换后的张量形状
    # 0表示保持维度值不变
    shape = [0, num_groups, -1]
    input_reshaped = symbolic_helper._reshape_helper(
        g, input, g.op("Constant", value_t=torch.LongTensor(shape))
    )

    # 创建weight和bias常量张量
    weight_ = g.op(
        "Constant",
        value_t=torch.tensor(
            [1.0] * num_groups,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        ),
    )
    bias_ = g.op(
        "Constant",
        value_t=torch.tensor(
            [0.0] * num_groups,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        ),
    )

    # 执行实例归一化操作，并重新调整形状
    norm_reshaped = g.op(
        "InstanceNormalization", input_reshaped, weight_, bias_, epsilon_f=eps
    )
    norm = symbolic_helper._reshape_helper(g, norm_reshaped, g.op("Shape", input))

    # 如果weight为None或其节点必须为None，则创建默认权重张量
    if weight is None or weight.node().mustBeNone():
        weight_value = torch.tensor(
            [1.0], dtype=_type_utils.JitScalarType.from_value(input).dtype()
        )
        weight = g.op("Constant", value_t=weight_value)
    # 如果偏置（bias）为None或其节点（node）必须为None，则执行以下操作
    if bias is None or bias.node().mustBeNone():
        # 根据输入值的数据类型，创建一个包含单个0.0的张量作为偏置值
        bias_value = torch.tensor(
            [0.0], dtype=_type_utils.JitScalarType.from_value(input).dtype()
        )
        # 使用Graph对象（g）的操作（op）方法创建一个常量节点，将上述偏置值作为其值
        bias = g.op("Constant", value_t=bias_value)
    
    # Norm（规范化）的形状为[N, C, *]，因此我们将权重（weight）和偏置（bias）重新调整为形状为[C, *]
    # 计算需要展开的轴列表，从1到输入维度排名（input_rank - 1）
    axes = list(range(1, input_rank - 1))
    
    # 返回如下计算结果：
    return add(
        g,
        # 计算norm与通过辅助函数_unsqueeze_helper展开权重（weight）的乘积
        mul(g, norm, symbolic_helper._unsqueeze_helper(g, weight, axes)),
        # 通过辅助函数_unsqueeze_helper展开偏置（bias）
        symbolic_helper._unsqueeze_helper(g, bias, axes),
    )
# 定义权重归一化函数，用于 ONNX 符号化
@_onnx_symbolic("aten::_weight_norm")
# 解析参数，参数类型为向量、向量、整数
@symbolic_helper.parse_args("v", "v", "i")
# 使用 beartype 进行类型检查
@_beartype.beartype
def _weight_norm(g: jit_utils.GraphContext, weight_v, weight_g, dim):
    # 获取权重向量的秩
    rank = symbolic_helper._get_tensor_rank(weight_v)
    if rank is not None:
        # 计算 L2 范数的权重归一化
        # 如果 dim 为 None，则表示在所有维度上计算
        # torch 的 weight_norm 模块在 dim 为 None 时将其设置为 -1
        # 这与负轴访问维度的逻辑冲突
        # TODO: 可能需要在 torch group_norm 模块中进行修复
        axes = list(range(rank))
        if dim is not None:
            if dim < -1:
                dim += rank
            if dim != -1:
                axes.remove(dim)
        # 计算权重向量的 L2 范数
        norm_v = norm(g, weight_v, 2, axes, 1)
        div = g.op("Div", weight_v, norm_v)
        return g.op("Mul", div, weight_g)
    # 抛出符号值错误
    raise errors.SymbolicValueError(
        "Unsupported: ONNX export of _weight_norm for tensor of unknown rank.",
        weight_v,
    )


# 定义 dim 函数，实现 ONNX 中 pytorch 张量的 dim 功能
@_onnx_symbolic("aten::dim")
@_beartype.beartype
def dim(g: jit_utils.GraphContext, self):
    # 在此 opset 中，ONNX 不直接支持 dim，因此可以使用 2 个操作来获取信息
    shape = g.op("Shape", self)
    return g.op("Size", shape)


# 定义 __contains__ 函数，实现 ONNX 中的 __contains__ 功能
@_onnx_symbolic("aten::__contains_")
@_beartype.beartype
def __contains_(g: jit_utils.GraphContext, self, element):
    # 解包列表
    unpacked_list = symbolic_helper._unpack_list(self)
    # 如果列表中的所有元素和 element 都是常量
    if all(
        symbolic_helper._is_constant(x) for x in unpacked_list
    ) and symbolic_helper._is_constant(element):
        return g.op(
            "Constant",
            value_t=torch.tensor(
                symbolic_helper._node_get(element.node(), "value")
                in (symbolic_helper._node_get(x.node(), "value") for x in unpacked_list)
            ),
        )
    # 抛出符号值错误
    raise errors.SymbolicValueError(
        "Unsupported: ONNX export of __contains__ for non-constant list or element.",
        self,
    )


# 定义 __getitem__ 函数，实现 ONNX 中的 __getitem__ 功能
@_onnx_symbolic("aten::__getitem_")
@_beartype.beartype
def __getitem_(g: jit_utils.GraphContext, self, i):
    return select(g, self, g.op("Constant", value_t=torch.tensor([0])), i)


# 定义 item 函数，实现 ONNX 中的 item 功能
@_onnx_symbolic("aten::item")
@_beartype.beartype
def item(g: jit_utils.GraphContext, self):
    return self


# 定义 take 函数，实现 ONNX 中的 take 功能
@_onnx_symbolic("aten::take")
@_beartype.beartype
def take(g: jit_utils.GraphContext, self, index):
    # 将 self 展平
    self_flattened = symbolic_helper._reshape_helper(
        g, self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    )
    # 使用 index_select 进行索引选择
    out = index_select(g, self_flattened, 0, index)
    out = reshape_as(g, out, index)
    return out


# 定义 _kl_div_log_target_impl 函数，实现 KL 散度计算
@_beartype.beartype
def _kl_div_log_target_impl(g: jit_utils.GraphContext, input, target):
    diff_ = sub(g, target, input)
    exp_ = exp(g, target)
    output = mul(g, exp_, diff_)
    return output


# 定义 _kl_div_non_log_target_impl 函数
@_beartype.beartype
def _kl_div_non_log_target_impl(g: jit_utils.GraphContext, input, target):
    # 计算对数 log(g, target)
    log_ = log(g, target)
    # 计算差值 diff_ = g - log_
    diff_ = sub(g, log_, input)
    # 计算输出位置 output_pos = g * target * diff_
    output_pos = mul(g, target, diff_)
    # 生成与 output_pos 相同形状的全零张量 zeros_
    zeros_ = zeros_like(g, output_pos)
    # 创建一个布尔掩码，标识 target 大于零的位置
    mask_ = gt(g, target, g.op("Constant", value_t=torch.tensor(0)))
    # 根据 mask_ 条件选择 output_pos 或 zeros_，形成最终的输出 output
    output = where(g, mask_, output_pos, zeros_)
    # 返回最终的输出张量
    return output
# 使用装饰器将函数注册为ONNX操作符"aten::kl_div"
# 使用装饰器解析函数参数，期望参数类型为"v", "v", "i", "b"
# 使用装饰器对函数进行类型检查和类型注解
def kl_div(g: jit_utils.GraphContext, input, target, reduction, log_target):
    # 如果log_target为真，调用基于对数目标实现的kl_div操作
    if log_target:
        output = _kl_div_log_target_impl(g, input, target)
    else:
        # 否则，调用基于非对数目标实现的kl_div操作
        output = _kl_div_non_log_target_impl(g, input, target)

    # 根据reduction的值进行不同的输出处理
    if reduction == 0:
        return output
    elif reduction == 1:
        # 返回输出的平均值，保持维度不变
        return g.op("ReduceMean", output, keepdims_i=0)
    elif reduction == 2:
        # 返回输出的和，保持维度不变
        return symbolic_helper._reducesum_helper(g, output, keepdims_i=0)
    else:
        # 如果reduction的值不是0、1、2中的任何一个，返回不支持的错误信息
        return symbolic_helper._onnx_unsupported(
            "kl_div with reduction other than none, mean, or sum.", input
        )


# 使用装饰器将函数注册为ONNX操作符"aten::mse_loss"
# 使用装饰器解析函数参数，期望参数类型为"v", "v", "i"
# 使用装饰器对函数进行类型检查和类型注解
def mse_loss(g: jit_utils.GraphContext, input, target, reduction):
    # 计算均方误差损失，使用mul、sub函数分别计算input与target之间的差值的平方
    output = mul(g, sub(g, input, target), sub(g, input, target))
    
    # 根据reduction的值进行不同的输出处理
    if reduction == 0:
        return output
    elif reduction == 1:
        # 返回输出的平均值，保持维度不变
        return g.op("ReduceMean", output, keepdims_i=0)
    elif reduction == 2:
        # 返回输出的和，保持维度不变
        return symbolic_helper._reducesum_helper(g, output, keepdims_i=0)
    else:
        # 如果reduction的值不是0、1、2中的任何一个，返回不支持的错误信息
        return symbolic_helper._onnx_unsupported(
            "mse_loss with reduction other than none, mean, or sum.", input
        )


# 使用装饰器将函数注册为ONNX操作符"aten::as_strided"
# 使用装饰器设置量化参数为True
# 使用装饰器解析函数参数，期望参数类型为"v", "v", "is", "i"
# 使用装饰器对函数进行类型检查和类型注解
def as_strided(g: jit_utils.GraphContext, self, sizes, strides, offset=None):
    # 获取sizes的常数值，如果sizes不是一个值而是一个变量，则返回None
    sizes = symbolic_helper._maybe_get_const(sizes, "is")
    # 计算strides的长度，即rank的值
    rank = len(strides)
    # 将self转换为一维数组，用于后续的gather操作
    self_1d = symbolic_helper._reshape_helper(
        g, self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    )
    ind: Optional[torch.Tensor]
    # 如果sizes不是一个值，而是一个常量数组，则进行以下操作
    if not symbolic_helper._is_value(sizes):
        # 初始化索引为0的Tensor
        ind = torch.tensor([0], dtype=torch.long)
        # 遍历sizes和strides，计算ind的索引值
        for i, (size, stride) in enumerate(zip(sizes, strides)):
            r_size = [1] * rank
            r_size[i] = -1
            ind = ind + torch.arange(size).view(r_size) * stride
        # 如果存在offset，则将offset添加到ind中
        if offset:
            ind = ind + offset
        # 使用Gather操作，根据ind从self_1d中收集数据
        return g.op("Gather", self_1d, g.op("Constant", value_t=ind))
    # 否则，初始化索引变量为 None
    else:
        ind = None
        # 遍历 strides 列表的每个元素及其索引值
        for i, stride in enumerate(strides):
            # 创建一个大小为 rank 的全 1 列表 r_size，并将第 i 个位置设为 -1
            r_size = [1] * rank
            r_size[i] = -1
            # 使用 g.op 方法创建一个表示常数的节点，其值为 torch.tensor([0]) 和 torch.tensor(i)
            size = select(
                g,
                sizes,
                g.op("Constant", value_t=torch.tensor([0])),
                g.op("Constant", value_t=torch.tensor(i)),
            )
            # 调用 symbolic_helper._reshape_helper 方法执行重塑操作，返回一个临时索引 tmp_ind
            tmp_ind = symbolic_helper._reshape_helper(
                g,
                arange(g, size, 4, None, None, None),
                g.op("Constant", value_t=torch.tensor(r_size)),
            )
            # 创建一个表示乘法操作的节点，计算 tmp_ind 乘以 stride
            tmp_ind = g.op(
                "Mul", tmp_ind, g.op("Constant", value_t=torch.tensor([stride]))
            )
            # 如果 ind 为 None，则将 tmp_ind 赋值给 ind
            if ind is None:
                ind = tmp_ind
            # 否则，创建一个表示加法操作的节点，将 ind 和 tmp_ind 相加，并更新 ind
            else:
                ind = g.op("Add", ind, tmp_ind)
        # 如果 offset 为真，则创建一个表示加法操作的节点，将 ind 和 offset 相加
        if offset:
            ind = g.op("Add", ind, g.op("Constant", torch.tensor([offset])))
        # 返回一个表示 Gather 操作的节点，使用 self_1d 和 ind 作为参数
        return g.op("Gather", self_1d, ind)
# 注册一个 ONNX 符号化函数，将 aten::__derive_index 映射到 __derive_index 函数
# 通过装饰器 @_onnx_symbolic 实现
@_onnx_symbolic("aten::__derive_index")
# 使用装饰器 @_beartype 对函数进行类型检查和类型注解
@_beartype.beartype
# 定义 __derive_index 函数，接受图上下文 g，以及 index、start 和 step 三个参数
def __derive_index(g: jit_utils.GraphContext, index, start, step):
    # 使用 ONNX 操作 "Add" 实现 start + index * step
    return g.op("Add", start, g.op("Mul", index, step))


# 注册一个 ONNX 符号化函数，将 aten::__range_length 映射到 __range_length 函数
@_onnx_symbolic("aten::__range_length")
# 使用装饰器 @_beartype 对函数进行类型检查和类型注解
@_beartype.beartype
# 定义 __range_length 函数，接受图上下文 g，以及 lo、hi 和 step 三个参数
def __range_length(g: jit_utils.GraphContext, lo, hi, step):
    # 计算 hi - lo
    sub = g.op("Sub", hi, lo)
    # 计算 (hi - lo) / step 的上取整结果
    div = g.op("Ceil", true_divide(g, sub, step))
    # 将结果转换为 INT64 类型
    return g.op("Cast", div, to_i=_C_onnx.TensorProtoDataType.INT64)


# 注册一个 ONNX 符号化函数，将 aten::linear 映射到 linear 函数
@_onnx_symbolic("aten::linear")
# 使用装饰器 @_beartype 对函数进行类型检查和类型注解
@_beartype.beartype
# 定义 linear 函数，接受图上下文 g，以及 input、weight 和 bias 三个参数
def linear(g: jit_utils.GraphContext, input, weight, bias):
    # 获取输入张量 input 的维度
    rank = symbolic_helper._get_tensor_rank(input)
    # 转换 weight 张量到图 g 所在的设备上
    weight = t(g, weight)
    
    # 如果 input 的维度是 2，并且 bias 不为 None
    if rank == 2 and not bias.node().mustBeNone():
        # 创建值为 1 的常量张量 alpha 和 beta
        alpha = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        beta = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        # 使用 addmm 操作计算 bias + input @ weight
        output = addmm(g, bias, input, weight, alpha, beta)
    else:
        # 使用 matmul 操作计算 input @ weight
        output = matmul(g, input, weight)
        # 如果 bias 不为 None，将 output 和 bias 相加
        if not bias.node().mustBeNone():
            output = add(g, bias, output)

    return output


# 注册一个 ONNX 符号化函数，将 aten::hann_window 映射到 hann_window 函数
@_onnx_symbolic("aten::hann_window")
# 使用装饰器 @_beartype 对函数进行类型检查和类型注解
@_beartype.beartype
# 定义 hann_window 函数，接受图上下文 g 和多个可选参数
def hann_window(
    g: jit_utils.GraphContext,
    window_length,
    periodic=True,
    dtype: Optional[int] = None,
    layout=None,
    device=None,
    pin_memory=None,
    requires_grad=False,
):
    # 如果未指定 dtype，则使用默认的浮点数类型
    if dtype is None:
        dtype_ = torch.get_default_dtype()
        if not dtype_ or not dtype_.is_floating_point:
            dtype_ = torch.float
        scalar_type = _type_utils.JitScalarType.from_dtype(dtype_)
    else:
        # 否则，使用指定的 dtype
        scalar_type = _type_utils.JitScalarType(dtype)

    # 创建一个以 window_length 为长度、步长为 4 的数组 n_array
    n_array = arange(g, window_length, 4, None, None, None)
    # 将 n_array 转换为 FLOAT 类型
    output = g.op("Cast", n_array, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    # 将 output 中的每个元素乘以 π
    output = mul(
        g, g.op("Constant", value_t=torch.tensor(math.pi, dtype=torch.float)), output
    )

    # 如果 periodic 参数为 False，则将 window_length 减去 1
    if periodic is False:
        window_length = sub(
            g, window_length, g.op("Constant", value_t=torch.tensor(1, dtype=torch.int))
        )
    # 将 output 中的每个元素除以 window_length
    output = div(g, output, window_length)
    # 计算 sin(output) 的平方，并将结果转换为 scalar_type 类型
    output = g.op(
        "Cast",
        square(g, sin(g, output)),
        to_i=scalar_type.onnx_type(),
    )

    return output


# 注册一个 ONNX 符号化函数，将 aten::mv 映射到 mv 函数
@_onnx_symbolic("aten::mv")
# 使用装饰器 @_beartype 对函数进行类型检查和类型注解
@_beartype.beartype
# 定义 mv 函数，接受图上下文 g 和两个张量参数 self 和 vec
def mv(g: jit_utils.GraphContext, self, vec):
    # 使用 matmul 操作计算 self @ vec
    return matmul(g, self, vec)


# 注册一个 ONNX 符号化函数，将 aten::dot 映射到 dot 函数
@_onnx_symbolic("aten::dot")
# 使用装饰器 @_beartype 对函数进行类型检查和类型注解
@_beartype.beartype
# 定义 dot 函数，接受图上下文 g 和两个张量参数 self 和 other
def dot(g: jit_utils.GraphContext, self, other):
    # 使用 matmul 操作计算 self @ other
    return matmul(g, self, other)


# 注册一个 ONNX 符号化函数，将 aten::movedim 映射到 movedim 函数
@_onnx_symbolic("aten::movedim")
# 使用装饰器 @_beartype 对函数进行类型检查和类型注解
@_beartype.beartype
# 定义 movedim 函数，接受图上下文 g 和三个参数 self、source 和 destination
def movedim(g: jit_utils.GraphContext, self, source, destination):
    # 从source和destination张量中展平后创建视图，以便进行后续操作
    source = source.view(-1)
    destination = destination.view(-1)

    # 断言source和destination张量的大小相同，确保操作的一致性
    assert source.size() == destination.size()

    # 检查source和destination张量是否完全相等，如果是则返回当前对象自身
    if (source == destination).all():
        return self

    # 获取当前对象self的张量秩（维度数量），确保获取成功
    self_rank = symbolic_helper._get_tensor_rank(self)
    assert self_rank is not None

    # 创建一个表示张量维度排列顺序的初始排列列表
    perm = list(range(self_rank))

    # 复制perm列表以备后续操作使用
    src_dims = perm.copy()
    dst_dims = perm.copy()

    # 遍历source和destination张量展平后的元素，并更新perm、src_dims和dst_dims列表
    for src, dst in zip(source.tolist(), destination.tolist()):
        perm[dst] = src
        src_dims[src] = -1
        dst_dims[dst] = -1

    # 从src_dims和dst_dims列表中删除所有标记为-1的元素，得到最终的源和目标维度列表
    src_dims = [dim for dim in src_dims if dim != -1]
    dst_dims = [dim for dim in dst_dims if dim != -1]

    # 使用最终的源和目标维度列表更新perm列表，完成最终的维度排列顺序
    for src, dst in zip(src_dims, dst_dims):
        perm[dst] = src

    # 使用ONNX操作"Transpose"创建一个新的转置操作节点，并返回结果
    return g.op("Transpose", self, perm_i=perm)
# 使用装饰器定义 ONNX 符号化函数 "aten::fill"
@_onnx_symbolic("aten::fill")
# 使用装饰器解析参数 "v", "v"
@symbolic_helper.parse_args("v", "v")
# 应用 beartype 装饰器，类型检查
@_beartype.beartype
# 定义 fill 函数，接受 GraphContext g, self, value 作为参数
def fill(g: jit_utils.GraphContext, self, value):
    # 从 value 中获取标量类型，默认为 FLOAT
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.FLOAT
    )
    # 调用 full_like 函数，返回填充后的张量
    return full_like(g, self, value, scalar_type)


# 使用装饰器定义 ONNX 符号化函数 "aten::index_add"
@_onnx_symbolic("aten::index_add")
# 应用 beartype 装饰器，类型检查
@_beartype.beartype
# 定义 index_add 函数，接受 GraphContext g, self, dim, index, other, alpha（可选）作为参数
def index_add(g: jit_utils.GraphContext, self, dim, index, other, alpha=None):
    # 发出警告，ONNX 导出不支持在 'index' 字段中存在重复值
    warnings.warn(
        "Warning: ONNX export does not support duplicated values in 'index' field, "
        + "this will cause the ONNX model to be incorrect."
    )

    # ONNX 不支持 "alpha" 参数，与 aten index_add 不同
    # 参考：https://github.com/pytorch/pytorch/pull/65993#issuecomment-953151102
    if alpha and symbolic_helper._scalar(symbolic_helper._maybe_get_scalar(alpha)) != 1:
        # 如果 alpha 不为 1，返回未实现的错误
        return symbolic_helper._unimplemented("index_add", "alpha != 1", self)

    # 获取常量维度 dim 的值，如果为 None，则抛出错误
    dim = symbolic_helper._maybe_get_const(dim, "i")
    if dim is None:
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting 'index_add_()' function with "
            "unknown 'dim' value.",
            self,
        )

    # 获取 self 和 other 张量的维度秩
    self_dim_rank = symbolic_helper._get_tensor_rank(self)
    other_dim_rank = symbolic_helper._get_tensor_rank(other)

    # 如果 self 或 other 的维度秩未知，则抛出错误
    if self_dim_rank is None or other_dim_rank is None:
        raise errors.SymbolicValueError(
            "ONNX export does NOT support exporting 'index_add_()' function while "
            "the rank of self tensor or tensor to be added is unknown.",
            self,
        )

    # 如果 other 的维度秩不等于 self 的维度秩，则进行维度扩展
    if other_dim_rank != self_dim_rank:
        delta = self_dim_rank - other_dim_rank
        for i in range(delta):
            other = symbolic_helper._unsqueeze_helper(
                g, other, [symbolic_helper._get_tensor_rank(other)]
            )

    # 获取 other 在维度 dim 上的大小和 self 在维度 dim 上的大小
    other_dim_size = symbolic_helper._get_tensor_dim_size(other, dim)
    self_dim_size = symbolic_helper._get_tensor_dim_size(self, dim)

    # 如果 other 在维度 dim 上的大小大于 self 在维度 dim 上的大小，则抛出错误
    if (other_dim_size is not None) and (self_dim_size is not None):
        if other_dim_size > self_dim_size:
            raise errors.SymbolicValueError(
                "ONNX export does not support exporting 'index_add_()' function with "
                "duplicated values in 'index' parameter yet.",
                self,
            )

    # 构造新的形状，除了维度 dim 外，其它维度与 self 相同，维度 dim 的大小为 1
    new_shape_axes = list(range(self_dim_rank))
    new_shape_starts = [0 for i in range(self_dim_rank)]
    new_shape_ends = [sys.maxsize if (i != dim) else 1 for i in range(self_dim_rank)]

    # 对 self 进行切片操作，获取新的形状
    new_shape = symbolic_helper._slice_helper(
        g, self, axes=new_shape_axes, starts=new_shape_starts, ends=new_shape_ends
    )

    # 将 other 根据新的形状进行扩展
    other = expand_as(g, other, new_shape)

    # 对维度 dim 进行展开操作
    for i in range(dim):
        index = symbolic_helper._unsqueeze_helper(g, index, [0])
    # 对于给定的维度范围进行迭代，从 self_dim_rank - dim - 1 开始到 0 结束
    for i in range(self_dim_rank - dim - 1):
        # 调用符号助手的 _unsqueeze_helper 函数，将 g 张量在指定维度上进行unsqueeze操作，
        # 使用当前的 index 张量作为输入，同时传递一个包含 index 张量秩的列表参数
        index = symbolic_helper._unsqueeze_helper(
            g, index, [symbolic_helper._get_tensor_rank(index)]
        )

    # 返回调用 scatter_add 函数的结果，该函数用于在指定维度上进行张量相加的scatter操作，
    # 使用 self 张量、dim 维度、g 张量、index 张量、other 张量作为输入
    return scatter_add(g, self, dim, expand_as(g, index, other), other)
# 为 `roll` 函数添加符号化装饰器和类型检查装饰器
@_onnx_symbolic("aten::roll")
@symbolic_helper.parse_args("v", "is", "is")
@_beartype.beartype
def roll(g: jit_utils.GraphContext, self, shifts, dims):
    # 断言 `shifts` 和 `dims` 的长度相同
    assert len(shifts) == len(dims)

    # 将 `self` 赋值给 `result`，用于迭代过程中的累积结果
    result = self
    # 遍历 `shifts` 列表的长度
    for i in range(len(shifts)):
        # 创建一个空列表 `shapes` 用于存储切片后的形状
        shapes = []
        # 使用 `_slice_helper` 函数对 `result` 进行切片，从 `-shifts[i]` 到最大整数
        shape = symbolic_helper._slice_helper(
            g, result, axes=[dims[i]], starts=[-shifts[i]], ends=[sys.maxsize]
        )
        # 将切片后的形状添加到 `shapes` 列表中
        shapes.append(shape)
        # 再次使用 `_slice_helper` 函数对 `result` 进行切片，从 `0` 到 `-shifts[i]`
        shape = symbolic_helper._slice_helper(
            g, result, axes=[dims[i]], starts=[0], ends=[-shifts[i]]
        )
        # 将切片后的形状添加到 `shapes` 列表中
        shapes.append(shape)
        # 使用 `g.op` 函数进行拼接操作，将 `shapes` 列表中的形状按照 `dims[i]` 轴进行连接
        result = g.op("Concat", *shapes, axis_i=dims[i])

    # 返回拼接后的结果 `result`
    return result


# 为 `cross` 函数添加符号化装饰器和类型检查装饰器
@_onnx_symbolic("aten::cross")
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype
def cross(g: jit_utils.GraphContext, input, other, dim=None):
    # 调用 `_get_dim_for_cross` 函数获取用于交叉乘积的维度 `dim`
    dim = symbolic_helper._get_dim_for_cross(input, dim)
    
    # 对 `input` 和 `other` 进行第一次滚动操作，分别在 `dim` 维度上滚动
    roll_x_1 = roll(g, input, [2], [dim])
    roll_y_1 = roll(g, other, [1], [dim])
    
    # 对 `input` 和 `other` 进行第二次滚动操作，分别在 `dim` 维度上滚动
    roll_x_2 = roll(g, input, [1], [dim])
    roll_y_2 = roll(g, other, [2], [dim])
    
    # 计算交叉乘积，通过 `mul` 函数和 `sub` 函数实现
    # 先计算 (roll_x_1 * roll_y_1) - (roll_x_2 * roll_y_2)
    return sub(g, mul(g, roll_x_1, roll_y_1), mul(g, roll_x_2, roll_y_2))


# 为 `cdist` 函数添加符号化装饰器和类型检查装饰器
@_onnx_symbolic("aten::cdist")
@_beartype.beartype
def cdist(
    g: jit_utils.GraphContext,
    x1,
    x2,
    p=2.0,
    compute_mode="use_mm_for_euclid_dist_if_necessary",
):
    # 获取 `x1` 的张量秩
    rank = symbolic_helper._get_tensor_rank(x1)
    # 断言张量秩不为空
    assert rank is not None
    
    # 使用 `_unsqueeze_helper` 函数在 `x1` 的倒数第二个维度上添加一个维度
    broadcasted_x1 = symbolic_helper._unsqueeze_helper(g, x1, [rank - 1])
    # 使用 `_unsqueeze_helper` 函数在 `x2` 的倒数第三个维度上添加一个维度
    broadcasted_x2 = symbolic_helper._unsqueeze_helper(g, x2, [rank - 2])
    
    # 调用 `pairwise_distance` 函数计算广播后的 `x1` 和 `x2` 之间的距离
    return pairwise_distance(
        g, broadcasted_x1, broadcasted_x2, p, eps=1e-06, keepdim=False
    )


# 为 `lerp` 函数添加符号化装饰器和类型检查装饰器
@_onnx_symbolic("aten::lerp")
@_beartype.beartype
def lerp(g: jit_utils.GraphContext, self, end, weight):
    # 计算 `end - self`，得到差值 `diff`
    diff = g.op("Sub", end, self)
    # 使用 PyTorch 的图（graph）操作构建一个条件表达式
    return where(
        # 在图 g 中使用 Less 操作比较 weight 和 0.5 的值
        g,
        g.op("Less", weight, g.op("Constant", value_t=torch.tensor(0.5))),
        # 如果条件为真，则返回 self 加上 weight 乘以 diff 的结果
        g.op("Add", self, g.op("Mul", weight, diff)),
        # 如果条件为假，则返回 end 减去 diff 乘以 (1.0 - weight) 的结果
        g.op(
            "Sub",
            end,
            g.op(
                "Mul",
                diff,
                g.op("Sub", g.op("Constant", value_t=torch.tensor(1.0)), weight),
            ),
        ),
    )
# 注册一个 ONNX 符号化函数，处理 "aten::broadcast_tensors" 操作符
# 这个函数使用装饰器 @_beartype.beartype 进行类型检查
@_onnx_symbolic("aten::broadcast_tensors")
@_beartype.beartype
def broadcast_tensors(g: jit_utils.GraphContext, self):
    # 解包输入列表中的所有张量
    all_tensors = symbolic_helper._unpack_list(self)
    # 创建一个形状与第一个张量相同的零张量
    t_with_final_shape = zeros_like(g, all_tensors[0])

    # 循环遍历所有张量，并使用 add 函数对它们进行广播操作
    for t in all_tensors:
        t_with_final_shape = add(g, t_with_final_shape, t)

    # 对所有输入张量进行扩展，使它们与最终形状相匹配
    t_list = [expand_as(g, t, t_with_final_shape) for t in all_tensors]
    # 使用 "prim::ListConstruct" 操作符将所有扩展后的张量组合成一个列表
    return g.op("prim::ListConstruct", *t_list)


# 注册一个 ONNX 符号化函数，处理 "aten::is_pinned" 操作符
@_onnx_symbolic("aten::is_pinned")
def is_pinned(g: jit_utils.GraphContext, self, device=None):
    # 在 ONNX 中未被使用，直接返回 None
    return None


# 注册一个 ONNX 符号化函数，处理 "prim::ConstantSplit" 操作符
@_onnx_symbolic("prim::ConstantSplit")
@_beartype.beartype
def prim_constant_split(g: jit_utils.GraphContext, self, split_size, dim):
    # 获取指定维度上的张量大小
    size = symbolic_helper._get_tensor_dim_size(self, dim)
    # 如果大小未知，返回未实现的提示信息
    if size is None:
        return symbolic_helper._unimplemented(
            "prim::ConstantSplit", "unknown dimension size", self
        )
    # 计算分割点列表
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    # 使用 "Split" 操作符在指定维度上对张量进行分割
    return g.op("Split", self, split_i=splits, axis_i=dim, outputs=len(splits))


# 注册一个 ONNX 符号化函数，处理 "prim::ConstantChunk" 操作符
@_onnx_symbolic("prim::ConstantChunk")
@_beartype.beartype
def prim_constant_chunk(g: jit_utils.GraphContext, self, chunks, dim):
    # 获取指定维度上的张量大小
    dim_size = symbolic_helper._get_tensor_dim_size(self, dim)
    # 如果大小未知，返回未实现的提示信息
    if dim_size is None:
        return symbolic_helper._unimplemented(
            "prim::ConstantChunk", "unknown dimension size", self
        )
    # 计算每个分块的大小
    split_size = (dim_size + chunks - 1) // chunks
    # 调用 prim_constant_split 函数进行分块操作
    return prim_constant_split(g, self, split_size, dim)


# 注册一个 ONNX 符号化函数，处理 "prim::shape" 操作符
@_onnx_symbolic("prim::shape")
@_beartype.beartype
def prim_shape(g: jit_utils.GraphContext, self):
    # 使用 "Shape" 操作符获取张量的形状
    return g.op("Shape", self)


# 注册一个 ONNX 符号化函数，处理 "prim::max" 操作符
@_onnx_symbolic("prim::max")
@_beartype.beartype
def prim_max(g: jit_utils.GraphContext, self, other):
    # 使用带有可选浮点数转换的 _op_with_optional_float_cast 函数执行 "Max" 操作
    return symbolic_helper._op_with_optional_float_cast(
        g, "Max", self, other, opset_before=12
    )


# 注册一个 ONNX 符号化函数，处理 "prim::min" 操作符
@_onnx_symbolic("prim::min")
@_beartype.beartype
def prim_min(g: jit_utils.GraphContext, self, other=None):
    # 如果未提供第二个参数，且输入是 packed list，则将其与常量 0 合并后执行最小值操作
    if not other:
        if symbolic_helper._is_packed_list(self):
            self = stack(g, self, g.op("Constant", value_t=torch.tensor([0])))
        return min(g, self)
    # 否则，执行两个张量之间的最小值操作
    return min(g, self, other)


# 注册一个 ONNX 符号化函数，处理 "prim::data" 操作符
@_onnx_symbolic("prim::data")
@_beartype.beartype
def prim_data(g: jit_utils.GraphContext, self):
    # 返回输入自身，即数据本身
    return self


# 注册一个 ONNX 符号化函数，处理 "prim::layout" 操作符
@_onnx_symbolic("prim::layout")
def prim_layout(g: jit_utils.GraphContext, self):
    # 始终返回 'torch.strided'，因为 JIT 'TensorType' 不支持其他布局类型
    # 布局类型定义在 'c10/core/Layout.h' 中的 Layout 类中
    return 'torch.strided'
    # 使用 PyTorch 的操作符创建一个常量张量，值为0
    return g.op("Constant", value_t=torch.tensor(0))
# 使用装饰器 @_onnx_symbolic 标记函数处理 ONNX 符号
# 使用装饰器 @_beartype.beartype 对函数进行类型检查
@_onnx_symbolic("prim::ListConstruct")
@_beartype.beartype
def prim_list_construct(g: jit_utils.GraphContext, *inputs, **kwargs):
    # 返回 None，表示不执行特定的 ONNX 符号操作
    return None


# 使用装饰器 @_onnx_symbolic 标记函数处理 ONNX 符号
# 使用装饰器 @_beartype.beartype 对函数进行类型检查
@_onnx_symbolic("prim::ListUnpack")
@_beartype.beartype
def prim_list_unpack(
    g: jit_utils.GraphContext, *inputs, **kwargs
) -> Optional[List[_C.Value]]:
    # 如果输入只有一个并且其类型是 "prim::ListConstruct"，则取消前一个节点并返回其输入
    # TODO(justinchuby): 使用助手模块中的公共方法
    if len(inputs) == 1 and inputs[0].node().kind() == "prim::ListConstruct":
        return symbolic_helper._unpack_list(inputs[0])

    # 返回 None，表示不执行特定的 ONNX 符号操作
    return None


# 使用装饰器 @_onnx_symbolic 标记函数处理 ONNX 符号
# 使用装饰器 @_beartype.beartype 对函数进行类型检查
def prim_tuple_construct(g: jit_utils.GraphContext, *inputs, **kwargs):
    # 返回 None，表示不执行特定的 ONNX 符号操作
    return None


# 使用装饰器 @_onnx_symbolic 标记函数处理 ONNX 符号
# 使用装饰器 @_beartype.beartype 对函数进行类型检查
def prim_uninitialized(g: jit_utils.GraphContext, *inputs, **kwargs):
    # 返回 None，表示不执行特定的 ONNX 符号操作
    return None


# 使用装饰器 @_onnx_symbolic 标记函数处理 ONNX 符号
# 使用装饰器 @_beartype.beartype 对函数进行类型检查
def prim_unchecked_cast(g: jit_utils.GraphContext, self):
    # 直接返回 self，用于将 x 强制转换为 Tensor 类型
    return self


# 使用装饰器 @_onnx_symbolic 标记函数处理 ONNX 符号
# 使用装饰器 @_beartype.beartype 对函数进行类型检查
def prim_dtype(g: jit_utils.GraphContext, self):
    # 获取 scalar_type，如果获取失败，则默认为 FLOAT 类型
    scalar_type = symbolic_helper._try_get_scalar_type(self)
    if scalar_type is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    # 返回一个表示 torch dtype 的整数常量
    return g.op("Constant", value_t=torch.tensor(scalar_type))


# 使用装饰器 @_onnx_symbolic 标记函数处理 ONNX 符号
# 使用装饰器 @_beartype.beartype 对函数进行类型检查
def prim_tolist(g: jit_utils.GraphContext, input, dim_val, elem_ty_val):
    """tolist 目前仅支持 1D 输入张量。

    dim_val 和 elem_ty_val 分别表示需要匹配的输入张量的维度和类型注解。
    """
    # 获取 dim 的值，如果其大于 1，则返回未实现的错误信息
    dim = symbolic_helper._maybe_get_const(dim_val, "i")
    if dim > 1:
        return symbolic_helper._unimplemented("prim::tolist", "dim_val > 1", input)
    # 返回输入本身，表示不执行特定的 ONNX 符号操作
    return input


# -----------------------------------------------------------------------------
# 需要额外上下文的符号函数
# -----------------------------------------------------------------------------
# 使用装饰器 @_onnx_symbolic 标记函数处理 ONNX 符号
# 使用装饰器 @_beartype.beartype 对函数进行类型检查
def prim_device(g: jit_utils.GraphContext, *inputs, **kwargs) -> None:
    # 获取输出类型
    output_type = g.original_node.output().type()
    # 如果输出类型是 _C.DeviceObjType 类型，则返回 None
    if isinstance(output_type, _C.DeviceObjType):
        return None

    # 否则返回未实现的错误信息，指出输出类型应该是 'DeviceObjType' 而不是当前类型
    return symbolic_helper._unimplemented(
        "prim::device",
        f"output type should be 'DeviceObjType', not '{output_type.kind()}'",
        g.original_node.output(),
    )


# 使用装饰器 @_onnx_symbolic 标记函数处理 ONNX 符号
# 使用装饰器 @_beartype.beartype 对函数进行类型检查
def prim_loop(g: jit_utils.GraphContext, *inputs, **attrs) -> List[_C.Value]:
    # 获取原始节点、环境、环境中的值以及参数字典
    node = g.original_node
    env = g.env
    values_in_env = g.values_in_env
    params_dict = g.params_dict
    # 返回空列表，表示不执行特定的 ONNX 符号操作
    return []
    # 从全局变量中获取操作符导出类型
    operator_export_type = GLOBALS.operator_export_type
    # 从全局变量中获取导出 ONNX 的 opset 版本
    opset_version = GLOBALS.export_onnx_opset_version

    # 获取当前节点的所有子块作为元组
    old_blocks = tuple(node.blocks())
    # 使用 jit_utils.add_op_with_blocks 函数向图 g 中添加带块的操作 "Loop"
    # inputs 为输入参数列表，outputs=node.outputsSize() 为输出数量，n_blocks=len(old_blocks) 为块数量
    new_op_outputs, new_block_contexts, new_node = jit_utils.add_op_with_blocks(
        g, "Loop", *inputs, outputs=node.outputsSize(), n_blocks=len(old_blocks)
    )

    # 遍历旧块列表和新块上下文列表
    for old_block, new_block_context in zip(old_blocks, new_block_contexts):
        # 将输入元数据复制到子块中
        #
        #   prim::Loop(iter, cond, input_1, ..., input_n)
        #     block0(iter, input_1, ..., input_n)
        #
        # 对于 `Loop` 节点，复制 `iter`, `input_1`, ..., `input_n` 的元数据。
        for i, b_in in enumerate(old_block.inputs()):
            if i == 0 and i < len(inputs):
                b_in.setType(inputs[i].type())
            # 对于可选的块输入，在循环体内可能在 None 和非 None 之间切换，
            # 因此如果循环输入不是可选的，则块输入可能仍然需要是可选的。
            if (
                i > 0
                and (i + 1) < len(inputs)
                and not isinstance(b_in.type(), _C.OptionalType)
            ):
                b_in.setType(inputs[i + 1].type())
        
        # 将旧块转换为新块上下文中的 ONNX 块
        torch._C._jit_pass_onnx_block(
            old_block,
            new_block_context.block,
            operator_export_type,
            env,
            values_in_env,
            False,
        )

    # 修正 ONNX 控制流节点的输出
    fixed_outputs = torch._C._jit_pass_fixup_onnx_controlflow_node(
        new_node, opset_version
    )

    # 如果启用了全局的 ONNX 形状推断
    if GLOBALS.onnx_shape_inference:
        # 对 Loop 节点进行形状类型推断
        torch._C._jit_pass_onnx_node_shape_type_inference(
            new_node, params_dict, opset_version
        )

    # 返回修正后的输出
    return fixed_outputs
# 声明函数prim_if，用于处理ONNX图中的prim::If操作
@_onnx_symbolic("prim::If")
@_beartype.beartype
# 函数接收一个GraphContext对象g作为参数，返回一个值列表(List[_C.Value])
def prim_if(g: jit_utils.GraphContext, *inputs, **attrs) -> List[_C.Value]:
    # 从GraphContext对象g中获取原始节点n、块block、环境env、values_in_env、参数字典params_dict
    n = g.original_node
    block = g.block
    env = g.env
    values_in_env = g.values_in_env
    params_dict = g.params_dict

    # 获取全局变量中的运算符导出类型和导出ONNX操作集版本号
    operator_export_type = GLOBALS.operator_export_type
    opset_version = GLOBALS.export_onnx_opset_version

    # 判断输入的第一个参数是否是静态if条件（即是否是"onnx::Constant"类型）
    static_if = inputs[0].node().kind() == "onnx::Constant"
    if static_if:
        # 如果是静态if条件，则折叠静态if
        #
        # Torch IR示例图
        # graph(%embedding_matrix.1 : Float(10, 15, strides=[15, 1], requires_grad=0, device=cpu),
        #    %input.1 : Long(6, strides=[1], requires_grad=0, device=cpu), ...
        # %65 : Bool(requires_grad=0, device=cpu) = prim::Constant[value={0}]()
        # %21 : Long(device=cpu) = aten::eq(%20, %64)
        # %22 : Long(device=cpu) = prim::If(%21)
        #     block0():
        #     %23 : Long(device=cpu) = aten::is_floating_point(%input.1)
        #     -> (%23)
        #     block1():
        #     -> (%65)
        # %input.53 : Tensor, %weight : Tensor = prim::If(%22)
        #     block0():
        #     -> (%embedding_matrix.1, %input.1)
        #     block1():
        #     -> (%input.1, %embedding_matrix.1)
        # %26 : int[] = aten::size(%input.53)
        #
        # 转换为ONNX图示例
        # %10 : Bool(device=cpu) = onnx::Constant[value={0}]()
        # %14 : Bool(device=cpu) = onnx::Equal(%13, %8)
        # %15 : Bool(requires_grad=0, device=cpu) = onnx::Constant[value={0}]()
        # %16 : Long(1, strides=[1], device=cpu) = onnx::Shape(%input.1)
        input_flag = symbolic_helper._node_get(inputs[0].node(), "value").tolist()
        const_value = (
            all(input_flag) if isinstance(input_flag, list) else bool(input_flag)
        )
        # 根据静态if条件的值选择执行的块索引
        block_idx = 0 if const_value else 1
        current_b = list(n.blocks())[block_idx]
        # 在ONNX块中进行符号转换
        env = torch._C._jit_pass_onnx_block(
            current_b,
            block,
            operator_export_type,
            env,
            values_in_env,
            True,
        )
        # 获取原始节点n的输出列表和当前块current_b的输出列表
        if_output_list = list(n.outputs())
        current_b_list = list(current_b.outputs())

        # 存储最终的块列表
        final_b_list = []
        # 遍历if_output_list中的每个索引
        for idx in range(len(if_output_list)):
            # 检查当前块的输出是否在环境env中，如果不在则引发SymbolicValueError异常
            if current_b_list[idx] not in env:
                raise errors.SymbolicValueError(
                    f"The sub block ATen output {current_b_list[idx]} is not in env.",
                    current_b_list[idx],
                )  # type:ignore[operator]
            # 将环境env中的ONNX块添加到最终块列表中
            onnx_b = env[current_b_list[idx]]
            final_b_list.append(onnx_b)
        # 返回最终块列表作为函数的结果
        return final_b_list
    else:
        # 获取当前节点的所有块
        old_blocks = tuple(n.blocks())
        # 添加带有块的操作到图中，返回新的输出、块上下文和新节点
        new_op_outputs, new_block_contexts, new_node = jit_utils.add_op_with_blocks(
            g, "If", *inputs, outputs=n.outputsSize(), n_blocks=len(old_blocks)
        )

        # 遍历旧块和新块上下文，转换为ONNX块并导出
        for old_block, new_block_context in zip(old_blocks, new_block_contexts):
            torch._C._jit_pass_onnx_block(
                old_block,
                new_block_context.block,
                operator_export_type,
                env,
                values_in_env,
                False,
            )
        
        # 修正ONNX控制流节点，确保控制流正确
        fixed_outputs = torch._C._jit_pass_fixup_onnx_controlflow_node(
            new_node, opset_version
        )
        
        # 如果启用了全局ONNX形状推断，运行形状类型推断
        if GLOBALS.onnx_shape_inference:
            torch._C._jit_pass_onnx_node_shape_type_inference(
                new_node, params_dict, opset_version
            )
        
        # 返回修正后的输出
        return fixed_outputs
# 用于处理 ONNX 符号化的函数装饰器，指定操作为 prim::Constant
@_onnx_symbolic("prim::Constant")
# 应用 Beartype 装饰器，确保函数参数类型正确
@_beartype.beartype
def prim_constant(g: jit_utils.GraphContext, *inputs, **attrs):
    # 获取原始节点
    node = g.original_node

    # 如果节点必须为 None，则返回 None
    if node.mustBeNone():
        return None
    
    # 如果节点输出类型为 _C.DeviceObjType，返回 None，保持设备类型不变以支持 eq() 操作
    if isinstance(node.output().type(), _C.DeviceObjType):
        return None
    
    # 如果节点值类型为 "t"，返回 Constant 操作节点，值为节点的 "value"
    if node.kindOf("value") == "t":
        return g.op("Constant", value_t=symbolic_helper._node_get(node, "value"))
    
    # 如果节点值类型为 "s"，返回 Constant 操作节点，值为节点的 "value"
    if node.kindOf("value") == "s":
        return g.op("Constant", value_s=symbolic_helper._node_get(node, "value"))
    
    # 如果节点输出类型为整数列表或浮点数列表，返回 Constant 操作节点，值为节点的 "value" 转换为 torch.tensor
    if node.output().type().isSubtypeOf(_C.ListType.ofInts()) or node.output().type().isSubtypeOf(_C.ListType.ofFloats()):
        return g.op("Constant", value_t=torch.tensor(symbolic_helper._node_get(node, "value")))
    
    # 如果节点输出类型为字符串列表，返回由字符串值创建的 Constant 操作节点列表，并使用 prim::ListConstruct 包装
    if node.output().type().isSubtypeOf(_C.ListType.ofStrings()):
        str_constants = [
            g.op("Constant", value_s=s)
            for s in symbolic_helper._node_get(node, "value")
        ]
        return g.op("prim::ListConstruct", *str_constants)
    
    # 抛出错误，表示不支持的 prim::Constant 类型，提供错误信息和节点输出
    raise errors.SymbolicValueError(
        f"Unsupported prim::Constant kind: '{node.kindOf('value')}'. "
        f"Please send a bug report at {_constants.PYTORCH_GITHUB_ISSUES_URL}.",
        node.output(),
    )


# 用于处理 ONNX 符号化的函数装饰器，指定操作为 prim::type
@_onnx_symbolic("prim::type")
# 应用 Beartype 装饰器，确保函数参数类型正确
@_beartype.beartype
def prim_type(g: jit_utils.GraphContext, device_value: _C.Value, *args, **kwargs):
    # 如果设备值节点类型为 prim::device
    if device_value.node().kind() == "prim::device":
        # 从节点输入获取设备类型
        device = jit_utils.get_device_from_value(device_value.node().input())
        # 如果设备类型不为 None，则返回 Constant 操作节点，值为设备类型字符串表示
        if device is not None:
            return g.op("Constant", value_s=str(device))
    
    # 调用未实现函数处理异常情况，提供未实现的信息和相关设备值
    return symbolic_helper._unimplemented(
        "prim::type",
        "Device type cannot be statically determined.",
        device_value,
    )


# 用于处理 ONNX 符号化的函数装饰器，指定操作为 onnx::Placeholder
@_onnx_symbolic("onnx::Placeholder")
# 应用 Beartype 装饰器，确保函数参数类型正确
@_beartype.beartype
def onnx_placeholder(g: jit_utils.GraphContext, *inputs, **attrs):
    # 获取原始节点、块和环境信息
    node = g.original_node
    block = g.block
    env = g.env
    values_in_env = g.values_in_env
    
    # 调用 torch._C._jit_onnx_convert_pattern_from_subblock 处理占位符转换为 ONNX 模式
    return torch._C._jit_onnx_convert_pattern_from_subblock(
        block, node, env, values_in_env
    )


# 用于处理 ONNX 符号化的函数装饰器，指定操作为 aten::resolve_conj 和 aten::resolve_neg
@_onnx_symbolic("aten::resolve_conj")
@_onnx_symbolic("aten::resolve_neg")
# 应用 Beartype 装饰器，确保函数参数类型正确
@_beartype.beartype
def noop_complex_operators(g: jit_utils.GraphContext, input: _C.Value):
    # ONNX 不直接支持操作处理实部和虚部，但某些 Torch API（例如 .tolist()）在处理实数时执行复杂操作，
    # 这会导致缺少复杂数操作符而失败
    
    # `aten::resolve_conj` 和 `aten::resolve_neg` 可以安全地实现为无操作
    return input


# 用于处理 ONNX 符号化的函数装饰器，指定操作为 aten::_conj 和 aten::conj_physical
@_onnx_symbolic("aten::_conj")
@_onnx_symbolic("aten::conj_physical")
# 应用 Beartype 装饰器，确保函数参数类型正确
@_beartype.beartype
def unsupported_complex_operators(g: jit_utils.GraphContext, input: _C.Value):
    # 这个函数还没有实现，但是应该用于处理不支持的复杂数操作
    # ONNX 不直接支持操作实数/虚数部分
    # 然而，一些 torch API（例如 .tolist()）在输入为实数时使用复杂操作，导致复杂数操作缺失而失败

    # 当输入是复杂数时，`aten::_conj` 和 `aten::conj_physical` 会引发异常
    if symbolic_helper.is_complex_value(input):
        # FIXME(justinchuby): 报告正在执行的符号操作的正确名称
        return symbolic_helper._onnx_unsupported(
            "aten::_conj, aten::conj_physical",
            input,
        )

    # 对于只有实数的情况，可以安全地将它们实现为无操作
    return noop_complex_operators(g, input)
# 使用装饰器将函数注册为ONNX符号化函数，处理aten::logit操作
# 使用beartype进行函数参数的类型检查和注解
@_onnx_symbolic("aten::logit")
@_beartype.beartype
def logit(g: jit_utils.GraphContext, self: torch._C.Value, eps: torch._C.Value):
    # 创建一个常量节点，值为1.0
    one = g.op("Constant", value_t=torch.tensor(1.0))

    # 如果eps不为None，则进行以下操作
    if not symbolic_helper._is_none(eps):
        # 将eps强制转换为self的类型
        eps = g.op(
            "Cast", eps, to_i=_type_utils.JitScalarType.from_value(self).onnx_type()
        )
        # 计算1 - eps
        one_sub_eps = g.op("Sub", one, eps)
        # 比较self和1 - eps的大小
        self_less_equal_one_sub_eps = g.op("Greater", one_sub_eps, self)
        # 如果self <= 1 - eps，则选择self，否则选择1 - eps
        temporary_self = g.op("Where", self_less_equal_one_sub_eps, self, one_sub_eps)

        # 比较temporary_self和eps的大小
        temporary_self_less_eps = g.op("Less", temporary_self, eps)
        # 如果temporary_self < eps，则选择eps，否则选择temporary_self
        z = g.op("Where", temporary_self_less_eps, eps, temporary_self)
    else:
        # 如果eps为None，则z等于self
        z = self

    # 计算1 - z
    sub = g.op("Sub", one, z)
    # 计算z / (1 - z)
    div = g.op("Div", z, sub)
    # 计算log(div)
    return g.op("Log", div)
```