# `.\pytorch\torch\_refs\__init__.py`

```
# mypy: allow-untyped-defs
# 导入必要的内置模块和标准库模块
import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings

# 导入抽象基类 Iterable
from collections.abc import Iterable
# 导入枚举类型 Enum
from enum import Enum
# 导入偏函数 partial，函数组合 reduce，单分派装饰器 singledispatch，装饰器 wraps
from functools import partial, reduce, singledispatch, wraps
# 导入类型提示相关模块
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union

# 导入 PyTorch 库
import torch

# 导入 PyTorch 内部模块 prims 和 prims_common
import torch._prims as prims
import torch._prims_common as utils
# 导入 sym_float 和 sym_int 两种类型
from torch import sym_float, sym_int
# 从 prims_common 模块导入各种类型和函数
from torch._prims_common import (
    BoolLike,
    DeviceLikeType,
    Dim,
    DimsSequenceType,
    DimsType,
    dtype_to_type,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    FloatLike,
    FloatWithoutSymFloat,
    IntLike,
    is_weakly_lesser_type,
    Number,
    NumberType,
    RealNumberType,
    REDUCTION_OUTPUT_TYPE_KIND,
    ShapeType,
    StrideType,
    TensorLike,
    TensorLikeType,
    TensorOrNumberLikeType,
    TensorSequenceType,
)
# 从 prims_common 的 wrappers 模块导入一些函数
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    _safe_copy_out,
    elementwise_type_promotion_wrapper,
    elementwise_unary_scalar_wrapper,
    out_wrapper,
)

# 实验性模块，包含现有 PyTorch 操作的原型 Python 引用
# 定义 __all__ 列表，列出此模块的公开接口
__all__ = [
    #
    # Elementwise Unary References
    #
    "abs",  # 绝对值
    "acos",  # 反余弦
    "acosh",  # 反双曲余弦
    "asinh",  # 反双曲正弦
    "asin",  # 反正弦
    "atan",  # 反正切
    "atanh",  # 反双曲正切
    "bitwise_not",  # 按位取反
    # "cbrt",  # 无对应的 torch 操作
    "ceil",  # 向上取整
    "conj_physical",  # 物理共轭
    "cos",  # 余弦
    "cosh",  # 双曲余弦
    "count_nonzero",  # 计算非零元素数量
    "deg2rad",  # 度数转弧度
    "digamma",  # Digamma 函数
    "erf",  # 误差函数
    "erfinv",  # 误差函数的反函数
    "erfc",  # 余误差函数
    "exp",  # 指数函数
    "expm1",  # exp(x) - 1
    "exponential",  # 指数分布
    "exp2",  # 2 的指数函数
    "fill",  # 填充张量
    "fill_",  # 填充张量（原位）
    "floor",  # 向下取整
    "frac",  # 小数部分
    "geometric",  # 几何分布
    "index_add",  # 索引相加
    "index_copy",  # 索引复制
    "index_copy_",  # 索引复制（原位）
    "index_select",  # 索引选择
    "index_fill",  # 索引填充
    "index_fill_",  # 索引填充（原位）
    "isfinite",  # 是否为有限数
    "isinf",  # 是否为无穷数
    "isposinf",  # 是否为正无穷
    "isneginf",  # 是否为负无穷
    "isnan",  # 是否为 NaN
    "isreal",  # 是否为实数
    "i0",  # 修正 Bessel 函数第一类
    "lerp",  # 线性插值
    "lgamma",  # Gamma 函数的自然对数的绝对值
    "log",  # 自然对数
    "log1p",  # log(1 + x)
    "log2",  # 以 2 为底的对数
    "log10",  # 以 10 为底的对数
    "log_normal",  # 对数正态分布
    "log_softmax",  # 对数 Softmax
    "mvlgamma",  # 多元 Gamma 函数的对数的绝对值
    "norm",  # 范数
    "normal",  # 正态分布
    "nan_to_num",  # NaN 替换为数值
    "neg",  # 负数
    "positive",  # 正数
    "rad2deg",  # 弧度转度数
    "reciprocal",  # 倒数
    "round",  # 四舍五入（TODO: 模型 kwargs）
    "sigmoid",  # Sigmoid 函数
    "sgn",  # 符号函数
    "sign",  # 符号位
    "signbit",  # 符号位是否为 1
    "sin",  # 正弦
    "sinc",  # 同步函数
    "sinh",  # 双曲正弦
    "softmax",  # Softmax 函数
    "sqrt",  # 平方根
    "square",  # 平方
    "tan",  # 正切
    "tanh",  # 双曲正切
    "trace",  # 矩阵的迹
    "trunc",  # 截断
    #
    # Elementwise Binary References
    #
    "add",  # 加法
    "atan2",  # 反正切函数 atan(y, x)
    "bitwise_and",  # 按位与
    "bitwise_left_shift",  # 按位左移
    "bitwise_or",  # 按位或
    "bitwise_right_shift",  # 按位右移
    "bitwise_xor",  # 按位异或
    "clamp_min",  # 最小值截断
    "clamp_max",  # 最大值截断
    "copysign",  # 复制符号
    "div",  # 除法
    "eq",  # 相等比较
    "float_power",  # 浮点数的幂
    "floor_divide",  # 整数除法
    "fmax",  # 最大值
    "fmin",  # 最小值
    "fmod",  # 浮点数取模
    "gcd",  # 最大公约数
    "ge",  # 大于等于比较
    "gt",  # 大于比较
    "heaviside",  # 海维赛德阶跃函数
    "hypot",  # 欧几里得距离
    "igamma",  # 下不完全 Gamma 函数
    "igammac",  # 上不完全 Gamma 函数补
    "imag",  # 虚部
    "isclose",  # 是否接近
    "lcm",  # 最小公倍数
    # 'ldexp',  # 左移指数
    "le",  # 小于等于比较
    "logaddexp",  # 对数和指数的和
    "logaddexp2",  # 对数和 2 的指数的和
    "logical_and",  # 逻辑与
    "logical_not",  # 逻辑非
    "logical_or",  # 逻辑或
    "logical_xor",  # 逻辑异或
    "logsumexp",  # 对数
    # 'max', # implement with reductions
    "maximum",
    # 'min', # implement with reductions
    "minimum",
    "mul",
    "ne",
    "nextafter",
    # 'polar',  # abs, cos, sin
    "pow",
    "real",
    "rpow",
    "remainder",
    "rsub",
    "rtruediv",
    "rfloordiv",
    "sub",
    "true_divide",
    "trunc_divide",
    "xlogy",
    #
    # Elementwise Ternary References
    #
    "addcdiv",
    "addcmul",
    "clamp",
    #
    # Conditional references
    #
    "masked_fill",
    "masked_fill_",
    "where",
    #
    # Data conversion and movement references
    #
    "clone",
    "copy_to",  # TODO: add OpInfo (or implement .to)
    "item",
    "to",
    #
    # Reduction ops
    #
    "all",
    "amax",
    "amin",
    "any",
    "cumsum",
    "cumprod",
    "mean",
    "dot",
    "vdot",
    "std",
    "std_mean",
    "sum",
    "sum_to_size",
    "prod",
    "var",
    "var_mean",
    #
    # Linear algebra ops
    #
    "addr",
    #
    # View & Shape Ops
    #
    "alias",
    "alias_copy",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "as_strided",
    "as_strided_copy",
    "as_strided_scatter",
    "block_diag",
    "broadcast_shapes",
    "broadcast_tensors",
    "broadcast_to",
    "cat",
    "chunk",
    "column_stack",
    "conj",
    "constant_pad_nd",
    "contiguous",
    "diag_embed",
    "diag",
    "diagonal",
    "diagonal_copy",
    "diagonal_scatter",
    "dsplit",
    "dstack",
    "expand",
    "expand_as",
    "flatten",
    "flip",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "meshgrid",
    "movedim",
    "narrow",
    "narrow_copy",
    "native_group_norm",
    "native_layer_norm",
    "permute",
    "ravel",
    "repeat",
    "reshape",
    "reshape_as",
    "roll",
    "rot90",
    "rsqrt",
    "stack",
    "swap_axes",  # alias for transpose
    "squeeze",
    "t",
    "T",
    "take_along_dim",
    "tensor_split",
    "transpose",
    "unfold",
    "unfold_copy",
    "unsqueeze",
    "view",
    "view_as",
    "vsplit",
    "vstack",
    "view_as_complex",
    "unflatten",
    "unbind",
    "triu",
    "tril",
    "triu_indices",
    "tril_indices",
    #
    # Tensor Creation
    #
    "arange",
    "cauchy",
    "empty",
    "empty_like",
    "empty_permuted",
    "empty_strided",
    "eye",
    "full",
    "full_like",
    "linspace",
    "logspace",
    "new_empty",
    "new_empty_strided",
    "new_full",
    "new_ones",
    "new_zeros",
    "ones",
    "ones_like",
    "randn",
    "scalar_tensor",
    "zero",
    "zeros",
    "zeros_like",
    #
    # Test-related functions
    #
    "allclose",
    "equal",
    #
    # Statistical operations
    #
    "bucketize",
    #
    # Misc
    #
    "is_complex",
    "renorm",
    "stft",
    "istft",
#`
# 导入必要的库和模块
Tensor = torch.Tensor
DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]
aten = torch._ops.ops.aten

# 提示信息，说明该文件中公共方法的文档字符串在 torch/_torch_docs.py 文件中

def is_noncontiguous_supported(device):
    # 如果设备不为空且设备类型为 "hpu"，返回 False；否则返回 True
    if device is not None and device.type == "hpu":
        return False
    return True

def handle_noncontiguous_outputs(input_tlist, output):
    # 初始化设备为 None
    device = None
    from torch._subclasses.fake_tensor import FakeTensor

    # 遍历输入列表中的每个 tensor，查找是否为 FakeTensor 实例，并获取其伪设备
    for t in input_tlist:
        if isinstance(t, FakeTensor):
            device = t.fake_device
            break

    # 如果设备不支持非连续数据，执行 output 的 contiguous 操作
    if not is_noncontiguous_supported(device):
        output = output.contiguous()

    return output

def _broadcast_shapes(*_shapes):
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 处理输入形状，确保其为元组形式，并过滤掉 None 的形状
    shapes = tuple(
        (x,) if isinstance(x, IntLike) else x
        for x in filter(lambda x: x is not None, _shapes)
    )

    # 如果没有输入，直接返回 None
    if len(shapes) == 0:
        return None

    # 类型检查，确保所有形状都是序列
    for shape in shapes:
        assert isinstance(shape, Sequence)

    # 初始化公共形状，长度为输入形状中最长的维度数，每个维度初始值为 1
    common_shape = [
        1,
    ] * reduce(max, (len(shape) for shape in shapes))
    for arg_idx, shape in enumerate(shapes):
        for idx in range(-1, -1 - len(shape), -1):
            # 判断当前维度是否为 1，且符合 guard_size_oblivious 条件
            if guard_size_oblivious(common_shape[idx] == 1):
                if shape[idx] < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[idx] = shape[idx]
            elif guard_size_oblivious(shape[idx] != 1):
                if common_shape[idx] != shape[idx]:
                    raise RuntimeError(
                        f"Attempting to broadcast a dimension of length {shape[idx]} at {idx}! "
                        f"Mismatching argument at index {arg_idx} had {shape}; but expected shape "
                        f"should be broadcastable to {common_shape}"
                    )

    return common_shape

def _maybe_broadcast(*args, preserve_cpu_scalar_tensors=True):
    # 计算所有输入张量的公共形状
    common_shape = _broadcast_shapes(
        *(t.shape if isinstance(t, TensorLike) else None for t in args)
    )

    def __maybe_broadcast(x, shape):
        # 处理 None 的情况
        if x is None:
            return None
        elif isinstance(x, Number):
            return x
        elif isinstance(x, TensorLike):
            # 如果保留 CPU 标量张量并且是 CPU 标量张量，则直接返回
            if preserve_cpu_scalar_tensors and utils.is_cpu_scalar_tensor(x):
                return x

            # 如果张量的形状与公共形状不一致，则扩展张量到公共形状
            if not utils.same_shape(x.shape, common_shape):
                return x.expand(common_shape)

            return x
        else:
            raise RuntimeError(
                "Unexpected type when broadcasting: " + str(type(x)) + "!"
            )

    # 返回处理后的张量元组
    return tuple(__maybe_broadcast(x, common_shape) for x in args)

# 该部分应该在导入工具函数之前
# 从 torch._decomp 模块中导入 register_decomposition 函数
from torch._decomp import register_decomposition

#
# Elementwise unary references
#

# 定义一个用于推断的标志对象
infer_aten_op = object()

# TODO: add type promotion support
# 定义一个生成元素级一元引用的函数
def _make_elementwise_unary_reference(
    type_promotion_kind,
    *,
    aten_op=infer_aten_op,
    extra_meta=None,
) -> Callable:
    # 内部函数，装饰给定的一元原语函数 prim
    def inner(prim: Callable):
        nonlocal aten_op

        # 包装 prim 函数，处理输出
        @wraps(prim)
        @out_wrapper()
        @elementwise_unary_scalar_wrapper
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a",),
            type_promotion_kind=type_promotion_kind,
        )
        def _ref(a: TensorLikeType) -> TensorLikeType:
            # 如果存在额外的元信息函数，则调用它
            if extra_meta is not None:
                extra_meta(a)

            # 调用原语函数 prim 处理输入 a，并获取输出
            output = prim(a)
            # 处理非连续的输出结果
            return handle_noncontiguous_outputs([a], output)

        # 如果 aten_op 是推断值，则从 utils 模块获取 prim 函数的 aten_op
        if aten_op is infer_aten_op:
            aten_op = utils.get_aten_op(prim, prim.__name__)
        # 如果 aten_op 不为 None，则注册其分解方式
        if aten_op is not None:
            register_decomposition(aten_op)(_ref)

        return _ref

    return inner


# 定义一个生成函数的别名的函数
def _make_alias(fn, name):
    """
    This function defines an alias of another function and sets its __name__ argument.
    It also sets its __module__ argument to the module of the caller.
    Note that when naively doing `alias = fn`, we have that `alias.__name__ == "fn"`, and
    `alias.__module__ == fn.__module__`.
    """
    
    # 内部函数 _fn 被定义为 fn 的别名，设置其 __name__ 和 __module__ 属性
    def _fn(*args, **kwargs):
        return fn(*args, **kwargs)

    _fn.__name__ = name
    _fn.__module__ = inspect.currentframe().f_back.f_globals["__name__"]  # type: ignore[union-attr]
    return _fn


# 定义一个生成函数的原地操作版本的函数
def _make_inplace(fn):
    """
    Given a function with out variant (i.e. using `out_wrapper()`), it returns its in-place variant.
    See https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-do-in-place-operations-work-in-pytorch
    """

    # 内部函数 _fn 是 fn 的原地操作版本
    @wraps(fn)
    def _fn(a, *args, **kwargs):
        return fn(a, *args, out=a, **kwargs)

    # 原地操作的名称
    inplace_name = f"{fn.__name__}_"
    # 注册对应的分解方式
    _fn = register_decomposition(getattr(aten, inplace_name))(_fn)

    # 获取 fn 所在模块的 __all__ 属性，并将原地操作的名称添加进去
    from inspect import getmodule

    _all = getmodule(fn).__all__  # type: ignore[union-attr]
    if inplace_name not in _all:
        _all.append(inplace_name)
    return _fn


# 以下是一系列使用 _make_elementwise_unary_reference 装饰器定义的函数

# 定义一个取绝对值的函数，复数到浮点数的类型提升
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT)
def abs(a):
    return prims.abs(a)

# 定义一个反余弦函数，整数到浮点数的类型提升
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def acos(a):
    return prims.acos(a)

# 定义一个反双曲余弦函数，整数到浮点数的类型提升
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def acosh(a):
    return prims.acosh(a)

# 定义一个反正弦函数，整数到浮点数的类型提升
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def asin(a):
    return prims.asin(a)

# 定义一个反双曲正弦函数，整数到浮点数的类型提升
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def asinh(a):
    # 返回参数 a 的反双曲正弦值
    return prims.asinh(a)
# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `atan`，将整数参数提升为浮点数后调用底层 `prims.atan` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def atan(a):
    return prims.atan(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `atanh`，将整数参数提升为浮点数后调用底层 `prims.atanh` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def atanh(a):
    return prims.atanh(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `bitwise_not`，使用默认类型提升并调用底层 `prims.bitwise_not` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def bitwise_not(a):
    return prims.bitwise_not(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `ceil`，使用默认类型提升并调用底层 `prims.ceil` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def ceil(a):
    return prims.ceil(a)


# 注册函数 `is_complex` 的分解，检查输入参数的数据类型是否为复数
@register_decomposition(aten.is_complex)
def is_complex(input: TensorLikeType):
    return utils.is_complex_dtype(input.dtype)


# 注册函数 `conj_physical` 的分解，并包装返回结果
@out_wrapper()
@register_decomposition(aten.conj_physical)
def conj_physical(input: TensorLikeType):
    # 如果输入数据类型不是复数，则直接返回输入
    if not utils.is_complex_dtype(input.dtype):
        return input
    # 否则调用底层 `prims.conj_physical` 函数返回共轭复数
    return prims.conj_physical(input)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `cos`，将整数参数提升为浮点数后调用底层 `prims.cos` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def cos(a):
    return prims.cos(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `cosh`，将整数参数提升为浮点数后调用底层 `prims.cosh` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def cosh(a):
    return prims.cosh(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `digamma`，将整数参数提升为浮点数后调用底层 `prims.digamma` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def digamma(a):
    return prims.digamma(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `erf`，将整数参数提升为浮点数后调用底层 `prims.erf` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def erf(a):
    return prims.erf(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `erfinv`，将整数参数提升为浮点数后调用底层 `prims.erf_inv` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def erfinv(a):
    return prims.erf_inv(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `erfc`，将整数参数提升为浮点数后调用底层 `prims.erfc` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def erfc(a):
    return prims.erfc(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `exp`，将整数参数提升为浮点数后调用底层 `prims.exp` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def exp(a):
    return prims.exp(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `expm1`，将整数参数提升为浮点数后调用底层 `prims.expm1` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def expm1(a):
    return prims.expm1(a)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `exp2`，将整数参数提升为浮点数后调用底层 `prims.exp2` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def exp2(a):
    return prims.exp2(a)


# `fill` 函数有自己的实现，因为它有一个 `value` 参数
# 使用装饰器 `out_wrapper` 和 `elementwise_type_promotion_wrapper` 对 `fill` 函数进行包装，
# 使用 `ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH` 类型的类型提升策略
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a,"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
)
def fill(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    assert isinstance(value, Number)

    python_type = utils.dtype_to_type(a.dtype)
    # 检查 `value` 参数的类型是否可以安全地转换为 `a` 的数据类型
    if not utils.is_weakly_lesser_type(type(value), python_type):
        msg = f"value argument of type {type(value)} cannot be safely cast to type {python_type}!"
        raise ValueError(msg)

    # 调用底层 `prims.fill` 函数来填充 `a` 的数据并返回
    return prims.fill(a, value)


# `fill_` 函数的实现也有所不同，因为它使用了 `prims.fill` 和 `prims.copy_to` 函数
def fill_(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    # 使用 `prims.fill` 函数填充 `a` 的数据
    r = prims.fill(a, value)
    # 使用 `prims.copy_to` 函数将填充的结果复制到 `a` 上
    prims.copy_to(a, r)
    return a


# 注册函数 `zero` 的分解，并包装返回结果
@register_decomposition(aten.zero)
@out_wrapper()
def zero(input: TensorLikeType) -> TensorLikeType:
    # 返回与输入形状相同的零张量
    return torch.zeros_like(input)


# 使用装饰器 `_make_elementwise_unary_reference` 来创建一元函数 `floor`，使用默认类型提升并调用底层 `prims.floor` 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def floor(a):
    # 返回参数 a 的 floor 值
    return prims.floor(a)
# 使用装饰器 `_make_elementwise_unary_reference` 包装函数 `frac`，指定默认的元素级一元引用类型
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
# 定义函数 `frac`，计算输入张量 `x` 的分数部分
def frac(x: TensorLikeType) -> TensorLikeType:
    # 计算 x 的绝对值向下取整后乘以其符号，得到 x 的截断值
    trunc_x = torch.mul(torch.floor(torch.abs(x)), torch.sign(x))
    # 返回 x 减去截断值后的结果
    return torch.sub(x, trunc_x)


# 函数 `imag` 不使用 `_make_elementwise_unary_reference` 装饰器，因为它不支持输出
# 定义函数 `imag`，计算输入张量 `a` 的虚部
def imag(a: TensorLikeType) -> TensorLikeType:
    # 断言 `a` 是 `TensorLike` 类型
    assert isinstance(a, TensorLike)
    # 检查 `a` 的数据类型是否为复数类型
    torch._check(
        utils.is_complex_dtype(a.dtype), lambda: "imag only supports complex tensors."
    )
    # 返回张量 `a` 的虚部
    return prims.imag(a)


# 使用装饰器 `_make_elementwise_unary_reference` 包装函数 `isfinite`，指定总是返回布尔类型
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=None,  # CompositeImplicitAutograd
)
# 定义函数 `isfinite`，判断输入张量 `a` 的每个元素是否有限
def isfinite(a: TensorLikeType) -> TensorLikeType:
    # 如果 `a` 的数据类型是浮点型或复数类型
    if utils.is_float_dtype(a.dtype) or utils.is_complex_dtype(a.dtype):
        # 返回张量 `a` 中每个元素是否有限的结果
        return prims.isfinite(a)
    
    # 对于其他数据类型，返回一个与 `a` 相同形状的全为 True 的张量
    return ones_like(a, dtype=torch.bool)


# 使用装饰器 `_make_elementwise_unary_reference` 包装函数 `isinf`，指定总是返回布尔类型
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
# 定义函数 `isinf`，判断输入张量 `a` 的每个元素是否为无穷大
def isinf(a: TensorLikeType) -> TensorLikeType:
    # 如果 `a` 的数据类型是复数类型
    if utils.is_complex_dtype(a.dtype):
        # 返回张量 `a` 的实部或虚部是否为无穷大的逻辑或
        return torch.logical_or(isinf(torch.real(a)), isinf(torch.imag(a)))
    # 如果 `a` 的数据类型是浮点型
    if utils.is_float_dtype(a.dtype):
        # 返回张量 `a` 的每个元素是否为正无穷大或负无穷大的结果
        return torch.abs(a) == float("inf")
    # 对于其他数据类型，返回一个与 `a` 相同形状的全为 False 的张量
    return torch.zeros_like(a, dtype=torch.bool)


# 使用装饰器 `_make_elementwise_unary_reference` 包装函数 `isposinf`，指定总是返回布尔类型
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
# 定义函数 `isposinf`，判断输入张量 `a` 的每个元素是否为正无穷大
def isposinf(a: TensorLikeType) -> TensorLikeType:
    # 检查 `a` 的数据类型是否为复数类型，如果是，则抛出异常
    torch._check(
        not utils.is_complex_dtype(a.dtype),
        lambda: f"Complex dtype is not supported for isposinf, got dtype {a.dtype}",
    )
    # 如果 `a` 的数据类型是浮点型
    if utils.is_float_dtype(a.dtype):
        # 返回张量 `a` 的每个元素是否为正无穷大的结果
        return a == float("inf")
    # 对于其他数据类型，返回一个与 `a` 相同形状的全为 False 的张量
    return torch.zeros_like(a, dtype=torch.bool)


# 使用装饰器 `_make_elementwise_unary_reference` 包装函数 `isneginf`，指定总是返回布尔类型
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
# 定义函数 `isneginf`，判断输入张量 `a` 的每个元素是否为负无穷大
def isneginf(a: TensorLikeType) -> TensorLikeType:
    # 检查 `a` 的数据类型是否为复数类型，如果是，则抛出异常
    torch._check(
        not utils.is_complex_dtype(a.dtype),
        lambda: f"Complex dtype is not supported for isneginf, got dtype {a.dtype}",
    )
    # 如果 `a` 的数据类型是浮点型
    if utils.is_float_dtype(a.dtype):
        # 返回张量 `a` 的每个元素是否为负无穷大的结果
        return a == float("-inf")
    # 对于其他数据类型，返回一个与 `a` 相同形状的全为 False 的张量
    return torch.zeros_like(a, dtype=torch.bool)


# 使用装饰器 `_make_elementwise_unary_reference` 包装函数 `isnan`，指定总是返回布尔类型
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
# 定义函数 `isnan`，判断输入张量 `a` 的每个元素是否为 NaN
def isnan(a: TensorLikeType) -> TensorLikeType:
    # 返回张量 `a` 的每个元素是否不等于自身的结果（即判断是否为 NaN）
    return prims.ne(a, a)


# 定义别名 `mvlgamma`，指向 `_make_alias` 函数，用于生成特殊函数 `multigammaln` 的别名
mvlgamma = _make_alias(torch.special.multigammaln, "mvlgamma")  # type: ignore[has-type]


# 使用装饰器 `_make_elementwise_unary_reference` 包装函数 `isreal`，指定总是返回布尔类型
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    aten_op=None,  # CompositeImplicitAutograd
)
# 定义函数 `isreal`，判断输入张量 `a` 的每个元素是否为实数
def isreal(a: TensorLikeType) -> TensorLikeType:
    # 如果 `a` 的数据类型是复数类型
    if utils.is_complex_dtype(a.dtype):
        # 返回张量 `a` 的虚部是否为零的结果
        return torch.imag(a) == 0
    # 对于其他数据类型，返回一个与 `a` 相同形状的全为 True 的张量
    return torch.ones_like(a, dtype=torch.bool)


# 使用装饰器 `_make_elementwise_unary_reference` 包装函数 `i0`，指定将整数转换为浮点数
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, aten_op=aten.i0
)
# 定义函数 `i0`，计算输入张量 `a` 的修正贝塞尔函数
def i0(a):
    # 返回调用 `prims.bessel_i0` 函数计算结果
    return prims.bessel_i0(a)


# 使用装饰器 `_make_elementwise_unary_reference` 包装函数 `lgamma`，指定将整数转换为浮点数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
# 定义函数 `lgamma`，计算输入张量 `a` 的自然对数伽马函数
def lgamma(a):
    # 返回调用 `prims.lgamma` 函数计算结果
    return prims.lgamma(a)
# 使用装饰器 `_make_elementwise_unary_reference` 将 `log` 函数注册为一元函数，将整数类型提升为浮点数类型
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log(a):
    # 调用底层的数学库 `prims` 中的对数函数 `log`
    return prims.log(a)


# 使用装饰器 `_make_elementwise_unary_reference` 将 `log1p` 函数注册为一元函数，将整数类型提升为浮点数类型
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log1p(a):
    # 调用底层的数学库 `prims` 中的对数函数 `log1p`
    return prims.log1p(a)


# 使用装饰器 `_make_elementwise_unary_reference` 将 `log2` 函数注册为一元函数，将整数类型提升为浮点数类型
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log2(a):
    # 调用底层的数学库 `prims` 中的对数函数 `log2`
    return prims.log2(a)


# 使用装饰器 `_make_elementwise_unary_reference` 将 `log10` 函数注册为一元函数，将整数类型提升为浮点数类型
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def log10(a):
    # 调用底层的数学库 `prims` 中的对数函数 `log10`
    return prims.log10(a)


# 使用装饰器 `out_wrapper` 包装 `log_softmax` 函数，不注册为分解
@out_wrapper()
def log_softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # 若未指定数据类型，使用输入张量 `a` 的数据类型
    result_dtype = dtype or a.dtype
    # 根据结果数据类型获取计算数据类型
    computation_dtype = utils.get_computation_dtype(result_dtype)
    # 将输入张量 `a` 转换为计算数据类型 `computation_dtype`
    a_ = _maybe_convert_to_dtype(a, computation_dtype)
    # 计算 `log_softmax`，并将结果转换为结果数据类型 `result_dtype`
    return _maybe_convert_to_dtype(a_ - logsumexp(a_, dim, keepdim=True), result_dtype)  # type: ignore[return-value]


# 将 `logsumexp` 函数注册为分解 `aten.logsumexp` 的函数
@register_decomposition(aten.logsumexp)
# 使用装饰器 `out_wrapper` 包装 `logsumexp` 函数
@out_wrapper()
# 使用装饰器 `elementwise_type_promotion_wrapper`，将 `self` 参数类型提升为浮点数类型
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logsumexp(
    self: TensorLikeType, dim: DimsType, keepdim: bool = False
) -> TensorLikeType:
    # 如果 `dim` 不是可迭代对象，则转换为元组
    if not isinstance(dim, Iterable):
        dim = (dim,)
    # 如果张量 `self` 为空，则返回其指定维度上指数的和的对数
    if self.numel() == 0:
        return torch.sum(torch.exp(self), dim, keepdim).log()
    # 计算张量 `self` 指定维度上的最大值
    maxes = torch.amax(self, dim, keepdim=True)
    # 将无穷大值替换为零
    maxes = torch.masked_fill(maxes, maxes.abs() == float("inf"), 0)
    # 如果 `keepdim` 为真，则保持维度不变，否则挤压维度
    maxes_squeezed = maxes if keepdim else torch.squeeze(maxes, dim)
    # 计算 `self` 减去最大值后的指数和，并保持指定维度
    result = torch.sum(torch.exp(self - maxes), dim, keepdim)
    # 返回结果的对数并加上最大值
    return result.log().add(maxes_squeezed)


# 将 `nan_to_num` 函数注册为分解 `aten.nan_to_num` 的函数
@register_decomposition(aten.nan_to_num)
# 使用装饰器 `out_wrapper` 包装 `nan_to_num` 函数
@out_wrapper()
def nan_to_num(
    a: TensorLikeType,
    nan: Optional[NumberType] = 0.0,
    posinf: Optional[NumberType] = None,
    neginf: Optional[NumberType] = None,
) -> TensorLikeType:
    # 断言输入张量 `a` 是张量类别
    assert isinstance(a, TensorLike)

    # 如果输入张量 `a` 的数据类型是布尔型或整数型，则返回其副本
    if utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
        return a.clone()

    # 如果未指定 `nan`，则设置为零
    if nan is None:
        nan = 0.0

    # 如果未指定 `posinf`，则设置为 `a` 数据类型的最大值
    if posinf is None:
        posinf = torch.finfo(a.dtype).max

    # 如果未指定 `neginf`，则设置为 `a` 数据类型的最小值
    if neginf is None:
        neginf = torch.finfo(a.dtype).min

    # 使用 `torch.where` 函数将 NaN 替换为 `nan`
    result = torch.where(torch.isnan(a), nan, a)  # type: ignore[call-overload]
    # 使用 `torch.where` 函数将负无穷替换为 `neginf`
    result = torch.where(torch.isneginf(a), neginf, result)  # type: ignore[call-overload]
    # 使用 `torch.where` 函数将正无穷替换为 `posinf`
    result = torch.where(torch.isposinf(a), posinf, result)  # type: ignore[call-overload]
    # 返回结果张量
    return result


# 定义 `_neg_meta` 函数，用于检查输入张量的数据类型，确保不是布尔型
def _neg_meta(a: TensorLikeType):
    torch._check(
        a.dtype is not torch.bool,
        lambda: (
            "Negation, the `-` operator, on a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `logical_not()` "
            "operator instead."
        ),
    )


# 使用装饰器 `_make_elementwise_unary_reference` 将 `neg` 函数注册为一元函数，采用默认的类型提升策略，同时传递额外的元信息 `_neg_meta`
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, extra_meta=_neg_meta
)
def neg(a):
    # 返回底层的数学库 `prims` 中的取负函数 `neg`
    return prims.neg(a)
# positive does not use _make_elementwise_unary_reference because it does not support out
# CompositeImplicitAutograd - don't register decomp
def positive(a: TensorLikeType) -> TensorLikeType:
    # 断言参数 a 是 TensorLike 类型
    assert isinstance(a, TensorLike)
    # 如果 a 的数据类型为 torch.bool，则抛出 RuntimeError
    if a.dtype is torch.bool:
        msg = "positive does not support bool tensors."
        raise RuntimeError(msg)
    # 返回参数 a 本身
    return a


# real does not use _make_elementwise_unary_reference because it does not support out
def real(a: TensorLikeType) -> TensorLikeType:
    # 断言参数 a 是 TensorLike 类型
    assert isinstance(a, TensorLike)
    # 如果 a 的数据类型为复数类型，则调用 prims.real 函数处理并返回结果
    if utils.is_complex_dtype(a.dtype):
        return prims.real(a)
    # 否则返回参数 a 本身
    return a


# 使用 @_make_elementwise_unary_reference 装饰器注册 reciprocal 函数，
# 使用 INT_TO_FLOAT 类型提升，返回 prims.reciprocal(a) 的结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def reciprocal(a):
    return prims.reciprocal(a)


# 使用 @_make_elementwise_unary_reference 装饰器注册 round 函数，
# 在执行 aten.round 操作后进行分解注册，使用元素级类型提升并包装输出，
# 默认的类型提升模式为 DEFAULT
@register_decomposition(aten.round)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def round(a: TensorLikeType, *, decimals: int = 0) -> TensorLikeType:
    if decimals == 0:
        # 如果小数位数 decimals 为 0，则调用 prims.round(a) 并返回结果
        return prims.round(a)
    else:
        ten_pow = 10**decimals
        ten_neg_pow = 10 ** (-decimals)
        # 否则先将 a 乘以 ten_pow，再对结果取整，再乘以 ten_neg_pow，返回结果
        return prims.mul(prims.round(prims.mul(a, ten_pow)), ten_neg_pow)


# 使用 @_make_elementwise_unary_reference 装饰器注册 rsqrt 函数，
# 使用 INT_TO_FLOAT 类型提升，返回 prims.rsqrt(a) 的结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def rsqrt(a):
    return prims.rsqrt(a)


# 使用 @_make_elementwise_unary_reference 装饰器注册 sigmoid 函数，
# 使用 INT_TO_FLOAT 类型提升，返回 true_divide(1, add(1, exp(neg(a)))) 的结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sigmoid(a: TensorLikeType) -> TensorLikeType:
    return true_divide(1, add(1, exp(neg(a))))


# 使用 @_make_elementwise_unary_reference 装饰器注册 sgn 函数，
# 使用默认的类型提升模式，处理复数数据类型时使用 a.abs() 和 torch.where 处理结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def sgn(a):
    if utils.is_complex_dtype(a.dtype):
        a_abs = a.abs()
        return torch.where(a_abs == 0, 0, a / a_abs)
    else:
        return a.sign()


# 使用 @_make_elementwise_unary_reference 装饰器注册 sign 函数，
# 使用默认的类型提升模式，返回 prims.sign(a) 的结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def sign(a):
    return prims.sign(a)


# 使用 @_make_elementwise_unary_reference 装饰器注册 signbit 函数，
# 使用 ALWAYS_BOOL 类型提升，返回 prims.signbit(a) 的结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)
def signbit(a):
    return prims.signbit(a)


# 使用 @_make_elementwise_unary_reference 装饰器注册 sin 函数，
# 使用 INT_TO_FLOAT 类型提升，返回 prims.sin(a) 的结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sin(a):
    return prims.sin(a)


# 使用 @_make_elementwise_unary_reference 装饰器注册 sinc 函数，
# 使用 INT_TO_FLOAT 类型提升，根据 a 计算 sinc(a) 的结果
# 当 a 等于 0 时返回 1，否则返回 torch.sin(a) / a 的结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sinc(a):
    a = math.pi * a
    return torch.where(a == 0, 1, torch.sin(a) / a)


# 使用 @_make_elementwise_unary_reference 装饰器注册 sinh 函数，
# 使用 INT_TO_FLOAT 类型提升，返回 prims.sinh(a) 的结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sinh(a):
    return prims.sinh(a)


# 使用 @_make_elementwise_unary_reference 装饰器注册 sqrt 函数，
# 使用 INT_TO_FLOAT 类型提升，返回 prims.sqrt(a) 的结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def sqrt(a):
    return prims.sqrt(a)


# 使用 @_make_elementwise_unary_reference 装饰器注册 square 函数，
# 使用 BOOL_TO_LONG 类型提升，返回 a * a 的结果
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG,
    aten_op=None,  # CompositeImplicitAutograd,
)
def square(a: TensorLikeType) -> TensorLikeType:
    return mul(a, a)


# 使用 @_make_elementwise_unary_reference 装饰器注册 tan 函数，
# 使用 INT_TO_FLOAT 类型提升，返回 prims.tan(a) 的结果
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def tan(a):
    return prims.tan(a)
# 使用装饰器创建一个元素级的一元引用函数，将整数转换为浮点数类型并调用 prims.tanh 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def tanh(a):
    return prims.tanh(a)


# 使用装饰器创建一个元素级的一元引用函数，使用默认的类型提升策略，并调用 prims.trunc 函数
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def trunc(a):
    return prims.trunc(a)


# TODO: register this as a real ref/decomposition once TorchInductor supports complex!
# 将输入张量视为复数张量的函数定义
def view_as_complex(self: TensorLikeType) -> TensorLikeType:
    # 获取输入张量的数据类型
    input_dtype = self.dtype
    # 检查输入数据类型是否为浮点数类型
    torch._check(
        utils.is_float_dtype(input_dtype),
        lambda: f"view_as_complex is only supported for floating point"
        f"tensors, but got a tensor of scalar type: {input_dtype}",
    )
    # 获取输入张量的尺寸
    sizes = self.size()
    # 检查输入张量至少有一个维度
    torch._check(
        len(sizes) != 0,
        lambda: "Input tensor must have one or more dimensions",
    )
    # 检查输入张量的最后一个维度大小是否为2
    torch._check(
        sizes[-1] == 2,
        lambda: "Tensor must have a last dimension of size 2",
    )

    # 获取输入张量的旧步长
    old_strides = self.stride()
    # 检查输入张量最后一个维度的步长是否为1
    torch._check(
        old_strides[-1] == 1,
        lambda: "Tensor must have a last dimension with stride 1",
    )
    # 获取除了最后一个维度外的所有维度的步长
    dims = old_strides[:-1]
    # 检查除了最后一个维度外的所有维度的步长是否都可以被2整除
    torch._check(
        py_all(stride % 2 == 0 for stride in dims),
        lambda: "Tensor must have a stride divisible by 2 for all but last dimension",
    )
    # 检查张量的存储偏移是否可以被2整除
    torch._check(
        self.storage_offset() % 2 == 0,
        lambda: "Tensor must have a storage_offset divisible by 2",
    )
    # 使用 prims.view_element_type 函数将张量视为对应的复数数据类型，并在最后一维上进行挤压
    return prims.view_element_type(
        self, utils.corresponding_complex_dtype(input_dtype)
    ).squeeze(-1)


# 定义一个元素级的二元引用函数的工厂函数，根据指定的参数创建函数并返回
def _make_elementwise_binary_reference(
    type_promotion_kind,
    aten_op=infer_aten_op,
    name=None,
    has_out=True,
    supports_lhs_python_scalar=True,
    supports_rhs_python_scalar=True,
    supports_two_python_scalars=False,
    should_register_decomposition=True,
) -> Callable:
    pass  # 此处省略了函数的具体实现，函数体未提供
    def inner(prim: Callable):
        # 使用 nonlocal 关键字声明变量 aten_op 和 name 来自外部环境
        nonlocal aten_op, name
        # 如果 name 为 None，则使用传入函数 prim 的名称作为 name
        if name is None:
            name = prim.__name__
    
        # 定义内部函数 _ref，使用 wraps 装饰器保留原始函数的元数据
        @wraps(prim)
        # 使用 elementwise_type_promotion_wrapper 装饰器，指定类型提升参数和类型提升方式
        def _ref(
            a: Union[Tensor, NumberType],
            b: Union[Tensor, NumberType],
        ) -> Tensor:
            # 检查是否支持左操作数是 Python 标量，或者 a 不是 Number 类型
            torch._check_value(
                supports_lhs_python_scalar or not isinstance(a, Number),
                lambda: f"{name}: Received a lhs Python scalar to an elementwise binary "
                "operation that does not accept lhs scalars!",
            )
            # 检查是否支持右操作数是 Python 标量，或者 b 不是 Number 类型
            torch._check_value(
                supports_rhs_python_scalar or not isinstance(b, Number),
                lambda: f"{name}: Received a rhs Python scalar to an elementwise binary "
                "operation that does not accept rhs scalars!",
            )
            # 检查是否支持两个 Python 标量作为输入
            torch._check_value(
                supports_two_python_scalars
                or not (isinstance(a, Number) and isinstance(b, Number)),
                lambda: f"{name}: Receive two Number inputs to an elementwise binary operation!",
            )
            # 对 a 和 b 进行广播操作
            a, b = _maybe_broadcast(a, b)
            # 调用原始函数 prim 处理输入 a 和 b，得到输出
            output = prim(a, b)
            # 处理非连续输出，确保返回的输出张量是连续的
            return handle_noncontiguous_outputs([a, b], output)
    
        # 如果有指定输出参数，使用 out_wrapper 对 _ref 进行包装
        if has_out:
            _ref = out_wrapper()(_ref)
    
        # 设置 _ref 的名称为当前函数的名称 name
        _ref.__name__ = name
        # 如果 aten_op 是默认值 infer_aten_op，则通过 utils.get_aten_op 获取实际的操作符
        if aten_op is infer_aten_op:
            aten_op = utils.get_aten_op(prim, name)
        # 如果 aten_op 不为空，并且应该注册分解过程，则注册分解过程到 aten_op
        if aten_op is not None and should_register_decomposition:
            register_decomposition(aten_op)(_ref)
    
        # 返回内部函数 _ref
        return _ref
    
    # 返回内部函数 inner
    return inner
# 注册对 torch.add 的分解，包装输出，进行元素类型提升
@register_decomposition(aten.add)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def add(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    alpha: Optional[NumberType] = None,
):
    """
    Reference implementation of torch.add
    """

    # 可能对 a 和 b 进行广播处理
    a, b = _maybe_broadcast(a, b)

    # 如果指定了 alpha 参数，根据 a 或 b 的类型进行数据类型转换检查
    if alpha is not None:
        dtype = a.dtype if isinstance(a, TensorLike) else b.dtype  # type: ignore[union-attr]
        python_type = utils.dtype_to_type(dtype)
        # 如果 alpha 的类型不是布尔型且不能安全地转换为 a 或 b 的类型，则抛出 ValueError 异常
        if python_type != bool and not utils.is_weakly_lesser_type(
            type(alpha), python_type
        ):
            msg = f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!"
            raise ValueError(msg)
        # 如果 b 是 TensorLike 类型，则将 b 乘以 alpha
        if isinstance(b, TensorLike):
            b = prims.mul(b, alpha)
        else:
            b = b * alpha

    # 调用 prims.add 函数进行加法操作
    output = prims.add(a, b)
    # 处理非连续输出，返回处理后的输出结果
    return handle_noncontiguous_outputs([a, b], output)


# 创建 atan2 的元素级二进制引用
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def atan2(a, b):
    # 调用 prims.atan2 函数进行反正切操作
    return prims.atan2(a, b)


# 创建 bitwise_and 的元素级二进制引用
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_and(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 prims.bitwise_and 函数进行按位与操作
    return prims.bitwise_and(a, b)


# 创建 bitwise_left_shift 的元素级二进制引用
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_left_shift(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 prims.shift_left 函数进行按位左移操作
    return prims.shift_left(a, b)


# 创建 bitwise_or 的元素级二进制引用
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_or(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 prims.bitwise_or 函数进行按位或操作
    return prims.bitwise_or(a, b)


# 创建 bitwise_right_shift 的元素级二进制引用
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_right_shift(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 prims.shift_right_arithmetic 函数进行按位右移操作
    return prims.shift_right_arithmetic(a, b)


# 创建 bitwise_xor 的元素级二进制引用
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def bitwise_xor(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 prims.bitwise_xor 函数进行按位异或操作
    return prims.bitwise_xor(a, b)


# 创建 copysign 的元素级二进制引用
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    supports_lhs_python_scalar=False,
)
def copysign(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    # 如果 b 是数值并且 a 是 Tensor 类型，则将 b 转换为与 a 相同的张量类型
    if isinstance(b, Number) and isinstance(a, Tensor):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    # 如果 a 和 b 都是 Tensor 类型，并且它们在不同的设备上，引发运行时错误
    elif isinstance(a, Tensor) and isinstance(b, Tensor) and a.device != b.device:
        # 构造错误信息，指示除数（b）应该与被除数（a）在相同的设备上，但实际上在不同设备上
        msg = f"Expected divisor (b) to be on the same device ({a.device}) as dividend (a), but it is found on {b.device}!"
        # 抛出运行时异常，包含错误消息
        raise RuntimeError(msg)
    
    # 返回根据条件选择计算的结果：如果 b 是负数，则取相反数的绝对值，否则取绝对值
    return where(signbit(b), neg(abs(a)), abs(a))
# 使用 _make_elementwise_binary_reference 函数创建对 prims.complex 的元素级二进制引用
# 使用默认的类型提升方式 ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
complex = _make_elementwise_binary_reference(prims.complex, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)

# 注册对 aten.div 函数的分解装饰器，带有 @out_wrapper 装饰器
# 定义 div 函数，接受两个参数 a 和 b，可以是 TensorLikeType 或 NumberType 类型
# rounding_mode 是一个可选的字符串参数，默认为 None
# 如果 rounding_mode 是 None，则调用 true_divide(a, b)
# 如果 rounding_mode 是 "trunc"，则调用 trunc_divide(a, b)
# 如果 rounding_mode 是 "floor"，则调用 floor_divide(a, b)
# 否则抛出 ValueError 异常，提示 rounding_mode 参数错误
@register_decomposition(aten.div)
@out_wrapper()
def div(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    rounding_mode: Optional[str] = None,
):
    """
    Reference implementation of torch.div
    """

# 使用 _make_elementwise_binary_reference 函数创建对 prims.eq 的元素级二进制引用
# 类型提升方式为 ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
# 不支持左侧 Python 标量，即第一个参数不支持 Python 原生数值类型
def eq(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.eq(a, b)

# 使用 _make_elementwise_binary_reference 函数创建对 prims.pow 的元素级二进制引用
# 类型提升方式为 ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
def pow(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
) -> TensorLikeType:
    assert isinstance(a, TensorLikeType) or isinstance(b, TensorLikeType)

    # 根据参数类型进行不同的处理
    if isinstance(b, Number):
        if b == 1.0:
            return a.clone()  # type: ignore[return-value,union-attr]
        elif b == 2.0:
            return a * a  # type: ignore[return-value]
        elif b == 0.5:
            return torch.sqrt(a)  # type: ignore[arg-type]
    elif isinstance(a, Number):
        if a == 1.0:
            return torch.fill(b, True)
        if a == 2.0 and (
            utils.is_float_dtype(b.dtype) or utils.is_complex_dtype(b.dtype)
        ):
            return torch.exp2(b)

    return prims.pow(a, b)

# 不注册分解的 CompositeImplicitAutograd，带有 @out_wrapper 装饰器
# 定义 float_power 函数，接受两个参数 a 和 b，可以是 TensorLikeType 或 NumberType 类型
# 如果 a 和 b 都是 Number 类型，抛出 ValueError 异常
# 获取类型提升后的 dtype，若为复数则为 torch.complex128，否则为 torch.float64
# 将 a 和 b 转换为 dtype 类型，并进行可能的广播
# 调用 pow(a, b) 返回结果
@out_wrapper()
def float_power(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
) -> Tensor:
    if isinstance(a, Number) and isinstance(b, Number):
        raise ValueError(
            "Receive two Number inputs to an elementwise binary operation!"
        )

    dtype = utils.get_higher_dtype(a, b)
    assert dtype is not None
    if utils.is_complex_dtype(dtype):
        dtype = torch.complex128
    else:
        dtype = torch.float64

    a = _maybe_convert_to_dtype(a, dtype)
    b = _maybe_convert_to_dtype(b, dtype)

    a, b = _maybe_broadcast(a, b)
    return pow(a, b)
# 定义了一个双元素的引用函数，用于执行 floor_divide 操作。
# 其中包含了一些元数据，如类型提升的种类、是否支持两个 Python 标量等。
@_make_elementwise_binary_reference(
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_two_python_scalars=True,
    should_register_decomposition=False,
)
# floor_divide 函数定义，用于执行整除操作。
def floor_divide(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    # 如果 a 和 b 都是标量数字，则将它们包装成张量，因为有些引用只接受张量参数。
    if isinstance(a, Number) and isinstance(b, Number):
        a = scalar_tensor(a)
        b = scalar_tensor(b)
    # 如果 b 是数字且 a 是张量，则将 b 包装成张量，并保持 dtype 和 device 一致。
    elif isinstance(b, Number) and isinstance(a, Tensor):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    # 如果 a 是数字且 b 是张量，则将 a 包装成张量，并保持 dtype 和 device 一致。
    elif isinstance(a, Number) and isinstance(b, Tensor):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)
    # 如果 a 和 b 都是张量且位于不同设备上，则将 b 移动到与 a 相同的设备上。
    elif isinstance(a, Tensor) and isinstance(b, Tensor) and a.device != b.device:
        if a.device == torch.device("cpu"):
            msg = f"Expected divisor (b) to be on the same device ({a.device}) as dividend (a), but it is found on {b.device}!"
            raise RuntimeError(msg)
        else:
            b = prims.device_put(b, device=a.device)

    # 确保 a 和 b 都是张量类型
    assert isinstance(a, Tensor) and isinstance(b, Tensor)
    dtype = a.dtype
    # 如果 a 的数据类型是浮点型，则执行浮点数的 floor_divide 操作
    if utils.is_float_dtype(dtype):
        return _floor_divide_float(a, b)
    # 如果 a 的数据类型是整型，则执行整数的 floor_divide 操作
    elif utils.is_integer_dtype(dtype):
        return _floor_divide_integer(a, b)
    else:
        # 如果数据类型不支持 floor_divide 操作，则抛出异常
        torch._check(False, lambda: f"{dtype} not supported for floor_divide")


# 执行整数类型的 floor_divide 操作
def _floor_divide_integer(a: Tensor, b: Tensor) -> Tensor:
    # 对 a 和 b 进行广播，使它们具有相同的形状
    a, b = _maybe_broadcast(a, b)

    # 如果 a 的数据类型是无符号整数类型，则使用 prims.div 执行整数的除法操作
    if not a.dtype.is_signed:
        return prims.div(a, b)

    # 将截断转换为 floor 操作：
    offset = (torch.signbit(a) != torch.signbit(b)).logical_and(torch.fmod(a, b) != 0)
    return prims.div(a, b) - _maybe_convert_to_dtype(offset, a.dtype)


# 执行浮点数类型的 floor_divide 操作
def _floor_divide_float(a: Tensor, b: Tensor) -> Tensor:
    # 计算 a 对 b 的取模
    mod = fmod(a, b)
    # 计算真实除法的结果
    div = true_divide(sub(a, mod), b)

    # 确保余数与除数具有相同的符号
    different_signed_inputs = bitwise_xor(lt(a, 0), lt(b, 0))
    non_zero_remainder = ne(mod, 0)
    mask = bitwise_and(non_zero_remainder, different_signed_inputs)
    div = where(mask, sub(div, 1), div)

    # 将商映射到最接近的整数值
    floor_div = floor(div)
    mask = gt(sub(div, floor_div), 0.5)
    # 根据掩码条件，将 floor_div 中的每个元素加一，生成新的 floor_div 结果
    floor_div = where(mask, add(floor_div, 1), floor_div)

    # 计算 a 与 b 的真实除法结果，生成 basic_div
    basic_div = true_divide(a, b)
    # 创建一个与 basic_div 相同数据类型和设备的标量张量，值为 0，用于处理零除情况
    zero_tensor = scalar_tensor(0, dtype=basic_div.dtype, device=basic_div.device)

    # 如果除法的结果不为零，则将 floor_div 保留；否则根据 basic_div 的符号复制零张量的符号
    floor_div = where(ne(div, 0), floor_div, copysign(zero_tensor, basic_div))

    # 如果分母 b 不为零，则返回 floor_div；否则返回 basic_div，保持真实除法的行为
    return where(ne(b, 0), floor_div, basic_div)
# 使用装饰器 `_make_elementwise_binary_reference` 来定义函数 `fmax`，其类型提升为默认类型，不支持左右操作数为 Python 标量
def fmax(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 `prims.fmax` 函数，返回两个张量的元素级最大值
    return prims.fmax(a, b)


# 使用装饰器 `_make_elementwise_binary_reference` 来定义函数 `fmin`，其类型提升为默认类型，不支持左右操作数为 Python 标量
def fmin(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 `prims.fmin` 函数，返回两个张量的元素级最小值
    return prims.fmin(a, b)


# 使用装饰器 `_make_elementwise_binary_reference` 来定义函数 `fmod`，其类型提升为默认类型，支持左操作数为张量，右操作数为 Python 标量
def fmod(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 `prims.fmod` 函数，返回两个张量的元素级取模运算结果
    return prims.fmod(a, b)


# 使用装饰器 `register_decomposition` 和 `out_wrapper` 来定义函数 `frexp`，其类型为返回解压元组，返回两个张量的元素级最小值
def frexp(self: TensorLikeType) -> Tuple[TensorLikeType, TensorLikeType]:
    # 调用 `torch.return_types.frexp` 函数，返回两个张量的元素级最小值
    return torch.return_types.frexp(prims.frexp(self))


# 使用装饰器 `_make_elementwise_binary_reference` 来定义函数 `gcd`，其类型提升为默认类型，不支持左右操作数为 Python 标量
def gcd(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 `prims.gcd` 函数，返回两个张量的元素级最大公约数
    return prims.gcd(a, b)


# 使用装饰器 `_make_elementwise_binary_reference` 来定义函数 `ge`，其类型提升为布尔类型，不支持左操作数为 Python 标量
def ge(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 `prims.ge` 函数，返回两个张量的元素级大于等于比较结果（布尔类型）
    return prims.ge(a, b)


# 使用装饰器 `_make_elementwise_binary_reference` 来定义函数 `gt`，其类型提升为布尔类型，不支持左操作数为 Python 标量
def gt(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 `prims.gt` 函数，返回两个张量的元素级大于比较结果（布尔类型）
    return prims.gt(a, b)


# 使用装饰器 `_make_elementwise_binary_reference` 来定义函数 `heaviside`，其类型提升为默认类型，不支持左右操作数为 Python 标量
def heaviside(input: TensorLikeType, values: TensorLikeType) -> TensorLikeType:
    # 检查输入张量是否等于零
    input_eq_zero = torch.eq(input, 0)
    # 检查输入张量是否小于零或者是 NaN
    input_lt_zero = torch.logical_or(torch.lt(input, 0), torch.isnan(input))
    # 根据条件选择返回值，生成一个零和一组成的张量
    zeros_and_ones = torch.where(input_lt_zero, 0, 1)
    # 根据输入是否为零返回对应的值或者之前生成的张量
    output = torch.where(input_eq_zero, values, zeros_and_ones)
    return output


# 使用装饰器 `_make_elementwise_binary_reference` 来定义函数 `hypot`，其类型提升为默认类型，不支持左右操作数为 Python 标量
def hypot(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 `prims.hypot` 函数，返回两个张量的元素级欧几里得距离
    return prims.hypot(a, b)


# 使用装饰器 `_make_elementwise_binary_reference` 来定义函数 `igamma`，其类型提升为整数到浮点数类型，不支持左右操作数为 Python 标量
def igamma(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 `prims.igamma` 函数，返回两个张量的元素级不完全伽马函数
    return prims.igamma(a, b)
    # 设置参数 supports_lhs_python_scalar 为 False，表示不支持左操作数为 Python 标量
    supports_lhs_python_scalar=False,
    # 设置参数 supports_rhs_python_scalar 为 False，表示不支持右操作数为 Python 标量
    supports_rhs_python_scalar=False,
# 定义函数 igammac，接受两个类型为 TensorLikeType 的参数 a 和 b，并返回 TensorLikeType
def igammac(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 prims 模块的 igammac 函数，将参数 a 和 b 传递给它，并返回结果
    return prims.igammac(a, b)


# 定义函数 _check_close_args，用于检查是否可以进行“接近（close）”比较
def _check_close_args(
    name: str,
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float,
    atol: float,
) -> None:
    # 检查参数 a 和 b 的数据类型是否相同，如果不同则抛出异常
    torch._check_value(
        a.dtype == b.dtype,
        lambda: f"{name}: Attempting to compare tensors of different dtypes {a.dtype} and {b.dtype}!",
    )
    # 检查相对误差 rtol 是否大于等于零，如果不是则抛出异常
    torch._check(
        rtol >= 0,
        lambda: f"{name}: rtol must be greater than or equal to zero, but got {rtol}!",
    )
    # 检查绝对误差 atol 是否大于等于零，如果不是则抛出异常
    torch._check(
        atol >= 0,
        lambda: f"{name}: atol must be greater than or equal to zero, but got {atol}!",
    )


# 定义函数 isclose，用于比较两个张量 a 和 b 是否“接近”
def isclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> TensorLikeType:
    # 调用 _check_close_args 函数，检查参数的有效性
    _check_close_args(name="torch.isclose", a=a, b=b, rtol=rtol, atol=atol)

    # 使用 eq 函数比较张量 a 和 b 是否相等，得到布尔张量 close
    close = eq(a, b)
    
    # 如果 equal_nan 为 True，并且 a 和 b 的数据类型为浮点型或复数型，则进一步处理 NaN 的情况
    if equal_nan and (utils.is_float_dtype(a.dtype) or utils.is_complex_dtype(a.dtype)):
        # 将 close 更新为包含 NaN 的情况的逻辑或操作结果
        close = logical_or(close, logical_and(isnan(a), isnan(b)))

    # 如果 rtol 和 atol 都为零，则直接返回 close 结果，避免误差可能导致的误判
    if atol == 0 and rtol == 0:
        return close

    # 计算允许的误差范围
    allowed_error = add(atol, abs(mul(b, rtol)))
    # 计算实际误差
    actual_error = abs(sub(a, b))

    # 计算最终的“接近性”结果
    result = logical_or(
        close, logical_and(isfinite(actual_error), le(actual_error, allowed_error))
    )

    return result


# 定义函数 lcm，用于计算两个张量 a 和 b 的最小公倍数
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def lcm(a: TensorLikeType, b: TensorLikeType):
    # 获取参数 a 的数据类型
    dtype = a.dtype
    # 如果数据类型为 int8 或 int16，则将其提升为 int32，以保持与 C++ 的一致性并避免溢出
    promote_to_int = dtype in (torch.int8, torch.int16)
    # 如果需要将输入张量提升为整数类型
    if promote_to_int:
        # 将张量 a 和 b 转换为 torch.int32 类型
        a = prims.convert_element_type(a, torch.int32)
        b = prims.convert_element_type(b, torch.int32)

    # 计算张量 a 和 b 的最大公约数
    g = torch.gcd(a, b)
    # 处理避免在 gcd(0, 0) == 0 的情况下出现除以零的情况
    g = torch.where(g == 0, 1, g)
    # 计算调整后的结果，确保结果是整数
    res = torch.abs(prims.div(a, g) * b)
    # 如果需要将结果提升为整数类型，则进行类型转换
    return res if not promote_to_int else prims.convert_element_type(res, dtype)
# 使用装饰器 `_make_elementwise_binary_reference` 包装函数 `le`
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,  # 总是将操作数提升为布尔类型
    supports_lhs_python_scalar=False,  # 不支持左侧Python标量
)
def le(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.le(a, b)  # 调用 `prims.le` 函数比较 `a` 和 `b`


# 使用装饰器 `_make_elementwise_binary_reference` 包装函数 `logaddexp`
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,  # 默认的操作数提升方式
    supports_lhs_python_scalar=False,  # 不支持左侧Python标量
    supports_rhs_python_scalar=False,  # 不支持右侧Python标量
)
def logaddexp(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 注意：当 `a == b` 时，梯度分布不均匀
    mask = torch.real(a) >= torch.real(b)  # 创建布尔掩码，比较 `a` 和 `b` 的实部大小
    max_ = torch.where(mask, a, b)  # 选择较大的值作为 `max_`
    min_ = torch.where(mask, b, a)  # 选择较小的值作为 `min_`
    inf_mask = torch.logical_and(
        torch.logical_not(torch.isfinite(torch.real(a))), torch.real(a) == torch.real(b)
    )  # 创建一个布尔掩码，处理无穷大和相等实部的情况
    if utils.is_complex_dtype(a.dtype) or utils.is_complex_dtype(b.dtype):
        # 如果 `a` 或 `b` 的数据类型为复数
        neg_min_mask = torch.real(min_) < 0  # 创建一个负最小值的布尔掩码
        inf_vals = torch.where(
            neg_min_mask, min_, torch.log(torch.exp(min_) + torch.exp(max_))
        )  # 处理无穷值的情况
        non_nan_vals = torch.where(
            inf_mask, inf_vals, max_ + torch.log1p(torch.exp(min_ - max_))
        )  # 处理非 NaN 值的情况
        nan_mask = torch.isnan(min_)  # 创建一个NaN值的布尔掩码
        return torch.where(nan_mask, complex(float("nan"), float("nan")), non_nan_vals)  # 返回处理后的值
    else:
        return torch.where(inf_mask, a, max_ + torch.log1p(torch.exp(min_ - max_)))  # 返回处理后的值


# 使用装饰器 `_make_elementwise_binary_reference` 包装函数 `logaddexp2`
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,  # 默认的操作数提升方式
    supports_lhs_python_scalar=False,  # 不支持左侧Python标量
    supports_rhs_python_scalar=False,  # 不支持右侧Python标量
)
def logaddexp2(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    torch._check(
        not (utils.is_complex_dtype(a.dtype) or utils.is_complex_dtype(b.dtype)),
        lambda: "logaddexp2 doesn't support complex dtypes",
    )  # 检查数据类型是否为复数，若是则报错
    # 注意：当 `a == b` 时，梯度分布不均匀
    mask = a >= b  # 创建布尔掩码，比较 `a` 和 `b` 的大小
    max_ = torch.where(mask, a, b)  # 选择较大的值作为 `max_`
    min_ = torch.where(mask, b, a)  # 选择较小的值作为 `min_`
    inf_mask = torch.logical_and(torch.isinf(a), a == b)  # 创建一个布尔掩码，处理无穷和相等的情况
    inv_log_2 = 1.0 / math.log(2)  # 计算对数2的倒数
    result = max_ + torch.log1p(torch.exp2(min_ - max_)) * inv_log_2  # 计算最终结果
    return torch.where(inf_mask, a, result)  # 返回处理后的值


# 使用装饰器 `_make_elementwise_binary_reference` 包装函数 `logical_and`
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,  # 总是将操作数提升为布尔类型
)
def logical_and(a: TensorLikeType, b: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):  # 如果 `a` 的数据类型不是布尔型
        a = a != 0  # 将 `a` 转换为布尔类型
    if not utils.is_boolean_dtype(b.dtype):  # 如果 `b` 的数据类型不是布尔型
        b = b != 0  # 将 `b` 转换为布尔类型
    return a & b  # 返回逻辑与运算的结果


# 使用装饰器 `_make_elementwise_unary_reference` 包装函数 `logical_not`
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL)  # 总是将操作数提升为布尔类型
def logical_not(a: TensorLikeType):
    if not utils.is_boolean_dtype(a.dtype):  # 如果 `a` 的数据类型不是布尔型
        return a == 0  # 返回 `a` 是否等于0的布尔结果
    return ~a  # 返回逻辑非运算的结果
# 使用逻辑或运算符对两个张量或类型进行逻辑或运算
def logical_or(a: TensorLikeType, b: TensorLikeType):
    # 如果 a 的数据类型不是布尔型，则将其转换为布尔型
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    # 如果 b 的数据类型不是布尔型，则将其转换为布尔型
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    # 返回 a 和 b 的逐元素逻辑或运算结果
    return bitwise_or(a, b)


# 逻辑异或运算，元素逐个进行异或操作
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)
def logical_xor(a: TensorLikeType, b: TensorLikeType):
    # 如果 a 的数据类型不是布尔型，则将其转换为布尔型
    if not utils.is_boolean_dtype(a.dtype):
        a = a != 0
    # 如果 b 的数据类型不是布尔型，则将其转换为布尔型
    if not utils.is_boolean_dtype(b.dtype):
        b = b != 0
    # 返回 a 和 b 的逐元素逻辑异或运算结果
    return a ^ b


# 小于运算，返回 a 是否小于 b 的逐元素比较结果
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def lt(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.lt(a, b)


# 返回 a 和 b 的逐元素最大值
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def maximum(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.maximum(a, b)


# 返回 a 和 b 的逐元素最小值
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def minimum(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.minimum(a, b)


# 返回 a 和 b 的逐元素乘积
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    supports_two_python_scalars=True,
)
def mul(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.mul(a, b)


# 不等于运算，返回 a 和 b 的逐元素是否不相等的结果
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
    supports_lhs_python_scalar=False,
)
def ne(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.ne(a, b)


# 返回 a 和 b 的逐元素下一个浮点数
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
    supports_lhs_python_scalar=False,
    supports_rhs_python_scalar=False,
)
def nextafter(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.nextafter(a, b)


# 返回 a 和 b 的逐元素取余数结果
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def remainder(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.remainder(a, b)


# 反向减法，返回 b 减去 a 的结果，支持 TensorLikeType 或 NumberType
@register_decomposition(aten.rsub)
@out_wrapper()
def rsub(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    alpha: NumberType = 1,
):
    # 如果 a 是数字，则抛出错误，因为预期 a 是张量
    if isinstance(a, Number):
        msg = "Received a Number for the first argument, but expected a Tensor"
        raise ValueError(msg)

    # 返回 b 减去 a 的结果，alpha 是可选的乘法因子
    return torch.sub(b, a, alpha=alpha)


# 减法操作，支持 TensorLikeType 或 NumberType，并有一个 alpha 参数
@register_decomposition(aten.sub)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def sub(
    a: Union[TensorLikeType, NumberType],
    b: Union[TensorLikeType, NumberType],
    *,
    alpha: NumberType = 1,
):
    # TODO: consider refactoring this with add impl
    # 减法有自己的实现，因为它具有 alpha 参数
    pass
    """
    Reference implementation of torch.sub
    """
    
    # 尝试将输入参数 a, b 进行广播
    a, b = _maybe_broadcast(a, b)
    
    # 检查 a 和 b 是否都是 TensorLike 类型，如果是则检查其数据类型是否为布尔类型，不支持布尔类型的张量相减操作
    if isinstance(a, TensorLike) and isinstance(b, TensorLike):
        torch._check(
            not utils.is_boolean_dtype(a.dtype) and not utils.is_boolean_dtype(b.dtype),
            lambda: (
                "Subtraction, the `-` operator, with two bool tensors is not supported. "
                "Use the `^` or `logical_xor()` operator instead."
            ),
        )
    
    # 如果 alpha 不等于 1，则根据 a 或 b 的类型确定数据类型 dtype，然后根据 alpha 的类型将 b 扩展乘以 alpha
    if alpha != 1:
        dtype = a.dtype if isinstance(a, TensorLike) else b.dtype  # type: ignore[union-attr]
        python_type = utils.dtype_to_type(dtype)
        # 检查 alpha 的类型是否能够安全地转换为 b 的数据类型
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!"
            raise ValueError(msg)
        if isinstance(b, torch.Tensor):
            b = prims.mul(b, alpha)
        else:
            # 当 b 是标量或符号整数时，谨慎使用乘法，以免破坏类型提升
            b = b * alpha
    
    # 使用底层的 prims.sub 函数对 a 和 b 进行减法操作，得到输出
    output = prims.sub(a, b)
    # 处理非连续输出的情况，并返回结果
    return handle_noncontiguous_outputs([a, b], output)
# 使用装饰器 `_make_elementwise_binary_reference` 创建一个元素级别的二元参考函数，
# 它执行整数到浮点数的类型提升，命名为 "true_divide"
@_make_elementwise_binary_reference(
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    name="true_divide",
    aten_op=None,  # 该参数是 `CompositeImplicitAutograd` 的一部分
    supports_two_python_scalars=True,
)
def true_divide(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    # 调用 prims 模块中的 div 函数来执行真正的除法操作
    return prims.div(a, b)


# 使用装饰器注册对 aten.xlogy 函数的分解
@register_decomposition(aten.xlogy)
# 使用装饰器 out_wrapper 对返回值进行处理
@out_wrapper()
# 使用装饰器 elementwise_type_promotion_wrapper 进行类型提升，将整数提升为浮点数
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def xlogy(a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]):
    # 使用 torch._check 函数验证参数 a 或 b 是否是 TensorLike 类型，否则抛出异常
    torch._check(
        isinstance(a, TensorLike) or isinstance(b, TensorLike),
        lambda: 'Expected either argument a or b to be a Tensor"',
    )

    # 某些操作如 eq 和 log 不处理标量值，因此将它们转换为标量张量
    if isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)
    elif isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)

    # 使用 torch.where 函数进行条件选择，计算 rhs 的值
    # 如果 a 等于 0，则 rhs 为 0；否则 rhs 为 a 乘以 log(b)
    rhs = torch.where(torch.eq(a, 0), 0, torch.mul(a, torch.log(b)))
    # 如果 b 包含 NaN，则返回 NaN；否则返回 rhs
    return torch.where(torch.isnan(b), float("nan"), rhs)


# 使用装饰器 `_make_elementwise_binary_reference` 创建一个元素级别的二元参考函数，
# 使用默认的类型提升规则，没有指定 `name`，因此使用默认名称
@_make_elementwise_binary_reference(
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    aten_op=None,  # 该参数是 `CompositeImplicitAutograd` 的一部分
    supports_two_python_scalars=True,
)
def trunc_divide(
    a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]
):
    # 获取 a 的数据类型
    dtype = utils.get_dtype(a)
    # 如果 a 的数据类型是整数类型，则使用 prims 模块的 div 函数执行整数除法
    if utils.is_integer_dtype(dtype):
        return prims.div(a, b)

    # 否则执行浮点数除法，然后对结果进行截断
    return trunc(prims.div(a, b))


#
# Elementwise Ternary References
#


# 使用装饰器注册对 aten.addcdiv 函数的分解
@register_decomposition(aten.addcdiv)
# 使用装饰器 out_wrapper 对返回值进行处理
@out_wrapper()
# 使用装饰器 elementwise_type_promotion_wrapper 进行类型提升，将整数提升为浮点数
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "tensor1", "tensor2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def addcdiv(
    self: TensorLikeType,
    tensor1: TensorLikeType,
    tensor2: TensorLikeType,
    *,
    value: NumberType = 1,
) -> TensorLikeType:
    """
    Reference implementation of torch.addcdiv
    """
    # 如果 value 不为 None，则验证它的类型是否可以安全地转换为 self 的数据类型
    if value is not None:
        dtype = self.dtype  # 不允许标量，参考 add 函数
        python_type = utils.dtype_to_type(dtype)
        # 使用 torch._check_value 函数验证 value 的类型是否可以安全转换为 python_type
        torch._check_value(
            utils.is_weakly_lesser_type(type(value), python_type),
            lambda: f"value argument of type {type(value)} cannot be safely cast to type {python_type}!",
        )

    # 返回 self 加上 value 乘以 tensor1 除以 tensor2 的结果
    return self + value * tensor1 / tensor2


# 使用装饰器注册对 aten.addcmul 函数的分解
@register_decomposition(aten.addcmul)
# 使用装饰器 out_wrapper 对返回值进行处理
@out_wrapper()
# 使用装饰器 elementwise_type_promotion_wrapper 进行类型提升，使用默认的类型提升规则
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "tensor1", "tensor2"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def addcmul(
    self: TensorLikeType,
    tensor1: TensorLikeType,
    tensor2: TensorLikeType,
    *,
    value: NumberType = 1,
) -> TensorLikeType:
    """
    Reference implementation of torch.addcmul
    """
    # 如果 value 不为 None，则进行类型检查
    if value is not None:
        # 获取当前张量的数据类型
        dtype = self.dtype  # no scalars allowed, see add
        # 将数据类型转换为 Python 中的类型
        python_type = utils.dtype_to_type(dtype)
        # 检查 value 的类型是否可以安全地转换为当前张量的数据类型
        torch._check_value(
            utils.is_weakly_lesser_type(type(value), python_type),
            # 如果类型不兼容，则抛出错误信息
            lambda: f"value argument of type {type(value)} cannot be safely cast to type {python_type}!",
        )

    # 返回当前张量加上 value 乘以 tensor1 和 tensor2 的结果
    return self + value * tensor1 * tensor2
# 使用@register_decomposition装饰器将`aten.clamp`注册为分解函数，使其能够被外部调用
# 使用@out_wrapper装饰器对clamp函数进行输出包装处理
# 使用@elementwise_type_promotion_wrapper装饰器对元素进行类型提升处理，指定参数和处理类型

def clamp(
    a: TensorLikeType,  # 定义输入参数a的类型为TensorLikeType
    min: Optional[TensorOrNumberLikeType] = None,  # 定义可选输入参数min的类型为TensorOrNumberLikeType，默认值为None
    max: Optional[TensorOrNumberLikeType] = None,  # 定义可选输入参数max的类型为TensorOrNumberLikeType，默认值为None
) -> TensorLikeType:  # 函数返回类型为TensorLikeType

    # NOTE: grad behavior with implementation `where` is not consistent on `nan`
    # 如果min和max都为None，则抛出值错误异常
    if min is None and max is None:
        msg = "clamp called but both min and max are none!"
        raise ValueError(msg)

    # 如果min不为None，则执行以下操作
    if min is not None:
        a_isnan = torch.isnan(a)  # 检查a是否包含NaN值
        condition = torch.bitwise_or(torch.ge(a, min), a_isnan)  # 构造条件，当a大于等于min或者a是NaN时为True
        a = torch.where(condition, a, min)  # 使用torch.where根据条件选择a或min的值

    # 如果max不为None，则执行以下操作
    if max is not None:
        a_isnan = torch.isnan(a)  # 再次检查a是否包含NaN值
        condition = torch.bitwise_or(torch.le(a, max), a_isnan)  # 构造条件，当a小于等于max或者a是NaN时为True
        a = torch.where(condition, a, max)  # 使用torch.where根据条件选择a或max的值

    return a  # 返回处理后的张量a


# 使用@register_decomposition装饰器将`aten.clamp_min`注册为分解函数，使其能够被外部调用
# 使用@out_wrapper装饰器对clamp_min函数进行输出包装处理
def clamp_min(
    self: TensorLikeType,  # 定义输入参数self的类型为TensorLikeType
    min: Optional[TensorOrNumberLikeType] = None,  # 定义可选输入参数min的类型为TensorOrNumberLikeType，默认值为None
) -> TensorLikeType:  # 函数返回类型为TensorLikeType

    return torch.clamp(self, min=min)  # 使用torch.clamp函数对self进行最小值约束，返回处理后的张量self


# 使用@register_decomposition装饰器将`aten.clamp_max`注册为分解函数，使其能够被外部调用
# 使用@out_wrapper装饰器对clamp_max函数进行输出包装处理
def clamp_max(
    self: TensorLikeType,  # 定义输入参数self的类型为TensorLikeType
    max: Optional[TensorOrNumberLikeType] = None,  # 定义可选输入参数max的类型为TensorOrNumberLikeType，默认值为None
) -> TensorLikeType:  # 函数返回类型为TensorLikeType

    return torch.clamp(self, max=max)  # 使用torch.clamp函数对self进行最大值约束，返回处理后的张量self


#
# Conditional references
#


# https://pytorch.org/docs/stable/generated/torch.where.html
# TODO: implement alternate where
# 使用@register_decomposition装饰器将`aten.where`注册为分解函数，使其能够被外部调用
# 使用@out_wrapper装饰器对where函数进行输出包装处理
# 使用@elementwise_type_promotion_wrapper装饰器对元素进行类型提升处理，指定参数和处理类型
def where(
    pred: Tensor,  # 定义输入参数pred的类型为Tensor
    a: Optional[TensorOrNumberLikeType] = None,  # 定义可选输入参数a的类型为TensorOrNumberLikeType，默认值为None
    b: Optional[TensorOrNumberLikeType] = None,  # 定义可选输入参数b的类型为TensorOrNumberLikeType，默认值为None
):
    """ """

    # 如果a或者b为None，则抛出未实现错误
    if a is None or b is None:
        raise NotImplementedError

    # 检查pred、a、b三者的设备是否一致，允许CPU标量张量与其他设备张量混用
    utils.check_same_device(pred, a, b, allow_cpu_scalar_tensors=True)

    # 检查pred的数据类型是否为bool类型，如果不是则抛出类型错误
    torch._check(
        pred.dtype is torch.bool,
        lambda: f"expected predicate to be bool, got {pred.dtype}",
    )

    # 对pred、a、b进行可能的广播处理
    pred, a, b = _maybe_broadcast(pred, a, b)

    # 调用prims模块的where函数进行条件选择，返回处理后的结果
    return prims.where(pred, a, b)


#
# Data Movement References
#

# 使用@register_decomposition装饰器将`aten.clone`注册为分解函数，使其能够被外部调用
# 使用@out_wrapper装饰器对clone函数进行输出包装处理
def clone(
    a: TensorLikeType,  # 定义输入参数a的类型为TensorLikeType
    *,  # 后续参数强制使用关键字方式传递
    memory_format: torch.memory_format = torch.preserve_format  # 定义内存格式参数memory_format，默认为保持原格式
) -> TensorLikeType:  # 函数返回类型为TensorLikeType

    # 调用prims模块的clone函数对输入张量a进行克隆操作，按照指定的memory_format处理
    result = prims.clone(a, memory_format=memory_format)

    return result  # 返回克隆后的结果


# 定义copy_to函数，用于将张量a复制到张量b
def copy_to(a: Tensor, b: Tensor, *, allow_cross_device=True):
    # 如果不允许跨设备拷贝，并且源张量 b 的设备与目标张量 a 的设备不同
    if not allow_cross_device and a.device != b.device:
        # 构造错误消息，指出不允许进行跨设备拷贝的尝试
        msg = f"Attempting to copy from device {b.device} to device {a.device}, but cross-device copies are not allowed!"
        # 抛出运行时错误，包含错误消息
        raise RuntimeError(msg)

    # 调用 prims 模块中的 copy_to 函数，将张量 b 复制到张量 a
    return prims.copy_to(a, b)
# 注册函数装饰器，将 `aten.item` 函数注册为 `item` 函数的分解函数
@register_decomposition(aten.item)
def item(a: TensorLikeType) -> NumberType:
    # 如果张量 `a` 的元素数量不为1，则抛出异常
    if a.numel() != 1:
        msg = f"Can't convert a tensor with {a.numel()} elements to a number!"
        raise ValueError(msg)

    # NOTE: 明确转换布尔类型是必要的！
    # 参考：https://github.com/pytorch/pytorch/issues/78071
    # 将张量的数据类型转换为对应的 Python 数字类型
    number_type = utils.dtype_to_type(a.dtype)
    # 调用底层的 `item` 函数获取张量 `a` 的数据并返回
    return number_type(prims.item(a))


# 当 `to` 方法返回输入的别名时的快速路径。这模仿了 `aten` 中相同的功能
def _to_will_alias(
    a: TensorLikeType,
    device: Optional[DeviceLikeType] = None,
    dtype: Optional[torch.dtype] = None,
    copy: Optional[bool] = None,
    layout: Optional[torch.layout] = None,
    memory_format: Optional[torch.memory_format] = None,
    pin_memory: Optional[bool] = False,
    non_blocking: bool = False,  # 不使用非阻塞模式
) -> bool:
    return (
        not copy
        and (device is None or a.device == device)
        and (dtype is None or a.dtype == dtype)
        and (layout is None or a.layout == layout)
        # 处理 `is_pinned` 问题 #84925
        # and (pin_memory is None or pin_memory == a.is_pinned())
        and (
            memory_format is None
            or memory_format == torch.preserve_format
            or utils.is_contiguous_for_memory_format(a, memory_format=memory_format)
        )
    )


# 单分派函数，当没有特定类型的处理函数时会抛出 `NotImplementedError`
@singledispatch
def _to_dispatch(*args, **kwargs):
    raise NotImplementedError


# 注册 `_to_dispatch` 的处理函数，用于将数据移到指定设备
@_to_dispatch.register
def _to_device(
    device: torch.device,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Dict[str, Any]:
    kwargs = {
        "device": device,
        "dtype": dtype,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


# 注册 `_to_dispatch` 的处理函数，用于将数据移到指定设备字符串表示形式
@_to_dispatch.register
def _to_device_str(
    device: str,
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Dict[str, Any]:
    kwargs = {
        "device": torch.device(device),
        "dtype": dtype,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


# 注册 `_to_dispatch` 的处理函数，用于将数据移到指定数据类型
@_to_dispatch.register
def _to_dtype(
    dtype: torch.dtype,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Dict[str, Any]:
    kwargs = {
        "dtype": dtype,
        "non_blocking": non_blocking,
        "copy": copy,
        "memory_format": memory_format,
    }
    return kwargs


# 注册 `_to_dispatch` 的处理函数，用于将数据移到其他张量的设备和数据类型
@_to_dispatch.register
def _to_other(
    other: Tensor,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> Dict[str, Any]:
    # 获取其他张量 `other` 的设备和数据类型信息
    device = other.device
    dtype = other.dtype
    layout = other.layout
    # 处理 `is_pinned` 问题 #84925
    # pin_memory = other.is_pinned()
    # 创建一个包含多个键值对的字典 kwargs
    kwargs = {
        "device": device,           # 键 "device" 对应传入的变量 device，表示设备类型
        "dtype": dtype,             # 键 "dtype" 对应传入的变量 dtype，表示数据类型
        "layout": layout,           # 键 "layout" 对应传入的变量 layout，表示数据布局
        "non_blocking": non_blocking,   # 键 "non_blocking" 对应传入的变量 non_blocking，表示是否非阻塞操作
        "copy": copy,               # 键 "copy" 对应传入的变量 copy，表示是否进行数据复制
        "memory_format": memory_format, # 键 "memory_format" 对应传入的变量 memory_format，表示内存格式
    }
    # 返回包含各个参数设置的字典 kwargs
    return kwargs
# 从 `to_kwargs` 中移除已经存在于 `a` 中的 `to_kwargs` 参数
def _canonicalize_to_arguments(a: Tensor, to_kwargs: dict):
    # 要检查的选项列表
    options_to_check = ["dtype", "device", "layout", "memory_format"]
    
    # 如果 `to_kwargs` 中包含 "device" 并且其类型为 str，则转换为 torch.device 对象
    if "device" in to_kwargs and isinstance(to_kwargs["device"], str):
        to_kwargs["device"] = torch.device(to_kwargs["device"])

    # 遍历待检查的选项
    for kw in options_to_check:
        if kw in to_kwargs:
            # 处理不同的选项情况
            if (
                (kw == "memory_format" and to_kwargs[kw] is torch.preserve_format)
                or (
                    kw == "device"
                    and to_kwargs[kw].type == a.device.type
                    and (
                        not to_kwargs[kw].index or to_kwargs[kw].index == a.device.index
                    )
                )
                or (
                    getattr(a, kw, None) == to_kwargs[kw]
                )  # 这也处理了 {"memory_format": None} 的情况
            ):
                # 如果匹配条件，则从 `to_kwargs` 中删除该选项
                to_kwargs.pop(kw)


# 将输入 `a` 转换为张量，并应用指定的参数和选项
def to(a: TensorLikeType, *args, **kwargs) -> TensorLikeType:
    # 如果有位置参数，则通过 `_to_dispatch` 处理分发
    if len(args) != 0:
        kwargs = _to_dispatch(*args, **kwargs)

    # 检查是否有不支持的 `pin_memory` 参数
    assert "pin_memory" not in kwargs
    
    # 规范化 `to` 函数的参数
    _canonicalize_to_arguments(a, kwargs)

    # 如果 `_to_will_alias` 函数预测将别名化，则直接返回 `a`
    if _to_will_alias(a, **kwargs):
        return a

    # 从 `kwargs` 中获取 `copy` 和 `non_blocking` 参数，并弹出
    copy = kwargs.pop("copy") if "copy" in kwargs else False
    non_blocking = kwargs.pop("non_blocking") if "non_blocking" in kwargs else False

    # 如果只是数据类型变化且没有复制和非阻塞操作，并且没有指定 `memory_format` 和 `device` 等参数，
    # 则通过 `prims.convert_element_type` 直接转换数据类型
    if (
        (copy or (kwargs.get("dtype", a.dtype) != a.dtype))
        and (not non_blocking)
        and ("memory_format" not in kwargs)
        and ("device" not in kwargs)
        and ("layout" not in kwargs)
        # 处理 `pin_memory` 问题 #84925
        # and ("pin_memory" not in kwargs)
    ):
        return prims.convert_element_type(a, kwargs.get("dtype", a.dtype))

    # 根据 `a` 的形状创建一个空张量 `result`，并应用 `kwargs`
    result = torch.empty_like(a, **kwargs)

    # 将数据从 `a` 复制到 `result` 中
    copy_to(result, a)

    # 返回结果张量 `result`
    return result


#
# 缩减操作的参考
#


# 执行缩减操作，使用指定的 `prim` 函数
def _reduction(
    a: TensorLikeType,
    prim: Callable,
    *,
    has_identity: bool = True,
    accepts_dim_tuple: bool = True,  # 用于处理只接受单个维度的 min/argmin 等操作
    dims: Optional[DimsType] = None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,  # 对支持的操作应该指定
    out: Optional[Tensor] = None,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,
) -> TensorLikeType:  # 通常是相同的，但我希望参考作者们实际思考该放置在这里的内容
    # 断言输入 `a` 是 `TensorLike` 类型
    assert isinstance(a, TensorLike)
    
    # 如果输入张量 `a` 的维度超过 64，抛出错误
    if a.ndim > 64:
        raise RuntimeError(
            f"Received a tensor with {a.ndim} dimensions, but only tensors with up to 64 dims are supported!"
        )
    # 如果指定了输出对象 out
    if out is not None:
        # 断言 out 是 TensorLike 类型
        assert isinstance(out, TensorLike)
        # 如果指定了 dtype
        if dtype is not None:
            # 检查是否与输出对象的 dtype 匹配
            # 注意：目前在即时执行模式下是正确的，但对于复杂的规范来说，这是错误的行为
            if dtype != out.dtype:
                # 抛出运行时错误，要求在规约操作中 dtype 和 out 的 dtype 必须匹配
                raise RuntimeError(
                    "dtype argument and out dtype must match in reduction"
                )
    
    # 如果不接受维度元组
    if not accepts_dim_tuple:
        # 断言维度 dims 要么为 None，要么是 Dim 类型
        assert dims is None or isinstance(dims, Dim)
    
    # 如果 dims 是 Dim 类型，则转换为包含单一元素的元组
    if isinstance(dims, Dim):
        dims = (dims,)  # type: ignore[assignment]
    
    # 根据给定的维度 dims 计算有效的规约维度
    dims = utils.reduction_dims(a.shape, dims)
    
    # 如果没有单位元（identity），检查数组 a 是否可以在指定的维度上进行规约
    if not has_identity:
        # 验证数组 a 的形状是否是零维或者在指定维度上是否全部为非零值
        valid_shape = a.ndim == 0 or all(a.shape[i] for i in dims)
        if not valid_shape:
            # 如果存在零维的维度，在没有单位元的规约操作中抛出运行时错误
            raise RuntimeError(
                "reducing over zero-size dimension for reduction operation without identity"
            )
    
    # 确定计算时的数据类型和结果数据类型
    computation_dtype, result_dtype = utils.reduction_dtypes(
        a, output_dtype_kind, dtype
    )
    
    # 将数组 a 可能转换为计算数据类型 computation_dtype
    a = _maybe_convert_to_dtype(a, computation_dtype)  # type: ignore[method-assign]
    
    # 执行规约操作 prim，并得到结果
    result = prim(a, dims)
    
    # 如果保持维度 keepdims 为 True
    if keepdims:
        # 计算输出的形状，保留规约维度的长度为 1，其他维度与 a 的形状相同
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        # 计算需要广播的维度，即非规约维度的索引
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        # 对结果进行维度广播操作
        result = prims.broadcast_in_dim(result, output_shape, broadcast_dims)
    
    # 如果指定了输出对象 out
    if out is not None:
        # 断言结果数据类型 result_dtype 不为 None
        assert result_dtype is not None
        # 如果指定了 dtype 并且结果数据类型与 out 的 dtype 不匹配
        if dtype is not None and result_dtype != out.dtype:
            # 抛出运行时错误，要求规约结果的 dtype 与 out 的 dtype 匹配
            raise RuntimeError(
                "Expected the dtype of reduction result and out to match"
            )
        # 将结果复制到指定的输出对象 out 中
        out = _maybe_resize_out(out, result.shape)
        return _safe_copy_out(copy_from=result, copy_to=out)  # type: ignore[arg-type]
    
    # 如果结果的数据类型与期望的结果数据类型不匹配且结果数据类型不为 None
    if result.dtype != result_dtype and result_dtype is not None:
        # 将结果转换为期望的结果数据类型
        result = prims.convert_element_type(result, result_dtype)
    
    # 返回最终的规约结果
    return result
# 定义一个函数，用于生成给定视图函数的复制版本
def _make_copy_from_view(fn):
    # 获取视图函数的名称
    name = fn.__name__
    # 应用输出包装器到视图函数，返回包装后的函数
    fn = out_wrapper()(fn)

    # 定义生成的复制版本函数
    def _fn(*args, out=None, **kwargs):
        # 调用原始视图函数，并传递参数和关键字参数
        result = fn(*args, out=out, **kwargs)
        # 如果没有提供输出张量，则进行克隆并指定内存格式为连续格式
        if out is None:
            return result.clone(memory_format=torch.contiguous_format)
        return result
    
    # 构造复制版本函数的名称
    copy_name = f"{name}_copy"
    # 设置复制版本函数的名称属性
    _fn.__name__ = copy_name
    # 使用getattr从aten中获取copy_name所对应的对象，并注册分解
    _fn = register_decomposition(getattr(aten, copy_name))(_fn)
    # 返回生成的复制版本函数
    return _fn


# 保存Python内置函数all的引用
py_all = all


# 注册对aten.all的分解，并应用输出包装器
@register_decomposition(aten.all)
@out_wrapper()
# 定义all函数，接受张量样式类型a，可选的维度dim，默认为None，是否保持dim维度keepdim，默认为False，返回张量样式类型
def all(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
) -> TensorLikeType:
    # 计算所有元素的逻辑非，并取其结果
    result = torch.logical_not(torch.any(torch.logical_not(a), dim, keepdim=keepdim))

    # 如果a的数据类型为torch.uint8，则将结果转换为torch.uint8类型
    if a.dtype == torch.uint8:
        result = result.to(dtype=torch.uint8)

    # 返回计算结果
    return result


# 保存Python内置函数any的引用
py_any = any


# 注册对aten.any的分解，并应用输出包装器
@register_decomposition(aten.any)
@out_wrapper()
# 定义any函数，接受张量样式类型a，可选的维度dim，默认为None，是否保持dim维度keepdim，默认为False，返回张量样式类型
def any(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
) -> TensorLikeType:
    # 将a转换为torch.bool类型
    a_ = _maybe_convert_to_dtype(a, torch.bool)
    
    # 如果dim是列表或元组且长度为0，则复制a_
    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        result = a_.clone()
    else:
        # 在指定维度上计算和，并保持维度
        result = a_.sum(dim=dim, keepdim=keepdim).ne(False)

    # 如果a的数据类型为torch.uint8，则通过prims.convert_element_type将结果转换为torch.uint8类型
    if a.dtype is torch.uint8:
        return prims.convert_element_type(result, torch.uint8)

    # 返回计算结果
    return result


# 注册对aten.sum.dim_IntList和aten.sum.IntList_out的分解
@register_decomposition([aten.sum.dim_IntList, aten.sum.IntList_out])
# 定义sum函数，接受张量样式类型a，可选的维度dim，默认为None，是否保持dim维度keepdim，默认为False，关键字参数dtype，默认为None，输出张量out，默认为None，返回张量样式类型
def sum(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # 如果dtype为None
    if dtype is None:
        # 如果out不为None，则使用out的数据类型
        if out is not None:
            dtype = out.dtype
        # 如果a的数据类型是布尔型或整数型，则dtype为torch.int64
        elif utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
            dtype = torch.int64
        else:
            dtype = a.dtype
    
    # 如果dim为()或[]，则将dim设置为None，即减少所有维度
    if dim == () or dim == []:
        dim = None
    
    # 调用_reduction函数，进行张量a的减少操作
    return _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=out,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


# 定义sum_to_size函数，接受张量a和可变数量的形状参数，返回张量
def sum_to_size(
    a: Tensor,
    *shape,
) -> Tensor:
    # 从可变参数中提取形状，不进行验证
    shape = utils.extract_shape_from_varargs(shape, validate=False)
    # 检查形状shape是否可以扩展为张量a的形状
    torch._check(
        utils.is_expandable_to(shape, a.shape),
        lambda: f'sum_to_size: size "{shape}" is not expandable to size "{a.shape}"',
    )
    
    # 如果形状shape与张量a的形状相同且长度大于0，则返回a的视图
    if utils.is_same_shape(shape, a.shape) and len(shape) > 0:
        return prims.view_of(a)
    
    # 计算a的前导维度数量
    leading_dims = a.ndim - len(shape)
    # 根据给定条件确定需要压缩的维度，包括 leading_dims 及其后面所有维度中，对应的 shape 符合条件的维度
    reduce_dims = tuple(range(leading_dims)) + tuple(
        i
        for i in range(leading_dims, len(shape))
        if shape[i - leading_dims] == 1 and a.shape[i] != 1
    )
    # 对张量 a 按照 reduce_dims 所指定的维度进行求和操作
    # keepdim=True 表示保持输出张量的维度与输入张量一致，即使某些维度的大小为 1
    # dtype=None 表示保持输入张量的数据类型不变
    return torch.sum(a, dim=reduce_dims, keepdim=True, dtype=None)
@register_decomposition(aten.prod)
def prod(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # 如果 dtype 未指定：
    if dtype is None:
        # 如果指定了输出张量 out，则使用 out 的数据类型
        if out is not None:
            dtype = out.dtype
        # 否则，根据输入张量 a 的数据类型确定 dtype
        elif utils.is_boolean_dtype(a.dtype) or utils.is_integer_dtype(a.dtype):
            dtype = torch.int64
        else:
            dtype = a.dtype
    # 如果 dim 是空元组 () 或空列表 []，则将 dim 设为 None，表示沿着所有维度进行减少
    if dim == () or dim == []:
        dim = None
    return _reduction(
        a,
        prims.prod,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=out,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


@register_decomposition(aten.amin)
def amin(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # 如果 dim 是空元组 () 或空列表 []，则将 dim 设为 None，表示沿着所有维度进行减少
    if dim == () or dim == []:
        dim = None

    return _reduction(
        a,
        prims.amin,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=out,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


@register_decomposition(aten.amax)
def amax(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # 如果 dim 是空元组 () 或空列表 []，则将 dim 设为 None，表示沿着所有维度进行减少
    if dim == () or dim == []:
        dim = None

    return _reduction(
        a,
        prims.amax,
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=out,
        has_identity=False,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.SAME,
    )


def _dim_var_dispatch(dim=None, unbiased=None):
    # torch.var 函数有以下重载：
    # var(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
    # 需要将布尔类型的 dim 映射到 unbiased 参数
    if unbiased is None and isinstance(dim, bool):
        unbiased = dim
        dim = None
    return dim, unbiased


@register_decomposition(aten.var)
@out_wrapper()
def var(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[NumberType] = None,
) -> TensorLikeType:
    # 调用 _dim_var_dispatch 函数处理 dim 和 unbiased 参数
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    # 根据 unbiased 参数设置修正值 correction
    correction = utils.set_correction(unbiased, correction)
    # 如果 dim 是空元组 () 或空列表 []，则将 dim 设为 None，表示沿着所有维度进行减少
    if dim == () or dim == []:
        dim = None

    # 执行 _reduction 函数进行降维操作，使用 partial 函数指定 prims.var 以及 correction 参数
    result = _reduction(
        a,
        partial(prims.var, correction=correction),
        dims=dim,
        keepdims=keepdim,
        dtype=None,
        out=None,
        has_identity=True,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    )
    return result


@register_decomposition(aten.std)
@out_wrapper()
def std(
    a: TensorLikeType,
    dim: Union[Optional[int], Optional[List[int]]] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[NumberType] = None,
# 返回类型注释，表示函数返回的对象类型为 TensorLikeType
) -> TensorLikeType:
    # 调用 _dim_var_dispatch 函数，根据传入的 dim 和 unbiased 参数进行分发处理
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    # 调用 utils.set_correction 函数，设置修正值，根据 unbiased 和 correction 参数
    correction = utils.set_correction(unbiased, correction)

    # 调用 utils.reduction_dtypes 函数，确定操作和输出的数据类型
    opmath_dtype, dtype = utils.reduction_dtypes(
        a, REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    )
    # 调用 _maybe_convert_to_dtype 函数，可能将 a 转换为 opmath_dtype 类型
    a = _maybe_convert_to_dtype(a, opmath_dtype)
    # 调用 torch.var 函数，计算张量 a 的方差，沿着维度 dim，考虑修正值 correction 和是否保持维度 keepdim
    a_var = torch.var(a, dim, correction=correction, keepdim=keepdim)
    # 计算张量 a_var 的标准差
    a_std = torch.sqrt(a_var)
    # 断言语句，确保 dtype 不为 None
    assert dtype is not None
    # 将 a_std 转换为指定的 dtype 类型，并返回结果
    return _maybe_convert_to_dtype(a_std, dtype)


# 注册 aten.mean 函数的装饰器
@register_decomposition(aten.mean)
def mean(
    # 输入张量类型注释，参数 a 为 TensorLikeType 类型
    a: TensorLikeType,
    # 可选的维度参数，指定计算平均值的维度
    dim: Optional[DimsType] = None,
    # 是否保持维度不变的布尔值，默认为 False
    keepdim: bool = False,
    # 仅限关键字参数，指定输出结果的数据类型，默认为 None
    *,
    dtype=None,
    # 可选的输出张量参数，默认为 None
    out=None,
) -> TensorLikeType:
    # 如果 dim 参数为 () 或 []，则将 dim 设置为 None，表示对所有维度进行缩减
    if dim == () or dim == []:
        dim = None
    # 保存原始的 dtype 参数值
    orig_dtype = dtype
    # 如果 dtype 参数为 None，则将其设为输入张量 a 的数据类型
    if dtype is None:
        dtype = a.dtype
    # 断言语句，确保输出张量 out 的数据类型与指定的 dtype 一致，或者 out 为 None
    torch._check(
        out is None or out.dtype == dtype,
        lambda: f"Expected out tensor to have dtype {dtype}, but got {out.dtype} instead",
    )
    # 调用 _reduction 函数，执行张量 a 的缩减操作，使用 prims.sum 作为缩减操作的原语
    result = _reduction(
        a,
        prims.sum,
        dims=dim,
        keepdims=keepdim,
        dtype=dtype,
        out=None,
        output_dtype_kind=REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE,
    )
    # 断言语句，确保 dtype 是浮点数或复数类型，用于下一步的类型推断
    torch._check(
        utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
        lambda: (
            f"mean(): could not infer output dtype. "
            f"{'Input' if orig_dtype is None else 'Optional'} dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        ),
    )
    # 如果 dim 参数是 Dim 类型的实例，则将其转换为包含单个元素的元组
    if isinstance(dim, Dim):
        dim = (dim,)  # type: ignore[assignment]
    # 调用 utils.reduction_dims 函数，根据输入张量 a 的形状和 dim 参数，获取用于缩减的维度
    dims = utils.reduction_dims(a.shape, dim)  # type: ignore[arg-type]
    # 计算张量 a 的元素个数
    nelem = 1 if a.ndim == 0 else reduce(operator.mul, (a.shape[i] for i in dims), 1)
    # 对 result 进行除法运算，计算平均值
    result = true_divide(result, nelem)
    # 确定结果的数据类型为输入张量 a 的数据类型，或者指定的 dtype 类型
    result_dtype = a.dtype if dtype is None else dtype
    # 将 result 转换为指定的结果数据类型，并返回结果
    result = _maybe_convert_to_dtype(result, result_dtype)  # type: ignore[method-assign]
    # 如果提供了输出张量 out，则进行安全的复制操作，返回复制后的结果
    if out is not None:
        assert isinstance(out, TensorLike)
        # 调用 _maybe_resize_out 函数，根据 result 的形状调整输出张量 out 的大小
        out = _maybe_resize_out(out, result.shape)
        # 返回 _safe_copy_out 函数的结果，用于将计算结果复制到输出张量 out 中
        return _safe_copy_out(copy_from=result, copy_to=out)  # type: ignore[arg-type]
    # 返回计算得到的平均值结果
    return result


# 注册 aten.std_mean 函数的装饰器
@register_decomposition(aten.std_mean)
# 输出包装器装饰器，指定函数的输出名称为 "out0" 和 "out1"
@out_wrapper("out0", "out1")
def std_mean(
    # 输入张量类型注释，参数 a 为 TensorLikeType 类型
    a: TensorLikeType,
    # 可选的维度参数，指定计算标准差和平均值的维度
    dim: Optional[DimsType] = None,
    # 仅限关键字参数，指定是否使用无偏估计，默认为 None
    *,
    unbiased: Optional[bool] = None,
    # 是否保持维度不变的布尔值，默认为 False
    keepdim: bool = False,
    # 可选的修正值参数，用于标准差计算
    correction: Optional[NumberType] = None,
):
    # 调用 _dim_var_dispatch 函数，根据传入的 dim 和 unbiased 参数进行分发处理
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    # 调用 utils.set_correction 函数，设置修正值，根据 unbiased 和 correction 参数
    correction = utils.set_correction(unbiased, correction)
    # 调用 utils.reduction_dtypes 函数，确定操作和输出的数据类型
    opmath_dtype, dtype = utils.reduction_dtypes(
        a, REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    )
    # 保存原始的输入张量 a 的数据类型
    original_dtype = a.dtype
    # 调用 _maybe_convert_to_dtype 函数，可能将 a 转换为 opmath_dtype 类型
    a = _maybe_convert_to_dtype(a, opmath_dtype)
    # 调用 torch.var_mean 函数，同时计算张量 a 的方差和平均值
    a_var, a_mean = torch.var_mean(a, dim, correction=correction, keepdim=keepdim)
    # 计算张量 a_var 的标准差
    a_std = torch.sqrt(a_var)
    # 断言语句，确保 dtype 不为 None
    assert dtype is not None
    # 返回一个元组，包含经过可能的类型转换后的标准差和均值
    return (
        _maybe_convert_to_dtype(a_std, dtype),  # 调用函数 `_maybe_convert_to_dtype` 对 a_std 进行类型转换，并返回结果
        _maybe_convert_to_dtype(a_mean, original_dtype),  # 调用函数 `_maybe_convert_to_dtype` 对 a_mean 进行类型转换，并返回结果，使用原始数据类型
    )
# 注册变量分解函数 `var_mean` 到 `aten.var_mean`
# 使用 `out_wrapper` 装饰器将输出参数命名为 "out0" 和 "out1"
def var_mean(
    a: TensorLikeType,
    dim: Optional[DimsType] = None,
    unbiased: Optional[bool] = None,
    keepdim: bool = False,
    *,
    correction: Optional[NumberType] = None,
):
    # 调用 `_dim_var_dispatch` 函数，根据参数 `dim` 和 `unbiased` 处理维度和偏差
    dim, unbiased = _dim_var_dispatch(dim, unbiased)
    # 计算张量 `a` 的方差，根据给定的维度和偏差标志
    v = var(a, dim, unbiased, keepdim, correction=correction)
    # 计算张量 `a` 的均值，根据给定的维度和保持维度标志
    m = mean(a, dim, keepdim)
    # 返回方差 `v` 和均值 `m`
    return v, m


# 注册函数 `addr` 到 `aten.addr`
# 使用 `out_wrapper` 装饰器装饰函数，指定没有输出参数
# 使用 `elementwise_type_promotion_wrapper` 装饰器，指定类型提升的参数和默认类型提升种类
def addr(
    self: TensorLikeType,
    vec1: TensorLikeType,
    vec2: TensorLikeType,
    *,
    beta: NumberType = 1,
    alpha: NumberType = 1,
) -> TensorLikeType:
    # 检查 `vec1` 的维度是否为1
    torch._check(
        vec1.ndim == 1,
        lambda: f"addr: Expected 1-D argument vec1, but got {vec1.ndim}-D",
    )
    # 检查 `vec2` 的维度是否为1
    torch._check(
        vec2.ndim == 1,
        lambda: f"addr: Expected 1-D argument vec2, but got {vec2.ndim}-D",
    )
    # 将 `self` 张量扩展为与 `vec1` 和 `vec2` 张量形状相匹配
    self = self.expand(vec1.shape[0], vec2.shape[0])
    
    # 如果 `self` 张量的数据类型为布尔类型
    if utils.is_boolean_dtype(self.dtype):
        # 对于布尔类型，接受整数类型的 `beta`
        torch._check(
            is_weakly_lesser_type(type(beta), int),
            lambda: f"expected bool/int beta but got {type(beta)}",
        )
        # 对于布尔类型，接受整数类型的 `alpha`
        torch._check(
            is_weakly_lesser_type(type(alpha), int),
            lambda: f"expected bool/int alpha but got {type(beta)}",
        )
        # 如果 `beta` 为假值，则返回 `vec1` 和 `vec2` 的外积，否则返回与 `self` 相同形状的全假值张量
        if not beta:
            return torch.outer(vec1, vec2) if alpha else torch.full_like(self, False)
        else:
            # 否则，返回 `self` 与 `vec1` 和 `vec2` 的外积逻辑或运算的结果
            return torch.logical_or(
                self,
                torch.outer(vec1, vec2) if alpha else torch.full_like(self, False),
            )
    else:
        # 对于其他数据类型，检查 `beta` 是否能安全转换为 `self` 的数据类型
        torch._check(
            is_weakly_lesser_type(type(beta), dtype_to_type(self.dtype)),
            lambda: f"cannot safely convert {type(beta)} to {self.dtype}",
        )
        # 对于其他数据类型，检查 `alpha` 是否能安全转换为 `self` 的数据类型
        torch._check(
            is_weakly_lesser_type(type(alpha), dtype_to_type(self.dtype)),
            lambda: f"cannot safely convert {type(alpha)} to {self.dtype}",
        )
        # 如果 `beta` 为零，则返回 `alpha` 乘以 `vec1` 和 `vec2` 的外积
        if beta == 0:
            return alpha * torch.outer(vec1, vec2)
        else:
            # 否则，返回 `beta` 乘以 `self` 加上 `alpha` 乘以 `vec1` 和 `vec2` 的外积
            return beta * self + alpha * torch.outer(vec1, vec2)


# 不注册分解的 `CompositeImplicitAutograd` 类
def atleast_1d(
    arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args: TensorLikeType
) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_1d`."""
    # 如果没有额外参数且 `arg` 是 `collections.abc.Sequence` 类型
    if not args and isinstance(arg, collections.abc.Sequence):
        args_ = arg
    else:
        # 否则，确保 `arg` 不是 `collections.abc.Sequence` 类型，将参数和额外参数合并
        assert not isinstance(arg, collections.abc.Sequence)
        args_ = (arg,) + args
    # 对于每个参数 `a`，如果其维度大于等于1，则不改变，否则在第0维上增加一个维度
    res = tuple(a if a.ndim >= 1 else unsqueeze(a, 0) for a in args_)
    # 如果结果元组长度大于1，则返回元组；否则返回单个元素
    return res if len(res) > 1 else res[0]
# of incompatible type passed to unsqueeze
def _unsqueeze_atleast(
    at_least_fn: Callable, dim: int, arg: TensorLikeType
) -> TensorLikeType:
    arg_ = at_least_fn(arg)  # 调用给定的函数将参数至少转换为指定形式
    assert isinstance(arg_, TensorLike)  # 断言转换后的结果是张量样式的类型
    return unsqueeze(arg_, dim)  # 在指定维度上对结果进行展开操作


# CompositeImplicitAutograd - don't register decomp
def atleast_2d(
    arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args: TensorLikeType
) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_2d`."""
    if not args and isinstance(arg, collections.abc.Sequence):
        args_ = arg  # 如果参数不是多个，并且是序列类型，则直接使用参数本身
    else:
        assert not isinstance(arg, collections.abc.Sequence)  # 断言参数不是序列类型
        args_ = (arg,) + args  # 否则将参数与额外参数合并成一个元组
    unsqueeze_atleast_1d = partial(_unsqueeze_atleast, atleast_1d, 0)  # 创建一个函数，用于至少在第一维度上展开
    res = tuple(a if a.ndim >= 2 else unsqueeze_atleast_1d(a) for a in args_)  # 对每个参数进行至少展开到二维的操作
    return res if len(res) > 1 else res[0]  # 如果结果是多个，则返回元组；否则返回单个结果


# CompositeImplicitAutograd - don't register decomp
def atleast_3d(
    arg: Union[TensorLikeType, Sequence[TensorLikeType]], *args: TensorLikeType
) -> Union[TensorLikeType, Tuple[TensorLikeType, ...]]:
    """Reference implementation of :func:`torch.atleast_3d`."""
    if not args and isinstance(arg, collections.abc.Sequence):
        args_ = arg  # 如果参数不是多个，并且是序列类型，则直接使用参数本身
    else:
        assert not isinstance(arg, collections.abc.Sequence)  # 断言参数不是序列类型
        args_ = (arg,) + args  # 否则将参数与额外参数合并成一个元组
    unsqueeze_atleast_2d = partial(_unsqueeze_atleast, atleast_2d, -1)  # 创建一个函数，用于至少在最后一维度上展开
    res = tuple(a if a.ndim >= 3 else unsqueeze_atleast_2d(a) for a in args_)  # 对每个参数进行至少展开到三维的操作
    return res if len(res) > 1 else res[0]  # 如果结果是多个，则返回元组；否则返回单个结果


def as_strided(
    a: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: Optional[int] = None,
) -> TensorLikeType:
    storage_offset_int = (
        storage_offset if storage_offset is not None else a.storage_offset()
    )  # 计算存储偏移量，如果未提供则使用张量的默认偏移量
    return prims.as_strided(a, size, stride, storage_offset_int)  # 调用底层的as_strided函数进行操作


as_strided_copy = _make_copy_from_view(as_strided)


@register_decomposition(aten.as_strided_scatter)
@out_wrapper()
def as_strided_scatter(
    input: TensorLikeType,
    src: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: Optional[int] = None,
) -> TensorLikeType:
    storage_offset_int = 0 if storage_offset is None else storage_offset  # 计算存储偏移量，如果未提供则使用0
    return prims.as_strided_scatter(input, src, size, stride, storage_offset_int)  # 调用底层的as_strided_scatter函数进行操作


def broadcast_shapes(*shapes) -> ShapeType:
    return torch.Size(_broadcast_shapes(*shapes))  # 调用_broadcast_shapes函数并返回其结果的torch.Size


@aten.broadcast_tensors.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.broadcast_tensors.default.py_impl(DispatchKey.Meta)
def broadcast_tensors(*tensors) -> List[TensorLikeType]:
    if len(tensors) == 1 and not isinstance(tensors[0], Tensor):
        tensors = tensors[0]  # 如果只有一个张量且不是Tensor类型，则直接使用它
    return list(_maybe_broadcast(*tensors, preserve_cpu_scalar_tensors=False))  # 调用_maybe_broadcast函数并返回结果列表


# CompositeImplicitAutograd - don't register decomp
def broadcast_to(a: TensorLikeType, size: ShapeType) -> TensorLikeType:
    start = len(size) - len(a.shape)  # 计算开始展开的维度位置
    dims = tuple(range(start, len(a.shape) + start))  # 创建展开的维度元组
    # 调用 prims 模块中的 broadcast_in_dim 函数，执行在给定维度上的广播操作，并返回结果
    return prims.broadcast_in_dim(a, size, dims)
    # 注册函数的装饰器，将 aten.cat 函数注册到某个分解器中
    @register_decomposition(aten.cat)
    # 对输出进行包装的装饰器，可能涉及输出处理或转换
    @out_wrapper()
    # 对元素类型进行提升的装饰器，指定类型提升的参数和方式
    @elementwise_type_promotion_wrapper(
        type_promoting_args=("tensors",),
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH,
    )
    # 定义 cat 函数，接受一系列张量和一个维度参数，返回一个张量或张量样式的结果
    def cat(tensors: TensorSequenceType, dim: int = 0) -> TensorLikeType:
        # 定义一个内部函数，用于计算输出的内存格式
        def cat_compute_output_memory_format(inputs):
            format = None
            for t in inputs:
                # 建议给定张量的内存格式
                f = utils.suggest_memory_format(t)
                # 如果建议的格式是连续格式，则直接返回
                if f == torch.contiguous_format:
                    return f
                # 如果当前格式与之前格式不同，则返回连续格式
                if format is not None and format != f:
                    return torch.contiguous_format
                format = f
            # 确保最终有一个确定的格式返回
            assert format is not None
            return format

        # 如果输入的张量序列为空，则抛出值错误异常
        if len(tensors) == 0:
            msg = "cat expects at least one tensor, but received zero!"
            raise ValueError(msg)

        # 确保所有输入张量都是张量样式的
        for tensor in tensors:
            assert isinstance(tensor, TensorLike)

        # 检查所有输入张量是否在同一个设备上，不允许混合 CPU 和 GPU 张量
        utils.check_same_device(*tensors, allow_cpu_scalar_tensors=False)

        # 导入符号形状的守卫函数，用于在分析中忽略尺寸
        from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

        # 下面的部分是关于张量维度匹配和特殊情况处理的复杂逻辑
        # 详细说明了如何处理不同维度张量和特殊情况的逻辑

        # 如果所有输入张量都是 1 维，则 example 取第一个张量作为示例
        example = None
        for i, t in enumerate(tensors):
            if example is None:
                if t.ndim != 1:
                    example = t
            else:
                if t.ndim != 1:
                    torch._check(
                        t.ndim == example.ndim,
                        lambda: "Number of dimensions of tensors must match.  "
                        f"Expected {example.ndim}-D tensors, but got {t.ndim}-D for "
                        f"tensor number {i} in the list",
                    )

        # 如果 example 仍然为 None，则所有输入张量都是 1 维的情况，随意选择第一个张量作为示例
        if example is None:
            example = tensors[0]

        # 获取示例张量的形状作为输出张量的形状
        shape = example.shape
        # 初始化一个空列表，用于存储过滤后的张量
        filtered = []
    # 遍历张量列表，获取索引和张量对象
    for tensor_idx, tensor in enumerate(tensors):
        # 检查张量的维度是否与指定的形状长度相同
        if len(shape) != len(tensor.shape):
            # 断言张量的维度为1，这一点在之前已经检查过了
            assert tensor.ndim == 1
            # 在错误消息中不建议使用旧版本行为
            torch._check(
                tensor.shape[0] == 0,
                # 如果维度不匹配，返回错误消息
                lambda: f"Number of dimensions of tensors must match.  "
                f"Expected {example.ndim}-D tensors, but got 1-D for "
                f"tensor number {tensor_idx} in the list",
            )
        else:
            # 如果张量维度为1且大小为0，则跳过该输入
            if tensor.ndim == 1 and guard_size_oblivious(tensor.shape[0] == 0):
                continue
            # 不必检查大小匹配，prims.cat 将处理
            # 将符合条件的张量添加到过滤后的列表中
            filtered.append(tensor)

    # 计算输出的内存格式
    memory_format = cat_compute_output_memory_format(tensors)

    # 如果过滤后的列表为空
    if len(filtered) == 0:
        t = tensors[0]

        # TODO: fix this to work with meta tensors
        try:
            # 检查张量列表中是否有任何张量需要梯度
            requires_grad = any(x.requires_grad for x in tensors)
        except Exception:
            requires_grad = False

        # 返回一个空张量，与第一个张量的数据类型、设备、梯度需求和内存格式匹配
        return empty(
            (0,),
            dtype=t.dtype,
            device=t.device,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    # 规范化维度索引
    dim = utils.canonicalize_dim(filtered[0].ndim, dim)
    # 验证维度索引的有效性
    utils.validate_idx(filtered[0].ndim, dim)

    # 使用 prims.cat 函数在指定维度上拼接过滤后的张量，并克隆指定的内存格式
    return prims.cat(filtered, dim).clone(memory_format=memory_format)
# 带有装饰器 @out_wrapper() 的函数，用于将多个张量按列堆叠
@out_wrapper()
def column_stack(tensors: TensorSequenceType) -> TensorLikeType:
    # 将输入的张量序列中每个张量进行对齐处理，如果张量维度小于等于1，则将其转换为二维
    aligned_tensors = tuple(
        x if x.ndim > 1 else x.reshape((x.numel(), 1)) for x in tensors
    )
    # 使用 torch.cat 将对齐后的张量按列堆叠在一起
    return cat(aligned_tensors, 1)


# 函数用于获取张量的共轭，如果张量不是复数类型，则直接返回原张量
def conj(input: TensorLikeType) -> TensorLikeType:
    # 检查输入张量是否为复数类型
    if not utils.is_complex_dtype(input.dtype):
        return input
    # 如果输入张量是稀疏张量，则调用 torch.conj_physical 处理
    if input.is_sparse:
        return torch.conj_physical(input)
    # 否则调用 prims.conj 处理
    return prims.conj(input)


# 使用 @register_decomposition(aten.constant_pad_nd) 装饰器注册的函数，模拟 at::constant_pad_nd 函数的功能
@register_decomposition(aten.constant_pad_nd)
@out_wrapper()
def constant_pad_nd(
    input: TensorLikeType, pad: List[int], value: NumberType = 0
) -> TensorLikeType:
    # 检查 pad 列表长度是否为偶数
    torch._check(
        len(pad) % 2 == 0,
        lambda: f"Length of pad must be even but instead it equals {len(pad)}",
    )

    # 获取输入张量的形状信息
    input_sizes = input.shape
    l_inp = len(input_sizes)

    # 计算 pad 列表一半的长度及其差值
    l_pad = len(pad) // 2
    l_diff = l_inp - l_pad

    # 检查 pad 列表长度是否不超过输入张量维度的两倍
    torch._check(
        l_inp >= l_pad,
        lambda: "Length of pad should be no more than twice the number of "
        f"dimensions of the input. Pad length is {len(pad)} while the input has "
        f"{l_inp} dimensions.",
    )

    # 初始化变量 c_input 为输入张量
    c_input = input
    # 遍历输入张量的后几个维度进行填充处理
    for i in range(l_diff, l_inp):
        pad_idx = 2 * (l_inp - i - 1)
        # 如果负向填充小于0，则使用 narrow 函数对张量进行切片
        if pad[pad_idx] < 0:
            c_input = c_input.narrow(i, -pad[pad_idx], c_input.shape[i] + pad[pad_idx])

        # 如果正向填充小于0，则同样使用 narrow 函数对张量进行切片
        if pad[pad_idx + 1] < 0:
            c_input = c_input.narrow(i, 0, c_input.shape[i] + pad[pad_idx + 1])

    # 如果所有的填充值都不为正，则直接返回 c_input 的克隆
    if builtins.all(p <= 0 for p in pad):
        return c_input.clone()

    # 计算新的输出形状
    new_shape = list(input_sizes[:l_diff])

    # 遍历填充列表，计算每个维度的新大小
    for i in range(l_pad):
        pad_idx = len(pad) - ((i + 1) * 2)
        new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1]
        # 检查新的维度大小是否大于0
        torch._check(
            new_dim > 0,
            lambda: f"The input size {input_sizes[l_diff + i]}, plus negative padding "
            f"{pad[pad_idx]} and {pad[pad_idx + 1]} resulted in a negative output size, "
            f"which is invalid. Check dimension {l_diff + i} of your input.",
        )
        new_shape.append(new_dim)

    # 建议使用的内存格式
    memory_format = utils.suggest_memory_format(input)
    # 根据计算得到的新形状创建空的输出张量
    output = torch.empty(
        new_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
        memory_format=memory_format,
    )

    # 如果 value 等于 0 且输入张量的数据类型为 torch.bool，则将 value 设置为 False
    if value == 0 and input.dtype == torch.bool:
        value = False
    # 使用 torch.fill 填充张量，注意此处不允许复数值
    output = torch.fill(output, value)  # type: ignore[arg-type]

    # 将输出张量赋值给变量 c_output
    c_output = output
    # 遍历范围为 l_diff 到 l_inp 的索引值
    for i in range(l_diff, l_inp):
        # 计算当前索引对应的 pad 索引
        pad_idx = 2 * (l_inp - i - 1)
        # 如果 pad 中的第 pad_idx 位置的值大于 0
        if pad[pad_idx] > 0:
            # 对 c_output 进行裁剪操作，去除开头的 pad[pad_idx] 个元素
            c_output = c_output.narrow(
                i, pad[pad_idx], c_output.shape[i] - pad[pad_idx]
            )
        # 如果 pad 中的第 pad_idx + 1 位置的值大于 0
        if pad[pad_idx + 1] > 0:
            # 对 c_output 进行裁剪操作，去除结尾的 pad[pad_idx + 1] 个元素
            c_output = c_output.narrow(i, 0, c_output.shape[i] - pad[pad_idx + 1])

    # 将 c_input 的数据复制到 c_output 中
    prims.copy_to(c_output, c_input)
    # 返回结果 output
    return output
def contiguous(
    a: Tensor, *, memory_format: torch.memory_format = torch.contiguous_format
) -> Tensor:
    # 检查是否选择了保留内存格式，这是不支持的操作
    torch._check(
        memory_format != torch.preserve_format,
        lambda: "preserve memory format is unsupported by the contiguous operator",
    )

    # 如果张量已经符合所选的内存格式要求，则直接返回
    if utils.is_contiguous_for_memory_format(a, memory_format=memory_format):
        return a

    # 否则，通过克隆张量并使用指定的内存格式来创建一个连续的副本
    return torch.clone(a, memory_format=memory_format)


@out_wrapper()
def dstack(tensors: TensorSequenceType) -> TensorLikeType:
    # 检查输入的张量序列不为空
    torch._check(len(tensors) > 0, lambda: "dstack expects a non-empty TensorList")
    # 将所有张量至少转换为三维，然后沿第二维度进行拼接
    aligned_tensors = atleast_3d(*tensors)
    return cat(aligned_tensors, 2)


@register_decomposition(aten.expand)
def expand(a: Tensor, *shape) -> Tensor:
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 注意: 这里不能使用 utils.extract_shape_from_varargs
    # 因为它还会验证形状的有效性，但 expand 的形状可能是“无效”的
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])

    # 检查请求的形状维度不少于当前张量的维度
    torch._check(
        len(shape) >= len(a.shape),
        lambda: "expand: the requested shape has too few dimensions!",
    )

    offset = len(shape) - len(a.shape)
    shape_ = list(shape)
    for idx, x in enumerate(a.shape):
        offset_idx = idx + offset
        requested_length = shape[offset_idx]
        # 检查每个维度是否可以进行扩展
        torch._check(
            guard_size_oblivious(requested_length == x)
            or guard_size_oblivious(x == 1)
            or requested_length == -1,
            lambda: f"expand: attempting to expand a dimension of length {x}!",
        )

        shape_[offset_idx] = requested_length if requested_length != -1 else x

    # 确保最终的形状是有效的
    utils.validate_shape(shape_)

    # 使用广播操作在指定维度上扩展张量
    return prims.broadcast_in_dim(
        a, shape_, tuple(range(offset, len(a.shape) + offset))
    )


# CompositeImplicitAutograd - 不注册分解
def expand_as(a: Tensor, b: Tensor) -> Tensor:
    # 将张量 a 按照张量 b 的形状进行扩展
    return a.expand(b.shape)


def chunk(a: TensorLikeType, chunks: int, dim: int = 0) -> Tuple[TensorLikeType, ...]:
    if chunks <= 0:
        msg = f"Expected at least one chunk, but got {chunks}!"
        raise ValueError(msg)

    # 规范化维度索引
    dim = utils.canonicalize_dim(a.ndim, dim)
    length = a.shape[dim]
    chunk_size = math.ceil(length / chunks)
    full_chunks = math.floor(length / chunk_size)
    tail_chunk_size = length % chunk_size

    result = []
    for i in range(full_chunks):
        # 在指定维度上截取块大小的子张量，并加入结果列表中
        result.append(narrow(a, dim, i * chunk_size, chunk_size))

    # 如果有尾部块，也加入结果列表中
    if tail_chunk_size != 0:
        result.append(narrow(a, dim, full_chunks * chunk_size, tail_chunk_size))

    # 返回所有分块后的张量组成的元组
    return tuple(result)


# 注意: flatten 和其他形状操作不同，如果是对 0 维张量进行压平，则返回 1 维张量
# CompositeImplicitAutograd - 不注册分解
def flatten(a: TensorLikeType, start_dim: int = 0, end_dim: int = -1) -> TensorLikeType:
    # 使用 utils 模块的函数 canonicalize_dim 规范化起始维度，确保在有效范围内
    start_dim = utils.canonicalize_dim(a.ndim, start_dim)
    # 使用 utils 模块的函数 canonicalize_dim 规范化结束维度，确保在有效范围内
    end_dim = utils.canonicalize_dim(a.ndim, end_dim)

    # 如果起始维度和结束维度相同，并且数组的维度不为 0，则直接返回原数组，无需操作
    if start_dim == end_dim and a.ndim != 0:
        return a

    # 尝试创建一个视图
    # TODO: 可以考虑在这里指导 collapse_view 跳过其元函数 (unsafe_collapse_view)
    # 使用 prims 模块的 _collapse_view_helper 函数尝试合并视图的形状和步长
    new_shape, new_strides = prims._collapse_view_helper(a, start_dim, end_dim)
    # 如果成功创建视图，则返回视图
    if new_shape is not None:
        return prims.collapse_view(a, start_dim, end_dim)

    # 如果无法创建视图，则复制数组
    return prims.collapse(a, start_dim, end_dim)
# 将函数注册为对 aten.flip 的分解器
# 应用输出包装器装饰器
@register_decomposition(aten.flip)
@out_wrapper()
def flip(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    # 如果 dims 不是 tuple 或 list 类型，则引发数值错误
    if not isinstance(dims, tuple) and not isinstance(dims, list):
        raise ValueError("dims has to be a sequence of ints")
    # 规范化 dims 到有效的维度元组
    dims = utils.canonicalize_dims(a.ndim, dims)  # type: ignore[assignment]
    # 验证 dims 中没有重复的维度
    utils.validate_no_repeating_dims(dims)
    # 调用 prims.rev 函数进行反转操作
    return prims.rev(a, dims)


# CompositeImplicitAutograd - 不注册分解器
def fliplr(a: TensorLikeType) -> TensorLikeType:
    # 如果输入张量的维度小于 2，则引发运行时错误
    if a.ndim < 2:
        raise RuntimeError("Input must be >= 2-d.")
    # 调用 flip 函数进行左右翻转操作
    return flip(a, (1,))


# CompositeImplicitAutograd - 不注册分解器
def flipud(a: TensorLikeType) -> TensorLikeType:
    # 如果输入张量的维度小于 1，则引发运行时错误
    if a.ndim < 1:
        raise RuntimeError("Input must be >= 1-d.")
    # 调用 flip 函数进行上下翻转操作
    return flip(a, (0,))


# CompositeImplicitAutograd - 不注册分解器
def narrow(
    a: TensorLikeType, dim: int, start: Union[int, TensorLikeType], length: int
) -> TensorLikeType:
    # 支持 XLA 添加的对张量重载的支持
    if isinstance(start, TensorLike):
        # 检查 start 是否为零维整数张量
        torch._check(
            start.dim() == 0 and utils.is_integer_dtype(start.dtype),
            lambda: "start must be an 0-dim integral Tensor.",
        )
        start = start.item()  # type: ignore[assignment]
    # 检查张量 a 是否大于零维
    torch._check(a.dim() > 0, lambda: "narrow() cannot be applied to a 0-dim tensor.")
    # 检查长度是否为非负数
    torch._check(length >= 0, lambda: "narrow(): length must be non-negative.")
    # 规范化 dim 到有效的维度索引
    dim = utils.canonicalize_dim(a.ndim, dim)
    # 获取 dim 维度的长度
    dim_length = a.size(dim)
    # 检查 start 是否在有效范围内
    torch._check_with(
        IndexError,
        -dim_length <= start and start <= dim_length,  # type: ignore[arg-type]
        lambda: f"start out of range (expected to be in range of [{-dim_length}, {dim_length}], but got {start})",
    )
    # 如果 start 为负数，则将其转换为非负索引
    if start < 0:
        start = start + dim_length
    # 检查 start + length 是否超出维度大小范围
    torch._check(
        start <= dim_length - length,  # type: ignore[arg-type]
        lambda: f"start ({start}) + length ({length}) exceeds dimension size ({dim_length}).",
    )
    # 调用 prims.slice_in_dim 函数对 dim 维度进行切片操作
    return prims.slice_in_dim(a, start, start + length, axis=dim)


# TODO: 如果输入是稀疏张量，此函数必须返回稀疏张量，但 refs 不支持稀疏张量。参见 core 中的 narrow_copy_sparse 函数。
# 将 narrow 函数包装为从视图复制的函数
narrow_copy = _make_copy_from_view(narrow)


def _normalize(
    a: Tensor, norm_dims: DimsType, eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """计算张量沿 norm_dims 的均值和 1/std。

    用作标准化层的辅助函数。

    Args:
        a (Tensor): 输入张量
        norm_dims (DimsType): 要沿其进行标准化的维度
        eps (float): 数值稳定性的 epsilon

    Returns:
        out (Tensor): 标准化后的张量。
        mean (Tensor): 沿 norm_dims 的张量均值。
        rstd (Tensor): 沿 norm_dims 的张量的 1/std。
    """
    # 规范化 norm_dims 到有效的维度元组
    norm_dims = utils.canonicalize_dims(a.ndim, norm_dims)
    # 获取计算的数据类型
    computation_dtype = utils.get_computation_dtype(a.dtype)
    # 将变量 a 转换为指定的计算数据类型，返回结果赋给 a_acc
    a_acc = _maybe_convert_to_dtype(a, computation_dtype)
    # 使用 assert 断言确保 a_acc 是 TensorLike 类型，避免 mypy 错误（这与 var_mean 有关）
    assert isinstance(a_acc, TensorLike)
    # 计算在指定维度上的有偏方差（biased_var）和均值（mean）
    biased_var, mean = torch.var_mean(
        a_acc, dim=norm_dims, unbiased=False, keepdim=True
    )
    # 计算 biased_var 加上一个微小的值 eps 的平方根倒数（rstd）
    rstd = torch.rsqrt(biased_var + eps)
    # 使用归一化后的均值和 rstd 对变量 a 进行标准化处理
    out = (a - mean) * rstd
    # 返回标准化后的结果 out，以及计算得到的均值 mean 和 rstd
    return out, mean, rstd
# 添加所有指定的维度
def _unsqueeze_multiple(x: TensorLikeType, dimensions: List[int]) -> TensorLikeType:
    # 对给定的维度列表进行排序
    for dim in sorted(dimensions):
        # 使用 PyTorch 的 unsqueeze 函数在指定维度上增加维度
        x = torch.unsqueeze(x, dim)
    return x


# 将函数注册为 aten.native_group_norm.default 的分解函数
def native_group_norm(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    batch_size: int,
    num_channels: int,
    flattened_inner_size: int,
    num_groups: int,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    # 检查输入张量的维度是否至少为2
    torch._check(
        input.ndim >= 2,
        lambda: f"Expected at least 2 dimensions for input tensor but received {input.ndim}",
    )
    # 检查输入的通道数是否能被 num_groups 整除
    torch._check(
        num_channels % num_groups == 0,
        lambda: "Expected number of channels in input to be divisible by num_groups, "
        + f"but got input of shape {input.shape} and num_groups = {num_groups}",
    )

    # reduction_dims 中的维度会被用于压缩
    reduction_dims = [2, 3]
    # 将输入张量重塑为指定形状
    input_reshaped = torch.reshape(
        input,
        [batch_size, num_groups, num_channels // num_groups, flattened_inner_size],
    )
    # 对重塑后的张量进行归一化处理
    out, mean, rstd = _normalize(input_reshaped, reduction_dims, eps)
    # 将输出张量 out 重新调整为原始输入张量的形状
    out = out.view(input.shape)

    # 需要广播的维度
    broadcast_dims = [0] + list(range(2, input.ndim))
    unsqueeze_bias = None
    # 如果存在偏置项 bias，则在指定维度上增加维度
    if bias is not None:
        unsqueeze_bias = _unsqueeze_multiple(bias, broadcast_dims)
    unsqueeze_weight = None
    # 如果存在权重 weight，则在指定维度上增加维度
    if weight is not None:
        unsqueeze_weight = _unsqueeze_multiple(weight, broadcast_dims)

    # 如果权重存在，则将输出张量 out 乘以权重
    if unsqueeze_weight is not None:
        out = out * unsqueeze_weight
    # 如果偏置项存在，则将输出张量 out 加上偏置项
    if unsqueeze_bias is not None:
        out = out + unsqueeze_bias

    # 将输出张量 out、均值 mean 和标准差 rstd 转换为输入张量的数据类型
    out = _maybe_convert_to_dtype(out, input.dtype)  # type: ignore[assignment]
    mean = _maybe_convert_to_dtype(mean, input.dtype)  # type: ignore[assignment]
    rstd = _maybe_convert_to_dtype(rstd, input.dtype)  # type: ignore[assignment]

    # 从均值 mean 和标准差 rstd 中移除广播维度
    mean = torch.squeeze(mean, reduction_dims)
    rstd = torch.squeeze(rstd, reduction_dims)
    return (out, mean, rstd)


# 将函数注册为 aten.native_layer_norm 的分解函数，并将输出指定为 out0、out1 和 out2
@out_wrapper("out0", "out1", "out2")
def native_layer_norm(
    input: Tensor,
    normalized_shape: ShapeType,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    # 获取规范化形状的维度数
    normalized_ndim = len(normalized_shape)
    # 检查规范化形状的维度是否至少为1
    torch._check(
        normalized_ndim >= 1,
        lambda: "Expected normalized_shape to be at least 1-dimensional, i.e., "
        + "containing at least one element, but got normalized_shape = "
        + str(normalized_shape),
    )

    # 由于 torch.Size([1, 2, 3]) == [1, 2, 3] 为假，而 torch.Size([1, 2, 3]) == (1, 2, 3) 为真
    # 因此我们使用 tuple(normalized_shape) 来确保 normalized_shape 是元组类型
    # （而不是列表类型），以便进行后续的比较和操作
    tuple(normalized_shape)
    # 检查权重是否为None或者其形状与normalized_shape相同
    torch._check(
        weight is None or weight.shape == tuple(normalized_shape),
        lambda: "Expected weight to be of same shape as normalized_shape, but got "
        + "weight of shape "
        + str(weight.shape)  # type: ignore[union-attr]
        + " and normalized_shape = "
        + str(normalized_shape),
    )
    # 检查偏置是否为None或者其形状与normalized_shape相同
    torch._check(
        bias is None or bias.shape == tuple(normalized_shape),
        lambda: "Expected bias to be of same shape as normalized_shape, but got "
        + "bias of shape "
        + str(bias.shape)  # type: ignore[union-attr]
        + " and normalized_shape = "
        + str(normalized_shape),
    )
    # 检查输入张量的维度是否大于等于normalized_ndim，并且最后几维是否与normalized_shape相同
    torch._check(
        input.ndim >= normalized_ndim
        and input.shape[(input.ndim - normalized_ndim) :] == tuple(normalized_shape),
        lambda: "Given normalized_shape="
        + str(normalized_shape)
        + ", expected input with shape "
        + str(normalized_shape)
        + ", but got input of size "
        + str(input.shape),
    )

    # 使输入张量连续化
    input = input.contiguous()
    # 如果权重不为None，使权重张量连续化
    if weight is not None:
        weight = weight.contiguous()
    # 如果偏置不为None，使偏置张量连续化
    if bias is not None:
        bias = bias.contiguous()

    # 确定规范化操作所涉及的轴
    axis = input.ndim - normalized_ndim
    reduction_dims = list(range(axis, input.ndim))
    # 进行归一化处理，并返回结果以及均值和标准差
    out, mean, rstd = _normalize(input, reduction_dims, eps)

    # 根据权重和偏置对结果进行进一步处理
    if weight is None and bias is not None:
        out = out + bias
    elif weight is not None and bias is None:
        out = out * weight
    elif weight is not None and bias is not None:
        out = out * weight + bias

    # 将结果张量转换为与输入张量相同的数据类型
    out = _maybe_convert_to_dtype(out, input.dtype)  # type: ignore[assignment]
    # 如果输入张量位于CPU上，将均值和标准差张量转换为与输入张量相同的数据类型
    if input.device.type == "cpu":
        mean = _maybe_convert_to_dtype(mean, input.dtype)  # type: ignore[assignment]
        rstd = _maybe_convert_to_dtype(rstd, input.dtype)  # type: ignore[assignment]
    # 返回处理后的结果：归一化后的输出、均值和标准差
    return (out, mean, rstd)
# 在调试模式下，将此函数作为元函数添加会导致functorch测试失败。
# test/test_eager_transforms.py::TestFunctionalizeCPU::test_functionalize_fx_transpose_simple_cpu

# 将函数注册为aten.permute的分解函数
@register_decomposition(aten.permute)
def permute(a: TensorLikeType, *dims) -> TensorLikeType:
    # 规范化维度参数，提取可变参数中的维度
    _permutation = utils.canonicalize_dims(
        a.ndim, utils.extract_dims_from_varargs(dims)
    )
    # 调用prims中的transpose函数进行转置操作
    return prims.transpose(a, _permutation)


# 将函数注册为aten.renorm的分解函数，并添加输出包装器
@register_decomposition(aten.renorm)
@out_wrapper()
def renorm(
    input: TensorLikeType, p: RealNumberType, dim: int, maxnorm: RealNumberType
) -> TensorLikeType:
    # 检查p是否为实数值
    torch._check(not isinstance(p, complex), lambda: "renorm: p must be real-valued")
    # 检查p是否大于0
    torch._check(p > 0, lambda: "renorm: non-positive norm not supported")
    # 检查maxnorm是否为实数值
    torch._check(
        not isinstance(maxnorm, complex), lambda: "renorm: maxnorm must be real-valued"
    )
    # 检查maxnorm是否大于等于0
    torch._check(
        maxnorm >= 0, lambda: f"renorm: expected maxnorm to be >= 0 but got {maxnorm}"
    )
    # 获取输入张量的维度
    ndim = input.ndim
    # 检查输入张量的维度是否大于1
    torch._check(
        ndim > 1,
        lambda: f"renorm: input needs at least 2 dimensions, got {ndim} dimensions",
    )

    # 规范化dim参数
    dim = utils.canonicalize_dim(ndim, dim)
    # 创建要减少的维度列表，排除dim维度
    reduce_dims = list(range(ndim))
    del reduce_dims[dim]

    # 根据输入张量的数据类型确定计算精度
    acc_type = utils.get_computation_dtype(input.dtype)
    # 如果精度类型不等于输入张量的数据类型，则使用浮点精度计算范数并转换
    if acc_type != input.dtype:
        norm = torch.linalg.vector_norm(
            input, p, reduce_dims, keepdim=True, dtype=acc_type
        )
    else:
        norm = torch.linalg.vector_norm(input, p, reduce_dims, keepdim=True)

    # 设置一个极小值eps
    eps = 1e-7
    # 计算规范化因子，确保范数不超过maxnorm
    norm_factor = torch.where(norm > maxnorm, maxnorm / (norm + eps), 1.0)
    # 如果计算精度类型不等于输入张量的数据类型，则将规范化因子转换为输入张量的数据类型
    if acc_type != input.dtype:
        norm_factor = prims.convert_element_type(norm_factor, input.dtype)
    # 返回经过规范化处理的输入张量，并确保其是连续的
    return (input * norm_factor).contiguous()


# CompositeImplicitAutograd - 不要注册分解函数
@aten.stft.center.py_impl(DispatchKey.CompositeImplicitAutograd)
def stft(
    input: Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[Tensor] = None,
    center: bool = True,
    pad_mode: str = "reflect",
    normalized: bool = False,
    onesided: Optional[bool] = None,
    return_complex: Optional[bool] = None,
) -> Tensor:
    # 检查window张量和input张量必须在相同设备上
    torch._check(
        window is None or window.device == input.device,
        lambda: (
            f"stft input and window must be on the same device but got self on {input.device}"
            + f" and window on {window.device}"  # type: ignore[union-attr]
        ),
    )

    # 如果未提供hop_length，默认为n_fft的四分之一
    hop_length_ = hop_length if hop_length is not None else n_fft // 4
    # 如果未提供win_length，默认为n_fft
    win_length_ = win_length if win_length is not None else n_fft
    # 如果 return_complex 参数为 None，则根据输入是否复数和 window 的数据类型来确定 return_complex_
    return_complex_ = input.is_complex() or (
        window is not None and utils.is_complex_dtype(window.dtype)
    )
    # 检查 return_complex_ 的条件是否满足，如果不满足则抛出异常
    torch._check(
        return_complex_,
        (
            "stft requires the return_complex parameter be given for real inputs, "
            + "and will further require that return_complex=True in a future PyTorch release."
        ),
    )
    # 否则，使用指定的 return_complex 参数
    else:
        return_complex_ = return_complex

    # 检查输入的数据类型是否为浮点数或复数
    torch._check(
        utils.is_float_dtype(input.dtype) or utils.is_complex_dtype(input.dtype),
        lambda: "stft expected a tensor of floating point or complex values",
    )
    # 检查输入张量的维度是否在 1 到 2 之间
    torch._check(1 <= input.ndim <= 2, lambda: "stft expected a 1D or 2D tensor")

    # 记录原始输入张量的维度
    original_ndim = input.ndim
    # 如果原始输入张量是 1 维，则在最前面添加一个维度使其变为 2 维
    if original_ndim == 1:
        input = input.unsqueeze(0)

    # 如果 center 参数为 True，则在输入张量的末尾维度上进行填充
    if center:
        # 计算需要填充的数量
        extra_dims = 3 - input.ndim
        pad_amount = n_fft // 2
        # 构造填充后的形状
        extended_shape = [*itertools.repeat(1, extra_dims), *input.shape]
        # 执行填充操作，并根据指定的 pad_mode 进行填充
        input = aten.pad(input.view(extended_shape), [pad_amount, pad_amount], pad_mode)
        # 恢复原始的输入形状
        input = input.view(input.size()[extra_dims:])

    # 获取输入张量的批量大小和长度
    batch = input.size(0)
    length = input.size(1)
    # 检查 n_fft 参数是否在 0 到 length 之间
    torch._check(
        0 < n_fft <= length,
        lambda: f"stft expected 0 < n_fft <= {length}, but got n_fft={n_fft}",
    )
    # 检查 hop_length_ 参数是否大于 0
    torch._check(
        hop_length_ > 0,
        lambda: f"stft expected hop_length > 0 but got hop_length={hop_length_}",
    )
    # 检查 win_length_ 参数是否在 0 到 n_fft 之间
    torch._check(
        0 < win_length_ <= n_fft,
        lambda: f"stft expected 0 < win_length <= n_fft but got win_length={win_length_}",
    )
    # 检查 window 张量是否为 None 或者其形状是否与 win_length_ 相符
    torch._check(
        window is None or window.shape == (win_length_,),
        lambda: (
            f"expected a 1D window tensor of size equal to win_length={win_length_}, "
            + f"but got window with size {window.shape}"  # type: ignore[union-attr]
        ),
    )

    # 如果 win_length_ 小于 n_fft，则进行相应的处理
    if win_length_ < n_fft:
        if window is None:
            # 如果 window 为 None，则创建一个形状为 win_length_ 的全 1 张量
            window = torch.ones(win_length_, dtype=input.dtype, device=input.device)
        # 计算左侧需要填充的数量
        left = (n_fft - win_length_) // 2
        # 对 window 张量进行常数填充操作
        window = aten.constant_pad_nd(window, [left, n_fft - win_length_ - left])

    # 使用 unfold 方法对输入张量在末尾维度上进行展开操作，以便进行后续的短时傅里叶变换
    input = input.unfold(dimension=-1, size=n_fft, step=hop_length_)
    # 如果 window 不为 None，则对展开后的输入张量乘以 window 张量
    if window is not None:
        input = input * window

    # 检查输入张量的数据类型是否为复数
    complex_fft = utils.is_complex_dtype(input.dtype)
    # 根据是否为复数类型和是否指定 onesided 参数来确定是否执行单边 FFT
    onesided = onesided if onesided is not None else not complex_fft
    # 根据 normalized 参数确定是否进行正规化
    norm = "ortho" if normalized else None
    # 如果需要单边输出，则检查是否能执行单边 FFT
    if onesided:
        torch._check(
            not complex_fft,
            lambda: "Cannot have onesided output if window or input is complex",
        )
        # 执行单边实部 FFT
        out = torch.fft.rfft(input, dim=-1, norm=norm)
    else:
        # 执行完整 FFT
        out = torch.fft.fft(input, dim=-1, norm=norm)

    # 将输出张量的第 1 和第 2 维度进行转置操作
    out.transpose_(1, 2)

    # 如果原始输入张量是 1 维，则去除输出张量的第 0 维度，恢复为 1 维
    if original_ndim == 1:
        out = out.squeeze_(0)

    # 如果 return_complex_ 为 True，则返回复数形式的输出，否则返回实部
    return out if return_complex_ else torch.view_as_real(out)
# CompositeImplicitAutograd - don't register decomp
# 定义 istft 函数，用于计算短时傅里叶逆变换
@aten.istft.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def istft(
    input: Tensor,
    n_fft: int,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: Optional[Tensor] = None,
    center: bool = True,
    normalized: bool = False,
    onesided: Optional[bool] = None,
    length: Optional[int] = None,
    return_complex=False,
) -> Tensor:
    # 检查输入张量和窗口张量是否在同一设备上
    torch._check(
        window is None or window.device == input.device,
        lambda: (
            f"istft input and window must be on the same device but got self on {input.device}"
            + f" and window on {window.device}"  # type: ignore[union-attr]
        ),
    )

    # 如果未指定 hop_length，则默认为 n_fft 的四分之一
    hop_length_ = hop_length if hop_length is not None else n_fft // 4
    # 如果未指定 win_length，则默认为 n_fft
    win_length_ = win_length if win_length is not None else n_fft

    # 检查输入张量是否为复数数据类型
    torch._check(
        utils.is_complex_dtype(input.dtype),
        lambda: (
            "istft input and window must be on the same device but got self on "
            + f"{input.device} and window on {window.device}"  # type: ignore[union-attr]
        ),
    )
    
    # 获取输入张量的帧数和 FFT 大小
    n_frames = input.size(-1)
    fft_size = input.size(-2)

    # 计算期望的输出信号长度
    expected_output_signal_len = n_fft + hop_length_ * (n_frames - 1)
    # 检查输入张量是否为空
    torch._check(input.numel() > 0, lambda: "istft input tensor cannot be empty")
    # 检查输入张量维度是否在2到3之间
    torch._check(
        2 <= input.ndim <= 3,
        lambda: f"istft expected a tensor with 2 or 3 dimensions, but got {input.ndim}",
    )
    
    # 如果未指定 onesided，则根据 fft_size 和 n_fft 判断是否为单侧频谱
    onesided_ = onesided if onesided is not None else fft_size != n_fft
    
    # 如果是单侧频谱，检查频谱维度是否匹配
    if onesided_:
        torch._check(
            n_fft // 2 + 1 == fft_size,
            lambda: (
                "istft expected the frequency dimension (3rd to the last) of the input tensor "
                + "to match n_fft / 2 + 1 when onesided=True, but got {fft_size}"
            ),
        )
    else:
        # 如果不是单侧频谱，检查频谱维度是否匹配
        torch._check(
            n_fft == fft_size,
            lambda: (
                "istft expected the frequency dimension (3rd to the last) of the input tensor "
                + "to match n_fft when onesided=False, but got {fft_size}",
            ),
        )

    # 检查 hop_length 和 win_length 的取值范围
    torch._check(
        0 < hop_length_ <= win_length_,
        lambda: "istft expected 0 < hop_length <= win_length",
    )
    torch._check(
        0 < win_length_ <= n_fft, lambda: "istft expected 0 < win_length <= n_fft"
    )
    
    # 检查窗口张量的形状是否合法
    torch._check(
        window is None or window.shape == (win_length_,),
        lambda: "Invalid window shape. window has to be 1D and length of `win_length`",
    )

    # 如果未提供窗口张量，则创建一个与 win_length_ 相同长度的全1张量
    if window is None:
        real_dtype = utils.corresponding_real_dtype(input.dtype)
        window_ = torch.ones(win_length_, dtype=real_dtype, device=input.device)
    else:
        window_ = window

    # 如果 win_length_ 不等于 n_fft，则根据需要对窗口张量进行填充
    if win_length_ != n_fft:
        left = (n_fft - win_length_) // 2
        window_ = aten.constant_pad_nd(window_, (left, n_fft - win_length_ - left), 0)

    # 记录原始输入张量的维度
    original_ndim = input.ndim
    # 如果输入张量是二维的，则在第0维度上添加一个维度，使其变为三维张量
    if input.ndim == 2:
        input = input.unsqueeze(0)

    # 将输入张量的第1和第2个维度进行转置
    input = input.transpose(1, 2)

    # 根据参数normalized的取值确定是否使用"ortho"作为正则化方式
    norm = "ortho" if normalized else None

    # 如果需要返回复数结果，则进行以下操作
    if return_complex:
        # 检查是否可以输出单边谱，如果不能，则抛出错误
        torch._check(
            not onesided_,
            lambda: "cannot have onesided output if window or input is complex",
        )
        # 对输入张量在最后一个维度上进行逆FFT变换
        input = torch.fft.ifft(input, dim=-1, norm=norm)
    else:
        # 检查窗口是否为None或者窗口的数据类型是否为复数，如果是，则抛出错误
        torch._check(
            window is None or not utils.is_complex_dtype(window.dtype),
            lambda: "Complex windows are incompatible with return_complex=False",
        )
        # 如果不是单边谱，则对输入张量在最后一个维度上进行裁剪，保留前n_fft//2 + 1个元素
        if not onesided_:
            input = input.narrow(dim=-1, start=0, length=n_fft // 2 + 1)
        # 对输入张量在最后一个维度上进行逆实FFT变换
        input = torch.fft.irfft(input, dim=-1, norm=norm)

    # 断言输入张量的第2个维度大小为n_fft
    assert input.size(2) == n_fft

    # 计算y_tmp，即输入张量和窗口张量按元素相乘的结果
    y_tmp = input * window_.view([1, 1, n_fft])

    # 使用aten库中的unfold_backward函数对y_tmp进行反向展开操作，生成y
    y = aten.unfold_backward(
        y_tmp,
        input_sizes=(y_tmp.size(0), expected_output_signal_len),
        dim=1,
        size=n_fft,
        step=hop_length_,
    )

    # 计算窗口的包络，即窗口的平方在扩展后按元素的展开结果
    window_envelop = aten.unfold_backward(
        window_.pow(2).expand((1, n_frames, n_fft)),
        input_sizes=(y_tmp.size(0), expected_output_signal_len),
        dim=1,
        size=n_fft,
        step=hop_length_,
    )

    # 断言生成的y和窗口包络的长度与预期的输出信号长度相等
    assert expected_output_signal_len == y.size(1)
    assert expected_output_signal_len == window_envelop.size(1)

    # 根据参数center确定起始位置start
    start = n_fft // 2 if center else 0

    # 根据参数length确定结束位置end
    if length is not None:
        end = start + length
    elif center:
        end = expected_output_signal_len - n_fft // 2
    else:
        end = expected_output_signal_len

    # 计算实际长度
    length = max(0, end - start)

    # 对y和窗口包络在第1个维度上进行裁剪，以确保长度为length
    y = y.narrow(dim=1, start=start, length=length)
    window_envelop = window_envelop.narrow(dim=1, start=start, length=length)

    # 检查窗口包络的最小值是否小于1e-11，如果是，则抛出警告
    window_envelop_lowest = window_envelop.abs().min().lt(1e-11)
    torch._check(
        not window_envelop_lowest.item(),
        lambda: "window overlap add min less than 1e-11",
    )

    # 对y进行除法运算，以得到最终的输出信号
    y = y / window_envelop

    # 如果原始输入张量的维度为2，则压缩输出张量的第0维度
    if original_ndim == 2:
        y = y.squeeze(0)

    # 如果结束位置超过了预期的输出信号长度，发出警告并在尾部填充零
    if end > expected_output_signal_len:
        warnings.warn(
            "The length of signal is shorter than the length parameter. Result is being "
            + "padded with zeros in the tail. Please check your center and hop_length settings"
        )
        y = aten.constant_pad_nd(y, (0, end - expected_output_signal_len), 0)

    # 返回处理后的输出信号y
    return y
# Get the new shape and stride after applying unfold to an input tensor
def _get_unfold_shape_stride(
    a_shape: ShapeType, a_stride: StrideType, dimension: int, size: int, step: int
):
    # Determine the number of dimensions in the input tensor
    a_ndim = len(a_shape)
    # Canonicalize the specified dimension to ensure it is within valid range
    dim = utils.canonicalize_dim(a_ndim, dimension, wrap_scalar=True)
    # Determine the maximum size along the specified dimension
    max_size = 1 if a_ndim == 0 else a_shape[dim]
    # Obtain the stride of the specified dimension
    last_stride = 1 if a_ndim == 0 else a_stride[dim]

    # Check if the specified size is within bounds for the tensor dimension
    torch._check(
        size <= max_size,
        lambda: f"Maximum size for tensor at dimension {dim} is {max_size} but size is {size}",
    )

    # Ensure that the step size is greater than zero
    torch._check(
        step > 0,
        lambda: f"Step is {step} but must be > 0",
    )

    # Create copies of the shape and stride lists and append the new size and stride
    shape = list(a_shape)
    strides = list(a_stride)
    shape.append(size)
    strides.append(last_stride)
    
    # Adjust the shape and strides based on the specified dimension
    if dim < a_ndim:
        shape[dim] = (shape[dim] - size) // step + 1
        strides[dim] *= step
    
    # Return the updated shape and strides
    return shape, strides


@register_decomposition(aten.repeat)
@out_wrapper()
def repeat(a: Tensor, *repeat_shape) -> Tensor:
    # Extract the repeat shape from variable arguments
    repeat_shape = utils.extract_shape_from_varargs(repeat_shape, validate=False)
    # Check if the number of dimensions in repeat_shape is not smaller than tensor dimensions
    torch._check(
        len(repeat_shape) >= len(a.shape),
        lambda: "repeat: Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
    )

    # If repeat_shape is empty, return a clone of the input tensor
    if len(repeat_shape) == 0:
        return torch.clone(a)

    # Determine the number of new dimensions to be added to the tensor
    num_new_dimensions = len(repeat_shape) - a.ndim
    padded_shape = [1] * num_new_dimensions
    for dim_size in a.shape:
        padded_shape.append(dim_size)

    # Compute the target shape after padding and repeating dimensions
    target_shape = tuple(
        padded_size * repeat_size
        for padded_size, repeat_size in zip(padded_shape, repeat_shape)
    )

    # Return an empty tensor if any dimension in repeat_shape is zero
    if 0 in repeat_shape:
        return torch.empty(
            target_shape,
            dtype=a.dtype,
            device=a.device,
            requires_grad=a.requires_grad,
            memory_format=utils.suggest_memory_format(a),
        )

    # Initialize shape and stride for the unfolded tensor (urtensor)
    urtensor_shape = target_shape
    urtensor_stride = utils.make_contiguous_strides_for(target_shape)
    
    # Iterate over padded dimensions to unfold and repeat each dimension
    for dim, dim_size in enumerate(padded_shape):
        # Obtain the shape and stride after unfolding and repeating the current dimension
        urtensor_shape, urtensor_stride = _get_unfold_shape_stride(
            urtensor_shape, urtensor_stride, dim, dim_size, max(dim_size, 1)
        )

    # Derive permutation order by sorting urtensor strides
    enumerated_stride = list(enumerate(urtensor_stride))
    enumerated_stride.sort(key=operator.itemgetter(1), reverse=True)
    permute_order, sorted_stride = zip(*enumerated_stride)

    # Expand the input tensor to match urtensor's shape
    repeat_xtensor = a.expand(urtensor_shape)

    # Clone the tensor to finalize the expanded dimensions
    cloned_result = torch.clone(repeat_xtensor)

    # Permute tensor axes to ensure sorted strides order
    permuted_result = cloned_result.permute(permute_order)

    # Reshape the tensor to ensure it is contiguous with the correct target shape
    return permuted_result.reshape(target_shape)
# 定义一个辅助函数 `_reshape_view_helper`，用于处理张量重塑操作，支持符号化形状和视图操作
def _reshape_view_helper(a: TensorLikeType, *shape, allow_copy: bool) -> TensorLikeType:
    # 导入必要的模块和函数
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious, sym_eq

    # 从可变参数中提取形状信息，不进行验证
    shape = utils.extract_shape_from_varargs(shape, validate=False)
    # 推断形状，处理可能包含 -1 的情况，用于自动推断维度长度
    shape = utils.infer_size(shape, a.numel())

    # 处理元素数量为零的特殊情况
    if guard_size_oblivious(a.numel() == 0):
        # 返回一个按指定形状重塑后的视图，确保视图的连续性
        return as_strided(a, shape, utils.make_contiguous_strides_for(shape))

    # 处理零维张量的特殊重塑情况
    if a.ndim == 0:
        _a = a
        for length in shape:
            assert length == 1
            _a = unsqueeze(_a, -1)
        if _a is a:
            # 返回原张量的视图
            return prims.view_of(a)
        else:
            # 返回重塑后的张量
            return _a

    # 处理重塑成零维张量的特殊情况
    if len(shape) == 0:
        _a = a
        for length in a.shape:
            assert length == 1
            _a = squeeze(_a, -1)
        if _a is a:
            # 返回原张量的视图
            return prims.view_of(a)
        else:
            # 返回重塑后的张量
            return _a

    # 处理连续张量的特殊情况
    if a.is_contiguous():
        # 处理将多维张量特殊重塑为一维张量的情况
        if len(shape) == 1 and a.ndim > 1:
            return torch.as_strided(a, [a.numel()], [1])
        # 处理将一维张量特殊重塑为二维张量的情况
        if len(shape) == 2 and a.ndim == 1:
            dim0 = shape[0]
            dim1 = shape[1]
            return torch.as_strided(a, [dim0, dim1], [dim1, 1])

    # 处理一般情况：将任意维度的张量重塑为另一种不同的维度形状

    # 注意 [重塑算法]
    # 该算法通过贪婪地从左到右构建所需的输出形状维度来工作。
    # 它通过在原始张量上从左到右累积维度的概念性方法，直到维度可以使用 prims.split_dim 构造。
    # 该算法还对尾部的挤压/展开进行了特殊处理，例如从 (5, 5) 到 (5, 5, 1) 或反之。
    #
    # 该算法不会先将原始张量展平，然后根据需要拆分维度，因为这样做会比该算法更频繁地创建副本。
    # flatten 是下面唯一可能会创建视图或副本的操作，虽然它更倾向于创建视图，但如果张量的步幅不允许视图，则有时也会创建副本。
    # 因此，该算法试图最小化展平操作。
    #
    # 注意，可能存在更好版本的该算法。可以事先识别那些可以在不创建副本的情况下展平的区域，这可能允许更少的 flatten 调用或更快的短路复制操作。
    idx = 0
    a_ = a
    for length in shape:
        # 处理尾部的 unsqueeze 操作
        if idx >= a_.ndim:
            assert length == 1
            last_dim = a_.ndim - 1
            # 注意：在这里使用 split_dim 而不是 unsqueeze 可能看起来有些愚蠢，
            # 但这是为了确保得到正确的步幅
            a_ = prims.split_dim(a_, last_dim, a_.shape[last_dim])
            idx = idx + 1
            continue

        # 跳过已经是正确长度的维度
        if guard_size_oblivious(length == a_.shape[idx]):
            idx = idx + 1
            continue

        # 收集足够多的原始维度，以便创建新的维度
        # 注意：这个累积会终止，因为我们已经验证了 a 和 shape 上面指定的元素数量是相同的
        accum = a_.shape[idx]
        end = idx
        while guard_size_oblivious(accum % length != 0):
            end = end + 1
            accum = accum * a_.shape[end]
        if end != idx:
            # 注意：在这种情况下，多个维度必须被展平以创建所需的维度
            # 这种展平就是为什么 reshape 有时会创建一个副本 -- 因为展平可能返回副本的视图

            # 检查是否可以折叠成视图，并在不能时直接复制进行 reshape
            new_shape, new_strides = prims._collapse_view_helper(a_, idx, end)
            if new_shape is None:
                if allow_copy:
                    return prims.reshape(a, shape)

                msg = f"Cannot view a tensor with shape {a.shape} and strides {a.stride()} as a tensor with shape {shape}!"
                raise ValueError(msg)

            a_ = flatten(a_, idx, end)

        # 分割（可能已展平的）维度以创建所需的长度
        if guard_size_oblivious(accum != length):
            a_ = prims.split_dim(a_, idx, length)

        idx = idx + 1

    # 压缩尾部维度
    while idx < a_.ndim:
        torch._check(
            a_.shape[idx] == 1,
            lambda: f"a.size({idx}) expected to be 1 but got {a_.shape[idx]}",
        )
        a_ = squeeze(a_, idx)

    if a_ is a:
        return prims.view_of(a)
    else:
        return a_
# CompositeImplicitAutograd - don't register decomp
# 定义 reshape 函数，用于改变张量形状
# 注意：shape 是可变参数，因为 Tensor.reshape 可以以 Tensor.reshape(a, b, c) 或 Tensor.reshape((a, b, c)) 的形式调用。torch.reshape 不支持解包的形状
def reshape(a: TensorLikeType, *shape: ShapeType) -> TensorLikeType:
    return _reshape_view_helper(a, *shape, allow_copy=True)


# CompositeImplicitAutograd - don't register decomp
# 定义 reshape_as 方法，用于将张量按照另一个张量的形状进行改变
def reshape_as(self: TensorLikeType, other: TensorLikeType) -> TensorLikeType:
    return self.reshape(other.size())


# 注册 aten.roll 的分解器，并标记为输出装饰器
@register_decomposition(aten.roll)
@out_wrapper()
def roll(
    a: TensorLikeType, shifts: DimsType, dims: DimsType = tuple()
) -> TensorLikeType:
    """Reference implementation of :func:`torch.roll`."""
    # 规范化 dims，确保在合法范围内
    dims = utils.canonicalize_dims(a.ndim, dims)
    # ATen 指定 shifts 和 dims 的类型为 int[1]，会将整数扩展为长度为 1 的元组
    if not isinstance(shifts, Iterable):
        shifts = (shifts,)
    if not isinstance(dims, Iterable):
        dims = (dims,)

    # 避免除以零
    if a.numel() == 0:
        # 对于空张量，返回其克隆
        return a.clone()

    if a.dim() == 0 and len(dims) > 0:
        # 如果张量是零维且指定了维度，抛出索引错误
        raise IndexError(
            f"Dimension specified as {dims[0]} but tensor has no dimensions"
        )

    len_shifts = len(shifts)
    len_dims = len(dims)
    if len_shifts != 1 or len_dims != 1:
        if len_shifts == 0:
            # 当未指定 shifts 时，抛出运行时错误
            raise RuntimeError("`shifts` required")
        # 当未指定 dims 时，默认对张量进行展平后进行滚动，并恢复原始形状
        if len_dims == 0 and len_shifts == 1:
            return torch.roll(torch.flatten(a), shifts, 0).view(a.shape)
        if len_shifts != len_dims:
            # 当 shifts 和 dims 数量不一致时，抛出运行时错误
            raise RuntimeError(
                f"shifts and dimensions must align. shifts: {len_shifts}, dims: {len_dims}"
            )
        assert len_dims > 1
        tail_shifts = shifts[1:]
        tail_dims = dims[1:]
        first_dim_rolled = torch.roll(a, (shifts[0],), dims[0])
        # 对剩余维度进行滚动操作
        return torch.roll(first_dim_rolled, tail_shifts, tail_dims)

    # 当只有一个维度需要滚动时
    dim = dims[0]
    size = a.shape[dim]
    start = (size - shifts[0]) % size
    idx = torch.arange(size, device=a.device)
    # 返回按索引选取后的张量，实现滚动操作
    return a.index_select(dim, torch.fmod(start + idx, size))


# 注册 aten.rot90 的分解器，并标记为输出装饰器
@register_decomposition(aten.rot90)
@out_wrapper()
def rot90(
    a: TensorLikeType, k: int = 1, dims: DimsSequenceType = (0, 1)
) -> TensorLikeType:
    """Reference implementation of :func:`torch.rot90`."""
    if len(dims) != 2:
        # 若 dims 不等于 2，抛出运行时错误
        raise RuntimeError(
            f"expected total rotation dims == 2, but got dims = {len(dims)}"
        )
    if a.ndim < 2:
        # 若张量维度小于 2，抛出运行时错误
        raise RuntimeError(f"expected total dims >= 2, but got total dims = {a.ndim}")
    # 在初始检查之后执行此操作，以兼容核心的行为。
    # 规范化维度参数，确保与给定数组的维度兼容
    dims = utils.canonicalize_dims(a.ndim, dims)

    # 检查旋转维度是否相同，若相同则抛出运行时异常
    if dims[0] == dims[1]:
        raise RuntimeError(
            f"expected rotation dims to be different, but got dim0 = {dims[0]} and dim1 = {dims[1]}"
        )

    # 计算有效的旋转次数，确保在 0 到 3 的范围内
    k = k % 4  # 如果 k < 0，旋转方向是从第二个轴向第一个轴
    if k == 1:
        # 对数组进行 k = 1 的旋转操作：先沿 dims[1] 轴翻转，再转置 dims[0] 和 dims[1]
        return torch.transpose(torch.flip(a, (dims[1],)), dims[0], dims[1])
    elif k == 2:
        # 对数组进行 k = 2 的旋转操作：在所有维度上进行翻转
        return torch.flip(a, dims)
    elif k == 3:
        # 对数组进行 k = 3 的旋转操作：先沿 dims[0] 轴翻转，再转置 dims[0] 和 dims[1]
        return torch.transpose(torch.flip(a, (dims[0],)), dims[0], dims[1])
    else:
        # 若 k = 0，直接克隆数组并保持存储格式连续
        return a.clone(memory_format=torch.contiguous_format)
# 检查堆栈输入张量的形状是否相同
def _check_stack_inputs(tensors: TensorSequenceType) -> None:
    # 获取第一个张量的形状作为参考形状
    entry_shape = tensors[0].shape
    # 遍历所有张量，确保它们的形状与参考形状相同
    for i in range(1, len(tensors)):
        assert tensors[i].shape == entry_shape, (
            f"stack expects each tensor to be equal size, but got {entry_shape} at entry 0 "
            f"and {tensors[i].shape} at entry {i}"
        )


# 注册对应 torch.stack 的分解函数，并添加输出包装器
@out_wrapper()
def stack(tensors: TensorSequenceType, dim: int = 0) -> TensorLikeType:
    # 断言张量列表不为空
    assert len(tensors) > 0, "stack expects a non-empty TensorList"
    # 规范化堆叠的维度
    wrapped_dim = utils.canonicalize_dim(tensors[0].ndim + 1, dim)
    # 如果 wrapped_dim 小于第一个张量的维度，执行输入检查
    if wrapped_dim < tensors[0].ndim:  # and not tensors[0].is_sparse:
        _check_stack_inputs(tensors)
        # 构造结果张量的尺寸列表
        result_sizes = list(tensors[0].shape)
        result_sizes.insert(wrapped_dim, len(tensors))
        # 在指定维度上拼接张量
        out = torch.cat(tensors, wrapped_dim)
        # 调整视图以匹配结果尺寸
        return out.view(result_sizes)

    # 如果 wrapped_dim 等于第一个张量的维度，使用 unsqueeze 在指定维度上扩展张量
    return torch.cat([t.unsqueeze(wrapped_dim) for t in tensors], dim)


# 注册对应 torch.softmax 的分解函数，并添加输出包装器
@out_wrapper()
def softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # 确定结果的数据类型
    result_dtype = dtype or a.dtype
    # 确定计算过程中使用的数据类型
    computation_dtype = utils.get_computation_dtype(result_dtype)
    # 可能需要将输入张量转换为指定的计算数据类型
    a_ = _maybe_convert_to_dtype(a, computation_dtype)
    # 如果输入张量为空，返回空张量的指数
    if a.numel() == 0:
        a_exp = exp(a_)
    else:
        # 计算输入张量的最大值
        a_max = amax(a_, dim, keepdim=True)
        # 计算指数，并减去最大值以提高数值稳定性
        a_exp = exp(a_ - a_max)
    # 执行 softmax 操作，并根据指定维度计算归一化因子
    return _maybe_convert_to_dtype(
        true_divide(a_exp, sum(a_exp, dim, keepdim=True)), result_dtype
    )  # type: ignore[return-value]


# 注册对应 torch.hstack 的函数，并添加输出包装器
@out_wrapper()
def hstack(tensors: TensorSequenceType) -> TensorLikeType:
    # 检查张量列表不为空
    torch._check(len(tensors) > 0, lambda: "hstack expects a non-empty TensorList")
    # 将所有输入张量至少转换为一维张量
    aligned_tensors = atleast_1d(*tensors)
    # 如果第一个张量是一维的，沿着第一个维度拼接张量
    if aligned_tensors[0].ndim == 1:
        return cat(aligned_tensors, 0)
    # 否则，沿着第二个维度拼接张量
    return cat(aligned_tensors, 1)


# 注册对应 torch.vstack 的函数，并添加输出包装器
@out_wrapper()
def vstack(tensors: TensorSequenceType) -> TensorLikeType:
    # 检查张量列表不为空
    torch._check(len(tensors) > 0, lambda: "vstack expects a non-empty TensorList")
    # 将所有输入张量至少转换为二维张量
    aligned_tensors = atleast_2d(*tensors)
    # 沿着第一个维度拼接张量
    return cat(aligned_tensors, 0)


# 定义将张量展平的函数
def unflatten(a: TensorLikeType, dim: int, sizes: ShapeType) -> TensorLikeType:
    # 规范化展平的维度
    dim = utils.canonicalize_dim(a.ndim, dim)
    # 断言尺寸列表不为空
    torch._check(len(sizes) != 0, lambda: "unflatten: sizes must be non-empty")
    # 使用给定的尺寸将张量展平
    return a.view(tuple(a.shape[:dim]) + tuple(sizes) + tuple(a.shape[dim + 1 :]))


# 注册对应 torch.unbind 的分解函数
@register_decomposition(aten.unbind)
def unbind(t: TensorLikeType, dim: int = 0) -> TensorSequenceType:
    # 引入用于符号形状的大小保护
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 规范化解绑的维度
    dim = utils.canonicalize_dim(t.ndim, dim)
    # 调用内部函数 _check_index 进行索引检查，确保张量 t 至少有一个维度
    torch._check_index(
        len(t.shape) > 0,
        lambda: "Dimension specified as 0 but tensor has no dimensions",
    )
    # 如果 guard_size_oblivious 函数返回 True，表示维度 dim 处的大小为 0，返回空元组
    if guard_size_oblivious(t.shape[dim] == 0):
        return tuple()
    else:
        # 否则，使用 torch.tensor_split 函数按维度 dim 将张量 t 分割，并去除维度 dim 上的大小为 1 的维度
        return tuple(
            torch.squeeze(s, dim) for s in torch.tensor_split(t, t.shape[dim], dim)
        )
# 注册装饰器，将函数包装在一个外部函数中，可能执行某些前后处理
@out_wrapper()
# 定义一个函数，用于在给定维度上将索引处的值用另一个张量的值进行替换，返回替换后的张量
def index_copy(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    return x.clone(memory_format=torch.contiguous_format).index_copy_(
        dim, index, tensor
    )


# 在张量上原地进行索引复制操作，用给定的张量替换指定索引处的值，返回原始张量
def index_copy_(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    # 规范化维度参数，确保维度合法性
    dim = utils.canonicalize_dims(x.ndim, dim)
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    # 将标量视为 \R^1 中的元素
    y = x.unsqueeze(0) if x.ndim == 0 else x
    # 构造索引元组以进行索引操作
    idx = (slice(None),) * dim + (index,)
    # 在指定索引位置替换为给定张量的值
    y[idx] = tensor
    return x


# 注册装饰器，将函数注册到特定的分解函数上，可能执行某些前后处理
@register_decomposition(aten.index_fill)
@out_wrapper()
# 在张量上执行索引填充操作，将指定维度上索引处的值填充为给定值，返回填充后的张量
def index_fill(
    x: TensorLike, dim: int, index: TensorLike, value: Union[NumberType, TensorLike]
):
    return _index_fill(x, dim, index, value, inplace=False)


# 注册装饰器，将函数注册到特定的分解函数上
@register_decomposition(aten.index_fill_)
# 在张量上原地执行索引填充操作，将指定维度上索引处的值原地填充为给定值，返回原始张量
def index_fill_(
    x: TensorLike, dim: int, index: TensorLike, value: Union[NumberType, TensorLike]
):
    return _index_fill(x, dim, index, value, inplace=True)


# 辅助函数，执行索引填充的核心逻辑，支持原地和非原地操作
def _index_fill(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    value: Union[NumberType, TensorLike],
    *,
    inplace: bool,
):
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    if isinstance(value, TensorLike):
        torch._check(
            value.ndim == 0,
            lambda: "Only supports 0-dimensional value tensor. "
            f"Got a tensor with {value.ndim} dimensions.",
        )
    else:
        # 将值转换为张量标量
        value = torch.scalar_tensor(
            value, dtype=x.dtype, layout=x.layout, device=x.device
        )

    # 当输入张量是标量时，index_copy 对其有不必要的前提条件，通过此操作可以处理它们
    zero_dim = x.ndim == 0
    y = x.unsqueeze(0) if zero_dim else x
    # index_copy 不会对值进行广播，因此需要手动执行广播操作
    shape = list(y.shape)
    shape[dim] = index.numel()
    value = value.expand(shape)
    # 根据 inplace 参数选择使用 index_copy 或 torch.index_copy 进行索引替换
    index_copy = Tensor.index_copy_ if inplace else torch.index_copy
    out = index_copy(y, dim, index, value)
    if inplace:
        return x
    else:
        if zero_dim:
            # 必须克隆以返回一个新的张量，而不是视图
            out = out.squeeze(0).clone()
        # index_fill 保留张量的步幅。index_copy 总是返回连续的张量
        if out.stride() != x.stride():
            new_out = torch.empty_like(x)
            new_out.copy_(out)
            out = new_out
        return out


# 注册装饰器，将函数包装在一个外部函数中，可能执行某些前后处理
@out_wrapper()
# 在张量上执行索引加法操作，将指定维度上索引处的值加上给定张量的值乘以 alpha，返回加法后的张量
def index_add(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = 1,
):
    # index_add 总是返回一个新的连续张量
    return x.clone(memory_format=torch.contiguous_format).index_add_(
        dim, index, tensor, alpha=alpha
    )
    # 定义一个装饰器函数，用于检查函数的参数是否为正整数
    def positive_integer_check(func):
        # 定义一个包装函数，接收任意参数
        def wrapper(*args, **kwargs):
            # 检查所有位置参数
            for arg in args:
                # 如果参数不是整数或者小于等于0，抛出异常
                if not isinstance(arg, int) or arg <= 0:
                    raise ValueError("Arguments must be positive integers.")
            # 检查所有关键字参数的值
            for value in kwargs.values():
                # 如果关键字参数的值不是整数或者小于等于0，抛出异常
                if not isinstance(value, int) or value <= 0:
                    raise ValueError("Keyword arguments must have values that are positive integers.")
            # 如果所有参数检查通过，调用原始函数并返回结果
            return func(*args, **kwargs)
        # 返回包装函数
        return wrapper
# 将函数注册为指定操作的分解函数
# 将函数注册为输出包装器
@register_decomposition(aten.index_select)
@out_wrapper()
def index_select(x: TensorLike, dim: int, index: TensorLike):
    # 规范化维度参数，确保在合法范围内
    dim = utils.canonicalize_dims(x.ndim, dim)
    # 检查索引的维度是否为1或0
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    # 如果索引是0维，则扩展为1维
    if index.ndim == 0:
        index = index.unsqueeze(0)
    # 如果输入张量是0维，则将其视为\R^1中的元素
    if x.ndim == 0:
        # 由于无法使用 x[idx] 访问元素，使用此笨拙的构造方式
        return torch.empty_like(x).index_copy(0, index, x.expand_as(index))

    # 构造索引元组，用于在指定维度上选择索引
    idx = (slice(None),) * dim + (index,)
    return x[idx]


# 注册为指定操作的分解函数
@register_decomposition(aten.squeeze.dims)
def squeeze(a: TensorLikeType, dim: Optional[DimsType] = None) -> TensorLikeType:
    # 导入符号形状保护函数
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 如果未指定维度，则找到所有维度为1的维度
    if dim is None:
        dims = tuple(idx for idx, size in enumerate(a.shape) if size == 1)
        return prims.squeeze(a, dims) if dims else prims.view_of(a)

    # 获取张量的维度数，并规范化给定的维度参数
    ndim = a.ndim
    dim = utils.canonicalize_dims(ndim, dim)
    # 如果张量没有维度，则直接返回其视图
    if ndim == 0:
        assert len(dims) == 0 or dims == (0,)
        return prims.view_of(a)

    # 紧缩操作不会修改张量，如果指定的维度长度不为1
    dims = tuple(d for d in dims if guard_size_oblivious(a.shape[d] == 1))
    # 如果没有要删除的维度，直接返回张量视图
    if len(dims) == 0:
        return prims.view_of(a)
    # 如果只有一个维度要删除，进行紧缩操作
    if len(dims) == 1:
        return prims.squeeze(a, dims)
    # 多个维度要删除，递归紧缩操作
    dims_list = list(dims)
    dims_list = sorted(dims_list, reverse=True)
    for i in dims_list:
        a = squeeze(a, i)
    return a


# 注意：由于数据相关的控制流，此函数无法处理 TensorMetas
# CompositeImplicitAutograd - 不注册为分解函数
def tensor_split(
    a: TensorLikeType,
    indices_or_sections: Union[Tensor, DimsType],
    dim: int = 0,
) -> Tuple[TensorLikeType, ...]:
    # 规范化维度参数
    _dim = utils.canonicalize_dim(a.ndim, dim)
    # 如果输入张量是0维，则引发异常
    if a.ndim == 0:
        msg = "tensor_split: received a rank zero tensor, but expected a tensor of rank one or greater!"
        raise ValueError(msg)

    # 如果 indices_or_sections 是张量，则必须是 CPU 上的长整型张量
    if isinstance(indices_or_sections, TensorLike):
        if not indices_or_sections.device.type == "cpu":
            msg = (
                f"tensor_split: if indices_or_sections is a tensor it must be on the CPU, "
                f"but received one on {indices_or_sections.device}"
            )
            raise ValueError(msg)
        if indices_or_sections.dtype != torch.long:
            msg = "tensor_split: if indices_or_sections is a tensor it must have long dtype, "
            f" but received one with dtype {indices_or_sections.dtype}"
            raise ValueError(msg)

    # Case 0 -- indices_or_sections is an integer or a scalar tensor n and a is split along dim into n parts of equal-ish length
    # 如果 indices_or_sections 是 IntLike 类型或者是 TensorLike 类型且维度为 0
    # IntLike 类型是指类似整数的类型，TensorLike 是指类似张量的类型
    or (
        isinstance(indices_or_sections, TensorLike) and indices_or_sections.ndim == 0
    ):
        # 将 sections 定义为整数
        sections: int = (
            indices_or_sections  # type: ignore[assignment]
            # 如果 indices_or_sections 是 Number 类型，则直接使用
            if isinstance(indices_or_sections, Number)
            # 否则取其 item() 方法返回的值
            else indices_or_sections.item()
        )

        # 如果 sections 小于等于 0，则抛出 ValueError 异常
        if sections <= 0:
            msg = f"tensor_split: number of sections must be greater than 0, but was {sections}"
            raise ValueError(msg)

        # 初始化空列表 splits 存储切片后的结果
        splits = []
        # 获取 a 在指定维度 _dim 上的大小
        dim_size = a.shape[_dim]
        # 计算每个切片的最小大小
        min_split_size = math.floor(dim_size / sections)
        # 计算需要额外增加一个元素的切片数
        num_splits_one_extra = dim_size % sections
        # 初始化切片的起始索引
        start_idx = 0
        # 循环创建各个切片
        for split_idx in range(sections):
            # 计算当前切片的大小
            split_size = (
                min_split_size + 1
                if (split_idx < num_splits_one_extra)
                else min_split_size
            )
            # 利用 prims.slice_in_dim 函数切片 a，并添加到 splits 列表中
            s = prims.slice_in_dim(a, start_idx, start_idx + split_size, axis=_dim)
            splits.append(s)
            # 更新下一个切片的起始索引
            start_idx = start_idx + split_size

        # 返回切片结果的元组
        return tuple(splits)
    # Case 1 -- indices_or_sections 是一个整数序列或者描述切片的 1 维张量
    else:
        # 将 indices 定义为 indices_or_sections
        indices = indices_or_sections
        # 如果 indices_or_sections 是 TensorLike 类型
        if isinstance(indices_or_sections, TensorLike):
            # 如果 indices_or_sections 的维度不为 1，则抛出 ValueError 异常
            if indices_or_sections.ndim != 1:
                msg = "tensor_split: non-scalar indices_or_sections tensors must have only one dimension, "
                f"but received a tensor with {indices_or_sections.ndim} dimensions"
                raise ValueError(msg)

            # 将 indices 转换为 Python 列表
            indices = indices_or_sections.tolist()

        # 初始化空列表 splits 存储切片后的结果
        splits = []
        # 初始化切片的起始索引
        start_idx = 0
        # 遍历 indices 中的每个元素 x
        for x in indices:
            # 利用 prims.slice_in_dim 函数切片 a，并添加到 splits 列表中
            splits.append(prims.slice_in_dim(a, start_idx, x, axis=_dim))
            # 更新下一个切片的起始索引
            start_idx = x
        # 切片 a 的剩余部分，并添加到 splits 列表中
        splits.append(prims.slice_in_dim(a, start_idx, a.shape[_dim], axis=_dim))
        # 返回切片结果的元组
        return tuple(splits)
# CompositeImplicitAutograd - don't register decomp
# 定义函数hsplit，用于按指定维度水平分割张量
def hsplit(
    a: TensorLikeType, indices_or_sections: DimsType
) -> Tuple[TensorLikeType, ...]:
    # 检查张量a的维度是否至少为1
    torch._check(
        a.ndim >= 1,
        lambda: (
            "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with "
            + str(a.ndim)
            + " dimensions!"
        ),
    )
    # 确定分割的维度，如果张量a是一维的，则dim为0，否则为1
    dim = 0 if a.ndim == 1 else 1
    # 如果indices_or_sections是整数类型，表示按固定大小分割
    if isinstance(indices_or_sections, IntLike):
        split_size = indices_or_sections
        # 检查能否按指定大小分割，即a的dim维度大小能否被split_size整除
        torch._check(
            (split_size != 0 and a.shape[dim] % split_size == 0),
            lambda: (
                "torch.hsplit attempted to split along dimension "
                + str(dim)
                + ", but the size of the dimension "
                + str(a.shape[dim])
                + " is not divisible by the split_size "
                + str(split_size)
                + "!"
            ),
        )
        # 调用tensor_split函数进行分割
        return tensor_split(a, split_size, dim)

    # 如果indices_or_sections是列表或元组类型，表示按指定位置分割
    torch._check_type(
        isinstance(indices_or_sections, (list, tuple)),
        lambda: (
            "hsplit(): received an invalid combination of arguments. "
            "Expected indices_or_sections to be of type int, list of ints or tuple of ints "
            f"but got type {type(indices_or_sections)}"
        ),
    )

    # 将分割位置信息存储在split_sizes中，然后调用tensor_split进行分割
    split_sizes = indices_or_sections
    return tensor_split(a, split_sizes, dim)


# CompositeImplicitAutograd - don't register decomp
# 定义函数vsplit，用于按指定维度垂直分割张量
def vsplit(
    a: TensorLikeType, indices_or_sections: DimsType
) -> Tuple[TensorLikeType, ...]:
    # 检查张量a的维度是否至少为2
    torch._check(
        a.ndim >= 2,
        lambda: (
            "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with "
            + str(a.ndim)
            + " dimensions!"
        ),
    )
    # 如果indices_or_sections是整数类型，表示按固定大小分割
    if isinstance(indices_or_sections, IntLike):
        split_size = indices_or_sections
        # 检查能否按指定大小分割，即张量a的第0维大小能否被split_size整除
        torch._check(
            (split_size != 0 and a.shape[0] % split_size == 0),
            lambda: (
                f"torch.vsplit attempted to split along dimension 0"
                f", but the size of the dimension "
                f"{a.shape[0]}"
                f" is not divisible by the split_size "
                f"{split_size}"
                f"!"
            ),
        )
        # 调用tensor_split函数进行分割
        return tensor_split(a, split_size, 0)

    # 如果indices_or_sections是列表或元组类型，表示按指定位置分割
    torch._check_type(
        isinstance(indices_or_sections, (list, tuple)),
        lambda: (
            "vsplit(): received an invalid combination of arguments. "
            "Expected indices_or_sections to be of type int, list of ints or tuple of ints "
            f"but got type {type(indices_or_sections)}"
        ),
    )

    # 将分割位置信息存储在split_sizes中，然后调用tensor_split进行分割
    split_sizes = indices_or_sections
    return tensor_split(a, split_sizes, 0)


# 注册aten.diag.out的分解函数装饰器，且不注册分解
# 定义diag函数，用于提取对角线元素
@register_decomposition(aten.diag.out)
# 使用输出包装器
@out_wrapper()
# 函数签名指定参数和返回类型
def diag(
    self: TensorLikeType,
    offset: int = 0,
) -> TensorLikeType:
    # 获取张量的维度
    ndim = self.dim()
    # 检查张量是否为1维或2维
    torch._check(
        ndim in (1, 2), lambda: f"diag(): Supports 1D or 2D tensors. Got {ndim}D"
    )
    # 如果张量是一维的，则返回一个以该张量为对角线元素的对角矩阵，偏移量为offset
    if ndim == 1:
        return torch.diag_embed(self, offset)
    # 如果张量不是一维的，则返回一个复制该张量对角线元素的张量，偏移量为offset
    else:
        return torch.diagonal_copy(self, offset)
@register_decomposition(aten.diagonal_scatter)
@out_wrapper()
def diagonal_scatter(
    input: TensorLikeType,
    src: TensorLikeType,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
) -> TensorLikeType:
    # 复制输入张量并保留其步长信息
    out = utils.clone_preserve_strides(input)
    # 提取指定偏移量和维度上的对角线元素
    diag = out.diagonal(offset, dim1, dim2)
    # 检查对角线张量和源张量的形状是否一致
    torch._check(
        diag.shape == src.shape,
        lambda: "expected src to have a size equal to the diagonal of the input."
        f"Got {src.shape} for a diagonal of shape {diag.shape}",
    )
    # 将源张量的值复制到对角线张量中
    copy_to(diag, src)
    # 返回修改后的张量
    return out


@register_decomposition(aten.diagonal)
def diagonal(
    self: TensorLikeType,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
) -> TensorLikeType:
    """
    Reference implementation of torch.diagonal
    """
    # 获取张量的维度数
    num_dims = self.dim()
    # 规范化维度索引
    dim1 = utils.canonicalize_dim(idx=dim1, rank=num_dims)
    dim2 = utils.canonicalize_dim(idx=dim2, rank=num_dims)

    # 检查对角线的两个维度不能相同
    torch._check(
        dim1 != dim2, lambda: f"diagonal dimensions cannot be identical {dim1}, {dim2}"
    )

    # 获取存储偏移量
    storage_offset = self.storage_offset()

    # 根据偏移量计算对角线的大小
    if offset >= 0:
        diag_size = max(min(self.size()[dim1], self.size()[dim2] - offset), 0)
    else:
        diag_size = max(min(self.size()[dim1] + offset, self.size()[dim2]), 0)

    # 根据偏移量调整存储偏移量
    if diag_size > 0:
        if offset >= 0:
            storage_offset += offset * self.stride()[dim2]
        else:
            storage_offset -= offset * self.stride()[dim1]

    # 构建输出张量的大小和步长
    sizes = [s for i, s in enumerate(self.size()) if i not in (dim1, dim2)]
    sizes.append(diag_size)

    strides = [s for i, s in enumerate(self.stride()) if i not in (dim1, dim2)]
    strides.append(self.stride()[dim1] + self.stride()[dim2])

    # 使用给定的大小、步长和存储偏移量创建视图
    result = self.as_strided(size=sizes, stride=strides, storage_offset=storage_offset)

    # 返回创建的对角线张量视图
    return result


diagonal_copy = _make_copy_from_view(diagonal)


@register_decomposition(aten.diag_embed)
@out_wrapper()
def diag_embed(
    t: TensorLikeType,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> TensorLikeType:
    """
    Reference implementation of torch.diag_embed
    """
    # 将负维度索引转换为正值
    rank = t.ndim + 1
    dim1 = utils.canonicalize_dim(rank=rank, idx=dim1)
    dim2 = utils.canonicalize_dim(rank=rank, idx=dim2)

    # 根据文档，如果交换维度，等同于改变偏移量的符号
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
        offset = -offset

    # 检查对角线的两个维度不能相同
    torch._check(
        dim1 != dim2, lambda: f"diagonal dimensions cannot be identical {dim1}, {dim2}"
    )

    # 根据文档，将最后一个维度的大小放置在 dim1 和 dim2
    last_dim = t.size(-1)
    # 如果偏移量不为零，则进行填充以匹配新的大小
    t_shape = list(t.shape)  # 获取张量 t 的形状并转换为列表
    t_shape[-1] = builtins.abs(offset)  # 将最后一个维度设置为偏移量的绝对值
    z = torch.zeros(t_shape, dtype=t.dtype, device=t.device, requires_grad=False)  # 创建一个形状与 t_shape 相同的零张量 z
    pair = (z, t) if offset > 0 else (t, z)  # 根据偏移量的正负选择拼接顺序
    t = torch.cat(pair, dim=-1)  # 在最后一个维度上拼接张量对 pair，得到新的张量 t
    last_dim += builtins.abs(offset)  # 更新 last_dim，确保对角线始终具有相同的大小

    # 保留原始数据，但在 dim1 上放置 1 并将最后一个维度移动到 dim2
    t = t.unsqueeze(dim1).movedim(-1, dim2)

    # 基于偏移量生成移位索引的范围
    a_range = torch.arange(last_dim, device=t.device, dtype=torch.int64)  # 创建从 0 到 last_dim 的整数张量 a_range
    b_range = torch.arange(
        offset, last_dim + offset, device=t.device, dtype=torch.int64
    )  # 创建从 offset 到 last_dim+offset 的整数张量 b_range

    # 广播操作
    cond = a_range == b_range.unsqueeze(-1)  # 创建条件张量，判断 a_range 与 b_range 是否相等
    cond_shape = [last_dim if i in (dim1, dim2) else 1 for i in range(len(t.shape))]  # 创建 cond 的形状
    cond = cond.reshape(cond_shape)  # 将 cond 重塑为 cond_shape 形状

    # aten.diag_embed 总是返回一个新的连续张量
    # 需要使用 contiguous() 正确建模输出步幅
    return utils.mask_tensor(cond, t).contiguous()  # 调用 utils 中的函数 mask_tensor 对 t 应用条件掩码，并保证返回的张量是连续的
@register_decomposition(aten.block_diag)
@out_wrapper()
def _block_diag_iterable(tensors: List[TensorLikeType]) -> TensorLikeType:
    """
    Reference implementation of torch.block_diag
    """
    # 将所有输入张量变形为二维张量列表
    tensors_2d = [
        tensor.view(1, -1) if tensor.dim() <= 1 else tensor for tensor in tensors
    ]

    # 计算所有二维张量的总列数
    ncols = builtins.sum(tensor.shape[1] for tensor in tensors_2d)
    # 获取第一个张量的设备信息
    device = tensors_2d[0].device

    # 结果初始化为空列表
    result = []

    # 初始化列的起始位置
    col_start = 0
    # 遍历每个二维张量
    for i, tensor in enumerate(tensors_2d):
        # 检查张量维度是否为2
        torch._check(
            tensor.dim() == 2,
            lambda: "Input tensors must have 2 or fewer dimensions. "
            f"Input {i} has {tensor.dim()} dimensions",
        )
        # 检查张量是否在同一设备上
        torch._check(
            tensor.device == device,
            lambda: "Input tensors must all be on the same device. "
            f"Input 0 is on device {device} and input {i} is on device {tensor.device}.",
        )
        # 获取张量的行数和列数
        row, col = tensor.shape
        # 创建左侧和右侧的零张量
        left = torch.zeros((row, col_start), device=device, dtype=tensor.dtype)
        right = torch.zeros(
            (row, ncols - col_start - col), device=device, dtype=tensor.dtype
        )
        # 将左侧、当前张量和右侧连接起来，并添加到结果列表中
        result += [torch.cat((left, tensor, right), dim=1)]
        # 更新列的起始位置
        col_start += col

    # 将所有结果张量按行连接起来形成最终结果
    return torch.cat(result, dim=0)


def block_diag(*tensors: List[TensorLikeType]) -> TensorLikeType:
    """
    This is used as an input to PythonRefInfo. `torch.block_diag`
    expects arguments splatted, but `aten.block_diag` expects only
    one argument that is a list of Tensors.
    """
    # 调用 _block_diag_iterable 函数处理输入的张量列表
    return _block_diag_iterable(tensors)


# CompositeImplicitAutograd - don't register decomp
def dsplit(a: TensorLikeType, sections: DimsType) -> TensorSequenceType:
    # 检查输入张量维度是否小于3，如果是则抛出错误
    if a.ndim < 3:
        raise RuntimeError(
            f"torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with {a.ndim} dimensions!"
        )
    # 检查 sections 是否为整数且不为零，以及是否能整除张量的第二维度
    if isinstance(sections, IntLike) and (sections == 0 or a.shape[2] % sections != 0):
        raise RuntimeError(
            "torch.dsplit attempted to split along dimension 2, "
            + f"but the size of the dimension {a.shape[2]} is not divisible by the split_size {sections}!"
        )
    # 调用 tensor_split 函数进行张量分割操作
    return tensor_split(a, sections, 2)


@register_decomposition(aten.t.default)
def t(a: TensorLikeType):
    # TODO: Add sparse support
    # if a.is_sparse:
    #     sparse_dim = a.sparse_dim()
    #     dense_dim = a.dense_dim()
    #     if not (sparse_dim <= 2 and dense_dim == 0):
    #         raise RuntimeError(
    #             f"t() expects a tensor with <= 2 sparse and 0 dense dimensions, but got {sparse_dim} sparse and"
    #             f"{dense_dim} dense dimensions"
    #         )
    
    # 检查张量维度是否大于2，如果是则抛出错误
    if a.ndim > 2:
        raise RuntimeError(
            f"t() expects a tensor with <= 2 dimensions, but self is {a.ndim}D"
        )
    # 返回张量的转置，如果维度小于2则不改变
    return torch.transpose(a, 0, 0 if a.ndim < 2 else 1)


# CompositeImplicitAutograd - don't register decomp
def T(a: TensorLikeType) -> TensorLikeType:
    # 没有具体实现，仅作为占位符存在
    # 检查张量 `a` 的维度是否为 0 或 2
    torch._check(
        a.ndim in (0, 2),
        lambda: (
            "The use of `x.T` on tensors of dimension other than 0 or 2 "
            "to reverse their shape is not supported."
        ),
    )
    # 返回张量 `a` 的转置，如果满足维度条件
    return a.t()
@register_decomposition(aten.alias)
def alias(a: TensorLikeType) -> TensorLikeType:
    # 返回给定张量的视图
    return prims.view_of(a)


alias_copy = _make_copy_from_view(alias)


@register_decomposition(aten.transpose)
def transpose(a: TensorLikeType, dim0: int, dim1: int) -> TensorLikeType:
    _dim0, _dim1 = utils.canonicalize_dims(a.ndim, (dim0, dim1))  # type: ignore[misc]

    if a.ndim <= 1 or dim0 == dim1:
        # 如果张量的维度小于等于1或dim0等于dim1，则返回默认的别名张量
        return aten.alias.default(a)

    _permutation = list(range(0, a.ndim))
    _permutation[_dim0] = _dim1
    _permutation[_dim1] = _dim0
    # 使用给定的排列重新排列张量的维度并返回结果
    return torch.permute(a, _permutation)


# Aliases for transpose
swap_axes = transpose


@register_decomposition(aten.unfold)
def unfold(
    self: TensorLikeType, dimension: int, size: int, step: int
) -> TensorLikeType:
    # 计算张量按指定维度展开后的形状和步幅
    shape, strides = _get_unfold_shape_stride(
        self.shape, self.stride(), dimension, size, step
    )
    # 返回按指定维度展开后的张量
    return self.as_strided(shape, strides)


@register_decomposition(aten.unfold_copy)
@out_wrapper()
def unfold_copy(self: TensorLikeType, dimension: int, size: int, step: int):
    # 返回按指定维度展开后的张量的克隆
    return self.unfold(dimension, size, step).clone(
        memory_format=torch.contiguous_format
    )


def _cumsumprod_common(
    func,
    init,
    a: TensorLikeType,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # 实现一个通用的累加或累乘操作
    # 备注: 此分解可能不如专门的后端实现高效
    ndim = a.ndim
    dim = utils.canonicalize_dim(ndim, dim)
    if ndim == 0:
        return func(a.unsqueeze(0), dim=0, dtype=dtype, out=out)
    a = a.unsqueeze(dim + 1)
    rg = torch.arange(a.shape[dim], device=a.device)
    mask = rg.unsqueeze(1) <= rg
    for _ in range(ndim - dim - 1):
        mask = mask.unsqueeze(-1)
    masked_a = torch.where(mask, a, init)
    return func(masked_a, dim=dim, dtype=dtype, out=out)


@register_decomposition(aten.cumsum)
def cumsum(
    a: TensorLikeType,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # 对给定维度上的张量进行累加操作
    return _cumsumprod_common(func=sum, init=0, a=a, dim=dim, dtype=dtype, out=out)


@register_decomposition(aten.cumprod)
def cumprod(
    a: TensorLikeType,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> TensorLikeType:
    # 对给定维度上的张量进行累乘操作
    return _cumsumprod_common(func=prod, init=1, a=a, dim=dim, dtype=dtype, out=out)


# Note: although squeeze is documented as having the out= kwarg it doesn't
@register_decomposition(aten.unsqueeze)
def unsqueeze(a: TensorLikeType, dim: int) -> TensorLikeType:
    # 注意，unsqueeze会将张量的秩增加1，因为它允许指定一个新的最内层维度
    ndim = a.ndim + 1
    dim = utils.canonicalize_dim(ndim, dim)
    # 返回扩展了指定维度后的张量
    return prims.expand_dims(a, (dim,), ndim=ndim)


# NOTE: shape is a vararg because Tensor.reshape can be called with as
# 注册 aten.view.default 函数的装饰器，指定函数的类型签名
@register_decomposition(aten.view.default)
def view(a: TensorLikeType, *shape: ShapeType) -> TensorLikeType:
    # 调用 _reshape_view_helper 函数来执行视图重塑操作，禁止复制数据
    return _reshape_view_helper(a, *shape, allow_copy=False)


# 定义 CompositeImplicitAutograd 类中的 view_as 方法，不注册为分解函数
def view_as(self: TensorLikeType, other: TensorLikeType) -> TensorLikeType:
    # 调用 self 的 view 方法，使用 other 的大小来执行视图操作
    return self.view(other.size())


# 定义 CompositeImplicitAutograd 类中的 ravel 方法，不注册为分解函数
def ravel(a: TensorLikeType) -> TensorLikeType:
    # 调用 reshape 函数，将 a 摊平为一维数组
    return reshape(a, (-1,))


# 定义 CompositeImplicitAutograd 类中的 take_along_dim 方法，不注册为分解函数
# 使用装饰器 @out_wrapper() 对该方法进行包装
def take_along_dim(
    a: torch.Tensor, indices: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    # 检查输入张量 a 和索引张量 indices 的维度是否相同
    torch._check(
        a.ndim == indices.ndim,
        lambda: (
            "torch.take_along_dim(): input and indices should have the same "
            f"number of dimensions, but got {a.ndim} dimensions for input, and "
            f"{indices.ndim} dimensions for indices"
        ),
    )

    # 检查索引张量 indices 的数据类型是否为整数类型
    torch._check(
        utils.is_integer_dtype(indices.dtype),
        lambda: (
            "torch.take_along_dim(): dtype of indices should be int but got "
            f"{indices.dtype} instead"
        ),
    )

    if dim is None:
        # 若未指定维度 dim，则将张量 a 视图为一维，并按照 indices 进行 gather 操作
        return torch.gather(a.view(-1), 0, indices.view(-1))
    else:
        # 处理指定维度 dim 的情况
        # 获取张量 a 和 indices 的形状信息
        self_sizes = list(a.shape)
        self_sizes[dim] = indices.size(dim)
        # 推断广播后的形状
        broadcast_shape = utils.infer_size_shapes(self_sizes, indices.size())
        # 根据推断的形状进行广播
        indices_broadcast = broadcast_to(indices, broadcast_shape)

        # 获取 indices 的形状信息
        indices_sizes = list(indices.shape)
        indices_sizes[dim] = a.size(dim)
        # 推断广播后的形状
        broadcast_shape = utils.infer_size_shapes(indices_sizes, a.size())
        # 根据推断的形状进行广播
        self_broadcast = broadcast_to(a, broadcast_shape)

        # 对广播后的张量进行 gather 操作
        return torch.gather(self_broadcast, dim, indices_broadcast)


# 定义 empty 函数，使用 @out_wrapper() 装饰器包装
# 创建一个空张量，支持指定的数据类型、布局、设备、梯度是否跟踪、是否固定内存、内存格式等参数
@out_wrapper()
def empty(
    *shape,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> TensorLikeType:
    # 检查是否选择了保留内存格式，不支持 Preserve 内存格式
    torch._check(
        memory_format != torch.preserve_format,
        lambda: "torch.empty: the Preserve memory format is not supported",
    )

    # 提取变长参数中的形状信息
    shape = utils.extract_shape_from_varargs(shape)

    # 根据选择的内存格式，生成张量的步长信息
    if memory_format == torch.contiguous_format:
        strides = utils.make_contiguous_strides_for(shape)
    elif memory_format == torch.channels_last_3d:
        strides = utils.make_channels_last_3d_strides_for(shape)
    else:  # 如果 memory_format == torch.channels_last
        # 检查 memory_format 是否为 torch.channels_last，否则抛出异常信息
        torch._check(
            memory_format == torch.channels_last,
            lambda: f"torch.empty: received an unknown memory format {memory_format}!",
        )
        # 根据 shape 创建 channels_last 内存格式的二维步长
        strides = utils.make_channels_last_2d_strides_for(shape)
    
    # 返回一个新的空张量，使用指定的形状、步长、数据类型、布局、设备等参数
    return torch.empty_strided(
        shape,
        strides,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
# 定义一个装饰器函数，用于处理输出包装器
@out_wrapper()
def empty_permuted(
    shape,
    physical_layout,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> TensorLikeType:
    # 调用内部函数 prims.empty_permuted，返回结果
    return prims.empty_permuted(
        shape,
        physical_layout,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


# 注册一个 aten.new_empty 的分解函数，并应用输出包装器
@register_decomposition(aten.new_empty)
@out_wrapper()
def new_empty(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
) -> TensorLikeType:
    # 如果未指定 dtype、layout 或 device，则使用输入张量 a 的对应属性
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    # 调用 torch.empty 函数创建一个空张量，并返回结果
    return torch.empty(
        size,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory,
        layout=layout,
    )


# 注册一个 aten.new_empty_strided 的分解函数，并应用输出包装器
@out_wrapper()
def new_empty_strided(
    a: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.Tensor.new_empty_strided
    """

    # 如果未指定 dtype、layout 或 device，则使用输入张量 a 的对应属性
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    # 调用 torch.empty_strided 函数创建一个按指定步长的空张量，并返回结果
    return torch.empty_strided(
        size,
        stride,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory,
        layout=layout,
    )


# 注册一个 aten.zeros.default 的分解函数，并应用输出包装器
@out_wrapper()
def zeros(
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    # 从可变参数中提取张量的形状大小
    size = utils.extract_shape_from_varargs(size)

    # 如果未指定 dtype，则使用默认的 torch 数据类型
    if dtype is None:
        dtype = torch.get_default_dtype()

    # 调用 torch.full 函数创建一个填充的张量，并返回结果
    return torch.full(
        size,
        False if dtype == torch.bool else 0,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


# 注册一个 aten.new_zeros 的分解函数，并应用输出包装器
@out_wrapper()
def new_zeros(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    # 如果未指定 dtype、layout 或 device，则使用输入张量 a 的对应属性
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device
    # 返回一个填充满指定大小的张量
    return torch.full(
        size,  # 指定张量的大小，即张量的形状
        False if (dtype or a.dtype) == torch.bool else 0,  # 根据 dtype 或者 a 的数据类型是否为 torch.bool 来确定填充值，如果不是 bool 类型则填充 0
        dtype=dtype,  # 指定张量的数据类型
        layout=layout,  # 指定张量的布局方式
        device=device,  # 指定张量的存储设备
        pin_memory=pin_memory,  # 如果为 True，则将张量固定在内存中，使得 GPU 可以快速访问
        requires_grad=requires_grad,  # 如果为 True，则张量的操作会被跟踪，以便自动计算梯度
    )
@register_decomposition(aten.ones.default)
@out_wrapper()
def ones(
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    # 从变长参数中提取形状信息
    size = utils.extract_shape_from_varargs(size)

    # 如果未指定数据类型，则使用默认的数据类型
    if dtype is None:
        dtype = torch.get_default_dtype()

    # 创建一个填充满指定值的张量
    return torch.full(
        size,
        True if dtype == torch.bool else 1,  # 根据数据类型选择填充值
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition(aten.new_ones)
@out_wrapper()
def new_ones(
    a: TensorLikeType,
    size: ShapeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    # 如果未指定数据类型、布局或设备，则使用输入张量的对应属性
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    # 创建一个填充满指定值的张量，值为 True 或 1，根据数据类型确定
    return torch.full(
        size,
        True if (dtype or a.dtype) == torch.bool else 1,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )


@register_decomposition(aten.new_full)
@out_wrapper()
def new_full(
    a: TensorLikeType,
    size: ShapeType,
    fill_value: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
) -> TensorLikeType:
    # 如果未指定数据类型、布局或设备，则使用输入张量的对应属性
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    # 创建一个填充满指定值的张量
    return torch.full(
        size,
        fill_value,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )


@register_decomposition(aten.empty_like)
@out_wrapper()
def empty_like(
    a: TensorLikeType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: Optional[torch.layout] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> TensorLikeType:
    # 如果未指定数据类型、布局或设备，则使用输入张量的对应属性
    dtype = a.dtype if dtype is None else dtype
    layout = a.layout if layout is None else layout
    device = a.device if device is None else device

    # 如果需要按照特定内存格式创建张量，则使用指定的内存格式
    if memory_format != torch.preserve_format:
        return torch.empty(
            a.shape,
            dtype=dtype,
            layout=layout,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
            memory_format=memory_format,
        )

    # 如果内存格式为保持不变，则根据输入张量计算输出张量的逻辑到物理排列
    logical_to_physical_perm = (
        utils.compute_elementwise_output_logical_to_physical_perm(a)
    )
    # 创建一个空的张量，并按照指定的排列方式进行排列
    return torch.empty_permuted(
        a.shape,  # 使用张量 a 的形状作为新张量的形状
        logical_to_physical_perm,  # 使用给定的排列方式对张量进行重新排列
        dtype=dtype,  # 指定新张量的数据类型
        layout=layout,  # 指定新张量的布局方式
        device=device,  # 指定新张量所在的设备
        pin_memory=pin_memory,  # 如果为 True，则新张量会被锁定在内存中，用于异步数据传输
        requires_grad=requires_grad,  # 如果为 True，则新张量会跟踪其梯度
    )
# 将函数注册到特定的函数分解器列表中，用于处理torch.arange的不同情况
# 同时使用装饰器封装函数输出
@register_decomposition([aten.arange.start_step, aten.arange.start_out])
@out_wrapper()
def arange(
    start: NumberType = 0,  # 起始值，默认为0
    end: Optional[NumberType] = None,  # 终止值，可选，默认为None
    step: NumberType = 1,  # 步长，默认为1
    *,
    dtype: Optional[torch.dtype] = None,  # 数据类型，默认为None
    layout: torch.layout = torch.strided,  # 布局，默认为torch.strided
    device: Optional[DeviceLikeType] = None,  # 设备类型，默认为None
    pin_memory: bool = False,  # 是否使用固定内存，默认为False
    requires_grad: bool = False,  # 是否需要梯度，默认为False
) -> TensorLikeType:  # 返回类型为TensorLikeType

    # 检查布局是否有效
    utils.check_layout(layout)
    # 检查是否需要固定内存
    utils.check_pin_memory(pin_memory)
    # 将设备设置为默认设备或者指定的设备
    device = torch.device(utils.device_or_default(device))

    # 断言起始值、终止值和步长不是复数
    assert not isinstance(start, complex)
    assert not isinstance(end, complex)
    assert not isinstance(step, complex)

    # Case: torch.arange(5)，处理只有一个参数的情况
    if end is None:
        end = start
        start = 0

    # 检查步长不能为零
    torch._check(step != 0, lambda: "step must be nonzero")
    
    # 根据步长的正负性检查起始值和终止值是否一致
    if step > 0:
        torch._check(
            end >= start,
            lambda: "upper bound and lower bound inconsistent with step sign",
        )
    elif step < 0:
        torch._check(
            end <= start,
            lambda: "upper bound and lower bound inconsistent with step sign",
        )

    # 定义一个函数，用于检查值是否为有限浮点数或者特定类型
    def is_finite(x):
        return not isinstance(x, FloatWithoutSymFloat) or math.isfinite(x)

    # 检查起始值和终止值是否为有限数值
    torch._check(
        is_finite(start) and is_finite(end),
        lambda: f"unsupported range: {start} -> {end}",
    )
    # 检查步长是否为有限数值
    torch._check(
        is_finite(step),
        lambda: f"step must be finite but got {step}",
    )

    # 将起始值、终止值和步长组成元组
    args = (start, end, step)
    # 检查元组中的参数是否为整数类型
    integer_args = builtins.all(isinstance(arg, IntLike) for arg in args)

    # 如果数据类型为None，则根据整数类型设置默认的数据类型
    if dtype is None:
        dtype = torch.int64 if integer_args else torch.get_default_dtype()

    # 检查数据类型是否为整数类型
    is_integer = utils.is_integer_dtype(dtype)
    if is_integer:
        # 将起始值、终止值和步长转换为符号整数类型
        xstart = sym_int(start)
        xend = sym_int(end)
        xstep = sym_int(step)

    # 对于int64类型，截断参数到整数再计算长度，其他整数类型不需要这样做
    if dtype == torch.int64:
        # 使用floordiv避免ceil
        sgn = bool(xstep > 0) - bool(xstep < 0)  # type: ignore[possibly-undefined]
        length = (xend - xstart + xstep - sgn) // xstep  # type: ignore[possibly-undefined]
    else:
        # 对于其他数据类型，使用ceil计算长度
        length = math.ceil((end - start) / step)

    # 如果数据类型为整数类型，则调用prims.iota函数
    if is_integer:
        return prims.iota(
            length,
            start=xstart,  # type: ignore[possibly-undefined]
            step=xstep,  # type: ignore[possibly-undefined]
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    # 如果数据类型不是整数类型，则使用默认的int64类型调用prims.iota函数
    index = prims.iota(
        length,
        start=0,
        step=1,
        dtype=torch.int64,
        device=device,
        requires_grad=False,
    )

    # 计算使用的计算数据类型，如果参数都是整数则为torch.long，否则根据dtype和device获取
    computation_dtype = (
        torch.long if integer_args else utils.get_acc_type(dtype, device)
    )
    # 将index转换为计算数据类型
    index = _maybe_convert_to_dtype(index, computation_dtype)
    # 计算结果
    result = start + step * index
    # 将结果转换为指定的数据类型
    result = _maybe_convert_to_dtype(result, dtype)
    # 如果 requires_grad 参数为 True，则设置 result 张量的 requires_grad 属性为 True
    if requires_grad:
        result.requires_grad_(True)
    # 返回经处理后的 result 张量
    return result
@register_decomposition(aten.lerp)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("start", "end", "weight"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义线性插值函数，用于在张量之间按照权重进行插值计算
def lerp(start: Tensor, end: Tensor, weight: Union[Tensor, NumberType]):
    # 将输入张量统一放入列表
    inputs = [start, end]
    # 如果权重是数字类型，则将其转换为与 start 相同的张量类型
    if isinstance(weight, Number):
        weight = start.new_full((), weight)  # type: ignore[arg-type]
    else:
        inputs.append(weight)
    # 使用 assert 语句确保 weight 是张量类型，用于类型检查
    assert isinstance(weight, Tensor)  # mypy

    # 为了数值稳定性，采用这种实现方式。假设在稳定性优化中，我们有 0 <= weight <= 1。使用 abs 来处理复数
    # 我们希望在浮点数最精确的地方（接近零点）执行操作
    # 因此，我们执行以下优化：
    # 如果 weight.abs() >= 0.5:
    #    return (1 - weight) * (start - end) + end
    mask = weight.abs() >= 0.5
    coeff = torch.where(mask, weight - 1, weight)
    base = torch.where(mask, end, start)
    output = coeff * (end - start) + base

    # 确保分解输出的步长与非分解路径相同
    stride = utils.compute_elementwise_output_strides(*_maybe_broadcast(*inputs))
    if output.stride() != stride:
        output = prims.copy_strided(output, stride)

    # 返回处理非连续输出的函数结果
    return handle_noncontiguous_outputs(inputs, output)


@register_decomposition(aten.linspace)
@out_wrapper()
# 定义生成等间距序列的函数
def linspace(
    start: Union[NumberType, TensorLikeType],
    end: Union[NumberType, TensorLikeType],
    steps: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: torch.layout = torch.strided,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    # 如果 start 是张量类型，确保其维度为 0
    if isinstance(start, TensorLikeType):
        torch._check(
            start.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )
        start = _maybe_convert_to_dtype(start, torch.float64)
    # 如果 end 是张量类型，确保其维度为 0
    if isinstance(end, TensorLikeType):
        torch._check(
            end.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )
        end = _maybe_convert_to_dtype(end, torch.float64)

    # 如果 start、end 或 steps 中有复数，则推断出默认的复数 dtype
    if py_any(isinstance(arg, complex) for arg in (start, end, steps)):
        default_complex_dtype = utils.corresponding_complex_dtype(
            torch.get_default_dtype()
        )
        # 如果未指定 dtype，则使用推断出的默认复数 dtype
        if dtype is None:
            dtype = default_complex_dtype
        else:
            # 否则，确保指定的 dtype 是复数 dtype
            torch._check(
                utils.is_complex_dtype(dtype),
                lambda: f"linspace(): inferred dtype {default_complex_dtype} can't be safely cast to passed dtype {dtype}",
            )
    else:
        # 否则，dtype 使用默认 dtype 或者用户指定的 dtype
        dtype = dtype or torch.get_default_dtype()
    # 使用 assert 语句确保 dtype 是 torch 的合法 dtype
    assert isinstance(dtype, torch.dtype)

    # steps 在 dtype 计算中不参与
    torch._check_type(
        # 使用 torch 模块中的 _check_type 函数检查 steps 是否为 IntLike 类型
        isinstance(steps, IntLike),
        # 如果 steps 不是 IntLike 类型，则返回相应的错误消息
        lambda: f"received an invalid combination of arguments - got {type(steps).__name__} (expected IntLike)"
    ({type(start).__name__}, {type(end).__name__}, {type(steps).__name__})",
    )
    # 检查 steps 是否为 IntLike 类型（用于类型检查）
    assert isinstance(steps, IntLike)  # for mypy
    # 确保 steps 大于等于 0，否则引发异常
    torch._check(steps >= 0, lambda: "number of steps must be non-negative")

    # 构建 factory_kwargs 字典，包含创建张量所需的参数
    factory_kwargs = {
        "layout": layout,
        "device": device,
        "pin_memory": pin_memory,
        "requires_grad": requires_grad,
    }
    # 如果 steps 为 0，则返回一个空的张量，dtype 为指定的类型，其他参数从 factory_kwargs 获取
    if steps == 0:
        return torch.full((0,), 0, dtype=dtype, **factory_kwargs)  # type: ignore[arg-type]
    # 如果 steps 为 1，则根据 start 的类型创建张量，或者使用 start 值填充，其他参数从 factory_kwargs 获取
    if steps == 1:
        if isinstance(start, TensorLikeType):
            return torch.empty((steps,), dtype=dtype, **factory_kwargs).copy_(start)  # type: ignore[arg-type]
        else:
            return torch.full((steps,), start, dtype=dtype, **factory_kwargs)  # type: ignore[arg-type]

    # 使用 torch.arange 创建一个范围为 [0, steps) 的整数张量 rg，其他参数从 factory_kwargs 获取
    # 如果类型不支持，忽略类型提示
    rg = torch.arange(0, steps, **factory_kwargs)  # type: ignore[arg-type]

    # 根据 dtype 的类型选择计算时所需的精度
    dtype_red = (
        torch.int64
        if (utils.is_boolean_dtype(dtype) or utils.is_integer_dtype(dtype))
        else dtype
    )
    # 根据 rg 张量和 dtype_red 计算出计算所需的 dtype
    computation_dtype, _ = utils.reduction_dtypes(
        rg, REDUCTION_OUTPUT_TYPE_KIND.SAME, dtype_red
    )
    # 部分函数应用，转换 rg 张量为 computation_dtype 类型
    cast_rg = partial(_maybe_convert_to_dtype, dtype=computation_dtype)

    # 实现 torch.lerp，避免直接使用 rg / (steps - 1)
    # 保证 out[0] == start, out[-1] == end
    step = (end - start) / (steps - 1)
    # 使用 torch.where 在 rg < steps / 2 时计算 start + step * cast_rg(rg)，否则计算 end - step * cast_rg((steps - 1) - rg)
    out = torch.where(
        rg < steps / 2,
        start + step * cast_rg(rg),  # type: ignore[arg-type,operator]
        end - step * cast_rg((steps - 1) - rg),  # type: ignore[arg-type,operator]
    )
    # 将 out 张量转换为指定的 dtype 类型
    return _maybe_convert_to_dtype(out, dtype)  # type: ignore[return-value]


@register_decomposition(aten.logspace)
@out_wrapper()
def logspace(
    start: Union[NumberType, TensorLikeType],
    end: Union[NumberType, TensorLikeType],
    steps: NumberType,
    base: NumberType = 10,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: torch.layout = torch.strided,
    pin_memory: bool = False,
    requires_grad: bool = False,
) -> TensorLikeType:
    if dtype is None:
        dtype = torch.get_default_dtype()

    # 注意：NumPy 中没有这种类型转换
    # 检查 dtype 是否为整数类型
    if prims.utils.is_integer_dtype(dtype):
        # 如果 start 是 FloatLike 类型，则转换为对应的符号整数
        if isinstance(start, FloatLike):
            start = sym_int(start)
        # 如果 start 是 TensorLikeType 类型，则进行维度检查和类型转换
        elif isinstance(start, TensorLikeType):
            torch._check(
                start.dim() == 0,
                lambda: "logspace only supports 0-dimensional start and end tensors",
            )
            start = _maybe_convert_to_dtype(start, dtype)
        # 如果 end 是 FloatLike 类型，则转换为对应的符号整数
        if isinstance(end, FloatLike):
            end = sym_int(end)
        # 如果 end 是 TensorLikeType 类型，则进行维度检查和类型转换
        elif isinstance(end, TensorLikeType):
            torch._check(
                end.dim() == 0,
                lambda: "logspace only supports 0-dimensional start and end tensors",
            )
            end = _maybe_convert_to_dtype(end, dtype)

    # 检查 start、end、steps 中是否有复数类型的参数
    if py_any(isinstance(arg, complex) for arg in (start, end, steps)):
        # 获取默认的复数数据类型并设置为 dtype
        default_complex_dtype = utils.corresponding_complex_dtype(
            torch.get_default_dtype()
        )
        dtype = default_complex_dtype
        _dtype = None  # torch.linspace 将会更新正确的数据类型
    else:
        _dtype = torch.float64

    # 断言 base 不是复数类型（仅用于类型检查）
    assert not isinstance(base, complex)  # for mypy
    # 如果 base 小于 0，则抛出 NotImplementedError
    if base < 0:
        raise NotImplementedError
    # 使用 torch.linspace 生成等间隔的一维张量
    ret = torch.linspace(  # type: ignore[misc]
        start,  # type: ignore[arg-type] - 起始值
        end,  # type: ignore[arg-type] - 终止值
        steps,  # type: ignore[arg-type] - 步数
        dtype=_dtype,  # 指定数据类型
        layout=layout,  # 布局
        device=device,  # 设备
        pin_memory=pin_memory,  # 是否使用固定内存
        requires_grad=requires_grad,  # 是否需要梯度
    )
    # 将生成的一维张量应用指数运算，并可能转换为指定的数据类型
    return _maybe_convert_to_dtype(torch.pow(base, ret), dtype)  # type: ignore[arg-type,return-value]
@overload
def meshgrid(tensors: Sequence[TensorLikeType], indexing: str):
    pass


@overload
def meshgrid(*tensors: TensorLikeType, indexing: str):
    pass


@register_decomposition(aten.meshgrid)
def meshgrid(
    *tensors: Union[TensorLikeType, List[TensorLikeType], Tuple[TensorLikeType]],
    indexing: str,
) -> List[TensorLikeType]:
    # 此函数同时处理了两种重载情况（见上面的函数声明）
    # `indexing` 参数目前对于 torch.meshgrid 是可选的，但我们计划将其变为必选：https://github.com/pytorch/pytorch/issues/50276
    if isinstance(tensors[0], (list, tuple)):
        assert len(tensors) == 1
        tensors = tuple(tensors[0])

    torch._check(
        py_all(isinstance(a, TensorLike) for a in tensors),
        lambda: "meshgrid expects its inputs to be tensors",
    )

    torch._check(len(tensors) > 0, lambda: "meshgrid expects a non-empty TensorList")

    for i in range(len(tensors) - 1):
        torch._check(
            tensors[i].dtype == tensors[i + 1].dtype,  # type: ignore[union-attr]
            lambda: "meshgrid expects all tensors to have the same dtype",
        )
        torch._check(
            tensors[i].device == tensors[i + 1].device,  # type: ignore[union-attr]
            lambda: "meshgrid expects all tensors to have the same device",
        )

    swap_first_and_second_tensors = False
    if indexing == "xy":
        swap_first_and_second_tensors = len(tensors) >= 2
        if swap_first_and_second_tensors:
            tensors = (tensors[1], tensors[0], *tensors[2:])
    else:
        torch._check(
            indexing == "ij",
            lambda: (
                'torch.meshgrid: indexing must be one of "xy" or "ij", '
                f"but received: {indexing}"
            ),
        )

    result_shape: List[int] = []
    for t in tensors:
        assert isinstance(t, TensorLike)  # mypy
        torch._check(
            t.ndim == 0 or t.ndim == 1,
            lambda: f"torch.meshgrid: Expected 0D or 1D tensor in the tensor list but got: {t}",
        )
        result_shape.append(t.numel())

    grids: List[TensorLikeType] = []
    for i, t in enumerate(tensors):
        assert isinstance(t, TensorLike)  # mypy
        if t.ndim == 0:
            t = t.view((1,))
        grids.append(prims.broadcast_in_dim(t, result_shape, (i,)))

    if swap_first_and_second_tensors:
        # 如果最初在开头交换了张量，则交换输出
        grids[0], grids[1] = grids[1], grids[0]

    return grids


# CompositeImplicitAutograd - 不注册分解
def movedim(
    input: TensorLikeType,
    source: Union[int, DimsSequenceType],
    destination: Union[int, DimsSequenceType],
) -> TensorLikeType:
    """
    torch.movedim 的参考实现
    """
    if type(source) is int:
        source = (source,)
    if type(destination) is int:
        destination = (destination,)
    # 使用 torch._check() 函数检查源和目标张量的维度是否相等，生成兼容于 PyTorch 的错误消息格式，会打印带方括号的序列。
    torch._check(
        len(source) == len(destination),  # 检查源和目标张量的维度是否相等
        lambda: (
            "movedim: Invalid source or destination dims: source "  # 错误消息的一部分，描述维度不匹配的情况
            f"({list(source)} dims) should contain the same number "  # 拼接错误消息，显示源张量的维度
            f"of dims as destination ({list(destination)} dims)"  # 继续拼接错误消息，显示目标张量的维度
        ),
    )

    # 获取输入张量的维度
    rank = input.ndim

    # 将源索引通过 utils.canonicalize_dims() 规范化为元组
    ss = tuple(utils.canonicalize_dims(rank=rank, indices=source))  # 规范化源索引的维度

    # 将目标索引通过 utils.canonicalize_dims() 规范化为元组
    ds = tuple(utils.canonicalize_dims(rank=rank, indices=destination))  # 规范化目标索引的维度

    # 将规范化后的源索引转换为集合
    sss = set(ss)

    # 将规范化后的目标索引转换为集合
    dss = set(ds)

    # 再次使用 torch._check() 函数检查规范化后的源索引是否存在重复
    torch._check(
        len(ss) == len(sss),
        lambda: f"movedim: repeated dim in `source` ({list(source)})",  # 错误消息，指示源索引中存在重复的维度
    )

    # 检查规范化后的目标索引是否存在重复
    torch._check(
        len(ds) == len(dss),
        lambda: f"movedim: repeated dim in `destination` ({list(destination)})",  # 错误消息，指示目标索引中存在重复的维度
    )

    # 创建一个从目标索引到源索引的映射字典
    m = dict(zip(ds, ss))

    # 初始化一个空列表用于存储最终的维度顺序
    dims = []

    si = 0  # 源索引的起始值

    # 遍历输入张量的维度
    for di in range(rank):
        # 检查目标索引是否在映射字典中
        s = m.get(di)
        if s is not None:
            # 如果在映射字典中找到目标索引，则将对应的源索引添加到维度列表中
            dims.append(s)
        else:
            # 如果目标索引不在映射字典中，则顺序添加源索引到维度列表中，跳过映射字典中已有的索引
            while si in sss:
                si += 1
            dims.append(si)
            si += 1

    # 使用 torch.permute() 函数重新排列输入张量的维度，按照 dims 中的顺序
    result = torch.permute(input, tuple(dims))

    # 返回重新排列后的结果张量
    return result
# 使用装饰器注册对aten.empty_strided函数的分解方法，并且在输出中包装结果
# 使用装饰器注册对aten.empty_strided函数的分解方法，并且在输出中包装结果
@register_decomposition(aten.empty_strided)
@out_wrapper()
def empty_strided(
    shape: Union[ShapeType, Tuple[ShapeType]],  # shape参数可以是一个整数元组或包含整数元组的元组
    strides: StrideType,  # strides参数的类型为StrideType
    *,
    dtype: Optional[torch.dtype] = None,  # dtype参数是一个可选的torch.dtype
    device: Optional[DeviceLikeType] = None,  # device参数是一个可选的设备类型
    layout: torch.layout = torch.strided,  # layout参数默认为torch.strided
    requires_grad: bool = False,  # requires_grad参数默认为False
    pin_memory: bool = False,  # pin_memory参数默认为False
) -> TensorLikeType:  # 返回类型为TensorLikeType

    # 检查layout是否为strided，如果不是则会引发异常
    utils.check_layout(layout)
    # 检查pin_memory是否为False，如果不是则会引发异常
    utils.check_pin_memory(pin_memory)

    # 从变长参数中提取shape的实际形状
    shape = utils.extract_shape_from_varargs(shape)
    # 如果dtype为None，则使用默认的torch数据类型
    dtype = torch.get_default_dtype() if dtype is None else dtype
    # 如果device为None，则默认使用CPU设备
    device = torch.device("cpu") if device is None else device

    # 调用prims.empty_strided函数来创建空的张量
    return prims.empty_strided(
        shape,
        strides,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


# 使用装饰器注册对aten.eye函数的分解方法，并且在输出中包装结果
@register_decomposition(aten.eye)
@out_wrapper()
def eye(
    n: int,  # n参数表示行数，必须为整数
    m: Optional[int] = None,  # m参数表示列数，可以为None，默认与n相同
    *,
    dtype: Optional[torch.dtype] = None,  # dtype参数是一个可选的torch.dtype
    layout: torch.layout = torch.strided,  # layout参数默认为torch.strided
    device: Optional[DeviceLikeType] = None,  # device参数是一个可选的设备类型
    pin_memory: bool = False,  # pin_memory参数默认为False
    requires_grad: bool = False,  # requires_grad参数默认为False，未使用
) -> TensorLikeType:  # 返回类型为TensorLikeType

    """
    Reference implementation of torch.eye
    """

    # 如果m为None，则将m设置为n，确保创建方阵
    if m is None:
        m = n

    # 检查n是否大于等于0，否则引发异常
    torch._check(n >= 0, lambda: f"n must be greater or equal to 0, got {n}")
    # 检查m是否大于等于0，否则引发异常
    torch._check(m >= 0, lambda: f"m must be greater or equal to 0, got {m}")

    # 创建设备为device的整数张量range_n和range_m
    range_n = torch.arange(n, dtype=torch.int64, device=device, requires_grad=False)
    range_m = torch.arange(m, dtype=torch.int64, device=device, requires_grad=False)

    # 创建条件张量，表示行索引与列索引相等的位置为True，否则为False
    cond = range_n.unsqueeze(-1) == range_m

    # 如果dtype为torch.bool，则直接返回条件张量
    if dtype is torch.bool:
        return cond
    else:
        # 否则创建值为1的张量one，其形状为(1,)
        one = torch.ones(
            (1,),
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=False,
        )
        # 使用torch.where根据条件张量填充one或0，生成单位矩阵
        return torch.where(cond, one, 0)

    # TODO: 使用requires_grad。所有使用requires_grad参数的引用必须返回一个叶子张量。
    # result.requires_grad_(requires_grad)


# 使用装饰器注册对aten.full.default和aten.full.out函数的分解方法，并且在输出中包装结果
@register_decomposition([aten.full.default, aten.full.out])
@out_wrapper()
def full(
    shape: ShapeType,  # shape参数表示张量的形状
    fill_value: NumberType,  # fill_value参数表示填充的值，可以是数字类型
    *,
    dtype: Optional[torch.dtype] = None,  # dtype参数是一个可选的torch.dtype
    layout: torch.layout = torch.strided,  # layout参数默认为torch.strided
    device: Optional[DeviceLikeType] = None,  # device参数是一个可选的设备类型
    pin_memory: bool = False,  # pin_memory参数默认为False
    requires_grad: bool = False,  # requires_grad参数默认为False，未使用
) -> TensorLikeType:  # 返回类型为TensorLikeType

    # 检查layout是否为strided，如果不是则会引发异常
    utils.check_layout(layout)
    # 检查pin_memory是否为False，如果不是则会引发异常
    utils.check_pin_memory(pin_memory)

    # 如果dtype为None，则使用fill_value的类型作为dtype
    dtype = dtype if dtype is not None else utils.type_to_dtype(type(fill_value))
    # 如果device为None，则默认使用CPU设备
    device = device if device is not None else torch.device("cpu")

    # 创建一个与shape、dtype、layout和device匹配的空张量e
    e = empty(
        shape,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
    )
    # 使用fill_value填充张量e，生成填充后的张量并返回
    return torch.fill(e, fill_value)  # type: ignore[arg-type]
    a: TensorLikeType,
    # 参数a：表示接受的数据类型应符合TensorLikeType，即类似于张量的类型

    fill_value: NumberType,
    # 参数fill_value：表示填充值的数据类型应符合NumberType，即数字类型

    *,
    # 以下参数为关键字参数，必须使用关键字调用，不能按位置传递

    dtype: Optional[torch.dtype] = None,
    # 关键字参数dtype：张量的数据类型，默认为None，由torch.dtype类型控制

    layout: Optional[torch.layout] = None,
    # 关键字参数layout：张量的布局，默认为None，由torch.layout类型控制

    device: Optional[DeviceLikeType] = None,
    # 关键字参数device：张量的设备，默认为None，由DeviceLikeType类型控制

    pin_memory: bool = False,
    # 关键字参数pin_memory：是否将张量存储在锁页内存中，默认为False

    requires_grad: bool = False,
    # 关键字参数requires_grad：是否要在计算中跟踪梯度，默认为False

    memory_format: torch.memory_format = torch.preserve_format,
    # 关键字参数memory_format：张量的存储格式，默认为torch.preserve_format，由torch.memory_format类型控制
# 返回一个形状与给定张量 `a` 相同的空张量，填充方式由后续函数决定
def zeros_like(
    a: TensorLikeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> TensorLikeType:
    # 根据给定张量 `a` 的属性创建一个空张量 `e`
    e = torch.empty_like(
        a,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        memory_format=memory_format,
    )
    # 调用 `fill` 函数填充张量 `e`，并返回结果
    return fill(e, fill_value)


# 返回一个形状与给定张量 `a` 相同的张量，且所有元素填充为 1
def ones_like(
    a: TensorLikeType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> TensorLikeType:
    # 根据给定张量 `a` 的属性创建一个全为 1 的张量
    return torch.full_like(
        a,
        True if (dtype or a.dtype) == torch.bool else 1,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        memory_format=memory_format,
    )


# 返回一个具有指定形状的正态分布随机张量
def randn(
    *shape,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: Optional[torch.layout] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> TensorLikeType:
    # 检查是否需要将张量固定在内存中
    utils.check_pin_memory(pin_memory)

    # 从可变参数中提取形状信息
    shape_ = utils.extract_shape_from_varargs(shape)

    # 确定数据类型，如果未指定则使用默认值
    dtype = utils.dtype_or_default(dtype)
    # 确定设备，如果未指定则使用默认值
    device = utils.device_or_default(device)

    # 返回一个形状为 `shape_` 的正态分布随机张量
    return prims.normal(
        shape_,
        mean=0.0,
        std=1.0,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


# 返回一个标量值转换成的张量
def scalar_tensor(
    a: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[DeviceLikeType] = None,
    pin_memory: bool = False,
) -> TensorLikeType:
    # 检查张量的布局是否有效
    utils.check_layout(layout)
    # 检查是否需要将张量固定在内存中
    utils.check_pin_memory(pin_memory)
    # 如果未指定数据类型，则根据输入标量 `a` 的类型确定数据类型
    dtype = dtype if dtype is not None else utils.type_to_dtype(type(a))
    # 如果未指定设备，则默认使用 CPU
    device = device if device is not None else torch.device("cpu")
    # 返回一个包含指定标量值的张量
    return prims.scalar_tensor(a, dtype=dtype, device=device)


# 辅助函数：返回一个具有指定形状的均匀分布随机张量
def _uniform_helper(
    shape: ShapeType,
    low: Union[bool, int, float] = 0.0,
    high: Union[bool, int, float] = 1.0,
    *,
    dtype: torch.dtype,
    device: DeviceLikeType,
) -> TensorLikeType:
    # 验证给定形状是否有效
    utils.validate_shape(shape)
    # 断言low是Number类型
    assert isinstance(low, Number)
    # 断言high是Number类型
    assert isinstance(high, Number)
    # 将low转换为符号化浮点数
    low = sym_float(low)
    # 将high转换为符号化浮点数
    high = sym_float(high)
    
    # 断言dtype是torch的数据类型
    assert isinstance(dtype, torch.dtype)
    # 规范化设备字符串，返回标准化后的设备名称
    device = utils.canonicalize_device(device)
    
    # 调用prims模块的_uniform_helper函数，传入参数shape, low, high, dtype, device，并返回其结果
    return prims._uniform_helper(shape, low=low, high=high, dtype=dtype, device=device)
# 注册函数`aten.masked_fill`的装饰器
# 注册函数`out_wrapper`的装饰器
@register_decomposition(aten.masked_fill)
@out_wrapper()
def masked_fill(a: TensorLikeType, mask: TensorLikeType, value: TensorOrNumberLikeType):
    # 将张量 `a` 的数据类型转换为对应的 Python 类型
    python_type = utils.dtype_to_type(a.dtype)
    # 如果 `value` 是数字类型，则获取其类型
    if isinstance(value, Number):
        value_type = type(value)
    else:
        # 如果 `value` 不是数字类型，则进行以下检查和操作
        # 注意：尝试使用 `item(value)` 会导致 RuntimeError: Cannot cast FakeTensor(cpu) to number
        # 获取 `value` 的维度信息
        value_ndim = value.ndim
        # 检查 `value` 是否是 0 维张量，否则抛出异常
        torch._check(
            value_ndim == 0,
            lambda: f"only supports a 0-dimensional value tensor, but got tensor with {value_ndim} dimension",
        )
        # 检查是否允许将 CPU 标量移动到 cuda 或 xpu，但不允许其他情况
        is_cpu_scalar = (
            a.device.type in ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]
            and value.device.type == "cpu"
        )
        # 检查 `value` 是否与 `a` 在同一设备上，否则抛出异常
        torch._check(
            is_cpu_scalar or value.device == a.device,
            lambda: "Expected `value` to be on same device as `a`",
        )
        # 获取 `value` 的数据类型，并转换为对应的 Python 类型
        value_type = utils.dtype_to_type(value.dtype)

    # 如果 `value_type` 是复数类型，则进行以下检查
    if value_type is complex:
        # 仅允许从复数类型向较低类型的转换
        # 允许其他情况下将 `value` 转换为较低类型，例如 float -> int
        # 参考: https://github.com/pytorch/pytorch/issues/79195
        torch._check(
            utils.is_weakly_lesser_type(value_type, python_type),
            lambda: f"could not convert to type {python_type} without overflow",
        )

    # 将 `value` 转换为与 `a` 相同的数据类型，以便传递给 `torch.where`
    value = _maybe_convert_to_dtype(value, a.dtype)
    # 使用 `torch.where` 函数根据 `mask` 条件填充 `value` 或者保留 `a` 的值
    r = torch.where(mask, value, a)  # type: ignore[arg-type]

    # `aten.mask_fill` 函数始终返回一个新的连续张量
    # 使用 `contiguous()` 方法确保输出张量的步幅设置正确
    return r.contiguous()


# 注册函数`aten.masked_fill_`的装饰器
@register_decomposition(aten.masked_fill_)
def masked_fill_(
    a: TensorLikeType, mask: TensorLikeType, value: TensorOrNumberLikeType
) -> TensorLikeType:
    # 使用 `torch.masked_fill` 函数填充 `a` 根据 `mask` 和 `value` 的条件
    b = torch.masked_fill(a, mask, value)  # type: ignore[arg-type]
    # 将 `b` 的值复制到 `a` 中
    a.copy_(b)
    return a


# `CompositeImplicitAutograd` 类 - 不注册分解
# `torch.allclose` 的参考实现
def allclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Reference implementation of torch.allclose
    """
    # 检查 `torch.allclose` 的参数
    _check_close_args(name="torch.allclose", a=a, b=b, rtol=rtol, atol=atol)

    # 调用 `torch.isclose` 函数检查 `a` 和 `b` 是否在给定的容差范围内相等
    close_tensor = torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    # 检查 `close_tensor` 中的所有元素是否都为 `True`，并返回布尔值
    return bool(torch.all(close_tensor).item())


# 比较两个张量 `a` 和 `b` 是否完全相等
def equal(a: TensorLikeType, b: TensorLikeType) -> bool:
    # 检查 `a` 和 `b` 是否位于相同的设备上，不允许 CPU 标量张量
    utils.check_same_device(a, b, allow_cpu_scalar_tensors=False)
    # 检查 `a` 和 `b` 的数据类型是否相同
    utils.check_same_dtype(a, b)

    # 检查张量的形状是否相同
    if a.ndim != b.ndim:
        return False

    # 逐一比较张量的每个维度是否相等
    for x, y in zip(a.shape, b.shape):
        if x != y:
            return False

    # 如果张量中没有元素，则直接返回 `True`
    if a.numel() == 0:
        return True
    # 调用函数 `all`，传入一个生成器表达式 `eq(a, b)` 的所有结果作为参数
    # `eq(a, b)` 是一个比较函数，判断两个对象 `a` 和 `b` 是否相等
    # `all` 函数会返回一个布尔值，表示是否所有比较都为真
    # 最终返回 `item` 函数的结果，`item` 是一个接收参数并返回结果的函数
    # `# type: ignore[return-value]` 是类型提示语法，用来指示忽略对函数返回值类型的检查
    return item(all(eq(a, b)))
# 将 norm 函数注册为 aten.norm 的分解函数，确保输出的数据类型与输入一致
# 使用 out_wrapper 装饰器包装函数，确保输出精确匹配指定的数据类型
@register_decomposition(aten.norm)
@out_wrapper(exact_dtype=True)
def norm(
    input: TensorLikeType,
    p: Optional[Union[float, str]] = "fro",
    dim: Optional[DimsType] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # 在以下情况下计算 Frobenius 范数
    if (
        p == "fro" and (dim is None or isinstance(dim, Dim) or len(dim) <= 2)
    ) or p is None:
        p = 2
    if isinstance(dim, Dim):
        dim = [dim]
    if isinstance(p, str):
        # 如果维度 dim 为 None，则设置为所有输入张量的维度范围
        if dim is None:
            dim = tuple(range(input.ndim))
        # 调用 torch.linalg.matrix_norm 计算矩阵的核范数，或者调用 matrix_norm
        # 以某些参数调用，这将导致错误
        return torch.linalg.matrix_norm(input, p, dim, keepdim, dtype=dtype)
    else:
        # 调用 torch.linalg.vector_norm 计算向量的范数
        return torch.linalg.vector_norm(input, p, dim, keepdim, dtype=dtype)


# 将 trace 函数注册为 aten.trace 的分解函数
# 使用 out_wrapper 装饰器包装函数
@register_decomposition(aten.trace)
@out_wrapper()
def trace(self: TensorLikeType) -> TensorLikeType:
    # 检查输入张量是否为二维，否则抛出错误信息
    torch._check(
        self.ndim == 2, lambda: "expected a matrix, but got tensor with dim {self.ndim}"
    )
    # 返回对角线元素的和
    return torch.sum(torch.diag(self, 0))


# 创建一个基于 base_op 的右侧二元操作函数
def _make_r_binary_op(base_op):
    def rop(
        a: Union[TensorLikeType, NumberType],
        b: Union[TensorLikeType, NumberType],
    ) -> TensorLikeType:
        return base_op(b, a)

    return rop


# 创建 rtruediv 函数，其功能类似于 true_divide 的右侧二元操作
rtruediv = _make_r_binary_op(true_divide)
# 创建 rfloordiv 函数，其功能类似于 floor_divide 的右侧二元操作
rfloordiv = _make_r_binary_op(floor_divide)
# 创建 rpow 函数，其功能类似于 pow 的右侧二元操作
rpow = _make_r_binary_op(pow)


# 将 triu 函数注册为 aten.triu 的分解函数
# 使用 out_wrapper 装饰器包装函数
@register_decomposition(aten.triu)
@out_wrapper()
def triu(a: TensorLikeType, diagonal: int = 0) -> TensorLikeType:
    # 检查输入张量是否至少为二维，否则抛出错误信息
    torch._check(
        a.ndim >= 2, lambda: "triu: input tensor must have at least 2 dimensions"
    )
    h, w = a.shape[-2:]
    # 创建一个掩码，选择位于主对角线之上的元素
    mask = (
        torch.arange(w, device=a.device).unsqueeze(-2)
        - torch.arange(h, device=a.device).unsqueeze(-1)
    ) >= diagonal

    # aten.triu 总是返回一个新的连续张量
    # contiguous() 用于正确模拟输出步幅
    return utils.mask_tensor(mask, a).contiguous()


# 将 tril 函数注册为 aten.tril 的分解函数
# 使用 out_wrapper 装饰器包装函数
@register_decomposition(aten.tril)
@out_wrapper()
def tril(a: TensorLikeType, diagonal: int = 0) -> TensorLikeType:
    # 检查输入张量是否至少为二维，否则抛出错误信息
    torch._check(
        a.ndim >= 2, lambda: "tril: input tensor must have at least 2 dimensions"
    )
    h, w = a.shape[-2:]
    # 创建一个掩码，选择位于主对角线之下的元素
    mask = (
        torch.arange(w, device=a.device).unsqueeze(-2)
        - torch.arange(h, device=a.device).unsqueeze(-1)
    ) <= diagonal

    # aten.tril 总是返回一个新的连续张量
    # contiguous() 用于正确模拟输出步幅
    return utils.mask_tensor(mask, a).contiguous()


# 基于 aten/src/ATen/native/TensorFactories.h 中的 get_tril_size 实现
# 矩阵的下三角形成一个五边形，可以分解为一个上梯形和一个下矩形的形式
# 对于 tril_indices 的实现，我们需要这两者的大小，以及梯形顶边的长度。
# 如果行数或列数为零，则返回三个零值
def _get_tril_sizes(row: int, col: int, offset: int) -> Tuple[int, int, int]:
    if row == 0 or col == 0:
        return 0, 0, 0

    # 计算顶部梯形的第一行的最小值
    m_first_row = min(col, 1 + offset) if offset > 0 else int(row + offset > 0)
    # 计算顶部梯形的最后一行
    m_last_row = max(0, min(col, row + offset))
    # 所有行的数量
    n_row_all = max(0, min(row, row + offset))
    # 梯形区域的行数
    n_row_trapezoid = m_last_row - m_first_row + 1

    # 计算顶部梯形的元素数量
    trapezoid_size = (m_first_row + m_last_row) * n_row_trapezoid // 2
    # 计算底部矩形的元素数量
    diff_row = n_row_all - n_row_trapezoid
    rectangle_size = max(0, diff_row * col)

    return trapezoid_size, rectangle_size, m_first_row


# 对于给定的参数进行检查，确保行和列为非负数
def _trilu_checks(
    name: str,
    row: int,
    col: int,
    dtype: torch.dtype,
    layout: torch.layout,
    pin_memory: bool,
):
    torch._check(row >= 0, lambda: f"row must be non-negative, got {row}")
    torch._check(col >= 0, lambda: f"col must be non-negative, got {col}")
    torch._check(
        dtype in (torch.int32, torch.int64),
        lambda: f"\"{name}\" not implemented for '{dtype}'",
    )


# 这基于aten/src/ATen/native/cuda/TensorFactories.cu中的tril_indices_cuda
# 使用注册的分解函数和输出包装器
@register_decomposition(aten.tril_indices)
@out_wrapper()
def tril_indices(
    row: int,
    col: int,
    offset: int = 0,
    *,
    dtype: torch.dtype = torch.long,
    layout: torch.layout = torch.strided,
    device: DeviceLikeType = "cpu",
    pin_memory: bool = False,
) -> TensorLikeType:
    _trilu_checks("tril_indices", row, col, dtype, layout, pin_memory)

    # 调用_get_tril_sizes函数获取梯形和矩形的大小以及顶部梯形的第一行
    trapezoid_size, rectangle_size, m_first_row = _get_tril_sizes(row, col, offset)
    # 计算行的偏移量
    row_offset = max(0, -offset)

    # 使用partial函数设置arange_kw为torch.arange的部分应用，指定布局、设备和是否使用固定内存
    arange_kw = partial(
        torch.arange, layout=layout, device=device, pin_memory=pin_memory
    )

    # 首先处理顶部梯形的索引
    xs1 = arange_kw(0, trapezoid_size, dtype=torch.float64)
    b = m_first_row - 0.5
    row_inds1 = torch.floor(-b + torch.sqrt(b * b + 2 * xs1))
    col_inds1 = torch.floor(xs1 - (2 * m_first_row - 1 + row_inds1) * row_inds1 * 0.5)
    row_inds1 = _maybe_convert_to_dtype(row_inds1 + row_offset, dtype)
    col_inds1 = _maybe_convert_to_dtype(col_inds1, dtype)

    # 然后处理底部矩形的索引
    xs2 = arange_kw(0, rectangle_size, dtype=dtype)
    row_inds2 = xs2 // col + (col - m_first_row + 1 + row_offset)
    col_inds2 = xs2 % col

    # 返回堆叠的张量，包括顶部梯形和底部矩形的行和列索引
    return torch.stack(
        (torch.cat((row_inds1, row_inds2)), torch.cat((col_inds1, col_inds2)))
    )


# 类似于上面的_get_tril_sizes函数，但这里有一个顶部矩形和底部梯形
# 注意不能简化为_get_tril_sizes(col, row, -offset)，因为那会对应于将其分解为左侧梯形和右侧矩形
def _get_triu_sizes(row: int, col: int, offset: int) -> Tuple[int, int, int]:
    if row == 0 or col == 0:
        return 0, 0, 0

    # 计算顶部矩形的第一行
    m_first_row = max(0, col - offset) if offset > 0 else col

    # 返回顶部矩形、底部梯形和第一行的元组
    return 0, 0, 0  # 在此处添加注释
    # 计算矩形的面积，若行偏移为负则取0，行数不超过偏移的最小值与列数的乘积
    rectangle_size = max(0, min(row, -offset) * col)
    
    # 获取底部梯形的元素数量
    trapezoid_size_tril, rectangle_size_tril, _ = _get_tril_sizes(row, col, offset - 1)
    # 计算上三角形部分的元素数量
    triu_size = row * col - (trapezoid_size_tril + rectangle_size_tril)
    # 计算梯形的元素数量，即上三角形元素数量减去矩形的元素数量
    trapezoid_size = triu_size - rectangle_size
    
    # 返回计算得到的梯形大小、矩形大小和 m_first_row
    return trapezoid_size, rectangle_size, m_first_row
# 注册 triu_indices 函数的分解方法
# 使用 out_wrapper 装饰器对函数进行装饰，可能处理输出
def triu_indices(
    row: int,
    col: int,
    offset: int = 0,
    *,
    dtype: torch.dtype = torch.long,
    layout: torch.layout = torch.strided,
    device: DeviceLikeType = "cpu",
    pin_memory: bool = False,
) -> TensorLikeType:
    # 对输入参数进行检查，确保参数合法性
    _trilu_checks("triu_indices", row, col, dtype, layout, pin_memory)

    # 获取上三角区域、矩形区域和首行的大小信息
    trapezoid_size, rectangle_size, m_first_row = _get_triu_sizes(row, col, offset)
    col_offset = max(0, offset)

    # 创建一个偏函数 arange_kw，用于生成 torch.arange 对象
    arange_kw = partial(
        torch.arange, layout=layout, device=device, pin_memory=pin_memory
    )

    # 生成顶部矩形区域的索引
    xs2 = arange_kw(0, rectangle_size, dtype=dtype)
    row_inds2 = xs2 // col
    col_inds2 = xs2 % col

    # 生成底部梯形区域的索引
    xs1 = arange_kw(0, trapezoid_size, dtype=torch.float64)
    b = -0.5 - m_first_row
    row_inds1 = torch.floor(-b - torch.sqrt(b * b - 2 * xs1))
    col_inds1 = torch.floor(xs1 - ((2 * m_first_row - 1 - row_inds1) * row_inds1) * 0.5)
    row_inds1 = _maybe_convert_to_dtype(row_inds1, dtype)
    col_inds1 = _maybe_convert_to_dtype(col_inds1, dtype)

    # 如果列数不为零，调整底部梯形区域的行索引
    if col:
        row_inds1 = row_inds1 + (rectangle_size // col)
    col_inds1 = col_inds1 + col_offset

    # 返回堆叠后的结果，包括顶部和底部区域的行列索引
    return torch.stack(
        (torch.cat((row_inds2, row_inds1)), torch.cat((col_inds2, col_inds1)))
    )


# 注册 bucketize 函数的分解方法
# 使用 out_wrapper 装饰器对函数进行装饰，确保输出类型正确
def bucketize(
    a: TensorLikeType,
    boundaries: TensorLikeType,
    *,
    out_int32: bool = False,
    right: bool = False,
):
    # 检查边界张量维度必须为1维
    torch._check(
        boundaries.dim() == 1,
        lambda: f"boundaries tensor must be 1 dimension but got dim({boundaries.dim()})",
    )

    # 根据需要确定输出的数据类型是 int32 还是 int64
    out_dtype = torch.int32 if out_int32 else torch.int64
    n_boundaries = boundaries.shape[-1]

    # 如果边界张量为空，返回与输入张量 a 相同形状的零张量
    if n_boundaries == 0:
        return torch.zeros_like(a)

    # 使用二分搜索确定每个元素属于的桶的索引
    # 在并行处理所有元素的情况下，保证算法的对数复杂度
    start = torch.zeros(a.shape, device=a.device, dtype=torch.int64)
    end = start + n_boundaries

    # 二分搜索的最大深度
    # 由于不能在不同元素的不同点退出循环，所以采用最大迭代次数来保证搜索终止
    mid = start + (end - start) // 2
    mid_val = boundaries[mid]
    if right:
        cond_mid = mid_val > a
    else:
        cond_mid = mid_val >= a
    start = torch.where(cond_mid, start, mid + 1)
    # 如果有多个边界点
    if n_boundaries > 1:
        # 创建一个与 a 形状相同的全为 True 的布尔张量
        cond_update = torch.ones_like(a, dtype=torch.bool)
        # 计算迭代次数，以 2 为底 n_boundaries 的对数
        niters = int(math.log2(n_boundaries))
        # 循环 niters 次数
        for _ in range(niters):
            # 根据条件更新 mid 的值
            end = torch.where(cond_mid & cond_update, mid, end)
            # 更新条件：start < end
            cond_update = start < end
            # 如果 start 最终指向结束之后的位置，我们需要防范这种情况
            mid = torch.where(cond_update, start + (end - start) // 2, 0)
            # 获取 mid 所指的边界值
            mid_val = boundaries[mid]
            # 如果 right 为 True，表示区间左闭右开，类似于 C++ 中的 std::upper_bound
            # 否则为左闭右闭，类似于 C++ 中的 std::lower_bound
            if right:
                cond_mid = mid_val > a
            else:
                cond_mid = mid_val >= a
            # 更新 start 的值
            start = torch.where((~cond_mid) & cond_update, mid + 1, start)
    
    # 将 start 转换为指定的输出数据类型并返回
    return start.to(dtype=out_dtype)
# 注册 Cauchy 分布的分解函数，添加装饰器
# 包装输出的函数
# 对元素类型进行提升的包装器，指定类型提升的参数为 ("self",)
# 使用默认的元素类型提升方式
@register_decomposition(aten.cauchy)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def cauchy(self, median=0, sigma=1, generator=None):
    # 断言 generator 为 None
    assert generator is None
    # 检查数据类型不是复数、整数或布尔类型
    torch._check(
        not utils.is_complex_dtype(self.dtype)
        and not utils.is_integer_dtype(self.dtype)
        and not utils.is_boolean_dtype(self.dtype),
        lambda: f"Cauchy distribution is a continuous probability distribution. \
        dtype must be a floating point but you specified {self.dtype}",
    )
    # 检查 sigma 必须大于 0
    torch._check(
        sigma > 0.0,
        lambda: f"cauchy_ expects sigma > 0.0, but found sigma={sigma}",
    )
    # 返回 Cauchy 分布的随机数
    return median + sigma * torch.tan(math.pi * (torch.rand_like(self) - 0.5))


# 注册指数分布的分解函数，添加装饰器
# 包装输出的函数
# 对元素类型进行提升的包装器，指定类型提升的参数为 ("self",)
# 使用默认的元素类型提升方式
@register_decomposition(aten.exponential)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def exponential(self, rate=1, generator=None):
    # 断言 generator 为 None
    assert generator is None
    # 检查数据类型不是复数、整数或布尔类型
    torch._check(
        not utils.is_complex_dtype(self.dtype)
        and not utils.is_integer_dtype(self.dtype)
        and not utils.is_boolean_dtype(self.dtype),
        lambda: f"Exponential distribution is a continuous probability distribution. \
        dtype must be a floating point but you specified {self.dtype}",
    )
    # 检查 rate 必须大于 0
    torch._check(
        rate > 0.0,
        lambda: f"exponential_ expects lambda > 0.0, but found lambda={rate}",
    )

    # 生成与 self 相同形状的均匀分布随机数
    uniform_val = torch.rand_like(self)

    # 复制转换的数字：指数函数的注释见下：
    # curand_uniform 的范围是 (0,1]。log(1) 是 0，指数函数排除 0。
    # 我们需要确保 log 不为 0，并且在转换为半精度时不会下溢。
    # 快速的 __logf 近似可能会下溢，所以将 log 设置为 -epsilon/2，适用于接近 1 的参数
    epsilon = torch.finfo(uniform_val.dtype).eps / 2
    condition = uniform_val >= 1.0 - epsilon
    log_uniform = torch.where(condition, -epsilon, torch.log(uniform_val))

    # 返回指数分布的随机数
    return -1 / rate * log_uniform


# 注册几何分布的分解函数，添加装饰器
# 包装输出的函数
# 对元素类型进行提升的包装器，指定类型提升的参数为 ("self",)
# 使用默认的元素类型提升方式
@register_decomposition(aten.geometric)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def geometric(self, p, generator=None):
    # 断言 generator 为 None
    assert generator is None
    # 检查数据类型不是复数或布尔类型
    torch._check(
        not utils.is_complex_dtype(self.dtype)
        and not utils.is_boolean_dtype(self.dtype),
        lambda: f"geometric not implemented for {self.dtype}",
    )
    # 检查 p 必须在 (0, 1) 范围内
    torch._check(
        0 < p and p < 1,
        lambda: f"geometric_ expects p to be in (0, 1), but got p={p}",
    )
    # 返回几何分布的随机数
    return torch.floor(torch.log1p(-torch.rand_like(self)) / math.log1p(-p)) + 1


# 注册对数正态分布的分解函数，添加装饰器
# 包装输出的函数
# 对元素类型进行提升的包装器，指定类型提升的参数为 ("self",)
# 使用默认的元素类型提升方式
@register_decomposition(aten.log_normal)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,


注释：

# 设置变量 type_promotion_kind 为 ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
# 定义 log_normal 方法，计算正态分布的对数概率密度函数
def log_normal(self, mean=1, std=2, generator=None):
    # generator 参数必须为 None
    assert generator is None
    # 检查张量的数据类型不能是复数、整数或布尔类型
    torch._check(
        not utils.is_complex_dtype(self.dtype)
        and not utils.is_integer_dtype(self.dtype)
        and not utils.is_boolean_dtype(self.dtype),
        lambda: f"log_normal not implemented for {self.dtype}",
    )
    # 检查 std 必须大于 0
    torch._check(
        0 < std,
        lambda: f"log_normal_ expects std > 0.0, but found std={std}",
    )
    # 返回计算结果，对输入张量按照正态分布生成随机数，然后求指数
    return torch.exp(std * torch.randn_like(self) + mean)


# TODO: add support for functionalization aten.normal_functional
# NOTE: the device and dtype will be ignored when shape is None
# 注册 aten.normal 的分解函数，并且在 shape 为 None 时会忽略 device 和 dtype
@register_decomposition(aten.normal)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=(
        "mean",
        "std",
    ),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义 normal 方法，生成指定形状的正态分布随机数张量
def normal(
    mean=0,
    std=1,
    size=None,
    *,
    generator=None,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    # layout 必须为 None 或者 torch.strided
    assert layout is None or layout == torch.strided

    # 如果 std 不是 TensorLike 类型，检查 std 必须大于等于 0
    if not isinstance(std, TensorLike):
        torch._check(
            std >= 0, lambda: f"normal expects std >= 0.0, but found std {std}"
        )

    # 如果 size 为 None
    if size is None:
        # 将 mean 和 std 中的张量找出来，计算它们的广播形状
        tensors = tuple(t for t in (mean, std) if isinstance(t, TensorLike))
        # 检查至少有一个 tensor，或者需要定义 size
        torch._check(
            len(tensors) > 0,
            lambda: "normal expects that either mean or std is a tensor, or size is defined",
        )
        # 不应该传递 layout 或 pin_memory 当 size 为 None 时
        torch._check(
            layout is None and pin_memory is None,
            lambda: "Cannot pass layout, or pin_memory without size",
        )

        # 计算广播后的形状，设置 dtype 和 device
        size = _broadcast_shapes(*(t.shape for t in tensors))
        dtype = tensors[0].dtype
        device = tensors[0].device
    else:
        # 当 size 不为 None 时，mean 和 std 必须为标量
        torch._check(
            not isinstance(mean, TensorLike) and not isinstance(std, TensorLike),
            lambda: "normal expects mean and std to be scalars when size is defined",
        )
        # 如果未指定 dtype，默认使用 torch.get_default_dtype()
        dtype = torch.get_default_dtype() if dtype is None else dtype
        # 如果未指定 device，默认使用 CPU
        device = torch.device("cpu") if device is None else device

    # 调用 prims.normal 生成指定形状的正态分布随机数张量
    normal_samples = prims.normal(
        size,
        mean=0.0,
        std=1.0,
        dtype=dtype,
        device=device,
        requires_grad=False,
        generator=generator,
    )
    # 返回经过修正的正态分布随机数张量
    return std * normal_samples + mean


# 注册 aten.normal_ 的分解函数，直接修改张量为正态分布随机数
@register_decomposition(aten.normal_)
def normal_(self, mean=0, std=1, *, generator=None):
    return normal(mean, std, self.shape, out=self, generator=generator)


# 定义 rad2deg 方法，将弧度转换为角度
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def rad2deg(self: TensorLikeType):
    # 检查张量的数据类型不能是复数
    torch._check(
        not utils.is_complex_dtype(self.dtype),
        lambda: "rad2deg is not supported for complex tensors.",
    )
    # 计算弧度转换为角度的常数值
    M_180_PI = 57.295779513082320876798154814105170332405472466564
    # 返回角度值
    return self * M_180_PI


# 定义 deg2rad 方法，将角度转换为弧度
@_make_elementwise_unary_reference(ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)
def deg2rad(self: TensorLikeType):
    # 检查张量的数据类型是否为复数，如果是，则抛出异常，因为 deg2rad 不支持复数张量。
    torch._check(
        not utils.is_complex_dtype(self.dtype),
        lambda: "deg2rad is not supported for complex tensors.",
    )
    # 将弧度转换系数 M_PI_180 设置为常数值
    M_PI_180 = 0.017453292519943295769236907684886127134428718885417
    # 返回当前张量乘以弧度转换系数 M_PI_180 后的结果
    return self * M_PI_180
# 注册 `aten.count_nonzero` 函数的分解装饰器，使其能够支持输出包装器
@register_decomposition(aten.count_nonzero)
@out_wrapper()
def count_nonzero(self, dim: Optional[DimsType] = None):
    # 返回非零元素的数量，可选地沿指定维度求和
    return (self != 0).sum(dim)


def _dot_check(self, other):
    # 检查张量是否为一维，并在不符合预期时引发异常
    torch._check(
        self.dim() == 1 and other.dim() == 1,
        lambda: f"1D tensors expected, but got {self.dim()}D and {other.dim()}D tensors",
    )

    def numel_error():
        # 返回关于张量大小不一致的错误消息
        return (
            f"inconsistent tensor size, expected tensor [{self.numel()}] and src [{other.numel()}] to have the"
            f"same number of elements, but got {self.numel()} and {other.numel()} elements respectively"
        )

    # 检查张量的元素数量是否相同，否则引发异常
    torch._check(self.numel() == other.numel(), numel_error)


# 注册 `aten.dot` 函数的分解装饰器，支持输出包装器和逐元素类型提升
@register_decomposition(aten.dot)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "other"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def dot(self, other):
    if self.is_complex():
        if self.is_conj():
            if other.is_conj():
                # 返回共轭复数的点积的共轭
                return torch.dot(self.conj(), other.conj()).conj()
            else:
                # 返回共轭 self 和 other 的点积
                return torch.vdot(self.conj(), other)
        elif other.is_conj():
            # 返回共轭 other 和 self 的点积
            return torch.vdot(other.conj(), self)

    # 检查张量维度，并引发异常，如果维度不符合预期
    _dot_check(self, other)
    # 返回两个张量的点积
    return (self * other).sum()


# 注册 `aten.vdot` 函数的分解装饰器，支持输出包装器和逐元素类型提升
@register_decomposition(aten.vdot)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self", "other"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def vdot(self, other):
    if not self.is_complex():
        # 返回张量的点积
        return torch.dot(self, other)

    if self.is_conj():
        if other.is_conj():
            # 返回共轭复数的点积的共轭
            return torch.vdot(other.conj(), self.conj())
        else:
            # 返回共轭 self 和 other 的点积
            return torch.dot(self.conj(), other)
    elif other.is_conj():
        # 返回共轭 other 和 self 的点积的共轭
        return torch.dot(self, other.conj()).conj()

    # 检查张量维度，并引发异常，如果维度不符合预期
    _dot_check(self, other)
    # 如果进行 self.conj() ... 操作，分解失败，原因不明
    return (self.conj_physical() * other).sum()


# 注册 `aten.select_scatter` 函数的分解装饰器，支持输出包装器
@register_decomposition(aten.select_scatter)
@out_wrapper()
def select_scatter(x: TensorLikeType, src: TensorLikeType, dim: int, index: int):
    # 规范化维度参数，确保在张量维度范围内
    dim = utils.canonicalize_dim(x.ndim, dim)
    # 创建用于生成掩码的形状，维度上只有一个非1的值
    mask_shape = [1] * x.ndim
    mask_shape[dim] = -1
    if index < 0:
        # 将负的索引转换为正的索引
        index = index + x.shape[dim]
    # 创建掩码，选择性地设置为 True 的位置
    mask = torch.arange(x.shape[dim], device=x.device).view(mask_shape) == index
    # 将源张量按指定维度进行扩展，以匹配目标张量的形状
    src = torch.unsqueeze(src, dim).expand(x.shape)
    # 根据掩码选择性地替换目标张量的值
    return torch.where(mask, src, x)


# 使用 `_make_inplace` 函数创建原地操作的装饰器，以下为一系列原地操作函数

# inplace
abs_ = _make_inplace(abs)
acos_ = _make_inplace(acos)
acosh_ = _make_inplace(acosh)
add_ = _make_inplace(add)
addcmul_ = _make_inplace(addcmul)
addcdiv_ = _make_inplace(addcdiv)
asin_ = _make_inplace(asin)
asinh_ = _make_inplace(asinh)
atan_ = _make_inplace(atan)
atanh_ = _make_inplace(atanh)
atan2_ = _make_inplace(atan2)
bitwise_and_ = _make_inplace(bitwise_and)
bitwise_left_shift_ = _make_inplace(bitwise_left_shift)
bitwise_not_ = _make_inplace(bitwise_not)
bitwise_or_ = _make_inplace(bitwise_or)
bitwise_right_shift_ = _make_inplace(bitwise_right_shift)
# 将函数_make_inplace应用于bitwise_xor_，使其成为原位操作的版本
bitwise_xor_ = _make_inplace(bitwise_xor)
# 将函数_make_inplace应用于ceil_，使其成为原位操作的版本
ceil_ = _make_inplace(ceil)
# 将函数_make_inplace应用于clamp_，使其成为原位操作的版本
clamp_ = _make_inplace(clamp)
# 将函数_make_inplace应用于clamp_min_，使其成为原位操作的版本
clamp_min_ = _make_inplace(clamp_min)
# 将函数_make_inplace应用于clamp_max_，使其成为原位操作的版本
clamp_max_ = _make_inplace(clamp_max)
# 将函数_make_inplace应用于conj_physical_，使其成为原位操作的版本
conj_physical_ = _make_inplace(conj_physical)
# 将函数_make_inplace应用于copysign_，使其成为原位操作的版本
copysign_ = _make_inplace(copysign)
# 将函数_make_inplace应用于cos_，使其成为原位操作的版本
cos_ = _make_inplace(cos)
# 将函数_make_inplace应用于cosh_，使其成为原位操作的版本
cosh_ = _make_inplace(cosh)
# 将函数_make_inplace应用于cumsum_，使其成为原位操作的版本
cumsum_ = _make_inplace(cumsum)
# 将函数_make_inplace应用于cumprod_，使其成为原位操作的版本
cumprod_ = _make_inplace(cumprod)
# 将函数_make_inplace应用于deg2rad_，使其成为原位操作的版本
deg2rad_ = _make_inplace(deg2rad)
# 将函数_make_inplace应用于digamma_，使其成为原位操作的版本
digamma_ = _make_inplace(digamma)
# 将函数_make_inplace应用于div_，使其成为原位操作的版本
div_ = _make_inplace(div)
# 将函数_make_inplace应用于eq_，使其成为原位操作的版本
eq_ = _make_inplace(eq)
# 将函数_make_inplace应用于erf_，使其成为原位操作的版本
erf_ = _make_inplace(erf)
# 将函数_make_inplace应用于erfc_，使其成为原位操作的版本
erfc_ = _make_inplace(erfc)
# 将函数_make_inplace应用于erfinv_，使其成为原位操作的版本
erfinv_ = _make_inplace(erfinv)
# 将函数_make_inplace应用于exp_，使其成为原位操作的版本
exp_ = _make_inplace(exp)
# 将函数_make_inplace应用于exp2_，使其成为原位操作的版本
exp2_ = _make_inplace(exp2)
# 将函数_make_inplace应用于expm1_，使其成为原位操作的版本
expm1_ = _make_inplace(expm1)
# 将函数_make_inplace应用于float_power_，使其成为原位操作的版本
float_power_ = _make_inplace(float_power)
# 将函数_make_inplace应用于floor_，使其成为原位操作的版本
floor_ = _make_inplace(floor)
# 将函数_make_inplace应用于floor_divide_，使其成为原位操作的版本
floor_divide_ = _make_inplace(floor_divide)
# 将函数_make_inplace应用于fmod_，使其成为原位操作的版本
fmod_ = _make_inplace(fmod)
# 将函数_make_inplace应用于frac_，使其成为原位操作的版本
frac_ = _make_inplace(frac)
# 将函数_make_inplace应用于gcd_，使其成为原位操作的版本
gcd_ = _make_inplace(gcd)
# 将函数_make_inplace应用于ge_，使其成为原位操作的版本
ge_ = _make_inplace(ge)
# 将函数_make_inplace应用于gt_，使其成为原位操作的版本
gt_ = _make_inplace(gt)
# 将函数_make_inplace应用于heaviside_，使其成为原位操作的版本
heaviside_ = _make_inplace(heaviside)
# 将函数_make_inplace应用于hypot_，使其成为原位操作的版本
hypot_ = _make_inplace(hypot)
# 将函数_make_inplace应用于igamma_，使其成为原位操作的版本
igamma_ = _make_inplace(igamma)
# 将函数_make_inplace应用于igammac_，使其成为原位操作的版本
igammac_ = _make_inplace(igammac)
# 将函数_make_inplace应用于i0_，使其成为原位操作的版本
i0_ = _make_inplace(i0)
# 将函数_make_inplace应用于lcm_，使其成为原位操作的版本
lcm_ = _make_inplace(lcm)
# 将函数_make_inplace应用于le_，使其成为原位操作的版本
le_ = _make_inplace(le)
# 将函数_make_inplace应用于lerp_，使其成为原位操作的版本
lerp_ = _make_inplace(lerp)
# 将函数_make_inplace应用于lgamma_，使其成为原位操作的版本
lgamma_ = _make_inplace(lgamma)
# 将函数_make_inplace应用于log10_，使其成为原位操作的版本
log10_ = _make_inplace(log10)
# 将函数_make_inplace应用于log1p_，使其成为原位操作的版本
log1p_ = _make_inplace(log1p)
# 将函数_make_inplace应用于log2_，使其成为原位操作的版本
log2_ = _make_inplace(log2)
# 将函数_make_inplace应用于log_，使其成为原位操作的版本
log_ = _make_inplace(log)
# 将函数_make_inplace应用于logical_and_，使其成为原位操作的版本
logical_and_ = _make_inplace(logical_and)
# 将函数_make_inplace应用于logical_not_，使其成为原位操作的版本
logical_not_ = _make_inplace(logical_not)
# 将函数_make_inplace应用于logical_or_，使其成为原位操作的版本
logical_or_ = _make_inplace(logical_or)
# 将函数_make_inplace应用于logical_xor_，使其成为原位操作的版本
logical_xor_ = _make_inplace(logical_xor)
# 将函数_make_inplace应用于lt_，使其成为原位操作的版本
lt_ = _make_inplace(lt)
# 将函数_make_inplace应用于mul_，使其成为原位操作的版本
mul_ = _make_inplace(mul)
# 将函数_make_inplace应用于mvlgamma_，使其成为原位操作的版本
mvlgamma_ = _make_inplace(mvlgamma)
# 将函数_make_inplace应用于nan_to_num_，使其成为原位操作的版本
nan_to_num_ = _make_inplace(nan_to_num)
# 将函数_make_inplace应用于ne_，使其成为原位操作的版本
ne_ = _make_inplace(ne)
# 将函数_make_inplace应用于neg_，使其成为原位操作的版本
neg_ = _make_inplace(neg)
# 将函数_make_inplace应用于nextafter_，使其成为原位操作的版本
nextafter_ = _make_inplace(nextafter)
# 将函数_make_inplace应用于pow_，使其成为原位操作的版本
pow_ = _make_inplace(pow)
# 将函数_make_inplace应用于rad2deg_，使其成为原位操作的版本
rad2deg_ = _make_inplace(rad2deg)
# 将函数_make_inplace应用于reciprocal_，使其成为原位操作的版本
reciprocal_ = _make_inplace(reciprocal)
# 将函数_make_inplace应用于remainder_，使其成为原位操作的版本
remainder_ = _make_inplace(remainder)
# 将函数_make_inplace应用于rsqrt_，使其成为原位操作的版本
rsqrt_ = _make_inplace(rsqrt)
# 将函数_make
    # 当 seq 是 list 或 tuple 类型时执行循环，直到 seq 不再是列表或元组为止
    while isinstance(seq, (list, tuple)):
        # 获取 seq 的长度
        length = len(seq)
        # 如果 is_storage 为真，将长度调整为标量类型的元素大小
        if is_storage:
            length //= scalar_type.itemsize
        # 将长度添加到 sizes 列表中
        sizes.append(length)
        # 如果 sizes 列表超过了最大允许的维度数 MAX_DIMS，则抛出 ValueError 异常
        if len(sizes) > MAX_DIMS:
            raise ValueError(f"too many dimensions '{type(seq).__name__}'")
        # 如果 length 为 0，则跳出循环
        if length == 0:
            break
        try:
            # 尝试获取 seq 的第一个元素作为 handle
            handle = seq[0]
        except Exception:
            # 如果无法确定对象类型的形状，则抛出 ValueError 异常
            raise ValueError(
                f"could not determine the shape of object type '{type(seq).__name__}'"
            )
        # 将 seq 设置为 handle，继续下一轮循环
        seq = handle
    
    # 返回 sizes 列表，其中包含了 seq 及其嵌套结构的长度信息
    return sizes
# 推断对象的标量类型，类似于 torch/csrc/utils/tensor_new.cpp 中的 infer_scalar_type 函数
def _infer_scalar_type(obj):
    if isinstance(obj, FloatLike):
        # 如果对象是 FloatLike 类型，则返回当前默认的 torch 数据类型
        return torch.get_default_dtype()
    if isinstance(obj, IntLike) and not isinstance(obj, bool):  # 注意！
        # 如果对象是 IntLike 类型但不是布尔型，则返回 torch.int64 类型
        return torch.int64
    if isinstance(obj, BoolLike):
        # 如果对象是 BoolLike 类型，则返回 torch.bool 类型
        return torch.bool
    if isinstance(obj, complex):
        # 如果对象是复数类型
        default_dtype = torch.get_default_dtype()
        if default_dtype is torch.float:
            return torch.cfloat
        elif default_dtype is torch.double:
            return torch.cdouble
        elif default_dtype is torch.half:
            return torch.chalf
        else:
            raise RuntimeError("invalid default scalar type for complex")
    if isinstance(obj, torch.Tensor):
        # 如果对象是 torch.Tensor 类型，则返回其数据类型
        return obj.dtype
    if isinstance(obj, str):
        # 如果对象是字符串类型，则抛出类型错误异常
        raise TypeError(f"new(): invalid data type '{type(obj).__name__}'")
    # TODO: this is inaccurate, we actually test PySequence_Check
    if isinstance(obj, (list, tuple)):
        scalarType = None
        length = len(obj)
        # 符合 NumPy 语义，但使用默认的张量类型而不是 double 类型。
        if length == 0:
            return torch.get_default_dtype()
        for i in range(length):
            cur_item = obj[i]
            # TODO: test this
            """
            if cur_item is obj:
                raise TypeError("new(): self-referential lists are incompatible")
            """
            # 递归调用 _infer_scalar_type 函数
            item_scalarType = _infer_scalar_type(cur_item)  # 递归！
            if scalarType is not None:
                # 通过 torch.promote_types 提升数据类型
                scalarType = torch.promote_types(scalarType, item_scalarType)
            else:
                scalarType = item_scalarType
            if scalarType is torch.cdouble:
                # 这不会改变（除非我们遇到未定义，但那将在后面失败）
                return scalarType
        return scalarType
    # 如果无法推断出对象的数据类型，则抛出运行时错误异常
    raise RuntimeError(f"Could not infer dtype of {type(obj).__name__}")


# 递归构建函数，类似于 torch/csrc/utils/tensor_new.cpp 中的 recursive_store 函数
def _recursive_build(
    scalarType: torch.dtype, obj: Union[TensorOrNumberLikeType, TensorSequenceType]
):
    if isinstance(obj, Tensor) and obj.numel() == 1:
        # 如果对象是 Tensor 且只有一个元素，则返回该元素的标量张量
        return obj.detach().to(dtype=scalarType, device="cpu", copy=True).view(())
    elif isinstance(obj, Tensor):
        # 如果对象是 Tensor，则返回它的标量张量
        return obj.detach().to(dtype=scalarType, device="cpu", copy=True)
    elif isinstance(obj, Number):
        # 如果对象是数字类型，则返回其标量张量
        return torch.scalar_tensor(obj, dtype=scalarType)

    # seq 可以是张量列表
    seq = obj
    # 返回一个张量，其中每个元素是使用 _recursive_build 函数处理序列中每个元素得到的结果
    return torch.stack([_recursive_build(scalarType, item) for item in seq])
# xref: internal_new_from_data in torch/csrc/utils/tensor_new.cpp
def _internal_new_from_data(
    options,
    scalar_type,
    device_opt,
    data,
    copy_variables,
    copy_numpy,
    type_inference,
    pin_memory=False,
):
    if isinstance(data, torch.Tensor):
        # 检查是否允许内存固定，如果允许则抛出异常
        torch._check(
            not pin_memory, lambda: "Can't pin tensor constructed from a variable"
        )
        var = data
        if copy_variables:
            # 如果需要复制变量，执行变量的分离操作
            var = var.detach()
        # 推断标量类型（如果需要），或使用预定义的标量类型
        inferred_scalar_type = var.dtype if type_inference else scalar_type
        # 确定设备是显式提供的还是从变量中获取的
        device = device_opt if device_opt is not None else var.device
        # 将变量转换为指定的设备和数据类型
        return var.to(
            device=device,
            dtype=inferred_scalar_type,
            non_blocking=False,
            copy=copy_variables,
        )

    # 如果数据具有 __cuda_array_interface__ 属性，则返回 NotImplemented
    if hasattr(data, "__cuda_array_interface__"):
        return NotImplemented

    # TODO: 使用 PyArray_Check 测试是否为 NumPy 输入

    # 如果未提供显式设备，则使用选项中的默认设备
    device = device_opt if device_opt is not None else options["device"]
    # 推断标量类型（如果需要），或使用预定义的标量类型
    inferred_scalar_type = _infer_scalar_type(data) if type_inference else scalar_type

    # 注意：这里不需要避免追踪，因为我们不会执行任何手动的指针填充技巧
    if _isStorage(data):
        return NotImplemented
    else:
        # 如果设备类型是 "meta"，则返回 NotImplemented
        if torch.device(device).type == "meta":
            return NotImplemented

        # 在 C 实现中，我们会直接开始操作刚刚分配的 CPU 张量的内存。
        # 在这里，我们将使用一种非常慢的实现：将每个标量转换为张量，然后重复连接它们
        tensor = _recursive_build(inferred_scalar_type, data)

        # 将张量转移到指定设备上，并使用推断的标量类型，非阻塞复制
        tensor = tensor.to(device, inferred_scalar_type, non_blocking=False, copy=False)

    # 注意：在这种情况下，不需要 lift_fresh，因为我们从标量构建了张量，保证了一个新的张量
    return tensor


# xref: tensor_ctor in torch/csrc/utils/tensor_new.cpp
def tensor(data, *, dtype=None, device=None, pin_memory=False, requires_grad=False):
    # TODO (or not): 支持 names kwarg

    if isinstance(data, torch.Tensor):
        # 如果数据是张量，发出警告建议使用 clone().detach() 或 clone().detach().requires_grad_(True) 来复制构造张量
        warnings.warn(
            "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() "
            "or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)"
        )
    
    # 推断类型是否为 None
    type_inference = dtype is None
    # 使用 _internal_new_from_data 函数创建新张量
    new_tensor = _internal_new_from_data(
        # 因为使用 torch.tensor(2) 默认情况下是 "cpu" 设备，所以这里设置为 "cpu"
        {"device": "cpu"},  # TODO: 使用 torch.get_default_tensor_type
        dtype if dtype is not None else torch.get_default_dtype(),
        device,
        data,
        copy_variables=True,
        copy_numpy=True,
        type_inference=type_inference,
        pin_memory=pin_memory,
    )
    # 分离新张量，确保不会跟踪其计算历史
    new_tensor.detach_()
    # 如果需要计算梯度，则标记新张量需要计算梯度
    if requires_grad:
        new_tensor.requires_grad_(requires_grad)
    return new_tensor


# Views
# 以下代码块导入了一些名为 torch._refs 的模块下的子模块或成员，这些成员包含在不同的功能领域中：
# - torch._refs._conversions
# - torch._refs.fft
# - torch._refs.linalg
# - torch._refs.nn.functional
# - torch._refs.special
# 这些子模块可能提供了与类型转换、傅里叶变换、线性代数、神经网络功能和特殊数学函数相关的实现或引用。
# 每个模块可能包含了与其名字相对应的功能或实现，供在代码的后续部分使用。
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
```