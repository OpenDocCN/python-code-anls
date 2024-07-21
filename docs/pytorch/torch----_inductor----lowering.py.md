# `.\pytorch\torch\_inductor\lowering.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库
import functools  # 提供了一些有用的功能，如 functools.partial
import itertools  # 提供了用于操作迭代器的函数
import logging  # 日志记录工具
import math  # 数学函数库
import operator  # 提供了标准运算符的函数形式
import os  # 提供了与操作系统进行交互的功能
import warnings  # 警告控制功能

from collections import defaultdict  # 默认字典，提供了默认值的字典
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union  # 类型提示相关的功能
from unittest.mock import patch  # 用于在单元测试中模拟对象行为

import sympy  # 用于符号计算的库

import torch  # PyTorch 深度学习库
import torch.ao.quantization.fx._decomposed  # PyTorch AO量化功能的一部分
import torch.fx  # PyTorch FX框架
import torch.utils._pytree as pytree  # PyTorch 的一种数据结构

from torch._higher_order_ops.associative_scan import associative_scan_op  # 高阶操作相关功能
from torch._higher_order_ops.triton_kernel_wrap import (  # Triton内核包装相关函数
    triton_kernel_wrapper_functional,
    triton_kernel_wrapper_mutation,
)
from torch._prims_common import (  # 基本的操作相关函数
    canonicalize_dim,
    canonicalize_dims,
    check,
    dtype_to_type,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    get_computation_dtype,
    is_boolean_dtype,
    is_float_dtype,
    is_integer_dtype,
    Number,
)
from torch.fx.experimental.sym_node import magic_methods, method_to_operator  # 符号节点相关实验性功能
from torch.utils._sympy.functions import (  # Sympy函数相关功能
    CeilDiv,
    FloorDiv,
    Identity,
    IntTrueDiv,
    ModularIndexing,
)
from .._dynamo.utils import import_submodule  # 导入子模块的工具函数

# 导入本地模块和文件
from . import config, inductor_prims, ir, test_operators  # 导入配置、感应器操作、IR和测试操作
from .decomposition import decompositions, get_decompositions  # 导入分解和获取分解
from .ir import (  # 导入IR相关类和函数
    ExpandView,
    IndexingConstant,
    is_triton,
    ops_wrapper,
    PermuteView,
    Pointwise,
    Reduction,
    SqueezeView,
    TensorBox,
    validate_ir,
    View,
)
from .utils import (  # 导入各种工具函数
    ceildiv,
    decode_device,
    is_dynamic,
    is_gpu,
    is_pointwise_use,
    needs_fallback_due_to_atomic_add_limitations,
    pad_listlike,
    sympy_product,
    use_scatter_fallback,
)
from .virtualized import ops, V  # 导入虚拟化操作相关

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

lowerings: Dict[torch._ops.OpOverload, Callable[..., Any]] = {}  # OpOverload到函数的字典
layout_constraints: Dict[torch._ops.OpOverload, Callable[..., Any]] = {}  # OpOverload到布局约束函数的字典
fallbacks: Set[torch._ops.OpOverload] = set()  # OpOverload的集合，表示需要回退的操作
aten = torch.ops.aten  # 获取PyTorch aten操作的命名空间
tr_c10d = torch.ops.tr_c10d  # 获取PyTorch tr_c10d操作的命名空间
prims = torch.ops.prims  # 获取PyTorch prims操作的命名空间
needs_realized_inputs: Set[torch._ops.OpOverload] = set()  # OpOverload的集合，表示需要实现输入的操作
foreach_ops: Set[torch._ops.OpOverload] = set()  # OpOverload的集合，表示支持for each操作的操作
inplace_foreach_ops: Set[torch._ops.OpOverload] = set()  # OpOverload的集合，表示支持inplace for each操作的操作
inplaceable_foreach_ops: Dict[torch._ops.OpOverload, torch._ops.OpOverload] = dict()  # OpOverload到inplace OpOverload的映射
quantized_decomposed = torch.ops.quantized_decomposed  # 获取PyTorch quantized_decomposed操作的命名空间


def assert_nyi(cond, msg):
    # 如果条件不满足，则抛出NotImplementedError异常，指示感应器不支持某些操作
    if not cond:
        raise NotImplementedError(f"inductor does not support {msg}")


def add_needs_realized_inputs(fn):
    # 如果传入的fn是列表、元组或集合，则分别对其中的每个元素调用add_needs_realized_inputs
    if isinstance(fn, (list, tuple, set)):
        return [add_needs_realized_inputs(x) for x in fn]
    # 将fn添加到needs_realized_inputs集合中
    needs_realized_inputs.add(fn)
    # 如果fn是OpOverloadPacket类型，则遍历其所有overloads，并将其添加到needs_realized_inputs集合中
    if isinstance(fn, torch._ops.OpOverloadPacket):
        needs_realized_inputs.update(
            getattr(fn, overload) for overload in fn.overloads()
        )


def add_layout_constraint(fn, constraint):
    # 如果fn是OpOverloadPacket类型，则对其所有overloads添加布局约束
    if isinstance(fn, torch._ops.OpOverloadPacket):
        for overload in fn.overloads():
            layout_constraints[getattr(fn, overload)] = constraint
    else:
        # 如果条件不满足前面的任何情况，则执行这个分支
        # 将 layout_constraints 字典中键为 fn 的值设为 constraint
        layout_constraints[fn] = constraint
# 将一组函数添加到需要实现输入的集合中
add_needs_realized_inputs(
    [
        aten.as_strided,  # 添加函数 aten.as_strided
        aten.as_strided_copy,  # 添加函数 aten.as_strided_copy
        aten.avg_pool2d,  # 添加函数 aten.avg_pool2d
        aten.avg_pool2d_backward,  # 添加函数 aten.avg_pool2d_backward
        aten.bmm,  # 添加函数 aten.bmm
        aten.convolution,  # 添加函数 aten.convolution
        aten.convolution_backward,  # 添加函数 aten.convolution_backward
        aten.max_pool2d_with_indices,  # 添加函数 aten.max_pool2d_with_indices
        aten.max_pool2d_with_indices_backward,  # 添加函数 aten.max_pool2d_with_indices_backward
        aten.mm,  # 添加函数 aten.mm
        aten.upsample_nearest2d,  # 添加函数 aten.upsample_nearest2d
        aten._upsample_nearest_exact2d,  # 添加函数 aten._upsample_nearest_exact2d
        aten._int_mm,  # 添加函数 aten._int_mm
    ]
)

# TODO(jansel): ezyang says we won't need this in the future, try removing it
# 基于 https://github.com/pytorch/pytorch/blob/9e3eb329df8f701/c10/core/ScalarType.h#L28
# 在未来，根据 ezyang 的说法，可能不再需要这部分，可以尝试移除它
DTYPE_ID_LOOKUP = {
    0: torch.uint8,  # 表示类型 0 对应 torch.uint8
    1: torch.int8,  # 表示类型 1 对应 torch.int8
    2: torch.int16,  # 表示类型 2 对应 torch.int16
    3: torch.int32,  # 表示类型 3 对应 torch.int32
    4: torch.int64,  # 表示类型 4 对应 torch.int64
    5: torch.float16,  # 表示类型 5 对应 torch.float16
    6: torch.float32,  # 表示类型 6 对应 torch.float32
    7: torch.float64,  # 表示类型 7 对应 torch.float64
    8: torch.complex32,  # 表示类型 8 对应 torch.complex32
    9: torch.complex64,  # 表示类型 9 对应 torch.complex64
    10: torch.complex32,  # 表示类型 10 对应 torch.complex32
    11: torch.bool,  # 表示类型 11 对应 torch.bool
    15: torch.bfloat16,  # 表示类型 15 对应 torch.bfloat16
    # TODO(jansel): 添加量化类型？
    #  _(c10::qint8, QInt8) /* 12 */
    # _(c10::quint8, QUInt8) /* 13 */
    # _(c10::qint32, QInt32) /* 14 */
    # _(c10::quint4x2, QUInt4x2) /* 16 */
    # _(c10::quint2x4, QUInt2x4) /* 17 */
}


def decode_dtype(dtype: int):
    # 如果 dtype 不是整数，则直接返回 dtype
    if not isinstance(dtype, int):
        return dtype
    # 断言 dtype 在 DTYPE_ID_LOOKUP 中，否则报错
    assert dtype in DTYPE_ID_LOOKUP, f"id {dtype} missing from DTYPE_ID_LOOKUP"
    # 将 dtype 转换为对应的类型，并返回
    dtype = DTYPE_ID_LOOKUP[dtype]
    return dtype


def is_integer_type(x):
    # 如果 x 是 TensorBox 类型
    if isinstance(x, TensorBox):
        # 判断 x 的数据类型是否为整数或布尔类型
        return is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    # 如果 x 是 sympy.Expr 类型
    elif isinstance(x, sympy.Expr):
        # 返回 x 是否是整数类型的表达式
        return x.is_integer is True  # type: ignore[attr-defined]
    else:
        # 返回 x 是否是整数类型
        return isinstance(x, int)


def is_boolean_type(x):
    # 如果 x 是 TensorBox 类型
    if isinstance(x, TensorBox):
        # 返回 x 的数据类型是否为布尔类型
        return is_boolean_dtype(x.get_dtype())
    else:
        # 返回 x 是否是布尔类型
        return isinstance(x, bool)


def get_promoted_dtype(*args, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND):
    # 内部函数，构造输入参数 inp
    def construct_input(inp):
        # 如果 inp 是 Number 或 sympy.Basic 类型，则直接返回
        if isinstance(inp, (Number, sympy.Basic)):
            return inp
        else:
            # 否则，假定 inp 具有 get_dtype 属性，构造一个临时张量以便传给 torch.result_type
            assert hasattr(inp, "get_dtype")
            dim = len(inp.get_size())
            return torch.zeros([1] * dim, dtype=inp.get_dtype())

    # 构造所有输入的列表 inps
    inps = [construct_input(arg) for arg in args]
    # 调用 elementwise_dtypes 函数获取类型推广后的结果
    _, dtype = elementwise_dtypes(*inps, type_promotion_kind=type_promotion_kind)
    return dtype


def get_overloads(aten_fn):
    # 如果 aten_fn 不是列表或元组，将其转换为列表
    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)

    # 遍历 aten_fn 的副本
    for fn in list(aten_fn):
        # 如果 fn 是 torch._ops.OpOverloadPacket 类型
        if isinstance(fn, torch._ops.OpOverloadPacket):
            # 遍历其重载函数列表
            for overload in fn.overloads():
                # 获取重载函数 other_fn
                other_fn = getattr(fn, overload)
                # 如果 other_fn 不在 lowerings 中，则将其添加到 aten_fn 中
                if other_fn not in lowerings:
                    aten_fn.append(other_fn)

    # 返回处理后的 aten_fn 列表
    return aten_fn


def transform_args(args, broadcast, type_promotion_kind, convert_input_to_bool):
    # 获取所有 TensorBox 类型参数的索引
    indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    # 如果类型提升种类存在或者需要将输入转换为布尔类型，并且索引列表不为空时执行以下操作
    if (type_promotion_kind or convert_input_to_bool) and indices:
        # 如果需要将输入转换为布尔类型
        if convert_input_to_bool:
            # 设置数据类型为 torch 的布尔类型
            dtype = torch.bool
        else:
            # 否则，根据参数中的数值或 sympy.Basic 类型，或者其具有 dtype 属性的对象，生成用于提升类型的参数列表
            promoting_args = [
                a
                for a in args
                if isinstance(a, (Number, sympy.Basic))
                or getattr(a, "dtype", None) is not None
            ]
            # 获取通过类型提升规则确定的数据类型
            dtype = get_promoted_dtype(
                *promoting_args, type_promotion_kind=type_promotion_kind
            )

        # 有时参数是不可变列表，无法直接修改
        # 定义函数用于提升参数的数据类型
        def promote(arg):
            if isinstance(arg, TensorBox):
                return to_dtype(arg, dtype)
            elif isinstance(arg, ir.Constant):
                return ir.Constant(arg.value, dtype, args[indices[0]].get_device())
            else:
                return arg

        # 对参数列表中的每个参数执行数据类型提升
        args = [promote(a) for a in args]

    # 如果需要广播操作并且索引列表不为空时执行以下操作
    if broadcast and indices:
        # 使用广播函数对索引列表中的参数执行广播操作
        for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
            args[i] = x
        # 对参数列表中的每个参数执行扩展视图的创建，以便于后续处理
        for i in range(len(args)):
            if isinstance(args[i], ir.Constant):
                args[i] = ExpandView.create(args[i], list(args[indices[0]].get_size()))

    # 返回处理后的参数列表
    return args
# 将一个 foreach 降级操作注册到 lowerings 字典中
def _register_foreach_lowering(aten_fn, decomp_fn):
    """
    Add a foreach lowering to lowerings dict.

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
            用于降级操作的 torch.ops.aten.* 函数
        decomp_fn: alternate implementation on our IR
            我们内部 IR 的替代实现函数

    Returns:
        wrapped: 包装后的函数
            返回被包装的函数
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        assert len(args) <= 2
        # 调用传入的 decomp_fn 处理参数和关键字参数
        out = decomp_fn(*args, **kwargs)
        # 验证生成的内部 IR 是否有效
        validate_ir(out)
        return out

    # 获取 aten_fn 的重载函数列表
    aten_fns = get_overloads(aten_fn)
    # 将 aten_fn 的重载函数添加到 foreach_ops 字典中
    foreach_ops.update(aten_fns)
    # 将 aten_fn 的重载函数映射到 wrapped 函数并添加到 lowerings 字典中
    lowerings.update(dict.fromkeys(aten_fns, wrapped))
    return wrapped


# 将一个降级操作注册到 lowerings 字典中
def _register_lowering(
    aten_fn, decomp_fn, broadcast, type_promotion_kind, convert_input_to_bool
):
    """
    Add a lowering to lowerings dict

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
            用于降级操作的 torch.ops.aten.* 函数
        decomp_fn: alternate implementation on our IR
            我们内部 IR 的替代实现函数
        broadcast: True to apply broadcasting to tensor inputs
            是否对张量输入应用广播操作
        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
            应用于张量输入的类型提升方式，`None` 表示不进行类型提升
        convert_input_to_bool: some logical ops require inputs are converted to bool
            一些逻辑操作需要将输入转换为布尔类型

    Returns:
        wrapped: 包装后的函数
            返回被包装的函数
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        args: Union[List[Any], Tuple[Any, ...], Dict[Any, Any]] = list(args)
        unpacked = False
        # TODO maybe we need to use pytrees here
        # 如果参数长度为1且为列表或元组，则进行解包
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            unpacked = True
            args = args[0]

        # kwargs 中不支持张量，除非是后备操作
        if not all(fn in fallbacks for fn in aten_fn):
            assert not any(isinstance(x, TensorBox) for x in kwargs.values())
            # 对于 "out=" 操作，显式断言以获得更好的错误消息
            assert not any(
                x == "out" for x in kwargs.keys()
            ), "out= ops aren't yet supported"

        # 转换参数根据参数要求进行转换
        args = transform_args(
            args, broadcast, type_promotion_kind, convert_input_to_bool
        )

        if unpacked:
            args = [args]

        # 调用传入的 decomp_fn 处理转换后的参数和关键字参数
        out = decomp_fn(*args, **kwargs)
        # 验证生成的内部 IR 是否有效
        validate_ir(out)

        return out

    # 获取 aten_fn 的重载函数列表
    aten_fn = get_overloads(aten_fn)
    # 将 aten_fn 的重载函数映射到 wrapped 函数并添加到 lowerings 字典中
    lowerings.update(dict.fromkeys(aten_fn, wrapped))
    return wrapped


# 注册一个降级操作，支持装饰器语法
def register_lowering(
    aten_fn,
    broadcast=False,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
):
    """
    Shim to support decorator syntax.

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
            用于降级操作的 torch.ops.aten.* 函数
        broadcast: True to apply broadcasting to tensor inputs
            是否对张量输入应用广播操作，默认为 False
        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion
            应用于张量输入的类型提升方式，默认为 ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        convert_input_to_bool: some logical ops require inputs are converted to bool
            一些逻辑操作需要将输入转换为布尔类型，默认为 False

    Returns:
        functools.partial object:
            返回一个 functools.partial 对象，用于支持装饰器语法
    """
    return functools.partial(
        _register_lowering,
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
    )


# 基于符号形状进行广播逻辑
def broadcast_symbolic_shapes(a, b):
    """
    Broadcasting logic based on symbolic shapes.

    Arguments:
        a: shape of tensor a
            张量 a 的形状
        b: shape of tensor b
            张量 b 的形状
    """
    We give the shapes 0 and 1 concrete values, while all other shapes
    are symbolic sympy formulas.
    """
    # 初始化一个空列表，用于存放最终的输出结果
    output = []
    # 使用 itertools.zip_longest 函数同时遍历 a 和 b 的逆序序列
    # fillvalue=sympy.Integer(1) 表示在迭代中如果缺少值，默认用 sympy.Integer(1) 填充
    for x, y in itertools.zip_longest(
        reversed(a), reversed(b), fillvalue=sympy.Integer(1)
    ):
        # 如果 y 等于 1，则将 x 加入到 output 列表中
        if y == 1:
            output.append(x)
        # 如果 x 等于 1，则将 y 加入到 output 列表中
        elif x == 1:
            output.append(y)
        else:
            # 否则，调用 V.graph.sizevars.guard_equals(x, y) 函数
            V.graph.sizevars.guard_equals(x, y)
            # 如果 y 的自由符号数少于 x 的自由符号数
            if len(sympy.expand(y).free_symbols) < len(sympy.expand(x).free_symbols):
                # 优先选择较短的公式 y，将 y 加入到 output 列表中
                output.append(y)
            else:
                # 否则，将 x 加入到 output 列表中
                output.append(x)
    # 返回 output 列表的逆序作为最终结果的元组
    return tuple(reversed(output))
# 定义一个函数用于根据输入进行常量推广
def promote_constants(inputs, override_return_dtype=None, type_promotion_kind=None):
    # 断言：只能给定 override_return_dtype 或 type_promotion_kind 中的一个
    assert (
        override_return_dtype is None or type_promotion_kind is None
    ), "only one of override_return_dtype or type_promotion_kind may be given"

    # 如果 override_return_dtype 和 type_promotion_kind 都未指定，则使用默认的类型推广方式
    if override_return_dtype is None and type_promotion_kind is None:
        type_promotion_kind = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT

    # 如果所有输入都是 sympy.Basic、int 或 float 类型，则根据 override_return_dtype 或 type_promotion_kind 推导数据类型
    if all(isinstance(x, (int, float, sympy.Basic)) for x in inputs):
        # 获取推广后的数据类型
        dtype = override_return_dtype or get_promoted_dtype(
            *inputs, type_promotion_kind=type_promotion_kind
        )

        # 定义一个函数，根据输入类型创建常量对象
        def const_func(x):
            if isinstance(x, sympy.Basic):
                return ir.IndexingConstant(x, dtype, decode_device(None))
            else:
                return ir.Constant(x, dtype, decode_device(None))

        # 对输入列表中的每个元素应用 const_func 函数，返回常量对象列表
        return [const_func(x) for x in inputs]
    
    # 如果存在不是 sympy.Basic、int 或 float 的输入，则找到其中一个 TensorBox、ExpandView 或 ir.Constant 类型的对象
    ex = next(x for x in inputs if isinstance(x, (TensorBox, ExpandView, ir.Constant)))
    out = []
    # 遍历输入列表，根据元素类型创建 ExpandView 对象或保留原始元素
    for x in inputs:
        if isinstance(x, (int, float)):
            out.append(
                ExpandView.create(
                    ir.Constant(x, ex.get_dtype(), ex.get_device()), list(ex.get_size())
                )
            )
        elif isinstance(x, sympy.Basic):
            out.append(
                ExpandView.create(
                    IndexingConstant(x, ex.get_dtype(), ex.get_device()),
                    list(ex.get_size()),
                )
            )
        else:
            out.append(x)

    # 返回处理后的输出列表
    return out


# 定义一个函数用于创建逐点操作
def make_pointwise(
    fn,
    override_return_dtype=None,
    override_device=None,
    override_fn_when_input_bool=None,
    override_fn_when_cuda_float64=None,
    allow_alpha=False,
    triton_fallback=None,
):
    # 定义一个函数 inner，接受可变数量的 TensorBox 类型输入及一个 alpha 参数
    def inner(*inputs: List[TensorBox], alpha=None):
        # 如果存在 Triton 回退函数并且输入中有任何一个是 Triton 类型，则执行以下逻辑
        if triton_fallback is not None and any(map(is_triton, inputs)):
            # 断言不允许使用 alpha 参数（尚未实现）
            assert not allow_alpha  # not implemented
            # 返回 Triton 回退函数的结果
            return triton_fallback(*inputs)
    
        # 提升常量输入的优先级，使用指定的返回数据类型替代默认类型
        inputs = promote_constants(inputs, override_return_dtype)
        
        # 如果允许使用 alpha 参数
        if allow_alpha:
            # 如果 alpha 不为 None 且不等于 1，则对输入中的最后一个元素乘以 alpha
            if alpha is not None and alpha != 1:
                inputs = list(inputs)
                inputs[-1] = mul(inputs[-1], alpha)
        else:
            # 不允许使用 alpha 参数时，确保 alpha 为 None
            assert alpha is None
        
        # 创建输入中每个元素的加载器
        loaders = [x.make_loader() for x in inputs]
        # 获取第一个输入的尺寸范围
        ranges = inputs[0].get_size()
        # 获取覆盖的返回数据类型，否则使用第一个输入的数据类型
        dtype = override_return_dtype or inputs[0].get_dtype()
        # 判断第一个输入是否在 CUDA 设备上
        is_cuda = decode_device(inputs[0].get_device()).type == "cuda"
    
        # 遍历除第一个输入外的所有输入，确保它们是常量或者尺寸与第一个输入匹配
        for other in inputs[1:]:
            assert isinstance(other, ir.BaseConstant) or len(ranges) == len(
                other.get_size()
            ), f"ndim mismatch {fn} {ranges} {other.get_size()}"
    
        # 定义内部函数 inner_fn，接受一个索引作为参数
        def inner_fn(index):
            # 确保索引的维度与 ranges 相同
            assert len(index) == len(ranges), f"wrong ndim {index} {ranges}"
            # 根据数据类型和特定条件选择合适的函数进行计算并返回结果
            if dtype == torch.bool and override_fn_when_input_bool is not None:
                return override_fn_when_input_bool(*[load(index) for load in loaders])
            elif override_fn_when_cuda_float64 and is_cuda and dtype == torch.float64:
                return override_fn_when_cuda_float64(*[load(index) for load in loaders])
            else:
                return fn(*[load(index) for load in loaders])
    
        # 如果未覆盖设备，则根据输入选择 CUDA 设备或者使用第一个输入的设备
        if not override_device:
            device = None
            for i in inputs:
                if is_gpu(i.get_device().type):
                    device = i.get_device()
                    break
            if not device:
                device = inputs[0].get_device()
    
        # 使用覆盖的设备或者默认设备作为创建 Pointwise 对象时的设备
        device = override_device or device
    
        # 创建并返回一个 Pointwise 对象，设备、数据类型、内部函数和尺寸范围作为参数
        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
        )
def make_foreach_pointwise(pw_fn, allow_alpha=False):
    def inner(*inputs: List[List[TensorBox]], alpha=1):
        # 定义内部函数 inner，接受多个输入列表，每个列表元素是 TensorBox 对象，还可以接受一个 alpha 参数，默认为 1
        
        # 按设备分组，检查输入是否动态，以及它们的类型是否匹配（类型提升的代理）
        def group_args(arg_pairs):
            out = defaultdict(list)
            for i, args in enumerate(arg_pairs):
                use_foreach = not is_dynamic(*args)  # 判断是否动态输入
                device = None
                for t in args:
                    if isinstance(t, TensorBox):
                        device = t.data.get_device()  # 获取张量所在的设备
                        break
                assert (
                    device is not None
                ), "foreach op should have at least one tensor arg"  # 断言至少有一个张量输入
                out[(device, use_foreach)].append((i, args))
            return out
        
        # 是否实现输出取决于当前节点是否没有用户或当前节点是否在 inplace_foreach_ops 中
        realize_outputs = (
            len(V.graph.current_node.users) == 0
            or V.graph.current_node.target in inplace_foreach_ops
        )
        for node in V.graph.current_node.users:
            for user in node.users:
                if not (user.op == "call_function" and (user.target in foreach_ops)):
                    realize_outputs = True  # 检查节点用户是否为 foreach 操作，如果不是则实现输出

        a_list_input = None
        for input in inputs:
            if isinstance(input, (list, tuple)):
                a_list_input = input
                break
        assert (
            a_list_input is not None
        ), "at least one input must be a list to a foreach op"  # 断言至少有一个输入是列表，以便进行 foreach 操作

        # 广播标量输入以匹配列表输入的长度
        broadcast_inputs = []
        for input in inputs:
            if not isinstance(input, (list, tuple)):
                broadcast_inputs.append([input] * len(a_list_input))
            else:
                broadcast_inputs.append(input)

        groups = group_args(zip(*broadcast_inputs))  # 按设备和 foreach 使用情况分组输入参数

        outputs = [None] * len(a_list_input)
        for (device, use_foreach), group in groups.items():
            buffer_list = []
            for (
                output_ind,
                args,
            ) in group:
                if allow_alpha:
                    output = pw_fn(*args, alpha=alpha)  # 调用 pw_fn 处理参数 args，支持 alpha 参数
                else:
                    output = pw_fn(*args)  # 调用 pw_fn 处理参数 args

                outputs[output_ind] = output  # 将处理结果放入对应的输出位置

                # 如果图中支持 foreach 并且需要实现输出，则注册输出到缓冲区列表中
                if (
                    V.graph.has_feature(device, BackendFeature.FOREACH)
                    and use_foreach
                    and realize_outputs
                ):
                    buffer_list.append(output.realize())

            if buffer_list:
                V.graph.register_list(buffer_list)  # 注册缓冲区列表到图中

        assert all(x is not None for x in outputs)  # 断言所有输出都不为 None
        return outputs  # 返回处理后的输出列表

    return inner


def to_dtype(x: TensorBox, dtype: torch.dtype, copy=False):
    src_dtype = x.get_dtype()  # 获取张量 x 的数据类型
    if src_dtype == dtype:
        return clone(x) if copy else x  # 如果源数据类型和目标数据类型一致，返回克隆或原张量

    def _to_dtype(x):
        return ops.to_dtype(x, dtype, src_dtype=src_dtype)  # 定义内部函数 _to_dtype，用于类型转换

    return make_pointwise(_to_dtype, override_return_dtype=dtype)(x)
    # 调用 make_pointwise 函数，将 _to_dtype 作为处理函数，同时指定返回类型为 dtype
# 注册一个降级处理函数，将 prims.convert_element_type 函数注册为下转函数，不进行类型提升
@register_lowering(prims.convert_element_type, type_promotion_kind=None)
def _convert_element_type(x: TensorBox, dtype: torch.dtype):
    # 如果目标数据类型或者输入张量 x 的数据类型为复数类型
    if dtype.is_complex or x.get_dtype().is_complex:
        if x.get_size():
            # 创建一个与 x 具有相同形状和 dtype 的空张量 dst
            dst = empty_like(x, dtype=dtype)
            # 使用 aten 的后备方法进行就地复制操作
            ir.InplaceCopyFallback.create(dst, x)
            return dst
        else:
            # 如果 x 是空张量，则返回一个 fallback 处理函数的结果，不加入回退集合
            return fallback_handler(
                prims.convert_element_type.default, add_to_fallback_set=False
            )(x, dtype)
    # 否则，使用 to_dtype 函数将 x 转换为目标 dtype
    return to_dtype(x, dtype, copy=True)


def to_dtype_bitcast(x: TensorBox, dtype: torch.dtype, *, copy=False):
    # 获取输入张量 x 的数据类型
    x_dtype = x.get_dtype()
    # 如果 x 的数据类型与目标 dtype 相同
    if x_dtype == dtype:
        # 如果 copy 标志为 True，则克隆 x；否则直接返回 x
        return clone(x) if copy else x

    def _get_primitive_bitwidth(dtype):
        # 根据数据类型返回其基本数据宽度（比特数）
        if dtype.is_floating_point:
            return torch.finfo(dtype).bits
        else:
            return torch.iinfo(dtype).bits

    # 获取源数据类型 x_dtype 和目标数据类型 dtype 的基本数据宽度
    src_bits = _get_primitive_bitwidth(x_dtype)
    dst_bits = _get_primitive_bitwidth(dtype)
    # 如果源数据类型的宽度与目标数据类型的宽度不同
    if src_bits != dst_bits:
        # 使用 aten.view.dtype 函数进行回退，实现不同位宽的快速实现
        return fallback_handler(aten.view.dtype)(x, dtype)

    def _to_dtype_bitcast(x):
        # 由于可能从 float16 或 bfloat16 提升张量类型到 float，因此需要传递原始的源数据类型 x_dtype，
        # 用于在位转换前正确构造类型转换，要求输入张量类型的位宽与目标类型相同。
        return ops.to_dtype_bitcast(x, dtype, x_dtype)

    # 使用 make_pointwise 函数，指定返回 dtype 并执行 _to_dtype_bitcast 函数
    return make_pointwise(_to_dtype_bitcast, override_return_dtype=dtype)(x)


# 将 aten.view.dtype 函数注册为下转函数，不进行类型提升
@register_lowering(aten.view.dtype, type_promotion_kind=None)
def _view_dtype(x: TensorBox, dtype: torch.dtype):
    # 如果目标数据类型或者输入张量 x 的数据类型为复数类型
    if dtype.is_complex or x.get_dtype().is_complex:
        # 创建一个复数视图，使用 ir.ComplexView.create 构造复数视图操作
        return TensorBox.create(
            ir.ComplexView.create(torch.ops.aten.view.dtype, x, dtype)
        )
    # 否则，使用 to_dtype_bitcast 函数将 x 转换为目标 dtype
    return to_dtype_bitcast(x, dtype, copy=True)


def to_device(x: TensorBox, device: torch.device, *, copy=False):
    # 解码设备，获取设备对象
    device = decode_device(device)
    # 如果输入张量 x 的设备与目标设备相同
    if x.get_device() == device:
        # 如果 copy 标志为 True，则克隆 x；否则直接返回 x
        return clone(x) if copy else x
    # 创建一个新张量，复制 x 到目标设备上，并返回结果
    return TensorBox.create(ir.DeviceCopy.create(x, device))


# 将 prims.device_put 函数注册为下转函数，不进行类型提升
@register_lowering(prims.device_put, type_promotion_kind=None)
def _device_put(x: TensorBox, device: torch.device):
    # 使用 to_device 函数将输入张量 x 放置到目标设备上，并进行复制
    return to_device(x, device, copy=True)


def register_pointwise(
    aten_fn,
    name=None,
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
    override_return_dtype=None,
    override_fn_when_input_bool=None,
    allow_alpha=False,
    use_libdevice_for_f64=False,
    triton_fallback=None,
):
    """注册一个逐点函数，将 ops.{name} 映射到输入"""
    # 如果没有指定名称，则使用 aten_fn 函数的名称作为名称
    name = name or aten_fn.__name__
    # 使用 ops_wrapper 函数包装名称，返回操作函数
    fn = ops_wrapper(name)
    # 如果指定使用 libdevice 处理 float64 类型的操作
    if use_libdevice_for_f64:
        # 根据操作名称生成 libdevice 版本的操作包装器
        fn_libdevice = ops_wrapper("libdevice_" + name)
    
    # 如果指定了输入为布尔值时的操作重写函数
    if override_fn_when_input_bool is not None:
        # 使用操作包装器生成重写函数的包装器
        override_fn_when_input_bool = ops_wrapper(override_fn_when_input_bool)

    # 根据给定的参数创建一个 pointwise 函数
    fn = make_pointwise(
        fn,
        override_return_dtype=override_return_dtype,
        override_fn_when_input_bool=override_fn_when_input_bool,
        override_fn_when_cuda_float64=fn_libdevice if use_libdevice_for_f64 else None,  # type: ignore[possibly-undefined]
        allow_alpha=allow_alpha,
        triton_fallback=triton_fallback,
    )
    
    # 根据 aten_fn 注册一个下降（lowering）函数
    fn = register_lowering(
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
    )(fn)

    # 如果 prims 模块中存在指定名称的函数
    if hasattr(prims, name):
        # 根据 prims 模块中对应名称的函数注册一个下降（lowering）函数
        register_lowering(
            getattr(prims, name),
            type_promotion_kind=None,
            convert_input_to_bool=convert_input_to_bool,
        )(fn)
    
    # 返回生成的函数 fn
    return fn
# 定义一个函数，用于注册 frexp 操作的点逐函数
def register_frexp():
    """A pointwise function that maps ops.frexp to inputs"""
    # 设置操作的名称为 "frexp"
    name = "frexp"
    # 调用 ops_wrapper 函数获取 "frexp" 操作的封装函数
    frexp = ops_wrapper("frexp")

    # 定义 frexp0 函数，从 frexp 返回值中获取第一个元素
    def frexp0(*args, **kwargs):
        return frexp(*args, **kwargs)[0]

    # 定义 frexp1 函数，从 frexp 返回值中获取第二个元素
    def frexp1(*args, **kwargs):
        return frexp(*args, **kwargs)[1]

    # 创建点逐函数列表 pw_fns，分别应用于 frexp0 和 frexp1
    pw_fns = [
        make_pointwise(frexp0),
        make_pointwise(frexp1, override_return_dtype=torch.int32),
    ]

    # 定义 fn 函数，返回 pw_fns[0] 和 pw_fns[1] 的结果
    def fn(*args, **kwargs):
        return pw_fns[0](*args, **kwargs), pw_fns[1](*args, **kwargs)

    # 使用 register_lowering 注册 fn 函数作为 aten.frexp 的降低版本
    fn = register_lowering(
        aten.frexp,
    )(fn)

    # 如果 prims 模块中有与 name 变量同名的属性，注册对应的降低函数
    if hasattr(prims, name):
        register_lowering(
            getattr(prims, name),
            type_promotion_kind=None,
        )(fn)
    return fn


# 调用 register_frexp 函数
register_frexp()


# 定义 foreach_pointwise 函数，用于注册 foreach_pointwise 操作的点逐函数
def register_foreach_pointwise(
    aten_fn,
    pointwise_lowering_fn,
    allow_alpha=False,
):
    # 创建 foreach_pointwise 函数，通过 pointwise_lowering_fn 和 allow_alpha 参数创建点逐函数
    fn = make_foreach_pointwise(pointwise_lowering_fn, allow_alpha=allow_alpha)
    # 将创建的点逐函数注册为 aten_fn 的降低版本
    fn = _register_foreach_lowering(aten_fn, fn)
    return fn


# 使用 register_lowering 注册 aten.where 操作的降低版本
@register_lowering(aten.where, broadcast=False, type_promotion_kind=None)
def where(cond, a, b):
    # 定义 fn 函数，用于执行 ops.where 操作
    def fn(*args):
        return ops.where(*args)

    # 如果 a 是 float 或 int 类型，将其视为常量，创建相同形状的常量张量
    if isinstance(a, (float, int)):
        a = constant_like(a)(b)
    # 如果 b 是 float 或 int 类型，将其视为常量，创建相同形状的常量张量
    if isinstance(b, (float, int)):
        b = constant_like(b)(a)

    # 组合参数列表 args，并获取其推广后的数据类型
    args = [cond, a, b]
    dtype = get_promoted_dtype(
        args[1], args[2], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    # 找到 args 中的张量箱对象，并对其进行广播
    indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
    for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
        args[i] = x
    # 对于 args 中的常量进行扩展视图的创建
    for i in range(len(args)):
        if isinstance(args[i], ir.Constant):
            args[i] = ExpandView.create(args[i], list(args[indices[0]].get_size()))
    # 创建并返回 ops.where 操作的点逐函数，并将其应用于 args[0]、args[1] 和 args[2]
    return make_pointwise(fn, override_return_dtype=dtype)(
        args[0], to_dtype(args[1], dtype), to_dtype(args[2], dtype)
    )


# 使用 register_lowering 注册 aten.broadcast_tensors 操作的降低版本
@register_lowering(aten.broadcast_tensors, broadcast=False, type_promotion_kind=None)
def broadcast_tensors(*inputs):
    # 如果 inputs 的长度为 1，且为列表或元组类型，则递归调用 broadcast_tensors
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        return broadcast_tensors(*inputs[0])
    # 使用 reduce 函数计算目标符号形状，然后对 inputs 中的张量进行广播
    target: List[sympy.Expr] = functools.reduce(
        broadcast_symbolic_shapes, [x.get_size() for x in inputs], []
    )
    outputs = []
    for x in inputs:
        sizes = x.get_size()
        # 如果 sizes 的长度与 target 的长度不同，或者存在不匹配的维度，则扩展 x
        if len(sizes) != len(target) or any(
            ((a == 1 and b != 1) or (a != 1 and b == 1)) for a, b in zip(sizes, target)
        ):
            x = expand(x, target)
        outputs.append(x)
    return outputs


# 使用 register_lowering 注册 aten.alias、aten.detach、aten.detach_、aten.lift 和 prims.view_of 操作的降低版本
@register_lowering([aten.alias, aten.detach, aten.detach_, aten.lift, prims.view_of])
def nop(x):
    # 返回输入参数 x，此操作由 AOT autograd 处理
    return x  # AOT autograd handles this for us


# 如果 aten 模块中有 "lift_fresh" 属性，注册 aten.lift_fresh 操作的降低版本
if hasattr(aten, "lift_fresh"):
    register_lowering(aten.lift_fresh)(nop)


# 使用 register_lowering 注册 aten.squeeze 操作的降低版本
@register_lowering(aten.squeeze, type_promotion_kind=None)
def squeeze(x, dim=None):
    # 确保 x 是 TensorBox 类型的对象
    assert isinstance(x, TensorBox)
    # 如果 dim 为 None，则返回一个 SqueezeView 创建的 TensorBox 对象
    if dim is None:
        return TensorBox(SqueezeView.create(x.data))
    # 使用 canonicalize_dims 函数规范化维度参数，确保维度参数与 x 的尺寸长度匹配
    dim = canonicalize_dims(len(x.get_size()), dim)
    # 将 dim 转换为集合，如果 dim 不是元组，则将其转换为单元素集合
    dims = set((dim,) if not isinstance(dim, tuple) else dim)

    # 创建一个新的空列表，用于存储新的形状信息
    new_shape = []
    # 遍历 x 的尺寸信息
    for d, s in enumerate(x.get_size()):
        # 如果当前维度 d 在 dims 中，并且 V.graph.sizevars.evaluate_expr(sympy.Eq(s, 1)) 结果为假
        if not (d in dims and V.graph.sizevars.evaluate_expr(sympy.Eq(s, 1))):
            # 将当前尺寸 s 添加到新形状列表中
            new_shape.append(s)

    # 如果新形状 new_shape 不等于原始 x 的尺寸信息，则调用 view 函数创建视图，否则返回原始 x
    return view(x, new_shape) if new_shape != x.get_size() else x
@register_lowering(aten.squeeze_copy, type_promotion_kind=None)
def squeeze_copy(x, dim=None):
    # 注册对 aten.squeeze_copy 函数的降级处理，不进行类型提升
    # 返回对输入张量 x 在指定维度 dim 上挤压后的克隆张量
    return clone(squeeze(x, dim))


@register_lowering([aten.squeeze_])
def squeeze_(x, dim=None):
    # 注册对 aten.squeeze_ 函数的降级处理
    # 对输入张量 x 进行挤压操作，并确保 x 和结果 val 均为 TensorBox 类型
    val = squeeze(x, dim)
    assert isinstance(x, TensorBox)
    assert isinstance(val, TensorBox)
    # 将 val 的数据复制给 x 的数据，并返回 x 本身
    x.data = val.data
    return x


@register_lowering(aten.isinf)
def isinf(x):
    # 注册对 aten.isinf 函数的降级处理
    # 如果输入 x 是整数类型，则返回一个与 x 相同形状的全 False 的 torch.bool 张量
    if is_integer_type(x):
        return full_like(x, False, dtype=torch.bool)
    # 否则，获取 "isinf" 的操作包装器，并以 override_return_dtype=torch.bool 参数创建一个逐点运算函数，对输入 x 执行该操作
    fn = ops_wrapper("isinf")
    return make_pointwise(fn, override_return_dtype=torch.bool)(x)


@register_lowering(aten.isnan)
def isnan(x):
    # 注册对 aten.isnan 函数的降级处理
    # 如果输入 x 是整数类型，则返回一个与 x 相同形状的全 False 的 torch.bool 张量
    if is_integer_type(x):
        return full_like(x, False, dtype=torch.bool)
    # 否则，获取 "isnan" 的操作包装器，并以 override_return_dtype=torch.bool 参数创建一个逐点运算函数，对输入 x 执行该操作
    fn = ops_wrapper("isnan")
    return make_pointwise(fn, override_return_dtype=torch.bool)(x)


@register_lowering(aten.ceil)
def ceil(x):
    # 注册对 aten.ceil 函数的降级处理
    # 如果输入 x 是整数类型，则返回 x 的克隆
    if is_integer_type(x):
        return clone(x)
    # 否则，获取 "ceil" 的操作包装器，并创建一个逐点运算函数，对输入 x 执行该操作
    fn = ops_wrapper("ceil")
    return make_pointwise(fn)(x)


@register_lowering(aten.floor)
def floor(x):
    # 注册对 aten.floor 函数的降级处理
    # 如果输入 x 是整数类型，则返回 x 的克隆
    if is_integer_type(x):
        return clone(x)
    # 否则，获取 "floor" 的操作包装器，并创建一个逐点运算函数，对输入 x 执行该操作
    fn = ops_wrapper("floor")
    return make_pointwise(fn)(x)


@register_lowering(aten.round.default)
def round(x):
    # 注册对 aten.round.default 函数的降级处理
    # 如果输入 x 是整数类型，则返回 x 的克隆
    if is_integer_type(x):
        return clone(x)
    else:
        # 否则，获取 "round" 的操作包装器，并创建一个逐点运算函数，对输入 x 执行该操作
        fn = ops_wrapper("round")
        return make_pointwise(fn)(x)


@register_lowering(aten.trunc)
def trunc(x):
    # 注册对 aten.trunc 函数的降级处理
    # 如果输入 x 是整数类型，则返回 x 的克隆
    if is_integer_type(x):
        return clone(x)
    # 否则，获取 "trunc" 的操作包装器，并创建一个逐点运算函数，对输入 x 执行该操作
    fn = ops_wrapper("trunc")
    return make_pointwise(fn)(x)


@register_lowering(aten.expand, type_promotion_kind=None)
def expand(x, sizes):
    # 注册对 aten.expand 函数的降级处理，不进行类型提升
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    (x,) = promote_constants([x])
    if isinstance(x, ir.BaseConstant):
        # 如果 x 是常量，则创建一个 ExpandView 对象，将其尺寸扩展到指定 sizes
        return ExpandView.create(x, tuple(sizes))
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    # 如果 x 的当前尺寸与 sizes 相同，则直接返回 x
    if tuple(x.get_size()) == tuple(sizes):
        return x

    if not free_unbacked_symbols(x.get_size()):
        # 如果 x 的尺寸不含未支持的符号，计算 x 尺寸的乘积
        x_size_product = V.graph.sizevars.size_hint(sympy_product(x.get_size()))
        # TODO: 更好的方法是在广播之前实现输入，因为通常尺寸将不为零。
        # 但是，我们不能像下面这样直接做，因为我们会在这里卡住 size_hint
        if x_size_product > 0 and not free_unbacked_symbols(sizes):
            # 可能需要在广播之前实现输入
            x.mark_reuse(
                V.graph.sizevars.size_hint(sympy_product(sizes)) // x_size_product
            )
    # 返回一个 TensorBox 对象，将 x 的数据扩展到指定 sizes
    return TensorBox(ExpandView.create(x.data, tuple(sizes)))


@register_lowering(prims.broadcast_in_dim, type_promotion_kind=None)
def broadcast_in_dim(a, shape, broadcast_dimensions):
    # 注册对 prims.broadcast_in_dim 函数的降级处理，不进行类型提升
    s = list(shape)
    # 根据 broadcast_dimensions 将 s 中对应位置置为 -1
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    # 将 v 逐一在 s 中非 -1 的位置上插入维度
    for idx, x in enumerate(s):
        if x != -1:
            v = unsqueeze(v, idx)

    # 将 v 扩展到指定 shape
    return expand(v, shape)


@register_lowering(aten.expand_as, type_promotion_kind=None)
# 注册一个函数，使其能够在下降时处理 aten.repeat 操作
@register_lowering(aten.repeat)
def repeat(x, repeats):
    # 获取输入张量的原始大小
    old_size = list(x.get_size())
    # 如果重复次数数组比原始大小的维数多，则扩展原始大小
    if len(repeats) > len(old_size):
        old_size = [sympy.Integer(1)] * (len(repeats) - len(old_size)) + old_size
        x = view(x, list(old_size))  # 使用视图函数调整张量形状

    # 断言重复次数数组的长度与张量的维数相同
    assert len(repeats) == len(x.get_size())

    # 初始化新的张量大小
    new_size = list(x.get_size())

    zero_tensor = False  # 标志变量，用于检测是否有零张量的情况
    # 计算新的张量大小
    for i in range(len(repeats)):
        if repeats[i] == 0:
            zero_tensor = True  # 发现重复次数为0的情况
        new_size[i] = new_size[i] * repeats[i]

    # 如果有零张量情况，则返回一个空张量
    if zero_tensor:
        return empty(new_size, dtype=x.get_dtype(), device=x.get_device())
    # 如果所有维度的重复次数为1或原始大小为1，则返回扩展后的克隆张量
    if all((a == 1 or b == 1) for a, b in zip(repeats, old_size)):
        return clone(expand(x, new_size))

    x_loader: Callable[[Any], Any]  # 声明 x_loader 变量的类型为可调用对象

    # 定义内部函数 inner_fn，用于索引处理
    def inner_fn(index):
        assert len(index) == len(repeats)
        index = list(index)
        for i in range(len(repeats)):
            if repeats[i] != 1:
                if old_size[i] == 1:
                    index[i] = sympy.Integer(0)
                else:
                    index[i] = ModularIndexing(index[i], 1, old_size[i])
        return x_loader(index)

    # 计算旧大小的乘积
    old_size_product = V.graph.sizevars.size_hint(sympy_product(old_size))
    if old_size_product > 0:
        # 标记重用性，根据新大小和旧大小的乘积比例
        x.mark_reuse(
            V.graph.sizevars.size_hint(sympy_product(new_size)) // old_size_product
        )

    x_loader = x.make_loader()  # 获取加载器函数
    # 创建 Pointwise 对象，使用内部函数处理索引
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(new_size),
    )


# 注册一个函数，处理 aten._unsafe_view、aten.view 和 aten.reshape 操作的下降
@register_lowering(aten._unsafe_view, type_promotion_kind=None)
@register_lowering(aten.view, type_promotion_kind=None)
@register_lowering(aten.reshape, type_promotion_kind=None)
def view(x, sizes):
    assert isinstance(x, TensorBox)  # 断言 x 是 TensorBox 类型的对象
    assert isinstance(sizes, (list, tuple))  # 断言 sizes 是列表或元组类型
    return TensorBox(View.create(x.data, sizes))  # 返回一个包含视图创建的 TensorBox 对象


# 注册一个函数，处理 aten.permute 操作的下降
@register_lowering(aten.permute, type_promotion_kind=None)
def permute(x, dims):
    assert isinstance(x, TensorBox)  # 断言 x 是 TensorBox 类型的对象
    assert isinstance(dims, (list, tuple))  # 断言 dims 是列表或元组类型
    return TensorBox(PermuteView.create(x.data, tuple(dims)))  # 返回一个包含排列视图创建的 TensorBox 对象


# 注册一个函数，处理 aten.slice 操作的下降
@register_lowering(aten.slice, type_promotion_kind=None)
def slice_(x, dim=0, start=0, end=2**63, step=1, clamp=True):
    assert isinstance(x, TensorBox)  # 断言 x 是 TensorBox 类型的对象
    dim = _validate_dim(x, dim, 0)  # 验证并调整维度参数 dim
    return TensorBox(ir.SliceView.create(x.data, dim, start, end, step, clamp=clamp))  # 返回包含切片视图创建的 TensorBox 对象


# 注册一个函数，处理 aten.as_strided 操作的下降
@register_lowering(aten.as_strided, type_promotion_kind=None)
def as_strided(x, size, stride, storage_offset=None):
    if isinstance(x, TensorBox) and isinstance(x.data, ir.BaseView):
        # 如果 x 是 TensorBox 类型且其数据是 BaseView 类型，则忽略视图
        x = x.data.unwrap_view()
    x.realize()  # 实现张量 x
    if not ir.is_storage_and_layout(x):
        # 如果张量 x 没有存储和布局信息，则抛出 NotImplementedError 异常
        raise NotImplementedError(f"unrealized as_strided({x}, ...)")
    storage, old_layout = ir.as_storage_and_layout(x)  # 获取存储和旧布局信息
    # 创建一个新的固定布局对象 `new_layout`，该对象基于以下参数：
    # - `old_layout.device`: 继承自旧布局的设备信息
    # - `old_layout.dtype`: 继承自旧布局的数据类型信息
    # - `[sympy.expand(s) for s in size]`: 将 `size` 中每个元素进行符号表达式的扩展，并组成列表
    # - `[sympy.expand(s) for s in stride]`: 将 `stride` 中每个元素进行符号表达式的扩展，并组成列表
    # - `sympy.expand(storage_offset or 0)`: 如果 `storage_offset` 存在则扩展其值，否则为0
    # 创建一个 TensorBox 对象，使用 `storage` 数据和新的固定布局 `new_layout` 进行重新解释视图
    return TensorBox(ir.ReinterpretView(storage, new_layout))
@register_lowering(aten.as_strided_, type_promotion_kind=None)
# 注册一个降级操作，用于处理 as_strided_ 函数
def as_strided_(x, size, stride, storage_offset=None):
    assert isinstance(x, TensorBox)
    # 将输入张量 x 的数据字段替换为调用 as_strided 函数后的数据字段
    x.data = as_strided(x, size, stride, storage_offset).data
    return x


@register_lowering(aten.as_strided_copy, type_promotion_kind=None)
# 注册一个降级操作，用于处理 as_strided_copy 函数
def as_strided_copy(x, size, stride, storage_offset=None):
    # 调用 as_strided 函数生成新的张量 result
    result = as_strided(x, size, stride, storage_offset)
    # 返回 result 的克隆
    return clone(result)


def pointwise_cat(inputs, dim=0):
    # (inclusive, exclusive)
    # inputs_ranges 用于存储输入张量的维度范围
    inputs_ranges: List[Tuple[sympy.Expr, sympy.Expr]] = []
    prev_end = 0
    for inp in inputs:
        # 计算每个输入张量在指定维度上的范围，并添加到 inputs_ranges 中
        inputs_ranges.append((prev_end, prev_end + inp.get_size()[dim]))  # type: ignore[arg-type]
        prev_end = inputs_ranges[-1][-1]  # type: ignore[assignment]

    # inputs_loaders 用于存储每个输入张量的加载器
    inputs_loaders = [inp.make_loader() for inp in inputs]

    def inner_fn(idx):
        idx_dim = ops.index_expr(idx[dim], torch.int64)

        masks = []
        masked_loads = []
        for i in range(len(inputs)):
            # 计算每个输入张量在指定维度上的起始和结束条件
            start = (
                ops.constant(0, torch.int64)
                if i == 0
                else ops.index_expr(inputs_ranges[i][0], torch.int64)
            )
            end = ops.index_expr(inputs_ranges[i][1], torch.int64)

            # 创建条件掩码，用于选择合适的张量部分
            start_cond = ops.ge(idx_dim, start)
            end_cond = ops.lt(idx_dim, end)
            if i == 0:
                mask = end_cond
            elif i == len(inputs) - 1:
                mask = start_cond
            else:
                mask = ops.and_(start_cond, end_cond)

            masks.append(mask)
            idx_load = list(idx)

            # 调整索引以保持正确的张量连接
            idx_load[dim] = Identity(idx_load[dim] - inputs_ranges[i][0])

            # 创建带有条件的加载操作，根据掩码加载正确的张量部分
            masked_loads.append(
                ops.masked(
                    mask,
                    lambda: inputs_loaders[i](idx_load),
                    0.0,  # 这个值应该不会被使用到
                ),
            )

        # 根据掩码和加载的结果，生成最终的张量
        next_val = masked_loads[-1]
        for i in range((len(inputs)) - 2, -1, -1):
            next_val = ops.where(
                masks[i],
                masked_loads[i],
                next_val,
            )
        return next_val

    # 计算输出张量的新大小，以便创建 Pointwise 对象
    new_size = list(inputs[0].get_size())
    new_size[dim] = inputs_ranges[-1][-1]

    # 创建并返回 Pointwise 对象，用于执行 pointwise_cat 操作
    return Pointwise.create(
        device=inputs[0].get_device(),
        dtype=inputs[0].get_dtype(),
        inner_fn=inner_fn,
        ranges=new_size,
    )


@register_lowering(quantized_decomposed.quantize_per_channel, type_promotion_kind=None)
# 注册一个降级操作，用于处理 quantized_decomposed.quantize_per_channel 函数
def quantized_decomposed_quantize_per_channel(
    input: TensorBox,
    scales: TensorBox,
    zero_points: TensorBox,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> TensorBox:
    # 此函数用于按通道量化输入张量
    # 断言检查：确保 scales 的尺寸为 1
    assert len(scales.get_size()) == 1, "expect scales 1 dim"
    # 断言检查：确保 zero_points 的尺寸为 1
    assert len(zero_points.get_size()) == 1, "expect zero_points 1 dim"

    # 如果输入的数据类型为 torch.bfloat16，则将其转换为 torch.float32
    if input.get_dtype() == torch.bfloat16:
        input = to_dtype(input, torch.float32)
    # 断言检查：确保输入的数据类型为 torch.float32
    assert (
        input.get_dtype() == torch.float32
    ), f"Expecting input to have dtype torch.float32, but got dtype: {input.get_dtype()}"
    # 断言检查：确保 axis 的值小于输入张量的维度数
    assert axis < len(
        input.get_size()
    ), f"Expecting axis to be < {len(input.get_size())}"

    # 创建输入、scales 和 zero_points 的加载器
    input_loader = input.make_loader()
    scales_loader = scales.make_loader()
    zero_points_loader = zero_points.make_loader()

    # 定义内部函数 inner_fn，用于处理每个索引 idx
    def inner_fn(idx):
        channel_idx = (idx[axis],)

        # 加载对应索引的 input 数据
        input = input_loader(idx)
        # 加载对应通道索引的 scale 数据
        scale = scales_loader(channel_idx)
        # 加载对应通道索引的 zero_point 数据
        zero_point = zero_points_loader(channel_idx)
        # 创建量化范围的常量 qmin 和 qmax，并确保其数据类型为 torch.float32
        qmin, qmax = _create_constants(quant_min, quant_max, dtype=torch.float32)

        # 如果 scales 的数据类型不是 torch.float32，则转换为 torch.float32
        if scales.dtype != torch.float32:
            scale = ops.to_dtype(scale, torch.float32)
        # 如果 zero_points 的数据类型不是 torch.int32，则转换为 torch.int32
        if zero_points.dtype != torch.int32:
            zero_point = ops.to_dtype(zero_point, torch.int32)
        # 计算输入经过量化操作后的值
        inv_scale = ops.reciprocal(scale)
        val = ops.round(input * inv_scale) + zero_point
        # 将结果值夹紧在 qmin 和 qmax 之间
        clamped = ops.maximum(qmin, ops.minimum(qmax, val))
        # 将结果转换为指定的数据类型 dtype
        return ops.to_dtype(clamped, dtype)

    # 创建 Pointwise 对象并返回
    return Pointwise.create(
        device=input.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )
@register_lowering(
    quantized_decomposed.dequantize_per_channel, type_promotion_kind=None
)
def quantized_decomposed_dequantize_per_channel(
    input: TensorBox,
    scales: TensorBox,
    zero_points: TensorBox,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> TensorBox:
    # 确保scales是一维的张量
    assert len(scales.get_size()) == 1, "expect scales 1 dim"
    # 确保zero_points是一维的张量
    assert len(zero_points.get_size()) == 1, "expect zero_points 1 dim"
    # 确保输入的数据类型与指定的dtype一致
    assert (
        input.get_dtype() == dtype
    ), f"Expecting input to have dtype {dtype}, but got dtype: {input.get_dtype()}"
    # 确保指定的轴在输入张量的维度范围内
    assert axis < len(
        input.get_size()
    ), f"Expecting axis to be < {len(input.get_size())}"

    # 创建输入张量的加载器
    input_loader = input.make_loader()
    # 创建scales张量的加载器
    scales_loader = scales.make_loader()
    # 创建zero_points张量的加载器
    zero_points_loader = zero_points.make_loader()

    def inner_fn(idx):
        # 通过索引获取通道索引
        channel_idx = (idx[axis],)

        # 加载输入张量的数据
        input = input_loader(idx)
        # 加载指定通道的scale值
        scale = scales_loader(channel_idx)
        # 加载指定通道的zero_point值
        zero_point = zero_points_loader(channel_idx)

        # 如果scales的数据类型不是torch.float32，则转换为torch.float32
        if scales.dtype != torch.float32:
            scale = ops.to_dtype(scale, torch.float32)
        # 如果zero_points的数据类型不是torch.float32，则转换为torch.float32
        if zero_points.dtype != torch.float32:
            zero_point = ops.to_dtype(zero_point, torch.float32)
        
        # 计算量化后的值
        val = ops.sub(ops.to_dtype(input, torch.float32), zero_point) * scale
        return val

    # 创建Pointwise对象并返回
    return Pointwise.create(
        device=input.get_device(),
        dtype=torch.float32,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.quantize_per_tensor.default, type_promotion_kind=None
)
def quantized_decomposed_quantize_per_tensor_default(
    input: TensorBox,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> TensorBox:
    # 如果输入的数据类型是torch.bfloat16，则转换为torch.float32
    if input.get_dtype() == torch.bfloat16:
        input = to_dtype(input, torch.float32)
    # 确保输入的数据类型是torch.float32
    assert (
        input.get_dtype() == torch.float32
    ), f"Expecting input to have dtype torch.float32, but got dtype: {input.get_dtype()}"

    # 创建输入张量的加载器
    input_loader = input.make_loader()

    def inner_fn(idx, scale, zero_point):
        # 加载输入张量的数据
        input = input_loader(idx)
        # 创建逆标度和零点的常数
        inv_scale, zero_point = _create_constants(
            1.0 / scale, zero_point, dtype=torch.float32
        )
        # 计算量化后的值
        val = ops.round(input * inv_scale) + zero_point
        # 创建量化范围的常数
        qmin, qmax = _create_constants(quant_min, quant_max, dtype=torch.float32)
        # 对值进行夹紧操作
        clamped = ops.minimum(ops.maximum(val, qmin), qmax)
        # 将结果转换为指定的dtype
        return ops.to_dtype(clamped, dtype)

    # 创建Pointwise对象并返回
    return Pointwise.create(
        device=input.get_device(),
        dtype=dtype,
        inner_fn=functools.partial(
            inner_fn, scale=float(scale), zero_point=int(zero_point)
        ),
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.dequantize_per_tensor.default, type_promotion_kind=None
)
def quantized_decomposed_dequantize_per_tensor_default(
    input: TensorBox,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> TensorBox:
    # 确保输入的数据类型是torch.float32
    assert (
        input.get_dtype() == torch.float32
    ), f"Expecting input to have dtype torch.float32, but got dtype: {input.get_dtype()}"
    dtype: torch.dtype,


# 定义一个变量dtype，其类型为torch.dtype，表示数据类型
@register_lowering(
    quantized_decomposed.quantize_per_tensor.tensor, type_promotion_kind=None
)
def quantized_decomposed_quantize_per_tensor_tensor(
    input: TensorBox,
    scale: TensorBox,
    zero_point: TensorBox,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> TensorBox:
    # 检查输入张量的数据类型是否符合预期
    assert (
        input.get_dtype() == torch.float32
    ), f"Expecting input to have dtype torch.float32, but got dtype: {input.get_dtype()}"
    # 确保 scale 张量是标量
    assert len(scale.get_size()) == 0 or (
        len(scale.get_size()) == 1 and scale.get_size()[0] == 1
    ), "expect scale as scalar tensor"
    # 确保 zero_point 张量是标量
    assert len(zero_point.get_size()) == 0 or (
        len(zero_point.get_size()) == 1 and zero_point.get_size()[0] == 1
    ), "expect zero_point as scalar tensor"

    # 创建输入张量的加载器
    input_loader = input.make_loader()
    # 创建 scale 张量的加载器
    scale_loader = scale.make_loader()
    # 创建 zero_point 张量的加载器
    zero_point_loader = zero_point.make_loader()

    # 定义内部函数 inner_fn，接收索引 idx
    def inner_fn(idx):
        # 加载输入张量的数据
        input = input_loader(idx)
        # 获取 scale 和 zero_point 的常量值，确保它们是 torch.float32 类型
        _scale = scale_loader((0,) if len(scale.get_size()) == 1 else ())
        _zero_point = zero_point_loader((0,) if len(scale.get_size()) == 1 else ())
        if scale.dtype != torch.float32:
            _scale = ops.to_dtype(_scale, torch.float32)
        if zero_point.dtype != torch.float32:
            _zero_point = ops.to_dtype(_zero_point, torch.float32)
        # 计算量化后的值
        val = ops.round(input * ops.reciprocal(_scale)) + _zero_point
        # 获取量化的最小值和最大值
        qmin, qmax = _create_constants(quant_min, quant_max, dtype=torch.float32)
        # 对值进行截断，确保在 qmin 和 qmax 范围内
        clamped = ops.minimum(ops.maximum(val, qmin), qmax)
        # 将结果转换为指定的 dtype 类型
        return ops.to_dtype(clamped, dtype)

    # 创建并返回 Pointwise 对象
    return Pointwise.create(
        device=input.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


@register_lowering(
    quantized_decomposed.dequantize_per_tensor.tensor, type_promotion_kind=None
)
def quantized_decomposed_dequantize_per_tensor_tensor(
    input: TensorBox,
    scale: TensorBox,
    zero_point: TensorBox,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> TensorBox:
    # 确保 scale 张量是标量
    assert len(scale.get_size()) == 0 or (
        len(scale.get_size()) == 1 and scale.get_size()[0] == 1
    ), "expect scale as scalar tensor"
    # 确保零点张量的大小为空或者长度为1且唯一元素为1，否则抛出异常信息"expect zero_point as scalar tensor"
    assert len(zero_point.get_size()) == 0 or (
        len(zero_point.get_size()) == 1 and zero_point.get_size()[0] == 1
    ), "expect zero_point as scalar tensor"
    
    # 确保输入张量的数据类型为指定的dtype，否则抛出异常信息指示预期的dtype和实际的dtype
    assert (
        input.get_dtype() == dtype
    ), f"Expecting input to have dtype {dtype}, but got dtype: {input.get_dtype()}"

    # 创建输入张量的加载器
    input_loader = input.make_loader()
    # 创建缩放因子张量的加载器
    scale_loader = scale.make_loader()
    # 创建零点张量的加载器
    zero_point_loader = zero_point.make_loader()

    # 定义内部函数inner_fn，根据索引idx加载输入张量并执行张量运算
    def inner_fn(idx):
        # 使用输入加载器加载输入张量
        input = input_loader(idx)
        # 如果缩放因子张量的大小为1，则加载其值，否则加载默认索引(0,)
        _scale = scale_loader((0,) if len(scale.get_size()) == 1 else ())
        # 如果零点张量的大小为1，则加载其值，否则加载默认索引(0,)
        _zero_point = zero_point_loader((0,) if len(scale.get_size()) == 1 else ())
        
        # 如果缩放因子的数据类型不是torch.float32，则将其转换为torch.float32
        if scale.dtype != torch.float32:
            _scale = ops.to_dtype(_scale, torch.float32)
        # 如果零点张量的数据类型不是torch.float32，则将其转换为torch.float32
        if zero_point.dtype != torch.float32:
            _zero_point = ops.to_dtype(_zero_point, torch.float32)
        
        # 执行张量运算：先将输入张量转换为torch.float32类型，然后减去零点张量，再乘以缩放因子
        val = ops.sub(ops.to_dtype(input, torch.float32), _zero_point) * _scale
        return val

    # 返回使用Pointwise类创建的对象，其中包含设备信息、数据类型、内部函数以及输入张量的大小信息作为范围
    return Pointwise.create(
        device=input.get_device(),
        dtype=torch.float32,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )
@register_lowering(aten.cat)
# 注册一个针对 torch.cat 函数的优化降低策略
def cat(inputs, dim=0):
    # 判断第一个输入的设备是否是 CPU
    cpu_device = inputs[0].get_device().type == "cpu"
    # 如果所有输入的数据类型都是 torch.int8 或者 torch.uint8
    if cpu_device and all(
        input.get_dtype() in [torch.int8, torch.uint8] for input in inputs
    ):
        # TODO <leslie> 当我们直接支持 uint8 数据类型的向量化代码生成时，移除此回退
        # 对于每个输入，实现其内容
        for input in inputs:
            input.realize()
        # 如果所有输入的维度都是 4 维
        if all(len(input.get_size()) == 4 for input in inputs):
            # 要求输入以通道为最后一维的形式进行处理
            inputs, _ = require_channels_last(aten.cat, *inputs)
        # 使用默认的回退处理函数处理合并操作
        return fallback_handler(aten.cat.default)(inputs, dim)

    # 如果输入的长度为 1，直接克隆第一个输入并返回
    if len(inputs) == 1:
        return clone(inputs[0])

    # 验证维度参数的有效性，并获取推广后的数据类型
    dim = _validate_dim(inputs[0], dim, 0)
    dtype = get_promoted_dtype(
        *inputs, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    # 将所有输入转换为推广后的数据类型
    inputs = [to_dtype(inp, dtype) for inp in inputs]

    # 定义解包张量的函数，返回内部的 IR 节点
    def unwrap_tensor(x: Union[TensorBox, ir.StorageBox]) -> ir.IRNode:
        if isinstance(x, TensorBox):
            if isinstance(x.data, ir.BaseView):
                return x.data.unwrap_view()
            else:
                return x.data

        if isinstance(x, ir.StorageBox):
            return x.data

        return x

    # 判断是否为减少操作
    def is_reduction(t):
        return isinstance(t, ir.ComputedBuffer) and isinstance(t.data, ir.Reduction)

    # 判断是否可以融合减少操作
    def can_fuse_reduction(t):
        if isinstance(t, (TensorBox, ir.StorageBox)):
            return can_fuse_reduction(unwrap_tensor(t))
        return (
            is_reduction(t)
            or isinstance(t, ir.Pointwise)
            and any(
                can_fuse_reduction(V.graph.get_buffer(read))
                for read in t.get_read_names()
            )
        )

    # 判断是否存在可以融合的减少操作
    fusable_reduction = any(can_fuse_reduction(t) for t in inputs)

    # 判断是否应该降低合并操作的输入
    def should_lower_cat_input(x) -> bool:
        # 未实现的输入不会是存储和布局，我们不希望在想要融合的情况下将它们实现
        if ir.is_storage_and_layout(x):
            storage, _ = ir.as_storage_and_layout(x, freeze=False)
            return not ir.ConcatKernel.can_realize_into_without_copy(storage)

        if isinstance(x, (TensorBox, ir.StorageBox)):
            return should_lower_cat_input(unwrap_tensor(x))

        if isinstance(x, ir.Pointwise):
            return True

        return False

    # TODO: 我们观察到 pointwise_cat 优化对 CPU 的性能有负面影响，因此禁用它。
    # 在启用索引表达式的向量化后，我们将在稍后重新审视它。
    if cpu_device:
        return TensorBox(ir.ConcatKernel.create(inputs, dim))
    # 定义一个函数，用于计算操作数量
    def op_count(x):
        # 如果输入是 TensorBox 或 ir.StorageBox 类型，则递归调用 unwrap_tensor 函数
        if isinstance(x, (TensorBox, ir.StorageBox)):
            return op_count(unwrap_tensor(x))

        # 如果输入不是 ir.Pointwise 类型，则直接返回操作数量为 0
        # 这对应于直接的内存读取操作
        if not isinstance(x, ir.Pointwise):
            return 0

        # 调用 x 对象的 inner_fn_opcount 方法，获取其操作数量
        count = x.inner_fn_opcount()
        # 遍历 x 对象读取的名称列表，递归调用 op_count 获取操作数量并累加
        for read in x.get_read_names():
            count += op_count(V.graph.get_buffer(read))

        return count

    # 设置最大复杂 pointwise 合并输入的阈值
    MAX_COMPLEX_POINTWISE_CAT = 8
    # 设置最大简单操作数量的阈值
    MAX_SIMPLE_OP_COUNT = 2

    # 如果输入数量小于等于 MAX_COMPLEX_POINTWISE_CAT，或者：
    # 输入数量小于等于 config.max_pointwise_cat_inputs 并且所有输入的操作数量都小于等于 MAX_SIMPLE_OP_COUNT
    if len(inputs) <= MAX_COMPLEX_POINTWISE_CAT or (
        (len(inputs) <= config.max_pointwise_cat_inputs)
        and all(op_count(t) <= MAX_SIMPLE_OP_COUNT for t in inputs)
    ):
        # 检查当前节点的用户是否全部是 pointwise 使用
        pointwise_uses = all(is_pointwise_use(use) for use in V.current_node.users)
        
        # 判断是否应该在 pointwise 节点中进行融合，并且有任何输入可以避免实体化
        fuse_pointwise_use = (
            any(should_lower_cat_input(inp) for inp in inputs) and pointwise_uses
        )

        # 判断是否应该水平融合，在所有输入都需要复制核的情况下水平融合
        # 只有在所有 pointwise 内核都不可融合的情况下才进行水平融合
        horizontal_fuse_cat = all(
            should_lower_cat_input(inp) for inp in inputs
        ) and not any(can_fuse_reduction(t) for t in inputs)
        
        # 如果应该在 pointwise 使用中进行融合，或者应该进行水平融合并且不能进行规约融合，则返回 pointwise_cat 函数的结果
        if fuse_pointwise_use or (horizontal_fuse_cat and not fusable_reduction):
            return pointwise_cat(inputs, dim)

    # 如果不满足融合条件，则返回 ir.ConcatKernel.create 函数的结果
    return TensorBox(ir.ConcatKernel.create(inputs, dim))
@register_lowering(aten.diagonal, type_promotion_kind=None)
def diagonal(input, offset: int = 0, dim1: int = 0, dim2: int = 1):
    # 获取输入张量的原始形状
    original_shape = input.get_size()
    # 确定张量的维度数
    num_dims = len(original_shape)
    # 规范化 dim1 和 dim2 的维度索引
    dim1 = canonicalize_dim(idx=dim1, rank=num_dims)
    dim2 = canonicalize_dim(idx=dim2, rank=num_dims)

    # 检查 dim1 和 dim2 是否相同，不允许对角线维度相同
    check(
        dim1 != dim2, lambda: f"diagonal dimensions cannot be identical {dim1}, {dim2}"
    )

    # 判断 offset 是否为负数
    offset_negative = V.graph.sizevars.evaluate_expr(sympy.Lt(offset, 0))
    if offset_negative:
        # 计算对角线的大小，当 offset 为负时
        diag_size = V.graph.sizevars.evaluate_max(
            V.graph.sizevars.evaluate_min(
                original_shape[dim1] + offset, original_shape[dim2]
            ),
            0,  # type: ignore[arg-type]
        )
    else:
        # 计算对角线的大小，当 offset 为非负时
        diag_size = V.graph.sizevars.evaluate_max(
            V.graph.sizevars.evaluate_min(
                original_shape[dim1], original_shape[dim2] - offset
            ),
            0,  # type: ignore[arg-type]
        )

    # 计算基础索引位置
    base_idx = (0, 0)
    if offset_negative:
        base_idx = (-offset, 0)
    else:
        base_idx = (0, offset)

    # 构建结果张量的形状，移除 dim1 和 dim2 维度，并添加对角线的大小维度
    sizes = [s for i, s in enumerate(original_shape) if i not in (dim1, dim2)]
    sizes.append(diag_size)

    # 定义重新索引函数，用于根据对角线偏移重新计算原始索引
    def reindexer(idx):
        diag_idx = idx[-1]
        original_idx = [0] * len(original_shape)
        cur_dim = 0
        for d in range(num_dims):
            if d == dim1:
                original_idx[d] = diag_idx + base_idx[0]
            elif d == dim2:
                original_idx[d] = diag_idx + base_idx[1]
            else:
                original_idx[d] = idx[cur_dim]
                cur_dim += 1

        # 断言保证 cur_dim 应为原始形状减去两个维度后的长度
        assert cur_dim == len(original_shape) - 2
        return original_idx

    # 返回使用重索引函数生成的泛化视图张量盒子
    return TensorBox(ir.GenericView.create(input, sizes, reindexer))


@register_lowering(aten.diagonal_copy, type_promotion_kind=None)
def diagonal_copy(input, offset: int = 0, dim1: int = 0, dim2: int = 1):
    # 返回对角线函数的克隆
    return clone(diagonal(input, offset, dim1, dim2))


@register_lowering(aten.diagonal_scatter, type_promotion_kind=None)
def diagonal_scatter(input, src, offset: int = 0, dim1: int = 0, dim2: int = 1):
    # 克隆输入张量作为输出
    output = clone(input)
    # 获取对角线视图
    target = diagonal(output, offset, dim1, dim2)
    # 将 src 的值传播到目标视图中
    mutate_to(target, src)
    # 返回输出张量
    return output


@register_lowering(aten.select, type_promotion_kind=None)
def select(x, dim, idx):
    # 处理可能的负索引
    idx = View.handle_negative_index(idx, x.get_size()[dim])
    # 在指定维度上挤压张量的切片
    return squeeze(slice_(x, dim, idx, idx + 1), dim)


@register_lowering(aten.split, type_promotion_kind=None)
def split(x, sizes, dim=0, clamp=True):
    # 验证维度索引是否有效
    dim = _validate_dim(x, dim, 0)
    if isinstance(sizes, sympy.Expr):
        # 如果 sizes 是表达式，使用静态形状评估器对其进行评估
        sizes = V.graph.sizevars.evaluate_static_shape(sizes)
    if isinstance(sizes, (int, sympy.Integer)):
        # 如果 sizes 是整数，根据 x 在指定维度上的静态形状计算分割大小
        x_size = V.graph.sizevars.evaluate_static_shape(x.get_size()[dim])
        sizes = [sizes] * ((x_size + sizes - 1) // sizes)
    result = []
    start = 0
    # 遍历给定的sizes列表中的每个size值
    for size in sizes:
        # 计算当前切片的结束位置
        end = start + size
        # 调用slice_函数对输入x进行切片操作，返回切片结果并添加到结果列表中
        result.append(slice_(x, dim, start, end, clamp=clamp))
        # 更新下一个切片的起始位置为当前切片的结束位置
        start = end
    # 返回包含所有切片结果的列表
    return result
@register_lowering(aten.split_with_sizes, type_promotion_kind=None)
def split_with_sizes(x, sizes, dim=0):
    # 注册一个降低函数，用于处理 aten.split_with_sizes 操作，不进行类型提升
    return split(x, sizes, dim, clamp=False)


@register_lowering(aten.unbind, type_promotion_kind=None)
def unbind(x, dim=0):
    # 验证维度的有效性
    dim = _validate_dim(x, dim, 0)
    # 获取维度大小并进行静态形状计算
    x_size = V.graph.sizevars.evaluate_static_shape(x.get_size()[dim])
    result = []
    # 遍历指定维度上的所有索引，并将选择的结果添加到列表中
    for i in range(x_size):
        result.append(select(x, dim, i))
    return result


@register_lowering(aten.unfold, type_promotion_kind=None)
def unfold(x, dimension, size, step):
    # 获取输入张量的大小
    sizes = x.get_size()
    ndim = len(sizes)
    # 规范化维度索引
    dim = canonicalize_dim(ndim, dimension)

    if ndim == 0:
        # 如果张量是零维的，返回一个切片
        return slice_(unsqueeze(x, 0), end=size)

    dim_size = sizes[dim]
    sizevars = V.graph.sizevars
    # 断言 size 小于等于 dim_size
    sizevars.guard_leq(size, dim_size)
    # 断言 step 大于零
    sizevars.guard_lt(0, step)  # type: ignore[arg-type]

    # 计算新维度大小
    new_dim_size = FloorDiv(dim_size - size, step) + 1
    if sizevars.size_hint(dim_size) > 0:
        # 标记张量重用
        x.mark_reuse(sizevars.size_hint(CeilDiv(new_dim_size * size, dim_size)))

    # 构建输出张量的大小
    out_size = [*sizes[:dim], new_dim_size, *sizes[dim + 1 :], size]

    def reindexer(idx):
        # 重新索引张量
        dim_idx = idx[-1] + idx[dim] * step
        return (*idx[:dim], dim_idx, *idx[dim + 1 : -1])

    return TensorBox(ir.GenericView.create(x, out_size, reindexer))


@register_lowering(aten.unsqueeze, type_promotion_kind=None)
def unsqueeze(x, dim):
    # 验证维度的有效性
    dim = _validate_dim(x, dim, 1)
    # 在指定维度上插入大小为 1 的维度
    new_shape = list(x.get_size())
    new_shape.insert(dim, sympy.Integer(1))
    return view(x, new_shape)


@register_lowering(aten.unsqueeze_, type_promotion_kind=None)
def unsqueeze_(x, dim):
    # 调用 unsqueeze 函数
    val = unsqueeze(x, dim)
    # 断言 x 和 val 是 TensorBox 类型
    assert isinstance(x, TensorBox)
    assert isinstance(val, TensorBox)
    # 将 val 的数据赋值给 x 的数据
    x.data = val.data
    return x


def _validate_dim(x, dim, offset=0):
    # 断言 dim 是整数
    assert isinstance(dim, int)
    # 获取张量的维度数
    ndim = len(x.get_size())
    # 处理负数索引的情况
    if dim < 0:
        dim += ndim + offset
    # 断言 dim 在有效范围内
    assert 0 <= dim < ndim + offset
    return dim


@register_lowering(aten.glu)
def glu(x, dim=-1):
    # 验证维度的有效性
    dim = _validate_dim(x, dim, 0)
    # TODO: 在此处不要对静态形状进行保护
    # 计算新长度
    new_len = V.graph.sizevars.evaluate_static_shape(x.get_size()[dim]) // 2
    # 对张量进行切片操作
    a = slice_(x, dim, 0, new_len)
    b = slice_(x, dim, new_len, new_len * 2)
    # 返回 a 与 sigmoid(b) 的乘积
    return mul(a, sigmoid(b))


def fallback_handler(kernel, add_to_fallback_set=True):
    # 如果需要添加到回退集合中，则将核函数添加到回退集合
    if add_to_fallback_set:
        fallbacks.add(kernel)

    def handler(*args, **kwargs):
        # 对所有参数应用 TensorBox.create，并创建回退核函数
        return pytree.tree_map(
            TensorBox.create, ir.FallbackKernel.create(kernel, *args, **kwargs)
        )

    return handler


@functools.lru_cache(None)
def _warn_complex_not_supported():
    # 发出警告，不支持复杂运算符的代码生成
    warnings.warn(
        "Torchinductor does not support code generation for complex operators. Performance may be worse than eager."
    )


# There are some types (CPU) which we accept as input but not as
# output.
def unsupported_input_tensor(t: torch._subclasses.FakeTensor, parent=None):
    # 不支持读取或写入此张量类型
    "Do not support reading or writing to this tensor"
    # 检查是否为复杂类型张量
    if t.is_complex():
        # 如果是复杂类型视图，使用 IR ComplexView 支持
        if parent and parent.target in (
            torch.ops.aten.view.dtype,
            torch.ops.prims.convert_element_type.default,
        ):
            # 如果父节点存在且其目标在指定的操作列表中，则返回 False
            return False
        # 如果不是上述特定情况，则发出警告表明不支持复杂类型张量
        _warn_complex_not_supported()
        # 返回 True，表示不支持复杂类型张量
        return True
    # 如果不是复杂类型张量，则返回 False
    return False
def unsupported_output_tensor(t: torch._subclasses.FakeTensor, parent=None):
    # 不支持写入张量，但可以从中读取数据
    "Do not support writing tensor but can read from it"
    # 检查是否支持输入张量，如果是，则返回 True
    if unsupported_input_tensor(t, parent):
        return True
    # 返回是否为 CPU 张量并且禁用了 C++ 代码生成
    return t.is_cpu and config.disable_cpp_codegen


def fallback_node_due_to_unsupported_type(node: torch.fx.Node, allow_cpu_inputs=True):
    # 自定义的降级处理
    if node.target is aten.view_as_complex.default:
        return False

    # 当 `disable_cpp_codegen` 被废除后，我们应该能够移除此特殊情况
    if node.target is aten.lift_fresh_copy.default:
        return False

    def check_skip_condition(node, parent, is_output):
        # 如果 node 不是 torch.fx.Node 类型，则返回 False
        if not isinstance(node, torch.fx.Node):
            return False

        # 如果 node.meta 中没有 "val" 属性，则返回 False
        if "val" not in node.meta:
            return False

        # 遍历 node.meta["val"] 中的每个元素
        for meta in pytree.tree_leaves(node.meta["val"]):
            # 如果 meta 不是 torch._subclasses.FakeTensor 类型，则继续下一个循环
            if not isinstance(meta, torch._subclasses.FakeTensor):
                continue

            # 如果是输出节点，则检查是否为不支持的输出张量
            if is_output:
                if unsupported_output_tensor(meta, parent):
                    return True
            # 如果是输入节点，则检查是否为不支持的输入张量
            else:
                if unsupported_input_tensor(meta, parent):
                    return True

        return False

    # 只有在存在 CPU 输出时跳过代码生成，而不是输入
    for arg in pytree.arg_tree_leaves(*node.args, **node.kwargs):
        # 检查是否满足跳过条件
        if check_skip_condition(arg, node, is_output=False):
            return True

    # 检查当前节点是否满足跳过条件（作为输出节点）
    return check_skip_condition(node, node, is_output=True)


def make_fallback(op, layout_constraint=None, warn=True):
    # 断言确保在 decompositions 中不存在 op，否则引发 AssertionError
    assert op not in decompositions, f"both a fallback and a decomp for same op: {op}"
    # 如果满足以下条件，引发 AssertionError
    if (
        warn
        and bool(os.getenv("CI"))
        and get_decompositions([op])
        # 如果 fallback_random，允许不分解随机操作
        and not (
            config.fallback_random
            and op in torch._decomp.decompositions_for_rng.extra_random_decomps
        )
    ):
        # 注意：'warn' 保留自警告时期，但对于之前设置 warn=False 的操作，我们不希望在 CI 中出现错误。
        # 在 CI 中忽略“suppress errors”配置，因为此警告通常发生在启动时，不太可能优先触发一个 CI 配置而不是另一个。
        if torch._dynamo.config.suppress_errors:
            torch._dynamo.config.suppress_errors = False
            log.warning(
                "A make_fallback error occurred in suppress_errors config,"
                " and suppress_errors is being disabled to surface it."
            )
        # 抛出 AssertionError，说明应该切换到分解版本
        raise AssertionError(
            f"make_fallback({op}): a decomposition exists, we should switch to it."
            " To fix this error, either add a decomposition to core_aten_decompositions (preferred)"
            " or inductor_decompositions, and delete the corresponding `make_fallback` line."
            " Get help from the inductor team if unsure, don't pick arbitrarily to unblock yourself.",
        )
    # 定义注册回退函数，接受操作重载对象作为参数
    def register_fallback(op_overload):
        # 调用函数，将操作重载对象添加到需要实现输入的列表中
        add_needs_realized_inputs(op_overload)
        
        # 如果存在布局约束对象，则将布局约束添加到操作重载对象上
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)
        
        # 注册操作重载对象的降级处理函数，并使用回退处理程序包装它
        return register_lowering(op_overload, type_promotion_kind=None)(
            fallback_handler(op_overload)
        )

    # 如果操作对象是 OpOverloadPacket 类型的实例
    if isinstance(op, torch._ops.OpOverloadPacket):
        # 遍历操作对象的所有重载函数
        for ol in op.overloads():
            # 获取操作对象中的具体重载函数
            op_overload = getattr(op, ol)
            # 注册该重载函数的回退处理
            register_fallback(op_overload)
    
    # 如果操作对象是 OpOverload 或 HigherOrderOperator 类型的实例
    elif isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        # 直接注册操作对象的回退处理
        register_fallback(op)
    
    # 如果操作对象类型不受支持，抛出运行时异常
    else:
        raise RuntimeError(f"Unsupported fallback {op} with type {type(op)}")
# 计算给定形状的元素数量
def philox_rand_offset(shape):
    numel = 1
    for s in shape:
        numel = numel * s
    # 返回以 torch.int64 类型存储的张量
    return tensor(numel, dtype=torch.int64)


# 注册降级操作，处理 philox_rand 操作
@register_lowering(torch.ops.rngprims.philox_rand, type_promotion_kind=None)
def philox_rand(size, seed, offset, stride, device, dtype):
    # stride 参数是可选的，将来用于分布式随机操作，目前未使用
    random_pos = ir.FixedLayout(
        device,
        dtype,
        size,
        ir.FlexibleLayout.contiguous_strides(size),
    ).make_indexer()
    seed_loader = seed.make_loader()
    offset_loader = offset.make_loader()

    def inner_fn(index):
        # philox_rand 操作中的 seed 和 offset 都是张量
        # torch 中的 seed 和 offset 类型为 int64，但 tl.rand 接受 int32
        seed_index_expr = ops.to_dtype(seed_loader([]), torch.int32)
        offset_index_expr = ops.to_dtype(offset_loader([]), torch.int32)
        # 获取偏移后的位置
        rand_index_expr = ops.add(
            ops.index_expr(random_pos(index), torch.int32), offset_index_expr
        )
        # 执行随机操作
        result = ops.rand(
            seed_index_expr,
            rand_index_expr,
        )
        # 将结果转换为指定的 dtype 类型
        return ops.to_dtype(result, dtype)

    # 创建 Pointwise 对象，用于生成随机值
    random_values_node = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(size),
    )

    # 计算随机偏移量节点
    offset_node = philox_rand_offset(size)
    return random_values_node, offset_node


# 注册降级操作，处理 aten.native_dropout 操作
@register_lowering(aten.native_dropout, type_promotion_kind=None)
def native_dropout(x, p, train):
    if config.fallback_random:
        # 如果启用回退随机模式，返回使用 ir.FallbackKernel 创建的 TensorBox 对象
        return pytree.tree_map(
            TensorBox.create,
            ir.FallbackKernel.create(aten.native_dropout.default, x, p, train),
        )
    else:
        # 否则抛出断言错误
        raise AssertionError("should be handled in replace_random.py")


# 注册降级操作，处理 aten.bernoulli_ 操作
@register_lowering(aten.bernoulli_, type_promotion_kind=None)
def bernoulli_(x, *args):
    assert config.fallback_random or x.get_device() == torch.device(
        "cpu"
    ), "this should be handled in decomps unless config.fallback_random or the device is CPU"
    x.realize()
    # 根据参数确定重载操作
    op_overload = (
        aten.bernoulli_.float
        if len(args) == 0 or isinstance(args[0], float)
        else aten.bernoulli_.Tensor
    )
    # 使用 InplaceBernoulliFallback 类处理操作重载
    ir.InplaceBernoulliFallback(op_overload, x, *args)
    return x


# 注册降级操作，处理 aten.bernoulli.p 操作
@register_lowering(aten.bernoulli.p, type_promotion_kind=None)
def bernoulli_p(x, *args):
    assert config.fallback_random or x.get_device() == torch.device(
        "cpu"
    ), "this should be handled in decomps unless config.fallback_random or the device is CPU"
    # 返回调用 bernoulli_ 函数的克隆张量
    return bernoulli_(clone(x), *args)


# 不应该通常调用此函数
# 注册降级操作，处理 aten._foobar 操作
@register_lowering(aten._foobar)
def _foobar(_):
    raise AssertionError


# 使用 functools.lru_cache(1) 装饰器缓存函数调用结果
# 定义一个内部函数，用于记录使用 Triton 随机数时的警告信息
def _warn_triton_random(salt):
    log.info("using triton random, expect difference from eager")

# 定义一个函数，用于在图形创建时只警告一次使用 Triton 随机数的情况
def warn_triton_random():
    # 调用内部函数，传入图形创建时间作为盐值
    _warn_triton_random(V.graph.creation_time)

# 创建针对不同类型的随机数生成器的回退处理器
fallback_rand_default = fallback_handler(aten.rand.default)
fallback_rand_generator = fallback_handler(aten.rand.generator)
fallback_randn_default = fallback_handler(aten.randn.default)
fallback_randn_generator = fallback_handler(aten.randn.generator)
# 处理 randint 函数的回退
make_fallback(aten.randint)

# 注册对 aten.rand 函数进行降级处理的函数
@register_lowering(aten.rand)
def rand(*args, **kwargs):
    if kwargs.get("generator", None) is not None:
        return fallback_rand_generator(*args, **kwargs)
    elif config.fallback_random:
        kwargs.pop("generator", None)
        return fallback_rand_default(*args, **kwargs)
    raise AssertionError("should have been handled in replace_random.py")

# 注册对 aten.randn 函数进行降级处理的函数
@register_lowering(aten.randn)
def randn(*args, **kwargs):
    if kwargs.get("generator", None) is not None:
        return fallback_randn_generator(*args, **kwargs)
    elif config.fallback_random:
        kwargs.pop("generator", None)
        return fallback_randn_default(*args, **kwargs)
    raise AssertionError("should have been handled in replace_random.py")

# 注册对 inductor_prims.force_stride_order 函数进行降级处理的函数
@register_lowering(inductor_prims.force_stride_order, type_promotion_kind=None)
def inductor_force_stride_order(input_tensor, stride):
    # 获取输入张量的步长顺序
    stride_order = ir.get_stride_order(stride)
    # 要求外部内核使用指定的步长顺序
    return ir.ExternKernel.require_stride_order(input_tensor, stride_order)

# 注册对 inductor_prims.seed 函数进行降级处理的函数
@register_lowering(inductor_prims.seed, type_promotion_kind=None)
def inductor_seed(device: torch.device):
    raise AssertionError("should be handled in fuse_seed_creation_pass()")

# 注册对 inductor_prims.seeds 函数进行降级处理的函数
@register_lowering(inductor_prims.seeds, type_promotion_kind=None)
def inductor_seeds(count, device):
    # 警告使用 Triton 随机数
    warn_triton_random()
    # 创建一个随机种子对象
    return TensorBox.create(ir.RandomSeeds(count, decode_device(device)))

# 注册对 inductor_prims.lookup_seed 函数进行降级处理的函数
@register_lowering(inductor_prims.lookup_seed, type_promotion_kind=None)
def inductor_lookup_seed(seeds, index):
    # 定义内部函数，用于加载种子数据
    def inner_fn(_):
        return ops.load_seed(seeds.get_name(), index)

    # 创建点对点操作，使用种子的设备和数据类型
    return Pointwise.create(
        device=seeds.get_device(),
        dtype=seeds.get_dtype(),
        inner_fn=inner_fn,
        ranges=[],  # 空范围列表
    )

# 注册对 inductor_prims.random 函数进行降级处理的函数
@register_lowering(inductor_prims.random, type_promotion_kind=None)
def inductor_random(size: List[int], seed: TensorBox, mode: str, *, offset: int = 0):
    # 断言不使用回退随机数生成器
    assert not config.fallback_random
    # 断言模式是 'rand' 或 'randn'
    assert mode in ("rand", "randn")
    # 复制 size 列表
    size = [*size]
    # 指定数据类型为 torch.float32
    dtype = torch.float32
    # 获取种子的设备
    device = seed.get_device()
    # 创建固定布局对象，指定设备、数据类型、大小、灵活布局和偏移量
    random_pos = ir.FixedLayout(
        device, dtype, size, ir.FlexibleLayout.contiguous_strides(size), offset=offset
    ).make_indexer()
    # 创建种子加载器
    seed_loader = seed.make_loader()

    # 定义内部函数，用于生成随机数
    def inner_fn(index):
        return getattr(ops, mode)(
            seed_loader([]),
            ops.index_expr(random_pos(index), torch.int32),
        )

    # 创建点对点操作，使用指定的设备、数据类型和内部函数
    result = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[*size],  # 大小范围列表
    )
    result.realize()
    # 调用 result 对象的 realize() 方法，执行特定的操作或计算，修改 result 对象的内部状态或生成结果

    return result
    # 返回经过 realize() 方法处理后的 result 对象
@register_lowering(inductor_prims.randint, type_promotion_kind=None)
def inductor_randint(
    low: int, high: int, size: List[int], seed: TensorBox, *, offset: int = 0
):
    # 确保不使用回退随机生成器
    assert not config.fallback_random
    # 将 size 转换为列表以确保可变性
    size = [*size]
    # 设置数据类型为 torch.int64
    dtype = torch.int64
    # 获取种子张量所在设备
    device = seed.get_device()
    # 创建随机位置对象，使用固定布局和偏移量
    random_pos = ir.FixedLayout(
        device, dtype, size, ir.FlexibleLayout.contiguous_strides(size), offset=offset
    ).make_indexer()
    # 创建种子加载器
    seed_loader = seed.make_loader()

    def inner_fn(index):
        # 调用 ops.randint64 函数生成随机整数
        return ops.randint64(
            seed_loader([]),
            ops.index_expr(random_pos(index), torch.int32),
            ops.index_expr(low, torch.int64),
            ops.index_expr(high, torch.int64),
        )

    # 返回 Pointwise 对象，用于执行逐点操作
    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[*size],
    )


@register_lowering(aten.bucketize, type_promotion_kind=None)
def bucketize(
    input: TensorBox,
    boundaries: TensorBox,
    *,
    out_int32: bool = False,
    right: bool = False,
):
    # 确保 boundaries 张量维数为 1
    assert len(boundaries.get_size()) == 1

    if not (
        V.graph.has_feature(input, BackendFeature.BUCKETIZE)
        and V.graph.has_feature(boundaries, BackendFeature.BUCKETIZE)
    ):
        # 如果输入或边界不支持分桶化特性，则回退到处理程序
        return fallback_handler(aten.bucketize.Tensor, add_to_fallback_set=False)(
            input, boundaries, out_int32=out_int32, right=right
        )

    # 实现 boundaries 张量到全局内存，确保 boundaries.get_name() 可用
    boundaries.realize()
    # 获取 boundaries 张量的大小（第一个维度的大小）
    boundaries_size = boundaries.get_size()[0]
    # 获取输入张量所在设备
    device = input.get_device()
    # 创建输入加载器
    input_loader = input.make_loader()

    # 决定索引数据类型，根据 out_int32 参数
    index_dtype = torch.int32 if out_int32 else torch.int64

    def inner_fn(index):
        # 加载输入张量中的值
        val = input_loader(index)
        # 使用 ops.bucketize 函数进行桶化操作，返回索引
        indices = ops.bucketize(
            val,
            boundaries.get_name(),
            boundaries_size,
            index_dtype,
            right,
        )

        return indices

    # 返回 Pointwise 对象，用于执行逐点操作
    return Pointwise.create(
        device=device,
        dtype=index_dtype,
        inner_fn=inner_fn,
        ranges=input.get_size(),
    )


def require_dense(_, *args, **kwargs):
    # 对 args 和 kwargs 中的每个元素应用 require_stride1 函数
    args, kwargs = pytree.tree_map_only(
        ir.IRNode, ir.ExternKernel.require_stride1, (args, kwargs)
    )
    return args, kwargs


def require_contiguous(_, *args, **kwargs):
    # 对 args 和 kwargs 中的每个元素应用 require_contiguous 函数
    args, kwargs = pytree.tree_map_only(
        ir.IRNode, ir.ExternKernel.require_contiguous, (args, kwargs)
    )
    return args, kwargs


def require_channels_last(_, *args, **kwargs):
    # 对 args 和 kwargs 中的每个元素应用 require_channels_last 函数
    args, kwargs = pytree.tree_map_only(
        ir.IRNode, ir.ExternKernel.require_channels_last, (args, kwargs)
    )
    return args, kwargs


def constrain_to_fx_strides(fx_node, *args, **kwargs):
    # 此函数暂未提供注释，需要进一步添加内容
    pass
    # 定义一个函数 apply_constraint，接受两个参数 arg 和 fx_arg
    def apply_constraint(arg, fx_arg):
        # 如果 arg 是 ir.IRNode 的实例
        if isinstance(arg, ir.IRNode):
            # 获取 fx_arg 的 stride（步长），并根据其值获取顺序
            stride_order = ir.get_stride_order(fx_arg.meta["val"].stride())
            # 要求 arg 符合特定的 stride 顺序，并返回结果
            return ir.ExternKernel.require_stride_order(arg, stride_order)
        # 如果 arg 不是 ir.IRNode 的实例，则直接返回 arg
        return arg

    # 创建一个包含 apply_constraint 处理结果的元组 args，对每个参数 arg 和对应的 fx_node.args 进行处理
    args = tuple(
        apply_constraint(arg, fx_arg) for arg, fx_arg in zip(args, fx_node.args)
    )

    # 创建一个包含 apply_constraint 处理结果的字典 kwargs，对每个键值对 k, v 进行处理，其中 v 来自 fx_node.kwargs
    kwargs = {k: apply_constraint(v, fx_node.kwargs[k]) for k, v in kwargs.items()}

    # 返回处理后的 args 和 kwargs
    return args, kwargs
# TODO(jansel): we should implement decomps or lowerings for these
# https://github.com/pytorch/torchdynamo/issues/327
# 定义一个回退允许列表，包含 "torchvision::roi_align"
FALLBACK_ALLOW_LIST = {
    "torchvision::roi_align",
}


def sdpa_constraint(fx_node, *args, **kwargs):
    # sdpa requires dense last dimension
    # 定义一个内部函数，用于应用约束条件到参数
    def apply_constraint(arg, fx_arg):
        # 如果参数不是 IRNode 类型，则直接返回参数本身
        if not isinstance(arg, ir.IRNode):
            return arg
        
        # 获取参数的元数据中的值和步长信息
        meta_val = fx_arg.meta["val"]
        meta_stride = meta_val.stride()

        # 获取步长的顺序
        stride_order = ir.get_stride_order(meta_stride)
        # 如果存在步长顺序并且最后一个元素不为0，则将顺序反转
        if stride_order and stride_order[-1] != 0:
            stride_order = list(reversed(range(len(arg.get_size()))))

        # 如果不是 CUDA 设备上的张量，则要求按照指定的步长顺序
        if not meta_val.is_cuda:
            return ir.ExternKernel.require_stride_order(arg, stride_order)

        # SDPA 内核对 attention_bias 需要的最小对齐要求
        # 这个值可以在 pytorch/aten/src/ATen/native/transformers/attention.cpp 的 preprocess_mask 中找到
        ALIGNMENT = 8
        
        # 断言参数是 TensorBox 类型
        assert isinstance(arg, TensorBox)
        # 如果参数的维度不是 3 或者 4，则直接返回参数本身
        if len(arg.get_size()) not in (3, 4):
            return arg

        # 定义一个函数，用于检查是否张量已经对齐
        def is_aligned_realized_tensor(x):
            aligned_strides = all(
                (V.graph.sizevars.size_hint(x.get_stride()[i]) % ALIGNMENT) == 0
                for i in range(len(x.get_stride()) - 1)
            )
            return (
                V.graph.sizevars.size_hint(x.get_stride()[-1])
            ) == 1 and aligned_strides

        try:
            # 尝试获取参数的步长信息
            arg.get_stride()
            # 如果张量已经对齐，则尝试匹配无关紧要的步长
            if is_aligned_realized_tensor(arg):
                return V.graph.try_match_insignificant_strides(
                    ir.ExternKernel.realize_input(arg), meta_stride
                )
        except AttributeError:
            pass

        # 定义一个函数，用于检查是否张量对齐
        def is_aligned(x):
            return (V.graph.sizevars.size_hint(x.get_size()[-1]) % ALIGNMENT) == 0

        # 如果参数的数据是 BaseView 类型
        if isinstance(arg.data, ir.BaseView):
            # 如果未对齐，则尝试解包视图并检查对齐情况
            if not is_aligned(arg):
                if is_aligned(arg.unwrap_view()):
                    return V.graph.try_match_insignificant_strides(
                        ir.ExternKernel.realize_input(arg), meta_stride
                    )

        # 如果以上条件都不符合，则要求按照指定的步长顺序
        return ir.ExternKernel.require_stride_order(arg, stride_order)

    # 对传入的参数应用约束函数
    args = tuple(
        apply_constraint(arg, fx_arg) for arg, fx_arg in zip(args, fx_node.args)
    )
    kwargs = {k: apply_constraint(v, fx_node.kwargs[k]) for k, v in kwargs.items()}
    return args, kwargs


# WIP
# 调用 make_fallback 函数，对指定的函数进行回退处理
make_fallback(aten._adaptive_avg_pool3d)  # @isuruf
make_fallback(aten.adaptive_max_pool3d)  # @isuruf
make_fallback(aten.fractional_max_pool3d)  # @isuruf
make_fallback(aten.max_pool3d_with_indices)  # @isuruf (can this one be implemented?)


# 1) Easy
# 调用 make_fallback 函数，对指定的函数进行回退处理，禁用警告
make_fallback(aten.uniform, warn=False)
make_fallback(aten.exponential.default, warn=False)  # (fails accuracy on test_torch.py)
make_fallback(aten._pdist_forward)  # Has decomp. Needs benchmarks
make_fallback(aten.soft_margin_loss_backward, warn=False)  # py_impl?
# 将 aten.searchsorted 函数注册为可降级函数，实现已完成（见 eager 实现）
make_fallback(aten.searchsorted)  # bucketized is implemented (see eager impl)

# 1.5) Easy or Impossible
# 将 aten._cdist_forward 函数注册为可降级函数，预计 p=2 可行
make_fallback(aten._cdist_forward)  # p=2 should be feasible
# 将 aten._cdist_backward 函数注册为可降级函数
make_fallback(aten._cdist_backward)

# 2) Medium
# 将 aten.max_unpool2d 函数注册为可降级函数
make_fallback(aten.max_unpool2d)
# 将 aten.max_unpool3d 函数注册为可降级函数
make_fallback(aten.max_unpool3d)
# 将 aten._trilinear 函数注册为可降级函数
make_fallback(aten._trilinear)

# 3) Difficult
# 将 aten.segment_reduce.default 函数注册为可降级函数
# 查看详细讨论请参见 https://dev-discuss.pytorch.org/t/pytorch-sparse-gnn-compiler-rfc/1644/19
make_fallback(aten.segment_reduce.default)
# 将 aten._segment_reduce_backward.default 函数注册为可降级函数
make_fallback(aten._segment_reduce_backward.default)

# Histogram (need to implement Histogram IR)
# 将 aten.histc 函数注册为可降级函数
make_fallback(aten.histc)
# 将 aten.histogram.bin_ct 函数注册为可降级函数
make_fallback(aten.histogram.bin_ct)
# 将 aten._histogramdd_bin_edges.default 函数注册为可降级函数
make_fallback(aten._histogramdd_bin_edges.default)
# 将 aten._histogramdd_from_bin_cts.default 函数注册为可降级函数
make_fallback(aten._histogramdd_from_bin_cts.default)

# Need templated kernel
# 将 aten.addbmm 函数注册为可降级函数
make_fallback(aten.addbmm)
# 将 aten.addmv 函数注册为可降级函数，不发出警告
make_fallback(aten.addmv, warn=False)
# 将 aten._addmm_activation 函数注册为可降级函数，不发出警告
make_fallback(aten._addmm_activation, warn=False)

# Need templated kernel. Probably impossible to write efficiently
# 将 aten.convolution_backward 函数注册为可降级函数，使用 constrain_to_fx_strides
make_fallback(aten.convolution_backward, constrain_to_fx_strides)
# 将 aten._cudnn_rnn 函数注册为可降级函数，要求 dense
make_fallback(aten._cudnn_rnn, require_dense)
# 将 aten._cudnn_rnn_backward 函数注册为可降级函数，要求 contiguous
make_fallback(aten._cudnn_rnn_backward, require_contiguous)

# Haven't checked but sound difficult / impossible
# 将 aten._embedding_bag 函数注册为可降级函数，要求 contiguous
make_fallback(aten._embedding_bag, require_contiguous)
# 将 aten._embedding_bag_forward_only 函数注册为可降级函数，要求 contiguous
make_fallback(aten._embedding_bag_forward_only, require_contiguous)
# 将 aten._embedding_bag_dense_backward 函数注册为可降级函数
make_fallback(aten._embedding_bag_dense_backward)
# 将 aten._embedding_bag_per_sample_weights_backward 函数注册为可降级函数
make_fallback(aten._embedding_bag_per_sample_weights_backward)
# 将 aten._fused_moving_avg_obs_fq_helper 函数注册为可降级函数
make_fallback(aten._fused_moving_avg_obs_fq_helper)
# 将 aten._fused_moving_avg_obs_fq_helper_functional 函数注册为可降级函数
make_fallback(aten._fused_moving_avg_obs_fq_helper_functional)

# 4) Backwards (try py_impl'ing them) when fwd is written as a decomp
# 将 aten.max_pool3d_with_indices_backward 函数注册为可降级函数
make_fallback(aten.max_pool3d_with_indices_backward)
# 将 aten._adaptive_avg_pool2d_backward 函数注册为可降级函数，要求 dense
make_fallback(aten._adaptive_avg_pool2d_backward, require_dense)
# 将 aten._adaptive_avg_pool3d_backward 函数注册为可降级函数
make_fallback(aten._adaptive_avg_pool3d_backward)
# 将 aten.adaptive_max_pool2d_backward 函数注册为可降级函数
make_fallback(aten.adaptive_max_pool2d_backward)
# 将 aten.adaptive_max_pool3d_backward 函数注册为可降级函数
make_fallback(aten.adaptive_max_pool3d_backward)
# 将 aten.fractional_max_pool2d_backward 函数注册为可降级函数
make_fallback(aten.fractional_max_pool2d_backward)
# 将 aten.fractional_max_pool3d_backward 函数注册为可降级函数
make_fallback(aten.fractional_max_pool3d_backward)
# 将 aten.replication_pad1d_backward 函数注册为可降级函数
make_fallback(aten.replication_pad1d_backward)
# 将 aten.replication_pad2d_backward 函数注册为可降级函数
make_fallback(aten.replication_pad2d_backward)
# 将 aten.upsample_linear1d_backward 函数注册为可降级函数
make_fallback(aten.upsample_linear1d_backward)
# 将 aten.upsample_bicubic2d_backward 函数注册为可降级函数，要求 contiguous
make_fallback(aten.upsample_bicubic2d_backward, require_contiguous)
# 将 aten.upsample_trilinear3d_backward 函数注册为可降级函数
make_fallback(aten.upsample_trilinear3d_backward)
# 将 aten.grid_sampler_2d_backward 函数注册为可降级函数，要求 dense
make_fallback(aten.grid_sampler_2d_backward, require_dense)
# 将 aten._pdist_backward 函数注册为可降级函数

# 5) Impossible (missing triton/CPU features)

# Sorting / Sorting-like
# 将 aten.sort 函数注册为可降级函数
make_fallback(aten.sort)
# 将 aten.sort.stable 函数注册为可降级函数
make_fallback(aten.sort.stable)
# 将 aten.kthvalue 函数注册为可降级函数
make_fallback(aten.kthvalue)
# 将 aten.topk 函数注册为可降级函数
make_fallback(aten.topk)
# 将 aten.mode 函数注册为可降级函数
make_fallback(aten.mode)
# 将 aten.median 函数注册为可降级函数
make_fallback(aten.median)
# 将 aten.nanmedian 函数注册为可降级函数
make_fallback(aten.nanmedian)
# 将 aten.randperm 函数注册为可降级函数
make_fallback(aten.randperm)
# 查看 https://github.com/pytorch/pytorch/pull/121354
# 将 aten.resize_ 函数注册为可降级函数
make_fallback(aten.resize_)
# 将 aten.resize_as_ 函数注册为可降级函数

# Linalg
# 将 aten._linalg_det 函数注册为可降级函数
make_fallback(aten._linalg_det)
# 将 aten.linalg_householder_product 函数注册为可降级函数
make_fallback(aten.linalg_householder_product)
# 将 aten.linalg_inv_ex 函数注册为可降级函数
make_fallback(aten.linalg_inv_ex)
# 将 aten.linalg_ldl_factor_ex 函数注册为可降级函数
make_fallback(aten.linalg_ldl_factor_ex)
# 将 aten.linalg_ldl_solve 函数注册为可降级函数
make_fallback(aten.linalg_ldl_solve)
# 调用 make_fallback 函数，为 aten.linalg_lu 函数生成降级策略
make_fallback(aten.linalg_lu)

# 调用 make_fallback 函数，为 aten.linalg_lu_factor_ex 函数生成降级策略
make_fallback(aten.linalg_lu_factor_ex)

# 调用 make_fallback 函数，为 aten.linalg_lu_solve 函数生成降级策略
make_fallback(aten.linalg_lu_solve)

# 调用 make_fallback 函数，为 aten.linalg_matrix_exp 函数生成降级策略
make_fallback(aten.linalg_matrix_exp)

# 调用 make_fallback 函数，为 aten.linalg_qr 函数生成降级策略
make_fallback(aten.linalg_qr)

# 调用 make_fallback 函数，为 aten._linalg_slogdet 函数生成降级策略
make_fallback(aten._linalg_slogdet)

# 调用 make_fallback 函数，为 aten._linalg_solve_ex 函数生成降级策略
make_fallback(aten._linalg_solve_ex)

# 调用 make_fallback 函数，为 aten.linalg_solve_triangular 函数生成降级策略
make_fallback(aten.linalg_solve_triangular)

# 调用 make_fallback 函数，为 aten._linalg_svd 函数生成降级策略
make_fallback(aten._linalg_svd)

# 调用 make_fallback 函数，为 aten.lu_unpack 函数生成降级策略
make_fallback(aten.lu_unpack)

# 调用 make_fallback 函数，为 aten.ormqr 函数生成降级策略
make_fallback(aten.ormqr)

# 调用 make_fallback 函数，为 aten._linalg_check_errors 函数生成降级策略
make_fallback(aten._linalg_check_errors)

# 调用 make_fallback 函数，为 aten.linalg_pinv.atol_rtol_tensor 函数生成降级策略
make_fallback(aten.linalg_pinv.atol_rtol_tensor)

# 调用 make_fallback 函数，为 aten._linalg_eigh 函数生成降级策略
make_fallback(aten._linalg_eigh)

# 调用 make_fallback 函数，为 aten.triangular_solve 函数生成降级策略
make_fallback(aten.triangular_solve)

# 调用 make_fallback 函数，为 aten.linalg_cholesky_ex 函数生成降级策略
make_fallback(aten.linalg_cholesky_ex)

# 调用 make_fallback 函数，为 aten.cholesky_inverse 函数生成降级策略
make_fallback(aten.cholesky_inverse)

# 调用 make_fallback 函数，为 aten.cholesky_solve 函数生成降级策略
make_fallback(aten.cholesky_solve)

# 调用 make_fallback 函数，为 aten.geqrf 函数生成降级策略
make_fallback(aten.geqrf)

# 调用 make_fallback 函数，为 aten._fft_r2c 函数生成降级策略，需要 complex 类型支持
make_fallback(aten._fft_r2c)  # needs complex as well

# 调用 make_fallback 函数，为 aten.nonzero.default 函数生成降级策略，具体数据依赖
make_fallback(aten.nonzero.default)

# 调用 make_fallback 函数，为 aten.gcd.default 函数生成降级策略，不生成警告
make_fallback(aten.gcd.default, warn=False)

# 调用 make_fallback 函数，为 aten._thnn_fused_lstm_cell 函数生成降级策略，需要 dense 输入
make_fallback(aten._thnn_fused_lstm_cell, require_dense)

# 调用 make_fallback 函数，为 torch._prims.rng_prims.run_and_save_rng_state 函数生成降级策略
make_fallback(torch._prims.rng_prims.run_and_save_rng_state)

# 调用 make_fallback 函数，为 torch._prims.rng_prims.run_with_rng_state 函数生成降级策略
make_fallback(torch._prims.rng_prims.run_with_rng_state)

# 调用 make_fallback 函数，为 aten.masked_scatter 函数生成降级策略，实现部分或者未实现，CUDA 已实现但缺少 CPU 实现
make_fallback(aten.masked_scatter)

# 调用 make_fallback 函数，为 aten.masked_scatter_backward 函数生成降级策略，实现部分或者未实现，CUDA 已实现但缺少 CPU 实现
make_fallback(aten.masked_scatter_backward)

# 调用 make_fallback 函数，为 aten.view_as_complex 函数生成降级策略，要求连续内存
make_fallback(aten.view_as_complex, require_contiguous)

# 调用 make_fallback 函数，为 aten.angle 函数生成降级策略，需要 complex 类型支持
make_fallback(aten.angle)  # needs complex

# 调用 make_fallback 函数，为 aten._efficientzerotensor 函数生成降级策略
make_fallback(aten._efficientzerotensor)

# 调用 make_fallback 函数，为 aten._sparse_coo_tensor_with_dims_and_tensors 函数生成降级策略，需要 Sparse 支持
make_fallback(aten._sparse_coo_tensor_with_dims_and_tensors)

# 调用 make_fallback 函数，为 aten.to_sparse 函数生成降级策略
make_fallback(aten.to_sparse)

# 调用 make_fallback 函数，为 aten._to_sparse 函数生成降级策略
make_fallback(aten._to_sparse)

# 调用 make_fallback 函数，为 aten.zeros.names 函数生成降级策略，需要 dimname 支持
make_fallback(aten.zeros.names)

# 调用 make_fallback 函数，为 aten._scaled_dot_product_efficient_attention.default 函数生成降级策略，使用 sdpa_constraint
make_fallback(
    aten._scaled_dot_product_efficient_attention.default,
    sdpa_constraint,
    warn=False,
)

# 调用 make_fallback 函数，为 aten._scaled_dot_product_efficient_attention_backward.default 函数生成降级策略，使用 sdpa_constraint
make_fallback(
    aten._scaled_dot_product_efficient_attention_backward.default,
    sdpa_constraint,
    warn=False,
)

# 调用 make_fallback 函数，为 aten._scaled_dot_product_flash_attention.default 函数生成降级策略，使用 sdpa_constraint
make_fallback(
    aten._scaled_dot_product_flash_attention.default,
    sdpa_constraint,
    warn=False,
)

# 调用 make_fallback 函数，为 aten._scaled_dot_product_flash_attention_backward.default 函数生成降级策略，使用 sdpa_constraint
make_fallback(
    aten._scaled_dot_product_flash_attention_backward.default,
    sdpa_constraint,
    warn=False,
)

# 调用 make_fallback 函数，为 aten._scaled_dot_product_flash_attention_for_cpu.default 函数生成降级策略，使用 sdpa_constraint
make_fallback(
    aten._scaled_dot_product_flash_attention_for_cpu.default,
    sdpa_constraint,
    warn=False,
)

# 调用 make_fallback 函数，为 aten._scaled_dot_product_flash_attention_for_cpu_backward.default 函数生成降级策略，使用 sdpa_constraint
make_fallback(
    aten._scaled_dot_product_flash_attention_for_cpu_backward.default,
    sdpa_constraint,
    warn=False,
)

# 调用 make_fallback 函数，为 aten._flash_attention_forward.default 函数生成降级策略，使用 sdpa_constraint
make_fallback(aten._flash_attention_forward.default, sdpa_constraint)

# 调用 make_fallback 函数，为 aten._flash_attention_backward.default 函数生成降级策略，使用 sdpa_constraint
make_fallback(aten._flash_attention_backward.default, sdpa_constraint)

# 调用 make_fallback 函数，为 aten._efficient_attention_forward.default 函数生成降级策略，使用 sdpa_constraint
make_fallback(aten._efficient_attention_forward.default, sdpa_constraint)

# 调用 make_fallback 函数，为 aten._efficient_attention_backward
    # 检查当前对象和源对象的设备是否相同，如果不同则将 x 移动到当前对象的设备上
    if self.get_device() != src.get_device():
        x = to_device(x, self.get_device())

    # 检查当前对象和源对象的数据类型是否相同，如果不同则将 x 转换为当前对象的数据类型
    if self.get_dtype() != src.get_dtype():
        x = to_dtype(x, self.get_dtype())

    # 检查当前对象和源对象的大小是否相同，如果不同则扩展 x 至当前对象的大小，并返回其克隆
    if self.get_size() != src.get_size():
        # 使用 expand 函数将 x 扩展到当前对象的大小
        out = expand(x, self.get_size())
        # 克隆扩展后的 out 并返回
        return clone(out)

    # 如果大小相同，直接克隆 x 并返回
    return clone(x)
@register_lowering(aten.clone)
# 注册一个 lower 函数，用于克隆 aten.clone 操作
def clone(x, *, memory_format=None):
    # 创建一个 Pointwise 对象，用于执行克隆操作
    return Pointwise.create(
        device=x.get_device(),  # 获取张量 x 的设备信息
        dtype=x.get_dtype(),  # 获取张量 x 的数据类型信息
        inner_fn=x.make_loader(),  # 使用张量 x 的加载器作为内部函数
        ranges=list(x.get_size()),  # 获取张量 x 的维度大小，并转换为列表
    )


def clone_preserve_reinterpret_view(x):
    reinterpret_view_layouts = []
    # 检查 x 是否为 TensorBox 类型且其数据为 ir.ReinterpretView 类型
    if isinstance(x, TensorBox) and isinstance(x.data, ir.ReinterpretView):
        x = x.data  # 解包 TensorBox
        # 遍历嵌套的 ir.ReinterpretView，收集布局信息
        while isinstance(x, ir.ReinterpretView):
            reinterpret_view_layouts.append(x.get_layout())  # 获取布局信息并添加到列表中
            x = x.data  # 继续解包
        x = TensorBox(x)  # 封装为 TensorBox 类型

    x = clone(x)  # 调用克隆函数

    if reinterpret_view_layouts:
        x = x.data  # 解包 TensorBox
        # 反向遍历布局信息列表，重新创建 ir.ReinterpretView
        for layout in reinterpret_view_layouts[::-1]:
            x = ir.ReinterpretView(x, layout)
        x = TensorBox(x)  # 封装为 TensorBox 类型

    return x  # 返回处理后的张量或视图


if hasattr(aten, "lift_fresh_copy"):
    register_lowering(aten.lift_fresh_copy)(clone)


@register_lowering(prims.iota)
# 注册一个 lower 函数，用于生成 prims.iota 操作
def iota(
    length,
    *,
    start,
    step,
    dtype,
    device,
    requires_grad,
):
    def fn(index):
        # 根据索引计算元素值并返回
        return ops.index_expr(step * index[0] + start, dtype=dtype)

    return Pointwise.create(
        device=decode_device(device),  # 解码设备信息
        dtype=dtype,  # 指定数据类型
        inner_fn=fn,  # 使用指定的内部函数
        ranges=[length],  # 设置操作的范围
    )


@register_lowering(aten.select_scatter, type_promotion_kind=None)
# 注册一个 lower 函数，用于处理 aten.select_scatter 操作
def select_scatter(x, src, dim: int, index: int):
    assert x.get_dtype() == src.get_dtype()  # 断言张量 x 和 src 的数据类型相同
    x_loader = x.make_loader()  # 获取张量 x 的加载器
    dim = _validate_dim(x, dim, 0)  # 验证并获取有效的维度 dim
    # 若 index 小于 0，则将其转换为张量 x 在指定维度上的大小加上 index
    if V.graph.sizevars.evaluate_expr(sympy.Lt(index, 0)):
        index = index + x.get_size()[dim]
    V.graph.sizevars.guard_leq(0, index)  # 断言 index 大于等于 0
    V.graph.sizevars.guard_lt(index, x.get_size()[dim])  # 断言 index 小于张量 x 在指定维度上的大小
    src = expand(unsqueeze(src, dim), x.get_size())  # 根据张量 x 的大小扩展 src
    src_loader = src.make_loader()  # 获取扩展后的 src 的加载器

    def inner_fn(idx):
        return ops.where(
            ops.eq(
                ops.index_expr(idx[dim], torch.int32),  # 获取指定维度上的索引
                ops.index_expr(index, torch.int32),  # 获取指定维度上的 index
            ),
            src_loader(idx),  # 若条件成立则返回扩展后的 src
            x_loader(idx),  # 否则返回张量 x
        )

    return Pointwise.create(
        device=x.get_device(),  # 获取张量 x 的设备信息
        dtype=x.get_dtype(),  # 获取张量 x 的数据类型信息
        inner_fn=inner_fn,  # 使用指定的内部函数
        ranges=list(x.get_size()),  # 获取张量 x 的维度大小并转换为列表
    )


@register_lowering(aten.slice_scatter, type_promotion_kind=None)
# 注册一个 lower 函数，用于处理 aten.slice_scatter 操作
def slice_scatter(x, src, dim=0, start=None, end=None, step=1):
    assert x.get_dtype() == src.get_dtype()  # 断言张量 x 和 src 的数据类型相同
    x_loader = x.make_loader()  # 获取张量 x 的加载器
    dim = _validate_dim(x, dim, 0)  # 验证并获取有效的维度 dim
    dim_size = x.get_size()[dim]  # 获取指定维度上的大小

    start, end = ir.SliceView.normalize_start_end(x, dim, start, end)  # 根据给定的起始和结束位置进行规范化处理

    src_size = list(x.get_size())  # 获取张量 x 的大小并转换为列表
    src_size[dim] = FloorDiv(end - start + (step - 1), step)  # 根据计算规则调整 src 的大小
    src = expand(src, src_size)  # 根据新的大小扩展 src
    src_loader = src.make_loader()  # 获取扩展后的 src 的加载器
    def inner_fn(idx):
        # 如果选择所有元素，等同于直接返回 src_loader(idx)
        if start == 0 and end == dim_size and step == 1:
            return src_loader(idx)

        # 取出索引中指定维度的值，并将其转换为 torch.int64 类型
        idx_dim = ops.index_expr(idx[dim], torch.int64)
        
        # 复制一份索引列表
        src_idx = list(idx)
        
        # 根据起始值、步长，计算在源张量中的索引位置
        src_idx[dim] = FloorDiv(idx[dim] - start, step)

        # 初始化空的掩码列表
        mask = []
        
        # 如果起始值不为0，则创建一个大于等于起始值的掩码条件
        if start != 0:
            mask.append(
                ops.ge(
                    idx_dim,
                    ops.index_expr(sympy.expand(start), torch.int64),
                )
            )
        
        # 如果结束值不等于维度大小，则创建一个小于结束值的掩码条件
        if end != dim_size:
            mask.append(
                ops.lt(
                    idx_dim,
                    ops.index_expr(sympy.expand(end), torch.int64),
                )
            )
        
        # 如果步长不为1，则创建一个步长掩码条件
        if step != 1:
            mask.append(
                ops.eq(
                    ops.index_expr(
                        ModularIndexing(idx[dim] - start, 1, step), torch.int64
                    ),
                    ops.constant(0, torch.int64),
                )
            )
        
        # 确保掩码列表不为空
        assert mask
        
        # 使用逻辑与操作符合并所有掩码条件
        mask = functools.reduce(ops.and_, mask)
        
        # 使用掩码从源张量中加载值
        src_val = ops.masked(
            mask,
            lambda: src_loader(src_idx),
            0 if is_integer_type(x) else 0.0,
        )
        
        # 根据掩码条件选择返回值
        return ops.where(
            mask,
            src_val,
            x_loader(idx),
        )

    # 创建并返回一个 Pointwise 对象，使用指定的设备、数据类型、内部函数和维度大小列表
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=list(x.get_size()),
    )
# 用于递归解包列表或元组，返回第一个非列表或元组的元素
def _unwrap(x):
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _unwrap(x[0])
    return x


# 注册一个降低操作，接受 torch.tensor 和 aten.scalar_tensor 作为参数
def tensor(data, *, dtype=None, device=None, layout=None, pin_memory=False):
    # 断言布局参数为 None 或者 torch.strided
    assert_nyi(layout in (None, torch.strided), f"layout={layout}")
    # 断言不支持 pin_memory
    assert_nyi(not pin_memory, "pin_memory")
    
    # 如果 _unwrap(data) 返回整数，则将 dtype 设置为 torch.int64，否则使用默认的数据类型
    if isinstance(_unwrap(data), int):
        dtype = dtype or torch.int64
    else:
        dtype = dtype or torch.get_default_dtype()

    # 用于存储 sympy.Expr 类型的范围
    ranges: List[sympy.Expr] = []

    # 如果 data 是 sympy.Basic 类型
    if isinstance(data, sympy.Basic):

        # 定义内部函数 inner_fn，接受索引并返回 ops.index_expr(data, dtype) 的结果
        def inner_fn(index):
            return ops.index_expr(data, dtype)

    # 如果 data 是 float 或者 int 类型
    elif isinstance(data, (float, int)):

        # 定义内部函数 inner_fn，接受索引并返回 ops.constant(data, dtype) 的结果
        def inner_fn(index):
            return ops.constant(data, dtype)

    # 如果 data 长度为 0 或者首个元素为 float 或 int 且长度小于等于 8
    elif len(data) == 0 or isinstance(data[0], (float, int)) and len(data) <= 8:
        # 内联小张量
        ranges.append(sympy.Integer(len(data)))

        # 定义内部函数 inner_fn，接受索引并返回根据索引进行二分搜索的结果
        def inner_fn(index):
            def binary_search(start, end):
                assert start < end
                if end - start == 1:
                    return ops.constant(data[start], dtype)
                mid = (end - start) // 2 + start
                return ops.where(
                    ops.lt(
                        ops.index_expr(index[0], torch.int64),
                        ops.constant(mid, torch.int64),
                    ),
                    binary_search(start, mid),
                    binary_search(mid, end),
                )

            if len(data) == 0:
                return ops.constant(0, dtype)
            return binary_search(0, len(data))

    else:
        # 返回通过 torch.tensor 创建的常数张量
        return V.graph.add_tensor_constant(
            torch.tensor(data, dtype=dtype, device=device)
        )

    # 返回通过 Pointwise.create 创建的操作对象
    return Pointwise.create(
        device=decode_device(device),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=ranges,
    )


# 注册一个降低操作，接受 torch.as_tensor 作为参数
def as_tensor(data, dtype=None, device=None):
    # 如果 data 是 TensorBox 类型
    if isinstance(data, TensorBox):
        # 如果指定了 dtype，则转换数据类型
        if dtype is not None:
            data = to_dtype(data, dtype)
        # 如果指定了 device，则转换设备
        if device is not None:
            data = to_device(data, device)
        # 返回处理后的数据
        return data
    # 否则调用 tensor 函数处理数据
    return tensor(data, dtype=dtype, device=device)


# 注册一个降低操作，接受 torch.LongTensor 作为参数
def long_tensor(data):
    # 调用 tensor 函数处理数据，指定 dtype 为 torch.int64
    return tensor(data, dtype=torch.int64)


# 注册一个降低操作，接受 aten._local_scalar_dense 作为参数
def _local_scalar_dense(data):
    from torch.fx.experimental.symbolic_shapes import resolve_unbacked_bindings

    # 这里是一个有趣的情况！大多数降低操作返回张量，因此你可以简单地返回分配的缓冲区，
    # 它将被使用（或者如果它是死代码则不会被使用）。但是 _local_scalar_dense（又名 item）返回一个整数，
    # 而不是张量，因此如果返回一个缓冲区将导致类型不匹配；我们有义务返回一个 sympy 表达式。
    # 然而，我们确实需要编码生成 .item() 调用。
    # 通过为DynamicScalar IR节点注册一个虚拟缓冲区来完成绑定，该节点负责生成.item()。这个缓冲区实际上并没有被使用（注意我们将其丢弃）；
    # 在代码生成时，这个“缓冲区”会被赋值为None。
    unbacked_bindings = resolve_unbacked_bindings(
        V.graph.sizevars.shape_env, V.graph.current_node.meta["unbacked_bindings"]
    )
    assert len(unbacked_bindings) == 1, unbacked_bindings
    # 注意：这里必须非常小心。V.graph.current_node.meta["val"]似乎也包含一个符号，你希望对其进行绑定，但实际上并非如此。
    # 特别是，如果我们稍后执行了一个延迟运行时断言，断言说u0 == s0，你实际上会从expr中看到s0！这是不好的，因为我们实际上需要生成一个
    # 断言，说u0 == s0，所以我们需要知道从哪里获取u0（这个调用）。特别是，我们必须使用unbacked_bindings，它保证有问题的原始未替换符号。
    #
    # 注意2：我们还必须非常小心的另一件事是需要非平凡细化的符号绑定，例如，当你有一个绑定点x：Sym（u0 * 4）= y.item()。在这种情况下，
    # 代码生成必须执行除法以适当地绑定u0。这通过unbacked_bindings中的keypath进行通信，我们需要保留它以便为这种情况适当地生成代码。
    binding_sym, keypath = next(iter(unbacked_bindings.items()))
    buffer = ir.DynamicScalar(binding_sym, keypath, data)
    buffer.name = V.graph.register_buffer(buffer)
    # 注意：替换后的表达式在此处直接使用是可以的，我们希望在这种情况下进行简化！
    val = V.graph.current_node.meta["val"]
    if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return val.node.expr
    else:
        return sympy.sympify(val)
@register_lowering(aten._assert_scalar)
# 注册函数_​​assert_scalar到lowering系统，处理aten._assert_scalar操作
def _assert_scalar(data, msg):
    # 注意：这些内容将在代码生成时处理
    # 不确定是否能够从deferred_runtime_asserts中获取真实的值
    # TODO: 尝试执行这个assert语句
    # assert bool(data.scalar), data
    return None


def _full(fill_value, device, dtype, size):
    # 根据fill_value、device、dtype和size创建一个tensor
    value = fill_value
    if not isinstance(fill_value, (int, float)) and hasattr(value, "value"):
        value = value.value

    if isinstance(value, (int, float)):
        # 如果value是整数或浮点数，则创建一个返回常数值的函数
        def inner_fn(index):
            return ops.constant(value, dtype)

    elif isinstance(value, sympy.Basic):
        # 如果value是sympy.Basic类型，则创建一个返回索引表达式的函数
        def inner_fn(index):
            return ops.index_expr(value, dtype)

    else:
        assert len(value.get_size()) == 0
        # 否则，假设value是一个未命名的tensor，创建一个加载器函数
        value_loader = value.make_loader()

        def inner_fn(index):
            return value_loader([])

    return Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=list(size),
    )


@register_lowering(aten.full_like, type_promotion_kind=None)
# 注册函数full_like到lowering系统，处理aten.full_like操作，不进行类型提升
def full_like(x, fill_value, **kwargs):
    # 创建一个与x形状相同的填充值为fill_value的tensor
    return create_tensor_like(tensor_constructor(fill_value))(x, **kwargs)


def tensor_constructor(fill_value):
    # 返回一个内部函数，根据size创建相应的tensor构造器
    def inner(
        *size,
        names=None,
        dtype=None,
        device=None,
        layout=None,
        pin_memory=False,
        memory_format=None,
    ):
        assert_nyi(names is None, "named tensors")
        assert_nyi(layout in (None, torch.strided), f"layout={layout}")
        assert_nyi(not pin_memory, "pin_memory")
        device = decode_device(device)
        dtype = dtype or torch.get_default_dtype()
        if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
            size = tuple(size[0])
        # 参见https://github.com/pytorch/pytorch/issues/118102
        # 在lowering时所有的size应为sympy.Symbol，而不是SymInt！
        for s in size:
            assert not isinstance(s, torch.SymInt)
        size = [sympy.expand(s) for s in size]
        return _full(fill_value, device, dtype, size)

    return inner


@register_lowering([torch.empty, aten.empty])
# 注册函数empty和aten.empty到lowering系统
def empty(
    *size,
    names=None,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    assert_nyi(names is None, "named tensors")
    device = decode_device(device)
    if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
        size = tuple(size[0])
    return empty_strided(
        size, None, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


def create_tensor_like(creation_fn):
    """
    Shim to convert X_like(...) into X(...).  For example zeros_like() into zeros().
    """

    def _constant_like(
        x, *, dtype=None, device=None, layout=None, pin_memory=False, memory_format=None
    ):
        # 创建一个与x形状相同的常数tensor
        NotImplementedError
    ):
        # 如果是固定大小的张量创建函数，则执行以下操作
        assert_nyi(not pin_memory, "pin_memory")
        # 断言不支持 pin_memory，如果 pin_memory 为 True 则抛出异常
        assert_nyi(layout in (None, torch.strided), f"layout={layout}")
        # 断言布局在允许的范围内，如果不在指定范围内则抛出异常，使用 layout 参数的值作为错误信息
        if dtype is None:
            dtype = x.get_dtype()
        else:
            # 否则，解码 dtype 参数获取数据类型
            dtype = decode_dtype(dtype)
        # 获取张量所在的设备，如果未指定设备，则使用 x 的设备
        device = device or x.get_device()
        # 获取张量的尺寸，并将其转换为列表形式
        size = list(x.get_size())
        # 返回一个根据指定参数创建常量张量的函数
        return creation_fn(
            size, dtype=dtype, device=device, layout=layout, pin_memory=pin_memory
        )

    # 返回 _constant_like 函数作为结果
    return _constant_like
# 创建一个函数 constant_like，接受一个填充值作为参数，并返回一个调用 create_tensor_like 的结果
def constant_like(fill_value):
    return create_tensor_like(tensor_constructor(fill_value))


# 将 empty_like 定义为 register_lowering(aten.empty_like) 的结果，再调用 create_tensor_like(empty) 得到的结果
empty_like = register_lowering(aten.empty_like)(create_tensor_like(empty))


# 将 ones_like 定义为调用 create_tensor_like(tensor_constructor(1)) 的结果
ones_like = create_tensor_like(tensor_constructor(1))


# 将 zeros_like 定义为调用 create_tensor_like(tensor_constructor(0)) 的结果
zeros_like = create_tensor_like(tensor_constructor(0))


# 创建一个函数 new_constant，接受一个填充值作为参数，并返回一个函数 _new_constant
def new_constant(fill_value):
    # _new_constant 函数接受多个参数，包括 x（张量）、size（张量的尺寸）、dtype、layout、device 和 pin_memory
    def _new_constant(
        x, size, *, dtype=None, layout=None, device=None, pin_memory=None
    ):
        # 断言 size 是列表或元组
        assert isinstance(size, (list, tuple))
        # 断言 pin_memory 不支持，抛出异常 "pin_memory"
        assert_nyi(not pin_memory, "pin_memory")
        # 断言 layout 是 None 或 torch.strided，抛出异常 "layout={layout}"
        assert_nyi(layout in (None, torch.strided), f"layout={layout}")
        # 解码 dtype 或使用 x 的数据类型
        dtype = decode_dtype(dtype) or x.get_dtype()
        # 使用 x 的设备或者给定的设备
        device = device or x.get_device()
        # 将 size 中的每个元素转换为 sympy.Integer 类型
        size = [sympy.Integer(s) for s in size]
        # 调用 _full 函数，返回填充值为 fill_value 的张量，设备为 device，数据类型为 dtype，尺寸为 size
        return _full(fill_value, device, dtype, size)

    return _new_constant


# 将 new_empty 函数注册为 aten.new_empty 的降级版本，接受多个参数并返回一个张量
@register_lowering(aten.new_empty)
def new_empty(x, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    # 如果未提供 dtype，则使用 x 的数据类型
    if dtype is None:
        dtype = x.get_dtype()
    # 如果未提供 device，则使用 x 的设备
    if device is None:
        device = x.get_device()
    # 调用 empty_strided 函数，返回一个空张量，使用给定的 size、dtype、layout、device 和 pin_memory 参数
    return empty_strided(
        size, None, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# 将 empty_strided 函数注册为 aten.empty_strided 的降级版本，创建一个空张量
@register_lowering(aten.empty_strided)
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    # 断言 size 是列表或元组
    assert isinstance(size, (list, tuple))
    # 断言 stride 是列表、元组或 None
    assert isinstance(stride, (list, tuple, type(None)))
    # 断言 pin_memory 不支持，抛出异常 "pin_memory"
    assert_nyi(not pin_memory, "pin_memory")
    # 断言 layout 是 None 或 torch.strided，抛出异常 "layout={layout}"
    assert_nyi(layout in (None, torch.strided), f"layout={layout}")
    # 解码 dtype 或使用默认的 torch 数据类型
    dtype = decode_dtype(dtype) or torch.get_default_dtype()
    # 使用 torch.tensor(0.0).device 获取默认设备
    device = device or torch.tensor(0.0).device
    # 创建一个填充值为 0 的张量，设备为 device，数据类型为 dtype，尺寸为 size
    pointwise = _full(fill_value=0, device=device, dtype=dtype, size=size)
    # 实现 pointwise 张量的初始化
    pointwise.realize()
    # 获取 pointwise 张量的数据缓冲区
    buffer = pointwise.data.data
    # 明确设置缓冲区数据的范围为 size 中各维度的零
    buffer.data.ranges = [0] * len(size)
    # 断言 buffer 是 ir.ComputedBuffer 类的实例
    assert isinstance(buffer, ir.ComputedBuffer)
    # 将 size 中的每个元素扩展为 sympy 表达式
    size = [sympy.expand(s) for s in size]
    # 如果提供了 stride，则将每个元素扩展为 sympy 表达式，否则使用 ir.FlexibleLayout.contiguous_strides 生成连续的步长
    stride = (
        [sympy.expand(s) for s in stride]
        if stride
        else ir.FlexibleLayout.contiguous_strides(size)
    )
    # 将 buffer 的布局设置为固定布局，包括设备、数据类型、尺寸和步长信息
    buffer.layout = ir.FixedLayout(
        device=device,
        dtype=dtype,
        size=size,
        stride=stride,
    )
    # 返回 pointwise 张量
    return pointwise


# 将 new_empty_strided 函数注册为 aten.new_empty_strided 的降级版本，创建一个空张量
@register_lowering(aten.new_empty_strided)
def new_empty_strided(
    x, size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    # 如果未提供 dtype，则使用 x 的数据类型
    if dtype is None:
        dtype = x.get_dtype()
    # 如果未提供 device，则使用 x 的设备
    if device is None:
        device = x.get_device()
    # 调用 empty_strided 函数，返回一个空张量，使用给定的 size、stride、dtype、layout、device 和 pin_memory 参数
    return empty_strided(
        size, stride, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# 将 copy_strided 函数注册为 prims.copy_strided.default 的降级版本，复制张量并重新排序步长
@register_lowering(prims.copy_strided.default)
def copy_strided(x, stride):
    # 将 stride 中的每个元素视为大小提示，并排序返回排列顺序
    stride = [V.graph.sizevars.size_hint(s) for s in stride]
    # 根据步长的大小提示对 x 进行排序并返回结果
    stride_order = sorted(range(len(stride)), key=stride.__getitem__)
    return ir.ExternKernel.require_stride_order(x, stride_order)


# 将 full 函数注册为 torch.full 和 aten.full 的降级版本，返回一个填充值为 fill_value 的张量
@register_lowering([torch.full, aten.full])
def full(size, fill_value, **kwargs):
    # 确保关键字参数中包含 "dtype" 键，并且其值不为 None，否则抛出异常信息
    assert kwargs.get("dtype") is not None, "dtype should be handled by decomposition"
    # 使用给定的填充值创建张量对象，填充值由 tensor_constructor 提供，传入 size 和其他关键字参数 kwargs
    return tensor_constructor(fill_value)(size, **kwargs)
@register_lowering(aten.gather, type_promotion_kind=None)
# 定义一个降低操作的注册函数，针对 torch 中的 gather 操作
def gather(x, dim, index, sparse_grad=False):
    # sparse_grad 对前向计算无影响，而由 AOT Autograd 处理后向追踪
    assert isinstance(x, TensorBox)
    # 断言 x 是 TensorBox 类型
    if index.get_numel() == 0:
        # 处理空索引的情况，返回一个与 x 相同形状的空数组
        return new_empty(x, index.get_size())

    assert index.get_dtype() == torch.int64
    # 断言索引的数据类型为 torch.int64
    size = x.get_size()
    offset = len(size) == 0
    # 检查 x 的维度是否为零维
    dim = _validate_dim(x, dim, offset)
    # 验证维度是否有效，并返回调整后的维度值

    if offset:
        x = expand(x, [1])
        size = [1]
    # 如果是零维，扩展 x 的维度为 [1]，并更新 size 为 [1]

    x_loader = x.make_loader()
    index_loader = index.make_loader()
    # 创建 x 和 index 的数据加载器

    def fn(idx):
        idx = list(idx)
        # 将 idx 转换为列表形式
        gather_idx = ops.indirect_indexing(index_loader(idx), size[dim])
        # 间接索引获取 gather_idx
        if len(idx) == 0:
            idx = [gather_idx]
        else:
            idx[dim] = gather_idx
        # 更新 idx 中的 dim 维度为 gather_idx
        return x_loader(idx)
        # 返回 x_loader 加载的数据

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=index.get_size(),
    )
    # 创建 Pointwise 对象，用于执行指定设备上的 fn 函数


@register_lowering(aten.embedding, type_promotion_kind=None)
# 定义一个降低操作的注册函数，针对 torch 中的 embedding 操作
def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    assert not sparse
    # 确保 sparse 参数为 False
    assert isinstance(weight, TensorBox)
    # 断言 weight 是 TensorBox 类型
    assert isinstance(indices, TensorBox)
    # 断言 indices 是 TensorBox 类型
    assert "int" in str(indices.get_dtype())
    # 断言 indices 的数据类型包含 "int"

    weight_loader = weight.make_loader()
    indices_loader = indices.make_loader()
    # 创建 weight 和 indices 的数据加载器
    indices_ndim = len(indices.get_size())
    weight_size = weight.get_size()
    new_size = [*indices.get_size(), *weight_size[1:]]
    # 计算新的张量大小

    def fn(idx):
        assert len(idx) == len(new_size), f"{idx} != {new_size}"
        # 确保 idx 的长度与 new_size 相等
        var_index = indices_loader(idx[:indices_ndim])
        # 使用索引加载器加载 indices 的数据
        weight_idx = [ops.indirect_indexing(var_index, weight_size[0])] + [
            *idx[indices_ndim:]
        ]
        # 使用间接索引获取 weight_idx
        return weight_loader(weight_idx)
        # 返回 weight_loader 加载的数据

    return Pointwise.create(
        device=weight.get_device(),
        dtype=weight.get_dtype(),
        inner_fn=fn,
        ranges=new_size,
    )
    # 创建 Pointwise 对象，用于执行指定设备上的 fn 函数


def check_and_broadcast_indices(indices, device):
    assert all(
        i.get_dtype() in (torch.int64, torch.int32, torch.bool, torch.uint8)
        for i in indices
        if i is not None
    ), f"indices must be int64, byte or bool. Got {[i.get_dtype() for i in indices if i is not None]}"
    # 确保所有的 indices 数据类型是 torch.int64, torch.int32, torch.bool, torch.uint8 中的一种
    if any(
        i.get_dtype() in (torch.bool, torch.uint8) for i in indices if i is not None
    ):
        raise NotImplementedError("Fallback for bool indices")
    # 如果 indices 包含 torch.bool 或 torch.uint8 类型的数据，抛出 NotImplementedError

    valid_idxs = [i for i, x in enumerate(indices) if isinstance(x, TensorBox)]
    # 获取有效的索引
    assert len(valid_idxs) > 0, "requires at least 1 non-None index"
    # 确保至少有一个非空索引
    new_indices = [None] * len(indices)
    # 创建一个与 indices 长度相同的新索引数组
    # 遍历有效索引和广播张量对
    for i, x in zip(valid_idxs, broadcast_tensors(*[indices[i] for i in valid_idxs])):
        # Eager allows indices to be CPU tensor when running on CUDA
        # FIXME: Calling to_device(x, device) should work but
        # test_advancedindex_mixed_cpu_devices still fails
        # 检查张量 x 是否与指定设备不一致，如果不一致则抛出 NotImplementedError
        if x.get_device() != device:
            raise NotImplementedError("Fallback when indices is on a different device")
        # 将 x 放入新索引数组中的正确位置
        new_indices[i] = x
    # 返回更新后的新索引数组和有效索引列表
    return new_indices, valid_idxs
# 定义一个函数，用于计算索引输出的尺寸和内部函数
def index_output_size_and_inner_fn(
    x_size,
    indices,
    tensor_indices,
    tensor_size,
    indices_loaders,
    indexed_size,
    x_loader,
    check,
):
    # 当存在非连续张量时，索引行为有所不同。在这种情况下，张量索引被移到开头。
    #
    # 假设 a = torch.arange(3 * 4 * 5 * 6 * 7).view(3, 4, 5, 6, 7)
    #      x = torch.tensor([1, 2])
    # 那么，a[:, x, :, x, :] 的形状将是 (2, 3, 5, 7)，因为由于 x, :, x 的顺序，2 将被移到最前面。
    non_consecutive_tensors = False
    for previous, current in zip(tensor_indices, tensor_indices[1:]):
        if current - previous != 1:
            non_consecutive_tensors = True

    # 初始化输出尺寸为包含 None 的索引对应的 x_size
    output_size = [x_size[i] for i, val in enumerate(indices) if val is None]
    # 将剩余的 x_size 和 tensor_indices 对应的尺寸添加到 output_size
    output_size = [*output_size, *x_size[len(output_size) + len(tensor_indices):]]

    # 获取第一个张量索引
    first_tensor_index = tensor_indices[0]
    # 如果存在非连续张量，则将 tensor_size 加到 output_size 前面
    if non_consecutive_tensors:
        output_size = tensor_size + output_size
    else:
        # 否则，按顺序将 tensor_size 插入到 output_size 中
        output_size = (
            output_size[:first_tensor_index]
            + tensor_size
            + output_size[first_tensor_index:]
        )

    # 定义内部函数 fn，用于处理索引
    def fn(idx):
        assert len(idx) == len(output_size)
        assert len(indices_loaders) == len(indexed_size)

        rank = len(tensor_size)
        new_index = []
        first_tensor_index = tensor_indices[0]
        start_offset = 0 if non_consecutive_tensors else first_tensor_index
        next_idx = 0
        for i in range(tensor_indices[-1] + 1):
            if i == start_offset:
                next_idx += rank
            if indices[i] is None:
                assert next_idx < len(idx)
                new_index.append(idx[next_idx])
                next_idx += 1
            else:
                loader = indices_loaders[i]
                assert loader is not None
                size = indexed_size[i]
                # 执行间接索引操作
                new_index.append(
                    ops.indirect_indexing(
                        loader(idx[start_offset : start_offset + rank]),
                        size,
                        check=check,
                    )
                )
        # 添加剩余的 idx 到 new_index 中
        new_index = [
            *new_index,
            *idx[next_idx:],
        ]
        # 如果 x_loader 为 None，则直接返回 new_index，否则使用 x_loader 处理 new_index 后返回
        return new_index if x_loader is None else x_loader(new_index)

    # 返回计算后的 output_size 和定义的内部函数 fn
    return output_size, fn


# 定义 index_impl 函数，用于执行索引操作
def index_impl(x, indices, check):
    # 调用 index_impl_helper 获取 output_size、inner_fn 和 _
    output_size, inner_fn, _ = index_impl_helper(x, indices, check)

    # 创建 Pointwise 对象，返回结果
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=output_size,
    )


# 定义 index_impl_helper 函数，辅助执行索引操作
def index_impl_helper(x, indices, check):
    assert isinstance(indices, (list, tuple))
    # 获取 x 的 loader
    x_loader = x.make_loader()
    # 检查和广播索引，获取 indices 和 tensor_indices
    indices, tensor_indices = check_and_broadcast_indices(indices, x.get_device())
    # 断言至少有一个有效的索引
    assert len(tensor_indices) > 0, "Must have at least one valid idx"

    # 创建 indices_loaders 列表，用于加载每个索引
    indices_loaders = [i.make_loader() if i is not None else None for i in indices]
    # 输出大小没有任何保护措施，所有的保护都在 broadcast_tensors 中设置

    # 我们可以使用第一个张量的大小，因为它们都要求相同的大小
    tensor_size = list(indices[tensor_indices[0]].get_size())

    # 获取张量 x 的大小
    x_size = x.get_size()

    # 计算索引后的尺寸，仅计算非空的索引维度
    indexed_size = [x_size[i] for i in range(len(indices)) if indices[i] is not None]

    # 如果需要检查并且索引维度中存在大小为 0 的维度但张量维度中不存在大小为 0 的维度，则抛出 IndexError
    if check and 0 in indexed_size and 0 not in tensor_size:
        raise IndexError("index is out of bounds for dimension with size 0")

    # 重新计算索引后的尺寸，包括所有索引维度
    indexed_size = [x_size[i] for i in range(len(indices))]

    # 调用函数 index_output_size_and_inner_fn 计算输出大小和内部函数
    output_size, index_inner_fn = index_output_size_and_inner_fn(
        x_size,
        indices,
        tensor_indices,
        tensor_size,
        indices_loaders,
        indexed_size,
        None,
        check=check,
    )

    # 定义内部函数 inner_fn，用于加载经过索引后的张量 x 的数据
    def inner_fn(idx):
        return x_loader(index_inner_fn(idx))

    # 返回计算得到的输出大小、内部函数和索引内部函数
    return output_size, inner_fn, index_inner_fn
# 将 aten.index 函数注册为降级函数，不进行类型提升
@register_lowering(aten.index, type_promotion_kind=None)
def index(x, indices):
    try:
        # 调用 index_impl 函数处理索引操作，要求检查实现是否可用
        return index_impl(x, indices, check=True)
    except NotImplementedError:
        # 如果未实现，则回退到 ATen 的张量索引操作
        x.realize()
        # 调用 fallback_handler 处理回退操作
        return fallback_handler(aten.index.Tensor, add_to_fallback_set=False)(
            x, indices
        )


# 将 aten._unsafe_index 函数注册为降级函数，不进行类型提升
@register_lowering(aten._unsafe_index, type_promotion_kind=None)
def _unsafe_index(x, indices):
    # 直接调用 index_impl 函数处理索引操作，跳过检查
    return index_impl(x, indices, check=False)


# 所有索引分解都基于 index、index_put 和 index_put_ 函数编写
# 不能将此降级函数视为分解，因为它会在图中引入突变，这对 Aot Autograd 很不利。
# Aot Autograd 运行死代码消除和公共子表达式消除优化，假设图是无副作用的。
# 更多细节参见 https://github.com/pytorch/torchdynamo/issues/1235 和
# https://github.com/pytorch/torchdynamo/issues/1863
@register_lowering(aten.index_put)
def index_put(x, indices, values, accumulate=False):
    # 调用 index_put_ 函数，对 x 进行克隆后进行索引赋值操作
    return index_put_(clone(x), indices, values, accumulate)


# 将 aten._unsafe_index_put 函数注册为降级函数，不进行类型提升
@register_lowering(aten._unsafe_index_put)
def _unsafe_index_put(x, indices, values, accumulate=False):
    # 调用 index_put_impl_ 函数处理索引赋值操作，跳过检查
    return index_put_impl_(clone(x), indices, values, accumulate, check=False)


# 将 index_put_as_masked_fill 函数注册为索引赋值的回退函数
def index_put_as_masked_fill(self, indices, value, accumulate):
    # 如果 value 的设备与 self 的设备不同，将 value 转移到 self 的设备上
    if value.get_device() != self.get_device():
        value = to_device(value, self.get_device())
    # 如果 accumulate 为 True，则将 value 加到 self 上
    if accumulate:
        value = add(self, value)
    # 使用 where 函数根据 indices 对 self 进行条件替换操作，并返回结果
    return mutate_to(self, where(indices[0], value, self))


# 将 index_put_fallback 函数注册为索引赋值的回退函数
def index_put_fallback(self, indices, values, accumulate):
    # 检查是否启用了确定性算法
    deterministic = torch.are_deterministic_algorithms_enabled()
    # 如果 values 是 Triton，并且需要累加或者是确定性的操作
    if is_triton(values) and (accumulate or deterministic):
        msg = (
            "index put with accumulate."
            if not deterministic
            else "deterministic index put."
        )
        # 如果存在堆栈跟踪信息，则添加到消息中
        if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
            msg = f"{msg} Found from : \n {stack_trace}"
        # 禁用 cudagraphs，设置原因为 msg
        V.graph.disable_cudagraphs_reason = msg

    # 使用 ir.IndexPutFallback 创建索引赋值回退操作，并返回 self
    ir.IndexPutFallback(V.graph.current_node.target, self, indices, values, accumulate)
    return self


# 将 aten.index_put_ 函数注册为降级函数，不进行类型提升
@register_lowering(aten.index_put_, type_promotion_kind=None)
def index_put_(self, indices, values, accumulate=False):
    # 调用 index_put_impl_ 函数处理索引赋值操作，要求检查实现是否可用
    return index_put_impl_(self, indices, values, accumulate, check=True)


# 将 inductor_prims._unsafe_index_put_ 函数注册为降级函数，不进行类型提升
@register_lowering(inductor_prims._unsafe_index_put_, type_promotion_kind=None)
def _unsafe_index_put_(self, indices, values, accumulate=False):
    # 调用 index_put_impl_ 函数处理索引赋值操作，跳过检查
    return index_put_impl_(self, indices, values, accumulate, check=False)


# 定义 index_put_impl_ 函数，用于执行索引赋值操作
def index_put_impl_(self, indices, values, accumulate, check):
    # 如果 indices 是单一布尔索引且 values 只有一个元素
    if (
        values.get_numel() == 1
        and len(indices) == 1
        and indices[0].get_dtype() in {torch.bool, torch.uint8}
        # Dispatch to masked fill for single boolean index with single value
    ):
        # 用于索引张量中的索引，支持在给定的索引处放置新值，并根据需要填充掩码
        mask = indices[0]
        # 将掩码的维度扩展到与张量相同的维度
        for _ in range(len(mask.get_size()), len(self.get_size())):
            mask = unsqueeze(mask, -1)
        # 调用索引放置函数，将新值放置在张量中，并根据掩码进行填充
        return index_put_as_masked_fill(self, [mask], values, accumulate)

    # 在 PyTorch 的确定性模式下回退处理
    if torch.are_deterministic_algorithms_enabled():
        return index_put_fallback(self, indices, values, accumulate)

    # 如果存在布尔类型索引，也会回退处理
    for index in indices:
        if index is not None and index.get_dtype() in {torch.bool, torch.uint8}:
            return index_put_fallback(self, indices, values, accumulate)

    x_size = self.get_size()
    x_ndim = len(x_size)

    # 如果需要累积并且因原子加限制而需要回退处理时
    if accumulate and needs_fallback_due_to_atomic_add_limitations(self.get_dtype()):
        # 当 self 是标量张量时
        if x_ndim == 0:
            self = view(self, [1])
        # 调用回退处理函数，并根据情况调整 self 的视图
        self = index_put_fallback(self, indices, values, accumulate)
        if x_ndim == 0:
            self = view(self, [])
        return self

    # 将值转换为与 self 张量相同的数据类型
    values = to_dtype(values, self.get_dtype())

    try:
        # 检查和广播索引，通常在 dtype 为 uint32 时才会执行到这里
        indices, tensor_indices = check_and_broadcast_indices(
            indices, self.get_device()
        )
    except NotImplementedError:
        # 如果检查和广播索引不支持，则回退处理
        return index_put_fallback(self, indices, values, accumulate)

    # 创建索引加载器列表
    indices_loaders = [i.make_loader() if i is not None else None for i in indices]

    assert isinstance(self, TensorBox)
    # 确保 self 是 TensorBox 类型，并实现其实际值

    # 当 self 是标量张量时
    if x_ndim == 0:
        self = view(self, [1])

    # 取第一个索引的尺寸作为其他索引的预期值
    tensor_size = list(indices[tensor_indices[0]].get_size())
    # 获取索引后的尺寸列表
    indexed_size = [x_size[i] for i in range(len(indices))]

    # 计算输出的预期值尺寸和内部函数
    expected_vals_size, inner_fn = index_output_size_and_inner_fn(
        x_size,
        indices,
        tensor_indices,
        tensor_size,
        indices_loaders,
        indexed_size,
        None,
        check=check,
    )

    # 将值扩展到预期的尺寸
    values = expand(values, expected_vals_size)
    # 所有的保护措施都在广播张量和扩展过程中设置

    # 创建分散对象
    scatter = ir.Scatter(
        device=self.get_device(),
        dtype=self.get_dtype(),
        inner_fn=values.make_loader(),
        ranges=expected_vals_size,  # iter_ranges,
        output_indexer=inner_fn,
        scatter_mode="atomic_add" if accumulate else None,
    )
    # 创建计算缓冲区
    buffer = ir.ComputedBuffer(
        None,
        ir.MutationLayoutSHOULDREMOVE(self),
        scatter,
    )
    buffer.name = V.graph.register_buffer(buffer)

    # 当 self 是标量张量时，重新调整视图
    if x_ndim == 0:
        self = view(self, [])
    # 返回处理后的 self
    return self
# 使用 fallback_handler 函数处理 aten._unsafe_masked_index.default 函数，设置不加入回退集合
fallback__unsafe_masked_index = fallback_handler(
    aten._unsafe_masked_index.default, add_to_fallback_set=False
)

# 使用 fallback_handler 函数处理 aten._unsafe_masked_index_put_accumulate.default 函数，设置不加入回退集合
fallback__unsafe_masked_index_put_accumulate = fallback_handler(
    aten._unsafe_masked_index_put_accumulate.default, add_to_fallback_set=False
)

# 注册 _unsafe_masked_index 函数的降级实现
@register_lowering(aten._unsafe_masked_index, type_promotion_kind=None)
def _unsafe_masked_index(self, mask, indices, fill):
    # 获取索引操作的范围、加载器和不安全索引函数
    ranges, _, _unsafe_index_fn = index_impl_helper(self, indices, check=False)
    # 创建 mask 的加载器和 self 的加载器
    mask_loader = mask.make_loader()
    self_loader = self.make_loader()

    def inner_fn(idx):
        # 如果 mask 的数据类型不是 torch.bool，则将其转换为 torch.bool 类型
        if mask.dtype != torch.bool:
            mask_val = ops.to_dtype(mask_loader(idx), torch.bool)
        else:
            mask_val = mask_loader(idx)
        # 使用 ops.masked 函数进行掩码操作，填充值为 fill
        return ops.masked(mask_val, lambda: self_loader(_unsafe_index_fn(idx)), fill)

    # 创建并返回 Pointwise 对象，指定设备、数据类型、内部函数 inner_fn 和范围
    return Pointwise.create(
        device=self.get_device(),
        dtype=self.get_dtype(),
        inner_fn=inner_fn,
        ranges=ranges,
    )


# 注册 _unsafe_masked_index_put_accumulate 函数的降级实现
@register_lowering(aten._unsafe_masked_index_put_accumulate, type_promotion_kind=None)
def _unsafe_masked_index_put_accumulate(x, mask, indices, values):
    # 根据 mask 创建 masked_value
    masked_value = where(mask, values, 0)
    # 获取 x 的形状
    shape = x.get_size()
    # 对 indices 进行限制范围处理，并生成 clamped_indices
    clamped_indices = [
        clamp(indices[i], -shape[i], shape[i] - 1) if indices[i] else None
        for i in range(len(indices))
    ]
    # TODO: 使用 masked store 进行存储。目前仅 triton 支持 masked store，cpp 后端不支持
    # 调用 _unsafe_index_put 函数，传入 x、clamped_indices、masked_value 和 accumulate=True
    return _unsafe_index_put(x, clamped_indices, masked_value, accumulate=True)


# 定义 clamp 函数的降级实现
@make_pointwise
def clamp(a, min, max):
    # 返回 a、min 和 max 中的最大值和最小值
    return ops.maximum(min, ops.minimum(max, a))


# 注册 aten.as_strided_scatter 函数的降级实现
@register_lowering(aten.as_strided_scatter, type_promotion_kind=None)
def as_strided_scatter(self, src, size, stride, storage_offset=None):
    # 克隆 self，并将其存储为 output
    output = clone(self)
    # 使用 as_strided 函数创建 output_view
    output_view = as_strided(output, size, stride, storage_offset)
    # 将 src 复制到 output_view
    copy_(output_view, src)
    # 返回 output
    return output


# 注册 aten.scatter 函数的降级实现
@register_lowering(aten.scatter, type_promotion_kind=None)
def scatter(x, dim: int, index, src, **kwargs):
    # 克隆 x，并调用 scatter_ 函数
    return scatter_(clone(x), dim, index, src, **kwargs)


# 定义 scatter_fallback 函数，用于处理 scatter 函数的回退操作
def scatter_fallback(
    op_overload: torch._ops.OpOverload,
    self,
    dim: int,
    index,
    src,
    *,
    reduce: Optional[str] = None,
    include_self: bool = True,
):
    # 判断是否使用 scatter 回退
    src_is_tensor = isinstance(src, TensorBox)
    if use_scatter_fallback(
        op_overload,
        reduce,
        self.get_dtype(),
        src.get_dtype() if src_is_tensor else type(src),
        src.get_device().type if src_is_tensor else "not impl",
        src_is_tensor,
    ):
        # 创建 ScatterFallback 实例
        ir.ScatterFallback(
            op_overload,
            self,
            dim,
            index,
            src,
            reduce=reduce,
            include_self=include_self,
        )
        # 返回 self
        return self

    # 若不使用 scatter 回退，则返回 None
    return None


# 注册 aten.scatter_ 函数的降级实现
@register_lowering(aten.scatter_, type_promotion_kind=None)
def scatter_(self, dim: int, index, src, *, reduce: Optional[str] = None):
    # 断言 reduce 的取值范围为 {None, "add", "multiply"}
    assert reduce in {None, "add", "multiply"}
    # 如果 reduce 参数为 None，则尝试获取当前节点目标对象的 _overloadname 属性对应的方法
    op_overload = getattr(aten.scatter_, V.graph.current_node.target._overloadname)  # type: ignore[union-attr]
    
    # 调用 scatter_fallback 函数，尝试使用 op_overload 方法进行散步操作的备用方案
    fallback_result = scatter_fallback(
        op_overload, self, dim, index, src, reduce=reduce
    )
    
    # 如果 scatter_fallback 返回了非空结果，则直接返回该结果
    if fallback_result is not None:
        return fallback_result

    # 如果 reduce 参数为 "add"，则将其修改为 "sum"
    if reduce == "add":
        reduce = "sum"
    # 如果 reduce 参数为 "multiply"，则将其修改为 "prod"
    elif reduce == "multiply":
        reduce = "prod"
    
    # 调用 scatter_reduce_ 函数，使用指定的 reduce 方法进行散步操作
    return scatter_reduce_(self, dim, index, src, reduce)
@register_lowering(aten.scatter_add, type_promotion_kind=None)
# 注册一个降级操作，将 scatter_add 函数降级到低级操作

def scatter_add(x, dim: int, index, src):
    # 调用 scatter_add_ 函数，对 x 进行克隆后进行 scatter_add 操作
    return scatter_add_(clone(x), dim, index, src)


@register_lowering(aten.scatter_add_, type_promotion_kind=None)
# 注册一个降级操作，将 scatter_add_ 函数降级到低级操作

def scatter_add_(x, dim: int, index, src):
    # 调用 scatter_reduce_ 函数，执行 scatter_add_ 操作，对 x 进行 scatter_reduce 操作，reduction_type 为 "sum"
    return scatter_reduce_(x, dim, index, src, "sum")


@register_lowering(aten.scatter_reduce, type_promotion_kind=None)
# 注册一个降级操作，将 scatter_reduce 函数降级到低级操作

def scatter_reduce(x, dim: int, index, src, reduction_type, **kwargs):
    # 调用 scatter_reduce_ 函数，对 x 进行克隆后进行 scatter_reduce 操作，指定 reduction_type
    return scatter_reduce_(clone(x), dim, index, src, reduction_type, **kwargs)


@register_lowering(aten.scatter_reduce_, type_promotion_kind=None)
# 注册一个降级操作，将 scatter_reduce_ 函数降级到低级操作

def scatter_reduce_(self, dim: int, index, src, reduce, *, include_self: bool = True):
    # 确保 reduce 参数在支持的范围内
    assert reduce in {None, "sum", "prod", "mean", "amax", "amin"}
    # 确保 aten.scatter_reduce_.two 是 aten.scatter_reduce_ 的唯一重载
    assert (
        len(aten.scatter_reduce_.overloads()) == 1
        and "two" in aten.scatter_reduce_.overloads()
    ), "aten.scatter_reduce_.two 不是 aten.scatter_reduce_ 的唯一重载"

    # 如果 src 是数字，则将 self 克隆为与 src 相同形状的张量
    if isinstance(src, Number):
        src = full_like(self, src)

    # 使用 scatter_fallback 函数执行 scatter 操作的后备逻辑
    fallback_result = scatter_fallback(
        aten.scatter_reduce_.two,
        self,
        dim,
        index,
        src,
        reduce=reduce,
        include_self=include_self,
    )

    # 如果存在后备结果，则直接返回
    if fallback_result:
        return fallback_result

    # 确保 self 是 TensorBox 类型
    assert isinstance(self, TensorBox)
    # 确保 index 的数据类型是 int 型
    assert "int" in str(index.get_dtype())

    # 获取 self 的维度数
    ndim = len(self.get_size())
    # 如果 self 的维度数为 0，则将其视作单元素的向量
    if ndim == 0:
        self = view(self, [1])

    # 如果 src 是 TensorBox 类型且其维度数为 0，则将其视作单元素的向量
    if isinstance(src, TensorBox) and len(src.get_size()) == 0:
        src = view(src, [1])

    # 如果 index 是 TensorBox 类型且其维度数为 0，则将其视作单元素的向量
    if isinstance(index, TensorBox) and len(index.get_size()) == 0:
        index = view(index, [1])

    # 如果 index 中的元素个数为 0，则直接返回 self
    if index.get_numel() == 0:
        return self

    # 验证 dim 的有效性，确保其在 self 的维度范围内
    dim = _validate_dim(self, dim)

    # 实现 self 的实际化操作
    self.realize()
    # 创建 index 的加载器
    index_loader = index.make_loader()
    # 如果 src 是 TensorBox 类型，则创建 src 的加载器
    src_loader = src.make_loader() if isinstance(src, TensorBox) else None

    def output_indexer(idx):
        # 函数内部捕获了 self，因此可能会有 0 维的情况
        # 获取 self 的形状
        shape = self.get_size()
        # 获取 self 的维度数
        ndim = len(shape)
        # 处理间接索引，根据 dim 维度的形状对其进行索引
        indirect_idx = list(idx)
        indirect_idx[dim] = ops.indirect_indexing(
            index_loader(idx), 1 if ndim == 0 else shape[dim]
        )
        return indirect_idx

    def fn(idx):
        # 如果 src_loader 存在，则返回其加载的数据
        if src_loader:
            return src_loader(idx)
        else:
            # 否则，src 是标量，返回其常数形式
            return ops.constant(src, self.get_dtype())

    def backend_reduce_str(reduce):
        # 根据 reduce 类型返回对应的后端缩减操作名称
        if reduce == "sum":
            return "atomic_add"
        else:
            # TODO: 需要支持更多的缩减类型
            assert reduce is None
            return None
    if not include_self:
        # 首先将相应元素置零
        zero_out = ir.Scatter(
            device=self.get_device(),
            dtype=self.get_dtype(),
            inner_fn=lambda index: ops.constant(0, self.get_dtype()),
            ranges=index.get_size(),
            output_indexer=output_indexer,
            scatter_mode=None,
        )
        # 创建计算缓冲区对象
        buffer = ir.ComputedBuffer(
            None,
            ir.MutationLayoutSHOULDREMOVE(self),
            zero_out,
        )
        # 将缓冲区对象注册到计算图中
        buffer.name = V.graph.register_buffer(buffer)

    # 如果维度为0，则执行相应的索引操作
    # self[index[i][j][k]][j][k] += src[i][j][k]  # 如果 dim == 0
    # self[i][index[i][j][k]][k] += src[i][j][k]  # 如果 dim == 1
    # self[i][j][index[i][j][k]] += src[i][j][k]  # 如果 dim == 2
    scatter = ir.Scatter(
        device=self.get_device(),
        dtype=self.get_dtype(),
        inner_fn=fn,
        ranges=index.get_size(),
        output_indexer=output_indexer,
        scatter_mode=backend_reduce_str(reduce),
    )
    # 创建计算缓冲区对象
    buffer = ir.ComputedBuffer(
        None,
        ir.MutationLayoutSHOULDREMOVE(self),
        scatter,
    )
    # 将缓冲区对象注册到计算图中
    buffer.name = V.graph.register_buffer(buffer)

    # 如果维度为0，将 self 视图重新定义为一个一维张量
    if ndim == 0:
        self = view(self, [])
    # 返回结果张量
    return self
# 定义一个函数，用于最近邻上采样，对输入进行尺寸变换
def upsample_nearestnd(
    x,  # 输入张量，要进行上采样的数据
    output_size,  # 输出的目标尺寸
    scales_x: Tuple[Optional[float], ...],  # 缩放因子的元组，用于各个维度的缩放
    n: int = 2,  # 上采样的维度数量，默认为2（二维）
    exact: bool = False,  # 是否使用精确的最近邻算法，默认为False
):
    x.realize_hint()  # 对输入张量进行实现提示，以便重用元素
    x_loader = x.make_loader()  # 创建一个数据加载器
    i_sizes = x.get_size()[-n:]  # 获取输入张量最后n维的大小
    batch = x.get_size()[:-n]  # 获取输入张量除最后n维外的所有维度大小
    i_sizes = [V.graph.sizevars.evaluate_static_shape(i) for i in i_sizes]  # 计算输入张量每个维度的静态形状

    assert len(scales_x) == n  # 断言缩放因子的数量与维度数量相同
    o_sizes = output_size  # 输出的目标尺寸

    inv_scales = [i / o for i, o in zip(i_sizes, o_sizes)]  # 计算每个维度的反向缩放比例
    for i, scale in enumerate(scales_x):
        if scale is not None:
            inv_scales[i] = 1.0 / scale  # 如果缩放因子不为None，则更新反向缩放比例

    def scale_fn(x, scale, size):
        # 最近邻上采样函数
        # 精确模式：input_index = round(scale * (output_index + 0.5) - 0.5)
        # 普通模式：input_index = floor(scale * output_index)
        x = ops.index_expr(x, torch.float32)  # 将输入索引表达为浮点数
        if exact:
            x = ops.add(x, ops.constant(0.5, torch.float32))  # 如果是精确模式，增加0.5
        x = ops.mul(x, ops.constant(scale, torch.float32))  # 乘以缩放因子
        x = ops.to_dtype(x, torch.int32)  # 转换为整数类型
        return ops.indirect_indexing(x, size, check=False)  # 执行间接索引操作，返回索引结果

    def fn(idx):
        x = idx[-n:]  # 提取最后n维的索引
        b = idx[:-n]  # 提取除最后n维外的所有索引
        return x_loader(
            [*b, *[scale_fn(i, s, size) for i, s, size in zip(x, inv_scales, i_sizes)]]
            # 返回数据加载器的结果，包括批次索引和每个维度上的上采样结果
        )

    return Pointwise.create(
        device=x.get_device(),  # 使用输入张量的设备
        dtype=x.get_dtype(),  # 使用输入张量的数据类型
        inner_fn=fn,  # 内部函数为fn，用于实现具体的上采样逻辑
        ranges=[*batch, *o_sizes],  # 范围为批次索引和输出尺寸
    )


@register_lowering(aten.upsample_nearest1d.default)
def upsample_nearest1d(x, output_size, scales: Optional[float] = None):
    return upsample_nearestnd(x, output_size, (scales,), n=1)
    # 上采样最近邻一维版本的注册函数，使用了upsample_nearestnd函数


@register_lowering(aten._upsample_nearest_exact1d.default)
def _upsample_nearest_exact1d(x, output_size, scales: Optional[float] = None):
    return upsample_nearestnd(x, output_size, (scales,), n=1, exact=True)
    # 精确最近邻一维版本的注册函数，使用了upsample_nearestnd函数，并设置了精确模式为True


@register_lowering(aten.upsample_nearest2d.default)
def upsample_nearest2d(
    x, output_size, scales_h: Optional[float] = None, scales_w: Optional[float] = None
):
    return upsample_nearestnd(x, output_size, (scales_h, scales_w), n=2)
    # 上采样最近邻二维版本的注册函数，使用了upsample_nearestnd函数


@register_lowering(aten._upsample_nearest_exact2d.default)
def _upsample_nearest_exact2d(
    x, output_size, scales_h: Optional[float] = None, scales_w: Optional[float] = None
):
    return upsample_nearestnd(x, output_size, (scales_h, scales_w), n=2, exact=True)
    # 精确最近邻二维版本的注册函数，使用了upsample_nearestnd函数，并设置了精确模式为True


@register_lowering(aten.upsample_nearest3d.default)
def upsample_nearest3d(
    x,
    output_size,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    return upsample_nearestnd(x, output_size, (scales_d, scales_h, scales_w), n=3)
    # 上采样最近邻三维版本的注册函数，使用了upsample_nearestnd函数


@register_lowering(aten._upsample_nearest_exact3d.default)
def _upsample_nearest_exact3d(
    x,
    output_size,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
):
    return upsample_nearestnd(
        x, output_size, (scales_d, scales_h, scales_w), n=3, exact=True
    )
    # 精确最近邻三维版本的注册函数，使用了upsample_nearestnd函数，并设置了精确模式为True
    # 根据传入的正则表达式模式，编译成可重复使用的正则表达式对象
    regex = re.compile(pattern)

    # 在文件对象f的每一行中，查找匹配正则表达式模式的内容
    matches = [regex.search(line) for line in f]

    # 从所有匹配项中提取出第一个分组（即正则表达式模式中第一个括号内的内容）
    groups = [match.group(1) for match in matches if match]

    # 将提取的分组内容转换为整数列表
    numbers = [int(group) for group in groups if group]
# 创建一组常量张量，使用给定的值和数据类型
def _create_constants(*args, dtype):
    return tuple(ops.constant(a, dtype) for a in args)


# 注册反向传播的降低函数，用于aten.reflection_pad1d_backward等操作符
@register_lowering(aten.reflection_pad1d_backward)
@register_lowering(aten.reflection_pad2d_backward)
@register_lowering(aten.reflection_pad3d_backward)
def _reflection_padnd_backward(grad_output, x, padding):
    # 计算维度数
    dim = len(padding) // 2

    # 计算每个维度的反向传播宽度
    dhw = [h - 1 for h in x.get_size()[-dim:]]
    
    # 创建梯度输出的加载器
    grad_loader = grad_output.make_loader()

    # 提取左填充值
    padding_left = [padding[2 * (dim - 1 - i)] for i in range(dim)]
    
    # 提取右填充值
    padding_right = [padding[2 * (dim - 1 - i) + 1] for i in range(dim)]

    # 返回一个 Pointwise 对象，用于执行某些操作
    return Pointwise.create(
        device=grad_output.get_device(),
        dtype=grad_output.get_dtype(),
        inner_fn=fn,  # 此处应该是函数 fn 的引用，代码中未提供 fn 的定义
        ranges=list(x.get_size()),
    )


# 注册逆转操作的降低函数
@register_lowering(prims.rev.default)
def rev(x, dims):
    # 注意 - dims 已经被预先规范化
    x_loader = x.make_loader()
    sizes = x.get_size()

    # 定义加载器函数，用于返回逆转后的数据
    def loader(idx):
        idx = list(idx)
        assert len(idx) == len(sizes)
        for dim in dims:
            idx[dim] = (sizes[dim] - 1) - idx[dim]

        return x_loader(idx)

    # 返回一个 Pointwise 对象，用于执行某些操作
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=loader,
        ranges=sizes,
    )


# 注册 constant_pad_nd 操作的降低函数，不进行类型提升
@register_lowering(aten.constant_pad_nd, type_promotion_kind=None)
def constant_pad_nd(x, padding, fill_value=0):
    # 断言填充数量为偶数
    assert (len(padding) % 2) == 0
    # 如果所有填充值为零，则返回输入的克隆
    if all(p == 0 for p in padding):
        return clone(x)

    sizes = x.get_size()

    # 反转填充边界
    bounds = list(reversed(list(zip(padding[::2], padding[1::2]))))
    n = len(sizes) - len(bounds)

    # 预先计算填充边界
    bounds_precomp: List[Tuple[sympy.Symbol, Any]] = []
    for l, h in bounds:
        bounds_precomp.append((V.graph.sizevars.lookup_precomputed_size(l), h))  # type: ignore[arg-type]

    output_size = list(sizes[:n])
    mask_sizes = []
    for (low, high), size in zip(bounds, sizes[n:]):
        mask_sizes.append(size)
        output_size.append(sympy.expand(size + low + high))
    assert len(output_size) == len(sizes)
    fill_value = dtype_to_type(x.get_dtype())(fill_value)

    # 定义用于创建掩码的函数
    def mask(index):
        mask = []
        for idx, (low, high), length in zip(index[n:], bounds, mask_sizes):
            if low != 0:
                mask.append(range_mask_low(idx, 0))
            if high != 0:
                mask.append(range_mask_high(idx, length))
        mask = functools.reduce(ops.and_, mask)
        return ops.masked(mask, lambda: x_loader(index), fill_value)

    # 定义偏移函数，用于应用填充
    def offset_fn(index):
        new_index = list(index[:n])
        for idx, (low, high) in zip(index[n:], bounds_precomp):
            new_index.append(idx - low)
        assert len(new_index) == len(index)
        return mask(new_index)

    x_loader = x.make_loader()
    # 返回一个 Pointwise 对象，用于执行某些操作
    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=offset_fn,
        ranges=output_size,
    )


# 定义用于生成低层次操作的函数，参数为 sympy.Expr 和 low 值
def range_mask_low(i: sympy.Expr, low: Union[sympy.Expr, int]):
    # 返回一个比较操作的结果，比较两个索引表达式的值是否满足大于或等于关系
    return ops.ge(
        # 获取第一个索引表达式的值，并将其转换为 torch.int64 类型
        ops.index_expr(i, torch.int64),
        # 获取第二个索引表达式的值，并将其转换为 torch.int64 类型，使用 sympy.Integer(low) 的结果作为值
        ops.index_expr(sympy.Integer(low), torch.int64),
    )
# 检查给定索引是否小于高限，返回布尔结果
def range_mask_high(i: sympy.Expr, high: sympy.Expr):
    return ops.lt(
        ops.index_expr(i, torch.int64),
        ops.index_expr(high, torch.int64),
    )


# 检查给定索引是否同时满足低限和高限，返回布尔结果
def range_mask(i: sympy.Expr, high: sympy.Expr, low: sympy.Expr):
    return ops.and_(
        range_mask_low(i, low),
        range_mask_high(i, high),
    )


# 根据给定的维度信息和填充值，生成一个加载器函数
def constant_boundary_condition(
    x, fill_value, padding=None, pad_fill_value=1.0, dim=None
):
    h = x.get_size()[-dim:]  # 获取张量 x 的指定维度的大小
    x_loader = x.make_loader()  # 创建加载器函数，用于加载张量 x 的数据
    padding_h = padding or [0] * dim  # 如果未提供填充信息，则默认为零填充

    def load(index):
        prefix = index[:-dim]  # 截取前缀索引
        ih = index[-dim:]  # 截取当前维度索引

        # 使用 reduce 函数对维度进行逐个遍历，生成索引范围掩码
        mask = functools.reduce(
            ops.and_,
            [range_mask(ih[i], h[i] + padding_h[i], -padding_h[i]) for i in range(dim)],
        )

        # 根据是否有填充，使用不同的加载方法进行数据加载
        return (
            ops.masked(
                mask,
                lambda: constant_boundary_condition(x, pad_fill_value, dim=dim)(
                    [*prefix, *ih]
                ),
                fill_value,
            )
            if padding
            else ops.masked(mask, lambda: x_loader([*prefix, *ih]), fill_value)
        )

    return load


# 计算池化操作后的输出大小，根据 ceil_mode 参数进行向上取整或截断操作
def pooling_size(x, i, kernel_size, stride, padding, ceil_mode):
    x_out = FloorDiv(
        x + 2 * padding[i] - (kernel_size[i] - 1) + (stride[i] - 1), stride[i]
    )

    if ceil_mode:
        x_alt = FloorDiv(
            x + 2 * padding[i] - (kernel_size[i] - 1) + 2 * (stride[i] - 1), stride[i]
        )
        if V.graph.sizevars.size_hint((x_alt - 1) * stride[i] - x - padding[i]) >= 0:
            # 滑动窗口必须从输入或左填充开始
            x_alt -= 1  # 修改 x_alt 的值以符合条件
            V.graph.sizevars.guard_leq(0, x_alt * stride[i] - x - padding[i])  # 对 x_alt 的值进行保护
        if V.graph.sizevars.size_hint(x_out - x_alt) == 0:
            # 如果 ceil_mode 实际上没有操作，则进行保护
            V.graph.sizevars.guard_equals(x_out, x_alt)
            ceil_mode = False
        else:
            x_out = x_alt  # 更新 x_out 的值为 x_alt

    return x_out, ceil_mode  # 返回计算后的输出大小及更新后的 ceil_mode 值


# 根据核大小和扩张因子判断是否应该使用 max_pool2d 的替代方案
def should_fallback_max_pool2d_with_indices(kernel_size, dilation):
    kernel_size = pad_listlike(kernel_size, 2)  # 将核大小扩展为长度为 2 的列表
    window_size = kernel_size[0] * kernel_size[1]  # 计算窗口大小
    return (window_size > 25) or any(d > 1 for d in dilation)  # 判断是否应该回退到替代方案


# 检查 max_pool2d 函数的输入参数，并对其进行必要的填充处理
def max_pool2d_checks(
    x, kernel_size, stride, padding, dilation, *, assert_fallback=None
):
    if padding == 0:
        padding = [0, 0]  # 如果未提供填充信息，则默认为零填充
    if dilation == 1:
        dilation = [1, 1]  # 如果扩张因子为 1，则使用默认值 [1, 1]
    if not stride:
        stride = kernel_size  # 如果未提供步长信息，则使用核大小作为步长

    kernel_size = pad_listlike(kernel_size, 2)  # 将核大小扩展为长度为 2 的列表
    stride = pad_listlike(stride, 2)  # 将步长扩展为长度为 2 的列表
    padding = pad_listlike(padding, 2)  # 将填充信息扩展为长度为 2 的列表
    dilation = pad_listlike(dilation, 2)  # 将扩张因子扩展为长度为 2 的列表

    assert isinstance(x, TensorBox)  # 断言 x 是 TensorBox 类型的对象
    assert len(kernel_size) == 2  # 断言核大小列表长度为 2
    assert len(stride) == 2  # 断言步长列表长度为 2
    assert len(padding) == 2  # 断言填充列表长度为 2
    assert len(dilation) == 2  # 断言扩张因子列表长度为 2
    assert len(x.get_size()) in (3, 4)  # 断言张量 x 的维度为 3 或 4

    # 根据核大小和扩张因子判断是否需要使用替代方案
    use_fallback = should_fallback_max_pool2d_with_indices(kernel_size, dilation)
    # 如果 assert_fallback 参数不为 None，则进行断言检查，确保 use_fallback 参数与 assert_fallback 参数相等
    if assert_fallback is not None:
        assert use_fallback == assert_fallback

    # 返回 kernel_size, stride, padding, dilation, use_fallback 这五个变量作为结果
    return kernel_size, stride, padding, dilation, use_fallback
# 注册一个降低函数的装饰器，用于函数 prims._low_memory_max_pool2d_with_offsets
@register_lowering(prims._low_memory_max_pool2d_with_offsets, type_promotion_kind=None)
def _low_memory_max_pool2d_with_offsets(
    x,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode=False,
):
    # 断言我们不处于回退路径上，归纳器的分解应该保证了这一点
    kernel_size, stride, padding, dilation, _ = max_pool2d_checks(
        x, kernel_size, stride, padding, dilation, assert_fallback=False
    )

    # 确保张量 x 已经实现（realized）
    x.realize_hint()
    # 获取 x 的大小
    *batch, h, w = x.get_size()

    # 计算输出的高度 h_out 和是否使用 ceil_mode 的标志 ceil_mode1
    h_out, ceil_mode1 = pooling_size(h, 0, kernel_size, stride, padding, ceil_mode)
    # 计算输出的宽度 w_out 和是否使用 ceil_mode 的标志 ceil_mode2
    w_out, ceil_mode2 = pooling_size(w, 1, kernel_size, stride, padding, ceil_mode)

    # 获取 x 的数据类型
    dtype = x.dtype
    # 根据数据类型确定最小值
    min_value = (
        False
        if dtype is torch.bool
        else (float("-inf") if dtype.is_floating_point else torch.iinfo(dtype).min)
    )

    # 创建新的大小列表，包括 batch 维度和 h_out、w_out
    new_size = list(batch) + [h_out, w_out]
    # 如果有 padding 或 ceil_mode1 或 ceil_mode2，则使用常数边界条件创建 x_loader
    if padding[0] or padding[1] or ceil_mode1 or ceil_mode2:
        x_loader = constant_boundary_condition(x, min_value, dim=2)
    else:
        x_loader = x.make_loader()

    # 定义一个函数 fn，用于计算池化操作
    def fn(idx, return_index):
        # 解包 idx，获取 batch 之外的前缀和 bh、bw
        *prefix, bh, bw = idx
        maxval = None
        maxindex = None
        # 遍历 kernel_size[0] 和 kernel_size[1] 的组合
        for h_inc, w_inc in itertools.product(
            range(kernel_size[0]), range(kernel_size[1])
        ):
            # 计算输入张量中的索引 ih 和 iw
            ih = bh * stride[0] + h_inc - padding[0]
            iw = bw * stride[1] + w_inc - padding[1]
            # 从 x_loader 中获取值 val
            val = x_loader([*prefix, ih, iw])
            # 如果需要返回索引，则更新 maxindex
            if return_index:
                index = ops.index_expr(h_inc * kernel_size[1] + w_inc, torch.int8)
                if maxindex is None:
                    maxindex = index
                else:
                    maxindex = ops.where(ops.gt(val, maxval), index, maxindex)
            # 更新 maxval 为当前最大值
            if maxval is None:
                maxval = val
            else:
                maxval = ops.maximum(val, maxval)
        # 如果需要返回索引，则返回 maxindex，否则返回 maxval
        if return_index:
            return maxindex
        else:
            return maxval

    # 创建 Pointwise 对象 out，用于保存最大值
    out = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=functools.partial(fn, return_index=False),
        ranges=new_size,
    )
    # 创建 Pointwise 对象 offsets，用于保存最大值对应的索引
    offsets = Pointwise.create(
        device=x.get_device(),
        dtype=torch.int8,
        inner_fn=functools.partial(fn, return_index=True),
        ranges=new_size,
    )
    # 返回 out 和 offsets
    return out, offsets


# 注册一个降低函数的装饰器，用于函数 prims._low_memory_max_pool2d_offsets_to_indices
@register_lowering(
    prims._low_memory_max_pool2d_offsets_to_indices, type_promotion_kind=None
)
def _low_memory_max_pool2d_offsets_to_indices(
    offsets, kernel_width, input_width, stride, padding
):
    # TODO: Generalize to other max pooling flavors, and arbitrary dim

    # 创建 offsets 的加载器 offsets_loader
    offsets_loader = offsets.make_loader()

    # 定义一个函数 increments_to_index，用于计算偏移值到索引的映射
    def increments_to_index(h_inc, w_inc, bh, bw):
        # 定义输入宽度 w_in
        w_in = ops.index_expr(input_width, torch.int64)
        # 计算基础高度和宽度 hbase、wbase
        hbase = ops.index_expr(bh * stride[0] - padding[0], torch.int64)
        wbase = ops.index_expr(bw * stride[1] - padding[1], torch.int64)
        # 计算最终的索引 ih 和 iw
        ih = hbase + h_inc
        iw = wbase + w_inc
        # 返回 ih 和 iw 计算得到的索引值
        return ih * w_in + iw
    # 定义一个函数，将索引转换为增量，返回二维索引
    def offsets_to_indices(idx):
        # 将输入的索引 idx 解构为前缀列表和 bh、bw 两个变量
        *prefix, bh, bw = idx
        # 使用 offsets_loader 函数加载给定索引的偏移量数据
        offset = offsets_loader([*prefix, bh, bw])
        # 创建一个包含 kernel_width 值的常量张量，数据类型为 torch.int32
        kw_const = ops.constant(kernel_width, torch.int32)
        # 计算垂直方向的增量 h_inc，使用整数除法
        h_inc = offset // kw_const
        # 计算水平方向的增量 w_inc，使用取余运算
        w_inc = offset - (h_inc * kw_const)
        # 调用 increments_to_index 函数，将增量转换为二维索引，返回结果
        return increments_to_index(h_inc, w_inc, bh, bw)
    
    # 使用 Pointwise 类的静态方法 create 创建一个新的对象 indices
    indices = Pointwise.create(
        # 指定设备为 offsets 张量所在的设备
        device=offsets.get_device(),
        # 指定数据类型为 torch.int64
        dtype=torch.int64,
        # 指定内部函数为 offsets_to_indices 函数，用于处理每个索引
        inner_fn=offsets_to_indices,
        # 设置 ranges 参数为 offsets 张量的大小，用于确定迭代范围
        ranges=offsets.get_size(),
    )
    # 返回创建的 indices 对象
    return indices
# 当我们不降级到低内存路径时选择的回退选项。
make_fallback(aten.max_pool2d_with_indices)

# 将 aten.max_pool2d_with_indices_backward 默认值注册为回退处理程序，
# 不将其添加到回退集合中。
fallback_max_pool2d_with_indices_backward = fallback_handler(
    aten.max_pool2d_with_indices_backward.default,
    add_to_fallback_set=False,
)

# 注册 aten.max_pool2d_with_indices_backward 的降级函数。
@register_lowering(aten.max_pool2d_with_indices_backward, type_promotion_kind=None)
def max_pool2d_with_indices_backward(
    grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    # 如果 padding 是 0，则设为 [0, 0]
    if padding == 0:
        padding = [0, 0]
    # 如果 dilation 是 1，则设为 [1, 1]
    if dilation == 1:
        dilation = [1, 1]
    # 如果 stride 为空，则设为 kernel_size
    if not stride:
        stride = kernel_size

    # 断言 x 是 TensorBox 类型
    assert isinstance(x, TensorBox)
    # 断言 kernel_size 的长度为 2
    assert len(kernel_size) == 2
    # 断言 stride 的长度为 2
    assert len(stride) == 2
    # 断言 padding 的长度为 2
    assert len(padding) == 2
    # 断言 dilation 的长度为 2
    assert len(dilation) == 2
    # 断言 x 的尺寸维度为 3 或 4
    assert len(x.get_size()) in (3, 4)

    # 确保 grad_output 已经实现了提示
    grad_output.realize_hint()
    
    try:
        # 尝试获取 grad_output 的步长
        gO_stride = grad_output.get_stride()
    except AttributeError:
        # 某些类别没有 `get_stride` 方法
        # TODO 需要更好的方式来确定输入是否为通道最后
        gO_stride = None

    if isinstance(x, TensorBox) and isinstance(x.data.data, Pointwise):  # type: ignore[attr-defined]
        # 如果 x 是 TensorBox 类型，并且 x.data.data 是 Pointwise 类型
        data = x.data.data  # type: ignore[attr-defined]
        # 创建计算缓冲区 x_buffer
        x_buffer = ir.ComputedBuffer(
            name=None,
            layout=ir.FlexibleLayout(
                device=data.get_device(),
                dtype=data.get_dtype(),
                size=data.get_size(),
            ),
            data=data,
        )
        # 决定 x_buffer 的布局
        x_buffer.decide_layout()
        # 获取 x_buffer 的步长
        x_stride = x_buffer.get_stride()
    else:
        try:
            # 尝试获取 x 的步长
            x_stride = x.get_stride()
        except AttributeError:
            x_stride = None

    # 检查是否是通道最后布局
    is_channels_last = (x_stride is not None and x_stride[1] == 1) or (
        gO_stride is not None and gO_stride[1] == 1
    )

    if any(d != 1 for d in dilation):
        # dilation 尚未实现
        return fallback_max_pool2d_with_indices_backward(
            grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
        )

    # 解包 x 和 grad_output 的尺寸
    *batch, height, width = x.get_size()
    *_, pooled_height, pooled_width = grad_output.get_size()

    # 创建 indices_loader 和 grad_loader
    indices_loader = indices.make_loader()
    grad_loader = grad_output.make_loader()
    new_size = list(x.get_size())

    # 计算 h 和 w 的窗口大小
    h_window_size = max(
        max(h // stride[0] - max(0, (h - kernel_size[0]) // stride[0]), 1)
        for h in range(kernel_size[0] * 2)
    )
    w_window_size = max(
        max(w // stride[1] - max(0, (w - kernel_size[1]) // stride[1]), 1)
        for w in range(kernel_size[1] * 2)
    )

    # 计算窗口大小
    window_size = h_window_size * w_window_size

    if window_size > 25:
        # 窗口大小过大，导致 Triton 代码难以优化，使用回退
        return fallback_max_pool2d_with_indices_backward(
            grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
        )

    # 获取 indices 的尺寸
    indices_size = indices.get_size()
    # 定义函数 fn，接受一个索引 idx，返回一个梯度值
    def fn(idx):
        # 使用 *prefix 获取除最后两个元素之外的所有元素，将最后两个元素分配给 h 和 w
        *prefix, h, w = idx
        # 计算索引测试值，根据给定的 h 和 w 计算出的索引
        index_test = ops.index_expr(h * width + w, torch.int32)
        # 将 h 和 w 增加对应的填充值
        h = h + padding[0]
        w = w + padding[1]
        # 计算池化操作的起始行索引
        phstart = ops.index_expr(
            FloorDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32
        )
        # 计算池化操作的起始列索引
        pwstart = ops.index_expr(
            FloorDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32
        )
        # 计算池化操作的结束行索引
        phend = ops.index_expr(FloorDiv(h, stride[0]) + 1, torch.int32)
        # 计算池化操作的结束列索引
        pwend = ops.index_expr(FloorDiv(w, stride[1]) + 1, torch.int32)

        # 确保起始索引不小于 0
        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))
        # 确保结束索引不超过池化层的高度和宽度
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))

        # 初始化梯度值为 None
        gradient = None
        # 遍历窗口的高度
        for ph_ in range(h_window_size):
            # 遍历窗口的宽度
            for pw_ in range(w_window_size):
                # 计算当前位置的行索引
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))
                # 计算当前位置的列索引
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))
                # 构建梯度的索引
                grad_index = [
                    *prefix,
                    ops.indirect_indexing(
                        ops.minimum(ph, ops.sub(phend, ops.constant(1, torch.int32))),
                        indices_size[-2],
                        check=False,
                    ),
                    ops.indirect_indexing(
                        ops.minimum(pw, ops.sub(pwend, ops.constant(1, torch.int32))),
                        indices_size[-1],
                        check=False,
                    ),
                ]

                # 获取实际的索引值
                index_actual = indices_loader(grad_index)
                # 获取梯度值
                grad_part = grad_loader(grad_index)
                # 检查索引是否匹配测试索引
                check = ops.eq(index_actual, index_test)

                # 如果梯度为 None，则赋值为 grad_part；否则根据 mask 进行累加
                if gradient is None:
                    # 对于第一个梯度值，不需要进行 mask 操作
                    gradient = ops.where(
                        check, grad_part, ops.constant(0.0, torch.float32)
                    )
                else:
                    # 创建 mask，用于指定是否需要累加梯度
                    mask = ops.and_(
                        ops.and_(
                            ops.lt(ph, phend),
                            ops.lt(pw, pwend),
                        ),
                        check,
                    )
                    # 根据 mask 累加梯度值
                    gradient = ops.where(mask, ops.add(gradient, grad_part), gradient)
        
        # 断言梯度值不为空
        assert gradient is not None
        # 返回计算得到的梯度值
        return gradient

    # 使用 Pointwise.create 创建一个操作对象 out
    out = Pointwise.create(
        device=grad_output.get_device(),
        dtype=grad_output.get_dtype(),
        inner_fn=fn,
        ranges=new_size,
    )
    # 如果是 channels_last 格式，则返回 channels_last 形式的 out
    if is_channels_last:
        return ir.ExternKernel.require_channels_last(out)
    else:
        # 否则返回 out
        return out
# 定义一个函数，根据输入张量的大小获取其高度和宽度，忽略前面的维度
def pad_adaptive_loader(x, pad_val=0.0):
    *_, h, w = x.get_size()  # 获取输入张量 x 的高度 h 和宽度 w
    x_loader = x.make_loader()  # 创建一个数据加载器 x_loader

    # 定义内部函数 load，用于根据指定参数加载数据块
    def load(prefix, increments, start_indices, end_indices):
        ih, iw = increments  # 获取增量参数 ih 和 iw
        h_start_index, w_start_index = start_indices  # 获取起始索引参数 h_start_index 和 w_start_index
        h_end_index, w_end_index = end_indices  # 获取结束索引参数 h_end_index 和 w_end_index

        # 创建一个布尔类型的掩码，标识加载数据时的有效范围
        mask = ops.and_(
            ops.lt(
                ops.index_expr(h_start_index + ih, torch.int64),  # 检查高度索引是否在有效范围内
                ops.index_expr(h_end_index, torch.int64),
            ),
            ops.lt(
                ops.index_expr(w_start_index + iw, torch.int64),  # 检查宽度索引是否在有效范围内
                ops.index_expr(w_end_index, torch.int64),
            ),
        )

        # 使用掩码加载数据，如果掩码为真则使用 x_loader 加载数据，否则返回 pad_val
        return ops.masked(
            mask,
            lambda: x_loader([*prefix, h_start_index + ih, w_start_index + iw]),  # 调用 x_loader 加载数据块
            pad_val,
        )

    return load  # 返回内部函数 load


# 定义一个函数，根据输入和输出的维度计算适应性池化操作的索引范围
def compute_indices_adaptive_pooling(start_index, end_index, h_in, w_in, h_out, w_out):
    # 针对高度计算起始和结束索引函数
    h_start_index = functools.partial(start_index, out_dim=h_out, inp_dim=h_in)
    h_end_index = functools.partial(end_index, out_dim=h_out, inp_dim=h_in)

    # 针对宽度计算起始和结束索引函数
    w_start_index = functools.partial(start_index, out_dim=w_out, inp_dim=w_in)
    w_end_index = functools.partial(end_index, out_dim=w_out, inp_dim=w_in)

    return h_start_index, h_end_index, w_start_index, w_end_index  # 返回计算得到的四个函数


# 定义一个适应性池化函数，根据指定的池化函数和核大小执行池化操作
def _adaptive_pooling_fn(
    start_index, end_index, kernel_maxes, in_sizes, out_sizes, pooling_fn
):
    h_in, w_in = in_sizes  # 获取输入尺寸的高度和宽度
    h_out, w_out = out_sizes  # 获取输出尺寸的高度和宽度

    # 根据输入输出尺寸计算适应性池化的起始和结束索引函数
    (
        h_start_index_fn,
        h_end_index_fn,
        w_start_index_fn,
        w_end_index_fn,
    ) = compute_indices_adaptive_pooling(
        start_index, end_index, h_in, w_in, h_out, w_out
    )

    # 定义内部函数 fn，用于执行适应性池化操作
    def fn(idx, loader):
        *prefix, bh, bw = idx  # 解包索引参数，获取 batch 高度 bh 和宽度 bw

        # 计算当前 batch 高度的起始和结束索引
        h_start_index = h_start_index_fn(bh)
        h_end_index = h_end_index_fn(bh)

        # 计算当前 batch 宽度的起始和结束索引
        w_start_index = w_start_index_fn(bw)
        w_end_index = w_end_index_fn(bw)

        result = None  # 初始化结果变量为 None
        # 遍历核的最大值范围，执行池化操作
        for ih, iw in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
            val = loader(
                prefix,
                [ih, iw],
                [h_start_index, w_start_index],
                [h_end_index, w_end_index],
            )
            if result is None:
                result = val  # 初始化结果为当前值
            else:
                result = pooling_fn(val, result)  # 使用指定的池化函数更新结果
        return result  # 返回最终池化的结果

    return fn  # 返回内部函数 fn


# 定义一个带索引的适应性池化函数，与 _adaptive_pooling_fn 类似但增加了索引参数处理
def _adaptive_pooling_fn_with_idx(
    start_index, end_index, kernel_maxes, in_sizes, out_sizes, pooling_fn
):
    h_in, w_in = in_sizes  # 获取输入尺寸的高度和宽度
    h_out, w_out = out_sizes  # 获取输出尺寸的高度和宽度

    # 根据输入输出尺寸计算适应性池化的起始和结束索引函数
    (
        h_start_index_fn,
        h_end_index_fn,
        w_start_index_fn,
        w_end_index_fn,
    ) = compute_indices_adaptive_pooling(
        start_index, end_index, h_in, w_in, h_out, w_out
    )

    # 返回计算得到的四个函数作为结果
    return h_start_index_fn, h_end_index_fn, w_start_index_fn, w_end_index_fn
    # 定义一个函数 fn，接受 idx 和 loader 两个参数
    def fn(idx, loader):
        # 使用解构赋值获取 idx 中的前缀部分和 bh, bw 两个值
        *prefix, bh, bw = idx

        # 计算垂直方向上的起始和结束索引
        h_start_index = h_start_index_fn(bh)
        h_end_index = h_end_index_fn(bh)

        # 计算水平方向上的起始和结束索引
        w_start_index = w_start_index_fn(bw)
        w_end_index = w_end_index_fn(bw)

        # 初始化最大值和对应索引
        maxval = None
        maxindex = None

        # 使用 itertools.product 遍历两个范围，生成所有可能的 ih 和 iw 组合
        for ih, iw in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
            # 调用 loader 函数，传入参数 prefix, [ih, iw], [h_start_index, w_start_index], [h_end_index, w_end_index]
            val = loader(
                prefix,
                [ih, iw],
                [h_start_index, w_start_index],
                [h_end_index, w_end_index],
            )

            # 计算当前元素的索引
            index = ops.index_expr(
                (h_start_index + ih) * w_in + w_start_index + iw, torch.int64
            )

            # 判断是否是第一个元素，如果是则直接赋值给 maxindex
            if maxindex is None:
                maxindex = index
            else:
                # 否则使用 ops.where 判断是否更新 maxindex
                maxindex = ops.where(ops.gt(val, maxval), index, maxindex)

            # 判断是否是第一个元素，如果是则直接赋值给 maxval
            if maxval is None:
                maxval = val
            else:
                # 否则使用 pooling_fn 函数计算最大值
                maxval = pooling_fn(val, maxval)

        # 返回计算得到的最大索引 maxindex
        return maxindex

    # 返回函数 fn 本身作为结果
    return fn
# 创建一个函数 `fallback_adaptive_avg_pool2d`，调用 `fallback_handler` 来处理 `aten._adaptive_avg_pool2d.default`，并设置 `add_to_fallback_set` 参数为 False
fallback_adaptive_avg_pool2d = fallback_handler(
    aten._adaptive_avg_pool2d.default, add_to_fallback_set=False
)


# 注册一个降级函数，处理 `aten._adaptive_avg_pool2d` 的自定义操作
@register_lowering(aten._adaptive_avg_pool2d)
def _adaptive_avg_pool2d(x, output_size):
    # 断言输入 `x` 是 `TensorBox` 类型
    assert isinstance(x, TensorBox)
    # 断言输出大小 `output_size` 是二维的
    assert len(output_size) == 2
    # 实现一个实现提示的方法（具体细节未明确）

    # 解包 `x` 的尺寸信息，获取批次大小和输入的高度、宽度
    *batch, h_in, w_in = x.get_size()

    # 使用 `V.graph.sizevars.evaluate_static_shape` 方法评估静态形状，得到输入高度和宽度
    h_in = V.graph.sizevars.evaluate_static_shape(h_in)
    w_in = V.graph.sizevars.evaluate_static_shape(w_in)

    # 解包输出尺寸 `output_size` 的高度和宽度
    h_out, w_out = output_size

    # 如果输入和输出的高度和宽度相同，则返回 `x` 的克隆
    if h_in == h_out and w_in == w_out:
        return clone(x)

    # 如果输出的高度或宽度为零，则创建空的张量作为输出
    if h_out == 0 or w_out == 0:
        o_size = [*batch, h_out, w_out]
        return empty(o_size, dtype=x.get_dtype(), device=x.get_device())
    
    # 如果输入的高度和宽度可以被输出的高度和宽度整除，则计算池化核大小并使用平均池化函数
    if h_in % h_out == 0 and w_in % w_out == 0:
        kernel_size = [h_in // h_out, w_in // w_out]
        return avg_pool2d(x, kernel_size)

    # 计算适应性平均池化的核最大值
    h_kernel_max = ceildiv((h_in + h_out - 1), h_out)
    w_kernel_max = ceildiv((w_in + w_out - 1), w_out)

    # 创建新的尺寸列表和数据类型
    new_size = list(batch) + [h_out, w_out]
    dtype = x.get_dtype()

    # 计算窗口大小
    window_size = h_kernel_max * w_kernel_max
    if window_size > 25:
        # 如果窗口大小大于25，使用 `fallback_adaptive_avg_pool2d` 函数进行降级处理
        return fallback_adaptive_avg_pool2d(x, output_size)

    # 定义开始索引和结束索引的函数
    def start_index(index, out_dim, inp_dim):
        return FloorDiv((index * inp_dim), out_dim)

    def end_index(index, out_dim, inp_dim):
        return FloorDiv((index + 1) * inp_dim + out_dim - 1, out_dim)

    # 使用 `_adaptive_pooling_fn` 函数创建函数 `fn_sum`，用于汇总池化的值
    fn_sum = _adaptive_pooling_fn(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[h_in, w_in],
        out_sizes=[h_out, w_out],
        pooling_fn=ops.add,
    )

    # 使用 `pad_adaptive_loader` 函数创建全为1的加载器 `ones_loader`
    ones_loader = pad_adaptive_loader(ones_like(x))

    # 定义函数 `fn`，计算 `fn_sum` 的结果除以 `ones_loader` 的结果
    def fn(idx):
        return ops.truediv(
            fn_sum(idx, pad_adaptive_loader(x)), fn_sum(idx, ones_loader)
        )

    # 创建一个 `Pointwise` 对象 `rv`，使用 `fn` 函数在指定范围内生成张量
    rv = Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    # 返回结果 `rv`
    # TODO: 是否应该强制这些被实现？
    return rv


# 创建一个函数 `fallback_adaptive_max_pool2d`，调用 `fallback_handler` 处理 `aten.adaptive_max_pool2d.default`，并设置 `add_to_fallback_set` 参数为 False
fallback_adaptive_max_pool2d = fallback_handler(
    aten.adaptive_max_pool2d.default, add_to_fallback_set=False
)


# 注册一个降级函数，处理 `aten.adaptive_max_pool2d` 的自定义操作
@register_lowering(aten.adaptive_max_pool2d)
def adaptive_max_pool2d(x, output_size):
    # 断言输入 `x` 是 `TensorBox` 类型
    assert isinstance(x, TensorBox)
    # 断言输出大小 `output_size` 是二维的
    assert len(output_size) == 2
    # 实现一个实现提示的方法（具体细节未明确）

    # 解包 `x` 的尺寸信息，获取批次大小和输入的高度、宽度
    *batch, h_in, w_in = x.get_size()

    # 使用 `V.graph.sizevars.evaluate_static_shape` 方法评估静态形状，得到输入高度和宽度
    h_in = V.graph.sizevars.evaluate_static_shape(h_in)
    w_in = V.graph.sizevars.evaluate_static_shape(w_in)

    # 解包输出尺寸 `output_size` 的高度和宽度
    h_out, w_out = output_size

    # 如果输出的高度或宽度为零，则创建空的张量作为输出
    if h_out == 0 or w_out == 0:
        o_size = [*batch, h_out, w_out]
        return empty(o_size, dtype=x.get_dtype(), device=x.get_device()), empty(
            o_size, dtype=torch.int64, device=x.get_device()
        )
    # 检查输入高度和宽度是否可以整除输出高度和宽度
    if h_in % h_out == 0 and w_in % w_out == 0:
        # 计算卷积核大小
        kernel_size = [h_in // h_out, w_in // w_out]
        # 检查是否应该使用带索引的最大池化作为回退选项
        if should_fallback_max_pool2d_with_indices(kernel_size, dilation=[1, 1]):
            # 调用带索引的最大池化函数，并忽略类型检查
            return max_pool2d_with_indices(x, kernel_size)  # type: ignore[name-defined]   # noqa: F821
        else:
            # 使用低内存模式下的带偏移量的最大池化函数
            v, offsets = _low_memory_max_pool2d_with_offsets(
                x,
                kernel_size,
                stride=kernel_size,
                padding=[0, 0],
                dilation=[1, 1],
                ceil_mode=False,
            )
            # 将偏移量转换为索引
            indices = _low_memory_max_pool2d_offsets_to_indices(
                offsets, kernel_size[1], w_in, kernel_size, padding=[0, 0]
            )
            return v, indices

    # 计算适应性最大池化的卷积核最大尺寸
    h_kernel_max = ceildiv((h_in + h_out - 1), h_out)
    w_kernel_max = ceildiv((w_in + w_out - 1), w_out)

    # 构建新的大小列表，包含批处理维度和输出的高度、宽度
    new_size = list(batch) + [h_out, w_out]
    # 获取输入张量的数据类型
    dtype = x.get_dtype()

    # 计算窗口大小
    window_size = h_kernel_max * w_kernel_max
    if window_size > 25:
        # 如果卷积核大小太大，则使用回退的自适应最大池化函数
        return fallback_adaptive_max_pool2d(x, output_size)

    # 定义计算起始索引的函数
    def start_index(index, out_dim, inp_dim):
        return FloorDiv((index * inp_dim), out_dim)

    # 定义计算结束索引的函数
    def end_index(index, out_dim, inp_dim):
        return FloorDiv((index + 1) * inp_dim + out_dim - 1, out_dim)

    # 使用适应性池化函数进行最大值池化
    inner_func_max_val = _adaptive_pooling_fn(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[h_in, w_in],
        out_sizes=[h_out, w_out],
        pooling_fn=ops.maximum,
    )

    # 使用适应性池化函数进行最大值池化，并返回索引
    inner_func_max_idx = _adaptive_pooling_fn_with_idx(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[h_in, w_in],
        out_sizes=[h_out, w_out],
        pooling_fn=ops.maximum,
    )

    # 定义内部函数，用于计算最大值池化
    def inner_fn_max_val(idx):
        return inner_func_max_val(idx, pad_adaptive_loader(x, float("-inf")))

    # 定义内部函数，用于计算最大值池化并返回索引
    def inner_fn_max_idx(idx):
        return inner_func_max_idx(idx, pad_adaptive_loader(x, float("-inf")))

    # 创建基于点的操作，计算最大值池化
    rv = Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=inner_fn_max_val,
        ranges=new_size,
    )
    # 创建基于点的操作，计算最大值池化并返回索引
    ri = Pointwise.create(
        device=x.get_device(),
        dtype=torch.int64,
        inner_fn=inner_fn_max_idx,
        ranges=new_size,
    )
    return rv, ri
# 定义一个回退函数，用于处理分数最大池化的情况
fallback_fractional_max_pool2d = fallback_handler(
    aten.fractional_max_pool2d.default, add_to_fallback_set=False
)

# 定义一个函数，用于计算分数池化的偏移量
def _fractional_pooling_offsets(samples, in_sz, out_sz, kernel_sz, dim):
    # 获取指定维度的输出大小、输入大小和核大小
    out_sz = out_sz[dim]
    in_sz = in_sz[dim]
    kernel_sz = kernel_sz[dim]
    # 计算 alpha 值
    alpha = IntTrueDiv(in_sz - kernel_sz, out_sz - 1)
    # 创建样本加载器
    samples_loader = samples.make_loader()

    # 定义加载函数
    def load(prefix, i):
        # 加载样本
        sample = samples_loader([*prefix, dim])
        i_expr = ops.index_expr(i, samples.get_dtype())
        alpha_expr = ops.index_expr(alpha, samples.get_dtype())
        # 计算序列索引
        seq_i = ops.floor((i_expr + sample) * alpha_expr) - ops.floor(
            sample * alpha_expr
        )
        seq_i = ops.to_dtype(seq_i, torch.int64)

        # 创建掩码
        mask = ops.lt(
            i_expr,
            ops.index_expr(out_sz - 1, torch.int64),
        )
        return ops.where(mask, seq_i, ops.index_expr(in_sz - kernel_sz, torch.int64))

    return load

# 注册降级函数，处理分数最大池化
@register_lowering(aten.fractional_max_pool2d)
def fractional_max_pool2d(x, kernel_size, output_size, random_samples):
    # 实现提示
    x.realize_hint()
    # 获取批次大小和输入输出大小
    *batch, inp_h, inp_w = x.get_size()
    kernel_h, kernel_w = kernel_size
    h_out, w_out = output_size

    # 如果核大小大于等于 25，则使用回退函数处理
    if kernel_h * kernel_w >= 25:
        return fallback_fractional_max_pool2d(
            x, kernel_size, output_size, random_samples
        )

    # 生成维度偏移函数
    gen_offsets_for_dim = functools.partial(
        _fractional_pooling_offsets,
        samples=random_samples,
        in_sz=[inp_h, inp_w],
        out_sz=output_size,
        kernel_sz=kernel_size,
    )

    # 获取高度和宽度的偏移函数
    h_index_fn = gen_offsets_for_dim(dim=0)
    w_index_fn = gen_offsets_for_dim(dim=1)
    x_loader = x.make_loader()

    # 定义函数，用于计算最大值和索引
    def fn(idx, return_index):
        *prefix, bh, bw = idx

        h_start_index = ops.indirect_indexing(h_index_fn(prefix, bh), inp_h)
        w_start_index = ops.indirect_indexing(w_index_fn(prefix, bw), inp_w)

        maxval = None
        maxindex = None
        for ih, iw in itertools.product(range(kernel_size[0]), range(kernel_size[1])):
            val = x_loader([*prefix, h_start_index + ih, w_start_index + iw])
            if return_index:
                index = ops.index_expr(
                    (h_start_index + ih) * inp_w + w_start_index + iw, torch.int64
                )
                if maxindex is None:
                    maxindex = index
                else:
                    maxindex = ops.where(
                        ops.or_(ops.gt(val, maxval), ops.isnan(val)), index, maxindex
                    )
            if maxval is None:
                maxval = val
            else:
                maxval = ops.maximum(val, maxval)
        if return_index:
            return maxindex
        else:
            return maxval

    # 创建新的大小列表
    new_size = list(batch) + [h_out, w_out]
    # 创建 Pointwise 对象
    rv = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=functools.partial(fn, return_index=False),
        ranges=new_size,
    )
    # 使用 Pointwise 类的 create 方法创建一个对象 ri
    ri = Pointwise.create(
        # 设备参数设定为输入张量 x 的设备
        device=x.get_device(),
        # 数据类型设定为 64 位整数
        dtype=torch.int64,
        # inner_fn 参数使用 functools.partial 函数部分应用 fn 函数，并设置 return_index=True
        inner_fn=functools.partial(fn, return_index=True),
        # ranges 参数设置为 new_size
        ranges=new_size,
    )
    # 返回变量 rv 和 ri
    return rv, ri
@register_lowering(aten.upsample_nearest2d_backward.default)
def upsample_nearest2d_backward(
    x, output_size=None, input_size=None, scales_h=None, scales_w=None
):
    # 调用 x 的 realize_hint 方法，可能用于执行一些预测或优化操作
    x.realize_hint()

    # 解构 x 的大小，获取输入图像的高度和宽度
    *batch, inp_h, inp_w = x.get_size()
    # 使用静态形状评估函数获取输入高度和宽度的静态形状
    inp_h = V.graph.sizevars.evaluate_static_shape(inp_h)
    inp_w = V.graph.sizevars.evaluate_static_shape(inp_w)

    # 解构输入大小，获取输出图像的高度和宽度
    *batch, out_h, out_w = input_size

    # 如果输入高度和宽度可以被输出高度和宽度整除，则返回相应的平均池化结果
    if inp_h % out_h == 0 and inp_w % out_w == 0:
        return avg_pool2d(x, [inp_h // out_h, inp_w // out_w], divisor_override=1)

    # 计算高度和宽度的最大核心值
    h_kernel_max = ceildiv(inp_h, out_h)
    w_kernel_max = ceildiv(inp_w, out_w)

    # 定义计算起始索引的函数
    def start_index(index, out_dim, inp_dim):
        return CeilDiv(index * inp_dim, sympy.sympify(out_dim))

    # 定义计算结束索引的函数
    def end_index(index, out_dim, inp_dim):
        return start_index((index + 1), out_dim, inp_dim)

    # 使用自适应池化函数 _adaptive_pooling_fn 计算池化的结果
    fn_sum = _adaptive_pooling_fn(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[inp_h, inp_w],
        out_sizes=[out_h, out_w],
        pooling_fn=ops.add,
    )

    # 定义 fn 函数，将池化结果应用于输入 x
    def fn(idx):
        return fn_sum(idx, pad_adaptive_loader(x))

    # 创建 Pointwise 对象 rv，用于表示 fn 的计算结果
    rv = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=list(input_size),
    )

    # 返回 Pointwise 对象 rv，用于最终的向后传播
    return rv


# 定义回退处理函数，处理平均池化操作
fallback_avg_pool2d = fallback_handler(
    aten.avg_pool2d.default, add_to_fallback_set=False
)

# 定义回退处理函数，处理三维平均池化操作
fallback_avg_pool3d = fallback_handler(
    aten.avg_pool3d.default, add_to_fallback_set=False
)


@register_lowering(aten.avg_pool2d, type_promotion_kind=None)
def avg_pool2d(
    x,
    kernel_size,
    stride=(),
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    # 调用通用的池化函数 _avg_poolnd，指定维度为2
    return _avg_poolnd(
        x,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dim=2,
    )


@register_lowering(aten.avg_pool3d, type_promotion_kind=None)
def avg_pool3d(
    x,
    kernel_size,
    stride=(),
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    # 调用通用的池化函数 _avg_poolnd，指定维度为3
    return _avg_poolnd(
        x,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dim=3,
    )


def _avg_poolnd(
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dim,
):
    # 如果未指定步长，则使用核心大小作为步长
    if not stride:
        stride = kernel_size
    # 如果未指定填充，则将填充设置为零
    if not padding:
        padding = [0] * dim
    # 将核心大小、步长和填充扩展到与维度相匹配的列表
    kernel_size = pad_listlike(kernel_size, dim)
    stride = pad_listlike(stride, dim)
    padding = pad_listlike(padding, dim)

    # 断言输入 x 是 TensorBox 类型
    assert isinstance(x, TensorBox)
    # 断言核心大小、步长和填充的长度与维度相匹配
    assert len(kernel_size) == dim
    assert len(stride) == dim
    assert len(padding) == dim
    # 断言输入 x 的大小要么是 dim+1 要么是 dim+2
    assert len(x.get_size()) in (dim + 1, dim + 2)

    # 调用 x 的 realize_hint 方法，可能用于执行一些预测或优化操作
    x.realize_hint()
    # 获取批处理大小和维度 h 的大小
    batch = x.get_size()[:-dim]
    h = x.get_size()[-dim:]
    h_out, ceil_modes = zip(
        *[
            pooling_size(h[i], i, kernel_size, stride, padding, ceil_mode)
            for i in range(dim)
        ]
    )
    # 计算每个维度的输出大小和是否使用 ceil 模式

    if any(padding) or any(ceil_modes):
        # 如果存在任何一维有 padding 或 ceil_mode，则使用常数边界条件
        x_loader = constant_boundary_condition(x, 0.0, dim=dim)
        had_padding = True
    else:
        # 否则使用默认的数据加载器
        x_loader = x.make_loader()
        had_padding = False

    new_size = list(batch) + list(h_out)
    # 计算新的输出尺寸，包括批次维度和每个维度的输出大小
    dtype = x.get_dtype()

    window_size = functools.reduce(operator.mul, kernel_size)
    if window_size > 25:
        # 如果窗口大小大于25，则使用后备的平均池化方法
        if dim == 2:
            fallback = fallback_avg_pool2d
        elif dim == 3:
            fallback = fallback_avg_pool3d
        else:
            raise ValueError(f"Unknown dim: {dim}")
        # 调用后备的平均池化方法并返回结果
        return fallback(
            x,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    def fn_sum(idx, loader):
        # 计算在给定索引下的池化总和
        prefix = idx[:-dim]
        b = idx[-dim:]
        total = None
        for ih in itertools.product(*[range(kernel_size[i]) for i in range(dim)]):
            inp = [b[i] * stride[i] + ih[i] - padding[i] for i in range(dim)]
            val = loader([*prefix, *inp])
            if total is None:
                total = val
            else:
                total = ops.add(val, total)
        return total

    if not had_padding or divisor_override:
        if divisor_override:
            scale = 1 / divisor_override
        else:
            scale = 1.0 / window_size

        def fn(idx):
            # 对于无 padding 或者指定了 divisor_override 的情况，返回乘以 scale 后的总和
            return ops.mul(fn_sum(idx, x_loader), ops.constant(scale, dtype))

    else:

        def fn(idx):
            # 对于有 padding 的情况，计算分母并返回总和除以分母后的结果
            prefix = idx[:-dim]
            bh = idx[-dim:]

            divide_factors = []
            for i in range(dim):
                hstart = bh[i] * stride[i] - padding[i]
                hend = sympy.Min(hstart + kernel_size[i], h[i] + padding[i])
                if not count_include_pad:
                    hstart = sympy.Max(hstart, 0)
                    hend = sympy.Min(hend, h[i])
                factor = ops.index_expr(hend - hstart, torch.int32)
                divide_factors.append(factor)
            divide_factor = functools.reduce(ops.mul, divide_factors)
            return ops.truediv(fn_sum(idx, x_loader), divide_factor)

    rv = Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    # 创建 Pointwise 对象并返回结果
    # TODO(jansel): 应该强制这些被实现吗？（需要进一步讨论）
    return rv
# 使用 fallback_handler 函数处理 aten.avg_pool2d_backward.default，不添加到 fallback_set 中
fallback_avg_pool2d_backward = fallback_handler(
    aten.avg_pool2d_backward.default, add_to_fallback_set=False
)

# 将 avg_pool2d_backward 函数注册为 aten.avg_pool2d_backward 的降低（lowering）实现，不进行类型提升
@register_lowering(aten.avg_pool2d_backward, type_promotion_kind=None)
def avg_pool2d_backward(
    grad_output,             # 梯度输出张量
    x,                       # 输入张量 x
    kernel_size,             # 池化核大小
    stride,                  # 步幅
    padding,                 # 填充
    ceil_mode,               # 是否使用 ceil 模式
    count_include_pad,       # 计数是否包括填充
    divisor_override=None,   # 覆盖除数（可选）
):
    assert divisor_override is None or divisor_override != 0, "divisor must be not zero"  # 断言检查除数覆盖是否为 None 或非零

    if not stride:
        stride = kernel_size  # 如果步幅未指定，则设为池化核大小

    if not padding:
        padding = [0, 0]  # 如果填充未指定，则设为 [0, 0]

    assert isinstance(grad_output, TensorBox)  # 断言检查 grad_output 是否为 TensorBox 类型
    assert isinstance(x, TensorBox)  # 断言检查 x 是否为 TensorBox 类型
    assert len(kernel_size) == 2  # 断言检查池化核大小的长度是否为 2
    assert len(stride) == 2  # 断言检查步幅的长度是否为 2
    assert len(padding) == 2  # 断言检查填充的长度是否为 2
    assert len(x.get_size()) in (3, 4)  # 断言检查输入张量 x 的维度是否为 3 或 4

    grad_output.realize_hint()  # 实现提示，确保 grad_output 已计算

    *batch, height, width = x.get_size()  # 获取输入张量 x 的大小，分别获取批次维度、高度和宽度

    h_out, ceil_mode1 = pooling_size(height, 0, kernel_size, stride, padding, ceil_mode)  # 计算池化后的高度和 ceil_mode1
    w_out, ceil_mode2 = pooling_size(width, 1, kernel_size, stride, padding, ceil_mode)  # 计算池化后的宽度和 ceil_mode2

    grad_loader = grad_output.make_loader()  # 创建 grad_output 的加载器

    had_padding = padding[0] or padding[1] or ceil_mode1 or ceil_mode2  # 检查是否存在填充

    *_, pooled_height, pooled_width = grad_output.get_size()  # 获取池化后的输出高度和宽度

    new_size = list(x.get_size())  # 获取输入张量 x 的大小并转换为列表
    dtype = x.get_dtype()  # 获取输入张量 x 的数据类型

    h_window_size = max(
        max(h // stride[0] - max(0, (h - kernel_size[0]) // stride[0]), 1) for h in range(kernel_size[0] * 2)
    )  # 计算高度窗口大小

    w_window_size = max(
        max(w // stride[1] - max(0, (w - kernel_size[1]) // stride[1]), 1) for w in range(kernel_size[1] * 2)
    )  # 计算宽度窗口大小

    window_size = h_window_size * w_window_size  # 计算窗口大小

    if window_size > 25:
        # 如果窗口大小超过 25，池化核过大，难以优化 Triton 代码，使用回退函数处理
        return fallback_avg_pool2d_backward(
            grad_output,         # 梯度输出张量
            x,                   # 输入张量 x
            kernel_size,         # 池化核大小
            stride,              # 步幅
            padding,             # 填充
            ceil_mode,           # 是否使用 ceil 模式
            count_include_pad,   # 计数是否包括填充
            divisor_override,    # 覆盖除数
        )
    def compute_pool_size_without_padding(ph, pw):
        """
        This computes the scaling factor that we will divide an element
        by when `count_include_pad=False`
        """
        # 定义池化层的步幅（高度和宽度）
        stride_h = ops.constant(stride[0], torch.int32)
        stride_w = ops.constant(stride[1], torch.int32)
        # 定义填充的尺寸（高度和宽度）
        pad_h = ops.constant(padding[0], torch.int32)
        pad_w = ops.constant(padding[1], torch.int32)
        # 定义卷积核的尺寸（高度和宽度）
        kernel_h = ops.constant(kernel_size[0], torch.int32)
        kernel_w = ops.constant(kernel_size[1], torch.int32)
        # 计算池化区域的起始和结束位置（高度）
        hstart = ops.sub(ops.mul(ph, stride_h), pad_h)
        hend = ops.minimum(
            ops.add(hstart, kernel_h),
            ops.add(ops.index_expr(height, torch.int32), pad_h),
        )
        # 计算池化区域的起始和结束位置（宽度）
        wstart = ops.sub(ops.mul(pw, stride_w), pad_w)
        wend = ops.minimum(
            ops.add(wstart, kernel_w),
            ops.add(ops.index_expr(width, torch.int32), pad_w),
        )
        # 确保池化区域的起始位置不小于0
        hstart = ops.maximum(hstart, ops.constant(0, torch.int32))
        wstart = ops.maximum(wstart, ops.constant(0, torch.int32))
        # 确保池化区域的结束位置不超过输入的尺寸
        hend = ops.minimum(hend, ops.index_expr(height, torch.int32))
        wend = ops.minimum(wend, ops.index_expr(width, torch.int32))
        # 计算池化区域的面积
        divide_factor = ops.mul(ops.sub(hend, hstart), ops.sub(wend, wstart))
        return divide_factor
    def fn(idx):
        *prefix, h, w = idx  # 解构参数 idx，将除最后两个元素外的所有元素组成列表 prefix，最后两个元素分别赋给 h 和 w
        h = h + padding[0]  # 将 h 增加 padding 列表的第一个元素
        w = w + padding[1]  # 将 w 增加 padding 列表的第二个元素
        phstart = ops.index_expr(
            FloorDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32
        )  # 计算 phstart，即 h - kernel_size[0] + stride[0] 的整除结果，转换为 torch.int32 类型
        pwstart = ops.index_expr(
            FloorDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32
        )  # 计算 pwstart，即 w - kernel_size[1] + stride[1] 的整除结果，转换为 torch.int32 类型
        phend = ops.index_expr(FloorDiv(h, stride[0]) + 1, torch.int32)  # 计算 phend，即 h 整除 stride[0] 的结果加一，转换为 torch.int32 类型
        pwend = ops.index_expr(FloorDiv(w, stride[1]) + 1, torch.int32)  # 计算 pwend，即 w 整除 stride[1] 的结果加一，转换为 torch.int32 类型

        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))  # 取 phstart 和 0 中较大的值
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))  # 取 pwstart 和 0 中较大的值
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))  # 取 phend 和 pooled_height 的较小值，转换为 torch.int32 类型
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))  # 取 pwend 和 pooled_width 的较小值，转换为 torch.int32 类型

        gradient = None  # 初始化 gradient 变量为 None
        for ph_ in range(h_window_size):  # 循环遍历 h_window_size 次，ph_ 从 0 到 h_window_size-1
            for pw_ in range(w_window_size):  # 循环遍历 w_window_size 次，pw_ 从 0 到 w_window_size-1
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))  # 计算 ph，即 phstart + ph_，转换为 torch.int32 类型
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))  # 计算 pw，即 pwstart + pw_，转换为 torch.int32 类型

                if divisor_override is not None:
                    scale = divisor_override  # 如果 divisor_override 不为 None，则 scale 等于 divisor_override
                elif count_include_pad or not had_padding:
                    scale = kernel_size[0] * kernel_size[1]  # 如果 count_include_pad 为 True 或者没有 padding，则 scale 等于 kernel_size[0] * kernel_size[1]
                else:
                    scale = compute_pool_size_without_padding(ph, pw)  # 否则调用 compute_pool_size_without_padding 函数计算 scale

                part = ops.truediv(
                    grad_loader(
                        [
                            *prefix,
                            ops.indirect_indexing(
                                ops.minimum(
                                    ph, ops.sub(phend, ops.constant(1, torch.int32))
                                ),  # 计算 ph 和 phend-1 的较小值，转换为 torch.int32 类型
                                pooled_height,
                                check=False,
                            ),
                            ops.indirect_indexing(
                                ops.minimum(
                                    pw, ops.sub(pwend, ops.constant(1, torch.int32))
                                ),  # 计算 pw 和 pwend-1 的较小值，转换为 torch.int32 类型
                                pooled_width,
                                check=False,
                            ),
                        ]
                    ),
                    scale,
                )  # 使用 ops.truediv 计算 part，即 grad_loader(...) / scale

                mask = ops.and_(
                    ops.lt(ph, phend),  # 判断 ph 是否小于 phend
                    ops.lt(pw, pwend),  # 判断 pw 是否小于 pwend
                )
                if gradient is None:
                    gradient = ops.where(mask, part, ops.constant(0.0, torch.float32))  # 如果 gradient 为 None，则根据 mask 条件选择 part 或 0.0 赋值给 gradient
                else:
                    gradient = ops.where(mask, ops.add(gradient, part), gradient)  # 否则根据 mask 条件选择 gradient + part 或 gradient 赋值给 gradient
        assert gradient is not None  # 断言 gradient 不为 None
        return gradient  # 返回计算得到的 gradient

    rv = Pointwise.create(
        device=grad_output.get_device(),  # 创建 Pointwise 对象，指定设备为 grad_output 的设备
        dtype=dtype,  # 指定数据类型为 dtype
        inner_fn=fn,  # 将之前定义的 fn 函数作为内部函数传入 Pointwise 对象
        ranges=new_size,  # 指定范围参数为 new_size
    )
    return rv  # 返回创建的 Pointwise 对象
# 使用 fallback_handler 函数注册对 aten.avg_pool3d_backward.default 的回退处理函数，并禁止将其添加到回退集合中
fallback_avg_pool3d_backward = fallback_handler(
    aten.avg_pool3d_backward.default, add_to_fallback_set=False
)

# 注册针对 aten.avg_pool3d_backward 函数的降级操作，不进行类型提升
@register_lowering(aten.avg_pool3d_backward, type_promotion_kind=None)
def avg_pool3d_backward(
    grad_output,             # 梯度输出张量
    x,                       # 输入张量 x
    kernel_size,             # 池化核大小
    stride,                  # 池化步长
    padding,                 # 池化填充
    ceil_mode,               # 是否启用向上取整模式
    count_include_pad,       # 计算中是否包括填充
    divisor_override=None,   # 除数覆盖参数（可选）
):
    assert divisor_override is None or divisor_override != 0, "divisor must be not zero"
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0, 0]

    assert isinstance(grad_output, TensorBox)    # 断言 grad_output 是 TensorBox 类型
    assert isinstance(x, TensorBox)              # 断言 x 是 TensorBox 类型
    assert len(kernel_size) == 3                 # 断言 kernel_size 长度为 3
    assert len(stride) == 3                      # 断言 stride 长度为 3
    assert len(padding) == 3                     # 断言 padding 长度为 3
    assert len(x.get_size()) in (4, 5)           # 断言 x 的维度大小为 4 或 5

    grad_output.realize_hint()                   # 根据梯度输出提示实现梯度输出

    *batch, depth, height, width = x.get_size()  # 获取批次维度和空间维度（深度、高度、宽度）

    # 计算池化后的输出大小和是否使用了向上取整模式
    d_out, ceil_mode_d = pooling_size(depth, 0, kernel_size, stride, padding, ceil_mode)
    h_out, ceil_mode_h = pooling_size(height, 1, kernel_size, stride, padding, ceil_mode)
    w_out, ceil_mode_w = pooling_size(width, 2, kernel_size, stride, padding, ceil_mode)

    grad_loader = grad_output.make_loader()      # 创建梯度加载器
    had_padding = any(padding) or ceil_mode_d or ceil_mode_h or ceil_mode_w  # 检查是否进行了填充操作或启用了向上取整模式

    *_, pooled_depth, pooled_height, pooled_width = grad_output.get_size()  # 获取池化后的输出尺寸
    new_size = list(x.get_size())                # 获取输入张量 x 的尺寸并转换为列表形式
    dtype = x.get_dtype()                        # 获取输入张量 x 的数据类型

    # 计算池化窗口的深度、高度和宽度
    d_window_size, h_window_size, w_window_size = (
        max(
            max(d // stride[i] - max(0, (d - kernel_size[i]) // stride[i]), 1)
            for d in range(kernel_size[i] * 2)
        )
        for i in range(3)
    )

    window_size = d_window_size * h_window_size * w_window_size  # 计算池化窗口大小
    if window_size > 125:
        # 如果池化窗口大小超过 125，则返回使用回退函数处理的结果
        return fallback_avg_pool3d_backward(
            grad_output,
            x,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    # 定义一个函数，计算池化操作的大小，不包括填充部分
    def compute_pool_size_without_padding(pd, ph, pw):
        # 将步长转换为 Torch 的常量张量
        stride_d, stride_h, stride_w = (ops.constant(s, torch.int32) for s in stride)
        # 将填充转换为 Torch 的常量张量
        pad_d, pad_h, pad_w = (ops.constant(p, torch.int32) for p in padding)
        # 将核大小转换为 Torch 的常量张量
        kernel_d, kernel_h, kernel_w = (
            ops.constant(k, torch.int32) for k in kernel_size
        )

        # 计算池化操作在深度、高度、宽度上的起始位置
        dstart, hstart, wstart = (
            ops.sub(ops.mul(p, s), pad)
            for p, s, pad in zip(
                [pd, ph, pw], [stride_d, stride_h, stride_w], [pad_d, pad_h, pad_w]
            )
        )
        # 计算池化操作在深度、高度、宽度上的结束位置
        dend, hend, wend = (
            ops.minimum(
                ops.add(start, k), ops.add(ops.index_expr(dim, torch.int32), pad)
            )
            for start, k, dim, pad in zip(
                [dstart, hstart, wstart],
                [kernel_d, kernel_h, kernel_w],
                [depth, height, width],
                [pad_d, pad_h, pad_w],
            )
        )
        # 对起始位置进行修正，确保不小于零
        dstart, hstart, wstart = (
            ops.maximum(start, ops.constant(0, torch.int32))
            for start in [dstart, hstart, wstart]
        )
        # 对结束位置进行修正，确保不超过维度的最大值
        dend, hend, wend = (
            ops.minimum(end, ops.index_expr(dim, torch.int32))
            for end, dim in zip([dend, hend, wend], [depth, height, width])
        )
        # 计算池化操作的分母因子，即池化区域的体积
        divide_factor = ops.mul(
            ops.mul(ops.sub(dend, dstart), ops.sub(hend, hstart)), ops.sub(wend, wstart)
        )
        # 返回计算得到的池化操作的大小
        return divide_factor

    # 创建一个 Pointwise 对象，并返回
    rv = Pointwise.create(
        device=grad_output.get_device(),  # 设置设备
        dtype=dtype,  # 设置数据类型
        inner_fn=fn,  # 设置内部函数
        ranges=new_size,  # 设置新的尺寸范围
    )
    # 返回创建的 Pointwise 对象
    return rv
# 验证并规范化要进行减少操作的轴参数
def _validate_reduction_axis(x, axis):
    # 获取输入张量的尺寸信息
    size = x.get_size()
    # 如果 axis 是整数，则转换为列表
    if isinstance(axis, int):
        axis = [axis]
    # 如果 axis 为空，则设定为张量的所有维度范围
    elif not axis:
        axis = range(len(size))
    # 如果张量没有维度（零维张量），则检查轴参数是否有效
    if len(size) == 0:
        assert tuple(axis) in [(), (0,), (-1,)], f"invalid axis: {axis}"
        return []
    # 将 axis 转换为列表形式，并处理负数索引
    axis = list(axis)
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(size) if len(size) else 1
        # 检查轴的索引是否在张量的有效范围内
        assert 0 <= axis[i] < len(size) or (len(size) == 0 and axis[i] == 0)
    # 检查减少操作的轴是否唯一
    assert len(set(axis)) == len(axis), "reduction axis not unique"
    return axis


# 根据指定参数创建内部加载器，用于执行减少操作
def _make_reduction_inner(x, *, axis, keepdims, dtype, override_return_dtype):
    # 如果指定了 dtype，则将输入张量转换为指定的数据类型
    if dtype is not None:
        x = to_dtype(x, dtype)
    # 获取输入张量的尺寸信息
    size = x.get_size()
    # 获取经过验证的减少操作轴的集合
    axis = set(_validate_reduction_axis(x, axis))

    # 初始化保留和减少维度的信息
    kept_sizes = []
    kept_idx = []
    reduced_sizes = []
    reduced_idx = []
    for i in range(len(size)):
        if i in axis:
            reduced_idx.append(i)
            reduced_sizes.append(size[i])
        else:
            kept_idx.append(i)
            kept_sizes.append(size[i])

    # 定义加载器函数，用于加载指定索引和减少索引的数据
    def loader(index, reduction_index):
        assert len(reduction_index) == len(reduced_idx)
        if keepdims:
            assert len(index) == len(size)
            index = [index[i] for i in kept_idx]
        assert len(index) == len(kept_idx)
        new_index = [None] * (len(index) + len(reduction_index))
        for idx, var in itertools.chain(
            zip(kept_idx, index), zip(reduced_idx, reduction_index)
        ):
            new_index[idx] = var
        return inner_loader(new_index)

    # 如果 keepdims 为 True，则调整新尺寸以保持减少操作后的维度信息
    if keepdims:
        new_size = list(size)
        for i in reduced_idx:
            new_size[i] = sympy.Integer(1)
    else:
        new_size = kept_sizes

    # 获取内部加载器函数
    inner_loader = x.make_loader()
    # 返回包含各种元数据的字典，用于后续操作
    return dict(
        device=x.get_device(),
        dst_dtype=override_return_dtype or x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=loader,
        ranges=new_size,
        reduction_ranges=reduced_sizes,
    )


# 创建指定类型的减少操作函数
def make_reduction(reduction_type: str, override_return_dtype=None):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        # 根据指定参数生成内部元数据
        kwargs = _make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            override_return_dtype=override_return_dtype,
        )
        # 创建指定类型的减少操作结果
        result = Reduction.create(reduction_type=reduction_type, input_node=x, **kwargs)
        # 如果结果数据类型为 Reduction 类型，则进行实际数据实现
        if isinstance(
            result.data.data, Reduction
        ):  # Only realize if reduction isn't unrolled
            result.realize()
        return result

    return inner


# 创建指定轴的扫描操作的内部加载器
def _make_scan_inner(x, *, axis, dtype):
    # 如果指定了 dtype，则将输入张量转换为指定的数据类型
    if dtype is not None:
        x = to_dtype(x, dtype)
    # 获取输入张量的尺寸信息
    size = x.get_size()
    # 验证并获取扫描操作的轴参数
    axis = _validate_dim(x, axis)

    # 返回包含各种元数据的字典，用于后续扫描操作
    return dict(
        device=x.get_device(),
        dtypes=(x.get_dtype(),),
        inner_fns=(x.make_loader(),),
        size=x.get_size(),
        axis=axis,
    )


# 将函数注册为 ATen.mean 的降低版本的处理函数
@register_lowering(aten.mean)
# 计算给定数组的平均值
def mean(x, axis=None, keepdim=False, *, dtype=None):
    # 如果指定了dtype，则将输入数组转换为指定的数据类型
    if dtype is not None:
        x = to_dtype(x, dtype)
    # 获取输入数组的大小
    size = x.get_size()
    # 验证并返回有效的约简轴
    axis = _validate_reduction_axis(x, axis)
    # 确定输出的数据类型为输入数组的数据类型
    output_dtype = x.get_dtype()
    # 如果输出的数据类型是浮点16位或bfloat16，则将输入数组转换为浮点数类型
    if output_dtype in (torch.float16, torch.bfloat16):
        x = to_dtype(x, torch.float)
    # 计算数组在指定轴上的和
    sum_result = sum_(x, axis, keepdim)
    # 计算分母，即数组大小的乘积
    denom = sympy_product(size[i] for i in axis)
    # 创建一个常数索引，用于扩展视图，并指定数据类型和设备
    denom = ir.IndexingConstant(denom, x.get_dtype(), x.get_device())
    # 创建一个扩展视图，使其形状与和的结果一致
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    # 将和除以分母，并将结果转换为输出数据类型
    return to_dtype(div(sum_result, denom), output_dtype)


# 计算方差和均值的和
def var_mean_sum_(x, axis, correction, keepdim, return_mean):
    # 如果未指定校正值，则默认为1
    if correction is None:
        correction = 1

    # 获取输入数组的大小
    size = x.get_size()
    # 验证并返回有效的约简轴
    axis = _validate_reduction_axis(x, axis)
    # 计算输入数组在指定轴上的均值，并保持结果的维度
    x_mean = mean(x, axis, keepdim=True)
    # 如果需要返回均值，则实现均值计算
    if return_mean:
        x_mean.realize()

    # 计算输入数组与均值之间的差的平方
    diffs = square(sub(x, x_mean))
    # 计算差的平方在指定轴上的和
    sum_result = sum_(diffs, axis, keepdim)

    # 计算分母，即输入数组大小的乘积
    denom = sympy_product(size[i] for i in axis)
    # 如果需要校正，则减去校正值，并确保不小于0
    if correction:
        denom = sympy.Max(denom - correction, 0)
    # 创建一个常数索引，用于扩展视图，并指定数据类型和设备
    denom = ir.IndexingConstant(denom, x.get_dtype(), x.get_device())
    # 创建一个扩展视图，使其形状与和的结果一致
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    # 计算方差，并将结果存储在x_var中
    x_var = div(sum_result, denom)
    # 如果不需要返回均值，则仅返回方差
    if not return_mean:
        return (x_var,)

    # 如果需要返回均值，则根据keepdim标志决定返回的均值形状
    x_mean = x_mean if keepdim else squeeze(x_mean, axis)
    # 返回方差和均值
    return x_var, x_mean


# 使用两步骤的方差计算方法
def use_two_step_variance(x, axis, keepdim):
    # 验证并返回有效的约简轴
    axis = _validate_reduction_axis(x, axis)
    # 创建内部约简函数的参数
    kwargs = _make_reduction_inner(
        x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None
    )

    # 获取范围和约简的元素数
    ranges = kwargs["ranges"]
    reduction_numel = sympy_product(kwargs["reduction_ranges"])
    # 判断是否小于阈值，且范围不为1
    return (
        isinstance(reduction_numel, sympy.Integer)
        and int(reduction_numel) < config.unroll_reductions_threshold
        and sympy_product(ranges) != 1
    )


# 计算Welford方差和均值的和
def var_mean_welford_(x, axis, *, correction, keepdim, return_mean):
    # 如果未指定校正值，则默认为1
    if correction is None:
        correction = 1

    # 创建内部约简函数的参数
    kwargs = _make_reduction_inner(
        x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None
    )
    # 获取内部函数加载器和数据类型信息
    loader = kwargs.pop("inner_fn")
    kwargs.pop("dst_dtype")
    kwargs.pop("src_dtype")

    # 使用Welford算法创建均值、平方差和计数器
    mean, m2, _ = ir.WelfordReduction.create(
        inner_fns=(loader,),
        reduction_type="welford_reduce",
        dtype=x.get_dtype(),
        **kwargs,
    )
    # 实现m2的计算
    m2.realize()

    # 获取数据类型和数组大小
    dtype = x.get_dtype()
    size = x.get_size()
    # 验证并返回有效的约简轴
    axis = _validate_reduction_axis(x, axis)
    # 计算约简的元素数
    rnumel = sympy_product(size[i] for i in axis)

    # 获取常数索引表达式或索引表达式
    def get_constant_or_index_expr(x, dtype):
        if isinstance(x, sympy.Expr) and not x.is_number:
            return ops.to_dtype(ops.index_expr(x, torch.int64), dtype)
        return ops.constant(x, dtype)
    # 定义一个函数 scale_fn，接受一个数据参数
    def scale_fn(data):
        # 调用函数 get_constant_or_index_expr 获取修正常数或索引表达式，使用给定的数据类型
        c = get_constant_or_index_expr(correction, dtype)
        # 调用函数 get_constant_or_index_expr 获取rnumel的常数或索引表达式，使用给定的数据类型
        N = get_constant_or_index_expr(rnumel, dtype)
        # 创建一个常数张量 zero，其值为0，数据类型为 dtype
        zero = ops.constant(0, dtype)
        # 返回数据除以 N-c 的最大值，以进行数据缩放
        return data / ops.maximum(zero, N - c)

    # 调用 make_pointwise 函数，使用 scale_fn 函数创建一个变量 var，作用于 m2
    var = make_pointwise(scale_fn)(m2)

    # 如果 return_mean 为真
    if return_mean:
        # 调用 mean.realize() 方法，实现平均值计算（具体方法功能和实现略去）
        mean.realize()
        # 返回 var 和 mean 作为结果
        return var, mean
    # 如果 return_mean 为假，则只返回 var
    return (var,)
# 定义一个辅助函数，计算变量或均值相关操作
def var_mean_helper_(x, *, axis, correction, keepdim, return_mean):
    # 获取输入张量的数据类型
    out_dtype = x.get_dtype()
    # 根据输出数据类型获取计算数据类型
    compute_dtype = get_computation_dtype(out_dtype)
    # 将输入张量 x 转换为计算数据类型，如果不需要拷贝则直接引用
    x = to_dtype(x, compute_dtype, copy=False)
    # 构建参数字典
    kwargs = dict(
        x=x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=return_mean,
    )
    # 根据条件选择使用两步法方差计算或 Welford 方差计算
    output = (
        var_mean_sum_(**kwargs)
        if use_two_step_variance(x, axis=axis, keepdim=keepdim)
        else var_mean_welford_(**kwargs)
    )
    # 将输出结果转换为指定的输出数据类型
    output = tuple(to_dtype(x, out_dtype, copy=False) for x in output)
    # 如果不是返回均值，则返回第一个元素，否则返回全部结果
    return output[0] if not return_mean else output


# 注册对应的降级处理函数，处理 aten.var 和 prims.var 操作
@register_lowering([aten.var, prims.var])
def var_(x, axis=None, *, correction=None, keepdim=False):
    return var_mean_helper_(
        x, axis=axis, correction=correction, keepdim=keepdim, return_mean=False
    )


# 注册对应的降级处理函数，处理 aten.var_mean 操作
@register_lowering(aten.var_mean)
def var_mean(x, axis=None, *, correction=None, keepdim=False):
    return var_mean_helper_(
        x, axis=axis, correction=correction, keepdim=keepdim, return_mean=True
    )


# 递归计算幂函数，支持整数指数和负指数
def pow_recursive(x, y, dtype):
    if y < 0:
        return pow_recursive(ops.reciprocal(x), -y, dtype)
    if y == 0:
        return ops.constant(1, dtype)
    if y == 1:
        return x

    # 递归计算 y 的一半的幂
    result = pow_recursive(x, y // 2, dtype)
    # 将结果平方
    result = ops.mul(result, result)
    # 如果 y 是奇数，则再乘以 x
    if (y % 2) == 1:
        result = ops.mul(result, x)
    return result


# 使用内置的 ops.pow 实现的原生幂函数
@make_pointwise
def pow_native(a, b):
    return ops.pow(a, b)


# 处理 aten.pow.Tensor_Tensor 操作的降级处理
fallback_pow_tensor_tensor = fallback_handler(
    aten.pow.Tensor_Tensor, add_to_fallback_set=False
)

# 处理 aten.pow.Scalar 操作的降级处理
fallback_pow_scalar = fallback_handler(aten.pow.Scalar, add_to_fallback_set=False)

# 处理 aten.pow.Tensor_Scalar 操作的降级处理
fallback_pow_tensor_scalar = fallback_handler(
    aten.pow.Tensor_Scalar, add_to_fallback_set=False
)


# 注册对应的降级处理函数，处理 aten.pow 操作，支持广播
@register_lowering(aten.pow, broadcast=True)
def pow(a, b):
    # 如果指数 b 是浮点数且等于整数，则转换为整数
    if isinstance(b, float) and b == int(b):
        return pow(a, int(b))
    # 如果指数 b 是浮点数且等于 0.5，则返回 a 的平方根
    elif isinstance(b, float) and b == 0.5:
        return sqrt(a)
    # 如果指数 b 是整数且等于 1，则返回 a 的克隆
    elif isinstance(b, int) and b == 1:
        return clone(a)

    # 确保所有张量参数具有相同的数据类型
    dtype = next(x.get_dtype() for x in (a, b) if isinstance(x, ir.TensorBox))
    # 判断是否是整数类型的幂
    is_integer_pow = is_integer_dtype(dtype)

    # 优化小固定幂值，或者对于整数类型的避免回退到 ATen
    embed_exponent = isinstance(b, int) and (
        -32 < b < 32 or (is_integer_pow and b >= 0)
    )
    if embed_exponent:
        # 创建一个点对点的操作，逐元素计算 a 的 b 次幂
        loader = a.make_loader()

        def fn(idx):
            return pow_recursive(loader(idx), b, a.get_dtype())

        return Pointwise.create(
            device=a.get_device(),
            dtype=a.get_dtype(),
            inner_fn=fn,
            ranges=a.get_size(),
        )

    # 如果 a 是数字 1，则返回与 b 相同大小的全 1 张量
    if isinstance(a, Number):
        if a == 1:
            return full_like(b, 1)
        # 如果 a 是数字 2 且 b 是浮点数类型，则返回 2 的 b 次幂
        if a == 2 and is_float_dtype(b.get_dtype()):
            return exp2(b)
    # 如果 is_integer_pow 为真，则执行以下逻辑
    if is_integer_pow:
        # 当 ops.pow 对整数类型不适用时
        if isinstance(a, Number):
            # 如果 a 是数字类型，返回使用标准方法计算的结果
            return fallback_pow_scalar(a, b)
        elif isinstance(b, Number):
            # 如果 b 是数字类型，返回使用张量和标量计算的结果
            return fallback_pow_tensor_scalar(a, b)
        else:
            # 否则返回使用张量和张量计算的结果
            return fallback_pow_tensor_tensor(a, b)

    # 如果 is_integer_pow 不为真，则执行原生的幂运算
    return pow_native(a, b)
# 将对象 `changed` 变异为值 `val`，可选择使用不安全别名
def mutate_to(changed, val, unsafe_alias=False):
    # 如果 `changed` 是 `TensorBox` 类型，则获取其数据
    if isinstance(changed, TensorBox):
        changed_data = changed.data
    else:
        changed_data = changed
    # 如果 `val` 是 `TensorBox` 类型，则获取其数据
    if isinstance(val, TensorBox):
        val = val.data

    # 如果 `val` 不是 `ir.StorageBox` 类型
    if not isinstance(val, ir.StorageBox):
        # 引入一个副本来处理视图
        val = Pointwise.create(
            device=changed.get_device(),
            dtype=changed.get_dtype(),
            inner_fn=val.make_loader(),
            ranges=changed.get_size(),
        ).data
        # 确保 `val` 是 `ir.StorageBox` 类型
        assert isinstance(val, ir.StorageBox)

    # 如果 `changed_data` 是 `ir.StorageBox` 类型且不是以下情况之一
    if isinstance(changed_data, ir.StorageBox) and not (
        changed_data.is_input_buffer()
        or changed_data.is_module_buffer()
        or isinstance(changed_data.data, ir.NopKernel)
    ):
        # 快速路径，直接移动数据指针
        val.realize()
        changed_data.data = val.data
        return changed

    # 执行变异布局
    ir.MutationLayoutSHOULDREMOVE.realize_into(
        val, changed_data, unsafe_alias=unsafe_alias
    )
    return changed


# 注册对 `aten.fill_` 操作的降低处理
@register_lowering(aten.fill_)
def fill_(x, fill_value):
    return mutate_to(x, full_like(x, fill_value))


# 注册对 `aten.copy_` 操作的降低处理，不包含类型提升
@register_lowering(aten.copy_, type_promotion_kind=None)
def copy_(dst, src, non_blocking=False):
    # 将 `src` 移动到 `dst` 所在的设备
    src = to_device(src, dst.get_device())
    # 将 `src` 转换为 `dst` 的数据类型
    src = to_dtype(src, dst.get_dtype())
    # 扩展 `src` 的大小以匹配 `dst`
    src = expand(src, dst.get_size())
    return mutate_to(dst, src)


# 将 `floordiv` 函数转化为点对点操作
@make_pointwise
def floordiv(a, b):
    return ops.floordiv(a, b)


# 将 `truncdiv` 函数转化为点对点操作
@make_pointwise
def truncdiv(a, b):
    return ops.truncdiv(a, b)


# 注册对 `aten.div` 操作的降低处理，支持广播
@register_lowering(aten.div, broadcast=True)
def div_mode(a, b, rounding_mode=None):
    both_integer = is_integer_type(a) and is_integer_type(b)
    both_boolean = is_boolean_type(a) and is_boolean_type(b)

    # 在 Triton 上，整数张量的 floordiv 和 truncdiv 需要特殊处理
    if rounding_mode == "floor":
        assert not both_boolean, "floordiv operands can not be boolean at the same time"
        return floordiv(a, b) if both_integer else floor(div(a, b))
    if rounding_mode == "trunc":
        assert not both_boolean, "truncdiv operands can not be boolean at the same time"
        return truncdiv(a, b) if both_integer else trunc(div(a, b))
    return div(a, b)


# 注册对 `aten.mul` 操作的降低处理，支持广播
@register_lowering([aten.mul], broadcast=True)
def mul(a, b):
    both_bool = is_boolean_type(a) and is_boolean_type(b)
    if both_bool:
        return logical_and(a, b)
    else:
        fn = ops_wrapper(aten.mul.__name__)
        return make_pointwise(fn)(a, b)


# 尝试将任意 IR 节点转换为 `ir.Constant` 类型的值
def get_constant_value(x: ir.IRNode) -> Optional[ir.Constant]:
    """Try convert an arbitrary IR node into an ir.Constant value"""

    # 首先尝试解包 IRNode，看它是否已经是 `ir.Constant`
    # 可选步骤，但避免不必要的 inner_fn 评估
    if isinstance(x, ir.MutableBox):
        return get_constant_value(x.data)
    # 检查 x 是否是 ir.BaseView 类型的实例
    if isinstance(x, ir.BaseView):
        # 如果是，递归获取其展开视图的常量值
        return get_constant_value(x.unwrap_view())
    # 检查 x 是否是 ir.Constant 类型的实例
    if isinstance(x, ir.Constant):
        # 如果是常量，则直接返回
        return x

    # 如果展开后的节点不是 ir.Constant 类型，则尝试评估 inner_fn
    # 看返回值是否来自 `ops.constant` 调用
    if not isinstance(x, ir.Loops):
        # 如果不是循环节点类型，则返回 None
        return None

    # 创建一个处理器对象，用于从设备获取常量处理器
    handler = torch._inductor.ops_handler.ExtractConstantsHandler(x.get_device())
    # 使用上下文管理器设置操作处理器和允许索引的灵活布局
    with V.set_ops_handler(handler), patch.object(
        ir.FlexibleLayout, "allow_indexing", True
    ):
        # 调用 inner_fn，并传入其参数
        out = x.inner_fn(*x.inner_fn_args())

    # 断言 out 是 torch._inductor.virtualized.OpsValue 类型的实例
    assert isinstance(out, torch._inductor.virtualized.OpsValue)
    # 检查 out 的值是否是 ir.Constant 类型的实例
    if isinstance(out.value, ir.Constant):
        # 如果是常量，则返回其值
        return out.value
    # 否则返回 None
    return None
@register_lowering([prims.div], broadcast=True)
def div_prim(a, b):
    # 检查a和b是否均为整数类型或布尔类型，如果是，则执行截断除法；否则，执行真实除法
    is_integral = all(is_boolean_type(x) or is_integer_type(x) for x in [a, b])

    if is_integral:
        # 如果a和b均为整数类型或布尔类型，则执行截断除法
        return truncdiv(a, b)

    if (divisor := get_constant_value(b)) is not None:
        # 如果b是常量，则用其倒数替换除法操作，避免除以零情况
        if divisor.value == 0:
            reciprocal = math.copysign(float("inf"), divisor.value)
        else:
            reciprocal = 1.0 / divisor.value
        return mul(a, reciprocal)

    def fn(*args):
        return ops.truediv(*args)

    # 如果b不是常量，则返回一个执行真实除法的函数
    return make_pointwise(fn)(a, b)


@register_lowering(
    [aten.true_divide, aten.div.Tensor],
    broadcast=True,
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def div(a, b):
    # 提升a和b的类型至浮点型，以进行真实除法操作
    a, b = promote_constants(
        (a, b), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )
    return div_prim(a, b)


@register_lowering([aten.fmod, prims.fmod], broadcast=True)
def fmod(a, b):
    # 检查a是否为布尔类型或整数类型
    is_integral = is_boolean_type(a) or is_integer_type(a)

    if is_integral:
        # 如果a是整数类型，则返回计算余数的函数
        def fn(a, b):
            return ops.mod(a, b)

    else:
        # 如果a不是整数类型，则返回计算浮点数余数的函数
        def fn(a, b):
            return ops.fmod(a, b)

    # 返回应用所选函数的结果
    return make_pointwise(fn)(a, b)


@register_lowering(aten.rsqrt)
def rsqrt(x):
    dtype = x.get_dtype()
    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        x = to_dtype(x, torch.get_default_dtype())

    def _rsqrt(x):
        return ops.rsqrt(x)

    # 应用开方倒数操作到x上
    return make_pointwise(_rsqrt)(x)


@register_lowering([aten.sum, prims.sum])
def sum_(x, axis=None, keepdims=False, *, dtype=None):
    # 如果x是整数类型或布尔类型，并且未指定dtype，则将dtype设为torch.int64
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    # 创建sum操作的函数fn，并返回其结果
    fn = make_reduction("sum", override_return_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)


fallback_cumsum = fallback_handler(aten.cumsum.default)
fallback_cumprod = fallback_handler(aten.cumprod.default)
fallback_logcumsumexp = fallback_handler(aten.logcumsumexp.default)
fallback_cummax = fallback_handler(aten.cummax.default)
fallback_cummin = fallback_handler(aten.cummin.default)


@register_lowering(aten.cumsum)
def cumsum(x, axis=None, dtype=None):
    # 如果x是整数类型或布尔类型，并且未指定dtype，则将dtype设为torch.int64
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    # 如果x的维度为0，则返回其自身（进行类型转换为指定dtype）
    if len(x.get_size()) == 0:
        assert axis in [0, -1]
        dtype = dtype or x.get_dtype()
        return to_dtype(x, dtype, copy=True)

    # 定义结合函数combine_fn，用于累积求和操作
    def combine_fn(a_tuple, b_tuple):
        (a,) = a_tuple
        (b,) = b_tuple
        return (ops.add(a, b),)

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    (result,) = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    if result is None:
        return fallback_cumsum(x, dim=axis, dtype=dtype)
    return result
@register_lowering(aten.cumprod)
# 注册对 torch.cumprod 操作的降级实现
def cumprod(x, axis=None, dtype=None):
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        # 如果输入张量是整数类型或布尔类型，并且未指定 dtype，则设置为 torch.int64
        dtype = torch.int64

    if len(x.get_size()) == 0:
        # 如果输入张量的维度为 0
        assert axis in [0, -1]
        dtype = dtype or x.get_dtype()
        # 将输入张量转换为指定的 dtype 类型，并返回
        return to_dtype(x, dtype, copy=True)

    def combine_fn(a_tuple, b_tuple):
        # 定义组合函数，对两个输入元组中的元素执行累积乘法
        (a,) = a_tuple
        (b,) = b_tuple
        return (ops.mul(a, b),)

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    # 使用 _make_scan_inner 函数生成扫描操作所需的参数
    (result,) = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    # 调用 Scan.create 方法执行累积乘法操作
    if result is None:
        return fallback_cumprod(x, dim=axis, dtype=dtype)
    # 如果 result 为空，调用备用的 cumprod 函数进行计算
    return result


@register_lowering(aten.logcumsumexp)
# 注册对 torch.logcumsumexp 操作的降级实现
def logcumsumexp(x, dim):
    def log_add_exp_helper(a_tuple, b_tuple):
        # 定义辅助函数，执行对数累加指数操作
        (a,) = a_tuple
        (b,) = b_tuple
        min_v = ops.minimum(a, b)
        max_v = ops.maximum(a, b)
        mask = (min_v != max_v) | (~ops.isinf(min_v))
        return (ops.where(mask, ops.log1p(ops.exp(min_v - max_v)) + max_v, a),)

    dtype = x.get_dtype()
    if len(x.get_size()) == 0:
        # 如果输入张量的维度为 0
        assert dim in [0, -1]
        return clone(x)
        # 克隆输入张量并返回

    kwargs = _make_scan_inner(x, axis=dim, dtype=dtype)
    # 使用 _make_scan_inner 函数生成扫描操作所需的参数
    (result,) = ir.Scan.create(**kwargs, combine_fn=log_add_exp_helper)
    # 调用 Scan.create 方法执行对数累加指数操作
    if result is None:
        return fallback_logcumsumexp(x, dim=dim)
    # 如果 result 为空，调用备用的 logcumsumexp 函数进行计算
    return result


@register_lowering(aten.cummax, type_promotion_kind=None)
# 注册对 torch.cummax 操作的降级实现，且不进行类型提升
def cummax(x, axis=None):
    if len(x.get_size()) == 0:
        # 如果输入张量的维度为 0
        assert axis in [0, -1]
        return clone(x), empty_like(x, dtype=torch.int64)
        # 克隆输入张量并返回，同时返回一个与输入张量形状相同的空张量，dtype 设置为 torch.int64

    dtype = x.get_dtype()
    combine_fn = ir.get_reduction_combine_fn(
        "argmax", dtype=dtype, arg_break_ties_left=False
    )
    # 获取用于 argmax 操作的组合函数

    min_value = (
        False
        if dtype is torch.bool
        else (
            torch.finfo(dtype).min
            if dtype.is_floating_point
            else torch.iinfo(dtype).min
        )
    )
    # 根据数据类型确定最小值

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    kwargs["dtypes"] = (dtype, torch.int64)
    kwargs["inner_fns"] = (x.make_loader(), lambda _: "rindex")
    # 设置扫描操作所需的参数

    values, indices = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    # 调用 Scan.create 方法执行累积最大值操作
    if values is None:
        return fallback_cummax(x, dim=axis)
    # 如果 values 为空，调用备用的 cummax 函数进行计算
    return values, indices


@register_lowering(aten.cummin, type_promotion_kind=None)
# 注册对 torch.cummin 操作的降级实现，且不进行类型提升
def cummin(x, axis=None):
    if len(x.get_size()) == 0:
        # 如果输入张量的维度为 0
        assert axis in [0, -1]
        return clone(x), empty_like(x, dtype=torch.int64)
        # 克隆输入张量并返回，同时返回一个与输入张量形状相同的空张量，dtype 设置为 torch.int64

    dtype = x.get_dtype()
    combine_fn = ir.get_reduction_combine_fn(
        "argmin", dtype=dtype, arg_break_ties_left=False
    )
    # 获取用于 argmin 操作的组合函数

    max_value = (
        True
        if dtype is torch.bool
        else (
            torch.finfo(dtype).max
            if dtype.is_floating_point
            else torch.iinfo(dtype).max
        )
    )
    # 根据数据类型确定最大值

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    kwargs["dtypes"] = (dtype, torch.int64)
    kwargs["inner_fns"] = (x.make_loader(), lambda _: "rindex")
    # 设置扫描操作所需的参数
    # 使用参数 kwargs 创建一个 ir.Scan 对象，并指定 combine_fn 函数
    values, indices = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    # 如果 values 为 None，则调用 fallback_cummin 函数，并返回其结果
    if values is None:
        return fallback_cummin(x, dim=axis)
    # 否则返回 values 和 indices 作为结果
    return values, indices
@register_lowering(aten.prod)
def prod(x, axis=None, keepdims=False, *, dtype=None):
    # 如果 x 的数据类型是整数或布尔型，并且未指定 dtype，则设置 dtype 为 torch.int64
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    # 创建 "prod" 类型的降维操作函数
    fn = make_reduction("prod", override_return_dtype=dtype)
    # 调用降维函数 fn，并返回结果
    return fn(x, axis, keepdims, dtype=dtype)


@register_lowering(aten.any)
def reduce_any(x, dim=None, keepdim=False):
    # 将 x 转换为 torch.bool 类型
    x = to_dtype(x, torch.bool)
    # 创建 "any" 类型的降维操作函数，并调用
    return make_reduction("any")(x, axis=dim, keepdims=keepdim)


@register_lowering(aten.max, type_promotion_kind=None)
def reduce_max(x, dim=None, keepdim=False):
    # 如果指定了 dim，则返回 reduce_amax 和 reduce_argmax 的结果
    if dim is not None:
        return (
            reduce_amax(x, axis=dim, keepdims=keepdim),
            reduce_argmax(x, axis=dim, keepdims=keepdim),
        )

    # 否则，返回 reduce_amax 的结果
    return reduce_amax(x, axis=None, keepdims=keepdim)


@register_lowering(aten.min, type_promotion_kind=None)
def reduce_min(x, dim=None, keepdim=False):
    # 如果指定了 dim，则返回 reduce_amin 和 reduce_argmin 的结果
    if dim is not None:
        return (
            reduce_amin(x, axis=dim, keepdims=keepdim),
            reduce_argmin(x, axis=dim, keepdims=keepdim),
        )

    # 否则，返回 reduce_amin 的结果
    return reduce_amin(x, axis=None, keepdims=keepdim)


register_lowering(prims.xor_sum)(make_reduction("xor_sum"))
# 使用 make_reduction 函数创建 "xor_sum" 类型的降维操作函数，并注册到 prims.xor_sum

reduce_amax = register_lowering(aten.amax)(make_reduction("max"))
# 使用 make_reduction 函数创建 "max" 类型的降维操作函数，并注册到 aten.amax，同时赋值给 reduce_amax

reduce_amin = register_lowering(aten.amin)(make_reduction("min"))
# 使用 make_reduction 函数创建 "min" 类型的降维操作函数，并注册到 aten.amin，同时赋值给 reduce_amin

reduce_argmax = register_lowering(aten.argmax)(
    make_reduction("argmax", override_return_dtype=torch.int64)
)
# 使用 make_reduction 函数创建 "argmax" 类型的降维操作函数，并注册到 aten.argmax，同时指定返回 dtype 为 torch.int64

reduce_argmin = register_lowering(aten.argmin)(
    make_reduction("argmin", override_return_dtype=torch.int64)
)
# 使用 make_reduction 函数创建 "argmin" 类型的降维操作函数，并注册到 aten.argmin，同时指定返回 dtype 为 torch.int64

add = register_pointwise(
    aten.add, allow_alpha=True, override_fn_when_input_bool="logical_or"
)
# 注册逐点操作函数，支持 alpha 参数和当输入为布尔型时的逻辑或操作

sort_fallback = fallback_handler(aten.sort.stable, add_to_fallback_set=False)
# 创建排序的回退处理函数，将 aten.sort.stable 作为默认值，但不将其添加到回退集合中


@register_lowering(aten.sort.stable, type_promotion_kind=None)
def sort_stable(x, *, stable=None, dim=-1, descending=False):
    # 如果 stable 未指定，则默认为 False
    if stable is None:
        stable = False

    # 获取张量 x 的形状和设备信息
    shape = x.get_size()
    device = x.get_device()
    # 规范化维度参数 dim
    dim = canonicalize_dim(len(shape), dim)
    # 如果张量 x 是标量，则返回其克隆和对应的索引张量
    if len(shape) == 0:
        return clone(x), _full(0, device, torch.int64, shape)

    # 获取指定维度的大小
    dim_size = shape[dim] if len(shape) else 1
    # 创建索引张量，以便按照指定维度进行排序
    indices = iota(
        dim_size, start=0, step=1, dtype=torch.int64, device=device, requires_grad=False
    )
    view_shape = [1] * len(shape)
    if len(shape):
        view_shape[dim] = dim_size
    indices = view(indices, view_shape)
    indices = expand(indices, shape)

    # 调用 Sort.create 函数进行排序操作，并返回排序后的值和索引
    values, indices = ir.Sort.create(
        device=device,
        dtypes=(x.dtype, indices.dtype),
        inner_fns=(x.make_loader(), indices.make_loader()),
        size=shape,
        axis=dim,
        stable=stable,
        descending=descending,
    )
    # 如果无法排序，则回退到 sort_fallback 处理函数
    if values is None:
        return sort_fallback(x, stable=stable, dim=dim, descending=descending)

    # 返回排序后的值和索引
    return values, indices


@register_lowering(aten.sort.default, type_promotion_kind=None)
def sort(x, dim=-1, descending=False):
    # 该函数还未实现，待完善
    pass
    # 调用稳定排序函数sort_stable，返回排序后的列表
    return sort_stable(x, stable=False, dim=dim, descending=descending)
def register_pointwise_numeric(op, name=None, triton_fallback=None):
    # 注册一个数值型的逐点操作，返回注册的结果
    return register_pointwise(
        op,
        name=name,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        triton_fallback=triton_fallback,
    )


def register_pointwise_numeric_ldf64(op):
    # 注册一个操作，将整数转换为浮点数类型，使用 libdevice 处理双精度浮点数
    return register_pointwise(
        op,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        use_libdevice_for_f64=True,
    )


exp = register_pointwise_numeric_ldf64(aten.exp)  # 注册指数函数操作，处理双精度浮点数
exp2 = register_pointwise_numeric(aten.exp2)  # 注册2的指数函数操作，处理整数转浮点数
expm1 = register_pointwise_numeric(aten.expm1)  # 注册指数函数减1操作，处理整数转浮点数
relu = register_pointwise(aten.relu)  # 注册ReLU激活函数操作
sigmoid = register_pointwise_numeric_ldf64(aten.sigmoid)  # 注册Sigmoid函数操作，处理双精度浮点数
sqrt = register_pointwise_numeric_ldf64(aten.sqrt)  # 注册平方根函数操作，处理双精度浮点数
square = register_pointwise(aten.square)  # 注册平方函数操作
sub = register_pointwise(aten.sub, allow_alpha=True)  # 注册减法操作，允许alpha参数
register_pointwise_numeric_ldf64(aten.cos)  # 注册余弦函数操作，处理双精度浮点数
register_pointwise_numeric_ldf64(aten.sin)  # 注册正弦函数操作，处理双精度浮点数
abs = register_pointwise(aten.abs)  # 注册绝对值函数操作
bitwise_and = register_pointwise(aten.bitwise_and)  # 注册按位与操作
bitwise_left_shift = register_pointwise(aten.bitwise_left_shift)  # 注册按位左移操作
bitwise_not = register_pointwise(
    aten.bitwise_not, override_fn_when_input_bool="logical_not"
)  # 注册按位取反操作，如果输入为布尔型则使用逻辑非
bitwise_or = register_pointwise(aten.bitwise_or)  # 注册按位或操作
bitwise_right_shift = register_pointwise(aten.bitwise_right_shift)  # 注册按位右移操作
bitwise_xor = register_pointwise(aten.bitwise_xor)  # 注册按位异或操作
register_pointwise_numeric(aten.lgamma)  # 注册伽马函数的对数操作，处理整数转浮点数
erf = register_pointwise_numeric(aten.erf)  # 注册误差函数操作，处理整数转浮点数
register_lowering(
    aten.special_erf, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)(erf)  # 降低特殊的误差函数操作，处理整数转浮点数

register_pointwise_numeric(aten.log1p)  # 注册对数函数操作，处理整数转浮点数
register_pointwise_numeric(aten.tan)  # 注册正切函数操作，处理整数转浮点数
register_pointwise_numeric_ldf64(aten.log)  # 注册对数函数操作，处理双精度浮点数
logical_and = register_pointwise(
    aten.logical_and,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)  # 注册逻辑与操作，不进行类型提升，将输入转换为布尔型，返回值类型为torch.bool
logical_not = register_pointwise(
    aten.logical_not,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)  # 注册逻辑非操作，不进行类型提升，将输入转换为布尔型，返回值类型为torch.bool
logical_or = register_pointwise(
    aten.logical_or,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)  # 注册逻辑或操作，不进行类型提升，将输入转换为布尔型，返回值类型为torch.bool
logical_xor = register_pointwise(
    aten.logical_xor,
    type_promotion_kind=None,
    convert_input_to_bool=True,
    override_return_dtype=torch.bool,
)  # 注册逻辑异或操作，不进行类型提升，将输入转换为布尔型，返回值类型为torch.bool
maximum = register_pointwise(aten.maximum)  # 注册最大值操作
minimum = register_pointwise(aten.minimum)  # 注册最小值操作
register_lowering(aten.clamp_min)(maximum)  # 降低 clamp_min 操作的注册，应用于最大值操作
register_lowering(aten.clamp_max)(minimum)  # 降低 clamp_max 操作的注册，应用于最小值操作
neg = register_pointwise(aten.neg)  # 注册取负操作
abs = register_pointwise(aten.abs)  # 注册绝对值操作
reciprocal = register_pointwise_numeric(aten.reciprocal)  # 注册倒数操作，处理整数转浮点数
register_pointwise(aten.remainder)  # 注册取余数操作
sign = register_pointwise(
    aten.sign, override_fn_when_input_bool="identity"
)  # 注册符号函数操作，如果输入为布尔型则使用恒等函数
register_pointwise(aten.ceil)  # 注册向上取整操作
register_pointwise(
    aten.signbit, override_return_dtype=torch.bool
)  # 注册符号位判断操作，返回值类型为torch.bool

register_lowering(aten._neg_view)(neg)  # 降低 _neg_view 操作的注册，应用于取负操作

register_pointwise(aten.le, override_return_dtype=torch.bool)  # 注册小于等于比较操作，返回值类型为torch.bool
register_pointwise(aten.lt, override_return_dtype=torch.bool)  # 注册小于比较操作，返回值类型为torch.bool
# 注册 pointwise 操作，使用 torch.bool 作为返回数据类型覆盖
register_pointwise(aten.ge, override_return_dtype=torch.bool)
# 将返回值为 register_pointwise 的结果赋给变量 gt
gt = register_pointwise(aten.gt, override_return_dtype=torch.bool)
# 注册 pointwise 操作，使用 torch.bool 作为返回数据类型覆盖
register_pointwise(aten.eq, override_return_dtype=torch.bool)
# 注册 pointwise 操作，使用 torch.bool 作为返回数据类型覆盖
register_pointwise(aten.ne, override_return_dtype=torch.bool)

# 注册数值型的 pointwise 操作
register_pointwise_numeric(aten.cosh)
register_pointwise_numeric(aten.sinh)
register_pointwise_numeric(aten.acos)
register_pointwise_numeric(aten.acosh)
register_pointwise_numeric(aten.asin)
register_pointwise_numeric(aten.asinh)
register_pointwise_numeric(aten.atan2)
register_pointwise_numeric(aten.atan)
register_pointwise_numeric(aten.atanh)
register_pointwise_numeric(aten.copysign)
register_pointwise_numeric(aten.erfc)
register_pointwise_numeric(aten.erfinv)
register_pointwise_numeric(aten.hypot)
register_pointwise_numeric(aten.log10)
register_pointwise_numeric(aten.log2)
register_pointwise_numeric(aten.nextafter)

# 从代码生成通用的后端特性和 pointwise 覆盖数据导入
from .codegen.common import BackendFeature, pointwise_overrides_data

# 定义函数用于获取指定命名空间和名称的 pointwise 覆盖
def _get_pointwise_overrides(ns, name):
    # 获取 pointwise 覆盖的数据
    data = pointwise_overrides_data[name]
    # 获取命名空间 ns 中名称为 data.name 的操作
    op = getattr(ns, data.name, None)
    if op is None:
        return

    # 定义 triton_fallback 函数
    def make_triton_fallback(op):
        # 如果 data.triton 为 None，则返回 fallback_handler(op) 的结果

    # 如果 op 是 torch._ops.OpOverloadPacket 的实例
    if isinstance(op, torch._ops.OpOverloadPacket):
        # 遍历 op 的重载列表
        for olname in op.overloads():
            # 获取每个重载的操作对象
            ol = getattr(op, olname)
            # 返回重载操作对象 ol、类型提升方式 data.type_promotion_kind 和 triton_fallback 函数
            yield ol, data.type_promotion_kind, make_triton_fallback(ol)
    else:
        # 返回操作对象 op、类型提升方式 data.type_promotion_kind 和 triton_fallback 函数
        yield op, data.type_promotion_kind, make_triton_fallback(op)

# 遍历 pointwise_overrides_data 中的每个名称
for name in pointwise_overrides_data:
    # 调用 _get_pointwise_overrides 函数获取 aten 命名空间中的 pointwise 覆盖
    for op, type_promotion_kind, triton_fallback in _get_pointwise_overrides(
        aten, name
    ):
        # 注册 pointwise 操作，设置名称 name、类型提升方式 type_promotion_kind 和 triton_fallback 函数
        register_pointwise(
            op,
            name=name,
            type_promotion_kind=type_promotion_kind,
            triton_fallback=triton_fallback,
        )

    # 调用 _get_pointwise_overrides 函数获取 prims 命名空间中的 pointwise 覆盖
    for op, type_promotion_kind, triton_fallback in _get_pointwise_overrides(
        prims, name
    ):
        # 注册 pointwise 操作，设置名称 name、类型提升方式 type_promotion_kind 和 triton_fallback 函数
        register_pointwise(
            op,
            name=name,
            type_promotion_kind=type_promotion_kind,
            triton_fallback=triton_fallback,
        )

# 注册 foreach_pointwise 操作，针对列表元素进行加法操作，并允许 alpha 值存在
foreach_add_list = register_foreach_pointwise(
    aten._foreach_add.List, add, allow_alpha=True
)
# 注册 foreach_pointwise 操作，针对标量元素进行加法操作，并允许 alpha 值存在
foreach_add_scalar = register_foreach_pointwise(
    aten._foreach_add.Scalar, add, allow_alpha=True
)
# 注册 foreach_pointwise 操作，针对张量元素进行加法操作
register_foreach_pointwise(aten._foreach_add.Tensor, add, allow_alpha=True)
# 注册 foreach_pointwise 操作，针对列表元素进行乘法操作
foreach_mul_list = register_foreach_pointwise(aten._foreach_mul.List, mul)
# 注册 foreach_pointwise 操作，针对标量元素进行乘法操作
foreach_mul_scalar = register_foreach_pointwise(aten._foreach_mul.Scalar, mul)
# 注册 foreach_pointwise 操作，针对列表元素进行减法操作
register_foreach_pointwise(aten._foreach_sub.List, sub)
# 注册 foreach_pointwise 操作，针对标量元素进行减法操作
register_foreach_pointwise(aten._foreach_sub.Scalar, sub)
# 注册 foreach_pointwise 操作，针对默认元素进行取反操作
register_foreach_pointwise(aten._foreach_neg.default, neg)
# 注册 foreach_pointwise 操作，针对默认元素进行取绝对值操作
register_foreach_pointwise(aten._foreach_abs.default, abs)
# 注册 foreach_pointwise 操作，针对标量元素进行幂操作
register_foreach_pointwise(aten._foreach_pow.Scalar, pow)
# 注册 foreach_pointwise 操作，针对标量和张量元素进行幂操作
register_foreach_pointwise(aten._foreach_pow.ScalarAndTensor, pow)
# 注册 foreach_pointwise 操作，针对列表元素进行除法操作
foreach_div_list = register_foreach_pointwise(aten._foreach_div.List, div)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_div.Scalar 操作到 div 函数
foreach_div_scalar = register_foreach_pointwise(aten._foreach_div.Scalar, div)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_sqrt 操作到 sqrt 函数
register_foreach_pointwise(aten._foreach_sqrt, sqrt)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_maximum.List 操作到 maximum 函数
register_foreach_pointwise(aten._foreach_maximum.List, maximum)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_maximum.Scalar 操作到 maximum 函数
register_foreach_pointwise(aten._foreach_maximum.Scalar, maximum)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_minimum.List 操作到 minimum 函数
register_foreach_pointwise(aten._foreach_minimum.List, minimum)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_minimum.Scalar 操作到 minimum 函数
register_foreach_pointwise(aten._foreach_minimum.Scalar, minimum)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_clamp_min.List 操作到 maximum 函数
register_foreach_pointwise(aten._foreach_clamp_min.List, maximum)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_clamp_min.Scalar 操作到 maximum 函数
register_foreach_pointwise(aten._foreach_clamp_min.Scalar, maximum)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_clamp_max.List 操作到 minimum 函数
register_foreach_pointwise(aten._foreach_clamp_max.List, minimum)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_clamp_max.Scalar 操作到 minimum 函数
register_foreach_pointwise(aten._foreach_clamp_max.Scalar, minimum)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_reciprocal 操作到 reciprocal 函数
register_foreach_pointwise(aten._foreach_reciprocal, reciprocal)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_sign 操作到 sign 函数
register_foreach_pointwise(aten._foreach_sign, sign)
# 使用 register_foreach_pointwise 函数注册 aten._foreach_copy 操作到 copy 函数
register_foreach_pointwise(aten._foreach_copy, copy)


# 将 outplace_aten_op 映射到 aten_op，以及将 aten_op 添加到 inplace_foreach_ops 中
# 用于生成 inplace 版本的 foreach 操作
def register_foreach_inplace(aten_op, outplace_aten_op, outplace_op):
    inplaceable_foreach_ops[outplace_aten_op] = aten_op
    inplace_foreach_ops.add(aten_op)

    # 定义一个函数 fn，用于执行 outplace_op 操作，然后对结果进行 in-place 修改
    def fn(*args, **kwargs):
        # 执行 outplace_op 操作获取结果
        results = outplace_op(*args, **kwargs)
        mut_results = []
        # 遍历输入的第一个参数 args[0] 和对应的结果 results，执行 mutate_to 操作
        for arg, result in zip(args[0], results):
            mut_results.append(mutate_to(arg, result, unsafe_alias=True))

        return mut_results

    # 使用 _register_foreach_lowering 函数注册 aten_op 和 fn 函数的关系
    _register_foreach_lowering(aten_op, fn)


# 使用 register_foreach_inplace 函数注册 in-place 版本的 aten._foreach_add_.List 操作到 foreach_add_list 函数
register_foreach_inplace(
    aten._foreach_add_.List, aten._foreach_add.List, foreach_add_list
)
# 使用 register_foreach_inplace 函数注册 in-place 版本的 aten._foreach_add_.Scalar 操作到 foreach_add_scalar 函数
register_foreach_inplace(
    aten._foreach_add_.Scalar, aten._foreach_add.Scalar, foreach_add_scalar
)
# 使用 register_foreach_inplace 函数注册 in-place 版本的 aten._foreach_mul_.List 操作到 foreach_mul_list 函数
register_foreach_inplace(
    aten._foreach_mul_.List, aten._foreach_mul.List, foreach_mul_list
)
# 使用 register_foreach_inplace 函数注册 in-place 版本的 aten._foreach_mul_.Scalar 操作到 foreach_mul_scalar 函数
register_foreach_inplace(
    aten._foreach_mul_.Scalar, aten._foreach_mul.Scalar, foreach_mul_scalar
)
# 使用 register_foreach_inplace 函数注册 in-place 版本的 aten._foreach_div_.List 操作到 foreach_div_list 函数
register_foreach_inplace(
    aten._foreach_div_.List, aten._foreach_div.List, foreach_div_list
)
# 使用 register_foreach_inplace 函数注册 in-place 版本的 aten._foreach_div_.Scalar 操作到 foreach_div_scalar 函数
register_foreach_inplace(
    aten._foreach_div_.Scalar, aten._foreach_div.Scalar, foreach_div_scalar
)


# 使用 register_lowering 函数注册 aten.add_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.add_, add)
# 使用 register_lowering 函数注册 aten.bitwise_and_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.bitwise_and_, bitwise_and)
# 使用 register_lowering 函数注册 aten.bitwise_left_shift_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.bitwise_left_shift_, bitwise_left_shift)
# 使用 register_lowering 函数注册 aten.bitwise_not_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.bitwise_not_, bitwise_not)
# 使用 register_lowering 函数注册 aten.bitwise_or_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.bitwise_or_, bitwise_or)
# 使用 register_lowering 函数注册 aten.bitwise_right_shift_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.bitwise_right_shift_, bitwise_right_shift)
# 使用 register_lowering 函数注册 aten.bitwise_xor_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.bitwise_xor_, bitwise_xor)
# 使用 register_lowering 函数注册 aten.mul_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.mul_, mul)
# 使用 register_lowering 函数注册 aten.div_.Tensor 操作，生成对应的 in-place 版本函数
register_inplace(aten.div_.Tensor, div)
# 使用 register_lowering 函数注册 aten.div_.Tensor_mode 操作，生成对应的 in-place 版本函数
register_inplace(aten.div_.Tensor_mode, div_mode)
# 使用 register_lowering 函数注册 aten.logical_and_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.logical_and_, logical_and)
# 使用 register_lowering 函数注册 aten.logical_not_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.logical_not_, logical_not)
# 使用 register_lowering 函数注册 aten.logical_or_ 操作，生成对应的 in-place 版本函数
register_inplace(aten.logical_or_, logical_or)
register_inplace(aten.logical_xor_, logical_xor)
# 注册 inplace 操作，将 torch 的 logical_xor_ 函数映射到 logical_xor 函数

register_inplace(aten.sub_, sub)
# 注册 inplace 操作，将 torch 的 sub_ 函数映射到 sub 函数

register_inplace(aten.relu_, relu)
# 注册 inplace 操作，将 torch 的 relu_ 函数映射到 relu 函数

register_inplace(aten.sigmoid_, sigmoid)
# 注册 inplace 操作，将 torch 的 sigmoid_ 函数映射到 sigmoid 函数


register_lowering(aten.__and__)(bitwise_and)
# 注册 lowering 函数，将 torch 的 __and__ 操作映射到 bitwise_and 函数

register_lowering(aten.__lshift__)(bitwise_left_shift)
# 注册 lowering 函数，将 torch 的 __lshift__ 操作映射到 bitwise_left_shift 函数

register_lowering(aten.__or__)(bitwise_or)
# 注册 lowering 函数，将 torch 的 __or__ 操作映射到 bitwise_or 函数

register_lowering(aten.__rshift__)(bitwise_right_shift)
# 注册 lowering 函数，将 torch 的 __rshift__ 操作映射到 bitwise_right_shift 函数

register_lowering(aten.__xor__)(bitwise_xor)
# 注册 lowering 函数，将 torch 的 __xor__ 操作映射到 bitwise_xor 函数


register_inplace(aten.__iand__, aten.__and__)
# 注册 inplace 操作，将 torch 的 __iand__ 操作映射到 torch 的 __and__ 函数

register_inplace(aten.__ilshift__, aten.__lshift__)
# 注册 inplace 操作，将 torch 的 __ilshift__ 操作映射到 torch 的 __lshift__ 函数

register_inplace(aten.__ior__, aten.__or__)
# 注册 inplace 操作，将 torch 的 __ior__ 操作映射到 torch 的 __or__ 函数

register_inplace(aten.__irshift__, aten.__rshift__)
# 注册 inplace 操作，将 torch 的 __irshift__ 操作映射到 torch 的 __rshift__ 函数

register_inplace(aten.__ixor__, aten.__xor__)
# 注册 inplace 操作，将 torch 的 __ixor__ 操作映射到 torch 的 __xor__ 函数


@register_lowering(aten.sym_constrain_range)
# 注册 lowering 函数，将 torch 的 sym_constrain_range 操作映射到 sym_constrain_range 函数
def sym_constrain_range(a, min=None, max=None):
    return None


@register_lowering(aten.sym_size.int)
# 注册 lowering 函数，将 torch 的 sym_size.int 操作映射到 sym_size 函数
def sym_size(a, dim):
    val = V.graph.current_node.meta["val"]
    # Note [Can val be an int?]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # In principle, someone could construct an FX graph where
    # a call to size/stride has a val that is a plain int (not
    # SymInt).  However, we will maintain the invariant that
    # this is not possible: if you are constructing an FX graph
    # where there is a call to size/stride that returns an
    # int, but you KNOW that int must always be a constant,
    # then you do not need trace that call at all (and just
    # constant propagate the integer as is.)
    assert isinstance(val, torch.SymInt)
    return val.node.expr


@register_lowering(aten.sym_stride.int)
# 注册 lowering 函数，将 torch 的 sym_stride.int 操作映射到 sym_stride 函数
def sym_stride(a, dim):
    val = V.graph.current_node.meta["val"]
    # See Note [Can val be an int?]
    assert isinstance(val, torch.SymInt)
    return val.node.expr


@register_lowering(aten.sym_numel)
# 注册 lowering 函数，将 torch 的 sym_numel 操作映射到 sym_numel 函数
def sym_numel(a):
    return a.get_numel()


for method, func in magic_methods.items():
    register_lowering(method_to_operator(method))(func)
    # 遍历 magic_methods 字典中的每个方法和对应的函数，将方法注册为 lowering 函数


@register_lowering(aten._foobar)
# 注册 lowering 函数，将 torch 的 _foobar 操作映射到 foobar 函数
def foobar(self, *args, **kwargs):
    raise NotImplementedError("Helpful for debugging")


@register_lowering(torch.ops._inductor_test.realize)
# 注册 lowering 函数，将 torch 的 ops._inductor_test.realize 操作映射到 _realize 函数
def _realize(x):
    x.realize()
    return clone(x)


@register_lowering(torch.ops.inductor.resize_storage_bytes_)
# 注册 lowering 函数，将 torch 的 ops.inductor.resize_storage_bytes_ 操作映射到 resize_storage_bytes_ 函数
def resize_storage_bytes_(variable, new_size):
    variable.realize()
    ir.ResizeStorageBytes(variable, new_size)
    return variable


@register_lowering(torch.ops.aten.set_.source_Tensor)
# 注册 lowering 函数，将 torch 的 ops.aten.set_.source_Tensor 操作映射到 set__source_tensor 函数
def set__source_tensor(self, source_tensor):
    self.realize()
    source_tensor.realize()
    return TensorBox.create(ir.SetSourceTensorKernel(self, source_tensor))


@register_lowering(torch.ops.aten.resize)
# 注册 lowering 函数，将 torch 的 ops.aten.resize 操作映射到 resize 函数
def resize(x, size, *, memory_format=None):
    assert isinstance(x, TensorBox)
    assert isinstance(size, (list, tuple))

    if memory_format is None:
        memory_format = torch.contiguous_format
    if memory_format == torch.preserve_format:
        raise RuntimeError(f"unsupported memory format: {memory_format}")

    if memory_format == torch.channels_last:
        assert len(size) == 4
    # 如果内存格式为 torch.channels_last_3d，则确保 size 的长度为 5
    if memory_format == torch.channels_last_3d:
        assert len(size) == 5

    # 获取张量 x 的元素总数
    old_numel = x.get_numel()
    # 获取张量 x 的数据类型
    dtype = x.get_dtype()
    # 获取张量 x 的设备信息
    device = x.get_device()

    # 如果 x 的数据是一个视图，将其解开以获取原始数据
    if isinstance(x.data, ir.BaseView):
        x.data = x.data.unwrap_view()

    # 如果启用了确定性算法且填充未初始化内存的选项开启
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory  # type: ignore[attr-defined]
    ):
        # 根据数据类型设置未初始化值
        if is_float_dtype(dtype):
            uninitalized_val = float("nan")
        elif is_integer_dtype(dtype):
            uninitalized_val = torch.iinfo(dtype).max
        else:
            uninitalized_val = True
    else:
        # 否则使用零作为未初始化值，这与 empty 函数的行为相同
        uninitalized_val = 0.0

    # 如果张量 x 的元素数量为 0，则返回一个以 uninitalized_val 填充的指定大小的张量
    if V.graph.sizevars.statically_known_equals(old_numel, 0):  # type: ignore[arg-type]
        return full(size, uninitalized_val, dtype=dtype, device=device)

    # 将张量 x 展平为一维数组 x_flat
    x_flat = as_strided(
        x,
        [
            old_numel,
        ],
        [
            1,
        ],
    )
    # 创建 x_flat 的加载器 flat_loader
    flat_loader = x_flat.make_loader()
    # 根据内存格式和指定大小生成输出张量的步长信息 out_stride
    out_stride = ir.FlexibleLayout.stride_ordered_for_memory_format(size, memory_format)
    # 创建固定布局的索引器 out_indexer
    out_indexer = ir.FixedLayout(device, dtype, size, out_stride).make_indexer()

    # 定义内部函数 inner_fn，用于处理每个索引
    def inner_fn(idx):
        # 使用 out_indexer 将索引转换为平铺数组的索引
        flat_index = out_indexer(idx)
        # 将 flat_index 表达为 torch.int64 类型的索引表达式
        flat_index_expr = ops.index_expr(flat_index, torch.int64)
        # 创建限制表达式，限制 flat_index_expr 在 old_numel 内
        limit = ops.index_expr(old_numel, torch.int64)
        mask = ops.lt(flat_index_expr, limit)  # 创建掩码，标记有效索引
        # 使用 flat_loader 加载 flat_index 处的数据，根据掩码决定使用 uninitalized_val 填充无效数据
        return ops.masked(mask, lambda: flat_loader([flat_index]), uninitalized_val)

    # 创建 Pointwise 对象 out，使用 inner_fn 处理张量的每个索引
    out = Pointwise.create(
        device=device, dtype=dtype, inner_fn=inner_fn, ranges=list(size)
    )
    return out
# 导入 torch._higher_order_ops.auto_functionalize 模块中的 auto_functionalized 函数
from torch._higher_order_ops.auto_functionalize import auto_functionalized

# 将 auto_functionalized 函数传递给 make_fallback 函数，使其成为回退函数
make_fallback(auto_functionalized)


# 注册 triton_kernel_wrapper_mutation 作为降低函数的回调函数
@register_lowering(triton_kernel_wrapper_mutation)
def triton_kernel_wrap_(*, kernel_idx, constant_args_idx, grid, kwargs):
    # 从 torch._higher_order_ops.triton_kernel_wrap 模块导入 kernel_side_table
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    # 从 kernel_side_table 获取 constant_args_idx 对应的常量参数
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)
    
    # 创建一个 UserDefinedTritonKernel 实例，使用指定的参数
    ir.UserDefinedTritonKernel(
        kernel_idx=kernel_idx,
        grid=grid,
        kernel_args={**kwargs, **constant_args},
    )
    
    # 返回一个字典，其中包含所有值为 TensorBox 类型的 kwargs 的键值对
    return {key: val for key, val in kwargs.items() if isinstance(val, TensorBox)}


# 注册 triton_kernel_wrapper_functional 作为降低函数的回调函数
@register_lowering(triton_kernel_wrapper_functional)
def triton_kernel_wrap(
    *, kernel_idx, constant_args_idx, grid, kwargs, tensors_to_clone
):
    # 创建一个新的空字典 new_kwargs
    new_kwargs = {}
    
    # 遍历 kwargs 中的键值对
    for name, value in kwargs.items():
        # 检查值是否为 ir.TensorBox 类型
        if isinstance(value, ir.TensorBox):
            x = value.data
            has_non_rv_views = False
            while isinstance(x, ir.BaseView):
                if not isinstance(x, ir.ReinterpretView):
                    has_non_rv_views = True
                    break
                x = x.data
            if has_non_rv_views:
                # 如果存在非 ReinterpretView 的视图，将其实现为 ReinterpretView
                value = ir.TensorBox(ir.ExternKernel.realize_input(value))
            if name in tensors_to_clone:
                # 如果名称存在于 tensors_to_clone 中，克隆并保留 ReinterpretView
                value = clone_preserve_reinterpret_view(value)
        # 将处理后的值存入 new_kwargs 字典
        new_kwargs[name] = value

    # 调用 triton_kernel_wrap 函数，使用更新后的参数
    return triton_kernel_wrap_(
        kernel_idx=kernel_idx,
        constant_args_idx=constant_args_idx,
        grid=grid,
        kwargs=new_kwargs,
    )


# 注册 torch.ops.higher_order.cond 作为降低函数的回调函数
@register_lowering(torch.ops.higher_order.cond)
def cond(pred, true_fn, false_fn, operands):
    # 检查 pred 和 operands 中是否有任何一个是 triton 类型
    if is_triton(pred) or any(map(is_triton, operands)):
        msg = "control flow operator: torch.cond."
        if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
            msg = f"{msg} Found from : \n {stack_trace}"
        # 禁用 cudagraphs 的原因记录为 msg
        V.graph.disable_cudagraphs_reason = msg

    # 创建一个 ir.Conditional 实例，并返回其结果列表，每个元素为 TensorBox 类型
    result = ir.Conditional.create(pred, true_fn, false_fn, operands)
    return list(map(TensorBox.create, result))


# 注册 torch.ops.higher_order.while_loop 作为降低函数的回调函数
@register_lowering(torch.ops.higher_order.while_loop)
def while_loop(cond_fn, body_fn, carried_inputs, additional_inputs):
    # 检查 carried_inputs 和 additional_inputs 中是否有任何一个是 triton 类型
    if any(map(is_triton, carried_inputs + additional_inputs)):
        msg = "control flow operator: torch.while_loop."
        if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
            msg = f"{msg} Found from : \n {stack_trace}"
        # 禁用 cudagraphs 的原因记录为 msg
        V.graph.disable_cudagraphs_reason = msg

    # 创建一个 ir.WhileLoop 实例，并返回其结果列表，每个元素为 TensorBox 类型
    result = ir.WhileLoop.create(cond_fn, body_fn, carried_inputs, additional_inputs)
    return list(map(TensorBox.create, result))
@register_lowering(associative_scan_op, type_promotion_kind=None)
# 注册一个降低操作的函数，用于关联扫描操作，并指定类型提升的种类为None
def associative_scan(combine_fn: ir.Subgraph, input, dim: int):
    from .subgraph_lowering import InputDescriptor, lower_pointwise_subgraph

    # 创建输入描述符列表，用于降低子图
    subgraph_inputs = [
        InputDescriptor(dtype=x.get_dtype(), device=x.get_device())
        for x in itertools.chain(input, input)
    ]

    # 使用lower_pointwise_subgraph函数对combine_fn进行降低处理
    lowered_combine_fn = lower_pointwise_subgraph(combine_fn, subgraph_inputs)

    # 定义一个包装后的combine_fn函数，用于调用降低后的函数
    def wrapped_combine_fn(lhs, rhs):
        return lowered_combine_fn(
            *pytree.tree_leaves(lhs),
            *pytree.tree_leaves(rhs),
        )

    # 生成扫描操作的参数kwargs
    kwargs = _make_scan_inner(input[0], axis=dim, dtype=None)
    kwargs["dtypes"] = tuple(x.get_dtype() for x in input)
    kwargs["inner_fns"] = tuple(x.make_loader() for x in input)

    # 创建一个Scan对象，并指定combine_fn为wrapped_combine_fn
    result = ir.Scan.create(**kwargs, combine_fn=wrapped_combine_fn)

    # 如果结果为None，则抛出运行时异常
    if result[0] is None:
        raise RuntimeError("Unable to generate code for associative_scan op")

    # 返回结果
    return result


@register_lowering(torch.ops.prims._sink_tokens.default)
# 注册一个降低操作的函数，处理torch.ops.prims._sink_tokens.default操作
def _sink_tokens(tokens):
    # 返回None
    return None


@register_lowering(torch.ops.higher_order.with_effects)
# 注册一个降低操作的函数，处理torch.ops.higher_order.with_effects操作
def with_effects(token, op, *args, **kwargs):
    # 创建一个EffectfulKernel对象
    result = ir.EffectfulKernel.create(op, *args, **kwargs)

    # 导入get_effect_key函数
    from torch._higher_order_ops.effects import get_effect_key

    # 获取操作的效果类型
    effect_type = get_effect_key(op, args, kwargs)
    assert effect_type is not None

    # 获取effect_type对应的effectful_kernel对象
    effectful_kernel = V.graph.effectful_ops[effect_type]

    # 如果result为None，则返回(effectful_kernel,)
    if result is None:
        return (effectful_kernel,)

    # 将result中的TensorBox对象转换为ir.MultiOutput对象
    result = pytree.tree_map_only(ir.MultiOutput, TensorBox.create, result)

    # 如果result不是列表或元组，则返回(effectful_kernel, result)
    if not isinstance(result, (list, tuple)):
        return (effectful_kernel, result)
    else:
        # 否则返回(effectful_kernel, *result)
        return (effectful_kernel, *result)


try:
    import torch.distributed._functional_collectives

    _c10d_functional = torch.ops._c10d_functional

    @register_lowering(_c10d_functional.all_reduce)
    # 注册一个降低操作的函数，处理torch.ops._c10d_functional.all_reduce操作
    def _all_reduce(inp, reduce_op, group_name):
        # 复制输入的inp对象
        inp = clone(inp)

        # 创建一个原地操作的CollectiveKernel对象
        ir._CollectiveKernel.create_inplace(
            _c10d_functional.all_reduce_.default, inp, reduce_op, group_name
        )

        # 返回inp对象
        return inp

    @register_lowering(_c10d_functional.all_reduce_)
    # 注册一个降低操作的函数，处理torch.ops._c10d_functional.all_reduce_操作
    def _all_reduce_(inp, reduce_op, group_name):
        # 创建一个原地操作的CollectiveKernel对象
        ir._CollectiveKernel.create_inplace(
            _c10d_functional.all_reduce_.default, inp, reduce_op, group_name
        )

        # 返回inp对象
        return inp

    @register_lowering(_c10d_functional.all_reduce_coalesced)
    # 注册一个降低操作的函数，处理torch.ops._c10d_functional.all_reduce_coalesced操作
    def _all_reduce_coalesced(inputs, reduce_op, group_name):
        # 复制输入的inputs列表中的每个inp对象
        inputs = [clone(inp) for inp in inputs]

        # 创建一个原地操作的CollectiveKernel对象
        ir._CollectiveKernel.create_inplace(
            _c10d_functional.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )

        # 返回inputs列表
        return inputs

    @register_lowering(_c10d_functional.all_reduce_coalesced_)
    # 注册一个降低操作的函数，处理torch.ops._c10d_functional.all_reduce_coalesced_操作
    def _all_reduce_coalesced_(inputs, reduce_op, group_name):
        # 复制输入的inputs列表中的每个inp对象
        inputs = [clone(inp) for inp in inputs]

        # 创建一个原地操作的CollectiveKernel对象
        ir._CollectiveKernel.create_inplace(
            _c10d_functional.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )

        # 返回inputs列表
        return inputs
    def _all_reduce_coalesced_(inputs, reduce_op, group_name):
        # 创建一个 inplace 操作的 CollectiveKernel，用于执行 all_reduce_coalesced 操作
        ir._CollectiveKernel.create_inplace(
            _c10d_functional.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )
        # 返回输入的数据，此处实现的操作是原位操作
        return inputs

    @register_lowering(_c10d_functional.all_gather_into_tensor)
    def _all_gather_into_tensor(inp, group_size, group_name):
        # 创建一个 TensorBox 对象，通过 out-of-place CollectiveKernel 执行 all_gather_into_tensor 操作
        return ir.TensorBox.create(
            ir._CollectiveKernel.create_out_of_place(
                _c10d_functional.all_gather_into_tensor.default,
                inp,
                group_size,
                group_name,
            )
        )

    @register_lowering(_c10d_functional.all_gather_into_tensor_coalesced)
    def _all_gather_into_tensor_coalesced(inputs, group_size, group_name):
        # 对输入的每个元素应用 TensorBox.create，并通过 out-of-place CollectiveKernel 执行 all_gather_into_tensor_coalesced 操作
        return pytree.tree_map(
            ir.TensorBox.create,
            ir._CollectiveKernel.create_out_of_place(
                _c10d_functional.all_gather_into_tensor_coalesced.default,
                inputs,
                group_size,
                group_name,
            ),
        )

    @register_lowering(_c10d_functional.reduce_scatter_tensor)
    def _reduce_scatter_tensor(inp, reduce_op, group_size, group_name):
        # 创建一个 TensorBox 对象，通过 out-of-place CollectiveKernel 执行 reduce_scatter_tensor 操作
        return ir.TensorBox.create(
            ir._CollectiveKernel.create_out_of_place(
                _c10d_functional.reduce_scatter_tensor.default,
                inp,
                reduce_op,
                group_size,
                group_name,
            )
        )

    @register_lowering(_c10d_functional.reduce_scatter_tensor_coalesced)
    def _reduce_scatter_tensor_coalesced(inputs, reduce_op, group_size, group_name):
        # 对输入的每个元素应用 TensorBox.create，并通过 out-of-place CollectiveKernel 执行 reduce_scatter_tensor_coalesced 操作
        return pytree.tree_map(
            ir.TensorBox.create,
            ir._CollectiveKernel.create_out_of_place(
                _c10d_functional.reduce_scatter_tensor_coalesced.default,
                inputs,
                reduce_op,
                group_size,
                group_name,
            ),
        )

    @register_lowering(_c10d_functional.all_to_all_single)
    def _all_to_all_single(inp, output_split_sizes, input_split_sizes, group_name):
        # 创建一个 TensorBox 对象，通过 out-of-place CollectiveKernel 执行 all_to_all_single 操作
        return ir.TensorBox.create(
            ir._CollectiveKernel.create_out_of_place(
                _c10d_functional.all_to_all_single.default,
                inp,
                output_split_sizes,
                input_split_sizes,
                group_name,
            )
        )

    @register_lowering(_c10d_functional.broadcast)
    def _broadcast(inp, src, group_name):
        # 克隆输入的张量 inp，并在 CollectiveKernel 中执行 inplace broadcast 操作
        inp = clone(inp)
        ir._CollectiveKernel.create_inplace(
            _c10d_functional.broadcast_.default, inp, src, group_name
        )
        # 返回 inplace 操作后的张量 inp
        return inp

    @register_lowering(_c10d_functional.broadcast_)
    def _broadcast_(inp, src, group_name):
        # 在 CollectiveKernel 中执行 inplace broadcast 操作
        ir._CollectiveKernel.create_inplace(
            _c10d_functional.broadcast_.default, inp, src, group_name
        )
        # 返回 inplace 操作后的输入张量 inp
        return inp

    @register_lowering(_c10d_functional.wait_tensor)
    # 注册一个降低的函数用于 wait_tensor 操作，但此处代码中没有给出具体的实现
    # 定义一个名为 _wait_tensor 的函数，接受一个参数 inp
    def _wait_tensor(inp):
        # 调用 ir._WaitKernel.create_wait 方法，创建一个等待操作，使用 _c10d_functional.wait_tensor.default 作为参数
        ir._WaitKernel.create_wait(_c10d_functional.wait_tensor.default, inp)
        # 返回输入参数 inp
        return inp

    # 使用装饰器 @register_lowering 注册下面的函数作为 torch.ops._dtensor.shard_dim_alltoall 的降低函数
    @register_lowering(torch.ops._dtensor.shard_dim_alltoall)
    # 定义一个名为 _shard_dim_alltoall 的函数，接受四个参数 inp, gather_dim, shard_dim, group_name
    def _shard_dim_alltoall(inp, gather_dim, shard_dim, group_name):
        # 创建一个 TensorBox 对象，使用 ir._CollectiveKernel.create_out_of_place 方法生成的输出
        return ir.TensorBox.create(
            # 调用 ir._CollectiveKernel.create_out_of_place 方法，创建一个离位操作，使用 torch.ops._dtensor.shard_dim_alltoall.default 作为操作
            ir._CollectiveKernel.create_out_of_place(
                torch.ops._dtensor.shard_dim_alltoall.default,
                inp,
                gather_dim,
                shard_dim,
                group_name,
            )
        )
# 处理 AttributeError 或 ImportError 异常情况
except (AttributeError, ImportError):
    # 记录日志，说明分布式集合的感应器依赖于构建 torch.distributed
    log.info(
        "Inductor support for distributed collectives depends on building torch.distributed"
    )

# 导入并加载 kernel/* 中定义的下降函数
from . import kernel

# 导入子模块 kernel
import_submodule(kernel)

# 导入并注册量化下降函数
from . import quantized_lowerings

# 注册量化操作
quantized_lowerings.register_quantized_ops()

# 注册不带权重量化矩阵乘法操作
quantized_lowerings.register_woq_mm_ops()

# 导入并注册 MKLDNN 下降函数
from . import mkldnn_lowerings

# 注册 OneDNN 融合操作
mkldnn_lowerings.register_onednn_fusion_ops()

# 导入并注册 jagged 下降函数
from . import jagged_lowerings

# 注册 jagged 操作
jagged_lowerings.register_jagged_ops()
```