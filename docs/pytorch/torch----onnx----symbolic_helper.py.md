# `.\pytorch\torch\onnx\symbolic_helper.py`

```
# mypy: allow-untyped-defs
# 引入未类型化定义允许
from __future__ import annotations
# 导入未来的注解支持

import functools
# 导入 functools 模块，用于高阶函数操作
import inspect
# 导入 inspect 模块，用于解析 Python 对象结构和源代码
import math
# 导入 math 模块，提供数学函数
import sys
# 导入 sys 模块，提供与 Python 解释器交互的函数
import typing
# 导入 typing 模块，提供类型相关的支持
import warnings
# 导入 warnings 模块，用于警告控制

from typing import (
    Any,
    Callable,
    List,
    Literal,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
# 从 typing 模块中导入多种类型注解

import torch
# 导入 PyTorch 库
import torch._C._onnx as _C_onnx
# 导入 PyTorch ONNX 相关的 C 扩展模块
from torch import _C
# 导入 PyTorch 的 C 扩展模块

# Monkey-patch graph manipulation methods on Graph, used for the ONNX symbolics
# 导入用于 ONNX 符号计算的图操作方法的 Monkey-patch
from torch.onnx import _constants, _type_utils, errors, utils
# 从 torch.onnx 模块导入相关工具和错误处理
from torch.onnx._globals import GLOBALS
# 导入 ONNX 全局变量
from torch.onnx._internal import _beartype, jit_utils
# 导入内部工具和类型检查相关模块
from torch.types import Number
# 导入 Number 类型

# ---------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------

_ValueDescriptor = Literal[
    "v",
    "i",
    "is",
    "f",
    "fs",
    "b",
    "s",
    "t",
    "none",
]
# _ValueDescriptor 定义了多种字面值类型的别名，用于描述值的类型特征

@_beartype.beartype
# 使用 beartype 进行类型检查装饰器
def _parse_arg(
    value,
    desc: _ValueDescriptor,
    arg_name: Optional[str] = None,
    node_name: Optional[str] = None,
):
    # 解析参数函数，根据描述符处理输入的值
    if desc == "none":
        return value
    # 如果描述符为 'none'，直接返回值

    if desc == "v" or not _is_value(value):
        return value
    # 如果描述符为 'v' 或者值不是有效值，则直接返回值

    node = value.node()
    # 获取值的节点对象

    if node.mustBeNone():
        return None
    # 如果节点必须为 None，则返回 None

    if node.kind() == "onnx::Constant":
        # 如果节点类型为 'onnx::Constant'
        node_val = _node_get(node, "value")
        # 获取节点的值

        if desc == "i":
            return int(node_val)
        elif desc == "f":
            return float(node_val)
        elif desc == "b":
            return bool(node_val)
        elif desc == "s":
            return str(node_val)
        elif desc == "t":
            return node_val
        elif desc == "is":
            return [int(v) for v in node_val]
        elif desc == "fs":
            return [float(v) for v in node_val]
        else:
            raise errors.SymbolicValueError(
                f"ONNX symbolic does not understand the Constant node '{node}' "
                f"specified with descriptor '{desc}'.",
                value,
            )
        # 根据描述符处理 Constant 节点的值，并返回相应的结果

    elif node.kind() == "prim::ListConstruct":
        # 如果节点类型为 'prim::ListConstruct'
        if desc == "is":
            # 如果描述符为 'is'
            for v in node.inputs():
                element_node = v.node()
                # 获取每个输入节点
                if element_node.kind() != "onnx::Constant":
                    raise errors.SymbolicValueError(
                        f"Failed to export a node '{element_node}' "
                        f"(in list node {node}) "
                        f"because it is not constant. "
                        f"Please try to make things (e.g. kernel sizes) static if possible.",
                        value,
                    )
                # 如果元素节点不是常量，则抛出错误

            return [int(_node_get(v.node(), "value")) for v in value.node().inputs()]
            # 返回列表构造节点的整数值列表
        else:
            raise errors.SymbolicValueError(
                f"ONNX symbolic does not know how to unpack the ListConstruct node that "
                f"is not a list of integers: '{node}'",
                value,
            )
        # 抛出错误，说明 ONNX 符号无法处理非整数列表的 ListConstruct 节点
    # 如果参数 arg_name 或 node_name 为 None，则抛出符号值错误异常
    if arg_name is None or node_name is None:
        raise errors.SymbolicValueError(
            f"Expected node type 'onnx::Constant', got '{node.kind()}'.",
            value,
        )

    # 抛出符号值错误异常，指示预期的节点类型为 'onnx::Constant'，
    # 但实际上得到了节点的具体类型（通过 node.kind() 获取），用于参数 '{arg_name}' 的节点 '{node_name}'。
    raise errors.SymbolicValueError(
        "Expected node type 'onnx::Constant' "
        f"for argument '{arg_name}' of node '{node_name}', got '{node.kind()}'.",
        value,
    )
# 应用装饰器 beartype 对 _node_get 函数进行类型检查和注解
@_beartype.beartype
def _node_get(node: _C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    # 断言 node 是 _C.Node 类型的实例
    assert isinstance(node, _C.Node)
    # 获取 key 对应的节点的类型
    sel = node.kindOf(key)
    # 返回节点 key 对应的属性值
    return getattr(node, sel)(key)


# 应用装饰器 beartype 对 _is_onnx_constant 函数进行类型检查和注解
@_beartype.beartype
def _is_onnx_constant(value: _C.Value):
    """Whether a Value is an ONNX constant."""
    # 返回 value 对应的节点类型是否为 "onnx::Constant"
    return value.node().kind() == "onnx::Constant"


# 应用装饰器 beartype 对 _maybe_get_const 函数进行类型检查和注解
@_beartype.beartype
def _maybe_get_const(
    value: Optional[Union[_C.Value, torch.Tensor, Number, Sequence]],
    descriptor: _ValueDescriptor,
):
    # NOTE: prim::Constant at this stage usually means something not compatible in ONNX,
    # otherwise it'd be converted to onnx::Constant
    # TODO(justinchuby): Replace insinstance with _is_value once we figure out mypy
    # 如果 value 是 _C.Value 类型且是 ONNX 常量，则调用 _parse_arg 处理
    if isinstance(value, _C.Value) and _is_onnx_constant(value):
        return _parse_arg(value, descriptor)
    # 否则返回原始的 value
    return value


# 应用装饰器 beartype 对 _maybe_get_scalar 函数进行类型检查和注解
@_beartype.beartype
def _maybe_get_scalar(value):
    # 获取 value 的常量表示
    value_t = _maybe_get_const(value, "t")
    # 如果 value_t 是 torch.Tensor 类型且形状为 ()，则返回 value_t
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    # 否则返回原始的 value
    return value


# 应用装饰器 beartype 对 _get_const 函数进行类型检查和注解
@_beartype.beartype
def _get_const(value, desc, arg_name):
    # 如果 value 不是常量，则抛出错误
    if not _is_constant(value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected a constant value of the '{arg_name}' argument, "
            f"got '{value}'",
            value,
        )
    # 否则调用 _parse_arg 处理常量值，并返回结果
    return _parse_arg(value, desc)


# 应用装饰器 beartype 对 _unpack_list 函数进行类型检查和注解
@_beartype.beartype
def _unpack_list(list_value: _C.Value) -> List[_C.Value]:
    # 获取 list_value 对应的节点
    list_node = list_value.node()
    # 如果节点类型不是 'prim::ListConstruct'，则抛出错误
    if list_node.kind() != "prim::ListConstruct":
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected node type prim::ListConstruct, "
            f"got '{list_node}'.",
            list_value,
        )
    # 返回列表节点的输入列表
    return list(list_node.inputs())


# 应用装饰器 beartype 对 _unpack_tuple 函数进行类型检查和注解
@_beartype.beartype
def _unpack_tuple(tuple_value: _C.Value) -> Tuple[_C.Value, ...]:
    # 获取 tuple_value 对应的节点
    tuple_node = tuple_value.node()
    # 如果节点不是 'prim::TupleConstruct'，则抛出错误
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(
            f"ONNX symbolic expected node type 'prim::TupleConstruct', "
            f"got '{tuple_node.kind()}'.",
            tuple_value,
        )
    # 返回元组节点的输入元组
    return tuple(tuple_node.inputs())


# 应用装饰器 beartype 对 _unpack_quantized_tensor 函数进行类型检查和注解
@_beartype.beartype
def _unpack_quantized_tensor(tuple_value: _C.Value) -> Tuple[_C.Value, ...]:
    """Unpacks a quantized tensor into a tuple of tensor and scale/zero_point.
    Args:
        tuple_value: A tuple of tensor, scale, zero_point, and optionally axis.
    Returns:
        A tuple of tensor, scale, zero_point, and optionally axis.
    """
    # 获取 tuple_value 对应的节点
    tuple_node = tuple_value.node()
    # 量化张量表示为元组 (tensor, scale, zero_point, <axis>)
    # 检查传入的 tuple_value 是否符合 tuple 构造函数的要求，如果不是则抛出 SymbolicValueError 异常
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(
            # 构造异常信息，指示 ONNX 符号值期望 `{tuple_node}` 的输出为量化张量，
            # 可能是由于缺少对量化 `{tuple_node.kind()}` 的支持所导致。请在 {_constants.PYTORCH_GITHUB_ISSUES_URL} 上创建问题。
            f"ONNX symbolic expected the output of `{tuple_node}` to be a quantized "
            f"tensor. Is this likely due to missing support for quantized "
            f"`{tuple_node.kind()}`. Please create an issue on {_constants.PYTORCH_GITHUB_ISSUES_URL}",
            # 将引发异常的 tuple_value 作为异常的附加信息
            tuple_value,
        )
    # 将 tuple_node 的输入解包为元组 unpacked
    unpacked = tuple(tuple_node.inputs())
    # 确保解包后的元组长度为 3 或 4，否则引发断言错误
    assert len(unpacked) == 3 or len(unpacked) == 4
    # 返回解包后的元组 unpacked
    return unpacked
# 检查 list_value 是否是 prim::ListConstruct 的输出
# 这通常在 _unpack_list 之前调用，以确保列表可以被正确解包。
@_beartype.beartype
def _is_packed_list(list_value: Any) -> bool:
    # 返回一个布尔值，表示 list_value 是否是有效的 prim::ListConstruct 对象
    return _is_value(list_value) and list_value.node().kind() == "prim::ListConstruct"


@_beartype.beartype
def parse_args(*arg_descriptors: _ValueDescriptor):
    """一个装饰器，将来自 torch._C.Value 的参数转换为内置类型。

    例如:

    ```
    @parse_args('v', 'i', 'fs')
    foo(g, a, b, c):
        assert isinstance(a, torch._C.Value)
        assert isinstance(b, int)
        assert isinstance(c, list)
        assert isinstance(c[0], float)
    ```

    Args:
        arg_descriptors: 字符串列表，每个元素指定要转换的类型。有效的描述符包括:
            "v": 不转换，保持为 torch._C.Value 类型。
            "i": 整数
            "is": 整数列表
            "f": 浮点数
            "fs": 浮点数列表
            "b": 布尔值
            "s": 字符串
            "t": torch.Tensor
            "none": 变量未使用
    """
    def decorator(fn):
        # 将参数描述符存储在函数对象的属性中
        fn._arg_descriptors = arg_descriptors

        @functools.wraps(fn)
        def wrapper(g, *args, **kwargs):
            # 一些参数可能是可选的，因此长度可能较小
            FILE_BUG_MSG = (
                "If you believe this is not due to custom symbolic implementation within your code or "
                "an external library, please file an issue at "
                "https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml to report this bug."
            )
            # 断言确保参数描述符数量不少于参数数量
            assert len(arg_descriptors) >= len(args), (
                f"A mismatch between the number of arguments ({len(args)}) and "
                f"their descriptors ({len(arg_descriptors)}) was found at symbolic function '{fn.__name__}'. "
                f"{FILE_BUG_MSG}"
            )

            try:
                # 获取函数的签名和参数名列表（除去第一个 'g' 参数）
                sig = inspect.signature(fn)
                arg_names = list(sig.parameters.keys())[1:]
                fn_name = fn.__name__
            except Exception:
                # FIXME(justinchuby): 避免捕获 Exception。
                # 应捕获更具体的异常。
                # 如果出现异常，使用长度为参数数量的空列表作为参数名
                arg_names = [None] * len(args)  # type: ignore[list-item]
                fn_name = None
            # 解析每个参数，使用对应的描述符和参数名
            args = [
                _parse_arg(arg, arg_desc, arg_name, fn_name)  # type: ignore[method-assign]
                for arg, arg_desc, arg_name in zip(args, arg_descriptors, arg_names)
            ]
            # 仅支持 kwargs 中包含的单个 '_outputs' 参数
            assert len(kwargs) <= 1, (
                f"Symbolic function {fn.__name__}'s '**kwargs' can contain a single "
                f"key/value entry. "
                f"{FILE_BUG_MSG}"
            )

            if len(kwargs) == 1:
                # 断言确保 kwargs 中只有一个 '_outputs' 键
                assert "_outputs" in kwargs, (
                    f"Symbolic function {fn.__name__}'s '**kwargs' can only contain "
                    f"'_outputs' key at '**kwargs'. "
                    f"{FILE_BUG_MSG}"
                )
            # 调用原始函数，并返回其结果
            return fn(g, *args, **kwargs)

        # 返回装饰后的函数
        return wrapper

    # 返回装饰器函数
    return decorator
# 使用 @_beartype 装饰器来确保函数 quantized_args 被正确类型注解检查
@_beartype.beartype
# 定义 quantized_args 函数，接受一系列 bool 类型参数 arg_q_descriptors，
# 以及可选参数 scale（浮点数，默认为 None）、zero_point（整数，默认为 None）、quantize_output（布尔型，默认为 True）
def quantized_args(
    *arg_q_descriptors: bool,
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
    quantize_output: bool = True,
):
    """A decorator which extends support for quantized version of the base operator.

    Quantization is detected by examining the arguments that are annotated by
    `arg_q_descriptors`.

    If quantization is detected, the base operator symbolic function will be wrapped with
    argument de-quantization and output quantization.

    Otherwise, only the base symbolic function will be invoked.

    For example:

    ```
    @quantized_args(True, False)
    def foo(g, x, y):
        return x + y
    ```

    is equivalent to

    ```
    def q_foo(g, x, y):
        if is_quantized_tensor(x):
            x = dequantize(x)
            out = foo(g, x, y)
            return quantize(out)
        else:
            return foo(g, x, y)
    ```

    Args:
        arg_q_descriptors: A sequence of bool, where each element represents if the
          argument is QTensor for quantized version of this operator. It defaults
          to False for unspecified (variable length) arguments.
        scale: Quantized output scale. If None, derive from
          the first quantized input scale.
        zero_point: Quantized output zero point. If None,
          derive from the first quantized input zero point.
        quantize_output: If True, quantize the output of the base operator. Default is True
    """

    # 返回 decorator 函数
    return decorator


# 使用 @_beartype 装饰器确保函数 _scalar 被正确类型注解检查
@_beartype.beartype
# 定义 _scalar 函数，接受任意类型参数 x，返回一个可选的数字类型
def _scalar(x: Any) -> Optional[Number]:
    """Convert a scalar tensor into a Python value."""
    # 检查 x 是否为 torch.Tensor 类型且形状为 ()，如果是则返回其对应的 Python 值
    if isinstance(x, torch.Tensor) and x.shape == ():
        return x.item()
    # 否则返回 None
    return None


# 使用 @_beartype 装饰器确保函数 _if_scalar_type_as 被正确类型注解检查
@_beartype.beartype
# 定义 _if_scalar_type_as 函数，接受参数 self 和 tensor
def _if_scalar_type_as(self, tensor):
    """
    Convert self into the same type of tensor, as necessary.
    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    # 如果 self 是 _C.Value 类型，则直接返回 self
    if isinstance(self, _C.Value):
        return self

    # 获取 tensor 的标量类型
    scalar_type = _type_utils.JitScalarType.from_value(
        tensor, _type_utils.JitScalarType.UNDEFINED
    )
    # 如果标量类型不是 UNDEFINED，则根据标量类型进行处理
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        ty = scalar_type.scalar_name().lower()
        # 返回 self 的相应类型方法的调用结果
        return getattr(self, ty)()
    # 否则返回 self 本身
    return self


# 使用 @_beartype 装饰器确保函数 _is_none 被正确类型注解检查
@_beartype.beartype
# 定义 _is_none 函数，接受任意类型参数 x，返回布尔值
def _is_none(x: Any) -> bool:
    # 如果 x 是 None 或者 x 是 _C.Value 类型且其节点必须为 None，则返回 True，否则返回 False
    return x is None or (x.node().mustBeNone() if isinstance(x, _C.Value) else False)


# 使用 @_beartype 装饰器确保函数 _is_value 被正确类型注解检查
@_beartype.beartype
# 定义 _is_value 函数，接受任意类型参数 x，返回布尔值
def _is_value(x: Any) -> bool:
    # 返回 x 是否为 _C.Value 类型的判断结果
    return isinstance(x, _C.Value)


# 使用 @_beartype 装饰器确保函数 _is_constant 被正确类型注解检查
@_beartype.beartype
# 定义 _is_constant 函数，接受任意类型参数 value，返回布尔值
def _is_constant(value: Any) -> bool:
    # 如果 value 不是 _C.Value 类型或者其节点的类型是 'onnx::Constant' 或 'prim::Constant'，则返回 True，否则返回 False
    return not _is_value(value) or value.node().kind() in {
        "onnx::Constant",
        "prim::Constant",
    }


# 使用 @_beartype 装饰器确保函数 _is_tensor 被正确类型注解检查
@_beartype.beartype
# 定义 _is_tensor 函数，接受 _C.Value 类型参数 x，返回布尔值
def _is_tensor(x: _C.Value) -> bool:
    # 返回 x 的类型是否为 _C.TensorType 的子类型
    return x.type().isSubtypeOf(_C.TensorType.get())


# 定义未公开给 Python 的 _C.JitType 类型参数 jit_type 的转换函数，返回一个可选的 _C.ListType 类型
def _as_list_type(jit_type: _C.JitType) -> Optional[_C.ListType]:
    # 检查给定的 jit_type 是否为 _C.ListType 类型的实例
    if isinstance(jit_type, _C.ListType):
        # 如果是，则返回 jit_type
        return jit_type
    # 如果 jit_type 不是 _C.ListType 的实例，则返回 None
    return None
# Decorator to apply type validation using Beartype library to ensure `_is_list` function accepts `_C.Value` and returns a boolean
@_beartype.beartype
def _is_list(x: _C.Value) -> bool:
    return _as_list_type(x.type()) is not None

# Decorator to apply type validation using Beartype library to ensure `_is_tensor_list` function accepts `_C.Value` and returns a boolean
@_beartype.beartype
def _is_tensor_list(x: _C.Value) -> bool:
    # Retrieve the list type of x
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    # Check if each element type in the list is of type `_C.TensorType`
    return isinstance(x_type.getElementType(), _C.TensorType)

# Decorator to apply type validation using Beartype library to ensure `_is_scalar_list` function accepts `_C.Value` and returns a boolean
@_beartype.beartype
def _is_scalar_list(x: _C.Value) -> bool:
    """Checks if x is a scalar list, for example: List[float], List[int].

    Besides checking the type is ListType, we also check if the data type is
    a valid ONNX data type.
    """
    # Retrieve the list type of x
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    # Determine the scalar type from the value x and verify its compatibility with ONNX
    scalar_type = _type_utils.JitScalarType.from_value(x)
    return scalar_type.onnx_compatible()

# Decorator to apply type validation using Beartype library to ensure `_is_tuple_construct` function accepts `_C.Value` and returns a boolean
@_beartype.beartype
def _is_tuple_construct(x: _C.Value) -> bool:
    # Check if the node kind of x is "prim::TupleConstruct"
    return x.node().kind() == "prim::TupleConstruct"

# Function to check if x is a complex value based on its scalar type
@_beartype.beartype
def is_complex_value(x: _C.Value) -> bool:
    assert _is_value(x)  # Ensure x is a valid value
    # Check if the scalar type from value x is one of the complex types
    return _type_utils.JitScalarType.from_value(
        x, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.COMPLEX32,
        _type_utils.JitScalarType.COMPLEX64,
        _type_utils.JitScalarType.COMPLEX128,
    }

# Decorator to apply type validation using Beartype library to ensure `_get_tensor_rank` function accepts `_C.Value` and returns an optional integer
@_beartype.beartype
def _get_tensor_rank(x: _C.Value) -> Optional[int]:
    if not _is_tensor(x) or x.type() is None:
        return None
    # Cast x.type() to `_C.TensorType` and retrieve the rank (number of dimensions) of the tensor
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    return x_type.dim()

# Function to retrieve the sizes of a tensor, optionally allowing non-static sizes
@_beartype.beartype
def _get_tensor_sizes(x: _C.Value, allow_nonstatic: bool = True):
    if not _is_tensor(x) or x.type() is None:
        return None
    # Cast x.type() to `_C.TensorType` and retrieve varying sizes or static sizes based on the flag `allow_nonstatic`
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    if allow_nonstatic:
        # Return varying sizes of the tensor
        return x_type.varyingSizes()
    # Return static sizes of the tensor
    return x_type.sizes()

# Function to retrieve the size of a specific dimension of a tensor
@_beartype.beartype
def _get_tensor_dim_size(x: _C.Value, dim: int) -> Optional[int]:
    sizes = _get_tensor_sizes(x)
    # Return the size of the specified dimension `dim` if sizes are available
    return sizes[dim] if sizes else None

# Function to retrieve a dimension index for cross computation
@_beartype.beartype
def _get_dim_for_cross(x: _C.Value, dim: Optional[int]):
    if dim == -1:
        tensor_rank = _get_tensor_rank(x)
        assert tensor_rank is not None
        # Calculate and return the last dimension index using tensor rank
        return dim + tensor_rank
    # If dim is not given, find the first dimension index with size 3
    if dim is None:
        sizes = _get_tensor_sizes(x)
        assert sizes is not None
        for index, size in enumerate(sizes):
            if size is not None and size == 3:
                return index
    return dim  # Return the provided dimension index if specified

# Function to handle unimplemented operations
@_beartype.beartype
def _unimplemented(op: str, msg: str, value: Optional[_C.Value] = None) -> None:
    # For ONNX operator export type, raise an unsupported exception through `_onnx_unsupported`
    if GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX:
        _onnx_unsupported(f"{op}, {msg}", value)
# 定义一个函数，用于处理不支持的 ONNX 操作，会抛出异常并显示相应的错误消息
def _onnx_unsupported(op_name: str, value: Optional[_C.Value] = None) -> NoReturn:
    # 构建错误消息，指示不支持导出的操作名称，并提供 GitHub 请求支持或提交拉取请求的链接
    message = (
        f"Unsupported: ONNX export of operator {op_name}. "
        f"Please feel free to request support or submit a pull request "
        f"on PyTorch GitHub: {_constants.PYTORCH_GITHUB_ISSUES_URL}"
    )
    # 如果提供了值，并且值是 _C.Value 类型，抛出符号值错误异常
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    # 否则抛出 ONNX 导出器错误异常，显示错误消息
    raise errors.OnnxExporterError(message)


# 定义一个函数，用于处理不支持的 ONNX 操作集，会抛出异常并显示相应的错误消息
@_beartype.beartype
def _onnx_opset_unsupported(
    op_name: str,
    current_opset: int,
    supported_opset: int,
    value: Optional[_C.Value] = None,
) -> NoReturn:
    # 构建错误消息，指示不支持在指定操作集中导出的操作名称，并建议尝试支持的操作集版本
    message = (
        f"Unsupported: ONNX export of {op_name} in opset {current_opset}. "
        f"Please try opset version {supported_opset}."
    )
    # 如果提供了值，并且值是 _C.Value 类型，抛出符号值错误异常
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    # 否则抛出 ONNX 导出器错误异常，显示错误消息
    raise errors.OnnxExporterError(message)


# 定义一个函数，用于处理详细不支持的 ONNX 操作集，会抛出异常并显示相应的详细错误消息
@_beartype.beartype
def _onnx_opset_unsupported_detailed(
    op_name: str,
    current_opset: int,
    supported_opset: int,
    reason: str,
    value: Optional[_C.Value] = None,
) -> NoReturn:
    # 构建详细的错误消息，指示不支持在指定操作集中导出的操作名称和原因，并建议尝试支持的操作集版本
    message = (
        f"Unsupported: ONNX export of {op_name} in "
        f"opset {current_opset}. {reason}. Please try opset version {supported_opset}."
    )
    # 如果提供了值，并且值是 _C.Value 类型，抛出符号值错误异常
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(
            message,
            value,
        )
    # 否则抛出 ONNX 导出器错误异常，显示错误消息
    raise errors.OnnxExporterError(message)


# 定义一个函数，返回一个函数对象，用于表示在指定操作集中不支持的块列表
@_beartype.beartype
def _block_list_in_opset(name: str):
    # 定义一个内部函数，用于抛出 ONNX 导出器错误，指示未实现对指定操作集版本的块列表支持
    def symbolic_fn(*args, **kwargs):
        raise errors.OnnxExporterError(
            f"ONNX export failed on {name}, which is not implemented for opset "
            f"{GLOBALS.export_onnx_opset_version}. "
            "Try exporting with other opset versions."
        )
    # 返回内部函数对象
    return symbolic_fn


# 定义一个函数，用于尝试从参数中获取标量类型，返回获取到的标量类型对象或 None
@_beartype.beartype
def _try_get_scalar_type(*args) -> Optional[_type_utils.JitScalarType]:
    # 遍历所有参数，尝试获取每个参数的标量类型
    for arg in args:
        scalar_type = _type_utils.JitScalarType.from_value(
            arg, _type_utils.JitScalarType.UNDEFINED
        )
        # 如果找到非未定义的标量类型，返回该标量类型对象
        if scalar_type != _type_utils.JitScalarType.UNDEFINED:
            return scalar_type
    # 如果未找到任何标量类型，返回 None
    return None


# 定义一个函数，用于从给定的参数推断并返回推广后的标量类型
@_beartype.beartype
def _type_promote_from_values(*args) -> _type_utils.JitScalarType:
    # 定义未定义的标量类型常量
    undef = _type_utils.JitScalarType.UNDEFINED
    # 从所有参数中获取各自的标量类型，并存储在列表中
    jit_types = [_try_get_scalar_type(arg) for arg in args]
    # 如果没有任何标量类型被推断出来，返回未定义的标量类型
    if len(jit_types) == 0:
        return undef
    # 如果只有一个标量类型被推断出来，返回该标量类型
    if len(jit_types) == 1:
        return jit_types[0]
    # 否则，推广所有标量类型并返回推广后的标量类型
    new_dtype = jit_types[0].dtype()
    for t in jit_types:
        new_dtype = torch.promote_types(new_dtype, t.dtype())
    return _type_utils.JitScalarType.from_dtype(new_dtype)


# 定义一个函数，用于可能将值转换为指定的标量类型，根据条件判断是否需要转换
@_beartype.beartype
def _maybe_cast_to_type(
    g: jit_utils.GraphContext, value, jit_type: _type_utils.JitScalarType
):
    # 如果值的标量类型不等于指定的标量类型，执行可能的转换
    if (
        _type_utils.JitScalarType.from_value(value, _type_utils.JitScalarType.UNDEFINED)
        != jit_type
    )
    ):
        return g.op(
            "Cast",             # 在计算图 g 上执行类型转换操作
            value,              # 输入的值，需要进行类型转换
            to_i=jit_type.onnx_type(),  # 将 value 转换为 jit_type 对应的 ONNX 类型
        )
    return value   # 返回经过类型转换后的值
# 使用装饰器进行类型检查和修饰的函数定义，用于辅助选择操作
@_beartype.beartype
def _select_helper(g: jit_utils.GraphContext, self, dim, index, apply_reshape=True):
    # 获取 index 的常量标量值
    index_const = _maybe_get_scalar(index)
    # 获取 index 的张量秩
    index_dim = _get_tensor_rank(index)
    # 如果 index 不是常量标量，则将其转换为大小为1的常量张量
    if not _is_value(index_const):
        index = g.op("Constant", value_t=torch.LongTensor([index_const]))
    # 如果 index 是常量标量且需要应用 reshape 操作
    elif index_dim is not None and apply_reshape:
        # 如果 index 的秩为0，将其 reshape 成大小为1的张量
        if index_dim == 0:
            index = _reshape_helper(
                g, index, g.op("Constant", value_t=torch.LongTensor([1]))
            )
    
    # 获取 index 的标量类型
    index_scalar_type = _type_utils.JitScalarType.from_value(
        index, _type_utils.JitScalarType.UNDEFINED
    )
    # 如果 index 的标量类型不在 INT64 和 INT 中，则将其转换为 INT64 类型
    if index_scalar_type not in {
        _type_utils.JitScalarType.INT64,
        _type_utils.JitScalarType.INT,
    }:
        index = g.op("Cast", index, to_i=_C_onnx.TensorProtoDataType.INT64)
    
    # 执行 Gather 操作，根据 dim 维度从 self 中收集数据
    return g.op("Gather", self, index, axis_i=dim)


# 使用装饰器进行类型检查和修饰的函数定义，用于辅助切片操作
@_beartype.beartype
def _slice_helper(
    g: jit_utils.GraphContext,
    input,
    axes,
    starts,
    ends,
    steps=None,
):
    # 如果操作集版本小于等于9，则使用 symbol_opset9 中的 _slice 函数进行切片
    if g.opset <= 9:
        from torch.onnx.symbolic_opset9 import _slice as _slice9

        return _slice9(g, input, axes, starts, ends)
    # 否则使用 symbol_opset10 中的 _slice 函数进行切片
    else:
        from torch.onnx.symbolic_opset10 import _slice as _slice10

        return _slice10(g, input, axes, starts, ends, steps)


# 使用装饰器进行类型检查和修饰的函数定义，用于判断值是否为浮点数类型
@_beartype.beartype
def _is_fp(value) -> bool:
    return _type_utils.JitScalarType.from_value(
        value, _type_utils.JitScalarType.UNDEFINED
    ) in {
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.DOUBLE,
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.BFLOAT16,
    }


# 使用装饰器进行类型检查和修饰的函数定义，用于判断值是否为布尔类型
@_beartype.beartype
def _is_bool(value) -> bool:
    return _type_utils.JitScalarType.from_value(
        value, _type_utils.JitScalarType.UNDEFINED
    ) in {_type_utils.JitScalarType.BOOL}


# 使用装饰器进行类型检查和修饰的函数定义，用于生成包装的数字类型
@_beartype.beartype
def _generate_wrapped_number(g: jit_utils.GraphContext, scalar):
    """Creates a wrapped number based on https://github.com/pytorch/pytorch/issues/9515.

    A Tensor is a considered a "wrapped number" if it is
    auto-wrapped from a C++ or Python number type. Integer types are
    wrapped as 0-dim int64 tensors and floating-point types are
    wrapped as 0-dim double tensors.

    The input to this function is constant value. If the data type
    is a floating point type, it is converted to a 0-dim double
    tensor, else it is converted to a 0-dim tensor of its original type
    """
    # 断言 scalar 不是 torch.Tensor 类型
    assert not isinstance(scalar, torch.Tensor)
    # 如果 scalar 是 float 类型，返回一个常量张量，数据类型为 torch.double
    if isinstance(scalar, float):
        return g.op("Constant", value_t=torch.tensor(scalar, dtype=torch.double))
    # 否则返回一个常量张量，数据类型与 scalar 的原始类型相同
    return g.op("Constant", value_t=torch.tensor(scalar))


# 使用装饰器进行类型检查和修饰的函数定义，用于辅助排序操作
@_beartype.beartype
def _sort_helper(g: jit_utils.GraphContext, input, dim, decending=True, out=None):
    # 如果传入的 out 参数不为 None，则抛出未实现异常
    if out is not None:
        _unimplemented("Sort", "Out parameter is not supported")
    # 调用ONNX图操作（OP），获取输入张量的形状信息
    shape_ = g.op("Shape", input)
    # 调用ONNX图操作（OP），使用常量张量来收集指定维度的大小信息
    dim_size_ = g.op(
        "Gather",
        shape_,  # 使用前面获取的输入张量的形状信息
        g.op("Constant", value_t=torch.tensor([dim], dtype=torch.int64)),  # 使用包含维度值的常量张量
    )
    # 如果ONNX操作集版本小于或等于10
    if g.opset <= 10:
        # 如果不是降序排序，则抛出未实现的异常信息
        if not decending:
            _unimplemented("Sort", "Ascending is not supported")
        # 返回一个包含TopK操作结果的ONNX图操作（OP），包括前K个最大元素的值和对应的索引
        return g.op("TopK", input, dim_size_, axis_i=dim, outputs=2)
    # 如果ONNX操作集版本大于10
    else:
        # 返回一个包含TopK操作结果的ONNX图操作（OP），可以根据参数控制是求最大的K个元素还是最小的K个元素
        return g.op(
            "TopK", input, dim_size_, axis_i=dim, largest_i=decending, outputs=2
        )
@_beartype.beartype
def _topk_helper(
    g: jit_utils.GraphContext, input, k, dim, largest=True, sorted=False, out=None
):
    # 如果提供了输出参数，抛出未实现异常，因为不支持
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported")
    
    # 如果 k 不是一个值类型，则将其转换为常量张量
    if not _is_value(k):
        k = g.op("Constant", value_t=torch.tensor([k], dtype=torch.int64))
    else:
        # 调整 k 的形状以适应操作
        k = _reshape_helper(g, k, g.op("Constant", value_t=torch.tensor([1])))
        # 如果 k 的标量类型不是 INT64，则强制转换为 INT64 类型
        if _try_get_scalar_type(k) != _type_utils.JitScalarType.INT64:
            k = g.op("Cast", k, to_i=_C_onnx.TensorProtoDataType.INT64)
    
    # 根据操作集的版本选择适当的操作
    if g.opset <= 10:
        # 如果不是 largest 模式，抛出未实现异常
        if not largest:
            _unimplemented("TopK", "Ascending is not supported")
        # 使用 TopK 操作获取输入的前 k 个值和索引
        return g.op("TopK", input, k, axis_i=dim, outputs=2)
    else:
        # 使用带有更多参数选项的 TopK 操作获取输入的前 k 个值和索引
        return g.op(
            "TopK", input, k, axis_i=dim, largest_i=largest, sorted_i=sorted, outputs=2
        )


@_beartype.beartype
def _lt_helper(g: jit_utils.GraphContext, input, other):
    # 如果操作集版本低于等于 8，则使用 opset8 的 lt 函数
    if g.opset <= 8:
        from torch.onnx.symbolic_opset8 import lt as _lt8
        return _lt8(g, input, other)
    else:
        # 否则，使用 opset9 的 lt 函数
        from torch.onnx.symbolic_opset9 import lt as _lt9
        return _lt9(g, input, other)


@_beartype.beartype
def _interpolate_warning(interpolate_mode):
    # 根据全局变量的导出版本选择不同的 ONNX 操作名称
    onnx_op = (
        "onnx:Resize" if GLOBALS.export_onnx_opset_version >= 10 else "onnx:Upsample"
    )
    # 发出警告，说明导出模型的操作符可能会导致结果与 PyTorch 预期的结果不一致
    warnings.warn(
        "You are trying to export the model with "
        + onnx_op
        + " for ONNX opset version "
        "" + str(GLOBALS.export_onnx_opset_version) + ". "
        "This operator might cause results to not match the expected results by PyTorch.\n"
        "ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. "
        "Attributes to determine how to transform the input were added in onnx:Resize in opset 11 "
        "to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).\n"
        "We recommend using opset 11 and above for models using this operator."
    )


@_beartype.beartype
def _unsqueeze_helper(g: jit_utils.GraphContext, input, axes_i):
    # 如果轴是常量，则使用指定的轴创建常量张量
    if _is_constant(axes_i[0]):
        if g.opset >= 13:
            axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("Unsqueeze", input, axes)
        # 如果操作集版本不足 13，则直接使用动态轴创建 Unsquueze 操作
        return g.op("Unsqueeze", input, axes_i=axes_i)
    
    # 如果轴不是常量且操作集版本小于 13，则抛出符号值错误
    if g.opset < 13:
        raise errors.SymbolicValueError(
            "Opset version must be >= 13 for Unsqueeze with dynamic axes.", input
        )
    
    # 使用动态轴创建 Unsquueze 操作
    return g.op("Unsqueeze", input, axes_i[0])


@_beartype.beartype
def _squeeze_helper(g: jit_utils.GraphContext, input, axes_i):
    # 如果轴是常量，则使用指定的轴创建常量张量
    if _is_constant(axes_i[0]):
        if g.opset >= 13:
            axes = g.op("Constant", value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op("Squeeze", input, axes)
        # 如果操作集版本不足 13，则直接使用动态轴创建 Squeeze 操作
        return g.op("Squeeze", input, axes_i=axes_i)
    
    # 如果轴不是常量且操作集版本小于 13，则抛出符号值错误
    # 否则，使用动态轴创建 Squeeze 操作
    # 检查操作集合的版本是否小于13，如果是，则抛出符号值错误
    if g.opset < 13:
        raise errors.SymbolicValueError(
            "Opset version must be >= 13 for Squeeze with dynamic axes.", input
        )
    
    # 获取 axes_i 中的第一个元素作为 axes_t
    axes_t = axes_i[0]
    
    # 获取 axes_t 的张量秩
    axes_rank = _get_tensor_rank(axes_t)
    
    # 断言 axes_rank 不为 None
    assert axes_rank is not None
    
    # 如果 axes_rank 大于 1，则根据 ONNX 规范抛出符号值错误
    if axes_rank > 1:
        raise errors.SymbolicValueError(
            "For Squeeze axses as input, the axes rank must be one in ONNX spec.", input
        )
    
    # 如果 axes_rank 等于 0，则表示 axes 是一个标量，需要将其unsqueeze为秩为 1 的张量
    elif axes_rank == 0:
        # 使用 _unsqueeze_helper 函数将 axes_t 在指定维度 [0] 上进行 unsqueeze 操作
        axes_t = _unsqueeze_helper(g, axes_t, [0])
        # 返回通过 Squeeze 操作处理的结果
        return g.op("Squeeze", input, axes_t)
    
    # 如果以上条件均不满足，则直接返回通过 Squeeze 操作处理的结果
    return g.op("Squeeze", input, axes_t)
# 使用装饰器 beartype 对函数进行类型检查和类型提示
@_beartype.beartype
# 定义一个辅助函数 _reducesum_helper，用于在图形上下文 g 中执行 ReduceSum 操作
def _reducesum_helper(
    g: jit_utils.GraphContext,
    input,  # 输入参数
    axes_i=None,  # 要沿其求和的轴
    keepdims_i=1,  # 是否保留减少的维度
    noop_with_empty_axes_i=0,  # 是否处理空轴
):
    # 尝试获取 keepdims_i 的常量值
    keepdims_i = _maybe_get_const(keepdims_i, "i")
    # 如果 opset 版本 >= 13
    if g.opset >= 13:
        # 如果指定了轴 axes_i
        if axes_i:
            # 如果 axes_i 不是值，将其转换为常量张量
            if not _is_value(axes_i):
                axes_i = g.op(
                    "Constant", value_t=torch.tensor(axes_i, dtype=torch.long)
                )
            # 执行 ReduceSum 操作并返回结果
            return g.op(
                "ReduceSum",
                input,
                axes_i,
                keepdims_i=keepdims_i,
                noop_with_empty_axes_i=noop_with_empty_axes_i,
            )
        # 没有指定轴 axes_i，执行 ReduceSum 操作并返回结果
        return g.op(
            "ReduceSum",
            input,
            keepdims_i=keepdims_i,
            noop_with_empty_axes_i=noop_with_empty_axes_i,
        )
    else:
        # opset 版本 < 13，执行 ReduceSum 操作并返回结果
        return g.op("ReduceSum", input, axes_i=axes_i, keepdims_i=keepdims_i)


# 使用装饰器 beartype 对函数进行类型检查和类型提示
@_beartype.beartype
# 定义一个辅助函数 _interpolate_size_to_scales，用于在图形上下文 g 中处理插值大小到缩放比例的转换
def _interpolate_size_to_scales(g: jit_utils.GraphContext, input, output_size, dim):
    # 尝试获取 output_size 的常量值
    output_size = _maybe_get_const(output_size, "is")
    # 如果 output_size 是值
    if _is_value(output_size):
        # 定义偏移量为 2
        offset = 2
        # 创建一个常量张量，值为 [1.0, 1.0]，数据类型为 float32
        offsets = g.op("Constant", value_t=torch.ones(offset, dtype=torch.float32))
        # 将 output_size 强制转换为 float 类型
        dividend = g.op("Cast", output_size, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        # 获取输入张量的形状，并在指定的轴上进行切片
        divisor = _slice_helper(
            g, g.op("Shape", input), axes=[0], ends=[sys.maxsize], starts=[offset]
        )
        # 将 divisor 张量转换为 float 类型
        divisor = g.op("Cast", divisor, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        # 计算缩放比例
        scale_dims = g.op("Div", dividend, divisor)
        # 在指定轴上拼接偏移量和缩放比例
        scales = g.op("Concat", offsets, scale_dims, axis_i=0)
    else:
        # output_size 不是值，根据输入和输出维度计算缩放比例
        scales_constant = [
            1.0
            if i < 2
            else float(output_size[-(dim - i)])
            / float(input.type().sizes()[-(dim - i)])
            for i in range(0, dim)
        ]
        # 创建一个常量张量，值为 scales_constant，数据类型为 float32
        scales = g.op(
            "Constant", value_t=torch.tensor(scales_constant, dtype=torch.float32)
        )
    # 返回计算得到的缩放比例
    return scales


# 使用装饰器 beartype 对函数进行类型检查和类型提示
@_beartype.beartype
# 定义一个辅助函数 _interpolate_get_scales_if_available，用于在图形上下文 g 中检查并获取可用的缩放比例
def _interpolate_get_scales_if_available(g: jit_utils.GraphContext, scales):
    # 检查是否有可用的缩放比例
    available_scales = _maybe_get_const(scales[0], "fs") != -1 and not _is_none(
        scales[0]
    )

    # 如果没有可用的缩放比例，返回 None
    if not available_scales:
        return None

    # 创建一个常量张量，值为 [1.0, 1.0]，数据类型为 float32
    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    # 创建一个常量张量，值为 scales[0] 的常量值，数据类型为 float32
    scales_list = g.op(
        "Constant", value_t=torch.tensor(_maybe_get_const(scales[0], "fs"))
    )
    # 在指定轴上拼接偏移量和缩放比例列表
    scales = g.op("Concat", offsets, scales_list, axis_i=0)
    # 返回拼接后的缩放比例
    return scales


# 使用装饰器 beartype 对函数进行类型检查和类型提示
@_beartype.beartype
# 定义一个辅助函数 _get_interpolate_attributes，用于在图形上下文 g 中获取插值的属性
def _get_interpolate_attributes(g: jit_utils.GraphContext, mode, args):
    # 如果模式是 "nearest"
    if mode == "nearest":
        # 将 align_corners 设置为 None
        align_corners = None
        # 获取缩放参数 scales
        scales = args[0:]
    else:
        # 获取 align_corners 和 scales 参数
        align_corners = args[0]
        scales = args[1:]
    # 获取可用的缩放比例
    scales = _interpolate_get_scales_if_available(g, scales)
    # 返回获取到的缩放比例和 align_corners
    return scales, align_corners


# 使用装饰器 beartype 对函数进行类型检查和类型提示
@_beartype.beartype
# 定义一个辅助函数 _interpolate_get_scales，用于在图形上下文 g 中获取缩放比例
def _interpolate_get_scales(g: jit_utils.GraphContext, scale_factor, dim):
    # 创建一个常量张量，值为 [1.0, 1.0]，数据类型为 float32
    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    # 获取 scale_factor 的张量秩
    scale_factor_rank = _get_tensor_rank(scale_factor)
    # 检查 scale_factor 是否是列表类型或者其秩大于0
    if isinstance(scale_factor.type(), _C.ListType) or (
        scale_factor_rank is not None and scale_factor_rank > 0
    ):
        # 如果满足条件，使用 g.op 方法拼接 offsets 和 scale_factor，指定 axis_i=0
        return g.op("Concat", offsets, scale_factor, axis_i=0)
    else:
        # 否则，对 scale_factor 进行维度扩展，使其至少具有一个维度
        scale_factor = _unsqueeze_helper(g, scale_factor, [0])
        # 将 scale_factor 转换为 FLOAT 类型
        scale_factor = g.op(
            "Cast", scale_factor, to_i=_C_onnx.TensorProtoDataType.FLOAT
        )
        # 创建一个长度为 dim-2 的列表，用 scale_factor 填充
        scales = [scale_factor for i in range(dim - 2)]
    # 使用 g.op 方法将 offsets 和 scales 拼接成一个张量，指定 axis_i=0
    scale_factor = g.op("Concat", offsets, *scales, axis_i=0)
    # 返回拼接后的张量 scale_factor
    return scale_factor
# 使用装饰器对函数进行类型检查和修饰
@_beartype.beartype
# 定义一个函数，用于在图形上下文中进行插值操作的参数处理
def _interpolate_get_scales_and_mode(
    g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners
):
    # 尝试获取模式的常量值，如果不是常量则返回原始值
    mode = _maybe_get_const(mode, "s")
    # 如果模式字符串包含"linear"，则将模式设为"linear"
    if "linear" in mode:
        mode = "linear"
    # 如果模式字符串包含"cubic"，则将模式设为"cubic"
    if "cubic" in mode:
        mode = "cubic"
    # 发出插值警告，传递当前模式
    _interpolate_warning(mode)

    # 尝试获取align_corners的常量值，如果不是布尔值则返回原始值
    align_corners = _maybe_get_const(align_corners, "b")
    # 如果align_corners是布尔类型并且为True，则返回未实现的错误信息
    if isinstance(align_corners, bool) and align_corners:
        return _unimplemented("interpolate", "align_corners == True")

    # 如果输入的张量没有维度，则返回未实现的错误信息
    if not input.type().dim():
        return _unimplemented("interpolate", "missing input shape")
    # 获取输入张量的维度数
    dim = input.type().dim()

    # 如果scale_factor不是None，则使用插值获取比例尺，并传递给scale_factor
    if not _is_none(scale_factor):
        scale_factor = _interpolate_get_scales(g, scale_factor, dim)
    # 如果size不是None
    elif not _is_none(size):
        # 如果size不是打包列表，则检查是否为标量，如果是则进行扩展操作
        if not _is_packed_list(size):
            is_scalar = _maybe_get_const(size, "t").dim() == 0
            if is_scalar:
                size = _unsqueeze_helper(g, size, [0])
                size = [size for i in range(dim - 2)]
                size = g.op("Concat", *size, axis_i=0)
        # 使用插值将size转换为比例尺，并传递给scale_factor
        scale_factor = _interpolate_size_to_scales(g, input, size, dim)
    else:
        # 如果size和scale_factor都是None，则返回未实现的错误信息
        return _unimplemented(
            "interpolate", "Both size and scales are None in __interpolate"
        )
    # 返回计算得到的scale_factor和mode
    return scale_factor, mode


# 使用装饰器对函数进行类型检查和修饰
@_beartype.beartype
# 定义一个辅助函数，用于在图形上下文中处理argmin和argmax操作的参数
def _argmin_argmax_helper(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    dim: torch._C.Value,
    keepdim: bool,
    op_name: str,
):
    # 定义一个内部函数，用于对操作进行包装和处理
    def op_wrapper(input, axis_i, keepdims_i):
        # 如果操作集大于等于12，则执行带有select_last_index_i参数的操作
        if g.opset >= 12:
            return g.op(
                op_name,
                input,
                axis_i=axis_i,
                keepdims_i=keepdims_i,
                select_last_index_i=False,
            )
        # 否则执行不带select_last_index_i参数的操作
        return g.op(op_name, input, axis_i=axis_i, keepdims_i=keepdims_i)

    # 如果dim参数为None，则对输入张量进行扁平化处理，并执行相应操作
    if _is_none(dim):
        flattened = _reshape_helper(
            g, input, g.op("Constant", value_t=torch.tensor([-1]))
        )
        output = op_wrapper(flattened, axis_i=0, keepdims_i=False)
        # 如果keepdim为True，则进行重塑操作，保持相同的形状
        if keepdim:
            input_shape = g.op("Shape", input)
            input_shape_shape = g.op("Shape", input_shape)
            new_shape = g.op(
                "ConstantOfShape",
                input_shape_shape,
                value_t=torch.tensor([1], dtype=torch.int64),
            )
            output = g.op("Reshape", output, new_shape)
        # 返回处理后的输出
        return output

    # 解析dim参数为整数，并执行相应操作
    dim = _parse_arg(dim, "i")
    # 调用操作包装器，传递input和dim参数
    return op_wrapper(input, axis_i=dim, keepdims_i=keepdim)


# 使用装饰器对函数进行类型检查和修饰
@_beartype.beartype
# 定义一个辅助函数，用于在图形上下文中处理插值操作的辅助参数
def _interpolate_helper(name, dim, interpolate_mode):
    # 使用quantized_args装饰器进行量化参数处理
    @quantized_args(True, False, False)
    def symbolic_fn(g, input, output_size, *args):
        # 获取插值方法和参数，返回缩放比例和对齐角点信息
        scales, align_corners = _get_interpolate_attributes(g, interpolate_mode, args)
        
        # 确定坐标变换模式
        coordinate_transformation_mode = (
            "asymmetric"
            if interpolate_mode == "nearest"
            else "align_corners"
            if align_corners
            else "half_pixel"
        )
        
        # 如果缩放比例为空
        if scales is None:
            # 获取输入的形状
            input_size = g.op("Shape", input)
            # 对输入的形状进行切片以获取起始部分
            input_size_beg = _slice_helper(
                g, input_size, axes=[0], ends=[2], starts=[0]
            )
            # 将输出大小转换为INT64类型
            output_size = g.op(
                "Cast", output_size, to_i=_C_onnx.TensorProtoDataType.INT64
            )
            # 将输入的形状起始部分和输出大小连接起来
            output_size = g.op("Concat", input_size_beg, output_size, axis_i=0)

            # 根据ONNX操作集的版本，创建空的ROI和缩放比例
            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
                empty_scales = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )
                empty_scales = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )

            # 返回Resize操作，使用输入、空ROI、空缩放比例、输出大小和坐标变换模式
            return g.op(
                "Resize",
                input,
                empty_roi,
                empty_scales,
                output_size,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # 仅在mode="cubic"时有效
                mode_s=interpolate_mode,  # 插值模式，可以是nearest、linear或cubic
                nearest_mode_s="floor",  # 仅在mode="nearest"时有效
            )
        else:
            # 根据ONNX操作集的版本，创建空的ROI
            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op(
                    "Constant", value_t=torch.tensor([], dtype=torch.float32)
                )

            # 返回Resize操作，使用输入、空ROI、缩放比例、坐标变换模式等参数
            return g.op(
                "Resize",
                input,
                empty_roi,
                scales,
                coordinate_transformation_mode_s=coordinate_transformation_mode,
                cubic_coeff_a_f=-0.75,  # 仅在mode="cubic"时有效
                mode_s=interpolate_mode,  # 插值模式，可以是nearest、linear或cubic
                nearest_mode_s="floor",  # 仅在mode="nearest"时有效
            )

    return symbolic_fn
@_beartype.beartype
# 使用装饰器对函数进行类型检查和注解
def __interpolate_helper(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
):
    # 尝试获取模式的常量值，如果不是则保持原样
    mode = _maybe_get_const(mode, "s")
    
    # 如果模式中包含"linear"，则设定为"linear"
    if "linear" in mode:
        mode = "linear"
    
    # 如果模式中包含"cubic"，则设定为"cubic"
    if "cubic" in mode:
        mode = "cubic"
    
    # 尝试获取align_corners的常量值，如果不是则设为False
    align_corners = _maybe_get_const(align_corners, "b")
    align_corners = False if not isinstance(align_corners, bool) else align_corners
    
    # 根据模式和align_corners值确定坐标变换模式
    coordinate_transformation_mode = (
        "asymmetric" if mode == "nearest"
        else "align_corners" if align_corners
        else "half_pixel"
    )

    # 如果size不是None，则进行以下操作
    if not _is_none(size):
        # 获取输入张量的尺寸
        input_size = g.op("Shape", input)
        input_size = _slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
        
        # 尝试判断size是否为标量，如果出错则给出警告并假设不是标量
        try:
            is_scalar = not _is_packed_list(size) and (
                _maybe_get_const(size, "t").dim() == 0
            )
        except AttributeError:
            is_scalar = not _is_packed_list(size)
            if not is_scalar:
                warnings.warn(
                    "Cannot verify if the output_size is a scalar "
                    "while exporting interpolate. Assuming that it is not a scalar."
                )
        
        # 如果size是标量，则进行以下操作
        if is_scalar:
            # 获取输入张量的秩
            rank = _get_tensor_rank(input)
            if rank is None:
                return _unimplemented(
                    "interpolate (with a scalar output_size)",
                    "missing input shape (try giving an array of output_size values)",
                )
            # 将size转换为张量，并在第0维度上添加一个维度
            size = _unsqueeze_helper(g, size, [0])
            size = [size for i in range(rank - 2)]
            # 沿指定轴连接size张量
            size = g.op("Concat", *size, axis_i=0)
        
        # 将size转换为INT64类型的张量
        size = g.op("Cast", size, to_i=_C_onnx.TensorProtoDataType.INT64)
        # 沿指定轴连接输入尺寸和size
        size = g.op("Concat", input_size, size, axis_i=0)

        # 根据ONNX版本选择空的ROI和scales张量
        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
            empty_scales = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
            empty_scales = g.op(
                "Constant", value_t=torch.tensor([], dtype=torch.float32)
            )

        # 返回Resize操作的结果
        return g.op(
            "Resize",
            input,
            empty_roi,
            empty_scales,
            size,
            coordinate_transformation_mode_s=coordinate_transformation_mode,
            cubic_coeff_a_f=-0.75,  # 仅在mode="cubic"时有效
            mode_s=mode,  # nearest, linear, or cubic
            nearest_mode_s="floor",
        )
    # 如果没有指定缩放因子（scales），则进入此分支
    else:  # if not _is_none(scales)
        # 获取输入张量的维度
        rank = _get_tensor_rank(input)
        # 如果获取到的维度为 None，则返回未实现的错误信息
        if rank is None:
            return _unimplemented("interpolate (with scales)", "missing input shape")

        # 根据当前 opset 的版本选择创建空的 ROI 张量或常量张量
        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

        # 根据插值方法和输入张量的维度获取缩放因子
        scales = _interpolate_get_scales(g, scale_factor, rank)
        
        # 调用 "Resize" 操作来执行张量的调整大小操作
        return g.op(
            "Resize",
            input,  # 输入张量
            empty_roi,  # 空的 ROI 张量或常量
            scales,  # 缩放因子
            coordinate_transformation_mode_s=coordinate_transformation_mode,  # 坐标转换模式
            cubic_coeff_a_f=-0.75,  # 仅在 mode="cubic" 时有效的立方插值系数
            mode_s=mode,  # 插值模式，可以是 nearest（最近邻）、linear（线性）、cubic（立方）
            nearest_mode_s="floor",  # 当插值模式为 nearest 时有效的取整模式
        )  # 返回 Resize 操作的结果
# 使用装饰器_beartype.beartype对函数进行类型检查和注解
@_beartype.beartype
# 定义函数_unbind_helper，接受图形上下文g、self、维度dim和输出_outputs作为参数
def _unbind_helper(g: jit_utils.GraphContext, self, dim, _outputs):
    # 根据不同的操作集版本选择合适的unbind函数
    if g.opset < 11:
        from torch.onnx.symbolic_opset9 import unbind
    elif g.opset <= 12:
        from torch.onnx.symbolic_opset11 import unbind  # type: ignore[no-redef]
    else:
        from torch.onnx.symbolic_opset13 import unbind  # type: ignore[no-redef]
    # 调用选定的unbind函数执行操作
    return unbind(g, self, dim, _outputs)


@_beartype.beartype
# 定义函数_scatter_helper，接受图形上下文g、self、维度dim、索引index和源src作为参数
def _scatter_helper(g: jit_utils.GraphContext, self, dim, index, src):
    # 根据不同的操作集版本选择合适的scatter函数
    if g.opset <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        # 在较新版本的操作集中，scatter函数已经在前面两行被导入过了
        from torch.onnx.symbolic_opset11 import scatter  # type: ignore[no-redef]
    # 调用选定的scatter函数执行操作
    return scatter(g, self, dim, index, src)


@_beartype.beartype
# 定义函数_repeat_interleave_split_helper，接受图形上下文g、self、重复次数reps和维度dim作为参数
def _repeat_interleave_split_helper(g: jit_utils.GraphContext, self, reps, dim):
    # 根据不同的操作集版本选择合适的Split操作
    if g.opset <= 12:
        split_out = g.op("Split", self, split_i=[1] * reps, axis_i=dim, outputs=reps)
    else:
        from torch.onnx.symbolic_opset13 import split
        # 创建一个包含所有1的张量作为重复次数
        repeats = g.op("Constant", value_t=torch.tensor([1] * reps))
        # 调用split函数执行操作
        split_out = split(g, self, repeats, dim, _outputs=reps)
    # 如果重复次数大于1，则返回分割后的结果列表，否则返回单个分割结果
    return split_out if reps > 1 else [split_out]


@_beartype.beartype
# 定义函数_repeat_interleave_single_value_repeat_helper，接受图形上下文g、self、重复次数repeats和维度dim作为参数
def _repeat_interleave_single_value_repeat_helper(
    g: jit_utils.GraphContext, self, repeats, dim
):
    # 从torch.onnx.symbolic_opset9中导入flatten和unsqueeze函数
    from torch.onnx.symbolic_opset9 import flatten, unsqueeze

    # 如果重复次数repeats不是张量，则将其转换为常量张量
    if not _is_tensor(repeats):
        repeats = g.op("Constant", value_t=torch.LongTensor(repeats))

    # 检查重复次数repeats是否为常量
    const_repeats: bool = _is_constant(repeats)
    # 获取重复次数repeats的值
    reps = _maybe_get_const(repeats, "t")

    # 如果重复次数repeats是0维的，则将其reshape为1维张量
    if _get_tensor_rank(repeats) == 0:
        repeats = g.op("Reshape", repeats, g.op("Constant", value_t=torch.tensor([1])))

    # 创建一个维度大小为1的新维度，然后将其扩展为长度为'repeats'，最后合并它
    unsqueezed = unsqueeze(g, self, dim + 1)

    # 对新的unsqueezed维度以外的所有维度设置重复次数为1，对unsqueezed维度设置为'repeats'
    if const_repeats:
        # 如果'repeats'是常量，则'repeats_per_dim'可以是常量
        onehot = torch.ones(_get_tensor_rank(unsqueezed), dtype=torch.int64)
        onehot[dim + 1] = reps
        repeats_per_dim = g.op("Constant", value_t=onehot)
    else:
        # 如果'repeats'是变量，则'repeats_per_dim'不能是常量
        onehot = g.op(
            "OneHot",
            unsqueeze(g, dim + 1, 0),  # 索引，必须是至少1维
            g.op(
                "Constant", value_t=torch.tensor(_get_tensor_rank(unsqueezed))
            ),  # 深度
            g.op(
                "Concat", g.op("Constant", value_t=torch.tensor([1])), repeats, axis_i=0
            ),  # 开/关值
        )
        repeats_per_dim = flatten(g, onehot, 0, 1)

    # 使用Tile操作将unsqueezed张量扩展为tiled
    tiled = g.op("Tile", unsqueezed, repeats_per_dim)
    # 对tiled结果进行flatten操作，将维度dim和dim + 1合并
    return flatten(g, tiled, dim, dim + 1)


@_beartype.beartype
# 定义函数_arange_cast_helper，接受图形上下文g、self、重复次数repeats和维度dim作为参数
def _arange_cast_helper(
    g: jit_utils.GraphContext, end, start=None, step=None, dtype=None


# 定义函数参数：
# - g: jit_utils.GraphContext，表示一个图形上下文对象，用于执行某些图形相关操作
# - end，表示一个必需的参数，通常用于指定范围的结束值
# - start=None，表示一个可选的参数，默认为None，通常用于指定范围的起始值
# - step=None，表示一个可选的参数，默认为None，通常用于指定步长
# - dtype=None，表示一个可选的参数，默认为None，通常用于指定数据类型
def _index_fill_reshape_helper(g: jit_utils.GraphContext, self, dim, index):
    # 1. reshape index => [1, ..., 1, dim, 1, ..., 1]
    # 对索引进行 reshape，使其形状为 [1, ..., 1, dim, 1, ..., 1]
    from torch.onnx.symbolic_opset9 import expand

    # 2. expand index => [..., dim, ...], same shape as self except for dim.
    # 对索引进行扩展，使其形状与 self 相同，除了在 dim 维度上。
    if g.opset <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        # for mypy, scatter was imported two lines above
        from torch.onnx.symbolic_opset11 import scatter  # type: ignore[no-redef]

    if self.type().dim() is None:
        return _unimplemented("index_fill", "input rank not accessible")
    self_dim = self.type().dim()
    dim_value = _parse_arg(dim, "i")
    if dim_value < 0:
        dim_value += self_dim
    # 将索引在除了 dim_value 外的维度上进行 unsqueeze 操作
    unsqueezed_index = _unsqueeze_helper(
        g, index, [i for i in range(self_dim) if i != dim_value]
    )
    # 使用 scatter 函数在图 g 中进行操作，用于创建 expanded_index_shape 变量
    expanded_index_shape = scatter(
        g,                           # 使用的图对象 g
        g.op("Shape", self),         # 调用 g 中的操作 "Shape"，获取 self 的形状信息
        0,                           # scatter 操作的维度参数，这里为 0
        _unsqueeze_helper(g, dim, [0]),  # 调用 _unsqueeze_helper 函数，对 dim 进行 unsqueeze 操作得到的结果
        g.op("Shape", index)         # 获取 index 的形状信息
    )
    
    # 使用 expand 函数在图 g 中进行操作，创建 expanded_index 变量
    expanded_index = expand(
        g,                           # 使用的图对象 g
        unsqueezed_index,            # 要扩展的 unsqueezed_index 变量
        expanded_index_shape,        # 使用的 expanded_index_shape 变量作为扩展形状
        None                         # 没有指定输出形状，设为 None
    )
    
    # 返回结果变量 expanded_index_shape 和 expanded_index
    return expanded_index_shape, expanded_index
# 默认情况下，当 'shape' 输入中的任何值等于零时，将动态复制对应的维度值从输入张量中获取。
# allowzero=1 表示如果 'shape' 输入中有任何值设置为零，则会保留该零值，类似于 NumPy 的行为。
# allowzero=1 仅在 opset 版本 >= 14 下支持。
@_beartype.beartype
def _reshape_helper(g: jit_utils.GraphContext, input, shape, allowzero=0):
    shape = _maybe_get_const(shape, "is")  # 如果 'shape' 是常量，则获取其值
    if not _is_value(shape):  # 如果 'shape' 不是常量
        shape = g.op("Constant", value_t=torch.LongTensor(shape))  # 创建一个常量张量节点
    if g.opset <= 13:  # 如果 opset 版本 <= 13
        if allowzero == 1:  # 如果 allowzero 等于 1
            _onnx_opset_unsupported(
                "Reshape with allowzero=1", GLOBALS.export_onnx_opset_version, 14, input
            )  # 抛出不支持的操作异常
        return g.op("Reshape", input, shape)  # 使用 'Reshape' 操作对输入进行形状重塑
    else:  # 如果 opset 版本 > 13
        return g.op("Reshape", input, shape, allowzero_i=allowzero)  # 使用 'Reshape' 操作对输入进行形状重塑，设置 allowzero 参数


@_beartype.beartype
def _batchnorm_helper(
    g: jit_utils.GraphContext, input, weight, bias, running_mean, running_var
):
    from torch.onnx.symbolic_opset9 import _var_mean  # 导入 _var_mean 函数

    batch_size = _get_tensor_dim_size(input, 0)  # 获取输入张量的批量大小
    channel_size = _get_tensor_dim_size(input, 1)  # 获取输入张量的通道数

    if weight is None or _is_none(weight):  # 如果权重为 None 或者未知
        if channel_size is None:  # 如果通道数未知
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of batch_norm for unknown channel size.",
                input,
            )  # 抛出不支持的符号值错误
        weight_value = torch.tensor(
            [1.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )  # 创建一个值全为 1.0 的张量作为权重
        weight = g.op("Constant", value_t=weight_value)  # 创建一个常量节点表示权重
    if bias is None or _is_none(bias):  # 如果偏置为 None 或者未知
        if channel_size is None:  # 如果通道数未知
            raise errors.SymbolicValueError(
                "Unsupported: ONNX export of batch_norm for unknown channel size.",
                input,
            )  # 抛出不支持的符号值错误
        bias_value = torch.tensor(
            [0.0] * channel_size,
            dtype=_type_utils.JitScalarType.from_value(input).dtype(),
        )  # 创建一个值全为 0.0 的张量作为偏置
        bias = g.op("Constant", value_t=bias_value)  # 创建一个常量节点表示偏置
    # 如果 track_running_stats 设置为 False，则在评估时使用批次统计数据
    if (
        running_mean is None
        or _is_none(running_mean)
        or running_var is None
        or _is_none(running_var)
    ):  # 如果 running_mean 或 running_var 为 None 或者未知
        assert batch_size is not None and channel_size is not None  # 断言批次大小和通道数不为 None
        reshape_in = _reshape_helper(
            g,
            input,
            g.op(
                "Constant",
                value_t=torch.tensor([batch_size, channel_size, -1], dtype=torch.int64),
            ),  # 创建一个常量节点，表示形状为 [batch_size, channel_size, -1]
        )  # 对输入进行形状重塑
        trans_in = g.op("Transpose", reshape_in, perm_i=[0, 2, 1])  # 对重塑后的输入进行转置操作
        running_var, running_mean = _var_mean(
            g,
            trans_in,
            g.op("Constant", value_t=torch.tensor([0, 1], dtype=torch.int64)),  # 创建一个常量节点，表示维度索引 [0, 1]
            False,
            False,
        )  # 使用 _var_mean 函数计算方差和均值
    return weight, bias, running_mean, running_var  # 返回权重、偏置、运行时均值和方差
    tuple_fn: Callable[[Any], Sequence[int]],
    # tuple_fn 是一个函数类型的变量，接受任意类型的参数，返回一个整数序列

    padding: Union[int, Sequence[int]],
    # padding 是一个变量，可以是整数类型或者整数序列类型

    kernel_size,
    # kernel_size 是一个变量，用于表示核大小

    stride,
    # stride 是一个变量，表示步幅大小

    divisor_override,
    # divisor_override 是一个变量，表示除数覆盖

    name,
    # name 是一个变量，表示名称
@_beartype.beartype
def check_training_mode(op_train_mode: int, op_name: str) -> None:
    """Warns the user if the model's training mode and the export mode do not agree."""
    # 检查全局训练模式是否为保持模式，如果是，则不做任何操作
    if GLOBALS.training_mode == _C_onnx.TrainingMode.PRESERVE:
        return

    # 根据操作的训练模式设置操作模式枚举值
    if op_train_mode:
        op_mode_enum = _C_onnx.TrainingMode.TRAINING
    else:
        op_mode_enum = _C_onnx.TrainingMode.EVAL
    # 如果操作模式枚举与全局训练模式相同，则不做任何操作
    if op_mode_enum == GLOBALS.training_mode:
        return

    # 构建操作模式的文本描述
    op_mode_text = f"train={bool(op_train_mode)}"
    # 如果模型是 FuncModule，并且设置了模型模式可能导致操作模式与全局训练模式不一致，
    # 则根据操作模式进行导出，给用户警告状态
    warnings.warn(
        f"ONNX export mode is set to {GLOBALS.training_mode}, but operator '{op_name}' "
        f"is set to {op_mode_text}. Exporting with {op_mode_text}."
    )


@_beartype.beartype
def _flatten_helper(g: jit_utils.GraphContext, input, start_dim, end_dim, dim):
    # 获取输入的形状大小
    input_size = g.op("Shape", input)
    # 切片操作，获取输入的部分形状信息
    slice1 = _slice_helper(g, input_size, axes=[0], starts=[0], ends=[start_dim])
    slices = [slice1, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))]
    # 如果结束维度小于指定维度-1，则进行更多的切片操作
    if end_dim < dim - 1:
        slice3 = _slice_helper(
            g, input_size, axes=[0], starts=[end_dim + 1], ends=[dim]
        )
        slices = [
            slice1,
            g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long)),
            slice3,
        ]

    # 最终形状是通过 Concat 操作得到的
    final_shape = g.op("Concat", *slices, axis_i=0)
    # 调用 _reshape_from_tensor 函数，根据最终形状对输入进行重塑
    from torch.onnx.symbolic_opset9 import _reshape_from_tensor
    return _reshape_from_tensor(g, input, final_shape)


@_beartype.beartype
def _is_split_static(split_size_or_sizes, _outputs):
    # 如果 _outputs 为 None，则返回 False
    if _outputs is None:
        return False
    # 如果 split_size_or_sizes 是值并且不是常量，则返回 False
    if (
        _is_value(split_size_or_sizes)
        and split_size_or_sizes.node().kind() != "onnx::Constant"
    ):
        return False
    # 其他情况返回 True
    return True


@_beartype.beartype
def _optional_input_placeholder_tensor(g):
    # 创建一个表示常量的节点 n
    n = g.op("prim::Constant")
    # 将节点 n 设置为可选类型的 Tensor
    n.setType(_C.OptionalType.ofTensor())
    return n


@_beartype.beartype
def _handle_reduce_dim_none(g: jit_utils.GraphContext, self, op_name):
    # 获取输入张量的秩
    rank = _get_tensor_rank(self)
    # 如果秩不为 None 并且任何维度大小为 0，则根据 ONNX ReduceSum 定义，设置 keepdims=1
    if rank is not None and any(
        _get_tensor_dim_size(self, i) == 0 for i in range(rank)
    ):
        return g.op(op_name, self, keepdims_i=1)
    # 否则使用默认的 keepdims=0
    return g.op(op_name, self, keepdims_i=0)


@_beartype.beartype
def dequantize_helper(
    g: jit_utils.GraphContext,
    qtensor: _C.Value,
    # 以下省略部分参数和函数体，需要继续注释的内容在下一行
    qdtype: Optional[_C_onnx.TensorProtoDataType] = None,


    # 定义变量 qdtype，类型为 Optional[_C_onnx.TensorProtoDataType]，初始值为 None
# Appends ONNX nodes to graph `g` for dequantizing `qtensor` into `tensor`.
# Args:
#     g: Graph, the ONNX IR graph under construction.
#     qtensor: torch._C.Value, either a tuple of (quantized_tensor, scale, zero_point)
#         for per tensor quantization, or
#         (quantized_tensor, scale, zero_point, axis) for per channel quantization,
#         representing the quantized tensor.
#     qdtype: torch.onnx.TensorProtoDataType default None, if not None, specifies the
#         data type of quantized tensor, either UINT8 or INT8.
def dequantize_helper(
    g: Graph,
    qtensor: _C.Value,
    qdtype: Optional[_C.Value] = None,
) -> Tuple[_C.Value, _C.Value, _C.Value, Optional[_C.Value]]:
    """Appends to graph `g` ONNX nodes that dequantizes `qtensor` into `tensor`.

    Args:
        g: Graph, the ONNX IR graph that is under construction.
        qtensor: torch._C.Value, either a tuple of (quantized_tensor, scale, zero_point)
            for per tensor quantization, or
            (quantized_tensor, scale, zero_point, axis) for per channel quantization,
            representing the quantized tensor.
        qdtype: torch.onnx.TensorProtoDataType default None, if not None, represents the
            data type of quantized tensor. It must be either
            torch.onnx.TensorProtoDataType.UINT8 or torch.onnx.TensorProtoDataType.INT8.
    """
    # Unpacks the quantized tensor `qtensor` into its components.
    unpacked_qtensors = _unpack_quantized_tensor(qtensor)
    tensor, scale, zero_point = unpacked_qtensors[:3]
    axis = unpacked_qtensors[3] if len(unpacked_qtensors) >= 4 else None
    # Retrieves the constant axis index if available.
    axis_i = _get_const(axis, "i", "axis")
    # Determines the input quantized data type.
    input_qdtype = _type_utils.JitScalarType.from_value(tensor)
    # If qdtype is not provided, infers it from input_qdtype or defaults to UINT8.
    if qdtype is None:
        if input_qdtype is not None:
            qdtype = input_qdtype.onnx_type()
        else:
            qdtype = _C_onnx.TensorProtoDataType.UINT8
    # Casts the quantized tensor, scale, and zero point to desired data types.
    value = g.op("Cast", tensor, to_i=qdtype)
    scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    zero_point = g.op("Cast", zero_point, to_i=qdtype)

    # Checks if axis is supported depending on ONNX opset version.
    if axis_i is not None and GLOBALS.export_onnx_opset_version < 13:
        _onnx_opset_unsupported_detailed(
            "DequantizeLinear",
            GLOBALS.export_onnx_opset_version,
            13,
            "Attribute axis is not supported.",
            qtensor,
        )

    # Returns the dequantized tensor, scale, zero point, and axis.
    return (
        g.op("DequantizeLinear", value, scale, zero_point, axis_i=axis_i),
        scale,
        zero_point,
        axis,
    )


# Decorated function for quantizing `tensor` based on `scale`, `zero_point`, and optional `axis`.
# Appends ONNX nodes to graph `g`.
# Args:
#     g: Graph, the ONNX IR graph under construction.
#     tensor: torch._C.Value, representing the tensor to be quantized.
#     scale: torch._C.Value, quantized scale.
#     zero_point: torch._C.Value, quantized zero point.
#     axis: Optional[torch._C.Value] default None, represents per tensor or per channel quantization.
# Returns:
#     _C.Value, a quantized tensor.
@_beartype.beartype
def quantize_helper(
    g: jit_utils.GraphContext,
    tensor: _C.Value,
    scale: _C.Value,
    zero_point: _C.Value,
    axis: Optional[_C.Value] = None,
) -> _C.Value:
    """Appends to graph `g` ONNX nodes that quantizes `tensor` based on `scale`, `zero_point` and `axis`.

    Args:
        g: Graph, the ONNX IR graph that is under construction.
        tensor: torch._C.Value, representing the tensor to be quantized.
        scale: torch._C.Value, quantized scale.
        zero_point: torch._C.Value, quantized zero point.
        axis: Optional[torch._C.Value] default None, if None, represents per tensor quantization.
            Otherwise, represents per channel quantization, along given axis.

    Returns:
        A TupleConstruct storing information of the quantized tensor.
    """
    # Checks if axis is unsupported for the current ONNX opset version.
    if axis is not None and not _is_none(axis) and GLOBALS.export_onnx_opset_version < 13:
        _onnx_opset_unsupported_detailed(
            "QuantizeLinear",
            GLOBALS.export_onnx_opset_version,
            13,
            "Attribute axis is not supported.",
            tensor,
        )

    # Asserts that scale is not None for quantization.
    assert scale is not None
    # 检查 `scale` 的类型是否为 `FLOAT`，如果不是则进行类型转换为 `FLOAT`
    if (
        _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)

    # 断言 `zero_point` 参数不为空
    assert zero_point is not None
    # 检查 `zero_point` 的类型是否为 `UINT8` 或 `INT8`，如果不是则进行类型转换为 `UINT8`
    if _type_utils.JitScalarType.from_value(
        zero_point, _type_utils.JitScalarType.UNDEFINED
    ) not in {
        _type_utils.JitScalarType.UINT8,
        _type_utils.JitScalarType.INT8,
    }:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)

    # 使用 `QuantizeLinear` 操作对输入张量进行线性量化
    output = g.op(
        "QuantizeLinear",
        tensor,
        scale,
        zero_point,
        axis_i=_get_const(axis, "i", "axis"),
    )
    
    # 将输出张量、scale 和 zero_point 组成一个参数列表
    args = [output, scale, zero_point]
    
    # 如果 axis 参数不为空且不是 None，则将其添加到参数列表中
    if axis is not None and not _is_none(axis):
        args.append(axis)
    
    # 返回一个元组，包含所有的参数作为 `prim::TupleConstruct` 操作的输出
    return g.op("prim::TupleConstruct", *args)
# 使用 @_beartype.beartype 装饰器对函数进行类型检查和验证
@_beartype.beartype
# 定义一个辅助函数 requantize_bias_helper，用于在图形上下文 g 中重新量化偏置
def requantize_bias_helper(
    g: jit_utils.GraphContext, bias, input_scale, weight_scale, axis=None
):
    """In PyTorch, bias is float and is quantized to int32 implicitly inside the quantized ATen op kernel.
    In ONNX we need to make the quantization explicit because operators expect all of their inputs to be quantized.
    Since int32 is not a supported output type by ONNX operator `QuantizeLinear`, quantization is exported using
    regular operators.
    """
    # 计算偏置量化的比例，即 weight_scale 乘以 input_scale
    bias_scale = g.op("Mul", weight_scale, input_scale)
    # 获取 bias_scale 的形状
    bias_scale_shape = g.op("Shape", bias_scale)
    # 创建一个与 bias_scale_shape 形状相同的常量张量，值为 0，数据类型为 torch.int
    bias_zero_point = g.op(
        "ConstantOfShape", bias_scale_shape, value_t=torch.tensor([0], dtype=torch.int)
    )
    # 将 bias 除以 bias_scale，并将结果转换为 INT32 类型
    q_bias = g.op(
        "Cast", g.op("Div", bias, bias_scale), to_i=_C_onnx.TensorProtoDataType.INT32
    )
    # 如果指定了 axis 并且 axis 不为空，将其添加到 axis_args 列表中
    axis_args = []
    if axis is not None and not _is_none(axis):
        axis_args.append(axis)
    # 返回一个包含 q_bias、bias_scale、bias_zero_point 和 axis_args 的元组
    return g.op("prim::TupleConstruct", q_bias, bias_scale, bias_zero_point, *axis_args)


# 定义一个函数 args_have_same_dtype，用于检查参数列表中的所有参数是否具有相同的数据类型
@_beartype.beartype
def args_have_same_dtype(args):
    # 断言确保 args 列表不为空
    assert args
    # 获取第一个参数的 JitScalarType 类型
    base_dtype = _type_utils.JitScalarType.from_value(args[0])
    # 检查 args 中所有元素是否与 base_dtype 相同类型
    has_same_dtype = all(
        _type_utils.JitScalarType.from_value(elem) == base_dtype for elem in args
    )
    # 返回布尔值，指示所有参数是否具有相同的数据类型
    return has_same_dtype


# 定义一个函数 _op_with_optional_float_cast，用于处理带有可选浮点数转换的操作
@_beartype.beartype
def _op_with_optional_float_cast(g: jit_utils.GraphContext, op_name, *args, **kwargs):
    """Some PyTorch operators (e.g., Clip/Min/ReLU/Pad) are super set of ONNX in terms of data types.
    This function maximizes the exportability of PyTorch-ONNX by allowing ONNX-unsupported PyTorch
    operator data type. For example, `Cast<int>(Clip<float>(Cast<float>(INPUT)))` can be used to mimic
    `Clip<int>(INPUT)` (opset version < 12).

    Args:
        g (torch._C.Graph): graph to write the ONNX representation into.
        op_name (str): operator name in ONNX.
        *args (tuple): operands to the operator.
        **kwargs (dict): attributes to the operator along with "opset_before" (optional, None by default)
            indicating the smallest opset version to trigger such casting behavior and "target_float_t"
            (optional, torch.onnx.JitScalarType.FLOAT by default) indicating the data type of internal operator.

    Returns:
        Optional[torch._C.Value, Tuple[torch._C.Value, ...]]: output(s) of the operator.
    """
    # 获取 opset_before 参数，默认为 None
    opset_before = kwargs.pop("opset_before", None)
    # 获取 target_float_t 参数，默认为 torch.onnx.JitScalarType.FLOAT
    target_float_t = kwargs.pop("target_float_t", _type_utils.JitScalarType.FLOAT)

    # 将 args 转换为列表
    inputs = list(args)
    # 获取第一个输入参数的 JitScalarType 类型
    dtype_0 = _type_utils.JitScalarType.from_value(inputs[0])

    # 判断是否需要进行浮点数转换
    require_cast = not _is_fp(inputs[0]) and (
        opset_before is None or GLOBALS.export_onnx_opset_version < opset_before
    )
    # 如果需要类型转换
    if require_cast:
        # 遍历输入列表中的每个输入
        for input in inputs:
            # 检查输入是否为完整的张量
            if input.isCompleteTensor():
                # 根据输入值确定其标量类型
                input_scalar_type = _type_utils.JitScalarType.from_value(input)
                # 如果输入的标量类型不等于指定的 dtype_0
                if input_scalar_type != dtype_0:
                    # 抛出符号值错误，指出输入必须具有相同的数据类型
                    raise errors.SymbolicValueError(
                        f"Inputs of {op_name} must have same dtype."
                        f"Got {dtype_0.scalar_name()} and {input_scalar_type.scalar_name()}",
                        input,
                    )
        
        # 再次遍历输入列表中的每个输入
        for i, input in enumerate(inputs):
            # 如果输入是完整的张量且不是浮点数
            if input.isCompleteTensor() and not _is_fp(input):
                # 对输入执行类型转换，将其转换为目标浮点数类型的 ONNX 类型
                inputs[i] = g.op(
                    "Cast",
                    input,
                    to_i=target_float_t.onnx_type(),
                )

    # 创建当前操作节点，调用指定操作名称 op_name，并传入所有输入和额外参数
    self = g.op(op_name, *inputs, **kwargs)

    # 如果需要类型转换
    if require_cast:
        # 对当前操作节点进行类型转换，将其转换为指定的 dtype_0 类型的 ONNX 类型
        self = g.op("Cast", self, to_i=dtype_0.onnx_type())

    # 返回最终生成的操作节点
    return self
# 使用装饰器 `_beartype.beartype` 来装饰函数 `_maybe_cast_reduce_op_input`，增加类型检查和注解
@_beartype.beartype
def _maybe_cast_reduce_op_input(g: jit_utils.GraphContext, self):
    # 从 `self` 中获取标量类型，如果未定义则默认为 `_type_utils.JitScalarType.UNDEFINED`
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    # 如果标量类型不是 `_type_utils.JitScalarType.UNDEFINED`
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        # 对于已跟踪的模块（traced modules），存在 dtype 的情况下，PyTorch reduce-ops 会将所有其他整数类型转换为 int64
        if not _is_fp(self) and scalar_type != _type_utils.JitScalarType.INT64:
            # 使用 ONNX 操作 "Cast" 将 `self` 转换为 int64 类型
            self = g.op("Cast", self, to_i=_C_onnx.TensorProtoDataType.INT64)
    # 返回处理后的 `self`
    return self


# 定义一个内部函数 `_apply_params`，返回一个装饰器，用来将指定的参数传递给被装饰的函数
def _apply_params(*args, **kwargs):
    """Returns a decorator that calls the decorated (higher-order) function with the given parameters."""
    
    def _apply(fn):
        return fn(*args, **kwargs)
    
    return _apply


# 使用装饰器 `_beartype.beartype` 来装饰函数 `_reduce_op_symbolic_helper`，增加类型检查和注解
@_beartype.beartype
def _reduce_op_symbolic_helper(onnx_op_name, allow_multi_dim_support=True):
    # 定义一个内部函数 `symbolic`，用于构建 ONNX symbolic 函数
    @_beartype.beartype
    def symbolic(g, self, dim=None, keepdim=None):
        # 对输入的 `self` 进行类型转换处理
        self = _maybe_cast_reduce_op_input(g, self)
        
        # 如果 `dim` 为 None 或空元组，则执行全局减少路径处理
        if dim is None or dim == tuple():
            # 执行处理无维度的减少操作，调用 `_handle_reduce_dim_none` 函数
            return _handle_reduce_dim_none(g, self, onnx_op_name)
        else:
            # 执行有维度的减少操作路径
            # 获取 `keepdim` 常量值，如果未指定则默认为 True
            keepdim = _get_const(keepdim, "i", "keepdim")
            
            # 如果 ONNX 的操作集小于 18
            if g.opset < 18:
                desc = "is" if allow_multi_dim_support else "i"
                # 获取 `dim` 的常量值，如果 `allow_multi_dim_support` 为 True，则 `dim` 可以是一个列表
                dim = _get_const(dim, desc, "dim")
                dim_list = dim if allow_multi_dim_support else [dim]
                # 调用 ONNX 操作 `onnx_op_name`，传入 `self`，`axes_i` 和 `keepdims_i` 参数
                return g.op(onnx_op_name, self, axes_i=dim_list, keepdims_i=keepdim)
            else:
                # 如果 `dim` 是一个值类型
                if _is_value(dim):
                    axes = dim
                else:
                    # 根据 `allow_multi_dim_support` 的值，创建包含 `dim` 的常量张量
                    if allow_multi_dim_support:
                        axes = g.op(
                            "Constant", value_t=torch.tensor(dim, dtype=torch.long)
                        )
                    else:
                        axes = g.op(
                            "Constant", value_t=torch.tensor([dim], dtype=torch.long)
                        )
                # 调用 ONNX 操作 `onnx_op_name`，传入 `self`，`axes` 和 `keepdims_i` 参数
                return g.op(onnx_op_name, self, axes, keepdims_i=keepdim)

    return symbolic


# 使用装饰器 `_beartype.beartype` 来装饰函数 `_overload_by_arg_count`，增加类型检查和注解
@_beartype.beartype
def _overload_by_arg_count(fn):
    # 定义一个内部函数 `wrapper`，用于根据参数数量选择适当的重载函数
    @functools.wraps(fn)
    @_beartype.beartype
    def wrapper(g, *args):
        # 调用被装饰函数 `fn`，获取重载列表 `overloads`
        overloads = fn(g, *args)
        # 遍历重载列表
        for overload in overloads:
            # 获取重载函数的参数描述符
            arg_descriptors = overload._arg_descriptors
            # 如果参数数量与当前参数列表 `args` 的长度相同，则调用该重载函数
            if len(arg_descriptors) == len(args):
                return overload(g, *args)
        # 如果没有匹配的重载函数，则返回未实现的错误信息
        return _unimplemented(f"aten::{fn.__name__}", f"with {len(args)} arguments")

    return wrapper


# 使用装饰器 `_beartype.beartype` 来装饰函数 `_reduce_with_dtype_helper`，增加类型检查和注解
@_beartype.beartype
def _reduce_with_dtype_helper(
    onnx_op: str, name: str, allow_multi_dim_support: bool = True
):
    # 使用 `_reduce_op_symbolic_helper` 函数创建一个符号函数 `symbolic`
    symbolic = _reduce_op_symbolic_helper(
        onnx_op, allow_multi_dim_support=allow_multi_dim_support
    )
    
    # 定义一个内部函数 `_overload_by_arg_count`，根据参数数量选择适当的重载函数
    @_overload_by_arg_count
    def reduce(g, *args, **kwargs):
        # 定义内部函数 reduce_nodim，接收 g, self, dtype 作为参数，并进行量化参数处理
        @quantized_args(True)
        @parse_args("v", "none")
        def reduce_nodim(g, self, dtype):
            # 检查 dtype 是否为 onnx::Constant 类型的节点
            if dtype.node().kind() == "onnx::Constant":
                # 如果是常量节点，则从 dtype 中获取整数类型的常量值
                dtype = _get_const(dtype, "i", "dtype")
                # 将获取的整数类型转换为对应的 onnx 类型
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                # 将 self 张量转换为指定的 dtype_onnx 类型
                self = g.op("Cast", self, to_i=dtype_onnx)
            # 如果 dtype 不是 prim::Constant 类型的节点，则返回未实现的错误
            elif dtype.node().kind() != "prim::Constant":
                return _unimplemented(name, "dtype", dtype)
            # 对 self 进行符号计算，生成结果
            result = symbolic(g, self)
            # 如果存在 dtype_onnx 类型，进一步处理结果类型转换
            if dtype_onnx is not None:
                # 获取结果的 onnx 类型
                result_dtype_onnx = _type_utils.JitScalarType.from_value(result).onnx_type()
                # 如果结果类型与指定的 dtype_onnx 类型不匹配，则将结果转换为指定类型
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            # 返回处理后的结果
            return result

        # 根据 allow_multi_dim_support 的值确定维度描述字符串
        dim_desc = "is" if allow_multi_dim_support else "i"

        # 定义内部函数 reduce_dim，接收 g, self, dim, keepdim, dtype 作为参数，并进行量化参数处理
        @quantized_args(True)
        @parse_args("v", dim_desc, "i", "none")  # type: ignore[arg-type]
        def reduce_dim(g, self, dim, keepdim, dtype):
            # 检查 dtype 是否为 onnx::Constant 类型的节点
            if dtype.node().kind() == "onnx::Constant":
                # 如果是常量节点，则从 dtype 中获取整数类型的常量值
                dtype = _get_const(dtype, "i", "dtype")
                # 将获取的整数类型转换为对应的 onnx 类型
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                # 将 self 张量转换为指定的 dtype_onnx 类型
                self = g.op("Cast", self, to_i=dtype_onnx)
            # 如果 dtype 不是 prim::Constant 类型的节点，则返回未实现的错误
            elif dtype.node().kind() != "prim::Constant":
                return _unimplemented(name, "dtype", dtype)
            # 对 self 进行带维度参数的符号计算，生成结果
            result = symbolic(g, self, dim, keepdim)
            # 如果存在 dtype_onnx 类型，进一步处理结果类型转换
            if dtype_onnx is not None:
                # 获取结果的 onnx 类型
                result_dtype_onnx = _type_utils.JitScalarType.from_value(result).onnx_type()
                # 如果结果类型与指定的 dtype_onnx 类型不匹配，则将结果转换为指定类型
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            # 返回处理后的结果
            return result

        # 返回内部函数 reduce_nodim 和 reduce_dim
        return reduce_nodim, reduce_dim

    # 返回外部函数 reduce
    return reduce
# 带有类型检查装饰器的函数定义，用于计算输入张量在指定维度上的最大值和对应的索引
@_beartype.beartype
def _max_helper(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # 如果没有指定维度和keepdim参数，则对整个张量进行最大值的降维操作
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMax", self, keepdims_i=0)
    
    # 如果没有指定keepdim参数，则根据输入的第二个参数进行最大值计算，可能需要进行浮点数转换
    if keepdim is None:
        return _op_with_optional_float_cast(g, "Max", self, dim_or_y, opset_before=12)
    
    # 如果指定了dim和keepdim参数，则根据不同的opset版本选择合适的操作符
    else:
        # 获取真实的keepdim和dim值
        keepdim = _get_const(keepdim, "i", "keepdim")
        dim = _get_const(dim_or_y, "i", "dim")
        
        # 根据opset版本选择不同的操作符进行降维最大值计算
        if g.opset < 18:
            max = g.op("ReduceMax", self, axes_i=[dim], keepdims_i=keepdim)
        else:
            axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            max = g.op("ReduceMax", self, axes, keepdims_i=keepdim)
        
        # 计算在指定维度上的最大值对应的索引
        indices = g.op("ArgMax", self, axis_i=dim, keepdims_i=keepdim)
        
        # 返回计算得到的最大值和对应的索引
        return max, indices


# 带有类型检查装饰器的函数定义，用于计算输入张量在指定维度上的最小值和对应的索引
@_beartype.beartype
def _min_helper(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # 如果没有指定维度和keepdim参数，则对整个张量进行最小值的降维操作
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMin", self, keepdims_i=0)
    
    # 如果没有指定keepdim参数，则根据输入的第二个参数进行最小值计算，可能需要进行浮点数转换
    if keepdim is None:
        return _op_with_optional_float_cast(g, "Min", self, dim_or_y, opset_before=12)
    
    # 如果指定了dim和keepdim参数，则根据不同的opset版本选择合适的操作符
    else:
        # 获取真实的keepdim和dim值
        keepdim = _get_const(keepdim, "i", "keepdim")
        dim = _get_const(dim_or_y, "i", "dim")
        
        # 根据opset版本选择不同的操作符进行降维最小值计算
        if g.opset < 18:
            min = g.op("ReduceMin", self, axes_i=[dim], keepdims_i=keepdim)
        else:
            axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            min = g.op("ReduceMin", self, axes, keepdims_i=keepdim)
        
        # 计算在指定维度上的最小值对应的索引
        indices = g.op("ArgMin", self, axis_i=dim, keepdims_i=keepdim)
        
        # 返回计算得到的最小值和对应的索引
        return min, indices


# 带有类型检查装饰器的函数定义，用于计算输入张量的元素数量
@_beartype.beartype
def _numel_helper(g: jit_utils.GraphContext, self):
    # 计算输入张量的形状
    shape = g.op("Shape", self)
    # 对张量形状进行降维乘积操作，得到元素数量
    return g.op("ReduceProd", shape, keepdims_i=0)


# 带有参数解析装饰器的函数定义，用于计算输入张量在指定维度上的方差和均值
@parse_args("v", "is", "i", "i")
@_beartype.beartype
def _var_mean_helper(g: jit_utils.GraphContext, input, dim, correction, keepdim):
    # 如果ONNX操作集版本小于18
    if g.opset < 18:
        # 如果未指定dim（即对所有维度进行操作）
        if dim is None:
            # 计算输入张量的平均值，并保持维度信息
            mean = g.op("ReduceMean", input, keepdims_i=0)
            # t_mean与mean相同
            t_mean = mean
            # 计算输入张量的元素总数
            num_elements = _numel_helper(g, input)
        else:
            # 按指定维度dim计算输入张量的平均值，并保持维度信息
            mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=keepdim)
            # t_mean按指定维度dim计算输入张量的平均值，并保持维度信息
            t_mean = g.op("ReduceMean", input, axes_i=dim, keepdims_i=1)
            # 获取输入张量的形状信息
            redudced_dims = g.op("Shape", input)
            # dim可能包含一个或多个维度
            # 使用Gather操作从redudced_dims中收集指定的维度
            redudced_dims = g.op(
                "Gather",
                redudced_dims,
                g.op("Constant", value_t=torch.tensor(dim)),
                axis_i=0,
            )
            # 计算指定维度的元素总数
            num_elements = g.op("ReduceProd", redudced_dims, keepdims_i=0)
        
        # 计算输入张量与其均值的差
        sub_v = g.op("Sub", input, t_mean)
        # 计算差的平方
        sqr_sub = g.op("Mul", sub_v, sub_v)
        # 如果未指定dim，则保持维度信息为0，否则根据keepdim指定
        keepdim_mean = 0 if dim is None else keepdim
        # 计算输入张量方差，按指定维度dim保持维度信息
        var = g.op("ReduceMean", sqr_sub, axes_i=dim, keepdims_i=keepdim_mean)
        
        # 在计算方差时修正偏差，通过除以（N - correction）而不是N
        # 如果未提供修正值，则默认为1
        if correction is None:
            correction = 1
        # 如果修正值不为0，则进行修正
        if correction != 0:
            # 将num_elements转换为浮点数类型
            num_elements = g.op(
                "Cast", num_elements, to_i=_C_onnx.TensorProtoDataType.FLOAT
            )
            # 创建常量张量表示修正值
            one = g.op("Constant", value_t=torch.tensor(correction, dtype=torch.float))
            # 计算修正后的方差
            mul = g.op("Mul", var, num_elements)
            var = g.op("Div", mul, g.op("Sub", num_elements, one))
        
        # 返回方差和均值
        return var, mean
    else:
        # 如果未指定维度，则计算整个张量的均值
        axes = None
        if dim is None:
            # 计算整个输入张量的均值
            mean = g.op("ReduceMean", input, keepdims_i=0)
            # 将均值赋值给临时变量
            t_mean = mean
            # 计算输入张量的元素总数
            num_elements = _numel_helper(g, input)
        else:
            # 将维度转换为常量张量
            axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
            # 沿指定维度计算输入张量的均值
            mean = g.op("ReduceMean", input, axes, keepdims_i=keepdim)
            # 沿指定维度计算输入张量的均值，保持维度
            t_mean = g.op("ReduceMean", input, axes, keepdims_i=1)
            # 计算输入张量的形状
            redudced_dims = g.op("Shape", input)
            # 获取指定维度的形状
            redudced_dims = g.op(
                "Gather",
                redudced_dims,
                g.op("Constant", value_t=torch.tensor(dim)),
                axis_i=0,
            )
            # 计算指定维度下的元素总数
            num_elements = g.op("ReduceProd", redudced_dims, keepdims_i=0)
        
        # 计算输入张量与均值的差
        sub_v = g.op("Sub", input, t_mean)
        # 计算差的平方
        sqr_sub = g.op("Mul", sub_v, sub_v)
        # 如果未指定维度，则保持均值的维度为0；否则根据keepdim参数保持维度为1
        keepdim_mean = 0 if dim is None else keepdim
        
        # 根据是否指定维度，计算方差
        if axes is None:
            # 计算整个输入张量的方差
            var = g.op("ReduceMean", sqr_sub, keepdims_i=keepdim_mean)
        else:
            # 沿指定维度计算输入张量的方差
            var = g.op("ReduceMean", sqr_sub, axes, keepdims_i=keepdim_mean)
        
        # 修正方差的偏差，通过除以(N - correction)而不是N来纠正
        if correction is None:
            correction = 1
        if correction != 0:
            # 将元素总数转换为浮点数
            num_elements = g.op(
                "Cast", num_elements, to_i=_C_onnx.TensorProtoDataType.FLOAT
            )
            # 创建常量张量1
            one = g.op("Constant", value_t=torch.tensor(correction, dtype=torch.float))
            # 计算方差乘以元素总数
            mul = g.op("Mul", var, num_elements)
            # 通过(N - 1)来修正方差
            var = g.op("Div", mul, g.op("Sub", num_elements, one))
        
        # 返回方差和均值
        return var, mean
# 使用装饰器进行类型检查，确保函数参数符合预期的类型和约束
@_beartype.beartype
# 定义嵌入操作的辅助函数，接受多个参数
def _embedding_bag_helper(
    g: jit_utils.GraphContext,  # 图形上下文对象，用于构建图形操作
    embedding_matrix,           # 嵌入矩阵，用于查找嵌入向量
    indices,                    # 索引列表，指定需要从嵌入矩阵中提取的索引
    offsets,                    # 偏移量列表，指定每个“袋子”（bag）的起始索引
    scale_grad_by_freq,         # 是否按频率缩放梯度
    mode,                       # 嵌入模式（平均或求和）
    sparse,                     # 是否稀疏嵌入
    per_sample_weights,         # 每个样本的权重
    include_last_offset,        # 是否包含最后一个偏移量
    padding_idx,                # 填充索引，用于填充输入
):
    # 如果需要按频率缩放梯度且处于训练模式，则返回不支持 ONNX 格式的错误信息
    if scale_grad_by_freq and GLOBALS.export_training:
        return _onnx_unsupported(
            "embedding_bag with scale_grad_by_freq for training mode"
        )
    # 如果填充索引为非空且大于等于0，则引发运行时错误
    if padding_idx is not None and padding_idx >= 0:
        raise RuntimeError("embedding_bag with padding_idx")

    # 创建一个常量节点表示循环条件为真
    loop_condition = g.op("Constant", value_t=torch.tensor(1))
    # 将循环条件强制转换为布尔类型
    loop_condition = g.op("Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    # 创建一个常量节点表示0
    zero = g.op("Constant", value_t=torch.tensor([0]))

    # 计算索引的长度，并在第0维上增加一个维度
    indices_len = _unsqueeze_helper(
        g,
        _size_helper(g, indices, g.op("Constant", value_t=torch.tensor(0))),
        [0],
    )
    # 如果不包含最后一个偏移量，则将偏移量列表扩展添加索引长度，并沿第0维度进行连接
    if not include_last_offset:
        offsets = [offsets, indices_len]
        offsets = g.op("Concat", *offsets, axis_i=0)

    # 根据偏移量提取每个“袋子”（bag）的起始位置索引
    offsets_starts = _slice_helper(
        g, offsets, axes=[0], starts=[0], ends=[sys.maxsize], steps=[1]
    )
    # 根据偏移量提取每个“袋子”（bag）的结束位置索引
    offsets_ends = _slice_helper(
        g, offsets, axes=[0], starts=[1], ends=[sys.maxsize], steps=[1]
    )

    # 计算循环的长度
    loop_len = _size_helper(g, offsets_ends, g.op("Constant", value_t=torch.tensor(0)))

    # 添加一个具有块的循环操作，返回循环对象、上下文和结果
    loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
        g, "Loop", loop_len, loop_condition, n_blocks=1
    )
    # 获取循环块
    loop_block = loop_context.block

    # FIXME(justinchuby): 处理调用节点返回时的操作
    # 将输入添加到块中的输入迭代器
    block_input_iter = utils._add_input_to_block(loop_block)
    # 将输入添加到块中的条件
    cond = utils._add_input_to_block(loop_block)

    # 使用偏移量开始位置提取索引起始位置
    indices_start = loop_context.op(
        "Gather", offsets_starts, block_input_iter, axis_i=0
    )
    # 使用偏移量结束位置提取索引结束位置
    indices_end = loop_context.op("Gather", offsets_ends, block_input_iter, axis_i=0)
    # 在第0维度上增加一个维度，以处理索引起始位置
    indices_start = _unsqueeze_helper(loop_context, indices_start, [0])
    # 在第0维度上增加一个维度，以处理索引结束位置
    indices_end = _unsqueeze_helper(loop_context, indices_end, [0])

    # 使用切片操作从索引中提取特定的行范围
    indices_row = loop_context.op("Slice", indices, indices_start, indices_end, zero)
    # 使用索引行从嵌入矩阵中提取嵌入向量
    embeddings = loop_context.op("Gather", embedding_matrix, indices_row, axis_i=0)
    
    # 如果每个样本权重不为空，则对嵌入向量进行加权处理
    if not _is_none(per_sample_weights):
        # 使用切片操作从每个样本权重中提取特定的行范围
        per_sample_weights_row = loop_context.op(
            "Slice", per_sample_weights, indices_start, indices_end, zero
        )
        # 在第1维度上增加一个维度，以处理每个样本权重
        per_sample_weights_row = _unsqueeze_helper(
            loop_context, per_sample_weights_row, [1]
        )
        # 对嵌入向量进行加权处理
        embeddings = loop_context.op("Mul", embeddings, per_sample_weights_row)
    # 如果 mode 等于 0，则调用 _reducesum_helper 函数进行求和操作
    if mode == 0:
        embeddings = _reducesum_helper(
            loop_context, embeddings, axes_i=[0], keepdims_i=0
        )
    # 如果 mode 等于 1，则根据 opset 版本选择执行 ReduceMean 或 ReduceMean 操作
    elif mode == 1:
        if loop_context.opset < 18:
            embeddings = loop_context.op(
                "ReduceMean", embeddings, axes_i=[0], keepdims_i=0
            )
        else:
            # 创建一个表示轴的张量 axes，并执行 ReduceMean 操作
            axes = loop_context.op(
                "Constant", value_t=torch.tensor([0], dtype=torch.long)
            )
            embeddings = loop_context.op("ReduceMean", embeddings, axes, keepdims_i=0)
    # 如果 mode 不是 0 或 1，则根据 opset 版本选择执行 ReduceMax 或 ReduceMax 操作
    else:
        if loop_context.opset < 18:
            embeddings = loop_context.op(
                "ReduceMax", embeddings, axes_i=[0], keepdims_i=0
            )
        else:
            # 创建一个表示轴的张量 axes，并执行 ReduceMax 操作
            axes = loop_context.op(
                "Constant", value_t=torch.tensor([0], dtype=torch.long)
            )
            embeddings = loop_context.op("ReduceMax", embeddings, axes, keepdims_i=0)

    # 将 loop_condition 强制转换为 BOOL 类型，并将结果添加到 loop_block 中
    cond_out = loop_context.op(
        "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
    )
    utils._add_output_to_block(loop_block, cond_out)
    # 将 embeddings 添加到 loop_block 中作为输出
    utils._add_output_to_block(loop_block, embeddings)

    # 返回一个元组，包含 loop 节点的输出以及其他三个未使用的 None 值
    return loop.node().output(), None, None, None
# 使用装饰器进行类型检查和注解，确保输入参数符合要求
@_beartype.beartype
# 定义私有函数 _linalg_vector_norm_helper，用于计算向量的范数
def _linalg_vector_norm_helper(
    g: jit_utils.GraphContext,          # 图上下文对象，用于构建计算图
    self: torch._C.Value,               # 输入向量的张量表示
    ord: float,                         # 范数的阶数
    dim: Optional[Sequence[int]],       # 沿指定维度计算范数，可选参数
    keepdim: bool,                      # 是否保持维度
    dtype: torch._C.Value,              # 输出张量的数据类型
):
    axes = None                         # 默认情况下，轴为空

    # 根据 PyTorch 文档设置条件
    if _is_none(dim):                   # 如果 dim 为空
        # 重新整形输入张量为一维向量
        self = _reshape_helper(g, self, [-1])
        keepdim = False                 # 不保持维度
    elif g.opset >= 18:                 # 如果图操作集大于等于18
        axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
                                        # 使用指定维度创建常数节点

    if ord == math.inf:                 # 如果 ord 为正无穷
        if g.opset < 18:                # 如果图操作集小于18
            # 使用 ReduceMax 操作计算绝对值后的最大值
            result = g.op(
                "ReduceMax", g.op("Abs", self), axes_i=dim, keepdims_i=keepdim
            )
        else:                           # 如果图操作集大于等于18
            if axes is None:            # 如果轴为空
                # 使用 ReduceMax 操作计算绝对值后的最大值
                result = g.op("ReduceMax", g.op("Abs", self), keepdims_i=keepdim)
            else:
                # 使用 ReduceMax 操作计算绝对值后的最大值
                result = g.op("ReduceMax", g.op("Abs", self), axes, keepdims_i=keepdim)
    elif ord == -math.inf:              # 如果 ord 为负无穷
        if g.opset < 18:                # 如果图操作集小于18
            # 使用 ReduceMin 操作计算绝对值后的最小值
            result = g.op(
                "ReduceMin", g.op("Abs", self), axes_i=dim, keepdims_i=keepdim
            )
        else:                           # 如果图操作集大于等于18
            if axes is None:            # 如果轴为空
                # 使用 ReduceMin 操作计算绝对值后的最小值
                result = g.op("ReduceMin", g.op("Abs", self), keepdims_i=keepdim)
            else:
                # 使用 ReduceMin 操作计算绝对值后的最小值
                result = g.op("ReduceMin", g.op("Abs", self), axes, keepdims_i=keepdim)
    elif ord == 0:                      # 如果 ord 为0
        if g.opset < 11:                # 如果图操作集小于11
            # 返回不支持详细信息，表示不支持 ord=0 的情况
            return _onnx_opset_unsupported_detailed(
                "linalg_vector_norm", 9, 11, "ord=0 not supported", self
            )
        else:                           # 如果图操作集大于等于11
            if dim is None:             # 如果 dim 为空
                # 重新整形输入张量为一维向量
                self = _reshape_helper(
                    g,
                    self,
                    g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
                )
                keepdim = False         # 不保持维度

            # 创建条件操作，判断张量是否为非零值
            cond_op = g.op(
                "Not",
                g.op("Equal", self, g.op("Constant", value_t=torch.LongTensor([0]))),
            )
            # 将条件操作转换为指定类型的张量
            cond_op = g.op(
                "Cast",
                cond_op,
                to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
            )
            # 使用 ReduceSum 辅助函数计算满足条件的张量元素和
            return _reducesum_helper(g, cond_op, axes_i=dim, keepdims_i=keepdim)
    elif ord == 1:                      # 如果 ord 为1
        if g.opset < 18:                # 如果图操作集小于18
            # 使用 ReduceL1 操作计算绝对值的和
            result = _reduce_op_symbolic_helper("ReduceL1")(
                g, self, dim=dim, keepdim=keepdim
            )
        else:                           # 如果图操作集大于等于18
            if axes is None:            # 如果轴为空
                # 使用 ReduceL1 操作计算绝对值的和
                result = _reduce_op_symbolic_helper("ReduceL1")(
                    g, self, keepdim=keepdim
                )
            else:
                # 使用 ReduceL1 操作计算绝对值的和
                result = _reduce_op_symbolic_helper("ReduceL1")(
                    g, self, axes, keepdim=keepdim
                )
    # 如果 ord 等于 2
    elif ord == 2:
        # 如果 g.opset 小于 18
        if g.opset < 18:
            # 使用 ReduceL2 辅助函数来进行符号化帮助
            result = _reduce_op_symbolic_helper("ReduceL2")(
                g, self, dim=dim, keepdim=keepdim
            )
        else:
            # 如果 axes 为 None
            if axes is None:
                # 使用 ReduceL2 辅助函数来进行符号化帮助
                result = _reduce_op_symbolic_helper("ReduceL2")(
                    g, self, keepdim=keepdim
                )
            else:
                # 使用 ReduceL2 辅助函数来进行符号化帮助
                result = _reduce_op_symbolic_helper("ReduceL2")(
                    g, self, axes, keepdim=keepdim
                )
    # 如果 ord 不等于 2
    else:
        # 创建一个常量操作 ord_op，其值为 ord，类型为 torch.float32
        ord_op = g.op("Constant", value_t=torch.tensor(ord, dtype=torch.float32))
        # 使用 _reducesum_helper 辅助函数进行操作：计算 self 的 ord 次幂的绝对值的 g 和 axes 和 keepdim 参数
        result = _reducesum_helper(
            g, g.op("Pow", g.op("Abs", self), ord_op), axes_i=dim, keepdims_i=keepdim
        )
        # 对结果进行操作：计算 result 的 ord 次幂，使用 1 除以 ord_op 的结果
        result = g.op(
            "Pow",
            result,
            g.op(
                "Div",
                g.op("Constant", value_t=torch.tensor(1, dtype=torch.float32)),
                ord_op,
            ),
        )

    # 如果 dtype 不为 None
    if not _is_none(dtype):
        # 将 dtype 转换为整数型常量
        dtype = _get_const(dtype, "i", "dtype")
        # 使用 g.op 函数进行类型转换：将 result 转换为指定的 dtype 的 ONNX 类型
        result = g.op("Cast", result, to_i=_type_utils.JitScalarType(dtype).onnx_type())  # type: ignore[arg-type]

    # 返回结果
    return result
# Deprecated. Internally use _type_utils.ScalarType
# TODO: remove these once we support Type's in the JIT IR and we can once again
# use the unified toType operator
cast_pytorch_to_onnx = {
    "Byte": _C_onnx.TensorProtoDataType.UINT8,     # Map PyTorch 'Byte' type to ONNX UINT8
    "Char": _C_onnx.TensorProtoDataType.INT8,      # Map PyTorch 'Char' type to ONNX INT8
    "Double": _C_onnx.TensorProtoDataType.DOUBLE,  # Map PyTorch 'Double' type to ONNX DOUBLE
    "Float": _C_onnx.TensorProtoDataType.FLOAT,    # Map PyTorch 'Float' type to ONNX FLOAT
    "Half": _C_onnx.TensorProtoDataType.FLOAT16,   # Map PyTorch 'Half' type to ONNX FLOAT16
    "Int": _C_onnx.TensorProtoDataType.INT32,      # Map PyTorch 'Int' type to ONNX INT32
    "Long": _C_onnx.TensorProtoDataType.INT64,     # Map PyTorch 'Long' type to ONNX INT64
    "Short": _C_onnx.TensorProtoDataType.INT16,    # Map PyTorch 'Short' type to ONNX INT16
    "Bool": _C_onnx.TensorProtoDataType.BOOL,      # Map PyTorch 'Bool' type to ONNX BOOL
    "ComplexFloat": _C_onnx.TensorProtoDataType.COMPLEX64,    # Map PyTorch 'ComplexFloat' type to ONNX COMPLEX64
    "ComplexDouble": _C_onnx.TensorProtoDataType.COMPLEX128,  # Map PyTorch 'ComplexDouble' type to ONNX COMPLEX128
    "BFloat16": _C_onnx.TensorProtoDataType.BFLOAT16,          # Map PyTorch 'BFloat16' type to ONNX BFLOAT16
    "Undefined": _C_onnx.TensorProtoDataType.UNDEFINED,        # Map PyTorch 'Undefined' type to ONNX UNDEFINED
}

# Deprecated. Internally use _type_utils.ScalarType
scalar_name_to_pytorch = {
    "uint8_t": "Byte",          # Map 'uint8_t' scalar name to PyTorch 'Byte' type
    "int8_t": "Char",           # Map 'int8_t' scalar name to PyTorch 'Char' type
    "double": "Double",         # Map 'double' scalar name to PyTorch 'Double' type
    "float": "Float",           # Map 'float' scalar name to PyTorch 'Float' type
    "half": "Half",             # Map 'half' scalar name to PyTorch 'Half' type
    "int": "Int",               # Map 'int' scalar name to PyTorch 'Int' type
    "int64_t": "Long",          # Map 'int64_t' scalar name to PyTorch 'Long' type
    "int16_t": "Short",         # Map 'int16_t' scalar name to PyTorch 'Short' type
    "bool": "Bool",             # Map 'bool' scalar name to PyTorch 'Bool' type
    "complex64": "ComplexFloat",    # Map 'complex64' scalar name to PyTorch 'ComplexFloat' type
    "complex128": "ComplexDouble",  # Map 'complex128' scalar name to PyTorch 'ComplexDouble' type
    "qint8": "QInt8",           # Map 'qint8' scalar name to PyTorch 'QInt8' type
    "quint8": "QUInt8",         # Map 'quint8' scalar name to PyTorch 'QUInt8' type
    "qint32": "QInt32",         # Map 'qint32' scalar name to PyTorch 'QInt32' type
    "bfloat16": "BFloat16",     # Map 'bfloat16' scalar name to PyTorch 'BFloat16' type
}


# Deprecated. Internally use _type_utils.ScalarType
# This indicates each scalar type's corresponding
# torch type. Related source:
# https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
scalar_type_to_pytorch_type = [
    torch.uint8,    # 0: PyTorch's corresponding type for UINT8
    torch.int8,     # 1: PyTorch's corresponding type for INT8
    torch.short,    # 2: PyTorch's corresponding type for INT16
    torch.int,      # 3: PyTorch's corresponding type for INT32
    torch.int64,    # 4: PyTorch's corresponding type for INT64
    torch.half,     # 5: PyTorch's corresponding type for FLOAT16
    torch.float,    # 6: PyTorch's corresponding type for FLOAT32
    torch.double,   # 7: PyTorch's corresponding type for FLOAT64
    torch.complex32,# 8: PyTorch's corresponding type for COMPLEX32
    torch.complex64,# 9: PyTorch's corresponding type for COMPLEX64
    torch.complex128,# 10: PyTorch's corresponding type for COMPLEX128
    torch.bool,     # 11: PyTorch's corresponding type for BOOL
    torch.qint8,    # 12: PyTorch's corresponding type for QINT8
    torch.quint8,   # 13: PyTorch's corresponding type for QUINT8
    torch.qint32,   # 14: PyTorch's corresponding type for QINT32
    torch.bfloat16, # 15: PyTorch's corresponding type for BFLOAT16
]

# Deprecated. Internally use _type_utils.ScalarType
# source of truth is
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_dtypes.cpp
pytorch_name_to_type = {
    "Byte": torch.uint8,        # Map PyTorch 'Byte' type to its corresponding torch type
    "Char": torch.int8,         # Map PyTorch 'Char' type to its corresponding torch type
    "Double": torch.double,     # Map PyTorch 'Double' type to its corresponding torch type
    "Float": torch.float,       # Map PyTorch 'Float' type to its corresponding torch type
    "Half": torch.half,         # Map PyTorch 'Half' type to its corresponding torch type
    "Int": torch.int,           # Map PyTorch 'Int' type to its corresponding torch type
    "Long": torch.int64,        # Map PyTorch 'Long' type to its corresponding torch type
    "Short": torch.short,       # Map PyTorch 'Short' type to its corresponding torch type
    "Bool": torch.bool,         # Map PyTorch 'Bool' type to its corresponding torch type
    "ComplexFloat": torch.complex64,     # Map PyTorch 'ComplexFloat' type to its corresponding torch type
    "ComplexDouble": torch.complex128,   # Map PyTorch 'ComplexDouble' type to its corresponding torch type
    "QInt8": torch.qint8,       # Map PyTorch 'QInt8' type to its corresponding torch type
    "QUInt8": torch.quint8,     # Map PyTorch 'QUInt8' type to its corresponding torch type
    "QInt32": torch.qint32,     # Map PyTorch 'QInt32' type to its corresponding torch type
    "BFloat16": torch.bfloat16, # Map PyTorch 'BFloat16' type to its corresponding torch type
}

# Deprecated. Internally use _type_utils.ScalarType
scalar_type_to_onnx = [
    cast_pytorch_to_onnx["Byte"],       # 0: ONNX data type corresponding to PyTorch 'Byte'
    cast_pytorch_to_onnx["Char"],       # 1: ONNX data type corresponding to PyTorch 'Char'
    cast_pytorch_to_onnx["Short"],      # 2: ONNX data type corresponding to PyTorch 'Short'
    cast_pytorch_to_onnx["Int"],        # 3: ONNX data type corresponding to PyTorch 'Int'
    cast_pytorch_to_onnx["Long"],       # 4: ONNX data type corresponding to PyTorch 'Long'
    cast_pytorch_to_onnx["Half"],       # 5: ONNX data type corresponding to PyTorch 'Half'
    cast_pytorch_to_onnx["Float"],      # 6: ONNX data type corresponding to PyTorch 'Float'
    cast_pytorch_to_onnx["Double"],     # 7: ONNX data type corresponding to PyTorch 'Double'
    cast_pytorch_to_onnx["Undefined"],  # 8: ONNX data type corresponding to PyTorch 'Undefined'
]
    cast_pytorch_to_onnx["ComplexFloat"],  # 9
    # 获取 cast_pytorch_to_onnx 字典中键为 "ComplexFloat" 的值
    cast_pytorch_to_onnx["ComplexDouble"],  # 10
    # 获取 cast_pytorch_to_onnx 字典中键为 "ComplexDouble" 的值
    cast_pytorch_to_onnx["Bool"],  # 11
    # 获取 cast_pytorch_to_onnx 字典中键为 "Bool" 的值
    cast_pytorch_to_onnx["Char"],  # 12
    # 获取 cast_pytorch_to_onnx 字典中键为 "Char" 的值
    cast_pytorch_to_onnx["Byte"],  # 13
    # 获取 cast_pytorch_to_onnx 字典中键为 "Byte" 的值
    cast_pytorch_to_onnx["Int"],  # 14
    # 获取 cast_pytorch_to_onnx 字典中键为 "Int" 的值
    cast_pytorch_to_onnx["BFloat16"],  # 15
    # 获取 cast_pytorch_to_onnx 字典中键为 "BFloat16" 的值
# 全局变量，用于存储网络中量化操作的列表。
# 目前仅在将量化操作从PyTorch（PT）转换为Caffe2（C2）通过ONNX时使用。
# 使用集合（Set）来确保每个操作的唯一性。
_quantized_ops: Set[int] = set()
```