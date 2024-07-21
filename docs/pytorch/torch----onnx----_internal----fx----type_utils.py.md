# `.\pytorch\torch\onnx\_internal\fx\type_utils.py`

```
# mypy: allow-untyped-defs
"""Utilities for converting and operating on ONNX, JIT and torch types."""
from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import numpy  # 导入 numpy 库

import onnx  # 导入 onnx 库

import torch  # 导入 torch 库
from torch._subclasses import fake_tensor  # 导入 torch._subclasses 模块中的 fake_tensor 类

if TYPE_CHECKING:
    import onnx.defs.OpSchema.AttrType  # type: ignore[import]  # noqa: TCH004
    # 如果是类型检查阶段，导入 onnx.defs.OpSchema.AttrType 模块

# Enable both TorchScriptTensor and torch.Tensor to be tested
# for dtype in OpSchemaWrapper.
@runtime_checkable
class TensorLike(Protocol):
    @property
    def dtype(self) -> Optional[torch.dtype]:
        ...


def is_torch_complex_dtype(tensor_dtype: torch.dtype) -> bool:
    # NOTE: This is needed as TorchScriptTensor is not supported by torch.is_complex()
    # 检查给定的 torch 张量数据类型是否是复数类型
    return tensor_dtype in _COMPLEX_TO_FLOAT


def from_complex_to_float(dtype: torch.dtype) -> torch.dtype:
    # 将复数类型的 torch 数据类型转换为对应的浮点数类型
    return _COMPLEX_TO_FLOAT[dtype]


def from_sym_value_to_torch_dtype(sym_value: SYM_VALUE_TYPE) -> torch.dtype:
    # 将符号值类型转换为对应的 torch 数据类型
    return _SYM_TYPE_TO_TORCH_DTYPE[type(sym_value)]


def is_optional_onnx_dtype_str(onnx_type_str: str) -> bool:
    # 检查给定的字符串是否是可选的 ONNX 数据类型字符串
    return onnx_type_str in _OPTIONAL_ONNX_DTYPE_STR


def from_torch_dtype_to_onnx_dtype_str(dtype: Union[torch.dtype, type]) -> Set[str]:
    # 将给定的 torch 数据类型或类型转换为兼容的 ONNX 数据类型字符串集合
    return _TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS[dtype]


def from_python_type_to_onnx_attribute_type(
    dtype: type, is_sequence: bool = False
) -> Optional[onnx.defs.OpSchema.AttrType]:
    import onnx.defs  # type: ignore[import]
    # 定义 Python 类型到 ONNX 属性类型的映射关系

    _PYTHON_TYPE_TO_ONNX_ATTRIBUTE_TYPE = {
        float: onnx.defs.OpSchema.AttrType.FLOAT,
        int: onnx.defs.OpSchema.AttrType.INT,
        str: onnx.defs.OpSchema.AttrType.STRING,
        bool: onnx.defs.OpSchema.AttrType.INT,
    }

    _SEQUENCE_TYPE_TO_ONNX_ATTRIBUTE_TYPE = {
        float: onnx.defs.OpSchema.AttrType.FLOATS,
        int: onnx.defs.OpSchema.AttrType.INTS,
        str: onnx.defs.OpSchema.AttrType.STRINGS,
        bool: onnx.defs.OpSchema.AttrType.INTS,
    }

    if is_sequence:
        return _SEQUENCE_TYPE_TO_ONNX_ATTRIBUTE_TYPE.get(dtype)
    return _PYTHON_TYPE_TO_ONNX_ATTRIBUTE_TYPE.get(dtype)


def from_python_type_to_onnx_tensor_element_type(type: type):
    """
    Converts a Python type to the corresponding ONNX tensor element type.
    For example, `from_python_type_to_onnx_tensor_element_type(float)` returns
    `onnx.TensorProto.FLOAT`.

    Args:
      type (type): The Python type to convert.

    Returns:
      int: The corresponding ONNX tensor element type.

    """
    _PYTHON_TYPE_TO_ONNX_TENSOR_ELEMENT_TYPE = {
        float: onnx.TensorProto.FLOAT,  # type: ignore[attr-defined]
        int: onnx.TensorProto.INT64,  # type: ignore[attr-defined]
        bool: onnx.TensorProto.BOOL,  # type: ignore[attr-defined]
    }
    # 将给定的 Python 类型转换为对应的 ONNX 张量元素类型
    return _PYTHON_TYPE_TO_ONNX_TENSOR_ELEMENT_TYPE.get(type)


def is_torch_symbolic_type(value: Any) -> bool:
    # 检查给定的值是否是 Torch 的符号类型
    # 检查变量 value 是否是 torch 模块中的 SymBool、SymInt 或 SymFloat 类型的实例
    return isinstance(value, (torch.SymBool, torch.SymInt, torch.SymFloat))
# 将给定的 torch 数据类型映射为其缩写字符串
def from_torch_dtype_to_abbr(dtype: Optional[torch.dtype]) -> str:
    if dtype is None:
        return ""
    # 从 _TORCH_DTYPE_TO_ABBREVIATION 字典中获取对应的缩写，如果没有则返回空字符串
    return _TORCH_DTYPE_TO_ABBREVIATION.get(dtype, "")


# 将给定的 Python 标量类型映射为对应的 torch 数据类型
def from_scalar_type_to_torch_dtype(scalar_type: type) -> Optional[torch.dtype]:
    # 从 _SCALAR_TYPE_TO_TORCH_DTYPE 字典中获取给定类型对应的 torch 数据类型
    return _SCALAR_TYPE_TO_TORCH_DTYPE.get(scalar_type)


# NOTE: this is a mapping from torch dtype to a set of compatible onnx types
# It's used in dispatcher to find the best match overload for the input dtypes
# _TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS 是一个字典，将 torch 数据类型映射到与之兼容的 ONNX 数据类型的集合
_TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS: Dict[
    Union[torch.dtype, type], Set[str]
] = {
    torch.bfloat16: {"tensor(bfloat16)"},
    torch.bool: {"tensor(bool)"},
    torch.float64: {"tensor(double)"},
    torch.float32: {"tensor(float)"},
    torch.float16: {"tensor(float16)"},
    torch.float8_e4m3fn: {"tensor(float8_e4m3fn)"},
    torch.float8_e4m3fnuz: {"tensor(float8_e4m3fnuz)"},
    torch.float8_e5m2: {"tensor(float8_e5m2)"},
    torch.float8_e5m2fnuz: {"tensor(float8_e5m2fnuz)"},
    torch.int16: {"tensor(int16)"},
    torch.int32: {"tensor(int32)"},
    torch.int64: {"tensor(int64)"},
    torch.int8: {"tensor(int8)"},
    torch.uint8: {"tensor(uint8)"},
    str: {"tensor(string)"},
    int: {"tensor(int16)", "tensor(int32)", "tensor(int64)"},
    float: {"tensor(float16)", "tensor(float)", "tensor(double)"},
    bool: {"tensor(int32)", "tensor(int64)", "tensor(bool)"},
    complex: {"tensor(float)", "tensor(double)"},
    torch.complex32: {"tensor(float16)"},
    torch.complex64: {"tensor(float)"},
    torch.complex128: {"tensor(double)"},
}

# _OPTIONAL_ONNX_DTYPE_STR 是一个集合，包含所有可能的 optional(XXX) 类型字符串，用于 ONNX 的表示
_OPTIONAL_ONNX_DTYPE_STR: Set[str] = {
    f"optional({value})"
    for value_set in _TORCH_DTYPE_TO_COMPATIBLE_ONNX_TYPE_STRINGS.values()
    for value in value_set
}

# _PYTHON_TYPE_TO_TORCH_DTYPE 将 Python 原生类型映射为对应的 torch 数据类型
_PYTHON_TYPE_TO_TORCH_DTYPE = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float32,
    complex: torch.complex64,
}

# _COMPLEX_TO_FLOAT 是一个字典，将 torch 复数类型映射到对应的浮点数类型
_COMPLEX_TO_FLOAT: Dict[torch.dtype, torch.dtype] = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,  # NOTE: ORT doesn't support torch.float64
}

# _SYM_TYPE_TO_TORCH_DTYPE 将 torch 符号类型映射为对应的 torch 数据类型
_SYM_TYPE_TO_TORCH_DTYPE = {
    torch.SymInt: torch.int64,
    torch.SymFloat: torch.float32,
    torch.SymBool: torch.bool,
}

# _SCALAR_TYPE_TO_TORCH_DTYPE 将各种标量 Python 类型映射为对应的 torch 数据类型
_SCALAR_TYPE_TO_TORCH_DTYPE: Dict[Type, torch.dtype] = {
    **_PYTHON_TYPE_TO_TORCH_DTYPE,
    **_SYM_TYPE_TO_TORCH_DTYPE,
}

# _TORCH_DTYPE_TO_ABBREVIATION 将 torch 数据类型映射为其对应的缩写字符串
_TORCH_DTYPE_TO_ABBREVIATION = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.float8_e4m3fn: "e4m3fn",
    torch.float8_e4m3fnuz: "e4m3fnuz",
    torch.float8_e5m2: "f8e5m2",
    torch.float8_e5m2fnuz: "e5m2fnuz",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

# _TORCH_DTYPE_TO_NUMPY_DTYPE 将 torch 数据类型映射为其对应的 NumPy 数据类型
_TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: numpy.float16,
    torch.float32: numpy.float32,
    torch.float64: numpy.float64,
    torch.uint8: numpy.uint8,
}
    # 将torch中的数据类型映射到对应的numpy数据类型
    torch.int8: numpy.int8,
    torch.int16: numpy.int16,
    torch.int32: numpy.int32,
    torch.int64: numpy.longlong,
    torch.bool: numpy.bool_,
# 定义一个字典，将 ONNX 的张量元素类型映射到对应的 Torch 数据类型
_ONNX_TENSOR_ELEMENT_TYPE_TO_TORCH_DTYPE = {
    onnx.TensorProto.FLOAT: torch.float32,  # 对应 ONNX 的 FLOAT 类型映射为 Torch 的 float32
    onnx.TensorProto.FLOAT16: torch.float16,  # 对应 ONNX 的 FLOAT16 类型映射为 Torch 的 float16
    onnx.TensorProto.FLOAT8E5M2: torch.float8_e5m2,  # 对应 ONNX 的 FLOAT8E5M2 类型映射为 Torch 的 float8_e5m2
    onnx.TensorProto.FLOAT8E5M2FNUZ: torch.float8_e5m2fnuz,  # 对应 ONNX 的 FLOAT8E5M2FNUZ 类型映射为 Torch 的 float8_e5m2fnuz
    onnx.TensorProto.FLOAT8E4M3FN: torch.float8_e4m3fn,  # 对应 ONNX 的 FLOAT8E4M3FN 类型映射为 Torch 的 float8_e4m3fn
    onnx.TensorProto.FLOAT8E4M3FNUZ: torch.float8_e4m3fnuz,  # 对应 ONNX 的 FLOAT8E4M3FNUZ 类型映射为 Torch 的 float8_e4m3fnuz
    onnx.TensorProto.DOUBLE: torch.float64,  # 对应 ONNX 的 DOUBLE 类型映射为 Torch 的 float64
    onnx.TensorProto.BOOL: torch.bool,  # 对应 ONNX 的 BOOL 类型映射为 Torch 的 bool
    onnx.TensorProto.UINT8: torch.uint8,  # 对应 ONNX 的 UINT8 类型映射为 Torch 的 uint8
    onnx.TensorProto.INT8: torch.int8,  # 对应 ONNX 的 INT8 类型映射为 Torch 的 int8
    onnx.TensorProto.INT16: torch.int16,  # 对应 ONNX 的 INT16 类型映射为 Torch 的 int16
    onnx.TensorProto.INT32: torch.int32,  # 对应 ONNX 的 INT32 类型映射为 Torch 的 int32
    onnx.TensorProto.INT64: torch.int64,  # 对应 ONNX 的 INT64 类型映射为 Torch 的 int64
}

# 定义一个字典，将 Torch 的数据类型映射回对应的 ONNX 张量元素类型
_TORCH_DTYPE_TO_ONNX_TENSOR_ELEMENT_TYPE = {
    value: key for key, value in _ONNX_TENSOR_ELEMENT_TYPE_TO_TORCH_DTYPE.items()
}

# 定义一个类型别名，表示可能的符号化数值类型，包括 Torch 中的 SymInt, SymFloat, SymBool 类型
SYM_VALUE_TYPE = Union[torch.SymInt, torch.SymFloat, torch.SymBool]

# 定义一个类型别名，表示可能的元数据数值类型，包括 FakeTensor, 符号化数值类型 SYM_VALUE_TYPE, int, float, bool 等
META_VALUE_TYPE = Union[fake_tensor.FakeTensor, SYM_VALUE_TYPE, int, float, bool]

# 定义一个类型别名，表示节点参数的基本类型集合
BaseArgumentTypes = Union[
    str,
    int,
    float,
    bool,
    complex,
    torch.dtype,
    torch.Tensor,
    torch.device,
    torch.memory_format,
    torch.layout,
    torch._ops.OpOverload,
]

# 定义一个类型别名，表示节点的参数类型，可以是元组、列表、字典、切片、range、"torch.fx.Node" 或者 BaseArgumentTypes 类型的可选值
Argument = Optional[
    Union[
        Tuple[Any, ...],  # 实际上是 Argument，但是 mypy 不能表示递归类型
        List[Any],  # 实际上是 Argument
        Dict[str, Any],  # 实际上是 Argument
        slice,  # Slice[Argument, Argument, Argument]，但是 slice 不是 typing 中的模板类型
        range,
        "torch.fx.Node",
        BaseArgumentTypes,
    ]
]
```