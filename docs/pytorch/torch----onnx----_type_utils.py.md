# `.\pytorch\torch\onnx\_type_utils.py`

```py
# mypy: allow-untyped-defs
# 导入ONNX、JIT和torch类型转换和操作的实用工具
"""Utilities for converting and operating on ONNX, JIT and torch types."""
from __future__ import annotations

import enum  # 导入枚举类型模块
import typing  # 导入类型提示相关模块
from typing import Dict, Literal, Optional, Union  # 导入类型提示中的字典、字面量、可选和联合类型

import torch  # 导入PyTorch模块
from torch._C import _onnx as _C_onnx  # 导入torch._C中的_onnx模块
from torch.onnx import errors  # 导入torch.onnx中的errors模块
from torch.onnx._internal import _beartype  # 导入torch.onnx._internal中的_beartype模块

if typing.TYPE_CHECKING:
    # Hack to help mypy to recognize torch._C.Value
    from torch import _C  # noqa: F401  # 如果是类型检查阶段，则从torch中导入_C模块

ScalarName = Literal[
    "Byte",
    "Char",
    "Double",
    "Float",
    "Half",
    "Int",
    "Long",
    "Short",
    "Bool",
    "ComplexHalf",
    "ComplexFloat",
    "ComplexDouble",
    "QInt8",
    "QUInt8",
    "QInt32",
    "BFloat16",
    "Float8E5M2",
    "Float8E4M3FN",
    "Float8E5M2FNUZ",
    "Float8E4M3FNUZ",
]
# 定义ScalarName类型为字面量，包含多种标量类型的字符串表示

TorchName = Literal[
    "bool",
    "uint8_t",
    "int8_t",
    "double",
    "float",
    "half",
    "int",
    "int64_t",
    "int16_t",
    "complex32",
    "complex64",
    "complex128",
    "qint8",
    "quint8",
    "qint32",
    "bfloat16",
    "float8_e5m2",
    "float8_e4m3fn",
    "float8_e5m2fnuz",
    "float8_e4m3fnuz",
]
# 定义TorchName类型为字面量，包含多种Torch标量类型的字符串表示

class JitScalarType(enum.IntEnum):
    """Scalar types defined in torch.

    Use ``JitScalarType`` to convert from torch and JIT scalar types to ONNX scalar types.

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> # xdoctest: +IGNORE_WANT("win32 has different output")
        >>> JitScalarType.from_value(torch.ones(1, 2)).onnx_type()
        TensorProtoDataType.FLOAT

        >>> JitScalarType.from_value(torch_c_value_with_type_float).onnx_type()
        TensorProtoDataType.FLOAT

        >>> JitScalarType.from_dtype(torch.get_default_dtype).onnx_type()
        TensorProtoDataType.FLOAT

    """

    # Order defined in https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
    UINT8 = 0  # 定义UINT8为0
    INT8 = enum.auto()  # 通过enum.auto()生成INT8的枚举值
    INT16 = enum.auto()  # 通过enum.auto()生成INT16的枚举值
    INT = enum.auto()  # 通过enum.auto()生成INT的枚举值
    INT64 = enum.auto()  # 通过enum.auto()生成INT64的枚举值
    HALF = enum.auto()  # 通过enum.auto()生成HALF的枚举值
    FLOAT = enum.auto()  # 通过enum.auto()生成FLOAT的枚举值
    DOUBLE = enum.auto()  # 通过enum.auto()生成DOUBLE的枚举值
    COMPLEX32 = enum.auto()  # 通过enum.auto()生成COMPLEX32的枚举值
    COMPLEX64 = enum.auto()  # 通过enum.auto()生成COMPLEX64的枚举值
    COMPLEX128 = enum.auto()  # 通过enum.auto()生成COMPLEX128的枚举值
    BOOL = enum.auto()  # 通过enum.auto()生成BOOL的枚举值
    QINT8 = enum.auto()  # 通过enum.auto()生成QINT8的枚举值
    QUINT8 = enum.auto()  # 通过enum.auto()生成QUINT8的枚举值
    QINT32 = enum.auto()  # 通过enum.auto()生成QINT32的枚举值
    BFLOAT16 = enum.auto()  # 通过enum.auto()生成BFLOAT16的枚举值
    FLOAT8E5M2 = enum.auto()  # 通过enum.auto()生成FLOAT8E5M2的枚举值
    FLOAT8E4M3FN = enum.auto()  # 通过enum.auto()生成FLOAT8E4M3FN的枚举值
    FLOAT8E5M2FNUZ = enum.auto()  # 通过enum.auto()生成FLOAT8E5M2FNUZ的枚举值
    FLOAT8E4M3FNUZ = enum.auto()  # 通过enum.auto()生成FLOAT8E4M3FNUZ的枚举值
    UNDEFINED = enum.auto()  # 通过enum.auto()生成UNDEFINED的枚举值

    @classmethod
    @_beartype.beartype
    def _from_name(
        cls, name: Union[ScalarName, TorchName, Optional[str]]
        # 类方法，根据输入的标量类型名称，返回对应的JitScalarType枚举值
    ) -> JitScalarType:
        """Convert a JIT scalar type or torch type name to ScalarType.

        Note: DO NOT USE this API when `name` comes from a `torch._C.Value.type()` calls.
            A "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in several scenarios where shape info is not present.
            Instead use `from_value` API which is safer.

        Args:
            name: JIT scalar type name (Byte) or torch type name (uint8_t).

        Returns:
            JitScalarType

        Raises:
           OnnxExporterError: if name is not a valid scalar type name or if it is None.
        """
        if name is None:
            raise errors.OnnxExporterError("Scalar type name cannot be None")
        if valid_scalar_name(name):
            return _SCALAR_NAME_TO_TYPE[name]  # type: ignore[index]
        if valid_torch_name(name):
            return _TORCH_NAME_TO_SCALAR_TYPE[name]  # type: ignore[index]

        raise errors.OnnxExporterError(f"Unknown torch or scalar type: '{name}'")

    @classmethod
    @_beartype.beartype
    def from_dtype(cls, dtype: Optional[torch.dtype]) -> JitScalarType:
        """Convert a torch dtype to JitScalarType.

        Note: DO NOT USE this API when `dtype` comes from a `torch._C.Value.type()` calls.
            A "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in several scenarios where shape info is not present.
            Instead use `from_value` API which is safer.

        Args:
            dtype: A torch.dtype to create a JitScalarType from

        Returns:
            JitScalarType

        Raises:
            OnnxExporterError: if dtype is not a valid torch.dtype or if it is None.
        """
        if dtype not in _DTYPE_TO_SCALAR_TYPE:
            raise errors.OnnxExporterError(f"Unknown dtype: {dtype}")
        return _DTYPE_TO_SCALAR_TYPE[dtype]

    @classmethod
    @_beartype.beartype
    def from_onnx_type(
        cls, onnx_type: Optional[Union[int, _C_onnx.TensorProtoDataType]]
    ) -> JitScalarType:
        """Convert a ONNX data type to JitScalarType.

        Args:
            onnx_type: A torch._C._onnx.TensorProtoDataType to create a JitScalarType from

        Returns:
            JitScalarType

        Raises:
            OnnxExporterError: if dtype is not a valid torch.dtype or if it is None.
        """
        if onnx_type not in _ONNX_TO_SCALAR_TYPE:
            raise errors.OnnxExporterError(f"Unknown onnx_type: {onnx_type}")
        return _ONNX_TO_SCALAR_TYPE[typing.cast(_C_onnx.TensorProtoDataType, onnx_type)]

    @classmethod
    @_beartype.beartype
    def from_value(
        cls, value: Union[None, torch._C.Value, torch.Tensor], default=None
    ) -> JitScalarType:
        """Convert a torch value to JitScalarType.

        Args:
            value: A torch value to create a JitScalarType from
            default: Optional default value if `value` is None

        Returns:
            JitScalarType
        """
        if value is None:
            if default is None:
                raise errors.OnnxExporterError("Value cannot be None")
            value = default
        return cls.from_dtype(value.type())
    ) -> JitScalarType:
        """Create a JitScalarType from an value's scalar type.
        
        Args:
            value: An object to fetch scalar type from.
            default: The JitScalarType to return if a valid scalar cannot be fetched from value
        
        Returns:
            JitScalarType.
        
        Raises:
            OnnxExporterError: if value does not have a valid scalar type and default is None.
            SymbolicValueError: when value.type()'s info are empty and default is None
        """
        
        if not isinstance(value, (torch._C.Value, torch.Tensor)) or (
            isinstance(value, torch._C.Value) and value.node().mustBeNone()
        ):
            # 如果 value 不是 torch._C.Value 或者 torch.Tensor 对象，或者是一个必须为 None 的 torch._C.Value 对象
            # 则返回默认的 JitScalarType 值
            if default is None:
                raise errors.OnnxExporterError(
                    "value must be either torch._C.Value or torch.Tensor objects."
                )
            elif not isinstance(default, JitScalarType):
                raise errors.OnnxExporterError(
                    "default value must be a JitScalarType object."
                )
            return default
        
        # 每种值类型都有自己存储标量类型的方式
        if isinstance(value, torch.Tensor):
            # 如果 value 是 torch.Tensor 对象，则从其 dtype 创建 JitScalarType
            return cls.from_dtype(value.dtype)
        if isinstance(value.type(), torch.ListType):
            try:
                # 如果 value 的类型是 torch.ListType，则尝试从其元素类型的 dtype 创建 JitScalarType
                return cls.from_dtype(value.type().getElementType().dtype())
            except RuntimeError:
                # 如果运行时出错，则从其元素类型的名称创建 JitScalarType
                return cls._from_name(str(value.type().getElementType()))
        if isinstance(value.type(), torch._C.OptionalType):
            if value.type().getElementType().dtype() is None:
                # 如果 value 的类型是 torch._C.OptionalType 且其元素类型的 dtype 为 None，则返回默认值或引发异常
                if isinstance(default, JitScalarType):
                    return default
                raise errors.OnnxExporterError(
                    "default value must be a JitScalarType object."
                )
            return cls.from_dtype(value.type().getElementType().dtype())
        
        scalar_type = None
        if value.node().kind() != "prim::Constant" or not isinstance(
            value.type(), torch._C.NoneType
        ):
            # 如果 value 的节点类型不是 "prim::Constant" 或其类型不是 torch._C.NoneType，则获取其标量类型
            scalar_type = value.type().scalarType()
        
        if scalar_type is not None:
            # 如果成功获取标量类型，则根据标量类型名称创建 JitScalarType
            return cls._from_name(scalar_type)
        
        # 当一切尝试失败时，返回默认值或引发异常
        if default is not None:
            return default
        raise errors.SymbolicValueError(
            f"Cannot determine scalar type for this '{type(value.type())}' instance and "
            "a default value was not provided.",
            value,
        )

    @_beartype.beartype
    def scalar_name(self) -> ScalarName:
        """Convert a JitScalarType to a JIT scalar type name."""
        # 将 JitScalarType 转换为 JIT 标量类型名称
        return _SCALAR_TYPE_TO_NAME[self]

    @_beartype.beartype
    def torch_name(self) -> TorchName:
        """Convert a JitScalarType to a torch type name."""
        # 将 JitScalarType 转换为 torch 类型名称
        return _SCALAR_TYPE_TO_TORCH_NAME[self]
    # 使用装饰器_beartype.beartype来装饰dtype方法，确保输入参数类型符合预期
    @_beartype.beartype
    # 定义dtype方法，返回一个torch数据类型(torch.dtype)，将JitScalarType转换为torch数据类型
    def dtype(self) -> torch.dtype:
        """Convert a JitScalarType to a torch dtype."""
        # 返回_SCALAR_TYPE_TO_DTYPE字典中self对应的torch数据类型
        return _SCALAR_TYPE_TO_DTYPE[self]

    # 使用装饰器_beartype.beartype来装饰onnx_type方法，确保输入参数类型符合预期
    @_beartype.beartype
    # 定义onnx_type方法，返回一个_ONNX.TensorProtoDataType数据类型，将JitScalarType转换为ONNX数据类型
    def onnx_type(self) -> _C_onnx.TensorProtoDataType:
        """Convert a JitScalarType to an ONNX data type."""
        # 如果self不在_SCALAR_TYPE_TO_ONNX字典中，则抛出错误
        if self not in _SCALAR_TYPE_TO_ONNX:
            raise errors.OnnxExporterError(
                f"Scalar type {self} cannot be converted to ONNX"
            )
        # 返回_SCALAR_TYPE_TO_ONNX字典中self对应的ONNX数据类型
        return _SCALAR_TYPE_TO_ONNX[self]

    # 使用装饰器_beartype.beartype来装饰onnx_compatible方法，确保输入参数类型符合预期
    @_beartype.beartype
    # 定义onnx_compatible方法，返回一个布尔值，判断当前JitScalarType是否与ONNX兼容
    def onnx_compatible(self) -> bool:
        """Return whether this JitScalarType is compatible with ONNX."""
        # 返回条件：self在_SCALAR_TYPE_TO_ONNX中且不等于JitScalarType.UNDEFINED和JitScalarType.COMPLEX32
        return (
            self in _SCALAR_TYPE_TO_ONNX
            and self != JitScalarType.UNDEFINED
            and self != JitScalarType.COMPLEX32
        )
# 使用装饰器进行类型检查，确保输入的标量名称是有效的 JIT 标量类型名称
@_beartype.beartype
def valid_scalar_name(scalar_name: Union[ScalarName, str]) -> bool:
    """Return whether the given scalar name is a valid JIT scalar type name."""
    return scalar_name in _SCALAR_NAME_TO_TYPE


# 使用装饰器进行类型检查，确保输入的 torch 名称是有效的 torch 类型名称
@_beartype.beartype
def valid_torch_name(torch_name: Union[TorchName, str]) -> bool:
    """Return whether the given torch name is a valid torch type name."""
    return torch_name in _TORCH_NAME_TO_SCALAR_TYPE


# 定义从 JIT 标量类型到标量名称的映射字典
# https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
_SCALAR_TYPE_TO_NAME: Dict[JitScalarType, ScalarName] = {
    JitScalarType.BOOL: "Bool",
    JitScalarType.UINT8: "Byte",
    JitScalarType.INT8: "Char",
    JitScalarType.INT16: "Short",
    JitScalarType.INT: "Int",
    JitScalarType.INT64: "Long",
    JitScalarType.HALF: "Half",
    JitScalarType.FLOAT: "Float",
    JitScalarType.DOUBLE: "Double",
    JitScalarType.COMPLEX32: "ComplexHalf",
    JitScalarType.COMPLEX64: "ComplexFloat",
    JitScalarType.COMPLEX128: "ComplexDouble",
    JitScalarType.QINT8: "QInt8",
    JitScalarType.QUINT8: "QUInt8",
    JitScalarType.QINT32: "QInt32",
    JitScalarType.BFLOAT16: "BFloat16",
    JitScalarType.FLOAT8E5M2: "Float8E5M2",
    JitScalarType.FLOAT8E4M3FN: "Float8E4M3FN",
    JitScalarType.FLOAT8E5M2FNUZ: "Float8E5M2FNUZ",
    JitScalarType.FLOAT8E4M3FNUZ: "Float8E4M3FNUZ",
    JitScalarType.UNDEFINED: "Undefined",
}

# 定义从标量名称到 JIT 标量类型的映射字典
_SCALAR_NAME_TO_TYPE: Dict[ScalarName, JitScalarType] = {
    v: k for k, v in _SCALAR_TYPE_TO_NAME.items()
}

# 定义从 JIT 标量类型到 torch 标量名称的映射字典
_SCALAR_TYPE_TO_TORCH_NAME: Dict[JitScalarType, TorchName] = {
    JitScalarType.BOOL: "bool",
    JitScalarType.UINT8: "uint8_t",
    JitScalarType.INT8: "int8_t",
    JitScalarType.INT16: "int16_t",
    JitScalarType.INT: "int",
    JitScalarType.INT64: "int64_t",
    JitScalarType.HALF: "half",
    JitScalarType.FLOAT: "float",
    JitScalarType.DOUBLE: "double",
    JitScalarType.COMPLEX32: "complex32",
    JitScalarType.COMPLEX64: "complex64",
    JitScalarType.COMPLEX128: "complex128",
    JitScalarType.QINT8: "qint8",
    JitScalarType.QUINT8: "quint8",
    JitScalarType.QINT32: "qint32",
    JitScalarType.BFLOAT16: "bfloat16",
    JitScalarType.FLOAT8E5M2: "float8_e5m2",
    JitScalarType.FLOAT8E4M3FN: "float8_e4m3fn",
    JitScalarType.FLOAT8E5M2FNUZ: "float8_e5m2fnuz",
    JitScalarType.FLOAT8E4M3FNUZ: "float8_e4m3fnuz",
}

# 定义从 torch 标量名称到 JIT 标量类型的映射字典
_TORCH_NAME_TO_SCALAR_TYPE: Dict[TorchName, JitScalarType] = {
    v: k for k, v in _SCALAR_TYPE_TO_TORCH_NAME.items()
}

# 定义从 JIT 标量类型到 ONNX 标量数据类型的映射字典
_SCALAR_TYPE_TO_ONNX = {
    JitScalarType.BOOL: _C_onnx.TensorProtoDataType.BOOL,
    JitScalarType.UINT8: _C_onnx.TensorProtoDataType.UINT8,
    JitScalarType.INT8: _C_onnx.TensorProtoDataType.INT8,
    JitScalarType.INT16: _C_onnx.TensorProtoDataType.INT16,
    JitScalarType.INT: _C_onnx.TensorProtoDataType.INT32,
    JitScalarType.INT64: _C_onnx.TensorProtoDataType.INT64,
    JitScalarType.HALF: _C_onnx.TensorProtoDataType.FLOAT16,
    JitScalarType.FLOAT: _C_onnx.TensorProtoDataType.FLOAT,
}
    # 将 JitScalarType.DOUBLE 映射到 _C_onnx.TensorProtoDataType.DOUBLE
    JitScalarType.DOUBLE: _C_onnx.TensorProtoDataType.DOUBLE,
    # 将 JitScalarType.COMPLEX64 映射到 _C_onnx.TensorProtoDataType.COMPLEX64
    JitScalarType.COMPLEX64: _C_onnx.TensorProtoDataType.COMPLEX64,
    # 将 JitScalarType.COMPLEX128 映射到 _C_onnx.TensorProtoDataType.COMPLEX128
    JitScalarType.COMPLEX128: _C_onnx.TensorProtoDataType.COMPLEX128,
    # 将 JitScalarType.BFLOAT16 映射到 _C_onnx.TensorProtoDataType.BFLOAT16
    JitScalarType.BFLOAT16: _C_onnx.TensorProtoDataType.BFLOAT16,
    # 将 JitScalarType.UNDEFINED 映射到 _C_onnx.TensorProtoDataType.UNDEFINED
    JitScalarType.UNDEFINED: _C_onnx.TensorProtoDataType.UNDEFINED,
    # 将 JitScalarType.COMPLEX32 映射到 _C_onnx.TensorProtoDataType.UNDEFINED
    JitScalarType.COMPLEX32: _C_onnx.TensorProtoDataType.UNDEFINED,
    # 将 JitScalarType.QINT8 映射到 _C_onnx.TensorProtoDataType.INT8
    JitScalarType.QINT8: _C_onnx.TensorProtoDataType.INT8,
    # 将 JitScalarType.QUINT8 映射到 _C_onnx.TensorProtoDataType.UINT8
    JitScalarType.QUINT8: _C_onnx.TensorProtoDataType.UINT8,
    # 将 JitScalarType.QINT32 映射到 _C_onnx.TensorProtoDataType.INT32
    JitScalarType.QINT32: _C_onnx.TensorProtoDataType.INT32,
    # 将 JitScalarType.FLOAT8E5M2 映射到 _C_onnx.TensorProtoDataType.FLOAT8E5M2
    JitScalarType.FLOAT8E5M2: _C_onnx.TensorProtoDataType.FLOAT8E5M2,
    # 将 JitScalarType.FLOAT8E4M3FN 映射到 _C_onnx.TensorProtoDataType.FLOAT8E4M3FN
    JitScalarType.FLOAT8E4M3FN: _C_onnx.TensorProtoDataType.FLOAT8E4M3FN,
    # 将 JitScalarType.FLOAT8E5M2FNUZ 映射到 _C_onnx.TensorProtoDataType.FLOAT8E5M2FNUZ
    JitScalarType.FLOAT8E5M2FNUZ: _C_onnx.TensorProtoDataType.FLOAT8E5M2FNUZ,
    # 将 JitScalarType.FLOAT8E4M3FNUZ 映射到 _C_onnx.TensorProtoDataType.FLOAT8E4M3FNUZ
    JitScalarType.FLOAT8E4M3FNUZ: _C_onnx.TensorProtoDataType.FLOAT8E4M3FNUZ,
# 将 _SCALAR_TYPE_TO_ONNX 字典的键值对颠倒，形成从值到键的映射
_ONNX_TO_SCALAR_TYPE = {v: k for k, v in _SCALAR_TYPE_TO_ONNX.items()}

# 根据 PyTorch 源码中定义的标量类型映射表，将 JitScalarType 转换为对应的 torch 数据类型
# 参考源码位置：https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_dtypes.cpp
_SCALAR_TYPE_TO_DTYPE = {
    JitScalarType.BOOL: torch.bool,
    JitScalarType.UINT8: torch.uint8,
    JitScalarType.INT8: torch.int8,
    JitScalarType.INT16: torch.short,
    JitScalarType.INT: torch.int,
    JitScalarType.INT64: torch.int64,
    JitScalarType.HALF: torch.half,
    JitScalarType.FLOAT: torch.float,
    JitScalarType.DOUBLE: torch.double,
    JitScalarType.COMPLEX32: torch.complex32,
    JitScalarType.COMPLEX64: torch.complex64,
    JitScalarType.COMPLEX128: torch.complex128,
    JitScalarType.QINT8: torch.qint8,
    JitScalarType.QUINT8: torch.quint8,
    JitScalarType.QINT32: torch.qint32,
    JitScalarType.BFLOAT16: torch.bfloat16,
    JitScalarType.FLOAT8E5M2: torch.float8_e5m2,
    JitScalarType.FLOAT8E4M3FN: torch.float8_e4m3fn,
    JitScalarType.FLOAT8E5M2FNUZ: torch.float8_e5m2fnuz,
    JitScalarType.FLOAT8E4M3FNUZ: torch.float8_e4m3fnuz,
}

# 将 _SCALAR_TYPE_TO_DTYPE 字典的键值对颠倒，形成从值到键的映射
_DTYPE_TO_SCALAR_TYPE = {v: k for k, v in _SCALAR_TYPE_TO_DTYPE.items()}
```