# `.\pytorch\torchgen\api\types\types.py`

```py
"""
Where should I add a new type? `types_base.py` vs `types.py`

This file defines data model classes for torchgen typing system, as well as some base types such as int32_t.

`types.py` defines ATen Tensor type and some c10 types, along with signatures that use these types.

The difference between these two files, is `types_base.py` should be implementation-agnostic, meaning it shouldn't
contain any type definition that is tight to a specific C++ library (e.g., ATen), so that it can be easily reused
if we want to generate code for another C++ library.

Add new types to `types.py` if these types are ATen/c10 related.
Add new types to `types_base.py` if they are basic and not attached to ATen/c10.
"""

from __future__ import annotations

from dataclasses import dataclass

# 导入基础类型定义从 `types_base.py`
from torchgen.api.types.types_base import (
    BaseCppType,
    BaseCType,
    boolT,
    byteT,
    charT,
    CType,
    doubleT,
    floatT,
    int32T,
    longT,
    shortT,
)
# 导入基础类型的列表
TENSOR_LIST_LIKE_CTYPES = [
    "at::TensorList",
    "const c10::List<::std::optional<at::Tensor>> &",
    "const at::ITensorListRef &",
]

# 定义新的类型，以及它们的初始化
halfT = BaseCppType("at", "Half")
complexHalfT = BaseCppType(
    "c10", "complex<c10::Half>"
)  # stuffing template param here is an abuse
complexFloatT = BaseCppType("c10", "complex<float>")
complexDoubleT = BaseCppType("c10", "complex<double>")
bfloat16T = BaseCppType("at", "BFloat16")
float8_e5m2T = BaseCppType("at", "Float8_e5m2")
float8_e5m2fnuzT = BaseCppType("at", "Float8_e5m2fnuz")
float8_e4m3fnT = BaseCppType("at", "Float8_e4m3fn")
float8_e4m3fnuzT = BaseCppType("at", "Float8_e4m3fnuz")
stringT = BaseCppType("c10", "string_view")
generatorT = BaseCppType("at", "Generator")
scalarTypeT = BaseCppType("at", "ScalarType")
tensorT = BaseCppType("at", "Tensor")
optionalTensorRefT = BaseCppType("at", "OptionalTensorRef")
tensorListT = BaseCppType("at", "TensorList")
iTensorListRefT = BaseCppType("at", "ITensorListRef")
iOptTensorListRefT = BaseCppType("at", "IOptTensorListRef")
dimnameT = BaseCppType("at", "Dimname")
dimnameListT = BaseCppType("at", "DimnameList")
dimVectorT = BaseCppType("at", "DimVector")
layoutT = BaseCppType("at", "Layout")
deviceT = BaseCppType("at", "Device")
deviceIndexT = BaseCppType("at", "DeviceIndex")
scalarT = BaseCppType("at", "Scalar")
optionalScalarRefT = BaseCppType("at", "OptionalScalarRef")
memoryFormatT = BaseCppType("at", "MemoryFormat")
qschemeT = BaseCppType("at", "QScheme")
storageT = BaseCppType("at", "Storage")
streamT = BaseCppType("at", "Stream")
intArrayRefT = BaseCppType("at", "IntArrayRef")
optionalIntArrayRefT = BaseCppType("at", "OptionalIntArrayRef")
optionalSymIntArrayRefT = BaseCppType("at", "OptionalSymIntArrayRef")
tensorOptionsT = BaseCppType("at", "TensorOptions")
typeAndSizeT = BaseCppType("torch::autograd::generated", "TypeAndSize")
tensorGeometryT = BaseCppType("at", "TensorGeometry")
SymIntT = BaseCppType("c10", "SymInt")
symIntArrayRefT = BaseCppType("c10", "SymIntArrayRef")

# 定义了一个变量 symIntArrayRefT，表示一个基于 BaseCppType 类的实例，代表类型 "c10::SymIntArrayRef"

# Types representing template parameters.  Technically, we probably shouldn't
# represent them this way in codegen, but it was pretty convenient.
# 定义了几个变量用于表示模板参数类型，这种方式可能在代码生成中不是最佳实践，但非常方便。
scalar_t = BaseCppType("", "scalar_t")
opmath_t = BaseCppType("", "opmath_t")

# 定义了一个字典 ScalarTypeToCppMapping，将 ScalarType 枚举值映射到对应的 BaseCppType 实例。
ScalarTypeToCppMapping: dict[ScalarType, BaseCppType] = {
    ScalarType.Byte: byteT,
    ScalarType.Char: charT,
    ScalarType.Short: shortT,
    ScalarType.Int: int32T,
    ScalarType.Long: longT,
    ScalarType.Half: halfT,
    ScalarType.Float: floatT,
    ScalarType.Double: doubleT,
    ScalarType.ComplexHalf: complexHalfT,
    ScalarType.ComplexFloat: complexFloatT,
    ScalarType.ComplexDouble: complexDoubleT,
    ScalarType.Bool: boolT,
    ScalarType.Float8_e5m2: float8_e5m2T,
    ScalarType.Float8_e5m2fnuz: float8_e5m2fnuzT,
    ScalarType.Float8_e4m3fn: float8_e4m3fnT,
    ScalarType.Float8_e4m3fnuz: float8_e4m3fnuzT,
}

# 定义了一个字典 BaseTypeToCppMapping，将 BaseTy 枚举值映射到对应的 BaseCppType 实例。
BaseTypeToCppMapping: dict[BaseTy, BaseCppType] = {
    BaseTy.int: longT,
    BaseTy.float: doubleT,
    BaseTy.bool: boolT,
    BaseTy.str: stringT,
    BaseTy.Generator: generatorT,
    BaseTy.ScalarType: scalarTypeT,
    BaseTy.Tensor: tensorT,
    BaseTy.Dimname: dimnameT,
    BaseTy.DimVector: dimVectorT,
    BaseTy.Layout: layoutT,
    BaseTy.Device: deviceT,
    BaseTy.DeviceIndex: deviceIndexT,
    BaseTy.Scalar: scalarT,
    BaseTy.MemoryFormat: memoryFormatT,
    BaseTy.QScheme: qschemeT,
    BaseTy.Storage: storageT,
    BaseTy.Stream: streamT,
    BaseTy.SymInt: SymIntT,
}

# CTypes encode C++ type structure as needed for translation.
# 下面是几个用于表示不同类型的数据结构的类定义，它们都继承自 CType 类。

@dataclass(frozen=True)
class OptionalCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::optional<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"::std::optional<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> CType:
        return OptionalCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ListCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"c10::List<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"c10::List<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> CType:
        return ListCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ArrayRefCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"at::ArrayRef<{self.elem.cpp_type()}>"

    def cpp_type_registration_declarations(self) -> str:
        return f"ArrayRef<{self.elem.cpp_type_registration_declarations()}>"

    def remove_const_ref(self) -> CType:
        return ArrayRefCType(self.elem.remove_const_ref())
# 使用 @dataclass 装饰器定义一个冻结的数据类 VectorizedCType，表示向量化的 C 类型
@dataclass(frozen=True)
class VectorizedCType(CType):
    # elem 字段表示该向量化类型的元素类型，通常为特定的基本 C 类型，如 float, double 等
    # 在模板化上下文中，scalar_t 也是一个常见的参数
    elem: BaseCType

    # 返回该向量化 C 类型在 C++ 中的表示形式，使用 at::vec::Vectorized 包装元素类型的 C++ 表示
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return f"at::vec::Vectorized<{self.elem.cpp_type()}>"

    # 返回该向量化 C 类型在 C++ 中的注册声明，当前方法未实现，抛出 NotImplementedError 异常
    def cpp_type_registration_declarations(self) -> str:
        raise NotImplementedError

    # 移除常量引用，返回自身作为 CType 实例
    def remove_const_ref(self) -> CType:
        return self
```