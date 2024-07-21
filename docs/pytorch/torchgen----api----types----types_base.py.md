# `.\pytorch\torchgen\api\types\types_base.py`

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from torchgen.model import Argument, SelfArgument, TensorOptionsArguments


# An ArgName is just the str name of the argument in schema;
# but in some special circumstances, we may add a little extra
# context.  The Enum SpecialArgName covers all of these cases;
# grep for their construction sites to see when they can occur.
class SpecialArgName(Enum):
    possibly_redundant_memory_format = auto()
    # Enum defining special argument names that may carry additional context.


ArgName = Union[str, SpecialArgName]


# This class shouldn't be created directly; instead, use/create one of the singletons below.
@dataclass(frozen=True)
class BaseCppType:
    ns: str | None  # Namespace of the C++ type, None if not applicable
    name: str       # Name of the C++ type

    def __str__(self) -> str:
        if self.ns is None or self.ns == "":
            return self.name
        return f"{self.ns}::{self.name}"
    # Returns a string representation of the C++ type, including its namespace if applicable.


# The set of all non-templated, valid, fully-qualified names of C++ types that are used in the codegen.
# Templated types get their own dataclass, mainly to make namespace parsing easier.
byteT = BaseCppType("", "uint8_t")     # Define C++ type for 8-bit unsigned integer
charT = BaseCppType("", "int8_t")      # Define C++ type for 8-bit signed integer
shortT = BaseCppType("", "int16_t")    # Define C++ type for 16-bit signed integer
int32T = BaseCppType("", "int32_t")    # Define C++ type for 32-bit signed integer
longT = BaseCppType("", "int64_t")     # Define C++ type for 64-bit signed integer
doubleT = BaseCppType("", "double")    # Define C++ type for double precision floating point
floatT = BaseCppType("", "float")      # Define C++ type for single precision floating point
boolT = BaseCppType("", "bool")        # Define C++ type for boolean
voidT = BaseCppType("", "void")        # Define C++ type for void


class CType(ABC):
    @abstractmethod
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        raise NotImplementedError
    # Abstract base class defining methods for C++ type representation.

    @abstractmethod
    def cpp_type_registration_declarations(self) -> str:
        raise NotImplementedError
    # Abstract method for generating C++ type registration declarations.

    @abstractmethod
    def remove_const_ref(self) -> CType:
        return self
    # Abstract method to remove const and reference qualifiers from the type.


@dataclass(frozen=True)
class BaseCType(CType):
    type: BaseCppType  # Base C++ type

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return str(self.type)
    # Returns the string representation of the base C++ type.

    # For BC reasons, we don't want to introduce at:: namespaces to RegistrationDeclarations.yaml
    # 定义一个方法，生成 C++ 类型注册声明的字符串表示形式
    def cpp_type_registration_declarations(self) -> str:
        # 返回去除 "at::" 前缀后的 self.type 对象的字符串表示形式
        return str(self.type).replace("at::", "")
    
    # 定义一个方法，返回自身对象，用于移除常量引用
    def remove_const_ref(self) -> CType:
        # 直接返回自身，没有修改操作
        return self
@dataclass(frozen=True)
# 定义不可变的 ConstRefCType 类，继承自 CType 类
class ConstRefCType(CType):
    elem: CType

    # 定义返回 C++ 类型字符串的方法，支持是否去除引用
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            # 如果 strip_ref 为 True，则递归调用元素的 cpp_type 方法，去除引用
            return self.elem.cpp_type(strip_ref=strip_ref)
        # 返回 const 元素类型引用的字符串表示
        return f"const {self.elem.cpp_type()} &"

    # 定义返回 C++ 类型注册声明的字符串方法
    def cpp_type_registration_declarations(self) -> str:
        # 返回 const 元素类型注册声明的字符串表示
        return f"const {self.elem.cpp_type_registration_declarations()} &"

    # 定义去除 const 和引用的方法，返回元素的类型
    def remove_const_ref(self) -> CType:
        return self.elem.remove_const_ref()


@dataclass(frozen=True)
# 定义不可变的 VectorCType 类，继承自 CType 类
class VectorCType(CType):
    elem: CType

    # 定义返回 C++ 类型字符串的方法，支持是否去除引用，但不递归处理 strip_ref
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return f"::std::vector<{self.elem.cpp_type()}>"

    # 定义返回 C++ 类型注册声明的字符串方法
    def cpp_type_registration_declarations(self) -> str:
        return f"::std::vector<{self.elem.cpp_type_registration_declarations()}>"

    # 定义去除 const 和引用的方法，返回元素类型去除 const 和引用后的 VectorCType 对象
    def remove_const_ref(self) -> CType:
        return VectorCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
# 定义不可变的 ArrayCType 类，继承自 CType 类，包含元素类型和数组大小
class ArrayCType(CType):
    elem: CType
    size: int

    # 定义返回 C++ 类型字符串的方法，支持是否去除引用，但不递归处理 strip_ref
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return f"::std::array<{self.elem.cpp_type()},{self.size}>"

    # 定义返回 C++ 类型注册声明的字符串方法
    def cpp_type_registration_declarations(self) -> str:
        return f"::std::array<{self.elem.cpp_type_registration_declarations()},{self.size}>"

    # 定义去除 const 和引用的方法，返回元素类型去除 const 和引用后的 ArrayCType 对象
    def remove_const_ref(self) -> CType:
        return ArrayCType(self.elem.remove_const_ref(), self.size)


@dataclass(frozen=True)
# 定义不可变的 TupleCType 类，继承自 CType 类，包含一个类型元素列表
class TupleCType(CType):
    elems: list[CType]

    # 定义返回 C++ 类型字符串的方法，支持是否去除引用，但不递归处理 strip_ref
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return f'::std::tuple<{",".join([e.cpp_type() for e in self.elems])}>'

    # 定义返回 C++ 类型注册声明的字符串方法
    def cpp_type_registration_declarations(self) -> str:
        return f'::std::tuple<{",".join([e.cpp_type_registration_declarations() for e in self.elems])}>'

    # 定义去除 const 和引用的方法，返回包含每个元素去除 const 和引用后的类型的 TupleCType 对象
    def remove_const_ref(self) -> CType:
        return TupleCType([e.remove_const_ref() for e in self.elems])


@dataclass(frozen=True)
# 定义不可变的 MutRefCType 类，继承自 CType 类，表示带有引用的可变类型
class MutRefCType(CType):
    elem: CType

    # 定义返回 C++ 类型字符串的方法，支持是否去除引用
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            # 如果 strip_ref 为 True，则递归调用元素的 cpp_type 方法，去除引用
            return self.elem.cpp_type(strip_ref=strip_ref)
        # 返回元素类型引用的字符串表示
        return f"{self.elem.cpp_type()} &"

    # 定义返回 C++ 类型注册声明的字符串方法
    def cpp_type_registration_declarations(self) -> str:
        return f"{self.elem.cpp_type_registration_declarations()} &"

    # 定义去除 const 和引用的方法，返回元素的类型
    def remove_const_ref(self) -> CType:
        return self.elem.remove_const_ref()


# 定义 NamedCType 类，表示命名的 C++ 语义类型，包含类型名称和类型本身
@dataclass(frozen=True)
class NamedCType:
    name: ArgName
    type: CType
    # 返回与当前类型相关的 C++ 类型的字符串表示
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return self.type.cpp_type(strip_ref=strip_ref)

    # 由于兼容性原因，不希望在 RegistrationDeclarations.yaml 文件中引入 at:: 命名空间
    # TODO: 当最终移除这部分代码时，删除此函数！
    # 返回与类型注册声明相关的 C++ 类型的字符串表示
    def cpp_type_registration_declarations(self) -> str:
        return self.type.cpp_type_registration_declarations()

    # 返回移除当前类型常量和引用后的 NamedCType 对象
    def remove_const_ref(self) -> NamedCType:
        return NamedCType(self.name, self.type.remove_const_ref())

    # 返回一个具有指定名称的 NamedCType 对象，类型与当前对象相同
    def with_name(self, name: str) -> NamedCType:
        return NamedCType(name, self.type)
@dataclass(frozen=True)
class Binding:
    # 表示 C++ 形式参数的任何绑定点
    # 我们不区分不同 API 的绑定点；
    # 而是所有重要区别都编码在 NamedCType 中，
    # 您可以使用它来确定给定的 Binding 是否适合另一个上下文中使用。
    # （参见 torchgen.api.translate）

    name: str  # 绑定的名称
    nctype: NamedCType  # 命名类型
    argument: Argument | TensorOptionsArguments | SelfArgument  # 参数类型，可以是多种类型的联合
    # TODO: 可能不要在这里表示默认值
    default: str | None = None  # 默认值，可以是字符串或者 None

    def rename(self, name: str) -> Binding:
        # 返回一个新的 Binding 对象，名称被重命名为指定的 name
        return Binding(
            name=name,
            nctype=self.nctype,
            argument=self.argument,
            default=self.default,
        )

    @property
    def type(self) -> str:
        # 返回该绑定的类型的 C++ 表示
        return self.nctype.cpp_type()

    def no_default(self) -> Binding:
        # 返回一个新的 Binding 对象，没有默认值
        return Binding(
            name=self.name,
            nctype=self.nctype,
            default=None,
            argument=self.argument,
        )

    def decl(self, *, func_ptr_cast: bool = False) -> str:
        # 根据 func_ptr_cast 参数返回声明字符串
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"

        # 如果 func_ptr_cast 为真，只需要返回类型
        if func_ptr_cast:
            return f"{self.type}"
        else:
            return f"{self.type} {self.name}{mb_default}"

    def decl_registration_declarations(self) -> str:
        # 用于 RegistrationDeclarations.yaml 的声明字符串表示
        type_s = self.nctype.cpp_type_registration_declarations()
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"
        return f"{type_s} {self.name}{mb_default}"

    def defn(self) -> str:
        # 返回定义字符串，格式为 "<类型> <名称>"
        return f"{self.type} {self.name}"

    def with_name(self, name: str) -> Binding:
        # 返回一个新的 Binding 对象，名称被替换为指定的 name
        return Binding(
            name=name, nctype=self.nctype, argument=self.argument, default=self.default
        )


@dataclass(frozen=True)
class Expr:
    # 表达式是一个 C++ 表达式，它具有表示其语法的 C++ 字符串，
    # 以及指定其提供内容的 CType。

    expr: str  # 表达式的字符串表示
    type: NamedCType  # 表达式的类型
```