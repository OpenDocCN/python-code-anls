# `.\pytorch\torchgen\executorch\api\types\types.py`

```
# 从未来导入注解，以支持注解中的类型引用
from __future__ import annotations

# 导入用于定义数据类的模块
from dataclasses import dataclass

# 导入各种类型定义和绑定相关的类和函数
from torchgen.api.types import (
    BaseCppType,   # 基本的 C++ 类型
    BaseCType,    # 基本的 C 类型
    Binding,      # 表示绑定（变量绑定到类型的关系）
    boolT,        # 布尔类型
    CType,        # C 类型
    doubleT,      # 双精度浮点数类型
    Expr,         # 表达式
    longT,        # 长整型类型
    MutRefCType,  # 可变引用 C 类型
    NamedCType,   # 命名 C 类型
)
# 导入基础类型相关的模块
from torchgen.model import BaseTy


# 定义 Half 类型的 C++ 类型
halfT = BaseCppType("torch::executor", "Half")
# 定义 BFloat16 类型的 C++ 类型
bfloat16T = BaseCppType("torch::executor", "BFloat16")
# 定义 string_view 类型的 C++ 类型
stringT = BaseCppType("torch::executor", "string_view")
# 定义 ScalarType 类型的 C++ 类型
scalarTypeT = BaseCppType("torch::executor", "ScalarType")
# 定义 Tensor 类型的 C++ 类型
tensorT = BaseCppType("torch::executor", "Tensor")
# 定义 TensorList 类型的 C++ 类型
tensorListT = BaseCppType("torch::executor", "TensorList")
# 定义 Scalar 类型的 C++ 类型
scalarT = BaseCppType("torch::executor", "Scalar")
# 定义 MemoryFormat 类型的 C++ 类型
memoryFormatT = BaseCppType("torch::executor", "MemoryFormat")
# 定义 IntArrayRef 类型的 C++ 类型
intArrayRefT = BaseCppType("torch::executor", "IntArrayRef")
# 定义 optional 类型的 C++ 类型
optionalT = BaseCppType("torch::executor", "optional")
# 定义 KernelRuntimeContext 类型的 C++ 类型
contextT = BaseCppType("torch::executor", "KernelRuntimeContext")

# 创建一个表达式对象，表示一个 context 变量
contextExpr = Expr(
    expr="context",  # 表达式的内容是 context
    type=NamedCType(name="context", type=MutRefCType(BaseCType(contextT))),
)

# 创建一个绑定对象，表示一个 context 参数
contextArg = Binding(
    name="context",            # 绑定的名称为 context
    nctype=contextExpr.type,   # 绑定的类型为 contextExpr 的类型
    argument=None,             # 参数为 None（未指定具体参数）
    default=None,              # 默认值为 None（未指定默认值）
)

# 定义基础类型到 C++ 类型的映射关系字典
BaseTypeToCppMapping: dict[BaseTy, BaseCppType] = {
    BaseTy.int: longT,            # 整数映射到 longT 类型
    BaseTy.float: doubleT,        # 浮点数映射到 doubleT 类型
    BaseTy.bool: boolT,           # 布尔值映射到 boolT 类型
    BaseTy.str: stringT,          # 字符串映射到 stringT 类型
    BaseTy.ScalarType: scalarTypeT,  # ScalarType 映射到 scalarTypeT 类型
    BaseTy.Tensor: tensorT,       # Tensor 映射到 tensorT 类型
    BaseTy.Scalar: scalarT,       # Scalar 映射到 scalarT 类型
    BaseTy.MemoryFormat: memoryFormatT,  # MemoryFormat 映射到 memoryFormatT 类型
}

# 定义 OptionalCType 数据类，表示可选类型的 C 类型
@dataclass(frozen=True)
class OptionalCType(CType):
    elem: CType

    # 返回 C++ 中的类型字符串表示，包含可选类型的声明
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # 不递归去除引用
        return f"torch::executor::optional<{self.elem.cpp_type()}>"

    # 返回 C++ 中的类型注册声明字符串表示，包含可选类型的声明
    def cpp_type_registration_declarations(self) -> str:
        return f"torch::executor::optional<{self.elem.cpp_type_registration_declarations()}>"

    # 移除常量引用
    def remove_const_ref(self) -> CType:
        return OptionalCType(self.elem.remove_const_ref())


# 定义 ArrayRefCType 数据类，表示数组引用类型的 C 类型
@dataclass(frozen=True)
class ArrayRefCType(CType):
    elem: CType

    # 返回 C++ 中的类型字符串表示，包含数组引用类型的声明
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # 不递归去除引用
        return f"torch::executor::ArrayRef<{self.elem.cpp_type()}>"

    # 返回 C++ 中的类型注册声明字符串表示，包含数组引用类型的声明
    def cpp_type_registration_declarations(self) -> str:
        return f"torch::executor::ArrayRef<{self.elem.cpp_type_registration_declarations()}>"

    # 移除常量引用
    def remove_const_ref(self) -> CType:
        return ArrayRefCType(self.elem.remove_const_ref())
```