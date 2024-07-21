# `.\pytorch\torchgen\api\translate.py`

```py
# 从未来导入注释功能，用于支持类型注释的前向兼容性
from __future__ import annotations

# 导入类型提示相关模块
from typing import NoReturn, Sequence

# 导入具体的类型定义，用于程序绑定与转换
from torchgen.api.types import (
    ArrayRefCType,
    BaseCType,
    Binding,
    boolT,
    ConstRefCType,
    deviceT,
    Expr,
    intArrayRefT,
    iOptTensorListRefT,
    layoutT,
    ListCType,
    longT,
    memoryFormatT,
    MutRefCType,
    NamedCType,
    opmath_t,
    OptionalCType,
    optionalIntArrayRefT,
    optionalScalarRefT,
    optionalSymIntArrayRefT,
    optionalTensorRefT,
    scalar_t,
    scalarT,
    scalarTypeT,
    SpecialArgName,
    symIntArrayRefT,
    SymIntT,
    tensorOptionsT,
    tensorT,
    VectorCType,
)

# 本文件实现了一个小型程序合成引擎，用于实现不同 API 之间的转换。
#
# 在本文件中，关键的数据类型是 NamedCType，简称命名的 C++ 语义类型。
# NamedCType 表示一个 C++ 类型，以及关于其代表内容的语义信息。
# 例如，考虑参数 "bool pin_memory"；它的普通 C++ 类型是 "bool"，
# 但其 C++ 语义类型还跟踪此参数表示的是 "pin_memory"；
# 因此在需要 "pin_memory" 的上下文中，不能随意使用其他的布尔值！
#
# 翻译器接受一个所需的 NamedCType 列表，然后根据给定的绑定，决定如何构造
# 含有这些 NamedCType 的表达式。许多表达式是简单的（需要一个 Tensor other；
# 在作用域中有一个 Tensor other）；其他一些则更复杂，可能需要打包/解包。
# 一些非平凡动作的例子包括：
#   - 需要 "dtype" 绑定？也许在上下文中并没有 "dtype"，而是有了 "options"，
#     需要从中提取它。（聚合）
#   - 需要 "context" 绑定？也许在上下文中并没有 "context"，需要从 "dtype"、
#     "device" 等构造它。（分散）
#   - 需要 "memory_format" 绑定？实际上，它可以从 "memory_format" 和 "options"
#     中获取，因此必须确保它们保持一致。（连接）

# 定义一个 NamedCType，表示 "options"，是 ConstRefCType 类型，基础类型为 tensorOptionsT
options_ctype = NamedCType("options", ConstRefCType(BaseCType(tensorOptionsT)))

# 定义一个 NamedCType，表示 "out"，是 ConstRefCType 类型，基础类型为 tensorT
out_tensor_ctype = NamedCType("out", ConstRefCType(BaseCType(tensorT)))

# 定义一个 VectorCType，表示 BaseCType 为 longT 的向量类型
longVec_ctype = VectorCType(BaseCType(longT))

# 定义一个 VectorCType，表示 BaseCType 为 SymIntT 的向量类型
longSymVec_ctype = VectorCType(BaseCType(SymIntT))

# 定义一个 OptionalCType，表示其内部是 BaseCType 为 longT 的向量类型
optionalLongVec_ctype = OptionalCType(VectorCType(BaseCType(longT)))

# 定义一个 OptionalCType，表示其内部是 BaseCType 为 scalarT 的类型
optionalScalar_ctype = OptionalCType(BaseCType(scalarT))

# 定义一个 OptionalCType，表示其内部是 BaseCType 为 tensorT 的类型
optionalTensor_ctype = OptionalCType(BaseCType(tensorT))


class UnsatError(RuntimeError):
    # 表示在程序合成过程中出现的未满足条件的错误
    pass


# 给定一组已绑定的上下文和目标绑定，合成一个使用只有上下文绑定的表达式列表
# （bindings），这些表达式都满足目标类型要求。如果要为如下函数生成代码：
#   void f({args}) {
#     g({exprs}); // g 是另一个 API
#   }
# 则可能需要使用此函数来生成 "exprs"。
#
# 典型情况下，一个 Bindings 列表很方便获取（通常调用 arguments() 之类的方法获取它们）；
# 但技术上来说，你只需要更少的信息：对于 'bindings'，一个（无序的）Exprs 列表就足够了；
# 类似地，对于 'goals'，一个（有序的）NamedCType goals 列表就足够了。如果你在做更复杂的事情，
# 比如在上下文中跟踪绑定的集合，可能会发现使用这些更小的类型更方便。
def translate(
    bindings: Sequence[Expr | Binding],  # 输入参数 bindings，可以是 Expr 或 Binding 的序列
    goals: Sequence[NamedCType | Binding],  # 输入参数 goals，可以是 NamedCType 或 Binding 的序列
    *,
    method: bool = False,  # 是否在 Tensor 方法内部生成代码，默认为 False
    allow_expensive_conversions: bool = False,  # 是否允许昂贵的转换，默认为 False
) -> list[Expr]:  # 返回类型为 Expr 对象的列表
    binding_exprs: list[Expr] = []  # 初始化一个空列表，用于存储转换后的绑定表达式对象
    for b in bindings:  # 遍历 bindings
        if isinstance(b, Binding):  # 如果 b 是 Binding 对象
            binding_exprs.append(  # 添加转换后的表达式对象到列表中
                Expr(
                    expr=b.name,  # 使用 Binding 的名称创建表达式对象
                    type=b.nctype,  # 使用 Binding 的类型创建表达式对象
                )
            )
        else:  # 如果 b 不是 Binding 对象
            binding_exprs.append(b)  # 直接添加到列表中

    goal_ctypes: list[NamedCType] = []  # 初始化一个空列表，用于存储转换后的目标类型对象
    for g in goals:  # 遍历 goals
        if isinstance(g, Binding):  # 如果 g 是 Binding 对象
            goal_ctypes.append(g.nctype)  # 添加 Binding 的类型到目标类型列表中
        else:  # 如果 g 不是 Binding 对象
            goal_ctypes.append(g)  # 直接添加到列表中

    # 将所有绑定添加到上下文中
    ctx: dict[NamedCType, str] = {}  # 初始化一个空字典，用于存储命名类型到字符串的映射
    # 如果生成的代码位于 Tensor 方法内部，则添加隐式绑定
    if method:  # 如果 method 为 True
        ctx[  # 添加一个命名类型到字符串的映射
            NamedCType("self", MutRefCType(BaseCType(tensorT)))  # 创建一个命名类型对象
        ] = "const_cast<Tensor&>(*this)"  # 将字符串映射到命名类型对象
        ctx[  # 添加另一个命名类型到字符串的映射
            NamedCType("self", ConstRefCType(BaseCType(tensorT)))  # 创建另一个命名类型对象
        ] = "const_cast<Tensor&>(*this)"  # 将字符串映射到命名类型对象
        # This is better!  Byte-for-byte compat
        # ctx[NamedCType("self", ConstRefCType(BaseCType(tensorT)))] = "*this"

    def unsat(goal: NamedCType) -> NoReturn:  # 定义一个内部函数 unsat，参数为 NamedCType，返回类型为 NoReturn
        ctx_desc = "\n".join(  # 创建描述上下文的字符串，使用换行符连接
            f"  {t.cpp_type()} {t.name}; // {e}" for t, e in ctx.items()  # 遍历上下文字典，生成每个条目的描述字符串
        )
        raise UnsatError(  # 抛出未满足错误异常
            f"""
Failed to synthesize the expression "{goal.cpp_type()} {goal.name}".
When I failed, the following bindings were available in the context:

{ctx_desc}

This probably means there is a missing rule in the rules of torchgen.api.translate.
Check this module for more information.
"""
        )

    # 一个简陋的回溯搜索实现。之所以说它简陋，是因为它通过堆栈进行回溯（不好的想法！），
    # 大多数情况下尽量避免回溯。特别地，如果 direct=True，我们不会尝试进行任何复杂的合成，
    # 只做简单的转换（例如，“T a”对于“const T& a”是可以的）。因此，此函数中的所有现有规则
    # 都只是尝试立即解决，如果不起作用则退出。
    return [Expr(solve(g, direct=False), g) for g in goal_ctypes]  # 返回一个列表，包含目标类型的 Expr 对象和目标类型本身
```