# `.\pytorch\torchgen\api\ufunc.py`

```py
# 从未来模块导入 annotations，这样可以使用类型提示中的 NamedCType 类
from __future__ import annotations

# 导入 dataclass 模块，用于创建数据类
from dataclasses import dataclass

# 导入 torchgen.api.types 模块中的 api_types，用于类型引用
import torchgen.api.types as api_types

# 从 torchgen.api 模块中导入 cpp 和 structured，用于 API 调用
from torchgen.api import cpp, structured

# 从 torchgen.api.types 模块中导入多个类和常量，用于类型引用
from torchgen.api.types import (
    ArgName,
    BaseCppType,
    BaseCType,
    Binding,
    ConstRefCType,
    CType,
    NamedCType,
    scalarT,
)

# 从 torchgen.model 模块中导入多个类，用于模型相关定义
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    DispatchKey,
    FunctionSchema,
    NativeFunctionsGroup,
    Type,
)

# 函数定义，根据函数模式和分派键生成内核名称字符串
def schema_kernel_name(func: FunctionSchema, dispatch_key: DispatchKey) -> str:
    # 断言函数是否为输出函数，如果不是则抛出异常
    assert func.is_out_fn(), "ufunc.kernel_name should only be invoked on out schemas"
    # 返回生成的内核名称字符串
    return f"ufunc_{func.name.name}_{dispatch_key}"


# 函数定义，生成内核名称字符串，基于 NativeFunctionsGroup 和分派键
def kernel_name(g: NativeFunctionsGroup, dispatch_key: DispatchKey) -> str:
    # 调用 schema_kernel_name 函数，传入输出函数和分派键生成内核名称
    return schema_kernel_name(g.out.func, dispatch_key)


# 函数定义，生成分发存根类型，用于处理除张量外的所有参数类型
# 在 CPU 环境下使用
def dispatchstub_type(t: Type, *, binds: ArgName) -> NamedCType | None:
    # 调用 cpp 模块中的 valuetype_type 函数，生成指定类型的值类型
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    # 如果返回值不为空，则直接返回
    if r is not None:
        return r

    # 如果参数类型是标量，则返回带有常量引用类型的 NamedCType 对象
    if t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
    # 如果参数类型是张量，则返回空值
    elif t == BaseType(BaseTy.Tensor):
        return None
    else:
        # 抛出异常，表示遇到未识别的参数类型
        raise AssertionError(f"unrecognized type {repr(t)}")


# 函数定义，生成操作数数学类型，根据标量类型转换为对应的操作数类型
def opmath_type(scalar_t: BaseCppType) -> BaseCppType:
    # 如果标量类型为 api_types 模块中的标量类型，则返回对应的操作数数学类型
    if scalar_t == api_types.scalar_t:
        return api_types.opmath_t
    # 抛出未实现异常，表示遇到未处理的标量类型
    raise NotImplementedError


# 函数定义，生成仿函数构造函数类型，用于处理各种类型的参数
# 在 CUDA 环境下使用
def ufunctor_ctor_type(t: Type, *, binds: ArgName, scalar_t: BaseCppType) -> NamedCType:
    # 调用 cpp 模块中的 valuetype_type 函数，生成指定类型的值类型
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    # 如果返回值不为空，则直接返回
    if r is not None:
        return r

    # 如果参数类型是标量或张量，则返回带有操作数数学类型的 NamedCType 对象
    if t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, BaseCType(opmath_type(scalar_t)))
    elif t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(opmath_type(scalar_t)))
    else:
        # 抛出异常，表示遇到未识别的参数类型
        raise AssertionError(f"unrecognized type {repr(t)}")


# 函数定义，生成仿函数应用函数类型，用于处理张量类型的参数
# 在 CUDA 环境下使用
# 实际上，CPU 环境下也可以使用
def ufunctor_apply_type(
    t: Type, *, binds: ArgName, scalar_t: BaseCppType
) -> NamedCType:
    # 如果参数类型是张量，则返回带有标量类型的 NamedCType 对象
    if t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(scalar_t))
    else:
        # 抛出异常，表示遇到未识别的参数类型
        raise AssertionError(f"unrecognized type {repr(t)}")


# 函数定义，生成仿函数类型，用于用户编写的仿函数模板函数
# 所有操作都在计算类型中完成
# 在 CUDA 环境下，compute_t 是 opmath_t 类型，在 CPU 环境下是 scalar_t 类型
def ufunc_type(t: Type, *, binds: ArgName, compute_t: CType) -> NamedCType:
    # 调用 cpp 模块中的 valuetype_type 函数，生成指定类型的值类型
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    # 如果r不为None，则直接返回r，终止函数执行
    if r is not None:
        return r

    # 如果t的类型是BaseType(BaseTy.Scalar)，则返回一个NamedCType对象
    # 这个对象使用给定的绑定参数binds和计算函数compute_t
    elif t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, compute_t)
    
    # 如果t的类型是BaseType(BaseTy.Tensor)，同样返回一个NamedCType对象
    # 使用相同的绑定参数binds和计算函数compute_t
    elif t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, compute_t)
    
    # 如果以上条件都不满足，抛出一个AssertionError异常
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")
# 构建用于构造函数的绑定对象，即用于 CUDA 的函数对象构造器
def ufunctor_ctor_argument(a: Argument, scalar_t: BaseCppType) -> Binding:
    return Binding(
        nctype=ufunctor_ctor_type(a.type, binds=a.name, scalar_t=scalar_t),
        name=a.name,
        default=None,
        argument=a,
    )


# 构建用于应用函数的绑定对象，即用于 CUDA 的函数对象应用器
def ufunctor_apply_argument(a: Argument, scalar_t: BaseCppType) -> Binding:
    return Binding(
        nctype=ufunctor_apply_type(a.type, binds=a.name, scalar_t=scalar_t),
        name=a.name,
        default=None,
        argument=a,
    )


# 构建用于 Ufunc 的绑定对象列表，用于 CUDA 内核计算
def ufunc_argument(a: Argument, compute_t: CType) -> Binding:
    return Binding(
        nctype=ufunc_type(a.type, binds=a.name, compute_t=compute_t),
        name=a.name,
        default=None,
        argument=a,
    )


# 数据类，包含构造函数和应用函数的绑定列表
@dataclass(frozen=True)
class UfunctorBindings:
    ctor: list[Binding]
    apply: list[Binding]


# Ufunctors 是仅适用于 CUDA 的概念，表示函数对象，其中一部分参数在主机端构造，
# 另一部分在设备端应用。例如，
# ctor 指的是构造函数 CUDAFunctorOnSelf_add，apply 指的是 operator() 定义
def ufunctor_arguments(
    g: NativeFunctionsGroup, *, scalar_tensor_idx: int | None, scalar_t: BaseCppType
) -> UfunctorBindings:
    ctor = []
    apply = []
    for a in g.functional.func.arguments.flat_non_out:
        if a.type.is_tensor_like():
            if scalar_tensor_idx == 0:
                # 即使是标量张量索引，也要放入构造函数中
                ctor.append(ufunctor_ctor_argument(a, scalar_t=scalar_t))
                scalar_tensor_idx = None
            else:
                if scalar_tensor_idx is not None:
                    scalar_tensor_idx -= 1
                apply.append(ufunctor_apply_argument(a, scalar_t=scalar_t))
        else:
            ctor.append(ufunctor_ctor_argument(a, scalar_t=scalar_t))
    assert scalar_tensor_idx is None
    return UfunctorBindings(ctor=ctor, apply=apply)


# Ufuncs 是模板函数的内部循环函数，位于 ufunc/add.h 中，执行实际的计算。
# 在本文件中，我们称 T 为 compute_t，由调用者绑定
def ufunc_arguments(g: NativeFunctionsGroup, *, compute_t: CType) -> list[Binding]:
    return [
        ufunc_argument(a, compute_t=compute_t)
        for a in g.functional.func.arguments.flat_non_out
    ]


# Stubs 是 DispatchStub 的跳转，CPU 内核使用它们来获取其矢量化版本。
# 例如，
#
# 使用结构化的二进制函数指针类型 `structured_binary_fn_alpha`，其接受一个 `TensorIteratorBase` 对象和一个标量 `alpha` 作为参数，返回 `void` 类型。
# 声明一个名为 `add_stub` 的函数调度器（dispatch），其函数签名符合 `structured_binary_fn_alpha` 的定义。

def stub_arguments(g: NativeFunctionsGroup) -> list[Binding]:
    # 在给定的原生函数组 `g` 中，返回一个绑定列表，每个绑定描述一个参数。
    # Stubs（存根）会丢弃所有张量类型的参数（因为它们在 `TensorIterator` 参数中是隐式的），并保留其它所有参数。

    return [
        r
        for a in g.out.func.arguments.flat_non_out  # 遍历原生函数组中所有非输出参数的平坦列表
        if not a.type.is_tensor_like()  # 如果参数 `a` 的类型不像张量
        for r in structured.argument(a)  # 对于符合参数 `a` 的结构化参数，生成一个绑定
    ]
```