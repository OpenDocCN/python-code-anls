# `.\pytorch\torchgen\context.py`

```py
# 导入 __future__ 模块的 annotations 功能，使得可以在类型提示中使用 TypeVar
from __future__ import annotations

# 导入 contextlib 模块，用于创建上下文管理器
import contextlib
# 导入 functools 模块，用于创建高阶函数
import functools
# 导入 typing 模块中的类型
from typing import Any, Callable, Iterator, List, Optional, Tuple, TypeVar, Union

# 导入 torchgen.local 模块中的 local 对象
import torchgen.local as local
# 从 torchgen.model 模块导入多个类和类型
from torchgen.model import (
    BackendIndex,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
)
# 从 torchgen.utils 模块中导入 context 和 S, T 类型
from torchgen.utils import context, S, T

# TypeVar 用于泛型编程，定义多种类型的占位符

# 定义 F 类型变量，可以是 NativeFunction、NativeFunctionsGroup、NativeFunctionsViewGroup 的子类
F = TypeVar(
    "F",
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    Union[NativeFunction, NativeFunctionsGroup],
    Union[NativeFunction, NativeFunctionsViewGroup],
)

# 定义 F2 类型变量，可以是 NativeFunction、NativeFunctionsGroup、Optional[NativeFunction] 等类型
F2 = TypeVar(
    "F2",
    NativeFunction,
    NativeFunctionsGroup,
    Optional[NativeFunction],
    bool,
    str,
)

# 定义 F3 类型变量，可以是 Tuple[NativeFunction, Any] 或 List[NativeFunction]
F3 = TypeVar("F3", Tuple[NativeFunction, Any], List[NativeFunction])


# 定义 native_function_manager 上下文管理器函数，根据不同类型的 g 参数设置上下文环境
@contextlib.contextmanager
def native_function_manager(
    g: NativeFunctionsGroup | NativeFunctionsViewGroup | NativeFunction,
) -> Iterator[None]:
    if isinstance(g, NativeFunctionsGroup):
        # 如果 g 是 NativeFunctionsGroup 类型，则使用其 out 属性
        f = g.out
    elif isinstance(g, NativeFunctionsViewGroup):
        # 如果 g 是 NativeFunctionsViewGroup 类型，则使用其 view 属性
        f = g.view
    else:
        # 否则直接使用 g 本身
        f = g
    # 设置上下文环境，关联错误到特定的本地函数
    with context(lambda: f"in native_functions.yaml line {f.loc}:\n  {f.func}"):
        # 使用 local.parametrize 设置参数化上下文环境
        with local.parametrize(
            use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors,
            use_ilistref_for_tensor_lists=f.part_of_structured_group,
        ):
            yield


# 定义装饰器函数 with_native_function，用于将 func 包装成一个新函数，设置适当的上下文管理器
def with_native_function(func: Callable[[F], T]) -> Callable[[F], T]:
    @functools.wraps(func)
    def wrapper(f: F) -> T:
        # 使用 native_function_manager 设置上下文环境
        with native_function_manager(f):
            return func(f)

    return wrapper


# 定义装饰器函数 with_native_function_and，用于将 func 包装成一个新函数，同时接受两个参数 f 和 f2
def with_native_function_and(func: Callable[[F, F2], T]) -> Callable[[F, F2], T]:
    @functools.wraps(func)
    def wrapper(f: F, f2: F2) -> T:
        # 假设第一个 native_function 是具有适当上下文的函数
        with native_function_manager(f):
            return func(f, f2)

    return wrapper


# 定义方法装饰器函数 method_with_native_function，用于将 func 包装成一个新方法，接受参数 slf 和 f
def method_with_native_function(func: Callable[[S, F], T]) -> Callable[[S, F], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F) -> T:
        # 使用 native_function_manager 设置上下文环境
        with native_function_manager(f):
            return func(slf, f)

    return wrapper


# 定义方法装饰器函数 method_with_nested_native_function，用于将 func 包装成一个新方法，接受参数 slf 和 f3
def method_with_nested_native_function(
    func: Callable[[S, F3], T]
) -> Callable[[S, F3], T]:
    @functools.wraps(func)
    # 返回包装后的 func 函数
    def wrapper(slf: S, f3: F3) -> T:
        # 函数体暂未提供，应该在后续代码中继续完善

    return wrapper
    # 定义一个装饰器函数 wrapper，接受三个参数：slf 是类型 S 的对象，f 是类型 F3 的对象，返回类型为 T
    def wrapper(slf: S, f: F3) -> T:
        # 使用 native_function_manager 对 f[0] 进行管理，确保执行期间函数的正确行为
        with native_function_manager(f[0]):
            # 调用 func 函数，传入 slf 和 f 作为参数，并返回其结果
            return func(slf, f)

    # 返回装饰器函数 wrapper 本身
    return wrapper
# 为那些显式接受 BackendIndex 参数的函数提供便利的装饰器，而不是间接通过闭包接受
def with_native_function_and_index(
    func: Callable[[F, BackendIndex], T]
) -> Callable[[F, BackendIndex], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_index: BackendIndex) -> T:
        # 使用 native_function_manager 管理本地函数 f 的上下文
        with native_function_manager(f):
            return func(f, backend_index)

    return wrapper


# 为那些显式接受 BackendIndices 字典的函数提供便利的装饰器
def with_native_function_and_indices(
    func: Callable[[F, dict[DispatchKey, BackendIndex]], T]
) -> Callable[[F, dict[DispatchKey, BackendIndex]], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_indices: dict[DispatchKey, BackendIndex]) -> T:
        # 使用 native_function_manager 管理本地函数 f 的上下文
        with native_function_manager(f):
            return func(f, backend_indices)

    return wrapper
```