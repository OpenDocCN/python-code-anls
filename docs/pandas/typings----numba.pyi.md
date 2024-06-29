# `D:\src\scipysrc\pandas\typings\numba.pyi`

```
# pyright: reportIncompleteStub = false
# 引入必要的类型引用，包括 Any, Callable, Literal, overload
from typing import (
    Any,
    Callable,
    Literal,
    overload,
)

# 导入 numba 库
import numba

# 导入 pandas 库中的 F 类型
from pandas._typing import F

# 定义一个特殊方法 __getattr__，用于获取对象的属性（未完整实现）
def __getattr__(name: str) -> Any: ...

# 定义函数装饰器 jit 的重载版本，用于 JIT 编译函数或者方法
@overload
def jit(signature_or_function: F) -> F: ...

# 另一个 jit 的重载版本，接受多种参数：
@overload
def jit(
    signature_or_function: str
    | list[str]
    | numba.core.types.abstract.Type
    | list[numba.core.types.abstract.Type] = ...,
    locals: dict = ...,  # TODO: Mapping of local variable names to Numba types
    cache: bool = ...,
    pipeline_class: numba.compiler.CompilerBase = ...,
    boundscheck: bool | None = ...,
    *,
    nopython: bool = ...,
    forceobj: bool = ...,
    looplift: bool = ...,
    error_model: Literal["python", "numpy"] = ...,
    inline: Literal["never", "always"] | Callable = ...,
    # TODO: If a callable is provided it will be called with the call expression
    # node that is requesting inlining, the caller's IR and callee's IR as
    # arguments, it is expected to return Truthy as to whether to inline.
    target: Literal["cpu", "gpu", "npyufunc", "cuda"] = ...,  # deprecated
    nogil: bool = ...,
    parallel: bool = ...,
) -> Callable[[F], F]: ...
    
# 将 jit 装饰器赋值给别名 njit 和 generated_jit
njit = jit
generated_jit = jit
```