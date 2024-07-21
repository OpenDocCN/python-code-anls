# `.\pytorch\torch\_awaits\__init__.py`

```py
# 从 __future__ 模块导入 annotations 特性，使得类型注解的解析更加严格
from __future__ import annotations

# 导入必要的类型
from typing import cast, Callable, Generic, Type, TypeVar

# 导入 torch 库
import torch

# 定义公开的模块接口列表
__all__ = ['Await']

# 定义类型变量 W
W = TypeVar("W")

# 定义元类 _PyAwaitMeta，继承自 torch._C._Await 类和 Generic 泛型类
class _PyAwaitMeta(type(torch._C._Await), type(Generic)):  # type: ignore[misc, no-redef]
    pass

# 定义类 _Await，继承自 torch._C._Await 类和 Generic 泛型类，使用 _PyAwaitMeta 元类
class _Await(torch._C._Await, Generic[W], metaclass=_PyAwaitMeta):
    r"""
    对 ``torch._C.Await`` 进行包装，封装了可调用对象的延迟执行。
    所有操作都使用函数 ``torch.jit._awaitable``、``torch.jit._awaitable_wait``、``torch.jit._awaitable_nowait``。

    Torch 脚本操作：
    ``torch.jit._awaitable(func, *args)``
    创建 ``Await[W]`` 对象，其中 W 是 func 的返回类型。

    返回：
    ``torch.jit._awaitable_wait(Await[W])``
    返回函数调用的结果，该函数在 ``_awaitable`` 中指定，带有指定的参数。

    返回：
        函数调用的结果类型为 ``W``。结果由 ``Await[W]`` 持有，并在所有后续的 ``_awaitable_wait`` 调用中返回。

    ``torch.jit._awaitable_nowait(W)``
    返回：
        使用指定结果的平凡 ``Await[W]`` 对象。

    仅在急切模式下：
    ``fn() -> Callable[Tuple[Any], W]``
    返回：
        在 ``_awaitable`` 中指定的 Python 函数 ``func``。

    ``args() -> Tuple[Any]``
    返回：
        在 ``_awaitable`` 中指定的 Python 参数。

    ``is_nowait() -> _bool``
    返回：
        如果此对象是通过 ``_awaitable_nowait`` 调用创建的（平凡的 `Await[W]`），则返回 ``True``。

    在急切模式下，``Await[W]`` 可以被用作 ``W``，即可以调用 W 的属性，在调用 ``_awaitable_wait()`` 时会自动添加。
    """
    pass
```