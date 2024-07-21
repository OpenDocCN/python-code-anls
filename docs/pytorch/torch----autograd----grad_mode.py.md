# `.\pytorch\torch\autograd\grad_mode.py`

```
# 指定允许未标记的函数定义，用于类型检查
# 导入必要的类型
from typing import Any

# 导入 torch 库
import torch

# 从 torch.utils._contextlib 中导入所需的类和方法
from torch.utils._contextlib import (
    _DecoratorContextManager,
    _NoParamDecoratorContextManager,
    F,
)

# 将以下符号添加到模块的公共接口中
__all__ = [
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "inference_mode",
    "set_multithreading_enabled",
]

# 定义 no_grad 类，继承自 _NoParamDecoratorContextManager
class no_grad(_NoParamDecoratorContextManager):
    r"""Context-manager that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.
    There is an exception! All factory functions, or functions that create
    a new Tensor and take a requires_grad kwarg, will NOT be affected by
    this mode.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        No-grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.
        If you want to disable forward AD for a computation, you can unpack
        your dual tensors.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> @torch.no_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> z = doubler(x)
        >>> z.requires_grad
        False
        >>> @torch.no_grad
        ... def tripler(x):
        ...     return x * 3
        >>> z = tripler(x)
        >>> z.requires_grad
        False
        >>> # factory function exception
        >>> with torch.no_grad():
        ...     a = torch.nn.Parameter(torch.rand(10))
        >>> a.requires_grad
        True
    """

    # 初始化方法，检查是否处于脚本化环境，并调用父类初始化方法
    def __init__(self) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.prev = False

    # 进入上下文时调用的方法，记录当前梯度计算状态，并关闭梯度计算
    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    # 退出上下文时调用的方法，恢复之前的梯度计算状态
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_grad_enabled(self.prev)


class enable_grad(_NoParamDecoratorContextManager):
    r"""Context-manager that enables gradient calculation.

    Enables gradient calculation, if it has been disabled via :class:`~no_grad`
    or :class:`~set_grad_enabled`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.
    """
    # 略，未完整提供
    """
    # 进入上下文管理器时调用的方法，用于启用梯度计算
    def __enter__(self) -> None:
        # 记录当前梯度计算状态，并强制启用梯度计算
        self.prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(True)

    # 退出上下文管理器时调用的方法，用于恢复梯度计算状态
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # 恢复之前记录的梯度计算状态
        torch._C._set_grad_enabled(self.prev)
    """
class set_grad_enabled(_DecoratorContextManager):
    r"""Context-manager that sets gradient calculation on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.

    .. note::
        set_grad_enabled is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> is_train = False
        >>> with torch.set_grad_enabled(is_train):
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> _ = torch.set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> _ = torch.set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        False

    """

    def __init__(self, mode: bool) -> None:
        # 保存当前全局的梯度计算开关状态
        self.prev = torch.is_grad_enabled()
        # 设置新的梯度计算开关状态为传入的 mode
        self.mode = mode
        torch._C._set_grad_enabled(mode)  # 调用 C++ 后端设置梯度计算开关的函数

    def __call__(self, orig_func: F) -> F:
        # 恢复为初始化时的全局梯度计算开关状态
        torch._C._set_grad_enabled(self.prev)
        return super().__call__(orig_func)

    def __enter__(self) -> None:
        # 进入上下文时设置梯度计算开关状态为当前实例的 mode
        torch._C._set_grad_enabled(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # 退出上下文时恢复为初始化时的全局梯度计算开关状态
        torch._C._set_grad_enabled(self.prev)

    def clone(self) -> "set_grad_enabled":
        r"""
        Create a copy of this class
        """
        # 创建当前实例的深拷贝
        return self.__class__(self.mode)


class inference_mode(_DecoratorContextManager):
    r"""Context-manager that enables or disables inference mode.

    InferenceMode is a new context manager analogous to :class:`~no_grad`
    to be used when you are certain your operations will have no interactions
    with autograd (e.g., model training). Code run under this mode gets better
    performance by disabling view tracking and version counter bumps. Note that
    unlike some other mechanisms that locally enable or disable grad,
    entering inference_mode also disables to :ref:`forward-mode AD <forward-mode-ad>`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        Inference mode is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.
    Args:
        mode (bool or function): Either a boolean flag whether to enable or
            disable inference mode or a Python function to decorate with
            inference mode enabled

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> import torch
        >>> x = torch.ones(1, 2, 3, requires_grad=True)
        >>> with torch.inference_mode():
        ...     y = x * x
        >>> y.requires_grad
        False
        >>> # xdoctest: +SKIP("want string isnt quite right")
        >>> y._version
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        RuntimeError: Inference tensors do not track version counter.
        >>> @torch.inference_mode()
        ... def func(x):
        ...     return x * x
        >>> out = func(x)
        >>> out.requires_grad
        False
        >>> @torch.inference_mode
        ... def doubler(x):
        ...     return x * 2
        >>> out = doubler(x)
        >>> out.requires_grad
        False

    """
    # 初始化函数，设置推断模式的初始状态
    def __init__(self, mode: bool = True) -> None:
        # 如果当前不在 Torch 脚本化环境中
        if not torch._jit_internal.is_scripting():
            # 调用父类的初始化方法
            super().__init__()
        # 设置当前对象的推断模式属性
        self.mode = mode

    # 创建新的对象实例，根据传入的 mode 参数类型决定返回对象
    def __new__(cls, mode=True):
        # 如果 mode 是布尔类型，则调用父类的 __new__ 方法创建实例
        if isinstance(mode, bool):
            return super().__new__(cls)
        # 如果 mode 是函数，则以函数方式调用当前类构造实例并返回
        return cls()(mode)

    # 进入推断模式的上下文管理器
    def __enter__(self) -> None:
        # 创建推断模式上下文对象，并进入上下文管理器
        self._inference_mode_context = torch._C._InferenceMode(self.mode)
        self._inference_mode_context.__enter__()

    # 退出推断模式的上下文管理器
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # 退出推断模式的上下文管理器
        self._inference_mode_context.__exit__(exc_type, exc_value, traceback)

    # 克隆当前对象的副本
    def clone(self) -> "inference_mode":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)
# 进入推断模式的上下文管理器，设置推断模式
def _enter_inference_mode(mode):
    # 使用 torch._C._InferenceMode 创建推断模式的上下文
    mode_context = torch._C._InferenceMode(mode)
    # 进入推断模式
    mode_context.__enter__()
    # 返回推断模式上下文
    return mode_context


# 退出推断模式的上下文管理器
def _exit_inference_mode(mode):
    # 退出推断模式
    mode.__exit__(None, None, None)


# 控制多线程反向传播开关的上下文管理器
class set_multithreading_enabled(_DecoratorContextManager):
    r"""Context-manager that sets multithreaded backwards on or off.

    ``set_multithreading_enabled`` will enable or disable multithreaded backwards based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable multithreaded backwards (``True``), or disable
                     (``False``).

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    """

    def __init__(self, mode: bool) -> None:
        # 获取当前的多线程反向传播状态
        self.prev = torch._C._is_multithreading_enabled()
        # 设置新的多线程反向传播状态
        torch._C._set_multithreading_enabled(mode)
        # 记录当前的模式
        self.mode = mode

    def __enter__(self) -> None:
        # 进入上下文时无需执行任何操作
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # 恢复之前的多线程反向传播状态
        torch._C._set_multithreading_enabled(self.prev)

    def clone(self) -> "set_multithreading_enabled":
        r"""
        Create a copy of this class
        """
        # 返回当前实例的克隆
        return self.__class__(self.mode)


# 控制视图回放开关的上下文管理器
class _force_original_view_tracking(_DecoratorContextManager):
    r"""Context-manager that sets whether or not to always enable view-replay in autograd.

    ``set_view_replay_enabled`` will enable or disable view-replay based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    When a tensor view is mutated, the autograd engine needs to decide whether or not
    to regenerate the "updated view" by either replaying the chain of views from the updated base,
    or with a single call to as_strided.

    If set_view_replay_enabled is set to True, then autograd will always use view replay.
    Otherwise, it will fall back to its existing logic.

    Args:
        mode (bool): Flag whether to enable view-replay (``True``), or disable
                     (``False``).

    """

    def __init__(self, mode: bool) -> None:
        # 获取当前的视图回放状态
        self.prev = torch._C._is_view_replay_enabled()
        # 设置新的视图回放状态
        torch._C._set_view_replay_enabled(mode)
        # 记录当前的模式
        self.mode = mode

    def __enter__(self) -> None:
        # 进入上下文时无需执行任何操作
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        # 恢复之前的视图回放状态
        torch._C._set_view_replay_enabled(self.prev)

    def clone(self):
        # 返回当前实例的克隆
        return self.__class__(self.mode)


class _unsafe_preserve_version_counter(_DecoratorContextManager):
    r"""DO NOT USE THIS UNLESS YOU KNOW EXACTLY WHAT YOU'RE DOING.

    This context manager can lead to arbitrary silent-correctness issues in any other part of your code

    """

    # 这个上下文管理器用于保留版本计数器，使用时需要极大的注意
    # 用于管理 tensor 对象的版本计数器，允许在需要时隐藏对其的变化以避免 autograd 追踪
    """
    Context manager to preserve the version counter of a given PyTorch tensor.

    This context manager is used to preserve the version counter of a tensor,
    allowing mutations to be temporarily hidden from autograd tracking.
    This can be useful in scenarios where mutations to a tensor need to be
    managed manually to avoid interfering with autograd's gradient calculations.

    Ordinarily, autograd will track mutations to tensors by incrementing its `._version` attribute.
    This is generally important for correctness, as mutating a tensor that autograd has saved
    for the backwards pass can result in incorrect gradients. Autograd uses the version counter to detect
    and raise errors in such situations.

    However, there are rare instances where it might be useful to hide mutations from autograd. For example:
    if a tensor is very large, and you'd like to free its memory by storing it elsewhere, and re-populate
    the tensor right before it is needed by autograd.

    Args:
        tensor (torch.Tensor): the tensor in question, that you would like to preserve the version counter of.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    """

    # 初始化方法，保存给定 tensor 的当前版本计数器
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.prev_version = tensor._version

    # 进入上下文管理器时的操作，此处为空实现
    def __enter__(self) -> None:
        pass

    # 退出上下文管理器时的操作，恢复 tensor 的版本计数器为之前保存的值
    def __exit__(self, *args) -> None:
        torch._C._autograd._unsafe_set_version_counter(self.tensor, self.prev_version)
```