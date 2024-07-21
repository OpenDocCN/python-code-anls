# `.\pytorch\torch\nn\modules\module.py`

```
# 引入需要的模块和库
import functools  # 提供高阶函数操作
import itertools  # 提供迭代工具
import warnings  # 提供警告相关功能
import weakref  # 提供弱引用支持
from collections import namedtuple, OrderedDict  # 提供命名元组和有序字典的支持
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    overload,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import Self  # 引入自定义类型扩展

import torch  # 引入PyTorch库
from torch import device, dtype, Tensor  # 引入Tensor相关类型
from torch._prims_common import DeviceLikeType  # 引入设备类型
from torch.nn.parameter import Parameter  # 引入模型参数类型
from torch.utils._python_dispatch import is_traceable_wrapper_subclass  # 引入Python分发工具
from torch.utils.hooks import BackwardHook, RemovableHandle  # 引入反向传播钩子和可移除句柄

__all__ = [
    "register_module_forward_pre_hook",  # 导出的模块前向预钩子注册函数
    "register_module_forward_hook",  # 导出的模块前向钩子注册函数
    "register_module_full_backward_pre_hook",  # 导出的模块全反向传播前预钩子注册函数
    "register_module_backward_hook",  # 导出的模块反向传播钩子注册函数
    "register_module_full_backward_hook",  # 导出的模块全反向传播钩子注册函数
    "register_module_buffer_registration_hook",  # 导出的模块缓冲区注册钩子注册函数
    "register_module_module_registration_hook",  # 导出的模块模块注册钩子注册函数
    "register_module_parameter_registration_hook",  # 导出的模块参数注册钩子注册函数
    "Module",  # 导出的模块类
]

_grad_t = Union[Tuple[Tensor, ...], Tensor]  # 梯度类型可以是Tensor元组或单个Tensor
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar("T", bound="Module")  # 泛型T，用于表示Module的子类


class _IncompatibleKeys(  # 定义_IncompatibleKeys类，继承自命名元组
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


def _addindent(s_: str, numSpaces: int) -> str:
    """
    将字符串s_按照指定的空格数进行缩进处理。

    Args:
    s_: 输入的字符串
    numSpaces: 缩进的空格数

    Returns:
    缩进后的字符串
    """
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


r"""This tracks hooks common to all modules that are executed immediately before
.registering the buffer/module/parameter"""
_global_buffer_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_module_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_parameter_registration_hooks: Dict[int, Callable] = OrderedDict()


class _WrappedHook:
    def __init__(self, hook: Callable, module: Optional["Module"] = None):
        """
        初始化_WrappedHook对象。

        Args:
        hook: 要包装的回调函数
        module: 可选的Module对象，如果提供则使用弱引用引用它
        """
        self.hook: Callable = hook
        functools.update_wrapper(self, hook)

        self.with_module: bool = False

        if module is not None:
            self.module: weakref.ReferenceType[Module] = weakref.ref(module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        调用_WrappedHook对象时的行为，根据是否包含Module对象来选择调用方式。

        Args:
        *args: 位置参数
        **kwargs: 关键字参数

        Returns:
        调用结果
        """
        if self.with_module:
            module = self.module()
            if module is None:
                raise RuntimeError("You are trying to call the hook of a dead Module!")
            return self.hook(module, *args, **kwargs)
        return self.hook(*args, **kwargs)
    # 定义特殊方法 __getstate__，用于序列化对象状态为字典
    def __getstate__(self) -> Dict:
        # 初始化结果字典，包含 hook 和 with_module 两个状态信息
        result = {"hook": self.hook, "with_module": self.with_module}
        # 如果 with_module 为 True，将 module 对象也加入结果字典
        if self.with_module:
            result["module"] = self.module()

        # 返回序列化后的结果字典
        return result

    # 定义特殊方法 __setstate__，用于从字典中恢复对象状态
    def __setstate__(self, state: Dict):
        # 从状态字典中恢复 hook 和 with_module 的值
        self.hook = state["hook"]
        self.with_module = state["with_module"]

        # 如果 with_module 为 True，则尝试恢复 module 对象的弱引用
        if self.with_module:
            # 如果 module 为 None，则抛出运行时异常
            if state["module"] is None:
                raise RuntimeError(
                    "You are trying to revive the hook of a dead Module!"
                )
            # 使用弱引用将 module 重新绑定到对象上
            self.module = weakref.ref(state["module"])
# 全局变量：用于跟踪在调用前后执行的钩子，这些钩子对所有模块都是共享的，用于调试和性能分析的全局状态

# 全局钩子列表，用于在调用模块的 forward 和 backward 前执行的钩子
_global_backward_pre_hooks: Dict[int, Callable] = OrderedDict()

# 全局钩子列表，用于在调用模块的 backward 后执行的钩子
_global_backward_hooks: Dict[int, Callable] = OrderedDict()

# 全局标志，指示是否已经设置了完整的 backward hook
_global_is_full_backward_hook: Optional[bool] = None

# 全局钩子列表，用于在调用模块的 forward 前执行的钩子
_global_forward_pre_hooks: Dict[int, Callable] = OrderedDict()

# 全局钩子列表，用于在调用模块的 forward 后执行的钩子
_global_forward_hooks: Dict[int, Callable] = OrderedDict()

# 全局字典，用于记录始终会被调用的 forward 钩子的状态
_global_forward_hooks_always_called: Dict[int, bool] = OrderedDict()

# 全局变量：额外状态键的后缀，用于模块中额外状态的命名
_EXTRA_STATE_KEY_SUFFIX = "_extra_state"


def register_module_buffer_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a buffer registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_buffer` is invoked.
    It should have the following signature::

        hook(module, name, buffer) -> None or new buffer

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_buffer_registration_hooks)
    _global_buffer_registration_hooks[handle.id] = hook
    return handle


def register_module_module_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a module registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_module` is invoked.
    It should have the following signature::

        hook(module, name, submodule) -> None or new submodule

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_module_registration_hooks)
    _global_module_registration_hooks[handle.id] = hook
    return handle


def register_module_parameter_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    r"""Register a parameter registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_parameter` is invoked.
    It should have the following signature::

        hook(module, name, param) -> None or new parameter

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_parameter_registration_hooks)
    # 将 hook 与 handle.id 关联起来，注册到全局参数注册钩子中
    _global_parameter_registration_hooks[handle.id] = hook
    # 返回处理句柄
    return handle
# 注册一个全局的前向预处理钩子，适用于所有的模块。
def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a forward pre-hook common to all modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, input) -> None or modified input

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the input. User can either return a tuple or a
    single modified value in the hook. We will wrap the value into a tuple
    if a single value is returned(unless that value is already a tuple).

    This hook has precedence over the specific module hooks registered with
    ``register_forward_pre_hook``.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    # 创建一个可移除的句柄，用于后续移除添加的钩子
    handle = RemovableHandle(_global_forward_pre_hooks)
    # 将钩子函数添加到全局前向预处理钩子字典中
    _global_forward_pre_hooks[handle.id] = hook
    # 返回句柄，允许用户移除该钩子
    return handle


# 注册一个全局的前向后处理钩子，适用于所有的模块。
def register_module_forward_hook(
    hook: Callable[..., None],
    *,
    always_call: bool = False,
) -> RemovableHandle:
    r"""Register a global forward hook for all the modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the output. It can modify the input inplace but
    it will not have effect on forward since this is called after
    :func:`forward` is called.

    Parameters:
        hook (Callable): The user defined hook to be registered.
        always_call (bool): If ``True`` the ``hook`` will be run regardless of
            whether an exception is raised while calling the Module.
            Default: ``False``
    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    This hook will be executed before specific module hooks registered with
    ``register_forward_hook``.
    """
    # 创建一个可移除的句柄，用于后续移除添加的钩子
    handle = RemovableHandle(
        _global_forward_hooks, extra_dict=_global_forward_hooks_always_called
    )
    # 将钩子函数添加到全局前向钩子字典中
    _global_forward_hooks[handle.id] = hook
    # 如果设置了 always_call 标志，则将其标记为始终调用
    if always_call:
        _global_forward_hooks_always_called[handle.id] = True
    # 返回句柄，允许用户移除该钩子
    return handle


# 注册一个全局的反向钩子，适用于所有的模块。未完待续...
    hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],


    # 定义一个变量 hook，类型为 Callable，接受三个参数：
    # 第一个参数是字符串 "Module"
    # 第二个参数是 _grad_t 类型的变量
    # 第三个参数也是 _grad_t 类型的变量
    # 返回值可以是 None 或者 _grad_t 类型的变量
def register_module_full_backward_hook(
    hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],
) -> RemovableHandle:
    r"""Register a backward hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    Hooks registered using this function behave in the same way as those
    registered by :meth:`torch.nn.Module.register_full_backward_hook`.
    Refer to its documentation for more details.

    Hooks registered using this function will be called before hooks registered
    using :meth:`torch.nn.Module.register_full_backward_hook`.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    # 如果全局标志 _global_is_full_backward_hook 为 False，则抛出运行时错误
    if _global_is_full_backward_hook is False:
        raise RuntimeError(
            "Cannot use both regular backward hooks and full backward hooks as a "
            "global Module hook. Please use only one of them."
        )

    # 将全局标志 _global_is_full_backward_hook 设为 False
    _global_is_full_backward_hook = False

    # 创建一个可移除的句柄，用于后续移除已添加的钩子
    handle = RemovableHandle(_global_backward_hooks)
    # 将新的钩子函数 hook 添加到全局 _global_backward_hooks 字典中
    _global_backward_hooks[handle.id] = hook
    return handle
    # 设置全局变量 `_global_is_full_backward_hook` 为 True，用于标识全局是否使用完整的反向传播钩子
    
    # 创建一个可移除的句柄 `handle`，并将其添加到全局反向传播钩子字典 `_global_backward_hooks` 中
    handle = RemovableHandle(_global_backward_hooks)
    
    # 将指定的 `hook` 函数与 `handle` 关联，存储到 `_global_backward_hooks` 字典中，以便后续调用
    _global_backward_hooks[handle.id] = hook
    
    # 返回创建的 `handle` 对象，使调用者可以后续移除该钩子
    return handle
# 定义一个占位函数，以避免 mypy 在输入上应用逆变规则，将 forward 视为值而不是函数。
# 参考链接：https://github.com/python/mypy/issues/8795
def _forward_unimplemented(self, *input: Any) -> None:
    r"""Define the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        虽然需要在此函数内定义前向传播的步骤，但应该调用类的 :class:`Module` 实例，
        而不是直接调用此函数，因为前者会处理运行时注册的钩子，而后者会静默忽略它们。
    """
    raise NotImplementedError(
        f'Module [{type(self).__name__}] is missing the required "forward" function'
    )


class Module:
    r"""Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.

    .. note::
        根据上面的示例，在子类上分配属性之前必须先调用父类的 ``__init__()`` 方法。

    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    """

    dump_patches: bool = False

    _version: int = 1
    r"""This allows better BC support for :meth:`load_state_dict`. In
    :meth:`state_dict`, the version number will be saved as in the attribute
    `_metadata` of the returned state dict, and thus pickled. `_metadata` is a
    dictionary with keys that follow the naming convention of state dict. See
    ``_load_from_state_dict`` on how to use this information in loading.

    If new parameters/buffers are added/removed from a module, this number shall
    be bumped, and the module's `_load_from_state_dict` method can compare the
    version number and do appropriate changes if the state dict is from before
    the change."""

    training: bool
    _parameters: Dict[str, Optional[Parameter]]
    _buffers: Dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: Set[str]
    _backward_pre_hooks: Dict[int, Callable]
    _backward_hooks: Dict[int, Callable]
    _is_full_backward_hook: Optional[bool]
    _forward_hooks: Dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # JIT 不支持 Set[int]，因此这些字典被用作集合，其中所有表示的钩子都接受 kwargs。
    # 这些字典包含了所有具有 kwargs 参数的 forward hooks。
    _forward_hooks_with_kwargs: Dict[int, bool]
    
    # 这些是始终应该调用的 forward hooks，即使引发异常。
    _forward_hooks_always_called: Dict[int, bool]
    
    # 这是 forward pre hooks 的字典，它们在 forward 方法之前调用。
    _forward_pre_hooks: Dict[int, Callable]
    
    # 表示对应的 _forward_hooks 是否接受 kwargs 的标记。
    # JIT 不支持 Set[int]，因此这些字典被用作集合，其中所有表示的钩子都接受 kwargs。
    _forward_pre_hooks_with_kwargs: Dict[int, bool]
    
    # 用于存储状态字典 hooks 的字典。
    _state_dict_hooks: Dict[int, Callable]
    
    # 加载状态字典前调用的 hooks 的字典。
    _load_state_dict_pre_hooks: Dict[int, Callable]
    
    # 状态字典前调用的 hooks 的字典。
    _state_dict_pre_hooks: Dict[int, Callable]
    
    # 加载状态字典后调用的 hooks 的字典。
    _load_state_dict_post_hooks: Dict[int, Callable]
    
    # 模块的字典表示，其中键是模块的名称，值是 Optional 类型的 Module 对象。
    _modules: Dict[str, Optional["Module"]]
    
    # 是否调用超类的初始化方法。
    call_super_init: bool = False
    
    # 编译后的调用实现的可选回调函数。
    _compiled_call_impl: Optional[Callable] = None
    def __init__(self, *args, **kwargs) -> None:
        """Initialize internal Module state, shared by both nn.Module and ScriptModule."""
        # 记录 API 使用情况，这里是初始化 Module 状态时的一次调用记录
        torch._C._log_api_usage_once("python.nn_module")

        # 向后兼容性：当 call_super_init=False 时，不应该有任何参数
        if self.call_super_init is False and bool(kwargs):
            # 如果 call_super_init 为 False 且传入了关键字参数，则抛出 TypeError 异常
            raise TypeError(
                f"{type(self).__name__}.__init__() got an unexpected keyword argument '{next(iter(kwargs))}'"
                ""
            )

        # 向后兼容性：当 call_super_init=False 时，不应该有任何位置参数
        if self.call_super_init is False and bool(args):
            # 如果 call_super_init 为 False 且传入了位置参数，则抛出 TypeError 异常
            raise TypeError(
                f"{type(self).__name__}.__init__() takes 1 positional argument but {len(args) + 1} were"
                " given"
            )

        """
        Calls super().__setattr__('a', a) instead of the typical self.a = a
        to avoid Module.__setattr__ overhead. Module's __setattr__ has special
        handling for parameters, submodules, and buffers but simply calls into
        super().__setattr__ for all other attributes.
        """
        # 使用 super().__setattr__ 设置一些初始状态，避免 Module.__setattr__ 的开销
        super().__setattr__("training", True)
        super().__setattr__("_parameters", dict())
        super().__setattr__("_buffers", dict())
        super().__setattr__("_non_persistent_buffers_set", set())
        super().__setattr__("_backward_pre_hooks", OrderedDict())
        super().__setattr__("_backward_hooks", OrderedDict())
        super().__setattr__("_is_full_backward_hook", None)
        super().__setattr__("_forward_hooks", OrderedDict())
        super().__setattr__("_forward_hooks_with_kwargs", OrderedDict())
        super().__setattr__("_forward_hooks_always_called", OrderedDict())
        super().__setattr__("_forward_pre_hooks", OrderedDict())
        super().__setattr__("_forward_pre_hooks_with_kwargs", OrderedDict())
        super().__setattr__("_state_dict_hooks", OrderedDict())
        super().__setattr__("_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_post_hooks", OrderedDict())
        super().__setattr__("_modules", dict())

        # 如果 call_super_init 为 True，则调用父类的 __init__ 方法，并传递参数
        if self.call_super_init:
            super().__init__(*args, **kwargs)

    # 将 forward 方法设置为未实现的 _forward_unimplemented 函数
    forward: Callable[..., Any] = _forward_unimplemented

    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        # 如果持久性为 False，并且该模块是 torch.jit.ScriptModule 类型，则抛出异常
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError("ScriptModule does not support non-persistent buffers")

        # 如果 _buffers 不在模块的字典属性中，则抛出 AttributeError
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        # 如果 name 不是字符串类型，则抛出 TypeError
        elif not isinstance(name, str):
            raise TypeError(
                f"buffer name should be a string. Got {torch.typename(name)}"
            )
        # 如果 name 中包含 "."，则抛出 KeyError
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        # 如果 name 是空字符串，则抛出 KeyError
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        # 如果 self 已经存在 name 属性，并且 name 不在 _buffers 中，则抛出 KeyError
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        # 如果 tensor 不为 None 且不是 torch.Tensor 类型，则抛出 TypeError
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "
                "(torch Tensor or None required)"
            )
        else:
            # 遍历全局的 _global_buffer_registration_hooks 的值，并调用每个 hook
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, tensor)
                # 如果 hook 返回非空值，则更新 tensor
                if output is not None:
                    tensor = output
            # 将 tensor 注册为名为 name 的 buffer
            self._buffers[name] = tensor
            # 如果 persistent 为 True，则从 _non_persistent_buffers_set 中移除 name
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            # 如果 persistent 为 False，则将 name 添加到 _non_persistent_buffers_set 中
            else:
                self._non_persistent_buffers_set.add(name)
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Add a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        # 检查是否已经初始化了_parameters属性，如果没有，则抛出错误
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        # 检查参数name是否为字符串类型，如果不是则抛出类型错误
        elif not isinstance(name, str):
            raise TypeError(
                f"parameter name should be a string. Got {torch.typename(name)}"
            )
        # 检查参数name是否包含"."，如果包含则抛出键错误
        elif "." in name:
            raise KeyError('parameter name can\'t contain "."')
        # 检查参数name是否为空字符串，如果是则抛出键错误
        elif name == "":
            raise KeyError('parameter name can\'t be empty string ""')
        # 检查self是否已经存在同名的属性，并且该属性不在_parameters中，如果是则抛出键错误
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        # 如果param为None，则将_parameters中对应name的值设为None
        if param is None:
            self._parameters[name] = None
        # 检查param是否为Parameter类型，如果不是则抛出类型错误
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "
                "(torch.nn.Parameter or None required)"
            )
        # 检查param是否有梯度函数grad_fn，如果有则抛出值错误
        elif param.grad_fn:
            raise ValueError(
                f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "
                f"parameters must be created explicitly. To express '{name}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method."
            )
        else:
            # 遍历全局参数注册钩子的值，并调用钩子函数
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                # 如果钩子函数返回值不为None，则将param更新为返回值
                if output is not None:
                    param = output
            # 将参数param添加到_parameters中
            self._parameters[name] = param
    # 将一个子模块添加到当前模块中。

    # 参数:
    #     name (str): 子模块的名称。可以使用这个名称从当前模块访问子模块。
    #     module (Module): 要添加到当前模块的子模块。

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        # 如果 module 不是 Module 的子类且不为 None，则抛出类型错误异常
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{torch.typename(module)} is not a Module subclass")
        # 如果 name 不是字符串类型，则抛出类型错误异常
        elif not isinstance(name, str):
            raise TypeError(
                f"module name should be a string. Got {torch.typename(name)}"
            )
        # 如果当前对象已经有 name 这个属性，并且 name 不在 _modules 中，则抛出键错误异常
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        # 如果 name 中包含 "."，则抛出键错误异常
        elif "." in name:
            raise KeyError(f'module name can\'t contain ".", got: {name}')
        # 如果 name 是空字符串，则抛出键错误异常
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')

        # 遍历全局模块注册钩子，对当前模块、name 和 module 进行处理
        for hook in _global_module_registration_hooks.values():
            output = hook(self, name, module)
            # 如果钩子返回非空值，则将 module 更新为钩子的输出值
            if output is not None:
                module = output
        
        # 将 module 添加到 _modules 字典中，以 name 作为键
        self._modules[name] = module

    # register_module 是 add_module 的别名
    def register_module(self, name: str, module: Optional["Module"]) -> None:
        # 直接调用 add_module 方法
        self.add_module(name, module)
    def get_submodule(self, target: str) -> "Module":
        """Return the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        # 如果目标字符串为空，则直接返回当前模块 self
        if target == "":
            return self

        # 将目标字符串按点号分割成列表，每个元素是路径中的一个部分
        atoms: List[str] = target.split(".")
        # 从当前模块 self 开始逐级查找子模块
        mod: torch.nn.Module = self

        # 遍历路径中的每一个部分
        for item in atoms:
            # 检查当前模块是否具有名为 item 的属性
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no " "attribute `" + item + "`"
                )

            # 获取当前模块的子模块 item
            mod = getattr(mod, item)

            # 检查获取的子模块 mod 是否确实是 nn.Module 类型
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not " "an nn.Module")

        # 返回找到的子模块 mod
        return mod
    def set_submodule(self, target: str, module: "Module") -> None:
        """
        设置指定的子模块，如果存在则设置，否则抛出错误。

        例如，假设有一个名为 ``A`` 的 ``nn.Module``，结构如下：

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        （上面的图示展示了一个名为 ``A`` 的 ``nn.Module``。``A`` 包含一个嵌套的子模块 ``net_b``，
        ``net_b`` 本身又包含两个子模块 ``net_c`` 和 ``linear``。``net_c`` 还包含一个名为 ``conv`` 的子模块。）

        要用一个新的子模块 ``Linear`` 替换 ``Conv2d``，可以调用
        ``set_submodule("net_b.net_c.conv", nn.Linear(33, 16))``。

        Args:
            target: 要查找的完全限定字符串名称的子模块。
            module: 要设置的子模块对象。

        Raises:
            ValueError: 如果目标字符串为空。
            AttributeError: 如果目标字符串引用无效路径或解析为非 ``nn.Module`` 对象。
        """
        # 如果目标字符串为空，抛出 ValueError 异常
        if target == "":
            raise ValueError("Cannot set the submodule without a target name!")

        # 将目标字符串按 '.' 分割成列表 atoms
        atoms: List[str] = target.split(".")
        # 弹出列表 atoms 中的最后一个元素作为 submodule 的名称
        name = atoms.pop(-1)
        # 将当前对象 self 视为最顶层的模块 mod
        mod: torch.nn.Module = self

        # 遍历 atoms 列表中的每个元素
        for item in atoms:
            # 如果 mod 没有 item 这个属性，抛出 AttributeError 异常
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            # 获取 mod 的 item 属性，并将其作为新的 mod
            mod = getattr(mod, item)

            # 如果 mod 不是 torch.nn.Module 类型，抛出 AttributeError 异常
            if type(mod) is not torch.nn.Module:
                raise AttributeError("`" + item + "` is not an nn.Module")

        # 使用 setattr 函数设置 mod 的 name 属性为 module
        setattr(mod, name, module)
    def get_parameter(self, target: str) -> "Parameter":
        """Return the parameter given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
        # 从 target 中解析出模块路径和参数名
        module_path, _, param_name = target.rpartition(".")

        # 调用 get_submodule 方法获取模块对象
        mod: torch.nn.Module = self.get_submodule(module_path)

        # 检查模块是否具有指定的参数名
        if not hasattr(mod, param_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + param_name + "`"
            )

        # 获取模块中的参数对象
        param: torch.nn.Parameter = getattr(mod, param_name)

        # 检查获取的对象是否是 nn.Parameter 类型
        if not isinstance(param, torch.nn.Parameter):
            raise AttributeError("`" + param_name + "` is not an " "nn.Parameter")

        # 返回获取到的参数对象
        return param

    def get_buffer(self, target: str) -> "Tensor":
        """Return the buffer given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
        # 从 target 中解析出模块路径和缓冲区名称
        module_path, _, buffer_name = target.rpartition(".")

        # 调用 get_submodule 方法获取模块对象
        mod: torch.nn.Module = self.get_submodule(module_path)

        # 检查模块是否具有指定的缓冲区名称
        if not hasattr(mod, buffer_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + buffer_name + "`"
            )

        # 获取模块中的缓冲区对象
        buffer: torch.Tensor = getattr(mod, buffer_name)

        # 检查获取的对象是否在模块的缓冲区列表中
        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        # 返回获取到的缓冲区对象
        return buffer
    def get_extra_state(self) -> Any:
        """
        Return any extra state to include in the module's state_dict.

        Implement this and a corresponding :func:`set_extra_state` for your module
        if you need to store extra state. This function is called when building the
        module's `state_dict()`.

        Note that extra state should be picklable to ensure working serialization
        of the state_dict. We only provide provide backwards compatibility guarantees
        for serializing Tensors; other objects may break backwards compatibility if
        their serialized pickled form changes.

        Returns:
            object: Any extra state to store in the module's state_dict
        """
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug."
        )

    def set_extra_state(self, state: Any) -> None:
        """
        Set extra state contained in the loaded `state_dict`.

        This function is called from :func:`load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        :func:`get_extra_state` for your module if you need to store extra state within its
        `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`
        """
        raise RuntimeError(
            "Reached a code path in Module.set_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug."
        )
    def apply(self: T, fn: Callable[["Module"], None]) -> T:
        r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )

        """
        # 递归地对每个子模块（通过 .children() 返回的）以及自身应用 fn 函数
        for module in self.children():
            module.apply(fn)
        # 对当前模块自身应用 fn 函数
        fn(self)
        # 返回自身模块
        return self

    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        # 将模型的所有参数和缓冲区移动到 GPU 上
        # 这也会使关联的参数和缓冲区成为不同的对象，因此如果模块在 GPU 上进行优化，则应在构造优化器之前调用此方法
        # 注意：这个方法会原地修改模块
        return self._apply(lambda t: t.cuda(device))

    def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the IPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on IPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        # 将模型的所有参数和缓冲区移动到 IPU 上
        # 这也会使关联的参数和缓冲区成为不同的对象，因此如果模块在 IPU 上进行优化，则应在构造优化器之前调用此方法
        # 注意：这个方法会原地修改模块
        return self._apply(lambda t: t.ipu(device))
    def xpu(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the XPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on XPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        # 将模型的所有参数和缓冲区移动到指定的 XPU 设备
        return self._apply(lambda t: t.xpu(device))

    def cpu(self: T) -> T:
        r"""Move all model parameters and buffers to the CPU.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        # 将模型的所有参数和缓冲区移动到 CPU
        return self._apply(lambda t: t.cpu())

    def type(self: T, dst_type: Union[dtype, str]) -> T:
        r"""Casts all parameters and buffers to :attr:`dst_type`.

        .. note::
            This method modifies the module in-place.

        Args:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        # 将模型的所有参数和缓冲区转换为指定的数据类型 dst_type
        return self._apply(lambda t: t.type(dst_type))

    def float(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        # 将模型的所有浮点数参数和缓冲区转换为 float 数据类型
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    def double(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        # 将模型的所有浮点数参数和缓冲区转换为 double 数据类型
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    def half(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        # 将模型的所有浮点数参数和缓冲区转换为 half 数据类型
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    def bfloat16(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``bfloat16`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        # 将模型的所有浮点数参数和缓冲区转换为 bfloat16 数据类型
        return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)

    def to_empty(
        self: T, *, device: Optional[DeviceLikeType], recurse: bool = True
    ) -> T:
        r"""Move all model parameters and buffers to the specified device.

        Args:
            device (DeviceLikeType, optional): the device to move the parameters to
            recurse (bool, optional): if True, also moves the parameters of nested modules

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        # 将模型的所有参数和缓冲区移动到指定的设备上，并可以选择递归地移动嵌套模块的参数
        raise NotImplementedError
    ) -> T:
        r"""Move the parameters and buffers to the specified device without copying storage.

        Args:
            device (:class:`torch.device`): The desired device of the parameters
                and buffers in this module.
            recurse (bool): Whether parameters and buffers of submodules should
                be recursively moved to the specified device.

        Returns:
            Module: self
        """
        # 将模型的参数和缓冲区移动到指定的设备，但不复制存储
        return self._apply(
            lambda t: torch.empty_like(t, device=device), recurse=recurse
        )

    @overload
    def to(
        self,
        device: Optional[DeviceLikeType] = ...,
        dtype: Optional[dtype] = ...,
        non_blocking: bool = ...,
    ) -> Self:
        ...

    @overload
    def to(self, dtype: dtype, non_blocking: bool = ...) -> Self:
        ...

    @overload
    def to(self, tensor: Tensor, non_blocking: bool = ...) -> Self:
        ...

    def register_full_backward_pre_hook(
        self,
        hook: Callable[["Module", _grad_t], Union[None, _grad_t]],
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a backward pre-hook on the module.
        
        注册一个在模块上的反向预钩子。

        The hook will be called every time the gradients for the module are computed.
        
        每当计算模块的梯度时，钩子将被调用。

        The hook should have the following signature::
        
        钩子应该具有以下签名::

            hook(module, grad_output) -> tuple[Tensor] or None

        The :attr:`grad_output` is a tuple. The hook should
        not modify its arguments, but it can optionally return a new gradient with
        respect to the output that will be used in place of :attr:`grad_output` in
        subsequent computations.
        
        :attr:`grad_output` 是一个元组。钩子不应该修改其参数，但它可以选择返回一个相对于输出的新梯度，
        该梯度将在后续计算中代替 :attr:`grad_output` 使用。

        Entries in :attr:`grad_output` will be ``None`` for
        all non-Tensor arguments.
        
        :attr:`grad_output` 中的条目对于所有非 Tensor 参数将为 ``None``。

        For technical reasons, when this hook is applied to a Module, its forward function will
        receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
        of each Tensor returned by the Module's forward function.
        
        出于技术原因，当此钩子应用于模块时，其前向函数将接收到传递给模块的每个 Tensor 的视图。
        同样，调用者将接收到模块前向函数返回的每个 Tensor 的视图。

        .. warning ::
            Modifying inputs inplace is not allowed when using backward hooks and
            will raise an error.
            
        警告：
        在使用反向钩子时不允许原地修改输入，否则会引发错误。

        Args:
            hook (Callable): The user-defined hook to be registered.
            
            hook (Callable): 要注册的用户定义的钩子函数。

            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``backward_pre`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``backward_pre`` hooks
                on this :class:`torch.nn.modules.Module`. Note that global
                ``backward_pre`` hooks registered with
                :func:`register_module_full_backward_pre_hook` will fire before
                all hooks registered by this method.
                
            prepend (bool): 如果为 True，则提供的 ``hook`` 将在此模块上所有现有的 ``backward_pre`` 钩子之前触发。
                否则，提供的 ``hook`` 将在此模块上所有现有的 ``backward_pre`` 钩子之后触发。
                注意，使用 :func:`register_module_full_backward_pre_hook` 注册的全局 ``backward_pre`` 钩子将在此方法注册的所有钩子之前触发。

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
                
            :class:`torch.utils.hooks.RemovableHandle`:
                一个句柄，可以通过调用 ``handle.remove()`` 来移除添加的钩子。

        """
        handle = RemovableHandle(self._backward_pre_hooks)
        self._backward_pre_hooks[handle.id] = hook
        if prepend:
            self._backward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle
    ) -> RemovableHandle:
        r"""Register a backward hook on the module.
        
        在模块上注册一个反向传播钩子。

        This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
        the behavior of this function will change in future versions.
        
        此函数已弃用，推荐使用 :meth:`~torch.nn.Module.register_full_backward_hook`，并且此函数的行为将在将来版本中更改。

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

                返回一个句柄，可以通过调用 ``handle.remove()`` 来移除添加的钩子

        """
        if self._is_full_backward_hook is True:
            raise RuntimeError(
                "Cannot use both regular backward hooks and full backward hooks on a "
                "single Module. Please use only one of them."
            )

        self._is_full_backward_hook = False

        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle
    ) -> RemovableHandle:
        r"""Register a backward hook on the module.
        
        在模块上注册一个反向传播钩子。

        The hook will be called every time the gradients with respect to a module
        are computed, i.e. the hook will execute if and only if the gradients with
        respect to module outputs are computed. The hook should have the following
        signature::

            hook(module, grad_input, grad_output) -> tuple(Tensor) or None

        反向传播钩子将在每次计算模块的梯度时被调用，即只有在计算相对于模块输出的梯度时才会执行。钩子应具有以下签名：

            hook(module, grad_input, grad_output) -> tuple(Tensor) or None

        The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
        with respect to the inputs and outputs respectively. The hook should
        not modify its arguments, but it can optionally return a new gradient with
        respect to the input that will be used in place of :attr:`grad_input` in
        subsequent computations. :attr:`grad_input` will only correspond to the inputs given
        as positional arguments and all kwarg arguments are ignored. Entries
        in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
        arguments.

        :attr:`grad_input` 和 :attr:`grad_output` 是包含输入和输出梯度的元组。钩子不应修改其参数，
        但可以选择返回相对于输入的新梯度，该梯度将用于后续计算中替代 :attr:`grad_input`。 :attr:`grad_input`
        仅对应于作为位置参数给出的输入，所有关键字参数都将被忽略。 :attr:`grad_input` 和 :attr:`grad_output`
        中的条目对于所有非张量参数都将为 ``None``。

        For technical reasons, when this hook is applied to a Module, its forward function will
        receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
        of each Tensor returned by the Module's forward function.

        由于技术原因，当此钩子应用于模块时，其前向函数将接收传递给模块的每个张量的视图。同样，调用者将接收模块前向函数返回的每个张量的视图。

        .. warning ::
            Modifying inputs or outputs inplace is not allowed when using backward hooks and
            will raise an error.
        
        警告：
            在使用反向传播钩子时不允许原地修改输入或输出，并将引发错误。

        Args:
            hook (Callable): The user-defined hook to be registered.
                要注册的用户定义钩子。
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``backward`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``backward`` hooks on
                this :class:`torch.nn.modules.Module`. Note that global
                ``backward`` hooks registered with
                :func:`register_module_full_backward_hook` will fire before
                all hooks registered by this method.

                如果为 True，则提供的 ``hook`` 将在此模块的所有现有“backward”钩子之前触发。
                否则，提供的 ``hook`` 将在此模块的所有现有“backward”钩子之后触发。
                请注意，使用 :func:`register_module_full_backward_hook` 注册的全局“backward”钩子将在此方法注册的所有钩子之前触发。

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

                一个句柄，可以通过调用 ``handle.remove()`` 来删除添加的钩子。

        """
        if self._is_full_backward_hook is False:
            raise RuntimeError(
                "Cannot use both regular backward hooks and full backward hooks on a "
                "single Module. Please use only one of them."
            )

        如果 self._is_full_backward_hook 为 False，则抛出 RuntimeError 异常，指示不能在单个模块上同时使用常规反向传播钩子和完整反向传播钩子。

        self._is_full_backward_hook = True

        设置 self._is_full_backward_hook 为 True，表示现在使用完整的反向传播钩子。

        handle = RemovableHandle(self._backward_hooks)
        使用 self._backward_hooks 初始化一个 RemovableHandle 对象，并将其赋值给 handle。

        self._backward_hooks[handle.id] = hook
        将提供的 hook 添加到 self._backward_hooks 字典中，使用 handle.id 作为键。

        if prepend:
            如果 prepend 为 True，
            self._backward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
            将 handle.id 对应的 hook 移动到 self._backward_hooks 的最前面。

        返回 handle 作为结果，该 handle 可以通过调用 ``handle.remove()`` 来移除添加的钩子。
        return handle
    def _get_backward_hooks(self):
        r"""Return the backward hooks for use in the call function.

        It returns two lists, one with the full backward hooks and one with the non-full
        backward hooks.
        """
        # 初始化一个空列表用来存放完整的反向传播钩子函数
        full_backward_hooks: List[Callable] = []
        # 如果全局标志指示为使用完整的全局反向传播钩子，则将全局反向传播钩子函数添加到列表中
        if _global_is_full_backward_hook is True:
            full_backward_hooks += _global_backward_hooks.values()
        # 如果当前对象实例的标志指示为使用完整的对象实例反向传播钩子，则将其反向传播钩子函数添加到列表中
        if self._is_full_backward_hook is True:
            full_backward_hooks += self._backward_hooks.values()

        # 初始化一个空列表用来存放非完整的反向传播钩子函数
        non_full_backward_hooks: List[Callable] = []
        # 如果全局标志指示为不使用完整的全局反向传播钩子，则将全局反向传播钩子函数添加到列表中
        if _global_is_full_backward_hook is False:
            non_full_backward_hooks += _global_backward_hooks.values()
        # 如果当前对象实例的标志指示为不使用完整的对象实例反向传播钩子，则将其反向传播钩子函数添加到列表中
        if self._is_full_backward_hook is False:
            non_full_backward_hooks += self._backward_hooks.values()

        # 返回完整和非完整的反向传播钩子函数列表
        return full_backward_hooks, non_full_backward_hooks

    def _get_backward_pre_hooks(self):
        # 初始化一个空列表用来存放反向传播前钩子函数
        backward_pre_hooks: List[Callable] = []
        # 将全局反向传播前钩子函数添加到列表中
        backward_pre_hooks += _global_backward_pre_hooks.values()
        # 将当前对象实例的反向传播前钩子函数添加到列表中
        backward_pre_hooks += self._backward_pre_hooks.values()

        # 返回反向传播前钩子函数列表
        return backward_pre_hooks

    def register_forward_pre_hook(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...]], Optional[Any]],
            Callable[
                [T, Tuple[Any, ...], Dict[str, Any]],
                Optional[Tuple[Any, Dict[str, Any]]],
            ],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        r"""Register a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.

        If ``with_kwargs`` is false or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        input. User can either return a tuple or a single modified value in the
        hook. We will wrap the value into a tuple if a single value is returned
        (unless that value is already a tuple). The hook should have the
        following signature::

            hook(module, args) -> None or modified input

        If ``with_kwargs`` is true, the forward pre-hook will be passed the
        kwargs given to the forward function. And if the hook modifies the
        input, both the args and kwargs should be returned. The hook should have
        the following signature::

            hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``forward_pre`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward_pre`` hooks
                on this :class:`torch.nn.modules.Module`. Note that global
                ``forward_pre`` hooks registered with
                :func:`register_module_forward_pre_hook` will fire before all
                hooks registered by this method.
                Default: ``False``
            with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
                given to the forward function.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        # 创建一个 RemovableHandle 对象，用于管理和移除添加的前向预钩子
        handle = RemovableHandle(
            self._forward_pre_hooks, extra_dict=self._forward_pre_hooks_with_kwargs
        )
        # 将用户定义的 hook 添加到 _forward_pre_hooks 字典中
        self._forward_pre_hooks[handle.id] = hook
        # 如果 with_kwargs 为 True，将 handle.id 添加到 _forward_pre_hooks_with_kwargs 字典中
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs[handle.id] = True

        # 如果 prepend 为 True，将 handle.id 移动到 _forward_pre_hooks 字典的开头位置
        if prepend:
            self._forward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        # 返回创建的 handle 对象，用于后续移除该钩子
        return handle
    ) -> RemovableHandle:
        r"""Register a forward hook on the module.
        
        模块上注册一个前向钩子。

        The hook will be called every time after :func:`forward` has computed an output.
        
        钩子将在每次 :func:`forward` 计算输出后被调用。

        If ``with_kwargs`` is ``False`` or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        output. It can modify the input inplace but it will not have effect on
        forward since this is called after :func:`forward` is called. The hook
        should have the following signature::

            hook(module, args, output) -> None or modified output
        
        如果 ``with_kwargs`` 是 ``False`` 或未指定，输入仅包含传递给模块的位置参数。
        关键字参数不会传递给钩子，只会传递给 ``forward``。钩子可以修改输出。
        它可以原地修改输入，但对 forward 没有影响，因为它是在调用 :func:`forward` 后调用的。
        钩子应具有以下签名::

            hook(module, args, output) -> None or modified output

        If ``with_kwargs`` is ``True``, the forward hook will be passed the
        ``kwargs`` given to the forward function and be expected to return the
        output possibly modified. The hook should have the following signature::

            hook(module, args, kwargs, output) -> None or modified output
        
        如果 ``with_kwargs`` 是 ``True``，则前向钩子将传递给前向函数的 ``kwargs``，
        并且可能会返回修改后的输出。钩子应具有以下签名::

            hook(module, args, kwargs, output) -> None or modified output

        Args:
            hook (Callable): The user defined hook to be registered.
                要注册的用户定义钩子函数。
            prepend (bool): If ``True``, the provided ``hook`` will be fired
                before all existing ``forward`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward`` hooks on
                this :class:`torch.nn.modules.Module`. Note that global
                ``forward`` hooks registered with
                :func:`register_module_forward_hook` will fire before all hooks
                registered by this method.
                Default: ``False``
                如果为 ``True``，提供的 ``hook`` 将在此模块的所有现有前向钩子之前触发。
                否则，提供的 ``hook`` 将在此模块的所有现有前向钩子之后触发。
                注意，使用 :func:`register_module_forward_hook` 注册的全局前向钩子
                将在此方法注册的所有钩子之前触发。
                默认为 ``False``
            with_kwargs (bool): If ``True``, the ``hook`` will be passed the
                kwargs given to the forward function.
                Default: ``False``
                如果为 ``True``，则钩子将传递给前向函数的 kwargs。
                默认为 ``False``
            always_call (bool): If ``True`` the ``hook`` will be run regardless of
                whether an exception is raised while calling the Module.
                Default: ``False``
                如果为 ``True``，则无论在调用模块时是否引发异常，都将运行钩子。
                默认为 ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
                可用于通过调用 ``handle.remove()`` 删除添加的钩子的句柄。
        """
        handle = RemovableHandle(
            self._forward_hooks,
            extra_dict=[
                self._forward_hooks_with_kwargs,
                self._forward_hooks_always_called,
            ],
        )
        self._forward_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs[handle.id] = True
        if always_call:
            self._forward_hooks_always_called[handle.id] = True
        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle
    # 定义一个名为 _slow_forward 的方法，用于执行前向传播，支持追踪状态和脚本方法
    def _slow_forward(self, *input, **kwargs):
        # 获取当前的追踪状态
        tracing_state = torch._C._get_tracing_state()
        # 如果没有追踪状态或者 forward 方法是 ScriptMethod 类型，则直接调用 forward 方法
        if not tracing_state or isinstance(self.forward, torch._C.ScriptMethod):
            return self.forward(*input, **kwargs)
        # 判断是否记录作用域
        recording_scopes = torch.jit._trace._trace_module_map is not None
        # 如果正在记录作用域
        if recording_scopes:
            # 尝试获取当前模块在 trace_module_map 中的名称
            name = torch.jit._trace._trace_module_map[self] if self in torch.jit._trace._trace_module_map else None  # type: ignore[index, operator] # noqa: B950
            # 如果获取到了名称，则推入该作用域
            if name:
                tracing_state.push_scope(name)
            else:
                # 如果未获取到名称，则停止记录作用域
                recording_scopes = False
        try:
            # 执行 forward 方法
            result = self.forward(*input, **kwargs)
        finally:
            # 如果正在记录作用域，则弹出当前作用域
            if recording_scopes:
                tracing_state.pop_scope()
        # 返回执行结果
        return result

    # 定义一个名为 _wrapped_call_impl 的方法，用于根据是否存在编译后的调用实现来调用相应的方法
    def _wrapped_call_impl(self, *args, **kwargs):
        # 如果存在编译后的调用实现，则调用之
        if self._compiled_call_impl is not None:
            return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
        else:
            # 否则调用 _call_impl 方法
            return self._call_impl(*args, **kwargs)

    # 设置 __call__ 方法，使其引用 _wrapped_call_impl 方法，用于对象的调用
    __call__: Callable[..., Any] = _wrapped_call_impl

    # 定义一个名为 __getstate__ 的方法，用于获取对象的状态字典，去除 _compiled_call_impl 属性
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_compiled_call_impl", None)
        return state

    # 定义一个名为 __setstate__ 的方法，用于设置对象的状态，支持加载旧的检查点
    def __setstate__(self, state):
        self.__dict__.update(state)

        # 添加一些可能缺失的属性，以支持加载旧的检查点
        if "_forward_pre_hooks" not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()
        if "_forward_pre_hooks_with_kwargs" not in self.__dict__:
            self._forward_pre_hooks_with_kwargs = OrderedDict()
        if "_forward_hooks_with_kwargs" not in self.__dict__:
            self._forward_hooks_with_kwargs = OrderedDict()
        if "_forward_hooks_always_called" not in self.__dict__:
            self._forward_hooks_always_called = OrderedDict()
        if "_state_dict_hooks" not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if "_state_dict_pre_hooks" not in self.__dict__:
            self._state_dict_pre_hooks = OrderedDict()
        if "_load_state_dict_pre_hooks" not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()
        if "_load_state_dict_post_hooks" not in self.__dict__:
            self._load_state_dict_post_hooks = OrderedDict()
        if "_non_persistent_buffers_set" not in self.__dict__:
            self._non_persistent_buffers_set = set()
        if "_is_full_backward_hook" not in self.__dict__:
            self._is_full_backward_hook = None
        if "_backward_pre_hooks" not in self.__dict__:
            self._backward_pre_hooks = OrderedDict()

    # 关于返回类型的注释:
    # 定义一个特殊方法 __getattr__，用于获取对象的属性值
    # 返回类型为 Any，而不是更严格的 Union[Tensor, Module]
    # 这样设计可以更好地与各种类型检查器协作，以便终端用户使用
    # 使用更严格的返回类型会导致与 register_buffer() 方法不兼容，
    # 强制用户过多使用类型忽略、断言、强制转换等操作
    # 可参考这里关于返回 Union 的讨论：https://github.com/microsoft/pyright/issues/4213
    def __getattr__(self, name: str) -> Any:
        # 如果对象的 __dict__ 属性中存在 "_parameters"，则尝试获取其值
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            # 如果属性名 name 存在于 _parameters 字典中，则返回对应的值
            if name in _parameters:
                return _parameters[name]
        # 如果对象的 __dict__ 属性中存在 "_buffers"，则尝试获取其值
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            # 如果属性名 name 存在于 _buffers 字典中，则返回对应的值
            if name in _buffers:
                return _buffers[name]
        # 如果对象的 __dict__ 属性中存在 "_modules"，则尝试获取其值
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            # 如果属性名 name 存在于 modules 字典中，则返回对应的值
            if name in modules:
                return modules[name]
        # 如果以上条件都不满足，则抛出 AttributeError 异常
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # 定义一个特殊方法 __delattr__，用于删除对象的属性
    def __delattr__(self, name):
        # 如果属性名 name 存在于对象的 _parameters 字典中，则删除该属性
        if name in self._parameters:
            del self._parameters[name]
        # 如果属性名 name 存在于对象的 _buffers 字典中，则删除该属性
        elif name in self._buffers:
            del self._buffers[name]
            # 同时从 _non_persistent_buffers_set 集合中移除该属性名
            self._non_persistent_buffers_set.discard(name)
        # 如果属性名 name 存在于对象的 _modules 字典中，则删除该属性
        elif name in self._modules:
            del self._modules[name]
        else:
            # 否则，调用父类的 __delattr__ 方法删除属性
            super().__delattr__(name)

    # 定义一个方法 _register_state_dict_hook，用于注册状态字典的钩子函数
    def _register_state_dict_hook(self, hook):
        r"""Register a state-dict hook.

        These hooks will be called with arguments: `self`, `state_dict`,
        `prefix`, `local_metadata`, after the `state_dict` of `self` is set.
        Note that only parameters and buffers of `self` or its children are
        guaranteed to exist in `state_dict`. The hooks may modify `state_dict`
        inplace or return a new one.
        """
        # 创建一个可移除的处理句柄，将 hook 函数添加到 _state_dict_hooks 字典中
        handle = RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    # 定义一个方法 register_state_dict_pre_hook，用于注册状态字典前的钩子函数
    def register_state_dict_pre_hook(self, hook):
        r"""Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

        These hooks will be called with arguments: ``self``, ``prefix``,
        and ``keep_vars`` before calling ``state_dict`` on ``self``. The registered
        hooks can be used to perform pre-processing before the ``state_dict``
        call is made.
        """
        # 创建一个可移除的处理句柄，将 hook 函数添加到 _state_dict_pre_hooks 字典中
        handle = RemovableHandle(self._state_dict_pre_hooks)
        self._state_dict_pre_hooks[handle.id] = hook
        return handle
    # 将模块的状态保存到指定的字典 `destination` 中。
    # 这个方法在每个子模块的 `state_dict` 方法中被调用，用于保存模块自身的状态，但不包括其子模块的状态。

    for name, param in self._parameters.items():
        # 遍历模块的参数 `_parameters`，将参数保存到 `destination` 中，根据 `keep_vars` 决定是否保留梯度信息。
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()

    for name, buf in self._buffers.items():
        # 遍历模块的缓冲区 `_buffers`，将缓冲区保存到 `destination` 中，根据 `keep_vars` 决定是否保留梯度信息。
        if buf is not None and name not in self._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()

    # 生成额外状态的键值，用于保存可能存在的额外状态信息
    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX

    # 检查是否存在自定义的 `get_extra_state` 方法，如果有，则调用该方法获取额外的状态信息并保存到 `destination` 中
    if (
        getattr(self.__class__, "get_extra_state", Module.get_extra_state)
        is not Module.get_extra_state
    ):
        destination[extra_state_key] = self.get_extra_state()

# 用户可以选择传入一个可映射对象 `destination` 到 `state_dict` 方法中，此时方法会返回同样的对象。
# 如果未传入任何对象，则创建并返回一个有序字典 `OrderedDict`。
    def _register_load_state_dict_pre_hook(self, hook, with_module=False):
        r"""Register a pre-hook for the :meth:`~torch.nn.Module.load_state_dict` method.

        These hooks will be called with arguments: `state_dict`, `prefix`,
        `local_metadata`, `strict`, `missing_keys`, `unexpected_keys`,
        `error_msgs`, before loading `state_dict` into `self`. These arguments
        are exactly the same as those of `_load_from_state_dict`.

        If ``with_module`` is ``True``, then the first argument to the hook is
        an instance of the module.

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
            with_module (bool, optional): Whether or not to pass the module
                instance to the hook as the first parameter.
        """
        handle = RemovableHandle(self._load_state_dict_pre_hooks)
        self._load_state_dict_pre_hooks[handle.id] = _WrappedHook(
            hook, self if with_module else None
        )
        return handle


    def register_load_state_dict_post_hook(self, hook):
        r"""Register a post hook to be run after module's ``load_state_dict`` is called.

        It should have the following signature::
            hook(module, incompatible_keys) -> None

        The ``module`` argument is the current module that this hook is registered
        on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
        of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
        is a ``list`` of ``str`` containing the missing keys and
        ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

        The given incompatible_keys can be modified inplace if needed.

        Note that the checks performed when calling :func:`load_state_dict` with
        ``strict=True`` are affected by modifications the hook makes to
        ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
        set of keys will result in an error being thrown when ``strict=True``, and
        clearing out both missing and unexpected keys will avoid an error.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._load_state_dict_post_hooks)
        self._load_state_dict_post_hooks[handle.id] = hook
        return handle


    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        the module's own parameters and buffers, but not those of its
        descendants. This is called from :meth:`load_state_dict`. It returns
        the actual names of keys that were missing in :attr:`state_dict`.
        """
        pass


    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``,
        then the keys of :attr:`state_dict` must exactly match the keys
        returned by this module's :meth:`~torch.nn.Module.state_dict` function.
        """
        pass


    def _named_members(
        self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
    ):
        r"""Helper method for yielding various names and members of this module."""
        pass
    ):
        r"""Help yield various names + members of modules."""
        # 用于生成模块名称和成员的辅助函数
        memo = set()
        # 创建一个空集合，用于记录已经访问过的成员
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        # 如果递归为真，则获取所有命名模块的名称和成员，否则只包含当前模块
        for module_prefix, module in modules:
            # 遍历模块列表
            members = get_members_fn(module)
            # 获取模块的成员列表
            for k, v in members:
                # 遍历成员列表
                if v is None or v in memo:
                    continue
                # 如果成员值为None或者已经在memo中，则跳过
                if remove_duplicate:
                    memo.add(v)
                # 如果需要去重，则将成员值添加到memo中
                name = module_prefix + ("." if module_prefix else "") + k
                # 构造成员的完整名称
                yield name, v
                # 返回成员的完整名称和值

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Return an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        # 返回该模块参数的迭代器，通常用于优化器
        for name, param in self.named_parameters(recurse=recurse):
            yield param
            # 返回参数本身

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        r"""Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, param in self.named_parameters():
            >>>     if name in ['bias']:
            >>>         print(param.size())

        """
        # 返回一个迭代器，遍历模块参数并生成参数名称和参数本身的元组
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen
        # 从生成器中生成结果
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""Return an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        # 使用_named_members方法获取模块的缓冲区，返回一个生成器
        gen = self._named_members(
            lambda module: module._buffers.items(),  # 使用lambda函数获取模块的缓冲区项
            prefix="",  # 前缀为空字符串，不添加前缀
            recurse=recurse,  # 递归参数与函数参数相同
            remove_duplicate=True,  # 去除重复的缓冲区项，默认为True
        )
        # 从生成器中yield所有结果
        yield from gen

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        r"""Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool, optional): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

        Yields:
            (str, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, buf in self.named_buffers():
            >>>     if name in ['running_var']:
            >>>         print(buf.size())

        """
        # 使用_named_members方法获取模块的缓冲区，返回一个生成器
        gen = self._named_members(
            lambda module: module._buffers.items(),  # 使用lambda函数获取模块的缓冲区项
            prefix=prefix,  # 使用指定的前缀
            recurse=recurse,  # 递归参数与函数参数相同
            remove_duplicate=remove_duplicate,  # 去除重复的缓冲区项
        )
        # 从生成器中yield所有结果
        yield from gen

    def children(self) -> Iterator["Module"]:
        r"""Return an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        # 遍历模块的所有子模块，yield每一个子模块对象
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        r"""Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        # 使用_modules属性遍历模块的所有子模块，保证每个模块只被yield一次
        memo = set()  # 创建一个集合来存储已经yield过的模块
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)  # 将当前模块添加到memo集合中
                yield name, module  # yield模块的名称和模块对象
    def modules(self) -> Iterator["Module"]:
        r"""Return an iterator over all modules in the network.
        
        Yields:
            Module: a module in the network
        
        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        
        Example::
        
            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)
        
            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)
        
        """
        # 返回一个迭代器，遍历网络中的所有模块
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        r"""Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.
        
        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not
        
        Yields:
            (str, Module): Tuple of name and module
        
        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        
        Example::
        
            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)
        
            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
        
        """
        # 如果 memo 为空，创建一个空集合
        if memo is None:
            memo = set()
        # 如果当前模块不在 memo 中，将其加入 memo
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            # 返回当前模块的名称和模块本身
            yield prefix, self
            # 遍历当前模块的子模块字典
            for name, module in self._modules.items():
                # 如果子模块为空，继续下一个循环
                if module is None:
                    continue
                # 构建子模块的完整名称前缀
                submodule_prefix = prefix + ("." if prefix else "") + name
                # 递归调用 named_modules，生成子模块的迭代器
                yield from module.named_modules(
                    memo, submodule_prefix, remove_duplicate
                )
    def train(self: T, mode: bool = True) -> T:
        r"""Set the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        # 检查 mode 是否为布尔类型，否则抛出异常
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        # 设置当前模块的训练模式
        self.training = mode
        # 逐个设置当前模块下子模块的训练模式
        for module in self.children():
            module.train(mode)
        # 返回当前模块实例自身
        return self

    def eval(self: T) -> T:
        r"""Set the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        # 调用 train 方法将模块设为 evaluation 模式，并返回自身模块实例
        return self.train(False)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        r"""Change if autograd should record operations on parameters in this module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
        # 逐个设置当前模块下参数的 requires_grad 属性
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        # 返回当前模块实例自身
        return self
    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Reset gradients of all model parameters.

        See similar function under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        # 如果模型是通过 nn.DataParallel 创建的副本，则警告用户调用 zero_grad() 无效
        if getattr(self, "_is_replica", False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead."
            )

        # 遍历模型的所有参数
        for p in self.parameters():
            # 如果参数的梯度不为 None，则根据 set_to_none 参数决定将梯度置为 None 或者置零
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    # 如果梯度的 grad_fn 不为 None，则将其 detach
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        # 否则，将 requires_grad 设置为 False，并将梯度置零
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def share_memory(self: T) -> T:
        r"""See :meth:`torch.Tensor.share_memory_`."""
        # 调用 Tensor 的 share_memory_() 方法，实现共享内存
        return self._apply(lambda t: t.share_memory())

    def _get_name(self):
        # 返回当前类的类名
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        # 返回空字符串，子类可以重写这个方法以提供额外的描述信息
        return ""

    def __repr__(self):
        # 将额外的描述信息和子模块的表示形式组合成模块的字符串表示
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
    # 返回当前对象的属性列表，包括类属性、实例属性、参数、模块和缓冲区的键
    def __dir__(self):
        module_attrs = dir(self.__class__)  # 获取当前对象类的所有属性名
        attrs = list(self.__dict__.keys())  # 获取当前对象实例的所有属性名
        parameters = list(self._parameters.keys())  # 获取当前对象的所有参数名
        modules = list(self._modules.keys())  # 获取当前对象的所有模块名
        buffers = list(self._buffers.keys())  # 获取当前对象的所有缓冲区名
        keys = module_attrs + attrs + parameters + modules + buffers

        # 过滤掉不是合法 Python 变量名的属性名
        keys = [key for key in keys if not key[0].isdigit()]

        # 返回按字母顺序排序的属性名列表
        return sorted(keys)

    # 为数据并行复制当前对象
    def _replicate_for_data_parallel(self):
        replica = self.__new__(type(self))  # 创建当前对象类型的新实例
        replica.__dict__ = self.__dict__.copy()  # 复制当前对象的所有属性到新实例

        # 复制模块的缓冲区和模块本身的引用，副本本身没有参数，而是引用原始模块
        replica._parameters = dict()
        replica._buffers = replica._buffers.copy()
        replica._modules = replica._modules.copy()
        replica._is_replica = True  # 将 _is_replica 标记为 True，表示这是一个副本

        # 返回复制后的副本
        return replica

    # 编译该模块的 forward 方法，使用 torch.compile 进行编译
    def compile(self, *args, **kwargs):
        """
        Compile this Module's forward using :func:`torch.compile`.

        This Module's `__call__` method is compiled and all arguments are passed as-is
        to :func:`torch.compile`.

        See :func:`torch.compile` for details on the arguments for this function.
        """
        self._compiled_call_impl = torch.compile(self._call_impl, *args, **kwargs)
```