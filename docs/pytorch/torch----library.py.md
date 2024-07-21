# `.\pytorch\torch\library.py`

```
# mypy: allow-untyped-defs
# 引入需要的模块和库
import contextlib  # 上下文管理模块
import functools  # 函数装饰器和工具函数
import inspect  # 检查和获取调用框架信息
import re  # 正则表达式模块
import sys  # 系统相关的参数和函数
import traceback  # 跟踪异常信息
import weakref  # 弱引用对象
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union  # 引入类型提示需要的类型
from typing_extensions import deprecated  # 引入过时类型的支持

import torch  # 引入PyTorch库
import torch._library as _library  # 引入PyTorch的内部库
from torch._library.custom_ops import (
    _maybe_get_opdef,  # 引入获取操作定义的函数
    custom_op,  # 引入自定义操作函数
    CustomOpDef,  # 引入自定义操作定义类型
    device_types_t,  # 引入设备类型的类型
)
from torch._ops import OpOverload  # 引入操作重载类

__all__ = [
    "Library",  # 将类Library加入__all__列表，表明它是可以导入的
    "impl",  # 将impl函数加入__all__列表，表明它是可以导入的
    "define",  # 将define函数加入__all__列表，表明它是可以导入的
    "fallthrough_kernel",  # 将fallthrough_kernel函数加入__all__列表，表明它是可以导入的
    "impl_abstract",  # 将impl_abstract函数加入__all__列表，表明它是可以导入的
    "register_fake",  # 将register_fake函数加入__all__列表，表明它是可以导入的
    "get_ctx",  # 将get_ctx函数加入__all__列表，表明它是可以导入的
    "custom_op",  # 将custom_op函数加入__all__列表，表明它是可以导入的
]

# _impls集合，用于记录已注册新内核的组合（namespace，operator，DispatchKey）
# 集合中的键的格式为 `namespace + "/" + op_name + "/" + dispatch_key`
# 用于确保两个库不会尝试覆盖完全相同的功能，以避免调用意外的内核函数
_impls: Set[str] = set()

# _defs集合，用于记录已定义的操作符（namespace，operator）
_defs: Set[str] = set()

# prim是TorchScript解释器保留的命名空间
_reserved_namespaces = ["prim"]


def fallthrough_kernel():
    """
    A dummy function to pass to ``Library.impl`` in order to register a fallthrough.
    一个占位函数，传递给Library.impl以注册一个fallthrough。
    """
    raise NotImplementedError("fallthrough_kernel() should never be called.")


class Library:
    """
    用于创建库的类，可以用于从Python中注册新操作符或覆盖现有库中的操作符。
    用户可以选择传递一个调度键名，以便仅注册特定调度键对应的内核。

    Args:
        ns: 库的名称
        kind: "DEF"、"IMPL"（默认为"IMPL"）、"FRAGMENT"中的一种，用于指定库的类型
        dispatch_key: PyTorch调度键（默认为""）
    """
    # 初始化方法，接受命名空间（ns）、类型（kind）、调度键（dispatch_key）作为参数
    def __init__(self, ns, kind, dispatch_key=""):
        # 检查类型是否合法，只接受 "IMPL", "DEF", "FRAGMENT" 三种类型
        if kind not in ("IMPL", "DEF", "FRAGMENT"):
            raise ValueError("Unsupported kind: ", kind)

        # 如果命名空间在保留命名空间列表中，并且类型为 "DEF" 或 "FRAGMENT"，则引发异常
        if ns in _reserved_namespaces and (kind == "DEF" or kind == "FRAGMENT"):
            raise ValueError(
                ns,
                " is a reserved namespace. Please try creating a library with another name.",
            )

        # 获取当前堆栈信息，用于确定初始化的文件名和行号
        frame = traceback.extract_stack(limit=3)[0]
        filename, lineno = frame.filename, frame.lineno
        # 调用 Torch 库的特定函数，创建或获取分发库对象，存储在 self.m 中
        self.m: Optional[Any] = torch._C._dispatch_library(
            kind, ns, dispatch_key, filename, lineno
        )
        # 初始化命名空间（ns）、操作定义集合（_op_defs）、操作实现集合（_op_impls）、注册句柄列表（_registration_handles）
        self.ns = ns
        self._op_defs: Set[str] = set()
        self._op_impls: Set[str] = set()
        self._registration_handles: List[torch._library.utils.RegistrationHandle] = []
        self.kind = kind
        self.dispatch_key = dispatch_key
        # 使用弱引用的 finalize 方法设置析构器，代替常规的 __del__ 方法
        # Python 的 __del__ 方法可能存在奇怪行为（在 __del__ 调用时全局和局部变量可能已经消失）
        # finalize 方法有助于解决这种情况，因为它允许我们捕获引用并保持它们的存活状态
        weakref.finalize(
            self,
            _del_library,  # 调用 _del_library 函数进行析构操作
            _impls,  # 传递 _impls 参数给 _del_library 函数
            self._op_impls,  # 传递 self._op_impls 参数给 _del_library 函数
            _defs,  # 传递 _defs 参数给 _del_library 函数
            self._op_defs,  # 传递 self._op_defs 参数给 _del_library 函数
            self._registration_handles,  # 传递 self._registration_handles 参数给 _del_library 函数
        )

    # 返回描述对象的字符串表示形式，包括其类型、命名空间和调度键
    def __repr__(self):
        return f"Library(kind={self.kind}, ns={self.ns}, dispatch_key={self.dispatch_key})>"
    def define(self, schema, alias_analysis="", *, tags=()):
        r"""Defines a new operator and its semantics in the ns namespace.

        Args:
            schema: function schema to define a new operator.
            alias_analysis (optional): Indicates if the aliasing properties of the operator arguments can be
                                       inferred from the schema (default behavior) or not ("CONSERVATIVE").
            tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this
                                       operator. Tagging an operator changes the operator's behavior
                                       under various PyTorch subsystems; please read the docs for the
                                       torch.Tag carefully before applying it.

        Returns:
            name of the operator as inferred from the schema.

        Example::
            >>> my_lib = Library("mylib", "DEF")
            >>> my_lib.define("sum(Tensor self) -> Tensor")
        """
        # 检查是否提供了无效的 alias_analysis 类型，如果是则抛出异常
        if alias_analysis not in ["", "FROM_SCHEMA", "CONSERVATIVE"]:
            raise RuntimeError(f"Invalid alias_analysis type {alias_analysis}")
        # 确保 self.m 不为 None
        assert self.m is not None
        # 如果 tags 是 torch.Tag 类型，转换为元组
        if isinstance(tags, torch.Tag):
            tags = (tags,)

        # 解析 schema 获取操作符的名称
        name = schema.split("(")[0]
        # 提取包名，如果操作符名称包含点，则表示有包限定
        packet_name = name.split(".")[0] if "." in name else name
        # 检查是否已经存在同名的 OpOverloadPacket，如果存在则设置标志位
        has_preexisting_packet = hasattr(torch.ops, self.ns) and hasattr(
            getattr(torch.ops, self.ns), packet_name
        )

        # 调用 self.m 的 define 方法来定义新的操作符，并传入 alias_analysis 和 tags
        result = self.m.define(schema, alias_analysis, tuple(tags))
        # 再次提取操作符的名称
        name = schema.split("(")[0]
        # 构造全限定名称
        qualname = self.ns + "::" + name

        # 如果已存在同名的 OpOverloadPacket，则刷新该包以包含新的 OpOverload
        if has_preexisting_packet:
            ns = getattr(torch.ops, self.ns)
            packet = getattr(ns, packet_name)
            torch._ops._refresh_packet(packet)

        # 将 qualname 加入到 self._op_defs 和 _defs 中
        self._op_defs.add(qualname)
        _defs.add(qualname)
        # 返回定义操作后的结果
        return result
    def _register_fake(self, op_name, fn, _stacklevel=1):
        r"""Registers the fake impl for an operator defined in the library."""
        # 获取调用位置的源代码
        source = torch._library.utils.get_source(_stacklevel + 1)
        # 获取当前调用栈帧
        frame = sys._getframe(_stacklevel)
        # 获取调用者所在的模块
        caller_module = inspect.getmodule(frame)
        # 如果调用者模块不为None，则获取其模块名
        # 可能为None，例如在没有明确模块的地方调用（如__main__）
        caller_module_name = None if caller_module is None else caller_module.__name__

        # TODO(rzou): We're gonna need to stage this change with torchvision,
        # since torchvision is github first.
        # 如果调用者模块名不为None且以"torchvision."开头，则将模块名置为None
        if caller_module_name is not None and caller_module_name.startswith(
            "torchvision."
        ):
            caller_module_name = None

        # 构造限定名（qualname）
        qualname = f"{self.ns}::{op_name}"
        # 在简单注册表中查找限定名对应的条目
        entry = torch._library.simple_registry.singleton.find(qualname)
        # 如果调用者模块名不为None，则调用辅助函数_check_pystubs_once检查并返回注册的函数
        # 否则直接使用传入的函数fn
        if caller_module_name is not None:
            func_to_register = _check_pystubs_once(fn, qualname, caller_module_name)
        else:
            func_to_register = fn

        # 注册假实现，并返回句柄
        handle = entry.fake_impl.register(func_to_register, source)
        # 将句柄添加到注册句柄列表中
        self._registration_handles.append(handle)
    def _impl_with_aoti_compile(self, op_name, dispatch_key=""):
        r"""Register the operator to use the AOTI-compiled implementation.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.

        Example::
            >>> my_lib = Library("aten", "IMPL")
            >>> my_lib._impl_with_aoti_compile("div.Tensor", "CPU")
        """
        # 如果未提供 dispatch_key，则使用对象自身的 dispatch_key
        if dispatch_key == "":
            dispatch_key = self.dispatch_key
        
        # 断言检查指定的 dispatch_key 是否包含 Dense dispatch key
        assert torch.DispatchKeySet(dispatch_key).has(torch._C.DispatchKey.Dense)

        # 根据传入的 op_name 类型确定注册的操作名称
        if isinstance(op_name, str):
            name = op_name
        elif isinstance(op_name, OpOverload):
            name = op_name._schema.name
            overload_name = op_name._schema.overload_name
            if overload_name != "":
                name = name + "." + overload_name
        else:
            # 如果 op_name 既不是字符串也不是 OpOverload 对象，则抛出运行时错误
            raise RuntimeError(
                "_impl_with_aoti_compile should be passed either a name or an OpOverload object "
                "as the first argument"
            )

        # 构建用于唯一标识实现函数的键
        key = self.ns + "/" + name.split("::")[-1] + "/" + dispatch_key
        
        # 如果已经存在相同的键在 _impls 中，则抛出运行时错误
        if key in _impls:
            # TODO: 在未来添加更多关于现有函数注册位置的信息（此信息在调用 _impl_with_aoti_compile 时已由 C++ 警告返回，但在此之前我们会出错）
            raise RuntimeError(
                "This is not allowed since there's already a kernel registered from python overriding {}"
                "'s behavior for {} dispatch key and {} namespace.".format(
                    name.split("::")[-1], dispatch_key, self.ns
                )
            )

        # 断言确保 self.m 不为 None
        assert self.m is not None
        
        # 调用实现函数的具体实现方法
        impl_fn: Callable = self.m.impl_with_aoti_compile
        impl_fn(self.ns, name.split("::")[-1], dispatch_key)

        # 将当前键添加到 _impls 集合中，表示注册过程完成
        _impls.add(key)
        
        # 将当前键添加到对象的 _op_impls 集合中，以便跟踪注册的操作实现
        self._op_impls.add(key)
    # 定义一个方法用于销毁对象
    def _destroy(self):
        # 如果对象的成员变量 m 不为 None，则调用其 reset 方法进行重置
        if self.m is not None:
            self.m.reset()
        # 将对象的成员变量 m 置为 None
        self.m = None
        # 遍历注册句柄列表，并销毁每个句柄
        for handle in self._registration_handles:
            handle.destroy()
        # 清空注册句柄列表
        self._registration_handles.clear()
        # 引用全局变量 _impls，并减去对象的 _op_impls 成员变量的值
        global _impls
        _impls -= self._op_impls
        # 遍历对象的操作定义列表
        for name in self._op_defs:
            # 根据 "::" 分割操作名和命名空间
            ns, name_with_overload = name.split("::")
            # 根据 "." 分割操作名和重载相关信息
            name = name_with_overload.split(".")[0]
            # 如果 torch.ops 中不存在命名空间 ns，则继续下一次循环
            if not hasattr(torch.ops, ns):
                continue
            # 获取命名空间 ns 对应的对象
            namespace = getattr(torch.ops, ns)
            # 如果命名空间中不存在操作名 name，则继续下一次循环
            if not hasattr(namespace, name):
                continue
            # 删除命名空间中的操作 name
            delattr(namespace, name)
# 删除操作库中的实现和定义
def _del_library(
    captured_impls,
    op_impls,
    captured_defs,
    op_defs,
    registration_handles,
):
    # 从已捕获的实现中减去操作实现
    captured_impls -= op_impls
    # 从已捕获的定义中减去操作定义
    captured_defs -= op_defs
    # 遍历注册句柄列表，销毁每个句柄
    for handle in registration_handles:
        handle.destroy()


# 定义一个上下文管理器，用于创建作用域内的操作库
@contextlib.contextmanager
def _scoped_library(*args, **kwargs):
    try:
        # 创建一个Library对象
        lib = Library(*args, **kwargs)
        # 通过上下文管理器返回库对象
        yield lib
    finally:
        # 在退出上下文管理器时销毁库对象
        lib._destroy()


# 用于保持操作库对象的列表
_keep_alive: List[Library] = []


# 正则表达式模式，用于匹配没有名称的函数原型
NAMELESS_SCHEMA = re.compile(r"\(.*\) -> .*")


# 单分派函数定义，用于定义新的操作符
@functools.singledispatch
def define(qualname, schema, *, lib=None, tags=()):
    r"""Defines a new operator.

    In PyTorch, defining an op (short for "operator") is a two step-process:
    - we need to define the op (by providing an operator name and schema)
    - we need to implement behavior for how the operator interacts with
    various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.

    This entrypoint defines the custom operator (the first step)
    you must then perform the second step by calling various
    ``impl_*`` APIs, like :func:`torch.library.impl` or
    :func:`torch.library.register_fake`.

    Args:
        qualname (str): The qualified name for the operator. Should be
            a string that looks like "namespace::name", e.g. "aten::sin".
            Operators in PyTorch need a namespace to
            avoid name collisions; a given operator may only be created once.
            If you are writing a Python library, we recommend the namespace to
            be the name of your top-level module.
        schema (str): The schema of the operator. E.g. "(Tensor x) -> Tensor"
            for an op that accepts one Tensor and returns one Tensor. It does
            not contain the operator name (that is passed in ``qualname``).
        lib (Optional[Library]): If provided, the lifetime of this operator
            will be tied to the lifetime of the Library object.
        tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this
            operator. Tagging an operator changes the operator's behavior
            under various PyTorch subsystems; please read the docs for the
            torch.Tag carefully before applying it.

    Example::
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::sin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the operator
        >>> @torch.library.impl("mylib::sin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Call the new operator from torch.ops.
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.sin(x)
        >>> assert torch.allclose(y, x.sin())

    """
    # 检查qualname是否为字符串类型，若不是则抛出异常
    if not isinstance(qualname, str):
        raise ValueError(
            f"define(qualname, schema): expected qualname "
            f"to be instance of str, got {type(qualname)}"
        )
    # 使用 torch._library.utils 模块中的 parse_namespace 函数解析限定名称，将结果分配给 namespace 和 name 变量
    namespace, name = torch._library.utils.parse_namespace(qualname)

    # 如果 lib 参数为 None，则使用 namespace 和 "FRAGMENT" 创建一个 Library 对象，并将其添加到 _keep_alive 列表中
    if lib is None:
        lib = Library(namespace, "FRAGMENT")
        _keep_alive.append(lib)

    # 如果给定的 schema 不符合 NAMELESS_SCHEMA 的正则表达式匹配条件，则抛出 ValueError 异常
    if not NAMELESS_SCHEMA.fullmatch(schema):
        raise ValueError(
            # 错误消息说明要求 schema 形式为 "(Tensor x) -> Tensor" 的格式
            f"define(qualname, schema, ...): expected schema "
            f'to look like e.g. "(Tensor x) -> Tensor" but '
            f'got "{schema}"'
        )

    # 使用 lib 对象的 define 方法定义一个名为 name + schema 的函数或方法，并使用空字符串 alias_analysis 和指定的 tags
    lib.define(name + schema, alias_analysis="", tags=tags)
@define.register
def _(lib: Library, schema, alias_analysis=""):
    """The old torch.library.define.
    We're keeping this around for BC reasons
    """

    # 定义一个注册函数，将给定的函数 f 绑定到指定的库 lib 上，并返回该函数 f
    def wrap(f):
        # 使用给定的 schema 和 alias_analysis 定义名称，并将其注册到库 lib 上
        name = lib.define(schema, alias_analysis)
        # 将函数 f 实现绑定到库 lib 的名称上
        lib.impl(name, f)
        return f

    return wrap


@functools.singledispatch
def impl(qualname, types, func=None, *, lib=None):
    """Register an implementation for a device type for this operator.

    You may pass "default" for ``types`` to register this implementation as the
    default implementation for ALL device types.
    Please only use this if the implementation truly supports all device types;
    for example, this is true if it is a composition of built-in PyTorch operators.

    Some valid types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".

    Args:
        qualname (str): Should be a string that looks like "namespace::operator_name".
        types (str | Sequence[str]): The device types to register an impl to.
        lib (Optional[Library]): If provided, the lifetime of this registration
            will be tied to the lifetime of the Library object.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::mysin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the cpu device
        >>> @torch.library.impl("mylib::mysin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.mysin(x)
        >>> assert torch.allclose(y, x.sin())
    """

    # 如果 types 是字符串，则转换为包含该字符串的元组
    if isinstance(types, str):
        types = (types,)
    # 创建一个空集合 keys，用于存储设备类型的键值
    keys = set({})
    # 遍历所有设备类型
    for typ in types:
        # 检查是否是分发键
        is_dispatch_key = torch._C._parse_dispatch_key(typ)
        if is_dispatch_key:
            # 如果是分发键，将其添加到 keys 中
            keys.add(typ)
        else:
            # 否则，将设备类型转换为键，并添加到 keys 中
            keys.add(_device_type_to_key(typ))

    def register(func):
        # 解析 qualname 的命名空间
        namespace, _ = torch._library.utils.parse_namespace(qualname)
        # 如果未提供 lib，则创建一个新的 Library 对象并添加到 _keep_alive 中
        if lib is None:
            use_lib = Library(namespace, "FRAGMENT")
            _keep_alive.append(use_lib)
        else:
            # 否则，使用已提供的 lib
            use_lib = lib
        # 遍历 keys，为每个键注册函数实现
        for key in keys:
            use_lib.impl(qualname, func, key)

    # 如果 func 为 None，则返回 register 函数本身
    if func is None:
        return register
    else:
        # 否则，直接调用 register 函数，并传入 func
        register(func)


def _device_type_to_key(device_type: str) -> str:
    # 这个函数将设备类型转换为对应的键值，并返回该键值字符串
    # 如果设备类型为"default"，则返回字符串"CompositeExplicitAutograd"
    if device_type == "default":
        # 这个注释说明了技术上讲这并不正确，因为虽然所有的 device_type DispatchKeys
        # 都包含在 CompositeExplicitAutograd 中，但并非所有 CompositeExplicitAutograd 中的内容都与
        # device_type 相关联。但作者并不太关心这种区别。
        return "CompositeExplicitAutograd"
    
    # 如果设备类型不是"default"，调用 torch._C._dispatch_key_for_device 函数
    return torch._C._dispatch_key_for_device(device_type)
# 注册函数装饰器，用于将函数注册到给定库的特定实现中
@impl.register
def _(lib: Library, name, dispatch_key=""):
    """Legacy torch.library.impl API. Kept around for BC"""

    def wrap(f):
        # 调用给定库的impl方法，注册函数f作为名称为name的实现
        lib.impl(name, f, dispatch_key)
        return f

    return wrap


# 声明一个弃用警告的函数装饰器，提示用户使用torch.library.register_fake代替
@deprecated(
    "`torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that "
    "instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.",
    category=FutureWarning,
)
def impl_abstract(qualname, func=None, *, lib=None, _stacklevel=1):
    r"""This API was renamed to :func:`torch.library.register_fake` in PyTorch 2.4.
    Please use that instead.
    """
    if func is not None:
        _stacklevel = _stacklevel + 1
    # 调用register_fake函数注册虚构的函数qualname，同时传递lib和_stacklevel参数
    return register_fake(qualname, func, lib=lib, _stacklevel=_stacklevel)


# 声明一个Union类型，用于标识操作符的多种可能类型
_op_identifier = Union[
    str, "torch._ops.OpOverload", "torch._library.custom_ops.CustomOpDef"
]


# 注册一个针对操作符的实现函数到特定设备类型的函数
def register_kernel(
    op: _op_identifier,
    device_types: device_types_t,
    func: Optional[Callable] = None,
    /,
    *,
    lib: Optional[Library] = None,
):
    """Register an implementation for a device type for this operator.

    Some valid device_types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".
    This API may be used as a decorator.

    Args:
        fn (Callable): The function to register as the implementation for
            the given device types.
        device_types (None | str | Sequence[str]): The device_types to register an impl to.
            If None, we will register to all device types -- please only use
            this option if your implementation is truly device-type-agnostic.

    Examples::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>> import numpy as np
        >>>
        >>> # Create a custom op that works on cpu
        >>> @custom_op("mylib::numpy_sin", mutates_args=(), device_types="cpu")
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np)
        >>>
        >>> # Add implementations for the cuda device
        >>> @torch.library.register_kernel("mylib::numpy_sin", "cuda")
        >>> def _(x):
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> x_cpu = torch.randn(3)
        >>> x_cuda = x_cpu.cuda()
        >>> assert torch.allclose(numpy_sin(x_cpu), x_cpu.sin())
        >>> assert torch.allclose(numpy_sin(x_cuda), x_cuda.sin())

    """

    # 检查操作符类型是否合法，如果不是合法类型则引发异常
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError("register_kernel(op): got unexpected type for op: {type(op)}")
    # 如果op是OpOverload类型，则获取其名称
    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    # 获取操作符的定义对象（opdef）
    opdef = _maybe_get_opdef(op)
    # 如果 opdef 不是 None，则调用 opdef 的 register_kernel 方法注册内核
    if opdef is not None:
        return opdef.register_kernel(device_types, func)
    # 如果 opdef 是 None，则断言 op 是字符串类型
    assert isinstance(op, str)
    # 如果 device_types 是 None，则设定其默认值为 "CompositeExplicitAutograd"
    if device_types is None:
        device_types = "CompositeExplicitAutograd"
    # 调用 impl 函数，并传递 op、device_types、func 以及可能的 lib 参数
    return impl(op, device_types, func, lib=lib)
# 定义注册一个虚拟张量实现的函数，用于指定操作符的行为
def register_fake(
    op: _op_identifier,  # 第一个参数是操作符的标识符
    func: Optional[Callable] = None,  # 可选参数，表示虚拟张量实现的具体函数
    /,  # 使用位置参数，后续参数必须以位置方式传递
    *,
    lib: Optional[Library] = None,  # 命名关键字参数，表示依赖的库
    _stacklevel: int = 1,  # 默认参数，指定堆栈层级
):
    r"""Register a FakeTensor implementation ("fake impl") for this operator.

    Also sometimes known as a "meta kernel", "abstract impl".

    An "FakeTensor implementation" specifies the behavior of this operator on
    Tensors that carry no data ("FakeTensor"). Given some input Tensors with
    certain properties (sizes/strides/storage_offset/device), it specifies
    what the properties of the output Tensors are.

    The FakeTensor implementation has the same signature as the operator.
    It is run for both FakeTensors and meta tensors. To write a FakeTensor
    implementation, assume that all Tensor inputs to the operator are
    regular CPU/CUDA/Meta tensors, but they do not have storage, and
    you are trying to return regular CPU/CUDA/Meta tensor(s) as output.
    The FakeTensor implementation must consist of only PyTorch operations
    (and may not directly access the storage or data of any input or
    intermediate Tensors).

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html
    # 检查 op 参数是否是字符串、torch._ops.OpOverload 或 torch._library.custom_ops.CustomOpDef 类型之一
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError("register_fake(op): got unexpected type for op: {type(op)}")

    # 如果 op 是 OpOverload 类型，则转换为其名称字符串
    if isinstance(op, torch._ops.OpOverload):
        op = op._name

    # 获取 op 的操作定义，若存在则返回其注册假实现的方法
    opdef = _maybe_get_opdef(op)

    # 如果 opdef 存在
    if opdef is not None:
        # 如果 func 为 None，则返回 opdef 的注册假实现方法
        if func is None:
            return opdef.register_fake
        # 否则，返回 opdef 的注册假实现方法，并使用指定的 func
        else:
            return opdef.register_fake(func)

    # 如果 op 是字符串类型，则进行断言检查
    assert isinstance(op, str)

    # 获取调用栈深度信息
    stacklevel = _stacklevel
    # 定义一个装饰器函数，用于注册函数到 Torch 库中
    def register(func):
        # 解析操作名称的命名空间
        namespace, op_name = torch._library.utils.parse_namespace(op)
        # 如果库对象不存在，创建一个新的库对象
        if lib is None:
            use_lib = Library(namespace, "FRAGMENT")
            # 将创建的库对象添加到保持活动状态的列表中
            _keep_alive.append(use_lib)
        else:
            # 否则，使用已存在的库对象
            use_lib = lib
        # 在库对象中注册一个假的操作，指定函数和堆栈级别
        use_lib._register_fake(op_name, func, _stacklevel=stacklevel + 1)
        # 返回注册的函数对象
        return func

    # 如果没有传入函数，返回注册函数本身
    if func is None:
        return register
    else:
        # 否则，增加堆栈级别并注册函数
        stacklevel += 1
        return register(func)
# 注册自定义操作的自动求导函数
def register_autograd(
    # op 参数指定操作的标识符，类型为 _op_identifier
    op: _op_identifier,
    # backward 参数是反向传播函数，用于计算梯度
    backward: Callable,
    /,
    # setup_context 是可选参数，用于在前向传播期间保存需要的值
    *,
    setup_context: Optional[Callable] = None,
    # lib 参数未指定类型，默认为 None
    lib=None,
) -> None:
    r"""Register a backward formula for this custom op.

    In order for an operator to work with autograd, you need to register
    a backward formula:
    1. You must tell us how to compute gradients during the backward pass
    by providing us a "backward" function.
    2. If you need any values from the forward to compute gradients, you can
    use `setup_context` to save values for backward.

    ``backward`` runs during the backward pass. It accepts ``(ctx, *grads)``:
    - ``grads`` is one or more gradients. The number of gradients matches
    the number of outputs of the operator.
    The ``ctx`` object is `the same ctx object <context_method_mixins>`_ used by
    :class:`torch.autograd.Function`. The semantics of ``backward_fn`` are the
    same as :meth:`torch.autograd.Function.backward`.

    ``setup_context(ctx, inputs, output)`` runs during the forward pass.
    Please save quantities needed for backward onto the ``ctx`` object via
    either :meth:`torch.autograd.function.FunctionCtx.save_for_backward`
    or assigning them as attributes of ``ctx``. If your custom op has
    kwarg-only arguments, we expect the signature of ``setup_context``
    to be ``setup_context(ctx, inputs, keyword_only_inputs, output)``.

    Both ``setup_context_fn`` and ``backward_fn`` must be traceable. That is,
    they may not directly access :meth:`torch.Tensor.data_ptr` and they must
    not depend on or mutate global state. If you need a non-traceable backward,
    you can make it a separate custom_op that you call inside ``backward_fn``.
    # 检查输入的操作符类型，必须是字符串、OpOverload对象或CustomOpDef对象之一
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        # 如果类型不符合要求，抛出数值错误异常，显示实际接收到的类型
        raise ValueError(
            f"register_autograd(op): got unexpected type for op: {type(op)}"
        )
    
    # 如果op是OpOverload对象，将其转换为其名称的字符串形式
    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    
    # 尝试获取操作符op的操作定义
    opdef = _maybe_get_opdef(op)
    
    # 如果获取到操作定义opdef，则调用其register_autograd方法注册反向传播函数和上下文设置函数
    if opdef is not None:
        opdef.register_autograd(backward, setup_context=setup_context)
        return

    # 断言操作符op是一个字符串
    assert isinstance(op, str)
    
    # 根据操作符的限定名称查找操作符对象
    qualname = op
    op = torch._library.utils.lookup_op(qualname)
    
    # 获取操作符的模式（schema）
    schema = op._schema
    
    # 如果操作符的模式不是函数模式，抛出运行时错误，提示不支持注册非函数操作符的自动微分公式
    if not _library.utils.is_functional_schema(schema):
        raise RuntimeError(
            f"Cannot register autograd formula for non-functional operator "
            f"{op} with schema {schema}. Please create "
            f"a functional operator and register an autograd formula for that."
        )
    # 检查 schema 是否包含仅限关键字参数的张量，如果是则抛出 NotImplementedError
    if _library.utils.has_kwarg_only_tensors(schema):
        raise NotImplementedError(
            f"register_autograd with kwarg-only Tensor args. In the original "
            f"definition of the op, please make your tensors not kwarg-only. "
            f"Got: {schema}"
        )

    # 创建 autograd.Info 对象，用于管理反向传播和设置上下文
    info = _library.autograd.Info(backward, setup_context)
    
    # 根据 op 和 info 创建 autograd_kernel 对象
    autograd_kernel = _library.autograd.make_autograd_impl(op, info)
    
    # 解析限定名称（qualname）获取命名空间和操作名称
    namespace, opname = torch._library.utils.parse_namespace(qualname)
    
    # 如果 lib 为 None，则创建 Library 对象，并将其添加到 _keep_alive 列表以防止被垃圾回收
    if lib is None:
        lib = Library(namespace, "FRAGMENT")
        _keep_alive.append(lib)
    
    # 向 lib 中注册 autograd_kernel，指定类型为 "Autograd"，并设置 with_keyset=True
    lib.impl(opname, autograd_kernel, "Autograd", with_keyset=True)
# 检查 Python stubs 函数，确保对应的操作符是在 C++ 中定义的，并且存在 m.set_python_module(module, ...) 调用，并且该 module 与调用 torch.library.register_fake 的模块相同。
def _check_pystubs_once(func, qualname, actual_module_name):
    checked = False

    def inner(*args, **kwargs):
        nonlocal checked
        # 如果已经检查过，则直接调用原始函数并返回结果
        if checked:
            return func(*args, **kwargs)

        # 查找操作符 op 是否在 C++ 中定义
        op = torch._library.utils.lookup_op(qualname)
        if op._defined_in_python:
            # 如果操作符在 Python 中定义，则标记为已检查，并调用原始函数并返回结果
            checked = True
            return func(*args, **kwargs)

        # 否则，尝试获取可能的 Python stub
        maybe_pystub = torch._C._dispatch_pystub(
            op._schema.name, op._schema.overload_name
        )
        if maybe_pystub is None:
            # 如果没有找到 Python stub，并且需要设置 Python module，则抛出运行时错误
            if torch._library.utils.requires_set_python_module():
                namespace = op.namespace
                cpp_filename = op._handle.debug()
                raise RuntimeError(
                    f"Operator '{qualname}' was defined in C++ and has a Python "
                    f"fake impl. In this situation, we require there to also be a "
                    f'companion C++ `m.set_python_module("{actual_module_name}")` '
                    f"call, but we could not find one. Please add that to "
                    f"to the top of the C++ TORCH_LIBRARY({namespace}, ...) block the "
                    f"operator was registered in ({cpp_filename})"
                )
        else:
            # 如果找到了 Python stub，则检查其指定的模块是否与实际模块名相符合，否则抛出运行时错误
            pystub_module = maybe_pystub[0]
            if actual_module_name != pystub_module:
                cpp_filename = op._handle.debug()
                raise RuntimeError(
                    f"Operator '{qualname}' specified that its python fake impl "
                    f"is in the Python module '{pystub_module}' but it was actually found "
                    f"in '{actual_module_name}'. Please either move the fake impl "
                    f"or correct the m.set_python_module call ({cpp_filename})"
                )
        checked = True
        # 标记为已检查，并调用原始函数并返回结果
        return func(*args, **kwargs)

    return inner


# NOTE [ctx inside the fake implementation]
# 如果用户有一个具有数据相关输出形状的操作符，则在编写虚假实现时，必须查询当前的 ctx 并使用 ctx 上的方法构建一个新的未支持的 symint。
#
# 这通过在每次调用虚假实现时设置 global_ctx_getter 函数来完成。
def get_ctx() -> "torch._library.fake_impl.FakeImplCtx":
    """get_ctx() 返回当前的 AbstractImplCtx 对象。

    调用 ``get_ctx()`` 仅在虚假实现内部有效
    (详见 :func:`torch.library.register_fake` 以获取更多用法详情)。
    """
    return torch._library.fake_impl.global_ctx_getter()


# 默认的 opcheck 工具集合，用于测试不同方面的操作符注册和功能。
_OPCHECK_DEFAULT_UTILS = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def opcheck(
    op: Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket, CustomOpDef],
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    test_utils: Union[str, Sequence[str]] = _OPCHECK_DEFAULT_UTILS,
    raise_exception: bool = True,


    # 定义一个可选的关键字参数字典，如果未提供则默认为 None
    kwargs: Optional[Dict[str, Any]] = None,
    # 使用星号 (*) 表示后面的参数必须以关键字形式传递
    # 定义一个测试工具参数，可以是单个字符串或字符串序列，默认使用预设的工具集合 _OPCHECK_DEFAULT_UTILS
    test_utils: Union[str, Sequence[str]] = _OPCHECK_DEFAULT_UTILS,
    # 定义一个布尔型参数，控制是否抛出异常，默认为 True
    raise_exception: bool = True,
# 定义函数签名和文档字符串，描述了函数的作用和参数
def opcheck(
    op: Callable[..., Any],  # 参数op表示要测试的操作符，类型为可调用对象
    *args: Any,              # 不定长位置参数，传递给操作符op
    **kwargs: Any            # 不定长关键字参数，传递给操作符op
) -> Dict[str, str]:         # 函数返回一个字典，键和值的类型都为字符串

    """Given an operator and some sample arguments, tests if the operator is
    registered correctly.

    That is, when you use the torch.library/TORCH_LIBRARY APIs to create a
    custom op, you specified metadata (e.g. mutability info) about the custom op
    and these APIs require that the functions you pass them satisfy certain
    properties (e.g. no data pointer access in the fake/meta/abstract kernel)
    ``opcheck`` tests these metadata and properties.

    Concretely, we test the following:
    - test_schema: if the operator's schema is correct.
    - test_autograd_registration: if autograd was registered correctly.
    - test_faketensor: If the operator has a FakeTensor kernel
    (and if it is correct). The FakeTensor kernel is necessary (
    but not sufficient) for the operator to work with PyTorch compilation
    APIs (torch.compile/export/FX).
    - test_aot_dispatch_dynamic: If the operator has correct behavior
    with PyTorch compilation APIs (torch.compile/export/FX).
    This checks that the outputs (and gradients, if applicable) are the
    same under eager-mode PyTorch and torch.compile.
    This test is a superset of ``test_faketensor``.

    For best results, please call ``opcheck`` multiple times with a
    representative set of inputs. If your operator supports
    autograd, please use ``opcheck`` with inputs with ``requires_grad = True``;
    if your operator supports multiple devices (e.g. CPU and CUDA), please
    use ``opcheck`` with inputs on all supported devices.

    Args:
        op: The operator. Must either be a function decorated with
            :func:`torch.library.custom_op` or an OpOverload/OpOverloadPacket
            found in torch.ops.* (e.g. torch.ops.aten.sin, torch.ops.mylib.foo)
        args: The args to the operator
        kwargs: The kwargs to the operator
        test_utils: Tests that we should run. Default: all of them.
            Example: ("test_schema", "test_faketensor")
        raise_exception: If we should raise an exception on the first
            error. If False, we will return a dict with information
            on if each test passed or not.

    .. warning::

        opcheck and :func:`torch.autograd.gradcheck` test different things;
        opcheck tests if your usage of torch.library APIs is correct while
        :func:`torch.autograd.gradcheck` tests if your autograd formula is
        mathematically correct. Use both to test custom ops that support
        gradient computation.
    """
    """
    Import the internal testing operations module from Torch.
    导入 Torch 内部测试操作模块。

    Call the `opcheck` function from the imported module `optests`, passing `op`, `args`, `kwargs`,
    `test_utils=test_utils`, and `raise_exception=raise_exception` as arguments.
    调用从导入模块 `optests` 中的 `opcheck` 函数，传递 `op`、`args`、`kwargs`、
    `test_utils=test_utils` 和 `raise_exception=raise_exception` 作为参数。
    """
    import torch.testing._internal.optests as optests

    return optests.opcheck(
        op, args, kwargs, test_utils=test_utils, raise_exception=raise_exception
    )
```