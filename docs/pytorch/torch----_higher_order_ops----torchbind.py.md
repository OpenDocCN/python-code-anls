# `.\pytorch\torch\_higher_order_ops\torchbind.py`

```
# 设置类型检查工具 mypy，允许未声明类型的函数
mypy: allow-untyped-defs

# 导入 logging 模块，用于记录日志信息
import logging

# 导入 contextmanager 上下文管理器，用于创建上下文管理器
from contextlib import contextmanager

# 导入 torch 库，作为主要的计算库
import torch

# 导入 DispatchKey 枚举类型，用于指定分派键
from torch._C import DispatchKey  # @manual

# 导入 KNOWN_TYPES，用于标识已知类型
from torch._functorch._aot_autograd.utils import KNOWN_TYPES

# 导入 autograd_not_implemented，用于处理未实现的自动求导操作
from torch._higher_order_ops.utils import autograd_not_implemented

# 导入 _ns_and_class_name 和 FakeScriptObject，用于虚拟脚本对象管理
from torch._library.fake_class_registry import _ns_and_class_name, FakeScriptObject

# 导入 HigherOrderOperator，用于操作高阶操作符
from torch._ops import HigherOrderOperator

# 导入 FakeTensorMode，用于模拟张量的模式
from torch._subclasses.fake_tensor import FakeTensorMode

# 导入 ProxyTorchDispatchMode 和 track_tensor_tree，用于追踪代理 Torch 分派模式和张量树
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree

# 导入 has_side_effect，用于检查节点是否有副作用
from torch.fx.node import has_side_effect

# 导入 _pytree 模块，作为私有工具集的一部分
from torch.utils import _pytree as pytree

# 获取当前模块的日志记录器对象
log = logging.getLogger(__name__)

# call_torchbind 操作符表示在 torchbind 对象上的方法调用。其调用约定为：
#   call_torchbind(self: ScriptObject, method_name: str, *method_args, **method_kwargs)
# 我们不期望用户直接编写此操作符，而是在 Dynamo 追踪到 torchbind 对象时自动发出。
call_torchbind = HigherOrderOperator("call_torchbind")

# 在 FX 中注册此操作符为有副作用的。
# TODO: 这并不完全足够。虽然 passes（希望如此）检查 Node.is_impure() 并做出良好的决策，
# 我们也假设可以任意执行图形而不改变行为，但这对于改变 torchbind 对象状态的操作并非如此。
has_side_effect(call_torchbind)

# 保存 torch.ScriptMethod 的原始 __call__ 方法
_orig_scriptmethod_call = torch.ScriptMethod.__call__


def torchbind_method_redispatch(self, *args, **kwargs):
    # 如果 self.raw_owner 是 torch.ScriptObject 的实例，则调用 call_torchbind 进行分派
    if isinstance(self.raw_owner, torch.ScriptObject):
        return call_torchbind(self.raw_owner, self.name, *args, **kwargs)
    # 否则，调用原始的 __call__ 方法
    return _orig_scriptmethod_call(self, *args, **kwargs)


@contextmanager
def enable_torchbind_tracing():
    """作为功能标志的上下文管理器，用于启用 torchbind 追踪行为。
    一旦 torchbind 追踪稳定下来，我们可以移除此上下文管理器，并始终启用该功能。
    """
    try:
        # 将 torch.ScriptObject 添加到已知类型列表 KNOWN_TYPES 中
        KNOWN_TYPES.append(torch.ScriptObject)
        # 重定向 torch.ScriptMethod 的 __call__ 方法到 torchbind_method_redispatch
        torch.ScriptMethod.__call__ = torchbind_method_redispatch  # type: ignore[method-assign]
        yield
    finally:
        # 确保在追踪期间没有其他人修改了 KNOWN_TYPES，否则抛出异常
        assert (
            KNOWN_TYPES.pop() is torch.ScriptObject
        ), "Someone else messed with KNOWN_TYPES during tracing, exploding."
        # 恢复 torch.ScriptMethod 的原始 __call__ 方法
        torch.ScriptMethod.__call__ = _orig_scriptmethod_call  # type: ignore[method-assign]


@call_torchbind.py_impl(DispatchKey.CompositeExplicitAutograd)
def call_torchbind_impl(obj, method, *args, **kwargs):
    # 如果 obj 是 torch.ScriptObject 的实例，则调用其方法
    if isinstance(obj, torch.ScriptObject):
        return _orig_scriptmethod_call(getattr(obj, method), *args, **kwargs)
    # 如果 obj 是 FakeScriptObject 的实例，则调用其 wrapped_obj 对象的方法
    elif isinstance(obj, FakeScriptObject):
        return getattr(obj.wrapped_obj, method)(*args, **kwargs)
    # 否则，抛出运行时异常，表明不支持的第一个参数类型
    else:
        raise RuntimeError(f"Unsupported first arg type {type(obj)} for call_torchbind")


@call_torchbind.py_impl(ProxyTorchDispatchMode)
def inner(mode, *args, **kwargs):
    # 如果启用了跟踪模式
    if mode.enable_tracing:
        # 使用跟踪器的函数解包参数和关键字参数中的代理对象
        proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
        proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

        # 创建一个代理对象，用于调用 torchbind 方法
        out_proxy = mode.tracer.create_proxy(
            "call_function",
            call_torchbind,
            proxy_args,
            proxy_kwargs,
        )

        # 调用 torchbind 方法
        out = call_torchbind(*args, **kwargs)

        # 解包调用参数，获取对象、方法及其余参数
        obj, method, *rest_args = args

        # 如果对象是 torch.ScriptObject 类型，则记录警告信息
        if isinstance(obj, torch.ScriptObject):
            # 获取对象的命名空间和类名
            ns, class_name = _ns_and_class_name(
                obj._type().qualified_name()  # type: ignore[attr-defined]
            )
            # 记录警告信息，指出可能导致原始对象被修改
            log.warning(
                "Tracing torchbind method %s.%s with real ScriptObject. This may"
                " cause the original object being mutated. If this is not intended,"
                ' You can register a fake class with torch._library.register_fake_class("%s::%s").',
                class_name,
                method,
                ns,
                class_name,
            )

        # 跟踪张量树，并返回跟踪结果
        ret = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)

        # 如果代理节点的元数据中不包含 "val" 键
        if "val" not in out_proxy.node.meta:
            # 断言输出是 None 或者是 (int, float, bool) 中的一种类型
            assert out is None or isinstance(
                out, (int, float, bool)
            ), "Currently, only these constant dtypes are supported to be returned from torchbind methods."
            # 将输出值存储到代理节点的元数据中的 "val" 键下
            out_proxy.node.meta["val"] = out

        # 返回跟踪结果
        return ret
    else:
        # 如果未启用跟踪模式，则直接调用 torchbind 方法并返回其结果
        return call_torchbind(*args, **kwargs)
# TODO: 当前我们仅使用虚拟张量运行 C++ 实现。
# 但是我们应该使注册虚拟的 torchbind 实现成为可能。

# 使用装饰器将 call_torchbind_fake 函数注册为 torchbind.py_impl(FakeTensorMode) 的实现
@call_torchbind.py_impl(FakeTensorMode)
def call_torchbind_fake(mode, *args, **kwargs):
    # 使用 mode 上下文管理器执行以下代码块
    with mode:
        # 调用 call_torchbind_impl 函数，并返回其结果
        return call_torchbind_impl(*args, **kwargs)

# 调用 call_torchbind.py_impl 函数，参数为 DispatchKey.Autograd
# 并将其结果传递给 autograd_not_implemented 函数，参数为 call_torchbind 和 deferred_error=True
call_torchbind.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(call_torchbind, deferred_error=True)
)

# 使用装饰器将 call_torchbind_func 函数注册为 call_torchbind 的功能化实现
@call_torchbind.py_functionalize_impl
def call_torchbind_func(ctx, *args, **kwargs):
    # 从 torch._higher_order_ops.effects 导入 handle_effects 函数
    from torch._higher_order_ops.effects import handle_effects

    # 调用 handle_effects 函数，参数为：
    # ctx.mode._allow_token_discovery, ctx.mode._tokens, call_torchbind, args, kwargs
    return handle_effects(
        ctx.mode._allow_token_discovery, ctx.mode._tokens, call_torchbind, args, kwargs
    )
```