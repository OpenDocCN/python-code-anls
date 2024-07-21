# `.\pytorch\torch\fx\experimental\proxy_tensor.py`

```
# 忽略 mypy 类型检查错误

# 引入上下文管理器模块
import contextlib
# 引入 functools 模块
import functools
# 引入类型提示模块
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
# 引入 PyTorch 模块
import torch
import torch.utils._pytree as pytree
# 从 torch.fx 中引入 Tracer 和 GraphModule 类
from torch.fx import Tracer, GraphModule
# 从 torch.fx.graph_module 中引入 _assign_attr 函数
from torch.fx.graph_module import _assign_attr
# 引入弱引用字典模块
from weakref import WeakKeyDictionary
# 引入默认字典模块
from collections import defaultdict
# 从 torch._subclasses.fake_tensor 中引入相关类和函数
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily, is_fake
# 从 torch._dispatch.python 中引入相关函数
from torch._dispatch.python import enable_python_dispatcher
# 从 torch.fx 中引入 fx 模块
import torch.fx as fx
# 从 torch.fx.node 中引入 _side_effectful_need_to_be_preserved_pre_dispatch 函数
from torch.fx.node import _side_effectful_need_to_be_preserved_pre_dispatch
# 从 torch.fx.passes.shape_prop 中引入 _extract_tensor_metadata 函数
from torch.fx.passes.shape_prop import _extract_tensor_metadata
# 从 contextlib 中引入 contextmanager 和 nullcontext 函数
from contextlib import contextmanager, nullcontext
# 引入 inspect 模块
import inspect
# 从 dataclasses 中引入 dataclass 装饰器
from dataclasses import dataclass
# 引入 weakref 模块
import weakref
# 引入 operator 模块
import operator
# 引入 traceback 模块
import traceback
# 从 torch.utils._stats 中引入 count 函数
from torch.utils._stats import count
# 从 torch.utils._traceback 中引入 CapturedTraceback 类
from torch.utils._traceback import CapturedTraceback
# 引入日志记录模块
import logging
# 从 torch._library.fake_class_registry 中引入 FakeScriptObject 类
from torch._library.fake_class_registry import FakeScriptObject
# 引入警告模块
import warnings

# 从 torch.overrides 中引入 TorchFunctionMode 类
from torch.overrides import TorchFunctionMode

# 从 torch.utils._python_dispatch 中引入相关函数和类
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _disable_infra_mode,
    _push_mode,
    _unset_infra_mode,
)

# 从当前目录的相关模块中引入类和函数
from ._backward_state import BackwardState
from .sym_node import SymNode
from ._sym_dispatch_mode import SymDispatchMode
# 从 torch.fx 中引入 Proxy 类
from torch.fx import Proxy
# 从 torch.fx.traceback 中引入相关函数
import torch.fx.traceback as fx_traceback
# 从 torch 中引入 SymInt、SymFloat 和 SymBool 类
from torch import SymInt, SymFloat, SymBool
# 从 torch.utils.weak 中引入相关类和函数
from torch.utils.weak import WeakTensorKeyDictionary, WeakIdKeyDictionary, _WeakHashRef

# 定义 __all__ 列表，指定模块中公开的对象
__all__ = ["PythonKeyTracer", "dispatch_trace", "make_fx", "DecompositionInterpreter", "py_sym_types", "get_innermost_proxy_mode"]

# 引用 torch.ops.aten 和 torch.ops.prim 模块
aten = torch.ops.aten
prim = torch.ops.prim

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)
# 获取未实现功能的日志记录器
not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")

# 定义全局变量 CURRENT_DECOMPOSITION_TABLE，并初始化为空字典
CURRENT_DECOMPOSITION_TABLE: Dict[torch._ops.OperatorBase, Callable] = {}

# 定义常量 CONSTANT_NUMEL_LIMIT，设定为 1
CONSTANT_NUMEL_LIMIT = 1

# 获取 nullcontext 函数的类型
null_ctx_type = type(nullcontext)

# 注册 torch.Size 类型在 pytree 模块中的处理函数，用于序列化和反序列化
pytree.register_pytree_node(
    torch.Size,
    lambda xs: (list(xs), None),
    lambda xs, _: tuple(xs),
    flatten_with_keys_fn=lambda xs: (
        [(pytree.SequenceKey(i), x) for i, x in enumerate(xs)],
        None,
    ),
)

# 定义 fake_signature 函数，用于处理 FX 对于变长参数的困惑
def fake_signature(fn, nargs):
    """FX gets confused by varargs, de-confuse it"""
    # 生成一个 lambda 函数，用于简化函数签名
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})

# 定义 decompose 上下文管理器函数，用于临时设置当前的分解表
@contextmanager
def decompose(decomposition_table):
    global CURRENT_DECOMPOSITION_TABLE
    # 保存当前的分解表，并设置新的分解表
    old_decomposition_table = CURRENT_DECOMPOSITION_TABLE
    CURRENT_DECOMPOSITION_TABLE = decomposition_table
    try:
        yield CURRENT_DECOMPOSITION_TABLE
    finally:
        # 恢复之前保存的分解表
        CURRENT_DECOMPOSITION_TABLE = old_decomposition_table
    finally:
        # 在异常处理结束后，恢复原始的CURRENT_DECOMPOSITION_TABLE值
        CURRENT_DECOMPOSITION_TABLE = old_decomposition_table
# 确保我们不会与其他属性发生冲突
proxy_slot = object()
# 无默认值的对象，用于标记默认情况下不应该使用的值
no_default = object()

# Python 符号类型的元组
py_sym_types = (SymInt, SymFloat, SymBool)

# 检查节点是否是符号节点
def is_sym_node(node):
    # 断言：所有使用 proxy_tensor 跟踪的节点都应该有 meta 属性
    assert hasattr(node, 'meta'), "All nodes traced with proxy_tensor should have meta"
    # 检查节点的 meta 属性中是否有 'val' 键，并且值的类型属于 py_sym_types 中定义的类型
    return "val" in node.meta and isinstance(node.meta['val'], py_sym_types)

# 设置对象的代理槽位
def set_proxy_slot(obj, tracer, proxy):
    if isinstance(obj, torch.Tensor):
        # 对于张量，每当我们执行原地操作并影响代理的元数据时，我们确实希望覆盖代理
        tracer.tensor_tracker[obj] = proxy
    elif isinstance(obj, (torch.ScriptObject, FakeScriptObject)):
        # 对于脚本对象和伪脚本对象，出于类似张量的原因，我们确实希望覆盖代理
        tracer.script_object_tracker[obj] = proxy
    else:
        # 注意：永远不要覆盖已存在的代理。尽管代理在原则上是等效的，
        # 但在进行图分割时，我们不希望出现对切线输入的不必要依赖。
        # 这是因为首先设置原始值的 SymInt，然后再分配切线输入。
        # 确保如果可以从原始值推导出 SymInt，则使用它。
        assert isinstance(obj, py_sym_types), type(obj)
        if obj not in tracer.symnode_tracker:
            tracer.symnode_tracker[obj] = proxy

# 检查对象是否具有代理槽位
def has_proxy_slot(obj, tracer):
    assert isinstance(obj, (torch.Tensor, SymNode)), type(obj)
    return get_proxy_slot(obj, tracer, False, lambda _: True)

# 获取对象的代理槽位
# 默认参数是当槽位未设置时返回的值。
# transform 参数在需要从成功查找的结果中提取子字段时很方便（但不包括默认值）。
def get_proxy_slot(obj, tracer, default=no_default, transform=lambda x: x):
    if isinstance(obj, torch.Tensor):
        tracker = tracer.tensor_tracker
    elif isinstance(obj, (torch.ScriptObject, FakeScriptObject)):
        tracker = tracer.script_object_tracker
    else:
        assert isinstance(obj, py_sym_types), type(obj)
        tracker = tracer.symnode_tracker

    if obj not in tracker:
        if default is no_default:
            raise RuntimeError(f"{obj} is not tracked with proxy for {tracer}")
        return default
    return transform(tracker[obj])

# 快照伪对象的值
def snapshot_fake(val):
    return val.detach()

# 提取值的函数
def extract_val(val):
    if is_fake(val):
        return snapshot_fake(val)
    elif isinstance(val, py_sym_types):
        return val
    elif isinstance(val, (torch.ScriptObject, FakeScriptObject)):
        return val
    elif isinstance(val, BackwardState):
        return val
    elif isinstance(val, (list, tuple)):
        return val.__class__([extract_val(x) for x in val])
    # 如果 val 是 torch.Tensor 类型，并且不是稀疏张量
    elif isinstance(val, torch.Tensor):
        # 如果 val 不是稀疏张量，则执行以下操作
        if not val.is_sparse:
            # 注意：这里的方法可能有些巧妙，但我们应该尽可能在所有地方获取 val 作为元数据
            # TODO: 这里没有正确追踪存储。更健壮的方法是维护每个跟踪的 FakeTensorMode，并从真实张量创建假值（别忘了快照 snapshot_fake）
            # 创建一个允许回退内核的 FakeTensorMode 上下文
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
            # 在 fake_tensor_mode 上下文中执行以下操作
            with fake_tensor_mode:
                # 返回一个与 val 具有相同形状和步幅的空张量，使用 val 的设备和数据类型
                return torch.empty_strided(val.shape, val.stride(), device=val.device, dtype=val.dtype)
        else:
            # 如果 val 是稀疏张量，则返回 None
            return None
    # 如果 val 是 int、float 或 bool 类型之一
    elif isinstance(val, (int, float, bool)):
        # 直接返回 val
        return val
# 设置_meta字典中的'val'字段，将提取的val值作为其值
def set_meta(proxy, val):
    proxy.node.meta['val'] = extract_val(val)

    # 如果val是假数据（fake），则尝试设置'tensor_meta'字段为val的张量元数据
    if is_fake(val):
        proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(val)
    # 如果val是torch.Tensor类型且不是稀疏张量，则设置'tensor_meta'字段为val的张量元数据
    elif isinstance(val, torch.Tensor) and not val.is_sparse:
        proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(val)
    # 返回设置好meta后的proxy
    return proxy

# 延迟执行函数f，同时缓存结果
def thunkify(f, *args, **kwargs):
    """
    Delays computation of f until it's called again
    Also caches the result
    """
    return functools.lru_cache(1)(functools.partial(f, *args, **kwargs))

# 跟踪张量tensor的元数据，将其关联到指定的proxy上
def track_tensor(tensor, proxy, *, constant, tracer):
    # 尝试设置代理槽（proxy slot），将proxy关联到每个张量维度s的SymInt对象上
    def try_set_proxy_slot(outer_s, proxy_callable, *args):
        assert callable(proxy_callable)
        # 如果outer_s是SymInt类型，则设置代理槽为thunkify(proxy_callable, outer_s, *args)的结果
        if isinstance(outer_s, SymInt):
            set_proxy_slot(outer_s, tracer, thunkify(proxy_callable, outer_s, *args))
    
    # 遍历张量的每个维度s，并尝试设置代理槽
    for i, s in enumerate(tensor.shape):
        try_set_proxy_slot(s, lambda x, i: set_meta(
            tracer.create_proxy('call_function', torch.ops.aten.sym_size.int, (proxy, i), {}), x), i)

    # 遍历张量的每个步长维度s，并尝试设置代理槽
    for i, s in enumerate(tensor.stride()):
        try_set_proxy_slot(s, lambda x, i: set_meta(
            tracer.create_proxy('call_function', torch.ops.aten.sym_stride.int, (proxy, i), {}), x), i)

    # 尝试设置代理槽为张量的元素数量numel()的结果
    try_set_proxy_slot(tensor.numel(), lambda x: set_meta(
        tracer.create_proxy('call_function', torch.ops.aten.sym_numel.default, (proxy,), {}), x))
    # 尝试设置代理槽为张量的存储偏移量storage_offset()的结果
    try_set_proxy_slot(tensor.storage_offset(), lambda x: set_meta(
        tracer.create_proxy('call_function', torch.ops.aten.sym_storage_offset.default, (proxy,)), x))
    
    # 将proxy关联到张量的代理槽，并设置为_ProxyTensor(proxy, constant)
    set_proxy_slot(tensor, tracer, _ProxyTensor(proxy, constant))

# 跟踪张量树结构的内部结果inner_res和代理结果proxy_res，设置未备份绑定
def track_tensor_tree(inner_res, proxy_res, *, constant, tracer):
    _set_unbacked_bindings(inner_res, proxy_res)
    # 定义一个函数，将输入对象 e 包装为带有代理的对象，并设置元数据
    def wrap_with_proxy(e, proxy, constant):
        # 如果 e 是 torch.Tensor 类型
        if isinstance(e, torch.Tensor):
            # 跟踪该张量，使用 tracer 和常量进行跟踪
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            # 设置代理的元数据为 e
            set_meta(proxy, e)
        # 如果 e 是 py_sym_types 类型的实例
        elif isinstance(e, py_sym_types):
            # 注意：这里提前设置元数据，以保证编号是按顺序的
            set_meta(proxy, e)
            # 设置代理的槽位，使其与 tracer 和 proxy 绑定
            set_proxy_slot(e, tracer, lambda: proxy)
        # 如果 e 是 torch.ScriptObject 或 FakeScriptObject 类型
        elif isinstance(e, (torch.ScriptObject, FakeScriptObject)):
            # 设置代理的槽位，使其与 tracer 绑定
            set_proxy_slot(e, tracer, proxy)
            # 设置代理的元数据为 e
            set_meta(proxy, e)
        # 如果 e 是 tuple 或 list 类型
        elif isinstance(e, (tuple, list)):
            # 如果 proxy 是 fx.Proxy 类型
            if isinstance(proxy, fx.Proxy):
                # 设置代理的元数据为 e
                set_meta(proxy, e)

            # 示例用例：allreduce_ 返回 ([tensor], work)
            # 遍历 e 中的每个元素，递归调用 wrap_with_proxy 函数
            for idx, ee in enumerate(e):
                wrap_with_proxy(ee, proxy[idx], get_constant(idx))
        # 如果 e 是 dict 类型
        elif isinstance(e, dict):
            # 理论上可以支持在 proxy-tensor-tracing 时进行常量传播
            # 返回一个字典的操作符，但我们当前没有这样的用例
            assert constant is None

            # 如果 proxy 是 fx.Proxy 类型
            if isinstance(proxy, fx.Proxy):
                # 设置代理的元数据为 e
                set_meta(proxy, e)

            # 示例用例：triton_kernel_wrapper 以 kwargs 形式接受参数
            # 遍历 e 中的每个键值对，递归调用 wrap_with_proxy 函数
            for key, val in e.items():
                wrap_with_proxy(val, proxy[key], None)
        # 如果 e 是 BackwardState 类型
        elif isinstance(e, BackwardState):
            # 设置代理的元数据为 e
            set_meta(proxy, e)
            # 设置 e 的代理为 proxy
            e.proxy = proxy
        else:
            # 故意跳过基本类型
            pass

    # 定义一个函数，根据索引返回常量列表中的常量值
    def get_constant(idx):
        if constant is None:
            return None
        else:
            return constant[idx]

    # 调用 wrap_with_proxy 函数，处理 inner_res 和 proxy_res
    wrap_with_proxy(inner_res, proxy_res, constant)

    # 返回 inner_res
    return inner_res
def maybe_disable_fake_tensor_mode():
    # TODO: figure out if this API generally makes sense and integrate it into
    # the library
    # 考虑此 API 是否通常有意义，并将其整合到库中
    return unset_fake_temporarily()

@dataclass
class _ProxyTensor:
    # Dataclass representing a proxy tensor with a proxy and optional constant tensor
    # 代表具有代理和可选常量张量的数据类
    proxy: Proxy
    constant: Optional[torch.Tensor]

def fetch_sym_proxy(tracer):
    # Returns a function to fetch symbolic proxy values based on an expression
    # 根据表达式返回一个函数来获取符号代理值
    def inner(e):
        n = e.node
        if n.constant is not None:
            return n.constant
        if e.node.expr.is_number:
            if isinstance(e, SymBool):
                return bool(e.node.expr)
            elif isinstance(e, SymInt):
                return int(e.node.expr)
            return float(e.node.expr)
        else:
            # NB: we REQUIRE all symints to be tracked
            # 注意：我们要求所有 SymInts 都被跟踪
            return get_proxy_slot(e, tracer)()
    return inner

def fetch_object_proxy(tracer):
    # Returns a lambda function to fetch object proxies based on a tracer
    # 根据追踪器返回一个 lambda 函数来获取对象代理
    return lambda t: get_proxy_slot(t, tracer, t)

HANDLED_TYPES = (torch.Tensor, torch.nn.Parameter, FakeTensor)

def proxy_call(proxy_mode, func, pre_dispatch, args, kwargs):
    unrecognized_types = []
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))

    def can_handle_tensor(x):
        # Determines if a tensor can be handled by the proxy mode
        # 确定张量是否可以被代理模式处理
        r = type(x) in HANDLED_TYPES or has_proxy_slot(x, proxy_mode.tracer)
        if proxy_mode._allow_fake_constant:
            r = r or type(x) in (torch._subclasses.FakeTensor,)
        if not r:
            unrecognized_types.append(type(x))
        return r

    # If there are any tensor subclasses, we need to handle those tensor subclasses first
    # 如果存在任何张量子类，我们需要首先处理这些张量子类
    # TODO: we could use types to test this
    if not all(
        can_handle_tensor(x) for x in flat_args_kwargs if isinstance(x, torch.Tensor)
    ):
        not_implemented_log.debug(
            "ProxyTensorMode tensors without proxy had unrecognized subclasses: %s",
            unrecognized_types,
        )
        return NotImplemented

    r = maybe_handle_decomp(proxy_mode, func, args, kwargs)
    if r is not NotImplemented:
        return r

    # For pre-autograd tracing, we do not want to run CompositeImplicit decomps.
    # 对于预自动微分追踪，我们不希望运行 CompositeImplicit 解码
    if not pre_dispatch and func not in [
        torch.ops.aten.size.default,
        torch.ops.aten.stride.default,
        torch.ops.aten.storage_offset.default,
    ]:
        with proxy_mode:
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r

    tracer = proxy_mode.tracer
    f_flat_args_kwargs = [
        (
            fetch_object_proxy(tracer)(x)
            if isinstance(x, (torch.Tensor, torch.ScriptObject, FakeScriptObject))
            else x
        )
        for x in flat_args_kwargs
    ]

    # If there are SymInts, we also should not consider this constant.
    # However, fake tensor handling of SymInts is sufficiently broken that
    # I couldn't write a test for this case
    # 如果存在 SymInts，我们也不应考虑这个常量。
    # 然而，对 SymInts 的虚假张量处理已经足够破碎，我无法为这种情况编写测试
    # 检查所有的参数和关键字参数是否都是常量
    all_constant = (
        not any(
            t.constant is None  # 检查是否有代理张量且其常量为None
            for t in f_flat_args_kwargs  # 遍历扁平化后的参数和关键字参数
            if isinstance(t, _ProxyTensor)  # 检查是否为代理张量
        )
        # TODO: 或许常量SymInts也应该被允许？不确定这种情况是否会发生
        and not any(
            isinstance(x, (SymInt, SymFloat, SymBool))  # 检查是否为SymInt, SymFloat或SymBool类型
            for x in flat_args_kwargs  # 遍历扁平化的参数和关键字参数
        )
    )

    if torch.Tag.data_dependent_output in func.tags:
        # 检查张量输入是否全为常量
        if all_constant:
            # 如果所有输入都是常量，提取常量参数和关键字参数
            const_flat_args_kwargs = [
                t.constant if isinstance(t, _ProxyTensor) else t  # 提取代理张量的常量
                for t in f_flat_args_kwargs  # 遍历扁平化后的参数和关键字参数
            ]
            const_args, const_kwargs = pytree.tree_unflatten(
                const_flat_args_kwargs, spec  # 使用规范将其解构回参数和关键字参数
            )
            with maybe_disable_fake_tensor_mode():
                return func(*const_args, **const_kwargs)  # 调用函数并返回结果
        # 如果任何一个张量输入不是FakeTensor，允许此访问可能会错误地烧入常量。在这种情况下引发错误
        if proxy_mode._error_on_data_dependent_ops and pytree.tree_all_only(
            torch.Tensor, lambda t: not is_fake(t), (args, kwargs)  # 检查所有张量是否不是FakeTensor
        ):
            raise RuntimeError(
                f"It appears that you're trying to get value out of a tracing tensor with {func} - erroring out! "
                "It's likely that this is caused by data-dependent control flow or similar.  "
                "It may be possible to trace this with dynamic shapes; try setting tracing_mode='symbolic' "
                "in your make_fx call."
            )

    # 提取代理张量或直接使用原始输入创建代理参数和关键字参数
    proxy_flat_args_kwargs = [
        e.proxy if isinstance(e, _ProxyTensor) else e  # 如果是代理张量则提取代理，否则使用原始值
        for e in f_flat_args_kwargs  # 遍历扁平化后的参数和关键字参数
    ]
    # 对于符号整数、符号浮点数和符号布尔值，使用相应的函数获取符号代理，否则保持原始值
    proxy_flat_args_kwargs = [
        (
            fetch_sym_proxy(proxy_mode.tracer)(e)  # 获取符号代理
            if isinstance(e, (SymInt, SymFloat, SymBool))  # 检查是否为符号整数、符号浮点数或符号布尔值
            else e  # 否则保持原始值
        )
        for e in proxy_flat_args_kwargs  # 遍历扁平化后的参数和关键字参数
    ]
    proxy_args, proxy_kwargs = pytree.tree_unflatten(proxy_flat_args_kwargs, spec)  # 使用规范将其解构回参数和关键字参数

    # 当我们通过torch.tensor调用进行追踪时，实际上永远不会看到torch.ops.aten.tensor调用。
    # 相反，该函数在内部实现时会分配一个普通张量（这是*保证*的，我们在执行时禁用所有模式），
    # 然后对其调用at::lift_fresh（以便让模式有机会执行其操作）。
    # 此外，lift_fresh的张量参数保证是新分配的，因此我们希望lift_fresh成为一个无操作（直接返回输入参数）。
    #
    # 基本问题在于：当我们将这个调用序列追踪到FX图中时，这个调用序列会发生什么？
    # 传统上，张量常量在FX图模块上被作为缓冲区进行内部存储。但这是危险的。考虑：
    #
    #       x = torch.tensor(1)
    #       x.add_(2)
    #
    # Naively, this traces into:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh(t)
    #       x.add_(2)
    #
    # If lift_fresh returns t directly, the subsequent add_ call will
    # modify the tensor constant. Really, the problem is we've violated
    # the invariant the argument to lift is fresh.  So what we should
    # preserve the invariant by replacing lift_fresh with lift_fresh_copy:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh_copy(t)
    #       x.add_(2)
    #
    # This is what the overload modification does.
    如果 func 是 torch.ops.aten.lift_fresh.default：
        # 替换 func 为 torch.ops.aten.lift_fresh_copy.default，以保持参数是新鲜的不变性
        func = torch.ops.aten.lift_fresh_copy.default

    # 使用代理模式创建一个代理对象，用于调用函数
    proxy_out = proxy_mode.tracer.create_proxy(
        "call_function",
        func,
        proxy_args,
        proxy_kwargs,
        name=proxy_mode.tracer.graph._target_to_str(func.overloadpacket.__name__),  # 获取函数名字符串
    )

    # 这使得 DCE（死代码删除）对 inplace 操作的影响稍微降低
    # 这不是绝对必要的
    # 这是一种通过测试操作是否为 inplace 来测试操作的方法
    if (
        func.overloadpacket.__name__[-1] == "_"
        and func.overloadpacket.__name__[0] != "_"
    ):
        if isinstance(args[0], List):
            # 例如，c10d::allreduce_ 返回一个张量列表作为输出的第一个元素
            # 将代理对象 proxy_out 分配给列表中的每个张量
            for i, a in enumerate(args[0]):
                a.proxy = proxy_out[0][i]
        else:
            # 将代理对象 proxy_out 分配给第一个参数
            args[0].proxy = proxy_out

    # 调用函数 func，并获取返回值
    out = func(*args, **kwargs)

    # 在某些情况下，我们会在追踪过程中遇到一个张量在静态情况下已知是常量
    # （目前仅发生在运行 torch.tensor 时；像 torch.arange 这样的确定性工厂函数不会受到这种处理的影响）。
    # 当涉及的张量很小的时候，如果我们调用 item()，进行常量传播会很有帮助
    # （此时我们可以返回已知的常量值，而不是返回错误）。
    # 这里的逻辑测试是否可以进行常量传播（因为所有的输入都是常量）。如果可以，我们会禁用虚假张量模式（如果它开启的话），并对常量进行真实的计算。
    #
    # 值得强调的是，我们在这里做了一个策略决策。
    # 可能存在张量实际上非常大的情况，我们不想运行计算。
    # 张量非常大是为什么工厂函数不会受到这种处理的影响的原因之一（因为它们可能非常大；如果参数初始化为常量值，它将是！）。
    # 类似地，还有可能运行一个使小张量的大小增加的运算符；我们不保护这种情况，但我们可以强制，例如，只进行单元素常量计算通过在返回结果之前测试结果的 numel
    # 检查是否存在任何常量值，由于const-ness传播。同样，常量不一定需要存储在CPU上，但是可能会存在。
    any_constant = any(
        t.constant is not None  # 检查是否存在常量
        for t in f_flat_args_kwargs  # 遍历扁平化后的参数和关键字参数列表
        if isinstance(t, _ProxyTensor)  # 仅考虑代理张量类型
    )

    constant = None  # 初始化常量为None

    # 如果这是一个lift操作，并且输出张量元素个数小于等于CONSTANT_NUMEL_LIMIT
    if (
        func is torch.ops.aten.lift_fresh_copy.default  # 检查函数是否是lift操作的默认实现
        and out.numel() <= CONSTANT_NUMEL_LIMIT  # 检查输出张量元素个数是否不超过限制
    ):
        # 在可能的情况下禁用假张量模式，并复制输入的第一个参数作为常量
        with maybe_disable_fake_tensor_mode():
            constant = args[0].clone()
    # 否则，如果函数不带torch.Tag.nondeterministic_seeded标签、所有输入参数均为常量、存在任何一个常量、
    # 并且输出张量中所有张量元素个数均不超过CONSTANT_NUMEL_LIMIT
    elif (
        torch.Tag.nondeterministic_seeded not in func.tags  # 检查函数标签是否不包含nondeterministic_seeded标签
        and all_constant  # 检查所有输入参数是否都是常量
        and any_constant  # 检查是否存在任何一个常量
        and pytree.tree_all_only(
            torch.Tensor, lambda t: t.numel() <= CONSTANT_NUMEL_LIMIT, out  # 检查输出张量中所有张量的元素个数是否不超过限制
        )
    ):
        # 注意：不要将工厂函数包括在常量中
        with maybe_disable_fake_tensor_mode():
            # 构建仅包含常量的扁平化参数和关键字参数列表
            const_flat_args_kwargs = [
                t.constant if isinstance(t, _ProxyTensor) else t
                for t in f_flat_args_kwargs
            ]
            # 将扁平化的常量参数和关键字参数重新组合成原始结构
            const_args, const_kwargs = pytree.tree_unflatten(
                const_flat_args_kwargs, spec
            )
            # 通过调用函数生成常量值
            constant = func(*const_args, **const_kwargs)
    else:
        constant = None  # 否则将常量设置为None

    # 跟踪张量树的结构，包括输出张量、代理输出、以及确定的常量和追踪器
    track_tensor_tree(out, proxy_out, constant=constant, tracer=tracer)
    # 返回输出张量
    return out
class _SymNodeDict:
    """
    Wrapper around a dictionary that will hash SymInts with their nodes
    """
    def __init__(self):
        # Initialize an empty dictionary to store SymInt nodes as keys and corresponding values
        self.sym_node_dict = {}

    def __setitem__(self, key: py_sym_types, value: Any):
        # Set an item in sym_node_dict with the node of the key as the key and the provided value
        self.sym_node_dict[key.node] = value

    def __getitem__(self, key: py_sym_types):
        # Retrieve the value associated with the node of the key from sym_node_dict
        return self.sym_node_dict[key.node]

    def __contains__(self, key: py_sym_types):
        # Check if the node of the key exists in sym_node_dict
        return key.node in self.sym_node_dict

    def get(self, key: py_sym_types, default: Any = None):
        # Retrieve the value associated with the node of the key from sym_node_dict
        # If the key does not exist, return the default value provided
        return self.sym_node_dict.get(key.node, default)


class PythonKeyTracer(Tracer):
    def __init__(self):
        # Initialize the PythonKeyTracer object inheriting from Tracer
        super().__init__(autowrap_modules=())
        # Initialize an empty WeakTensorKeyDictionary to track tensor keys weakly
        self.tensor_tracker = WeakTensorKeyDictionary()
        # Initialize an instance of _SymNodeDict to track symbolic nodes and their values
        self.symnode_tracker = _SymNodeDict()  # type: ignore[var-annotated]
        # Initialize a WeakIdKeyDictionary for script object tracking, using _WeakHashRef for reference
        self.script_object_tracker = WeakIdKeyDictionary(dict=None, ref_type=_WeakHashRef)

        # Stores metadata of the torch function called during tracing
        self.torch_fn_metadata = None
        # Stores counts of every torch function call to distinguish different calls to the same function
        self.torch_fn_counts = {}

    # Override call_module to execute the forward method of a torch.nn.Module instance
    # with the given arguments and keyword arguments
    def call_module(
            self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        return forward(*args, **kwargs)

    # Override getattr to return the actual value of an attribute instead of creating a proxy
    def getattr(self, attr, attr_val, parameter_proxy_cache):
        return attr_val

    # Create an argument node representation for tracing
    def create_arg(self, a: Any):
        if isinstance(a, torch.nn.Parameter):
            # If the argument is a torch.nn.Parameter, create a node for accessing its attribute
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            qualname: Optional[str] = None

            if not qualname:
                qualname = self.get_fresh_qualname("_param_constant")
                setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {})
        elif isinstance(a, (SymInt, SymFloat, SymBool)):
            # If the argument is a symbolic type, return its constant node
            assert a.node.constant is not None
            return a.node.constant
        return super().create_arg(a)

    # Unwrap proxies to access underlying torch.Tensor, SymInt, SymFloat, SymBool, or ScriptObject
    def unwrap_proxy(self, e):
        if isinstance(e, torch.Tensor):
            return get_proxy_slot(e, self, e, lambda e: e.proxy)
        elif isinstance(e, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return get_proxy_slot(e, self, e, lambda e: e())
        elif isinstance(e, (torch.ScriptObject, FakeScriptObject)):
            return get_proxy_slot(e, self, e)
        else:
            return e


@contextmanager
def _temp_remove_pre_dispatch_torch_function_mode():
    # Import necessary functions for managing torch function dispatch mode
    from torch.overrides import _len_torch_function_stack, _pop_mode, _push_mode
    # 创建一个空列表，用于临时存储模式元素
    temp_elements = []
    # 初始化预分发模式为 None
    pre_dispatch_mode = None

    # 当前正在处理的 Torch 函数调用堆栈中仍有元素时执行循环
    while _len_torch_function_stack() > 0:
        # 从调用堆栈中弹出一个模式对象
        mode = _pop_mode()
        # 如果弹出的模式是预分发模式的实例
        if isinstance(mode, PreDispatchTorchFunctionMode):
            # 将当前模式设置为预分发模式，并结束循环
            pre_dispatch_mode = mode
            break
        else:
            # 将非预分发模式的模式对象添加到临时列表中
            temp_elements.append(mode)

    # 将临时列表中的模式对象按照相反的顺序重新推入调用堆栈中
    for mode in reversed(temp_elements):
        _push_mode(mode)

    try:
        # 执行 yield 表达式，暂停生成器的执行并返回调用者
        yield

    finally:
        # 在最终处理中，如果存在预分发模式
        if pre_dispatch_mode is not None:
            # 计算临时列表中的元素个数
            count = len(temp_elements)
            # 将临时列表中的元素按顺序弹出调用堆栈
            while count > 0:
                mode = _pop_mode()
                count -= 1

            # 将预分发模式添加到临时列表末尾
            temp_elements.append(pre_dispatch_mode)

            # 将临时列表中的模式对象按照相反的顺序重新推入调用堆栈中
            for mode in reversed(temp_elements):
                _push_mode(mode)
@torch._disable_dynamo
def dispatch_trace(
        root: Union[torch.nn.Module, Callable],
        tracer: Tracer,
        concrete_args: Optional[Tuple[Any, ...]] = None,
) -> GraphModule:
    # 使用 Tracer 对象对指定的根对象进行追踪，生成计算图
    graph = tracer.trace(root, concrete_args)
    # 导入函数：从 torch._inductor.fx_passes.dedupe_symint_uses 模块导入 dedupe_symints 函数
    from torch._inductor.fx_passes.dedupe_symint_uses import dedupe_symints
    # 对生成的计算图进行符号整合处理
    dedupe_symints(graph)
    # 获取根对象的名称作为图模块的名称，如果根对象是 torch.nn.Module 类型则使用其类名，否则使用对象的名称
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    # 使用 fx._lazy_graph_module._make_graph_module 函数创建图模块，并返回
    return fx._lazy_graph_module._make_graph_module(tracer.root, graph, name)


def wrap_key(f, tensors, tracer, pre_dispatch: bool):
    # 使用 pytree.tree_flatten 函数将 tensors 扁平化
    flat_tensors, tensors_spec = pytree.tree_flatten(tensors)

    @functools.wraps(f)
    def wrapped(*proxies):
        # 使用 pytree.tree_flatten 函数将 proxies 扁平化
        flat_proxies, proxies_spec = pytree.tree_flatten(proxies)
        # 断言：确保 flat_proxies 和 flat_tensors 的长度相等
        assert len(flat_proxies) == len(flat_tensors)
        # 使用 disable_proxy_modes_tracing 上下文管理器，禁用代理模式的追踪
        with disable_proxy_modes_tracing() as m:
            # 断言：确保 m 是 ProxyTorchDispatchMode 类的实例
            assert isinstance(m, ProxyTorchDispatchMode)
            # 跟踪 tensor 树的结构，传递 flat_tensors 和 flat_proxies，tracer 用于跟踪
            track_tensor_tree(flat_tensors, flat_proxies, constant=None, tracer=tracer)

        # 调用原始函数 f，传入 tensors 参数，并记录返回值到 out
        out = f(*tensors)
        # 使用 pytree.tree_map_only 函数，将 out 中的每个 torch.Tensor 类型的元素映射为其代理槽的值
        out = pytree.tree_map_only(
            torch.Tensor,
            lambda t: get_proxy_slot(t, tracer, t, lambda x: x.proxy),
            out
        )
        # 使用 pytree.tree_map_only 函数，将 out 中的每个 torch.ScriptObject 或 FakeScriptObject 类型的元素映射为其代理槽的值
        out = pytree.tree_map_only(
            (torch.ScriptObject, FakeScriptObject),
            lambda t: get_proxy_slot(t, tracer, t, lambda x: x),
            out
        )
        # 使用 pytree.tree_map_only 函数，将 out 中的每个 SymInt, SymFloat, SymBool 类型的元素映射为其代理槽的值
        out = pytree.tree_map_only(
            (SymInt, SymFloat, SymBool),
            lambda t: get_proxy_slot(t, tracer)(),
            out
        )
        # 返回处理后的 out
        return out

    # 返回 wrapped 函数
    return wrapped


ORIGINAL_ATEN = None
@contextmanager
def set_original_aten_op(func):
    # 定义全局变量 ORIGINAL_ATEN 用于存储原始的 ATen 操作函数
    global ORIGINAL_ATEN
    # 如果 ORIGINAL_ATEN 为空且 fx_traceback 模块保留了节点元数据
    if ORIGINAL_ATEN is None and fx_traceback.has_preserved_node_meta():
        # 将 func 赋值给 ORIGINAL_ATEN
        ORIGINAL_ATEN = func
        # 将 func 存储到 fx_traceback 当前的元数据中
        fx_traceback.current_meta['original_aten'] = func
        try:
            # 执行代码块
            yield
        finally:
            # 代码块执行结束后，将 ORIGINAL_ATEN 置空
            ORIGINAL_ATEN = None
            # 将 fx_traceback 当前的元数据中的 original_aten 置空
            fx_traceback.current_meta['original_aten'] = None
    else:
        # 如果 ORIGINAL_ATEN 不为空或者 fx_traceback 没有保留节点元数据，直接执行代码块
        yield


class TorchFunctionMetadataMode(TorchFunctionMode):

    def __init__(self, tracer):
        # 初始化函数，接受一个 tracer 参数
        self.tracer = tracer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        # 实现 __torch_function__ 方法，用于处理 torch 函数的元数据
        kwargs = kwargs or {}
        # 设置当前追踪器的 torch_fn_metadata 属性为 func
        self.tracer.torch_fn_metadata = func
        # 更新追踪器中 func 的调用次数
        self.tracer.torch_fn_counts[func] = self.tracer.torch_fn_counts.get(func, 0) + 1
        # 调用 func 函数，并返回其结果
        return func(*args, **kwargs)


# This mode is **only** used for pre_dispatch tracing.
# In particular, we need to make sure that autograd/autocast API's
# that do not desugar into dispatcher operators stay in the graph.
class PreDispatchTorchFunctionMode(TorchFunctionMode):

    def __init__(self, tracer):
        # 初始化函数，接受一个 tracer 参数
        self.tracer = tracer
    # 定义一个特殊方法 __torch_function__，用于处理 torch 函数调用的协议
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # 初始化 kwargs，确保不为 None
        kwargs = kwargs or {}
        # 检查 func 是否在需要保留前调度的副作用函数列表中
        if func in _side_effectful_need_to_be_preserved_pre_dispatch:
            # 这段代码用于通过导出验证器验证 meta['val']
            # TODO(tmanlaibaatar): 应该将其系统地与导出验证器耦合，而不是在此处硬编码。
            # 创建一个调用函数节点，并传入相应的函数、参数和空字典
            node = self.tracer.create_node("call_function", func, args, {})
            # 如果 func 是 torch._C._set_grad_enabled，则设置 node.meta['val'] 为 None
            if func is torch._C._set_grad_enabled:
                node.meta['val'] = None
            # 返回创建的节点，表示仅跟踪函数调用，不实际运行函数
            return node
            # 不实际运行函数！我们只是想要追踪调用，而不是改变全局自动求导状态。
        
        # 如果 func 不在需要保留前调度的副作用函数列表中，则实际调用 func，并传入 args 和 kwargs
        return func(*args, **kwargs)
class ProxyTorchDispatchMode(TorchDispatchMode):
    # 定义一个代理的Torch调度模式，继承自TorchDispatchMode

    def __init__(
        self,
        tracer,
        tracing_mode,
        pre_dispatch=False,
        _allow_fake_constant=False,
        _error_on_data_dependent_ops=True
    ):
        # 初始化方法，设置代理Torch调度模式的各种参数
        # tracer: 追踪器对象
        # tracing_mode: 追踪模式
        # pre_dispatch: 是否预调度，默认为False
        # _allow_fake_constant: 是否允许虚假常量，默认为False
        # _error_on_data_dependent_ops: 是否在数据依赖操作上报错，默认为True

        # 根据 pre_dispatch 设置 DispatchKey
        dk = torch._C.DispatchKey.PreDispatch if pre_dispatch else None
        super().__init__(dk)  # 调用父类的初始化方法

        # 设置实例变量
        self.tracer = tracer
        self.tracing_mode = tracing_mode
        self.enable_tracing = True
        self.pre_dispatch = pre_dispatch
        self._allow_fake_constant = _allow_fake_constant
        self._error_on_data_dependent_ops = _error_on_data_dependent_ops
        self.sym_mode = ProxySymDispatchMode(tracer)  # 创建代理符号调度模式对象
        self.trace_state = {}  # 追踪状态字典
        self._managers = []  # 管理器列表

        # 指示这是一个“基础设施”模式，其调度优先级较低
        self._mode_key = torch._C._TorchDispatchModeKey.PROXY

        # 每次进入模式时，我们维护一个堆栈，告诉我们先前的ProxyTorchDispatchMode状态是什么（如果有的话）
        # 这样可以在退出时正确重置状态
        self.enter_stack: List[Optional[ProxyTorchDispatchMode]] = []

    @count
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Torch分发函数，用于分发函数调用到正确的代理处理函数
        with self.sym_mode.enable(False), set_original_aten_op(func):
            return self.inner_torch_dispatch(func, types, args, kwargs)

    def __enter__(self):
        # 进入上下文管理器时的操作
        # 首先启用符号模式，然后我们...
        m = self.sym_mode.enable(True)
        self._managers.append(m)  # 将符号模式管理器添加到列表中
        m.__enter__()  # 进入符号模式上下文

        # 存储并记录先前的代理模式（可能有也可能没有）
        maybe_prev_proxy_mode = _unset_infra_mode(torch._C._TorchDispatchModeKey.PROXY)
        self.enter_stack.append(maybe_prev_proxy_mode)  # 将先前的代理模式存入堆栈
        return super().__enter__()  # 调用父类的进入方法

    def __exit__(self, exc_type, exc_value, traceback):
        # 退出上下文管理器时的操作
        m = self._managers.pop()  # 弹出符号模式管理器
        # 首先退出我们自己，然后退出符号模式
        b = super().__exit__(exc_type, exc_value, traceback)

        # 重新启用先前的代理模式，如果有的话
        mb_previous_proxy_mode = self.enter_stack.pop()
        if mb_previous_proxy_mode is not None:
            _push_mode(mb_previous_proxy_mode)

        # 如果退出时没有异常，返回符号模式的退出结果；否则返回符号模式的异常处理结果
        if not b:
            return m.__exit__(exc_type, exc_value, traceback)
        else:
            return m.__exit__(None, None, None)

    def inner_torch_dispatch(self, func, types, args=(), kwargs=None):
        # 内部的Torch分发函数，根据当前模式调用相应的代理函数处理
        if not self.enable_tracing:
            return func(*args, **kwargs)

        # 如果函数是在例外之中的默认设备，直接调用函数并返回结果
        if func in [prim.device.default]:
            return func(*args, **kwargs)

        # 否则使用代理调用函数进行处理
        return proxy_call(self, func, self.pre_dispatch, args, kwargs)


class ProxySymDispatchMode(SymDispatchMode):
    # 代理符号调度模式，继承自SymDispatchMode
    def __init__(self, tracer):
        super().__init__()
        self.tracer = tracer
        # 当 enable_tracing 为 False 时，不进行追踪操作。
        # 如果使用这个选项，必须在操作的所有结果上调用 track_tensor/track_tensor_tree，
        # 以确保我们能够充分追踪结果。
        self.enable_tracing = True

    @contextmanager
    def enable(self, b):
        old = self.enable_tracing
        self.enable_tracing = b
        try:
            yield
        finally:
            self.enable_tracing = old

    def _compute_proxy(self, func, args, out: Union[SymInt, SymFloat, SymBool]):
        n_args = tuple(
            # 如果参数是符号类型，则获取其代理槽的节点；否则直接使用参数本身。
            get_proxy_slot(a, self.tracer)().node if isinstance(a, py_sym_types) else a
            for a in args
        )

        # func 没有 __torch_function__ 可以由代理层插入，
        # 因此我们必须手动执行这个操作。
        n_out = self.tracer.create_node("call_function", func, n_args, {})
        # 创建代理对象，封装节点 n_out
        p_out = fx.Proxy(n_out, self.tracer)
        set_meta(p_out, out)
        return p_out

    def __sym_dispatch__(self, func, types, args, kwargs):
        if not self.enable_tracing:
            return func(*args, **kwargs)

        # 优化乘以 1 的情况
        # 注意：这里要小心不要触发保护条件！
        if func == operator.mul:
            if isinstance(args[1], int) and args[1] == 1:
                return args[0]
            elif isinstance(args[0], int) and args[0] == 1:
                return args[1]

        # 为了速度，我们假设没有嵌套的数据结构
        # （否则我们可以使用 tree_map）
        # 我们还假设没有关键字参数。
        assert not kwargs
        # 执行 func 函数并获取输出
        out = func(*args, **kwargs)

        # 如果 func 返回一个常量，我们不需要追踪；
        # 我们已经确定结果是常量（无论输入是否为符号），不再需要追踪计算过程。
        # 这可能发生在 func 触发某些保护条件时。
        if isinstance(out, py_sym_types):
            # 延迟追踪此操作的代理对象，直到我们真正需要它
            p_out_thunk = thunkify(self._compute_proxy, func=func, args=args, out=out)
            set_proxy_slot(out, self.tracer, p_out_thunk)

        return out
# TODO: I'm not sure what the point of this class is; you can just
# make_fx through a regular Interpreter
# 定义一个自定义的解析器类，继承自 torch.fx.Interpreter 类
class DecompositionInterpreter(torch.fx.Interpreter):
    def __init__(self, module: torch.fx.GraphModule, new_graph: torch.fx.Graph, decomposition_table=None, **kwargs):
        # 调用父类的构造函数进行初始化
        super().__init__(module, **kwargs)
        # 保存新的图形对象和一个用于追踪的实例
        self.new_graph = new_graph
        self.tracer = torch.fx.proxy.GraphAppendingTracer(self.new_graph)
        # 为追踪器设置张量追踪器和符号节点追踪器，使用弱引用字典
        self.tracer.tensor_tracker = WeakTensorKeyDictionary()  # type: ignore[attr-defined]
        self.tracer.symnode_tracker = weakref.WeakKeyDictionary()  # type: ignore[attr-defined]
        # 如果没有提供分解表，则初始化为空字典
        self.decomposition_table = decomposition_table
        if self.decomposition_table is None:
            self.decomposition_table = {}
        # 创建代理的 Torch 调度模式实例，用于追踪真实模式
        self.mode = ProxyTorchDispatchMode(self.tracer, tracing_mode="real")

        # 存储在追踪过程中调用的 Torch 函数的元数据
        self.tracer.torch_fn_metadata = None
        # 存储每个调用的 Torch 函数的计数，以区分对同一 Torch 函数的不同调用
        self.tracer.torch_fn_counts = {}

    # 用于处理占位符的方法
    def placeholder(self, target, args, kwargs):
        out = super().placeholder(target, args, kwargs)
        # 在新图中创建占位符的代理对象，并追踪张量树结构
        proxy = torch.fx.Proxy(self.new_graph.placeholder(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        # TODO 处理目标的第一个字符是 '*' 的情况
        return out

    # 用于获取属性的方法
    def get_attr(self, target, args, kwargs):
        out = super().get_attr(target, args, kwargs)
        # 在新图中获取属性的代理对象，并追踪张量树结构
        proxy = torch.fx.Proxy(self.new_graph.get_attr(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    # call_function, call_method, call_module 方法会自动被外部模式追踪

    # 输出方法，用于处理输出节点
    def output(self, target, args, kwargs):
        out = super().output(target, args, kwargs)

        # 定义一个函数，用于展开代理对象
        def unwrap(e):
            return get_proxy_slot(e, self.tracer, e, lambda x: x.proxy.node)
        # 在新图中输出解开代理后的节点
        self.new_graph.output(pytree.tree_map(unwrap, out))
        return out

    # 运行方法，进入至少一次模式以便稍后恢复
    def run(self, *args, **kwargs):
        # 使用分解和模式上下文管理器来运行超类的运行方法
        with decompose(self.decomposition_table), self.mode:
            return super().run(*args, **kwargs)


# 函数，用于封装 make_fx 的参数
def wrapper_and_args_for_make_fx(func, args, kwargs):
    # make_fx 不支持 kwargs，因此需要展平参数，并在调用 func 前解开
    flat_args, spec = pytree.tree_flatten((args, kwargs))

    def wrapped(flat_args):
        fn_args, fn_kwargs = pytree.tree_unflatten(flat_args, spec)
        return func(*fn_args, **fn_kwargs)
    return wrapped, flat_args


# 上下文管理器，用于禁用自动转换缓存
@contextmanager
def disable_autocast_cache():
    old_value = torch.is_autocast_cache_enabled()
    # 禁用 PyTorch 自动混合精度（Autocast）的缓存
    torch.set_autocast_cache_enabled(False)
    # 尝试执行代码块
    try:
        # 使用 yield 关键字使函数可以作为生成器函数调用
        yield
    # 无论 try 块是否成功执行，最终都会执行以下代码块
    finally:
        # 恢复 PyTorch 自动混合精度的缓存设置为先前的值
        torch.set_autocast_cache_enabled(old_value)
# 定义一个自定义异常类，继承自 NameError，用于表示模块未作为子模块安装的错误
class _ModuleNotInstalledAsSubmoduleError(NameError):
    pass

# 定义一个继承自 PythonKeyTracer 的定制版本 _ModuleStackTracer 类，
# 用于在节点的 meta["nn_module_stack"] 中保留模块堆栈信息。
class _ModuleStackTracer(PythonKeyTracer):
    r"""
    定制版的 PythonKeyTracer，用于在 node.meta["nn_module_stack"] 中保留模块堆栈信息。

    FX 符号跟踪已经实现了这一点，但它依赖于 `self.root` 是实际跟踪的模块。
    由于 make_fx 跟踪了我们创建的 lambda，事情不正常工作。

    因此，对于这个版本，我们保持对原始模块（scope_root）的引用，并使用它来匹配路径。
    当我们看到像下面这样的结构时，
            A
           / \
          B   C
           \ /
            D
    我们希望记录路径为 A.B.D，只记录一条路径。
    参见注释 [在非严格模式下导出时保留 nn 模块堆栈元数据]  # noqa: W605
    """

    def __init__(self, scope_root):
        super().__init__()
        self.scope_root = scope_root  # 设置对象的根模块引用
        self.proxy_paths = WeakKeyDictionary()  # 弱引用字典，用于存储代理对象的路径
        self.proxy_modules = WeakKeyDictionary()  # 弱引用字典，用于存储代理对象对应的原始模块
        self.counter = 0  # 计数器，用于跟踪代理对象数量

        self.module_id_cache = defaultdict(list)  # 默认字典，用于缓存模块的 ID 和名称列表
        # 遍历 scope_root 的命名模块，将每个模块的 ID 和名称添加到缓存中（允许重复名称）
        for name, mod in self.scope_root.named_modules(remove_duplicate=False):
            self.module_id_cache[id(mod)].append(name)

        self_ = self

        # 定义一个属性代理类 AttrProxy
        class AttrProxy:
            def __init__(self, base, path):
                # 将当前类的 __class__ 设置为与 base.__class__ 相同的类
                self.__class__ = type(
                    base.__class__.__name__,
                    (self.__class__, base.__class__),
                    {},
                )
                self.__dict__ = base.__dict__  # 复制 base 的 __dict__ 到当前对象
                self.__class__.__module__ = base.__class__.__module__  # 设置模块名
                self.__class__.__qualname__ = base.__class__.__qualname__  # 设置完全限定名称
                self_.proxy_paths[self] = path  # 记录代理对象的路径
                self_.proxy_modules[self] = base  # 记录代理对象对应的原始模块

            def __getattr__(self, name):
                assert isinstance(self, torch.nn.Module)  # 断言 self 是 torch.nn.Module 的实例
                attr_val = super().__getattr__(name)  # 获取属性值
                if isinstance(attr_val, AttrProxy):
                    attr_val = self_.proxy_modules[attr_val]  # 如果属性是 AttrProxy，则获取其原始模块
                elif not isinstance(attr_val, torch.nn.Module):
                    return attr_val  # 如果属性不是 Module 类型，则直接返回其值
                # 返回新的 AttrProxy 对象，路径添加当前属性名
                return AttrProxy(attr_val, self_.proxy_paths[self] + "." + name)

            @property
            def _modules(self):
                assert "_modules" in self.__dict__  # 断言当前对象有 "_modules" 属性
                submodules = self.__dict__["_modules"]  # 获取子模块字典
                assert isinstance(submodules, dict)  # 断言子模块是字典类型
                # 返回一个新的字典，其中键是子模块的名称，值是对应的 AttrProxy 对象
                return {
                    key: AttrProxy(value, self_.proxy_paths[self] + "." + str(key))
                    for key, value in submodules.items()
                }

        self.proxy_type = AttrProxy  # 设置类的代理类型为 AttrProxy
    def path_of_module(self, mod: torch.nn.Module) -> str:
        """
        根据模块追踪期间的访问路径获取模块的路径，而不是使用默认的广度优先搜索行为。
        仍然使用所有可能的模块路径来验证结果。
        """
        if mod is self.scope_root:
            return ""

        if isinstance(mod, self.proxy_type):
            # 如果模块是代理类型，则返回其代理路径
            return self.proxy_paths[mod]

        try:
            # 尝试使用 Tracer 类中定义的方法获取模块路径
            return Tracer.path_of_module(self, mod)
        except NameError as e:
            # 如果模块未安装为子模块，则引发异常
            raise _ModuleNotInstalledAsSubmoduleError from e

    def getattr(self, attr, attr_val, parameter_proxy_cache):
        if not isinstance(attr_val, torch.nn.Module) or isinstance(attr_val, torch.fx.GraphModule):
            # 如果属性值不是 torch.nn.Module 或者是 torch.fx.GraphModule 类型，则调用父类的 getattr 方法
            return super().getattr(attr, attr_val, parameter_proxy_cache)
        if isinstance(attr_val, self.proxy_type):
            # 如果属性值是代理类型，则直接返回属性值
            return attr_val
        # 否则，返回代理类型的实例化对象
        return self.proxy_type(attr_val, attr)

    def trace(self, root, concrete_args):
        res = super().trace(root, concrete_args)
        # 由于 AttrProxy 需要模仿原始子模块，在跟踪过程中，如果有人直接注册了一个模块到追踪器中，
        # 代理对象会首先注册。因此，我们需要用真实模块替换代理模块。
        proxy_module_names_to_be_replaced = []
        for name, module in self.root.named_modules():
            if module in self.proxy_modules:
                proxy_module_names_to_be_replaced.append((name, module))

        def _delete_proxy_attr(obj, target):
            # 从 fx/graph_module.py 复制过来，并为代理类型进行定制
            atoms = target.split(".")
            path, target_submod = atoms[:-1], atoms[-1]
            assert isinstance(obj, torch.nn.Module)
            mod = obj

            # 获取父模块
            for item in path:
                if not hasattr(mod, item):
                    return False
                mod = getattr(mod, item)

                if not isinstance(mod, (self.proxy_type, torch.nn.Module)):
                    return False

            if not hasattr(mod, target_submod):
                return False

            # 至少叶子模块应为代理类型
            if not isinstance(getattr(mod, target_submod), self.proxy_type):
                return False

            # 删除属性
            delattr(mod, target_submod)
            return True

        # 替换代理模块
        for (proxy_module_name, proxy_module) in proxy_module_names_to_be_replaced:
            _delete_proxy_attr(self.root, proxy_module_name)
            actual_module = self.proxy_modules[proxy_module]
            _assign_attr(actual_module, self.root, proxy_module_name)

        return res
    # 覆盖了 call_module 方法，但我们实际上需要它来处理作用域
    def call_module(self, m, forward, args, kwargs):
        """PythonKeyTracer overrides call_module to avoid the scope handling,
        but we actually want it.
        """
        # 导入优化模块和图形模块
        from torch._dynamo import OptimizedModule
        # FIXME (tmanlaibaatar)
        # 当我们在 HOO 内部调用 torch.compile 时，会调用到未在根上注册的模块。
        # 目前我们只是内联它们。但一旦我们开始支持导出中的 mark_strict，我们需要正确处理这一点。
        # 目前它不重要，因为当前的非严格使用情况不需要与 HOO 一起工作。
        if isinstance(m, (OptimizedModule, GraphModule)):
            return forward(*args, **kwargs)

        try:
            # 尝试调用 Tracer 的 call_module 方法
            return Tracer.call_module(self, m, forward, args, kwargs)
        except _ModuleNotInstalledAsSubmoduleError as e:
            # 如果模块未作为子模块安装，则发出警告
            warnings.warn(
                f"Unable to find the path of the module {m}. "
                "This might be because the module was not properly registered "
                "as a submodule, which is not good practice. We will trace "
                "through the module without recording stack information."
            )
            return forward(*args, **kwargs)


    # 始终返回 False，表示给定的模块不是叶子模块
    def is_leaf_module(self, m, module_qualified_name):
        return False
    '''
    Create node and add on metadata.
    Add nn_module_stack here instead of TracerBase,
    since calls to make_fx() might not want to record module stack metadata.
    Add torch_fn by looking at torch_fn_metadata and torch_fn_counts.
    Add stack_trace by filtering out forward() stack frames.
    '''
    # 调用父类方法创建节点，并返回节点对象
    node = super().create_node(*args, **kwargs)

    # nn_module_stack
    # 如果节点操作不是"placeholder"或"output"
    if node.op not in ["placeholder", "output"]:
        # 如果节点的元数据中没有"nn_module_stack"字段
        if "nn_module_stack" not in node.meta:
            # 将当前对象的模块堆栈赋给节点的"nn_module_stack"字段
            node.meta["nn_module_stack"] = self.module_stack
        # 将"nn_module_stack"的值从Dict[key, (FQN, class)]转换为Dict[str, Tuple[str, str]]
        for key, (fqn, mod_cls) in node.meta["nn_module_stack"].items():
            if isinstance(mod_cls, type):
                # 更新节点的"nn_module_stack"，使用模块的完全限定名和类名
                node.meta["nn_module_stack"][key] = (fqn, mod_cls.__module__ + "." + mod_cls.__qualname__)

    # torch_fn
    # 如果节点操作是"call_function"，且存在torch_fn_metadata和节点的元数据中没有"torch_fn"
    if node.op == "call_function" and self.torch_fn_metadata is not None and "torch_fn" not in node.meta:
        # 添加torch_fn元数据，使用torch_fn_metadata的名称和计数器中的值
        node.meta["torch_fn"] = (
            f"{self.torch_fn_metadata.__name__}_{self.torch_fn_counts[self.torch_fn_metadata]}",
            f"{self.torch_fn_metadata.__class__.__name__}.{self.torch_fn_metadata.__name__}"
        )

    # stack_trace
    # 如果节点的元数据中没有"stack_trace"字段，并且节点操作不是"placeholder"或"output"
    if 'stack_trace' not in node.meta and node.op not in ["placeholder", "output"]:
        # 提取用户代码的堆栈摘要
        user_frame_summary = CapturedTraceback.extract().summary()
        if user_frame_summary:
            # 保留来自"forward"调用或位于torch/__init__.py中的操作的堆栈帧
            stack_trace = [frame for frame in user_frame_summary if (
                frame.name == 'forward'
                or frame.filename.endswith('torch/__init__.py')
            )]
            # 过滤掉来自fx/_symbolic_trace.py和export/_trace.py中的forward()帧
            stack_trace = [
                frame for frame in stack_trace if not (
                    frame.filename.endswith('fx/_symbolic_trace.py')
                    or frame.filename.endswith('export/_trace.py')
                )
            ]
            if stack_trace:  # 对于严格模式，空列表应该由dynamo处理stack_trace
                # 格式化堆栈摘要并将其作为字符串存储在节点的"stack_trace"中
                stack_trace = traceback.StackSummary.from_list(stack_trace)
                node.meta["stack_trace"] = ''.join(stack_trace.format()).strip()

    # 返回处理后的节点对象
    return node
    # 定义 _MakefxTracer 类，用于实现特定功能的追踪器

    def __init__(
        self,
        decomposition_table: Optional[Dict[Callable, Callable]],  # 初始化分解表，可选参数，字典类型，键和值均为可调用对象
        tracing_mode: str,  # 追踪模式，字符串类型，指定当前追踪的模式
        _allow_non_fake_inputs: bool,  # 是否允许非虚假输入，布尔类型，控制是否接受非虚假输入
        pre_dispatch: bool,  # 预调度，布尔类型，控制是否进行预调度
        record_module_stack: bool,  # 记录模块堆栈，布尔类型，控制是否记录模块的调用堆栈
        _allow_fake_constant: bool,  # 是否允许虚假常量，布尔类型，控制是否允许虚假常量
        _error_on_data_dependent_ops: bool  # 数据相关操作的错误，布尔类型，控制是否在数据相关操作时报错
    ):
        # 初始化分解表，如果为空则设为一个空字典，并设置默认的对称元素数函数
        self.decomposition_table: Dict[Callable, Callable] = decomposition_table or {}
        self.decomposition_table.setdefault(torch.ops.aten.sym_numel.default, torch._decomp.decompositions.sym_numel)
        
        self.tracing_mode: str = tracing_mode  # 设置追踪模式
        self._allow_non_fake_inputs: bool = _allow_non_fake_inputs  # 设置是否允许非虚假输入
        self.pre_dispatch: bool = pre_dispatch  # 设置是否进行预调度
        self.record_module_stack: bool = record_module_stack  # 设置是否记录模块堆栈
        self._allow_fake_constant: bool = _allow_fake_constant  # 设置是否允许虚假常量
        self._error_on_data_dependent_ops: bool = _error_on_data_dependent_ops  # 设置数据相关操作是否报错

        # 初始化各种上下文管理器及其状态，初始状态为 nullcontext()，即无操作的上下文管理器
        self.fake_tensor_mode: Union[null_ctx_type, FakeTensorMode] = nullcontext()
        self.proxy_mode: Union[null_ctx_type, ProxyTorchDispatchMode] = nullcontext()
        self.proxy_function_mode: Union[null_ctx_type, PreDispatchTorchFunctionMode] = nullcontext()
        self.fx_tracer: Union[null_ctx_type, Tracer] = nullcontext()
        self.python_dispatcher_mode: Union[null_ctx_type, Any] = nullcontext()
        self.torch_fn_metadata_mode: Union[null_ctx_type, TorchFunctionMetadataMode] = nullcontext()

    def _checkpoint_modes(self) -> List[Any]:
        # 返回当前各种模式的状态列表，用于检查点
        return [
            self.fake_tensor_mode,
            self.proxy_mode,
            self.proxy_function_mode,
            self.fx_tracer,
            self.python_dispatcher_mode,
            self.torch_fn_metadata_mode
        ]

    def _restore_modes(
        self,
        prev_fake_tensor_mode: Union[null_ctx_type, FakeTensorMode],
        prev_proxy_mode: Union[null_ctx_type, ProxyTorchDispatchMode],
        prev_proxy_function_mode: Union[null_ctx_type, PreDispatchTorchFunctionMode],
        prev_fx_tracer: Union[null_ctx_type, Tracer],
        prev_python_dispatcher_mode: Union[null_ctx_type, Any],
        prev_torch_fn_metadata_mode : Union[null_ctx_type, TorchFunctionMetadataMode],
    ):
        # 恢复各种模式的状态，接受上一个状态作为参数进行恢复
        pass  # 此处应有具体的恢复逻辑，暂未提供
    ) -> None:
        # 从输入参数中恢复先前保存的模式设置
        self.fake_tensor_mode = prev_fake_tensor_mode
        self.proxy_mode = prev_proxy_mode
        self.proxy_function_mode = prev_proxy_function_mode
        self.fx_tracer = prev_fx_tracer
        self.python_dispatcher_mode = prev_python_dispatcher_mode
        self.torch_fn_metadata_mode = prev_torch_fn_metadata_mode

    @contextmanager
    def _init_modes_from_inputs(self, f, args):
        # 保存当前的模式设置
        prev_modes = self._checkpoint_modes()
        try:
            # 避免在模块级别导入 sympy
            from .symbolic_shapes import ShapeEnv
            # 如果函数 f 有 "_orig_mod" 属性并且记录模块堆栈，则使用 _ModuleStackTracer，否则使用 PythonKeyTracer
            if hasattr(f, "_orig_mod") and self.record_module_stack:
                scope_root = f._orig_mod
                self.fx_tracer = _ModuleStackTracer(scope_root)
            else:
                self.fx_tracer = PythonKeyTracer()

            # 根据追踪模式设置 fake_tensor_mode
            if self.tracing_mode == "fake":
                import torch._dynamo
                # 检测假张量模式
                fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
                if fake_tensor_mode is None:
                    import torch._functorch.config as _config
                    # 使用 _config.patch 设置 fake_tensor_allow_unsafe_data_ptr_access=False
                    with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                        # 创建 FakeTensorMode 对象
                        fake_tensor_mode = FakeTensorMode(
                            allow_fallback_kernels=True,
                            allow_non_fake_inputs=self._allow_non_fake_inputs,
                            shape_env=ShapeEnv(),
                            static_shapes=True,
                        )
                self.fake_tensor_mode = fake_tensor_mode
            elif self.tracing_mode == "symbolic":
                import torch._dynamo
                # 检测假张量模式
                fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
                if fake_tensor_mode is None:
                    shape_env = ShapeEnv()
                    import torch._functorch.config as _config
                    # 使用 _config.patch 设置 fake_tensor_allow_unsafe_data_ptr_access=False
                    with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                        # 创建 FakeTensorMode 对象
                        fake_tensor_mode = FakeTensorMode(
                            allow_fallback_kernels=False,
                            allow_non_fake_inputs=self._allow_non_fake_inputs,
                            shape_env=shape_env)
                # 断言确保在符号追踪模式下 shape_env 不为 None
                assert fake_tensor_mode.shape_env is not None, "shape_env should be set if tracing with 'symbolic'"
                self.fake_tensor_mode = fake_tensor_mode
            else:
                # 如果追踪模式不是 "fake" 或 "symbolic"，则抛出异常
                if not self.tracing_mode == "real":
                    raise AssertionError(f"Unexpected tracing type: {self.tracing_mode}")

            # 使用当前的 fx_tracer 构建模式
            self._construct_modes_with_fx_tracer(self.fx_tracer)
            # 执行 yield，让外部代码执行
            yield
        finally:
            # 恢复之前保存的模式设置
            self._restore_modes(*prev_modes)
    #`
    # 定义一个方法，用于构建带有函数追踪器的模式
    def _construct_modes_with_fx_tracer(self, fx_tracer):
        # 设置代理模式，用于 Torch 分发
        self.proxy_mode = ProxyTorchDispatchMode(
            fx_tracer,
            self.tracing_mode,
            pre_dispatch=self.pre_dispatch,
            _allow_fake_constant=self._allow_fake_constant,
            _error_on_data_dependent_ops=self._error_on_data_dependent_ops
        )

        # 如果启用了预分发模式，设置预分发 Torch 函数模式
        if self.pre_dispatch:
            self.proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)

        # 如果追踪模式为符号化或者启用了预分发，启用 Python 分发器模式
        if self.tracing_mode == "symbolic" or self.pre_dispatch:
            self.python_dispatcher_mode = enable_python_dispatcher()

        # 设置 Torch 函数元数据模式
        self.torch_fn_metadata_mode = TorchFunctionMetadataMode(fx_tracer)

    # 使用上下文管理器初始化来自父追踪器的模式
    @contextmanager
    def _init_modes_from_parent(self, parent_tracer):
        # 默认情况下，子追踪器根据父追踪器的配置创建新模式。
        # 但是有些情况下，我们希望与父追踪器共享相同的模式，
        # 例如，fake_tensor_mode，我们希望父图和子图的示例值的 fake_mode 相同。
        prev_modes = self._checkpoint_modes()
        try:
            # 设置 fake_tensor_mode 为父追踪器的 fake_tensor_mode
            self.fake_tensor_mode = parent_tracer.fake_tensor_mode

            # 创建子追踪器
            def _create_sub_fx_tracer(parent_tracer):
                if type(parent_tracer) == PythonKeyTracer:
                    sub_tracer = PythonKeyTracer()
                elif type(parent_tracer) == _ModuleStackTracer:
                    sub_tracer = _ModuleStackTracer(parent_tracer.scope_root)
                else:
                    raise RuntimeError(f"Unexpected tracer type: {type(parent_tracer)}.")

                return sub_tracer

            # 使用父追踪器的 fx_tracer 创建子追踪器的 fx_tracer，并构建模式
            self.fx_tracer = _create_sub_fx_tracer(parent_tracer.fx_tracer)
            self._construct_modes_with_fx_tracer(self.fx_tracer)
            # 返回 yield，用于执行 trace_subgraph 方法中的代码块
            yield
        finally:
            # 恢复先前的模式状态
            self._restore_modes(*prev_modes)

    # 对给定函数和参数进行追踪，返回 Torch 图模块
    def trace(self, f, *args) -> torch.fx.GraphModule      self._allow_non_fake_inputs,
        self.pre_dispatch,
        self.record_module_stack,
        self._allow_fake_constant,
        self._error_on_data_dependent_ops
    )
    # 使用父追踪器的配置初始化子图追踪器的模式
    with sub_tracer._init_modes_from_parent(self):
        # 执行子图追踪操作
        return sub_tracer._trace_inner(f, *args)
# 当前的 _MakefxTracer 类型的可选变量，初始值为 None
_CURRENT_MAKE_FX_TRACER : Optional[_MakefxTracer] = None

# 上下文管理器函数，用于设置当前的 make_fx 跟踪器
@contextmanager
def _set_make_fx_tracer(tracer: _MakefxTracer) -> None:
    global _CURRENT_MAKE_FX_TRACER
    # 保存当前的跟踪器
    prev_tracer = _CURRENT_MAKE_FX_TRACER
    try:
        # 设置新的 make_fx 跟踪器为传入的跟踪器
        _CURRENT_MAKE_FX_TRACER = tracer
        # 执行代码块
        yield
    finally:
        # 恢复之前保存的跟踪器
        _CURRENT_MAKE_FX_TRACER = prev_tracer

# 创建 make_fx 函数，用于函数的 FX (效果) 创建
def make_fx(
        f,
        decomposition_table=None,
        tracing_mode="real",
        _allow_non_fake_inputs=False,
        *,
        pre_dispatch=False,
        record_module_stack=False,
        _allow_fake_constant=False,
        _error_on_data_dependent_ops=True):

    # 断言跟踪模式在合法值列表中
    assert tracing_mode in ["real", "fake", "symbolic"]

    # 创建 _MakefxTracer 实例，用于跟踪 FX 创建过程
    make_fx_tracer = _MakefxTracer(
        decomposition_table,
        tracing_mode,
        _allow_non_fake_inputs,
        pre_dispatch,
        record_module_stack,
        _allow_fake_constant,
        _error_on_data_dependent_ops
    )

    # 使用 functools.wraps 装饰器封装原始函数 f
    @functools.wraps(f)
    def wrapped(*args):
        # 使用 make_fx_tracer 跟踪器对函数 f 进行跟踪
        return make_fx_tracer.trace(f, *args)

    # 返回封装后的函数 wrapped
    return wrapped

# 获取当前的 torch 分发模式堆栈
def get_torch_dispatch_modes():
    return torch.utils._python_dispatch._get_current_dispatch_mode_stack()

# 获取最内层的代理模式
def get_innermost_proxy_mode():
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)

# 上下文管理器函数，用于禁用代理模式的跟踪
@contextlib.contextmanager
def disable_proxy_modes_tracing():
    return _disable_infra_mode(torch._C._TorchDispatchModeKey.PROXY)

# 处理可能的分解操作
def maybe_handle_decomp(proxy_mode, op, args, kwargs):
    # 如果操作在当前的 CURRENT_DECOMPOSITION_TABLE 中
    if op in CURRENT_DECOMPOSITION_TABLE:
        # 使用代理模式上下文执行分解操作
        with proxy_mode:
            return CURRENT_DECOMPOSITION_TABLE[op](*args, **kwargs)
    return NotImplemented

# 获取给定函数的孤立图模块
def get_isolated_graphmodule(func, args, kwargs, tracing_mode="real"):
    """A helper function used to get the GraphModule for the given func.

    It's expected to be used in the ProxyTensor tracing context.
    It detaches the args and kwargs from the current tracer so that the trace of
    the current graph module can be created without any side-effects.
    """
    # 使用 wrapper_and_args_for_make_fx 函数获取包装函数及其参数
    wrapped, all_args = wrapper_and_args_for_make_fx(func, args, kwargs)

    # 禁用代理模式的跟踪上下文
    with disable_proxy_modes_tracing():
        # 使用 make_fx 创建跟踪模式为 tracing_mode 的 GraphModule
        gm = make_fx(wrapped, tracing_mode=tracing_mode)(all_args)
    return gm

# 设置无后端绑定的辅助函数，用于在目标 FX 图中设置未支持的绑定
def _set_unbacked_bindings(out, out_proxy):
    """A helper function for setting up unbacked_bindings on the destination FX graph."""
    from .symbolic_shapes import compute_unbacked_bindings

    # 无法使用 detect_fake_mode
    #
    # python test/distributed/_tensor/test_dtensor_compile.py -k
    # test_tp_compile_fullgraph_is_seq_parallel_False
    #
    # 将失败。非常奇怪，他们可能不应该在那里同时使用两个假模式...
    fake_mode = torch._C._get_dispatch_mode(
        torch._C._TorchDispatchModeKey.FAKE
    )
    # 如果 fake_mode 存在且具有 shape_env 属性
    if fake_mode and fake_mode.shape_env:
        # 计算未支持的绑定并将其添加到 out_proxy 节点的元数据中
        if symbol_to_path := compute_unbacked_bindings(fake_mode.shape_env, out):
            out_proxy.node.meta["unbacked_bindings"] = symbol_to_path
```