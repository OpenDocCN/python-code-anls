# `.\pytorch\torch\_dynamo\decorators.py`

```py
# mypy: allow-untyped-defs
# ruff: noqa: TCH004
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 导入 TYPE_CHECKING 用于类型检查
from typing import TYPE_CHECKING

# 导入 torch 库
import torch
# 导入 is_traceable_wrapper_subclass 函数
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
# 从当前包中导入 trace_rules 和 variables 模块
from . import trace_rules, variables
# 从 comptime 模块中导入 comptime 函数
from .comptime import comptime
# 从 eval_frame 模块中导入 DisableContext, innermost_fn, RunOnlyContext 类和 IncorrectUsage 异常类
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
# 从 exc 模块中导入 IncorrectUsage 异常类
from .exc import IncorrectUsage
# 从 external_utils 模块中导入 is_compiling 函数
from .external_utils import is_compiling

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从 torch._C._dynamo.eval_frame 模块中导入 reset_code, set_eval_frame, set_guard_error_hook, skip_code, unsupported 函数
    from torch._C._dynamo.eval_frame import (
        reset_code,
        set_eval_frame,
        set_guard_error_hook,
        skip_code,
        unsupported,
    )
# 否则
else:
    # 遍历 torch._C._dynamo.eval_frame 模块中的所有名称
    for name in dir(torch._C._dynamo.eval_frame):
        # 如果名称以双下划线开头，则跳过
        if name.startswith("__"):
            continue
        # 将模块中的函数或变量添加到全局命名空间中
        globals()[name] = getattr(torch._C._dynamo.eval_frame, name)


# 定义 run 函数，接受一个可选参数 fn
def run(fn=None):
    """Don't do any dynamic compiles, just use prior optimizations"""
    # 如果 fn 参数不为 None
    if fn is not None:
        # 获取 fn 的最内层函数
        fn = innermost_fn(fn)
        # 确保 fn 是可调用的
        assert callable(fn)
        # 返回 RunOnlyContext 的调用结果
        return RunOnlyContext()(fn)
    # 如果 fn 参数为 None，则返回 RunOnlyContext 的实例
    return RunOnlyContext()


# 定义 disable 函数，接受一个可选参数 fn 和一个递归参数 recursive，默认为 True
def disable(fn=None, recursive=True):
    """
    Decorator and context manager to disable TorchDynamo

    If recursive=True, Dynamo is completely skipped on the decorated function
    frame as well as the recursively invoked functions.

    If recursive=False, Dynamo skips frames associated with the function code,
    but still process recursively invoked frames.
    """
    # 如果 recursive 参数为 True
    if recursive:
        # 如果 fn 参数不为 None
        if fn is not None:
            # 获取 fn 的最内层函数
            fn = innermost_fn(fn)
            # 确保 fn 是可调用的
            assert callable(fn)
            # 返回 DisableContext 的调用结果
            return DisableContext()(fn)
        # 如果 fn 参数为 None，则返回 DisableContext 的实例
        return DisableContext()
    # 如果 recursive 参数为 False，则返回 skip 函数的调用结果
    else:
        return skip(fn)


# 定义 skip 函数，接受一个可选参数 fn
def skip(fn=None):
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """
    # 如果 fn 参数为 None，则返回 skip 函数自身
    if fn is None:
        return skip
    # 获取 fn 的最内层函数
    fn = innermost_fn(fn)
    # 确保 fn 是可调用的
    assert callable(fn)
    # 跳过 fn 函数相关的帧
    skip_code(fn.__code__)
    # 将 fn 的 _torchdynamo_disable 属性设置为 True
    fn._torchdynamo_disable = True
    # 返回 fn 函数
    return fn


# 定义 assume_constant_result 函数，接受一个 fn 参数
def assume_constant_result(fn):
    # 将 fn 的 _dynamo_marked_constant 属性设置为 True
    fn._dynamo_marked_constant = True
    # 返回 fn 函数
    return fn


# 定义 allow_in_graph 函数，接受一个 fn 参数
def allow_in_graph(fn):
    """
    Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
    and instead directly write it to the graph when encountered.

    See :func:`torch.compiler.allow_in_graph`'s docstring for the full documentation

    WARNING: this API can be a footgun, please read the documentation carefully.
    """
    # 如果 fn 是列表或元组类型
    if isinstance(fn, (list, tuple)):
        # 对列表或元组中的每个元素递归调用 allow_in_graph 函数
        return [allow_in_graph(x) for x in fn]
    # 断言 fn 是可调用的函数
    assert callable(fn), "allow_in_graph expects a callable"
    # 如果 trace_rules.lookup_callable(fn) 不是 variables.TorchInGraphFunctionVariable 类型
    if trace_rules.lookup_callable(fn) != variables.TorchInGraphFunctionVariable:
        # 从 _disallowed_callable_ids 集合中移除 fn 的 ID，并将其添加到 _allowed_callable_ids 集合中
        trace_rules._disallowed_callable_ids.remove(id(fn))
        trace_rules._allowed_callable_ids.add(id(fn))
    # 返回 fn 函数
    return fn


# 定义 _disallow_in_graph_helper 函数，接受一个 throw_if_not_allowed 参数
def _disallow_in_graph_helper(throw_if_not_allowed):
    # 定义一个名为 inner 的函数，接受一个参数 fn
    def inner(fn):
        # 如果 fn 是 list 或 tuple 类型，则对其中每个元素调用 disallow_in_graph 函数
        if isinstance(fn, (list, tuple)):
            return [disallow_in_graph(x) for x in fn]
        # 断言 fn 必须是可调用对象，否则抛出异常
        assert callable(fn), "disallow_in_graph expects a callable"
        # 如果 throw_if_not_allowed 为真，并且 fn 不在允许的调用对象列表中，抛出 IncorrectUsage 异常
        if (
            throw_if_not_allowed
            and trace_rules.lookup_callable(fn)
            != variables.TorchInGraphFunctionVariable
            and trace_rules.lookup(fn) != variables.TorchInGraphFunctionVariable
        ):
            raise IncorrectUsage(
                "disallow_in_graph is expected to be used on an already allowed callable (like torch.* ops). "
                "Allowed callables means callables that TorchDynamo puts as-is in the extracted graph."
            )
        # 从允许的调用对象集合中移除 fn 的 ID，并将 fn 的 ID 添加到不允许的调用对象集合中
        trace_rules._allowed_callable_ids.remove(id(fn))
        trace_rules._disallowed_callable_ids.add(id(fn))
        # 返回原始的 fn
        return fn

    # 返回 inner 函数对象
    return inner
# 定义一个装饰器函数，用于指定 TorchDynamo 在生成图形时要排除的函数，并在此函数处强制断开图形。
def disallow_in_graph(fn):
    """
    Customize which functions TorchDynamo will exclude in the generated
    graph and force a graph break on.
    ::

        torch._dynamo.disallow_in_graph(torch.sub)

        @torch._dynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            return x

        fn(...)

    Will break the graph on `torch.sub`, and give two graphs each with a
    single `torch.add()` op.
    """
    return _disallow_in_graph_helper(throw_if_not_allowed=True)(fn)


# 定义一个辅助函数，用于强制中断生成的图形。
@_disallow_in_graph_helper(throw_if_not_allowed=False)
def graph_break():
    """Force a graph break"""
    pass


# 定义一个装饰器函数，用于指定 TorchDynamo 在跟踪时断言不应存在的函数。
def forbid_in_graph(fn):
    """
    Customize which functions TorchDynamo will assert are not present while tracing.

    If you want a graph break on this function instead, use disallow_in_graph.
    TODO(voz): We now have allow_in_graph, disallow_in_graph, forbid_in_graph - some more robust
    documentation would not be amiss.
    """
    if isinstance(fn, (list, tuple)):
        return [forbid_in_graph(x) for x in fn]
    assert callable(fn), "forbid_in_graph applies only to callables"
    fn._dynamo_forbidden = True
    return fn


# 定义一个辅助函数，用于扁平化张量子类并对匹配外部维度的所有内部张量应用函数。
# 用于减少各种标记 API 中的重复。
def _apply_func_to_inner_tensors_of_same_dim(func, t, *args, **kwargs):
    assert is_traceable_wrapper_subclass(t)

    attrs, ctx = t.__tensor_flatten__()
    for attr in attrs:
        inner = getattr(t, attr)
        if inner.dim() == t.dim():
            func(inner, *args, **kwargs)


# 定义一个冻结数据类，表示张量的维度范围及其最小和最大值。
# 不要直接创建此类；而是使用 :func:`mark_dynamic`。
@dataclass(frozen=True)
class _DimRange:
    """
    This represents an dimension of a tensor and the corresponding
    min and max values it can take.  Don't create this
    class directly; instead, use :func:`mark_dynamic`.
    """

    dim: int
    min: int
    max: int


# 定义一个装饰器函数，用于标记张量的特定维度为动态维度。
# 这改变了操作的语义，始终报告大小不等于零或一，并将此维度上的断言转换为运行时断言。
# 如果尝试获取真实值，则会引发异常。
@forbid_in_graph
def mark_unbacked(t, index):
    """
    Mark a tensor as having an unbacked dim.  This changes the semantics of operations,
    we will always report the size does not equal zero/one, we will turn asserts
    on this index into runtime asserts, and if you try to get the real value we will
    raise an exception.  In other words, we will treat this dimension as if it was
    data dependent (we do not know anything about its value.)
    """
    # You could have copied the mark_dynamic behavior but I'm not convinced
    # it's what you want
    assert not is_traceable_wrapper_subclass(t), "not implemented yet"

    if isinstance(index, int):
        if not hasattr(t, "_dynamo_unbacked_indices"):
            t._dynamo_unbacked_indices = set()
        t._dynamo_unbacked_indices.add(index)
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        mark_unbacked(t, i)


# 定义一个装饰器函数，用于标记张量的特定维度为动态维度，并指定其最小和最大值。
@forbid_in_graph
def mark_dynamic(t, index, *, min=None, max=None):
    """
    Mark a tensor's dimension as dynamic.

    """
    # 如果给定的张量 t 是可追踪包装器的子类，则将 mark_dynamic() 函数应用于所有与 t 具有相同维度的内部张量
    # 默认行为：将 mark_dynamic() 函数镜像应用于所有与 t 具有相同维度的内部张量
    # TODO: 通过支持的公共 API 使此行为可配置
    _apply_func_to_inner_tensors_of_same_dim(
        mark_dynamic, t, index, min=min, max=max
    )
    
    # 如果 index 是一个整数
    if isinstance(index, int):
        # 如果张量 t 没有属性 _dynamo_dynamic_indices，则创建一个空集合用于存储动态维度的索引
        if not hasattr(t, "_dynamo_dynamic_indices"):
            t._dynamo_dynamic_indices = set()
            # 创建一个空集合用于存储动态维度的范围
            t._dynamo_dynamic_range = set()
        # 将 index 添加到动态维度的索引集合中
        t._dynamo_dynamic_indices.add(index)
        # 将 (index, min, max) 组成的 _DimRange 对象添加到动态维度范围的集合中
        t._dynamo_dynamic_range.add(_DimRange(index, min, max))
        return
    
    # 如果 index 是一个列表或元组
    assert isinstance(index, (list, tuple))
    # 遍历 index 中的每个元素 i，对每个元素都调用 mark_dynamic() 函数，递归标记其动态维度
    for i in index:
        mark_dynamic(t, i, min=min, max=max)
# 标记一个张量的某个维度为动态维度，但不强制执行（即如果这个维度最终被专门化，不会出错）
@forbid_in_graph
def maybe_mark_dynamic(t, index):
    """
    Mark a tensor as having a dynamic dim, but don't enforce it (i.e., if this
    dimension ends up getting specialized, don't error).
    """
    if is_traceable_wrapper_subclass(t):
        # 默认行为：在所有具有与 t 相同维度的内部张量上映射 maybe_mark_dynamic() 方法
        # TODO: 通过支持的公共 API 来使此操作可配置化
        _apply_func_to_inner_tensors_of_same_dim(maybe_mark_dynamic, t, index)

    if isinstance(index, int):
        if not hasattr(t, "_dynamo_weak_dynamic_indices"):
            t._dynamo_weak_dynamic_indices = set()
        # TODO(voz): 是否应该进行边界检查？
        t._dynamo_weak_dynamic_indices.add(index)
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        maybe_mark_dynamic(t, i)


def mark_static(t, index=None):
    """
    标记张量的某个维度为静态维度。

    这将阻止我们在 dynamic=True 时尝试动态编译它；这可以提高追踪时的性能。

    这比 mark_dynamic 优先级低。

    与 mark_dynamic 不同，这可以在图中完成，此时会在张量上引起专门化。
    """
    if is_compiling():
        if index is None:
            for s in t.size():
                comptime.force_static(s)
        else:
            comptime.force_static(t.size(index))
        return

    if is_traceable_wrapper_subclass(t):
        # 默认行为：在所有具有与 t 相同维度的内部张量上映射 mark_static() 方法
        # TODO: 通过支持的公共 API 来使此操作可配置化
        _apply_func_to_inner_tensors_of_same_dim(mark_static, t, index)

    if isinstance(index, int):
        if not hasattr(t, "_dynamo_static_indices"):
            t._dynamo_static_indices = set()
        # TODO(voz): 是否应该进行边界检查？
        t._dynamo_static_indices.add(index)
    elif index is None:
        for i in range(t.dim()):
            mark_static(t, i)
    else:
        assert isinstance(index, (list, tuple))
        for i in index:
            mark_static(t, i)


@forbid_in_graph
def mark_static_address(t, guard=True):
    """
    标记一个输入张量的 data_ptr 在多次调用动态编译函数期间不会更改。
    这向 cudagraphs 指示不需要为此输入进行额外分配。
    如果 guard=True，则会保护 data_ptr。注意：以这种方式标记的张量会一直保持存活状态，直到调用 `torch._dynamo.reset()`。
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"mark_static_address expects a tensor but recieved {type(t)}")

    if guard:
        t._dynamo_static_input_type = "guarded"  # type: ignore[attr-defined]
    else:
        t._dynamo_static_input_type = "unguarded"  # type: ignore[attr-defined]


# 注意：这里仔细避免了急切导入 einops。
# TODO: 我们应该在大约 2024 年 Q2 之前删除整个 _allow_in_graph_einops 逻辑
# 定义一个函数 `_allow_in_graph_einops`，用于配置允许在图计算中使用 einops 库的操作
def _allow_in_graph_einops():
    # 导入 einops 库
    import einops
    
    try:
        # 尝试导入特定于 torch 的 einops 操作，要求 einops 版本 > 0.6.1，torch 版本 >= 2.0
        from einops._torch_specific import (
            _ops_were_registered_in_torchdynamo,  # 导入特定于 torch 的操作注册函数
        )
        
        # 如果成功导入，einops > 0.6.1 会在导入时调用操作注册逻辑，此处仅占位
        pass
    
    except ImportError:
        # 如果导入失败，说明 einops 版本 <= 0.6.1
        
        # 允许在图计算中使用 einops.rearrange 操作
        allow_in_graph(einops.rearrange)
        
        # 允许在图计算中使用 einops.reduce 操作
        allow_in_graph(einops.reduce)
        
        # 如果 einops 拥有 repeat 属性，允许在图计算中使用 einops.repeat 操作（自 einops 0.2.0 起可用）
        if hasattr(einops, "repeat"):
            allow_in_graph(einops.repeat)
        
        # 如果 einops 拥有 einsum 属性，允许在图计算中使用 einops.einsum 操作（自 einops 0.5.0 起可用）
        if hasattr(einops, "einsum"):
            allow_in_graph(einops.einsum)
        
        # 如果 einops 拥有 pack 属性，允许在图计算中使用 einops.pack 操作（自 einops 0.6.0 起可用）
        if hasattr(einops, "pack"):
            allow_in_graph(einops.pack)
        
        # 如果 einops 拥有 unpack 属性，允许在图计算中使用 einops.unpack 操作（自 einops 0.6.0 起可用）
        if hasattr(einops, "unpack"):
            allow_in_graph(einops.unpack)

# 将 `_allow_in_graph_einops` 函数注册到 trace_rules 的模块初始化函数中，使得在 "einops" 模块初始化时执行此函数
trace_rules.add_module_init_func("einops", _allow_in_graph_einops)
```