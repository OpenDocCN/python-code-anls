# `.\pytorch\torch\_dynamo\_trace_wrapped_higher_order_op.py`

```py
# mypy: allow-untyped-defs
# 引入 torch 库
import torch
# 从 torch._C 模块中引入 DispatchKey
from torch._C import DispatchKey
# 引入 autograd_not_implemented 函数
from torch._higher_order_ops.utils import autograd_not_implemented

# 从 torch._ops 模块中引入 HigherOrderOperator 类
from torch._ops import HigherOrderOperator
# 从 torch._subclasses 模块中引入 FakeTensorMode 类
from torch._subclasses import FakeTensorMode
# 从 torch.fx.experimental._backward_state 模块中引入 BackwardState 类
from torch.fx.experimental._backward_state import BackwardState

# 从 torch.fx.experimental.proxy_tensor 模块中引入 ProxyTorchDispatchMode 和 track_tensor_tree 函数
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
# 从 torch.utils._python_dispatch 模块中引入 _get_current_dispatch_mode 函数
from torch.utils._python_dispatch import _get_current_dispatch_mode
# 从 torch.utils._pytree 模块中引入 tree_map_only 函数
from torch.utils._pytree import tree_map_only

# 设置 __all__ 列表，定义模块公开的接口
__all__ = ["trace_wrapped"]

# trace_wrapped(*args, fn) 函数的作用是对 fn(*args) 进行追踪，但有一个变化：
# 如果在此调用中进行 make_fx 追踪，我们不会实际追踪到 fn；相反，
# 我们将直接将其作为一个 call_function 插入到图中的 fn。
# （与 make_fx 不同，Dynamo 将内联到 fn 中。）
# 可以将此视为代理张量追踪中的一次性 allow_in_graph 等效操作。
#
# 由于代理张量追踪实际上并不运行该函数，因此对 fn 的行为有一些要求。我们仍在完善中，但以下是当前的要求：
#
# 1) fn 应该只接受一个参数，该参数必须是一个张量。
# 2) fn 必须返回一个与原始张量具有相同元数据的新张量
#    （例如，zeros_like(input) 是 fn 的一个允许的实现）。
#    这通过追踪图中插入的额外 assert 进行验证。
# 3) fn 可能具有副作用，但不能对参与代理张量追踪的其他张量执行元数据突变
#    （它可以改变其他张量，可以改变 Python 状态）。
# 这些要求源于我们需要继续执行代理张量追踪的需求，这种追踪假设准确的假张量元数据，而不实际运行 fn。
# 未来，我们可能会允许与 fn 关联的 "meta" 函数，以允许更有趣的输入输出模式。
#
# 注意，张量/Python 状态是允许被改变的。
# 这种放松的约束并不总是安全的，但对于使用假张量进行的反向追踪来说是安全的，
# 因为它发生在 AOTAutograd 中，由于反向传播不依赖于具体张量值（通过假张量）或 Python 状态
# （因为自动求导引擎不依赖于 Python）。
#
# 此函数的预期使用场景是允许 AOTAutograd 将复杂的反向钩子推迟到编译的自动求导中。
# AOTAutograd 执行 make_fx 追踪，保留图中的函数调用，只有当我们在编译的自动求导中 Dynamo 通过反向图时，
# 我们才会内联到函数中。

def trace_wrapped(*args, **kwargs):
    # 使用 torch.no_grad() 上下文管理器，确保在此期间不计算梯度
    with torch.no_grad():
        # 调用 _trace_wrapped_op 函数来执行实际的追踪操作
        return _trace_wrapped_op(*args, **kwargs)


# 定义 _trace_wrapped_op 变量，使用 HigherOrderOperator 类创建名为 "trace_wrapped" 的高阶运算符
# 这个运算符用于将 trace_wrapped 函数插入到图中，而不进行实际追踪
_trace_wrapped_op = HigherOrderOperator("trace_wrapped")


# _assert_meta 函数用于验证梯度的元数据与给定的 size、stride 和 dtype 是否匹配
def _assert_meta(grad, size, stride, dtype):
    assert grad.size() == size, "size mismatch"
    assert grad.stride() == stride, "stride mismatch"
    assert grad.dtype == dtype, "dtype mismatch"
    return grad


注释：


    # 返回函数中计算得到的梯度值
    return grad
# 使用装饰器将函数注册到特定的分发模式中，使其能够被追踪
@_trace_wrapped_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(mode, *args, bw_state=None, **kwargs):
    # 定义内部函数self_invoke，通过torch.no_grad()确保不追踪梯度
    def self_invoke(*args, **dyn_kwargs):
        with torch.no_grad():
            # 调用_trace_wrapped_op函数，并传递动态参数和kwargs
            return _trace_wrapped_op(*args, **dyn_kwargs, **kwargs)

    # 定义内部函数unwrap_proxies，根据输入的类型解包代理对象
    def unwrap_proxies(x):
        if isinstance(x, torch.Tensor):
            return mode.tracer.unwrap_proxy(x)
        if isinstance(x, (list, tuple)):
            return type(x)(map(unwrap_proxies, x))
        if x is None:
            return None
        # 抛出断言错误，处理未预期的类型
        raise AssertionError(f"unhandled type: {type(x)}")

    # 初始化空的代理关键字参数字典
    proxy_kwargs = {}
    # 如果bw_state不为None，确保其为BackwardState实例，并且其代理不为空
    if bw_state is not None:
        assert isinstance(bw_state, BackwardState) and bw_state.proxy is not None
        # 将bw_state的代理对象添加到proxy_kwargs中
        proxy_kwargs["bw_state"] = bw_state.proxy
    # 使用mode.tracer.create_proxy创建一个代理对象out_proxy
    out_proxy = mode.tracer.create_proxy(
        "call_function",  # 代理对象的类型为call_function
        self_invoke,  # 被代理的函数是self_invoke
        unwrap_proxies(args),  # 对输入的args进行解包代理
        proxy_kwargs,  # 传递的关键字参数为proxy_kwargs
        name="trace_wrapped",  # 代理对象的名称为trace_wrapped
    )

    # 根据args[0]的值确定grad的赋值
    if args[0] is None:
        grad = args[1]  # 如果args[0]为None，则grad为args[1]，通常为模块的backward hooks
    else:
        grad = args[0]  # 否则grad为args[0]，通常为其他backward hooks
    # 使用tree_map_only函数将grad中的所有元素转换为torch.Tensor类型，并用torch.empty_like初始化
    grad = tree_map_only(torch.Tensor, torch.empty_like, grad)
    # 调用track_tensor_tree函数追踪grad的张量树，使用out_proxy作为追踪目标
    track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    # 返回经过处理的grad
    return grad


# 将该函数标记为不应在当前上下文中调用的操作，并引发RuntimeError
@_trace_wrapped_op.py_impl(FakeTensorMode)
def inner_fake(*args, **kwargs):
    raise RuntimeError("This op should never be invoked here")


# 将函数注册到特定的分发模式中，用于处理稠密型操作
@_trace_wrapped_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _trace_wrapped_op_dense(*args, fn, **kwargs):
    # 获取当前的分发模式
    mode = _get_current_dispatch_mode()
    # 断言当前模式为None，表明该操作不应启用于CPU/CUDA键
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    # 调用传入的fn函数，传递args和kwargs
    return fn(*args, **kwargs)


# 将autograd_not_implemented函数应用到_trace_wrapped_op函数上，并将其注册到Autograd分发键中
_trace_wrapped_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_trace_wrapped_op, deferred_error=True)
)


# 使用py_functionalize_impl将函数_functionalize_impl函数化，支持上下文管理和张量包装
@_trace_wrapped_op.py_functionalize_impl
def _trace_wrapped_functionalized(ctx, *args, **kwargs):
    # 使用ctx.unwrap_tensors解包输入的args
    unwrapped_args = ctx.unwrap_tensors(args)
    # 使用ctx.redispatch_to_next上下文管理器
    with ctx.redispatch_to_next():
        # 调用_trace_wrapped_op函数，传递解包后的参数和kwargs，并使用ctx.wrap_tensors包装结果
        return ctx.wrap_tensors(_trace_wrapped_op(*unwrapped_args, **kwargs))
```