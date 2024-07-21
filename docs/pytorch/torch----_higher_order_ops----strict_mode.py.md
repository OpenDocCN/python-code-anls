# `.\pytorch\torch\_higher_order_ops\strict_mode.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 导入 torch._subclasses.functional_tensor 模块
import torch._subclasses.functional_tensor
# 导入 torch.utils._pytree 模块，命名为 pytree
import torch.utils._pytree as pytree
# 从 torch._C 导入 DispatchKey 枚举
from torch._C import DispatchKey
# 从 torch._functorch.utils 导入 exposed_in 函数
from torch._functorch.utils import exposed_in
# 从 torch._higher_order_ops.utils 导入 _set_compilation_env 和 autograd_not_implemented 函数
from torch._higher_order_ops.utils import _set_compilation_env, autograd_not_implemented
# 从 torch._ops 导入 HigherOrderOperator 类
from torch._ops import HigherOrderOperator
# 从 torch._subclasses.fake_tensor 导入 FakeTensorMode 枚举
from torch._subclasses.fake_tensor import FakeTensorMode
# 从 torch.fx.experimental.proxy_tensor 导入多个函数：disable_proxy_modes_tracing, make_fx, ProxyTorchDispatchMode, track_tensor_tree
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
# 从 torch.utils._python_dispatch 导入 _get_current_dispatch_mode 函数
from torch.utils._python_dispatch import _get_current_dispatch_mode


# 将 strict_mode 函数公开在 torch 模块中
@exposed_in("torch")
def strict_mode(callable, operands):
    # 如果正在使用动态编译器编译，则调用 strict_mode_op 处理
    if torch.compiler.is_dynamo_compiling():
        return strict_mode_op(callable, operands)

    # 设置编译环境，并关闭缓存限制
    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            # 使用 eager 模式编译 strict_mode_op 函数
            return torch.compile(strict_mode_op, backend="eager", fullgraph=True)(
                callable, operands
            )


# 创建 HigherOrderOperator 实例 strict_mode_op，并命名为 "strict_mode"
strict_mode_op = HigherOrderOperator("strict_mode")


# 将 strict_mode_op 定义为 DispatchKey.CompositeExplicitAutograd 的 Python 实现
@strict_mode_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def strict_mode_op_dense(callable, operands):
    # 获取当前的分发模式
    mode = _get_current_dispatch_mode()
    # 断言当前模式为 None，不应启用 CPU/CUDA 键的模式
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    # 调用传入的 callable 函数并返回结果
    return callable(*operands)


# 将 strict_mode_op 定义为 DispatchKey.Autograd 的 Python 实现
@strict_mode_op.py_impl(DispatchKey.Autograd)
def strict_mode_op_autograd(callable, operands):
    # 调用 autograd_not_implemented 函数处理 strict_mode_op 函数，并返回结果
    return autograd_not_implemented(strict_mode_op, deferred_error=True)


# 将 strict_mode_op 定义为 ProxyTorchDispatchMode 的 Python 实现
@strict_mode_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, callable, operands):
    # 如果启用追踪模式，则调用 trace_strict_mode 函数；否则直接调用 strict_mode_op 处理
    if mode.enable_tracing:
        return trace_strict_mode(mode, strict_mode_op, callable, operands)
    else:
        return strict_mode_op(callable, operands)


# 定义 trace_strict_mode 函数，用于在跟踪模式下处理 strict_mode_op 函数
def trace_strict_mode(mode, strict_mode_op, callable, operands):
    # 获取 mode 对象的 pre_dispatch 属性
    pre_dispatch = getattr(mode, "pre_dispatch", False)

    # 禁用代理模式的跟踪
    with disable_proxy_modes_tracing():
        # 使用 make_fx 函数创建 callable 的图形表示
        graph = make_fx(callable, pre_dispatch=pre_dispatch)(*operands)

    # 为图形生成唯一的名称
    graph_name = mode.tracer.get_fresh_qualname("strict_graph_")
    # 将 graph 注册到 mode.tracer.root 下
    mode.tracer.root.register_module(graph_name, graph)

    # 构建参数元组
    args = (graph, operands)

    # 将 args 中的代理对象解包为非代理对象
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)

    # 使用 mode.tracer 创建代理对象，表示调用 strict_mode_op 函数
    out_proxy = mode.tracer.create_proxy(
        "call_function", strict_mode_op, proxy_args, {}, name="strict_mode"
    )

    # 调用 graph 函数并获取输出结果
    out = graph(*operands)
    # 跟踪张量树的变化，并返回最终结果
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


# 将 strict_mode_op 定义为 FakeTensorMode 的 Python 实现
@strict_mode_op.py_impl(FakeTensorMode)
def strict_mode_fake_tensor_mode(mode, callable, operands):
    # 使用 mode 上下文执行 callable 函数，并返回结果
    with mode:
        true_outs = callable(*operands)
    return true_outs


# 将 strict_mode_op 定义为函数的 Python 函数化实现
@strict_mode_op.py_functionalize_impl
def strict_mode_func(ctx, callable, inputs):
    # 使用 ctx.unwrap_tensors 解包输入的张量
    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    # 重新调度到下一个函数上下文
    with ctx.redispatch_to_next():
        # 将 callable 函数转换为函数式调用
        functional_callable = ctx.functionalize(callable)

        # 调用 strict_mode_op 函数并返回结果
        cond_return = strict_mode_op(functional_callable, unwrapped_inputs)
        # 使用 ctx.wrap_tensors 包装结果并返回
        return ctx.wrap_tensors(cond_return)
```