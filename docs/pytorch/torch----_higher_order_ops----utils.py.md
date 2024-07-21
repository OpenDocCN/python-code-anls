# `.\pytorch\torch\_higher_order_ops\utils.py`

```
# 导入必要的模块和库
import functools  # 导入 functools 模块，用于高阶函数和函数工具
from contextlib import contextmanager  # 导入 contextmanager 类，用于创建上下文管理器
from dataclasses import dataclass  # 导入 dataclass 类，用于创建简单的类
from typing import Any, Callable  # 导入类型提示相关的类和函数

import torch  # 导入 PyTorch 库
import torch.fx.traceback as fx_traceback  # 导入 torch.fx.traceback 模块
import torch.utils._pytree as pytree  # 导入 torch.utils._pytree 模块
from torch._ops import OperatorBase  # 导入 OperatorBase 类
from torch.fx.experimental.proxy_tensor import make_fx  # 导入 make_fx 函数
from torch.multiprocessing.reductions import StorageWeakRef  # 导入 StorageWeakRef 类


@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    reason: str  # 定义 UnsupportedAliasMutationException 异常类，包含 reason 字段


def autograd_not_implemented_inner(
    operator: OperatorBase, delayed_error: bool, *args: Any, **kwargs: Any
) -> Any:
    """If autograd is enabled and any of the arguments require grad this will either
    raise an error or return a DelayedError depending on the value of delayed.

    Args:
        operator: The Operator to call with the *args and **kwargs with
        op_name: The name of the Operator
        delayed_error: If True, return a DelayedError instead of raising an error
        args: The flattened operands to the Operator
        kwargs: The keyword arguments to the Operator

    Raises:
        RuntimeError: If autograd is enabled and any of the arguments to the Operator
    """
    with torch._C._AutoDispatchBelowAutograd():  # 进入上下文管理器，设置自动分发到 Autograd 以下
        result = operator(*args, **kwargs)  # 调用 operator 对象，传入参数 *args 和 **kwargs
        flat_operands = pytree.arg_tree_leaves(*args)  # 获取 *args 中所有叶子节点

        # 检查是否启用了梯度追踪，并且有需要梯度的 Tensor 存在
        if torch.is_grad_enabled() and any(
            f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
        ):
            if delayed_error:
                # 创建一个 DelayedError 实例，返回处理后的结果
                err_fn = torch._C._functions.DelayedError(
                    f"Autograd not implemented for {str(operator)}",
                    1,
                )

                # 定义一个函数 fake_requires_grad，处理梯度要求的 Tensor
                def fake_requires_grad(tensor):
                    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                        tensor = tensor.detach()  # 分离 Tensor
                        tensor.requires_grad = True  # 设置 requires_grad 为 True
                    return tensor

                # 对结果应用 fake_requires_grad 函数，返回处理后的结果
                return pytree.tree_map_only(
                    torch.Tensor, lambda x: err_fn(fake_requires_grad(x)), result
                )
            else:
                # 如果 delayed_error 为 False，则抛出 RuntimeError 异常
                raise RuntimeError(f"Autograd not implemented for {str(operator)}")

        # 如果未有 Tensor 需要梯度，直接返回结果
        return result


def autograd_not_implemented(op: OperatorBase, deferred_error: bool) -> Callable:
    """Decorator function that returns a function wrapping autograd_not_implemented_inner.

    Args:
        op: The OperatorBase instance to be wrapped
        deferred_error: If True, return a function that returns a DelayedError instead of raising an error

    Returns:
        Callable: A function that wraps autograd_not_implemented_inner with the provided parameters
    """
    def inner(*args, **kwargs):
        return autograd_not_implemented_inner(op, deferred_error, *args, **kwargs)

    return inner


def _maybe_run_with_interpreter(fn):
    """Returns either the original function or a function wrapped in interpreter for FX graph handling.

    Args:
        fn: The function to potentially wrap with an interpreter

    Returns:
        Callable: The original function or a new function with interpreter handling
    """
    maybe_interpreted_fn = fn

    # 如果 fn 是 torch.fx.GraphModule 类型且具有保留节点元信息，则使用解释器运行图形
    if isinstance(fn, torch.fx.GraphModule) and fx_traceback.has_preserved_node_meta():
        def graph_with_interpreter(*args):
            with fx_traceback.preserve_node_meta():
                return torch.fx.Interpreter(fn).run(*args)

        maybe_interpreted_fn = graph_with_interpreter

    return maybe_interpreted_fn


def reenter_make_fx(fn):
    """Dummy function definition; typically it would have additional implementation details.

    Args:
        fn: Function to be reentered with make_fx

    Notes:
        This function serves as a placeholder for future implementation.
    """
    pass  # 占位函数，目前没有实际实现内容
    # 导入torch.fx.experimental.proxy_tensor模块中的_CURRENT_MAKE_FX_TRACER变量
    from torch.fx.experimental.proxy_tensor import _CURRENT_MAKE_FX_TRACER

    # 定义一个装饰器函数wrapped，用于包装给定的函数fn，并保留其元数据
    @functools.wraps(fn)
    def wrapped(*args):
        # 断言确保_CURRENT_MAKE_FX_TRACER不为None，以确保在make_fx跟踪会话中重新进入
        assert (
            _CURRENT_MAKE_FX_TRACER is not None
        ), "Cannot reenter make_fx when we're not under a make_fx tracing session"
        # 使用_CURRENT_MAKE_FX_TRACER对象的trace_subgraph方法跟踪子图
        return _CURRENT_MAKE_FX_TRACER.trace_subgraph(
            _maybe_run_with_interpreter(fn), *args
        )

    # 返回装饰后的函数wrapped
    return wrapped
@contextmanager
def _set_compilation_env():
    # 保存旧的 FX 追踪标志和内置 NN 模块内联设置
    _old_is_tracing = torch.fx._symbolic_trace._is_fx_tracing_flag
    _old_is_inlining = torch._dynamo.config.inline_inbuilt_nn_modules
    try:
        # 关闭 FX 追踪标志。在确认 FX 追踪与 Dynamo 兼容后，从 Dynamo 中移除此标志检查。
        torch.fx._symbolic_trace._is_fx_tracing_flag = False

        # 为了避免目前的问题，暂时禁止内置 NN 模块的内联
        torch._dynamo.config.inline_inbuilt_nn_modules = False
        yield
    finally:
        # 恢复旧的 FX 追踪标志和内置 NN 模块内联设置
        torch.fx._symbolic_trace._is_fx_tracing_flag = _old_is_tracing
        torch._dynamo.config.inline_inbuilt_nn_modules = _old_is_inlining


def _has_potential_branch_input_mutation(branch, inputs, pre_dispatch=False):
    """
    Dispatch-trace the branch with inputs and check if
    producing graph has mutable op on the input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        # 使用输入参数调用 make_fx 函数，并生成图模块
        gm = make_fx(branch, pre_dispatch=pre_dispatch)(*inputs)
    except UnsupportedAliasMutationException:
        # 当嵌套的 cond_op 被功能化时可能发生此异常
        return True
    except Exception as e:
        raise e

    def _detect_input_mutation(gm):
        # 检测图模块中的输入变异情况
        input_nodes = set()
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                input_nodes.add(node)
            if node.op == "call_function":
                target = node.target
                if (
                    isinstance(target, torch._ops.OpOverload)
                    and target._schema.is_mutable
                ):
                    for arg in node.args:
                        if arg in input_nodes:
                            return True

        for _, module in gm.named_children():
            if isinstance(module, torch.fx.GraphModule):
                if _detect_input_mutation(module):
                    return True

        return False

    return _detect_input_mutation(gm)


def _has_potential_branch_input_alias(branch, inputs, pre_dispatch=False):
    """
    Dispatch-trace the branch with inputs and check if
    producing graph has output aliasing the branch input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        # 使用输入参数调用 make_fx 函数，并生成图模块
        gm = make_fx(branch, pre_dispatch=pre_dispatch)(*inputs)
    except UnsupportedAliasMutationException:
        # 当嵌套的 cond_op 被功能化时可能发生此异常
        return True
    except Exception as e:
        raise e
    # 检测输入别名的私有函数
    def _detect_input_alias(gm):
        # 存储输入的集合
        input_storages = set()
        # 遍历计算图中的每个节点
        for node in gm.graph.nodes:
            # 如果节点操作为"placeholder"并且具有"val"元数据
            if node.op == "placeholder" and "val" in node.meta:
                # 将节点的"val"元数据的类型化存储添加到输入存储集合中
                input_storages.add(StorageWeakRef(node.meta["val"]._typed_storage()))
            
            # 如果节点操作为"output"
            if node.op == "output":
                
                # 定义检查别名的内部函数
                def check_alias(out):
                    # 如果输出不为None并且具有"val"元数据
                    if out is not None and "val" in out.meta:
                        # 获取输出的"val"元数据的类型化存储
                        out_storage = StorageWeakRef(out.meta["val"]._typed_storage())
                        # 返回该存储是否在输入存储集合中
                        return out_storage in input_storages
                    return False
                
                # 如果任何一个节点输出参数经过检查别名函数返回True
                if any(pytree.tree_leaves(pytree.tree_map(check_alias, node.args))):
                    return True

        # 遍历命名子模块
        for _, module in gm.named_children():
            # 如果子模块是torch.fx.GraphModule且其自身调用_detect_input_alias返回True
            if isinstance(module, torch.fx.GraphModule) and _detect_input_alias(module):
                return True

        # 如果没有检测到输入别名，则返回False
        return False

    # 返回调用_detect_input_alias函数并传入gm参数的结果
    return _detect_input_alias(gm)
# 根据给定的前缀和代理模式生成一个唯一的图形名称和ID，用于添加到代理模式追踪器中
def unique_graph_id(proxy_mode, prefix):
    """Returns a unique name and id for a graph to be added to a proxy_mode tracer"""
    # 初始化下一个名称为 None
    next_name = None
    # 初始索引为 0
    i = 0
    # 循环直到找到一个唯一的名称
    while not next_name:
        # 生成候选的名称
        candidate = f"{prefix}_{i}"
        # 检查代理模式追踪器的根节点是否已经有相同名称的属性
        if hasattr(proxy_mode.tracer.root, candidate):
            # 如果有，增加索引以尝试下一个名称
            i += 1
        else:
            # 如果没有，找到了唯一的名称，退出循环
            next_name = candidate
    # 返回索引和找到的唯一名称
    return i, next_name
```