# `.\pytorch\torch\_higher_order_ops\while_loop.py`

```py
# mypy: allow-untyped-defs
# 引入类型相关模块和函数
from typing import Callable, Tuple, Union

# 引入PyTorch相关模块
import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _set_compilation_env,
    autograd_not_implemented,
    reenter_make_fx,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree

# 定义WhileLoopOp类，继承自HigherOrderOperator类
class WhileLoopOp(HigherOrderOperator):
    def __init__(self):
        # 调用父类构造函数初始化名称为"while_loop"的操作符
        super().__init__("while_loop")

    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: Tuple[Union[torch.Tensor, int, float, bool]],
        additional_inputs: Tuple[Union[torch.Tensor, int, float, bool]],
        /,
    ):
        # 检查carried_inputs是否为元组类型，若不是则抛出运行时异常
        if not isinstance(carried_inputs, tuple):
            raise RuntimeError(
                f"carried_inputs must be a tuple, got {type(carried_inputs)}"
            )
        # 检查additional_inputs是否为元组类型，若不是则抛出运行时异常
        if not isinstance(additional_inputs, tuple):
            raise RuntimeError(
                f"additional_inputs must be a tuple, got {type(additional_inputs)}"
            )
        # 检查carried_inputs中的每个元素是否为torch.Tensor、int、float或bool类型，若不是则抛出运行时异常
        if not all(
            isinstance(t, (torch.Tensor, int, float, bool)) for t in carried_inputs
        ):
            raise RuntimeError(
                "carried_inputs must be a tuple of tensors, ints, floats, or bools, got "
                f"{carried_inputs}"
            )
        # 检查additional_inputs中的每个元素是否为torch.Tensor、int、float或bool类型，若不是则抛出运行时异常
        if not all(
            isinstance(t, (torch.Tensor, int, float, bool)) for t in additional_inputs
        ):
            raise RuntimeError(
                "additional_inputs must be a tuple of tensors, ints, floats, or bools, got "
                f"{additional_inputs}"
            )
        # 调用父类的__call__方法，传递cond_fn、body_fn、carried_inputs和additional_inputs参数
        return super().__call__(cond_fn, body_fn, carried_inputs, additional_inputs)

# 创建WhileLoopOp对象
while_loop_op = WhileLoopOp()
# 将while_loop_op的__module__属性设置为"torch.ops.higher_order"，确保在生成的图模块中while_loop节点的目标正确打印为torch.ops.higher_order.while_loop
while_loop_op.__module__ = "torch.ops.higher_order"

# 定义while_loop函数
def while_loop(cond_fn, body_fn, carried_inputs):
    r"""
    Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or
    initial carried_inputs.

    .. warning::
        `torch.while_loop` is a prototype feature in PyTorch. It has limited support for input and output types and
        doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `while_loop` is a structured control flow operator. It preserves the loop semantic across the torch.compile and torch.export.
    """
    """
    `while_loop` is equivalent to the following:
    
        def while_loop(cond_fn, body_fn, carried_inputs):
            val = carried_inputs
            while cond_fn(*val):
                val = body_fn(*val)
            return val
    
    Args:
        cond_fn (Callable): A callable function that returns a boolean Scalar tensor.
            条件函数，接受与 `carried_inputs` 相同的参数，返回布尔标量张量。
    
        body_fn (Callable): A callable function that takes the same inputs as `cond_fn` and returns a tuple of tensors
            主体函数，接受与 `cond_fn` 相同的参数，返回张量元组。
    
        carried_inputs (Tuple of possibly nested dict/list/tuple of tensors): A tuple of inputs to cond_fn and body_fn. It's also
            the initial value of states that are carried across iterations.
            传递给 `cond_fn` 和 `body_fn` 的输入参数元组，也是在迭代过程中保持的初始状态值。
    
    Example:
    
        def cond_fn(iter, x):
            return iter.sum() < 10
    
        def body_fn(iter, x):
            return iter + 1, x.sin()
    
        while_loop(cond_fn, body_fn, (torch.zeros(1), torch.randn(3, 4)))
    
    Restrictions:
    
        - body_fn must return tensors with the same metadata (e.g.shape, dtype) as inputs.
            `body_fn` 必须返回与输入张量具有相同的元数据（如形状、数据类型）的张量。
    
        - body_fn and cond_fn must not in-place mutate the carried_inputs. A clone before the mutation is required.
            `body_fn` 和 `cond_fn` 不能就地更改 `carried_inputs`。在变异之前需要进行克隆。
    
        - body_fn and cond_fn must not mutate python varialbles (e.g. list/dict) created outside of the body_fn.
            `body_fn` 和 `cond_fn` 不能变异在函数外创建的 Python 变量（例如列表/字典）。
    
        - body_fn and cond_fn's output cannot aliase any of the inputs. A clone is required.
            `body_fn` 和 `cond_fn` 的输出不能与任何输入共享引用。需要进行克隆。
    
    .. warning::
        Temporal Limitations:
    
        - 'while_loop' only supports **inference** right now. Autograd will be supported in the future.
            目前 'while_loop' 仅支持推断。自动求导将在未来支持。
    
    """
    
    # Currently, additional_inputs is not a user-facing input. It will be automatically set in dynamo.
    # parameters and buffers accessed in cond_fn or body_fn or tensor closures will become additional_inputs.
    additional_inputs: Tuple = tuple()
    if torch.compiler.is_dynamo_compiling():
        return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)
    
    def _validate_input(cond_fn, body_fn, carried_inputs):
        if not callable(cond_fn) or not callable(body_fn):
            raise RuntimeError("Expect cond_fn and body_fn to be callbale.")
    
        if not isinstance(carried_inputs, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), carried_inputs
        ):
            raise RuntimeError(
                "Expect carried_inputs to be a tuple of possibly nested dict/list/tuple that only"
                f"consists of tensor leaves, but got {carried_inputs}."
            )
    
    _validate_input(cond_fn, body_fn, carried_inputs)
    
    with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
        return torch.compile(while_loop_op, backend="eager", fullgraph=True)(
            cond_fn, body_fn, carried_inputs, additional_inputs
        )
@while_loop_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def while_loop_dense(cond_fn, body_fn, carried_inputs, additional_inputs):
    # 初始化 carried_vals 为 carried_inputs
    carried_vals = carried_inputs

    # 定义一个内部函数，用于检查是否是布尔标量张量
    def _is_boolean_scalar_tensor(pred):
        return (
            isinstance(pred, torch.Tensor)
            and pred.size() == torch.Size([])
            and pred.dtype == torch.bool
        )

    # 如果 carried_inputs 不是元组，则抛出 RuntimeError 异常
    if not isinstance(carried_inputs, tuple):
        raise RuntimeError(
            f"carried_inputs must be a tuple but got {type(carried_inputs)}"
        )

    # 进入循环，条件由 cond_fn 决定，调用时传入 carried_vals 和 additional_inputs
    while pred := cond_fn(*carried_vals, *additional_inputs):
        # 检查 cond_fn 返回值是否为布尔标量张量，如果不是则抛出异常
        if not _is_boolean_scalar_tensor(pred):
            raise RuntimeError(
                f"cond_fn must return a boolean scalar tensor but got {pred}"
            )
        # 执行 body_fn，并将结果赋值给 out
        out = body_fn(*carried_vals, *additional_inputs)
        # 断言 body_fn 返回值为元组，如果不是则抛出异常
        assert isinstance(
            out, tuple
        ), f"body_fn should return a tuple but got {type(out)}"
        # 断言 body_fn 返回的元素数量与 carried_inputs 相同，如果不是则抛出异常
        assert len(out) == len(
            carried_inputs
        ), "body_fn should return the same number of elements as carried_inputs"
        # 更新 carried_vals 为 body_fn 的返回值
        carried_vals = out

    # 循环结束后返回最终的 carried_vals
    return carried_vals


# 调用 Autograd 模式下的 while_loop_op，并传入 autograd_not_implemented 函数结果
while_loop_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(while_loop_op, deferred_error=True)
)


@while_loop_op.py_impl(ProxyTorchDispatchMode)
def while_loop_tracing(mode, cond_fn, body_fn, carried_inputs, additional_inputs):
    # 定义内部函数 _trace_while_loop，用于在追踪模式下执行循环
    def _trace_while_loop(
        proxy_mode, while_loop_op, cond_fn, body_fn, carried_inputs, additional_inputs
    ):
        # 生成 cond_fn 和 body_fn 的函数图
        cond_graph = reenter_make_fx(cond_fn)(*carried_inputs, *additional_inputs)
        body_graph = reenter_make_fx(body_fn)(*carried_inputs, *additional_inputs)

        # 为 cond_graph 和 body_graph 选择唯一的名称
        next_name = None
        i = 0
        while not next_name:
            candidate = f"while_loop_cond_graph_{i}"
            if hasattr(proxy_mode.tracer.root, candidate):
                i += 1
            else:
                next_name = candidate
        cond_graph_name = next_name
        body_graph_name = f"while_loop_body_graph_{i}"
        assert not hasattr(proxy_mode.tracer.root, body_graph_name)

        # 在追踪模式的根节点中注册 cond_graph 和 body_graph
        proxy_mode.tracer.root.register_module(cond_graph_name, cond_graph)
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

        # 准备调用参数列表
        args = (cond_graph, body_graph, carried_inputs, additional_inputs)

        # 使用代理模式解包参数列表
        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

        # 创建一个代理对象，用于调用 while_loop_op
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", while_loop_op, proxy_args, {}, name="while_loop"
        )

        # 调用 body_fn 一次，获取输出，并使用 track_tensor_tree 函数追踪输出
        out = body_fn(*carried_inputs, *additional_inputs)
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer
        )
    # 如果启用了追踪模式（mode.enable_tracing 为真），则调用 _trace_while_loop 函数来执行追踪循环操作，
    # 该函数接受 mode、while_loop_op、cond_fn、body_fn、carried_inputs 和 additional_inputs 六个参数。
    if mode.enable_tracing:
        return _trace_while_loop(
            mode, while_loop_op, cond_fn, body_fn, carried_inputs, additional_inputs
        )
    # 如果未启用追踪模式，直接调用 while_loop_op 函数来执行普通的循环操作，
    # 该函数接受 cond_fn、body_fn、carried_inputs 和 additional_inputs 四个参数。
    else:
        return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)
# 定义一个装饰器函数，将当前函数实现替换为由 while_loop_op.py_impl(FakeTensorMode) 提供的实现
@while_loop_op.py_impl(FakeTensorMode)
# 定义函数 while_loop_fake_tensor_mode，接收参数 mode, cond_fn, body_fn, carried_inputs, additional_inputs
def while_loop_fake_tensor_mode(
    mode, cond_fn, body_fn, carried_inputs, additional_inputs
):
    # 使用给定的 mode 上下文环境执行以下代码块
    with mode:
        # 调用 body_fn 函数，传入 carried_inputs 和 additional_inputs 作为参数，并返回其结果
        return body_fn(*carried_inputs, *additional_inputs)

# 定义一个装饰器函数，将当前函数实现替换为由 while_loop_op.py_functionalize_impl 提供的实现
@while_loop_op.py_functionalize_impl
# 定义函数 while_loop_func，接收参数 ctx, cond_fn, body_fn, carried_inputs, additional_inputs
def while_loop_func(ctx, cond_fn, body_fn, carried_inputs, additional_inputs):
    # 使用 ctx.unwrap_tensors 方法解包 carried_inputs 中的张量
    unwrapped_carried_inputs = ctx.unwrap_tensors(carried_inputs)
    # 使用 ctx.unwrap_tensors 方法解包 additional_inputs 中的张量
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    # 将解包后的输入合并为一个列表
    unwrapped_inputs = unwrapped_carried_inputs + unwrapped_additional_inputs
    # 使用 ctx.redispatch_to_next() 方法获取一个上下文 m，并执行以下代码块
    with ctx.redispatch_to_next() as m:
        # 将 cond_fn 和 body_fn 转换为功能化版本
        functional_cond_fn = ctx.functionalize(cond_fn)
        functional_body_fn = ctx.functionalize(body_fn)
        # 检查是否存在潜在的分支输入变异，如果存在则抛出异常
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for fn, fn_name in [
            (functional_cond_fn, "cond_fn"),
            (functional_body_fn, "body_fn"),
        ]:
            if _has_potential_branch_input_mutation(
                fn, unwrapped_inputs, pre_dispatch=pre_dispatch
            ):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be modifying the input!"
                )

            # 检查是否存在潜在的分支输入别名，如果存在则抛出异常
            if _has_potential_branch_input_alias(
                fn, unwrapped_inputs, pre_dispatch=pre_dispatch
            ):
                raise UnsupportedAliasMutationException(
                    f"torch.while_loop's {fn_name} might be aliasing the input!"
                )
        # 调用 while_loop_op 函数，传入 functional_cond_fn, functional_body_fn,
        # unwrapped_carried_inputs, unwrapped_additional_inputs 作为参数，并返回结果
        ret = while_loop_op(
            functional_cond_fn,
            functional_body_fn,
            unwrapped_carried_inputs,
            unwrapped_additional_inputs,
        )
        # 使用 ctx.wrap_tensors 方法封装返回结果 ret，并返回封装后的结果
        return ctx.wrap_tensors(ret)
```