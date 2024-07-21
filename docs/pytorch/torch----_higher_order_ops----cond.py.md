# `.\pytorch\torch\_higher_order_ops\cond.py`

```py
# 导入模块 'contextlib'，用于上下文管理器和资源管理相关的功能
import contextlib

# 导入 PyTorch 模块
import torch
# 导入 PyTorch 内部的函数式张量相关模块
import torch._subclasses.functional_tensor
# 导入 PyTorch 工具模块，用于处理 Python 树结构
import torch.utils._pytree as pytree
# 从 torch._C 模块中导入 DispatchKey 枚举
from torch._C import DispatchKey
# 从 torch._C._functorch 模块中导入几个函数
from torch._C._functorch import (
    _add_batch_dim,
    get_unwrapped,
    is_batchedtensor,
    maybe_get_bdim,
)
# 从 torch._functorch.utils 模块中导入 exposed_in 函数
from torch._functorch.utils import exposed_in
# 从 torch._guards 模块中导入 detect_fake_mode 函数
from torch._guards import detect_fake_mode
# 从 torch._higher_order_ops.utils 模块中导入多个函数和异常
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _set_compilation_env,
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
    UnsupportedAliasMutationException,
)
# 从 torch._ops 模块中导入 HigherOrderOperator 类
from torch._ops import HigherOrderOperator
# 从 torch._subclasses.fake_tensor 模块中导入 FakeTensorMode 类
from torch._subclasses.fake_tensor import FakeTensorMode
# 从 torch.fx.experimental.proxy_tensor 模块中导入几个函数和类
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_pre_dispatch_torch_function_mode,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
# 从 torch.fx.passes.shape_prop 模块中导入 _extract_tensor_metadata 函数
from torch.fx.passes.shape_prop import _extract_tensor_metadata
# 从 torch.utils._python_dispatch 模块中导入 _get_current_dispatch_mode 函数


@exposed_in("torch")
# 定义一个被 torch 暴露的函数 'cond'，用于条件执行 true_fn 或 false_fn
def cond(pred, true_fn, false_fn, operands):
    r"""
    Conditionally applies `true_fn` or `false_fn`.

    .. warning::
        `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and
        doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `cond` is structured control flow operator. That is, it is like a Python if-statement,
    but has restrictions on `true_fn`, `false_fn`, and `operands` that enable it to be
    capturable using torch.compile and torch.export.

    Assuming the constraints on `cond`'s arguments are met, `cond` is equivalent to the following::

        def cond(pred, true_branch, false_branch, operands):
            if pred:
                return true_branch(*operands)
            else:
                return false_branch(*operands)

    Args:
        pred (Union[bool, torch.Tensor]): A boolean expression or a tensor with one element,
          indicating which branch function to apply.

        true_fn (Callable): A callable function (a -> b) that is within the
          scope that is being traced.

        false_fn (Callable): A callable function (a -> b) that is within the
          scope that is being traced. The true branch and false branch must
          have consistent input and outputs, meaning the inputs have to be
          the same, and the outputs have to be the same type and shape.

        operands (Tuple of possibly nested dict/list/tuple of torch.Tensor): A tuple of inputs to the true/false functions.

    Example::

        def true_fn(x: torch.Tensor):
            return x.cos()
        def false_fn(x: torch.Tensor):
            return x.sin()
        return cond(x.shape[0] > 4, true_fn, false_fn, (x,))

    """
    # 函数说明文档：条件执行 true_fn 或 false_fn 函数

    # 如果 pred 是 True，则调用 true_fn，否则调用 false_fn
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)
    """
    If the Torch compiler is currently in Dynamo compiling mode,
    execute the conditional operation using the provided predicates
    and branch functions on the given operands.
    """

    if torch.compiler.is_dynamo_compiling():
        # Execute the conditional operation using pred, true_fn, false_fn, and operands
        return cond_op(pred, true_fn, false_fn, operands)

    """
    Validate the inputs for the cond function:
    - pred must be a boolean or a torch.Tensor with a single element.
    - true_fn and false_fn must be callable functions.
    - operands must be a tuple or list containing only torch.Tensor objects.
    """

    def _validate_input(pred, true_fn, false_fn, operands):
        # Check if pred is either bool, torch.Tensor, or torch.SymBool; raise error if not
        if not isinstance(pred, (bool, torch.Tensor, torch.SymBool)):
            raise RuntimeError(f"Expected pred to be bool or tensor, but got {pred}.")

        # If pred is a torch.Tensor, ensure it has exactly one element
        if isinstance(pred, torch.Tensor) and pred.numel() != 1:
            raise RuntimeError(
                f"Expected pred to be bool or single-element tensor, but got {pred}."
            )

        # Check if true_fn and false_fn are callable functions; raise error if not
        if not callable(true_fn) or not callable(false_fn):
            raise RuntimeError("Expect both branches to be callable.")

        # Ensure operands is a tuple or list and contains only torch.Tensor objects
        if not isinstance(operands, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), operands
        ):
            raise RuntimeError(
                "Expect operands to be a tuple of possibly nested dict/list/tuple that only"
                f" consists of tensor leaves, but got {operands}."
            )

    # Validate the provided inputs
    _validate_input(pred, true_fn, false_fn, operands)

    """
    Perform additional checks before compiling with Torch:
    - Ensure dynamo support is available.
    - Set compilation environment and disable caching and pre-dispatch torch functions temporarily.
    """

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("torch.cond requires dynamo support.")

    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_pre_dispatch_torch_function_mode():
                # Compile the conditional operation using the Torch compiler
                return torch.compile(cond_op, backend="eager", fullgraph=True)(
                    pred, true_fn, false_fn, operands
                )
"""
定义一个 `cond_op` 操作。
为了实现这个操作，我们需要为每个分发键提供实现。
"""
cond_op = HigherOrderOperator("cond")


def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands):
    # 断言操作数必须是张量的列表或元组
    assert isinstance(
        operands, (list, tuple)
    ), "Cond operands must be a list or tuple of tensors"
    # 断言所有操作数必须是张量
    assert all(
        isinstance(o, torch.Tensor) for o in operands
    ), "Cond operands must be a list of tensors"

    # 对 true_fn 和 false_fn 的重新进入进行函数化表示
    true_graph = reenter_make_fx(true_fn)(*operands)
    false_graph = reenter_make_fx(false_fn)(*operands)

    # 收集 true_graph 和 false_graph 的输出节点
    true_outs = []
    false_outs = []
    for node in true_graph.graph.nodes:
        if node.op == "output":
            true_outs.extend(node.args)

    for node in false_graph.graph.nodes:
        if node.op == "output":
            false_outs.extend(node.args)

    # 展开 true_outs 和 false_outs 的参数树叶
    flat_true_outs = pytree.arg_tree_leaves(*true_outs)
    flat_false_outs = pytree.arg_tree_leaves(*false_outs)

    # 检查返回的输出数量必须相同
    if len(flat_true_outs) != len(flat_false_outs):
        raise torch._dynamo.exc.CondOpArgsMismatchError(
            f"Expected to return same number of outputs but got:"
            f"\n  {true_fn.__name__} returns {len(flat_true_outs)} item(s)"
            f"\n  {false_fn.__name__} returns {len(flat_false_outs)} item(s)"
        )

    # 检查每个张量必须具有相同的元数据
    for i in range(0, len(flat_true_outs)):
        true_out = flat_true_outs[i]
        false_out = flat_false_outs[i]
        if true_out.meta["tensor_meta"] != false_out.meta["tensor_meta"]:
            raise torch._dynamo.exc.CondOpArgsMismatchError(
                f"Expected each tensor to have same metadata but got:"
                f"\n  {true_fn.__name__} returns {true_out.meta['tensor_meta']}"
                f"\n  {false_fn.__name__} returns {false_out.meta['tensor_meta']}"
            )

    # 生成唯一的图形 ID
    i, true_name = unique_graph_id(proxy_mode, prefix="true_graph")
    false_name = f"false_graph_{i}"
    # 确保 proxy_mode.tracer.root 中没有 false_name 属性
    assert not hasattr(proxy_mode.tracer.root, false_name)

    # 在代理模式的跟踪器根节点中注册 true_graph 和 false_graph
    proxy_mode.tracer.root.register_module(true_name, true_graph)
    proxy_mode.tracer.root.register_module(false_name, false_graph)

    # 构造参数元组
    args = (pred, true_graph, false_graph, operands)

    # 对参数元组中的每个元素应用 unwrap_proxy
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    # 创建代理对象，调用 func_overload 函数
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="conditional"
    )

    """
    现在，我们 *保证* 无论输出来自 true 分支还是 false 分支都无法区分。
    因此，这只是为了追踪的目的，选择 true 分支。
    """

    """
    TODO: 未备份的符号分配绝不能泄漏出来，如果想支持此功能，我们需要安排重新进入
    make_fx 使用未备份的 SymInts，并且我们需要安排两个分支之间的某种统一（但实际上不是统一；
    例如，如果一个分支返回 [u0]，另一个返回 [5]，这是可以的，但绝对不能...
    """
    # 在上下文管理器中创建一个空的 nullcontext，用于忽略未支持的新符号
    ignore_fresh_unbacked = contextlib.nullcontext()
    
    # 如果检测到 fake 模式并且 fake 模式的 shape_env 存在，
    # 则更新 ignore_fresh_unbacked 为 fake 模式的忽略新符号方法的上下文管理器
    if (fake_mode := detect_fake_mode()) and fake_mode.shape_env:
        ignore_fresh_unbacked = fake_mode.shape_env.ignore_fresh_unbacked_symbols()
    
    # 使用 ignore_fresh_unbacked 上下文管理器来执行 false_fn 函数，并传入操作数 operands
    with ignore_fresh_unbacked:
        out = false_fn(*operands)

    # 返回调用 track_tensor_tree 函数的结果
    # out 是 false_fn 函数的输出结果，out_proxy 是相关的代理对象，
    # constant 设置为 None，tracer 设置为 proxy_mode 的跟踪器
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
# 根据装饰器设置条件操作函数的实现，使用复合显式自动求导的分发键
@cond_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def cond_op_dense(pred, true_fn, false_fn, operands):
    # 获取当前的分发模式
    mode = _get_current_dispatch_mode()
    # 断言当前模式为空，因为 CPU/CUDA 键不应启用模式
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    # 如果预测条件为真，则执行 true_fn 函数并返回结果
    if pred:
        return true_fn(*operands)
    else:
        # 否则执行 false_fn 函数并返回结果
        return false_fn(*operands)


# 使用自动求导的分发键，标记自动求导未实现的 cond_op 函数
cond_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(cond_op, deferred_error=True)
)


# 根据代理 Torch 分发模式实现内部函数
@cond_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, pred, true_fn, false_fn, operands):
    # 如果模式启用追踪，则调用 trace_cond 函数进行追踪处理
    if mode.enable_tracing:
        return trace_cond(mode, cond_op, pred, true_fn, false_fn, operands)
    else:
        # 否则直接调用 cond_op 函数处理条件分支
        return cond_op(pred, true_fn, false_fn, operands)


# 根据虚拟张量模式实现条件操作函数
@cond_op.py_impl(FakeTensorMode)
def cond_fake_tensor_mode(mode, pred, true_fn, false_fn, operands):
    # 忽略此处，因为如果到达此处但未手动追踪内部图形，
    # 这意味着您打算直接重用图形，因此旧的未支持符号绑定是合适的。
    # 如果未支持的符号可以逃逸，则此策略将不起作用。
    ignore_fresh_unbacked = contextlib.nullcontext()
    # 如果模式具有形状环境，则忽略新的未支持符号
    if mode.shape_env:
        ignore_fresh_unbacked = mode.shape_env.ignore_fresh_unbacked_symbols()

    # 在给定模式和忽略新的未支持符号的上下文中执行以下操作
    with mode, ignore_fresh_unbacked:
        # 调用 true_fn 函数获取真值分支的输出
        true_outs = true_fn(*operands)
        # 将真值分支输出展平为列表
        flat_true_outs = pytree.tree_leaves(true_outs)
        # 调用 false_fn 函数获取假值分支的输出，并展平为列表
        flat_false_outs = pytree.tree_leaves(false_fn(*operands))
    
    # 如果真值分支输出和假值分支输出的长度不匹配，则引发运行时错误
    if len(flat_true_outs) != len(flat_false_outs):
        raise RuntimeError("Unmatched number of outputs from cond() branches.")

    # 对每对真值和假值输出进行比较
    for true_out, false_out in zip(flat_true_outs, flat_false_outs):
        # 提取真值和假值输出的张量元数据
        true_meta = _extract_tensor_metadata(true_out)
        false_meta = _extract_tensor_metadata(false_out)
        # 如果真值和假值输出的元数据不相同，则引发条件操作参数不匹配错误
        if true_meta != false_meta:
            raise torch._dynamo.exc.CondOpArgsMismatchError(
                f"Expected each tensor to have same metadata but got:"
                f"\n  {true_fn.__name__} returns {true_meta}"
                f"\n  {false_fn.__name__} returns {false_meta}"
            )
    
    # 返回真值分支的输出结果
    return true_outs


# 将函数功能化实现为条件操作函数
@cond_op.py_functionalize_impl
def cond_func(ctx, pred, true_fn, false_fn, inputs):
    # 使用上下文对象对输入进行解包
    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    # 使用上下文对象对预测条件进行解包
    unwrapped_pred = ctx.unwrap_tensors(pred)
    # 使用上下文对象的 redispatch_to_next 方法获取上下文管理器 m
    with ctx.redispatch_to_next() as m:
        # 使用上下文对象的 functionalize 方法对 true_fn 和 false_fn 进行功能化
        functional_true = ctx.functionalize(true_fn)
        functional_false = ctx.functionalize(false_fn)
        # 检查上下文对象是否有 mode 属性，并且检查 mode.pre_dispatch 属性
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        # 遍历功能化后的 true_fn 和 false_fn
        for branch in [functional_true, functional_false]:
            # 检查分支函数是否具有潜在的输入变异
            if _has_potential_branch_input_mutation(
                branch, unwrapped_inputs, pre_dispatch=pre_dispatch
            ):
                # 抛出异常，指示 torch.cond 的一个分支可能正在修改输入
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch might be modifying the input!"
                )
        # 遍历原始的 true_fn 和 false_fn
        for branch in [true_fn, false_fn]:
            # 检查分支函数是否具有潜在的输入别名
            if _has_potential_branch_input_alias(
                branch, unwrapped_inputs, pre_dispatch=pre_dispatch
            ):
                # 抛出异常，指示 torch.cond 的一个分支可能正在别名输入
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch might be aliasing the input!"
                )

        # 使用 cond_op 函数执行条件操作，传递解包后的条件表达式、功能化后的 true_fn 和 false_fn，以及解包后的输入数据
        cond_return = cond_op(
            unwrapped_pred, functional_true, functional_false, unwrapped_inputs
        )
        # 将 cond_return 使用上下文对象的 wrap_tensors 方法进行包装后返回
        return ctx.wrap_tensors(cond_return)
# 使用 @cond_op.py_impl 装饰器将 cond_batch_rule 函数实现注册到 functorch.TransformType.Vmap 上
@cond_op.py_impl(torch._C._functorch.TransformType.Vmap)
# 定义 cond_batch_rule 函数，用于条件批处理规则
def cond_batch_rule(interpreter, pred, true_fn, false_fn, inputs):
    # 断言输入 inputs 是列表或元组类型，并且每个元素都是 torch.Tensor 类型
    assert isinstance(
        inputs, (list, tuple)
    ), "Cond inputs must be a list or tuple of tensors"
    assert all(
        isinstance(i, torch.Tensor) for i in inputs
    ), "Cond inputs must be a list of tensors"

    # 获取未包装的 pred，如果 pred 是批处理的张量则进行处理
    pred_ = get_unwrapped(pred) if is_batchedtensor(pred) else pred

    # 对输入 tensors 进行处理，将非批处理张量解包，并获取可能的批维度
    tensors, in_dims = zip(
        *[
            (get_unwrapped(t), maybe_get_bdim(t)) if is_batchedtensor(t) else (t, None)
            for t in inputs
        ]
    )

    if is_batchedtensor(pred):
        # 如果 pred 是批处理的张量，则在 tensors 前加入 pred，并对所有输入进行批处理映射
        tensors = (pred_,) + tensors
        in_dims = (0,) + in_dims

        # 定义函数 fn，用于对输入进行真假条件函数的计算，利用 torch.vmap 对其进行批处理映射
        def fn(p, *args):
            t = true_fn(*args)
            f = false_fn(*args)
            return torch.where(p, t[0], f[0])

        # 使用 interpreter.lower() 将以下操作降低到 functorch 中
        with interpreter.lower():
            result = torch.vmap(fn, in_dims=in_dims)(*tensors)

    else:
        # 如果 pred 在此阶段已知，并且是布尔表达式或只有一个元素的张量
        # 对 true_fn 和 false_fn 进行批处理映射
        true_fn = torch.vmap(true_fn, in_dims=in_dims)
        false_fn = torch.vmap(false_fn, in_dims=in_dims)

        # 使用 interpreter.lower() 将以下操作降低到 functorch 中
        with interpreter.lower():
            result = cond_op(pred, true_fn, false_fn, tensors)

    # 如果 result 不是元组，则将其转换为元组
    if not isinstance(result, tuple):
        result = (result,)
    
    # 获取 interpreter 的层级，并为结果中的每个元素添加批处理维度
    lvl = interpreter.level()
    return tuple([_add_batch_dim(r, 0, lvl) for r in result])
```