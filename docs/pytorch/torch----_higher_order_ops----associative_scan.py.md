# `.\pytorch\torch\_higher_order_ops\associative_scan.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和函数
import functools
import itertools
from typing import Callable, List

import torch
import torch._prims_common as utils  # 导入 torch 内部常用函数模块
import torch._subclasses.functional_tensor  # 导入 torch 子类化的函数张量模块
import torch.utils._pytree as pytree  # 导入 torch 的 pytree 工具模块
from torch._C import DispatchKey  # 导入 torch 的 C 模块中的 DispatchKey
from torch._C._functorch import _add_batch_dim, get_unwrapped, maybe_get_bdim  # 导入 functorch 中的部分函数
from torch._higher_order_ops.utils import (  # 导入高阶操作的工具函数
    _set_compilation_env,
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator  # 导入高阶运算符
from torch._subclasses.fake_tensor import FakeTensorMode  # 导入虚拟张量模式
from torch.fx.experimental.proxy_tensor import (  # 导入代理张量的实验性模块
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

aten = torch._ops.ops.aten  # 设置 torch 的 aten 操作模块


def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves):
    # 确保参数数量与期望一致
    assert len(args) == 2 * num_leaves
    # 将输入的参数根据 spec 展开成树形结构的左右子树
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    # 使用给定的 combine_fn 结合左右子树
    combined = combine_fn(lhs, rhs)
    # 将组合后的结果再展开成叶子节点列表
    combined_leaves = pytree.tree_leaves(combined)
    # 确保组合后的叶子节点数量与预期一致
    assert num_leaves == len(combined_leaves)
    # 返回组合后的叶子节点列表
    return combined_leaves


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    input: pytree.PyTree,
    dim: int,
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with an associative pointwise combine function.

    .. warning::
        `torch.associative_scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    This operator requires runtime code generation and so requires support for
    ``torch.compile``. Further, only CUDA device codegen is supported at the moment.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, pointwise, and satisfy the associative property.
        input (torch.Tensor): The input tensor, or nested pytree of tensors.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(add, x, dim)

    """
    # 确保 combine_fn 是可调用的函数
    assert callable(combine_fn), "combine_fn must be a callable, but got {combine_fn}"
    # 确保 dim 是整数类型
    assert isinstance(dim, int), "dim must be an int, but got {type(dim)}"

    # 如果不处于编译状态，则使用运行时编译环境来执行
    if not torch._dynamo.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(associative_scan, fullgraph=True)(
                combine_fn, input, dim
            )

    # 将输入展开成叶子节点列表和 spec
    leaves, spec = pytree.tree_flatten(input)
    # 确保至少有一个输入叶子节点
    assert len(leaves) >= 1, "expected at least 1 input leaf"
    # 确保所有的叶子节点都是 torch.Tensor 对象
    assert all(
        isinstance(x, torch.Tensor) for x in leaves
    ), "input leaves must be a Tensor"
    
    # 获取第一个叶子节点的形状
    shape = leaves[0].shape
    
    # 获取形状的维度数
    ndim = len(shape)
    
    # 规范化 dim 参数，确保在有效范围内
    dim = utils.canonicalize_dim(ndim, dim)
    
    # 遍历除第一个外的所有叶子节点，确保它们与第一个叶子节点具有相同的形状
    for x in leaves[1:]:
        assert x.shape == shape, "All input tensors must have the same shape"
    
    # 使用 functools.partial 创建一个包装后的组合函数，用于扁平化处理
    combine_fn = functools.partial(
        wrap_combine_fn_flat, combine_fn=combine_fn, spec=spec, num_leaves=len(leaves)
    )
    
    # 调用 associative_scan_op 函数，使用组合函数和叶子节点列表进行关联扫描操作
    result_flat = associative_scan_op(combine_fn, leaves, dim)
    
    # 使用 pytree.tree_unflatten 将扁平化的结果重新组装成树形结构
    return pytree.tree_unflatten(result_flat, spec)
# 定义高阶操作符 "associative_scan_op"
associative_scan_op = HigherOrderOperator("associative_scan")

# 定义函数 trace_associative_scan，用于跟踪关联扫描操作
def trace_associative_scan(
    proxy_mode, func_overload, combine_fn: Callable, input: List[torch.Tensor], dim: int
):
    # 禁用代理模式的跟踪
    with disable_proxy_modes_tracing():
        # 创建用于样本输入的张量列表
        sample_inputs = [
            torch.full((), False, dtype=x.dtype, device=x.device)
            for x in itertools.chain(input, input)
        ]
        # 重新进入并生成组合函数图形
        combine_graph = reenter_make_fx(combine_fn)(*sample_inputs)

    # 初始化输出为 None
    outputs = None
    # 遍历组合图中的节点
    for node in combine_graph.graph.nodes:
        # 如果节点操作为 "output"
        if node.op == "output":
            # 断言输出为 None
            assert outputs is None
            # 断言节点参数长度为 1
            assert len(node.args) == 1
            # 将输出设为节点的第一个参数
            outputs = node.args[0]

    # 断言输出不为 None
    assert outputs is not None
    # 断言输出长度与输入长度相同
    assert len(outputs) == len(
        input
    ), f"expected combine_fn to return {len(input)} results but got {len(outputs)}"

    # 验证每对输入和输出张量
    for i, o in zip(input, outputs):
        # 获取输出元数据
        o_meta = o.meta["tensor_meta"]
        # 断言输出类型与输入类型相同
        assert o_meta.dtype == i.dtype, (
            f"combine_fn output type mismatch, expected {i.dtype} "
            + f"but got {o_meta.dtype}"
        )
        # 断言输出形状为标量张量
        assert (
            o_meta.shape == ()
        ), f"combine_fn must return a scalar tensor but got shape {o_meta.shape}"
        # 再次断言输出形状为标量张量
        assert (
            o_meta.shape == ()
        ), f"combine_fn must return a scalar tensor but got shape {o_meta.shape}"

    # 生成唯一的组合图形标识符
    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")
    # 在代理模式的跟踪器根节点注册组合图形
    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    # 准备参数元组
    args = (combine_graph, input, dim)
    # 使用跟踪器解包代理模式的参数
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    # 创建代理模式的函数调用代理
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )

    # 再次禁用代理模式的跟踪
    with disable_proxy_modes_tracing():
        # 对输入张量列表进行克隆操作
        out = [aten.clone(x) for x in input]

    # 返回跟踪张量树的结果
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


# 使用 CompositeExplicitAutograd 分派键定义 associative_scan_op_dense 函数
@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, input, dim):
    # 抛出未实现错误
    raise NotImplementedError("associative_scan is not implemented for eager")


# 使用 Autograd 分派键定义 associative_scan_op 的行为
associative_scan_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(associative_scan_op, deferred_error=True)
)


# 使用 ProxyTorchDispatchMode 分派键定义 associative_scan_proxy_mode 函数
@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, combine_fn, input, dim):
    # 如果启用跟踪，则调用 trace_associative_scan 函数
    if mode.enable_tracing:
        return trace_associative_scan(mode, associative_scan_op, combine_fn, input, dim)
    # 否则直接调用 associative_scan_op 函数
    else:
        return associative_scan_op(mode, associative_scan_op, combine_fn, input, dim)


# 使用 FakeTensorMode 分派键定义 assoiciative_scan_fake_tensor_mode 函数
@associative_scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, input, dim):
    # 在模式下克隆输入张量列表并返回
    with mode:
        return [x.clone() for x in input]


# 使用 py_functionalize_impl 装饰器功能化定义 associative_scan_functionalize 函数
@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, input, dim):
    # 解包上下文中的张量，并调度到下一个功能
    unwrapped_input = ctx.unwrap_tensors(input)
    with ctx.redispatch_to_next() as m:
        # 调用 associative_scan_op 函数
        ret = associative_scan_op(combine_fn, unwrapped_input, dim)
    # 将 ret 变量中的张量包装在上下文 ctx 中并返回
    return ctx.wrap_tensors(ret)
# 使用特定函数作为 associative_scan_op.py_impl 装饰器的实现，操作对象为 torch._C._functorch.TransformType.Vmap
@associative_scan_op.py_impl(torch._C._functorch.TransformType.Vmap)
# 定义 associative_scan_batch_rule 函数，接受 interpreter、input、dim 和 combine_fn 四个参数
def associative_scan_batch_rule(interpreter, input, dim, combine_fn):
    # 将 input 中的每个元素进行解包，存储在 input_ 列表中
    input_ = [get_unwrapped(x) for x in input]
    # 获取每个 input 元素的 batch 维度信息，存储在 input_bdims 列表中
    input_bdims = [maybe_get_bdim(x) for x in input]

    # 初始化 batch_size 为 None
    batch_size = None
    # 遍历 input 和 input_bdims 的元素，检查是否存在 batch 维度，并获取 batch 大小
    for inp, bdim in zip(input, input_bdims):
        if bdim is not None:
            batch_size = get_unwrapped(inp).shape[bdim]

    # 断言 batch_size 不为 None
    assert batch_size
    # 初始化 input_unwrapped 列表
    input_unwrapped = []
    # 遍历 input 和 input_bdims 的元素，将每个元素进行 unwrap 操作，如果 dim 为 None，则添加新的维度并扩展为指定的 batch_size
    # 如果 dim 不为 None，则调整维度顺序使得 batch 维度在最前面
    for x, bdim in zip(input, input_bdims):
        unwrap = get_unwrapped(x)
        if dim is None:
            unwrap = unwrap.unsqueeze(0).expand(batch_size, *x.shape)
        else:
            unwrap = unwrap.movedim(bdim, 0)
        input_unwrapped.append(unwrap)

    # 使用 combine_fn 和 input_unwrapped 执行 associative_scan_op 操作，指定维度为 dim + 1
    res = associative_scan_op(combine_fn, input_unwrapped, dim + 1)
    # 获取 interpreter 的当前层级
    lvl = interpreter.level()
    # 对 res 中的每个元素调用 _add_batch_dim 函数，添加 batch 维度，级别为 0
    return [_add_batch_dim(x, 0, lvl) for x in res]
```