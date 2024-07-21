# `.\pytorch\torch\_higher_order_ops\flex_attention.py`

```py
# mypy: allow-untyped-defs
from typing import Any, Callable, Tuple, Union  # 导入需要的类型提示模块

import torch  # 导入 PyTorch 库
import torch.utils._pytree as pytree  # 导入 PyTorch 内部的 pytree 模块
from torch._C import DispatchKey  # 导入 DispatchKey 类
from torch._higher_order_ops.utils import (  # 导入 Higher Order Ops 的一些实用函数
    _has_potential_branch_input_mutation,
    autograd_not_implemented,
    reenter_make_fx,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator  # 导入 HigherOrderOperator 类
from torch._subclasses import FakeTensorMode  # 导入 FakeTensorMode 类
from torch.fx.experimental.proxy_tensor import (  # 导入 proxy_tensor 模块的相关函数和类
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.graph_module import GraphModule  # 导入 GraphModule 类
from torch.overrides import TorchFunctionMode  # 导入 TorchFunctionMode 类


def transform_getitem_args(x: torch.Tensor, index_args) -> Tuple[Any, ...]:
    # 将索引参数 index_args 转换为元组形式，确保兼容性
    if isinstance(index_args, tuple):
        return (x, list(index_args))
    elif not isinstance(index_args, (list, tuple)):
        return (x, [index_args])
    return (x, index_args)


class TransformGetItemToIndex(TorchFunctionMode):
    # 此类用于重写 torch.Tensor 的 __torch_function__ 方法，以支持特定的索引行为
    # 当索引参数是标量张量时，不将其隐式转换为 Python 标量，而是保持为张量形式
    def __torch_function__(self, func, types, args, kwargs=None):
        if func == torch.Tensor.__getitem__:
            return torch.ops.aten.index(*transform_getitem_args(*args))
        return func(*args, **(kwargs or {}))


class FlexAttentionHOP(HigherOrderOperator):
    def __init__(self):
        super().__init__("flex_attention")  # 调用父类构造函数，指定操作符名称为 "flex_attention"

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Callable,
        sparse_kv_num_blocks: torch.Tensor,
        sparse_kv_indices: torch.Tensor,
        sparse_q_num_blocks: torch.Tensor,
        sparse_q_indices: torch.Tensor,
        SPARSE_KV_BLOCK_SIZE: int,
        SPARSE_Q_BLOCK_SIZE: int,
        *other_buffers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 检查其他缓冲区是否都是张量，若不是则引发运行时错误
        if not all(isinstance(buf, torch.Tensor) for buf in other_buffers):
            raise RuntimeError("Other buffers must be tensors.")
        # 调用父类的 __call__ 方法，传递所有参数
        return super().__call__(
            query,
            key,
            value,
            score_mod,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers,
        )


flex_attention = FlexAttentionHOP()  # 创建 FlexAttentionHOP 类的实例对象
flex_attention.__module__ = "torch.ops.higher_order"  # 设置模块名称为 "torch.ops.higher_order"


class FlexAttentionBackwardHOP(HigherOrderOperator):
    def __init__(self):
        super().__init__("flex_attention_backward")  # 调用父类构造函数，指定操作符名称为 "flex_attention_backward"
    # 定义一个特殊方法 __call__，用于调用对象实例，接受多个参数：
    #   - query: 查询张量
    #   - key: 键张量
    #   - value: 值张量
    #   - out: 输出张量
    #   - logsumexp: 对数求和指数张量
    #   - grad_out: 梯度输出张量
    #   - fw_graph: 前向图模块或可调用对象
    #   - joint_graph: 联合图模块
    #   - sparse_kv_num_blocks: 稀疏键值对数块张量
    #   - sparse_kv_indices: 稀疏键值索引张量
    #   - sparse_q_num_blocks: 稀疏查询数块张量
    #   - sparse_q_indices: 稀疏查询索引张量
    #   - SPARSE_KV_BLOCK_SIZE: 稀疏键值块大小（整数）
    #   - SPARSE_Q_BLOCK_SIZE: 稀疏查询块大小（整数）
    #   - *other_buffers: 其他缓冲区张量（可变数量）
    # 如果其他缓冲区不是全部都是张量，抛出运行时错误
    if not all(isinstance(buf, torch.Tensor) for buf in other_buffers):
        raise RuntimeError("Other buffers must be tensors.")
    # 调用父类的 __call__ 方法，并返回其结果，传递所有接收到的参数
    return super().__call__(
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        fw_graph,
        joint_graph,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE,
        *other_buffers,
    )
# 创建一个 FlexAttentionBackwardHOP 的实例对象
flex_attention_backward = FlexAttentionBackwardHOP()
# 将实例对象的 __module__ 属性设置为 "torch.ops.higher_order"
flex_attention_backward.__module__ = "torch.ops.higher_order"

# 定义 math_attention 函数，执行注意力机制计算
def math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager implementation

    This implementation uses vmap to vectorize the score_mod function over the batch, head, m, and n dimensions.
    We then apply the vectorized score_mod function to the scores matrix. Each wrap of vmap applies one of the
    batch, head, m, or n dimensions. We need to apply vmap 4 times to vectorized over all 4 dimensions.

    Args:
        query: The query tensor
        key: The key tensor
        value: The value tensor
        score_mod: The score_mod function
        other_buffers: Other buffers that are passed to the score_mod function
    """
    # 确定工作精度，如果 query 的数据类型是 torch.float64，则工作精度为 torch.float64，否则为 torch.float32
    working_precision = torch.float64 if query.dtype == torch.float64 else torch.float32

    # 计算注意力分数矩阵 scores，将其转换为指定的工作精度
    scores = (query @ key.transpose(-2, -1)).to(dtype=working_precision)

    # 创建大小与 scores 大小相同的索引张量 b, h, m, n，分别表示 batch, head, m, n 维度的索引
    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    # 使用 torch.vmap 对 score_mod 函数进行向量化，分别在 batch, head, m, n 四个维度上应用 score_mod 函数
    # in_dim_buffers 为其它参数的缓冲区
    in_dim_buffers = (None,) * len(other_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None) + in_dim_buffers)

    # 使用 TransformGetItemToIndex 上下文管理器对 scores 进行变换处理
    with TransformGetItemToIndex():
        # 应用 score_mod 函数处理 scores，同时将结果转换为指定的工作精度
        scores = score_mod(scores, b, h, m, n, *other_buffers).to(working_precision)

    # 对 scores 的最后一个维度进行 softmax 操作
    scores = scores.softmax(dim=-1)

    # 计算返回值中的注意力加权和与 logsumexp
    attention_weighted_values = scores.to(query.dtype) @ value
    logsumexp = scores.logsumexp(dim=-1)

    return attention_weighted_values, logsumexp


# 将函数 math_attention 标记为使用 DispatchKey.CompositeExplicitAutograd 的 sdpa_dense 的实现
@flex_attention.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 使用自定义的注意力机制函数进行注意力计算
    out, lse = math_attention(
        query,                  # 查询向量
        key,                    # 键向量
        value,                  # 值向量
        score_mod,              # 分数修正
        sparse_kv_num_blocks,   # 稀疏键值对的块数
        sparse_kv_indices,      # 稀疏键值对的索引
        sparse_q_num_blocks,    # 稀疏查询的块数
        sparse_q_indices,       # 稀疏查询的索引
        SPARSE_KV_BLOCK_SIZE,   # 稀疏键值对的块大小
        SPARSE_Q_BLOCK_SIZE,    # 稀疏查询的块大小
        *other_buffers,         # 其他缓冲区（可变参数）
    )
    # 将输出张量变换为连续内存中的连续张量
    out = out.contiguous()
    # 返回计算后的输出张量和注意力得分
    return out, lse
# 定义一个函数，用于追踪 flex_attention 操作符，使用给定的 score_mod 函数和其他缓冲区。
# 函数签名指定了输入和输出的类型信息。

"""Traces the flex_attention operator with the given score_mod function and other_buffers.

Trace SDPA will call make_fx with "fake" example vals and then trace the score_mod function
This will produce a GraphModule that will be stored on the root tracer as "sdpa_score". We
access this graph module in inductor to inline the score_mod function to the triton template.
"""
# 上述注释解释了函数的整体目的和如何使用它进行模型追踪的细节。

example_out = flex_attention(
    query,
    key,
    value,
    score_mod,
    sparse_kv_num_blocks,
    sparse_kv_indices,
    sparse_q_num_blocks,
    sparse_q_indices,
    SPARSE_KV_BLOCK_SIZE,
    SPARSE_Q_BLOCK_SIZE,
    *other_buffers,
)
# 调用 flex_attention 函数，生成一个示例输出 example_out，用于后续的模型追踪和优化。

example_vals = [
    torch.zeros((), dtype=query.dtype, requires_grad=query.requires_grad)
] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
# 创建示例值 example_vals，用于模型追踪中的 make_fx 函数。

with TransformGetItemToIndex():
    score_graph = reenter_make_fx(score_mod)(*example_vals, *other_buffers)
# 使用 TransformGetItemToIndex 上下文管理器，调用 reenter_make_fx 函数生成 score_graph，该函数将追踪 score_mod 函数并生成一个图形模块。

qualname = proxy_mode.tracer.get_fresh_qualname("sdpa_score")
proxy_mode.tracer.root.register_module(qualname, score_graph)
# 获取一个新的合格名称 qualname，并将 score_graph 注册到代理模式的跟踪器根中，以便后续在 triton 模板中内联 score_mod 函数。

node_args = (
    query,
    key,
    value,
    score_graph,
    sparse_kv_num_blocks,
    sparse_kv_indices,
    sparse_q_num_blocks,
    sparse_q_indices,
    SPARSE_KV_BLOCK_SIZE,
    SPARSE_Q_BLOCK_SIZE,
    *other_buffers,
)
# 准备将传递给 flex_attention 函数的参数 node_args，包括 query、key、value、score_graph 等。

proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
# 使用 pytree.tree_map 将 node_args 中的参数解包，并将其传递给代理模式的跟踪器，以便跟踪和处理这些参数。

out_proxy = proxy_mode.tracer.create_proxy(
    "call_function", flex_attention, proxy_args, {}
)
# 使用代理模式的跟踪器创建一个代理对象 out_proxy，表示 flex_attention 函数的调用。

return track_tensor_tree(
    example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
)
# 返回通过 track_tensor_tree 处理后的输出结果，包括示例输出 example_out 和代理对象 out_proxy。
    # 如果模式启用追踪，则调用追踪灵活注意力函数
    if mode.enable_tracing:
        return trace_flex_attention(
            mode,                   # 模式参数
            query,                  # 查询张量
            key,                    # 键张量
            value,                  # 值张量
            score_mod,              # 分数修改器
            sparse_kv_num_blocks,   # 稀疏键值对数块
            sparse_kv_indices,      # 稀疏键值索引
            sparse_q_num_blocks,    # 稀疏查询数块
            sparse_q_indices,       # 稀疏查询索引
            SPARSE_KV_BLOCK_SIZE,   # 稀疏键值块大小
            SPARSE_Q_BLOCK_SIZE,    # 稀疏查询块大小
            *other_buffers,         # 其他缓冲区（可变数量）
        )
    else:
        # 否则调用普通的灵活注意力函数
        return flex_attention(
            query,                  # 查询张量
            key,                    # 键张量
            value,                  # 值张量
            score_mod,              # 分数修改器
            sparse_kv_num_blocks,   # 稀疏键值对数块
            sparse_kv_indices,      # 稀疏键值索引
            sparse_q_num_blocks,    # 稀疏查询数块
            sparse_q_indices,       # 稀疏查询索引
            SPARSE_KV_BLOCK_SIZE,   # 稀疏键值块大小
            SPARSE_Q_BLOCK_SIZE,    # 稀疏查询块大小
            *other_buffers,         # 其他缓冲区（可变数量）
        )
# 定义了一个装饰器函数，用于将下面的函数实现注册为灵活注意力操作的一部分
@flex_attention.py_functionalize_impl
# 定义了灵活注意力操作的函数，接受多个参数并返回两个张量
def flex_attention_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,  # 上下文对象，用于处理张量的功能化API
    query: torch.Tensor,  # 查询张量
    key: torch.Tensor,  # 键张量
    value: torch.Tensor,  # 值张量
    score_mod: Callable,  # 分数修改函数，接受输入并返回修改后的分数张量
    sparse_kv_num_blocks: torch.Tensor,  # 稀疏键值块的数量张量
    sparse_kv_indices: torch.Tensor,  # 稀疏键值索引张量
    sparse_q_num_blocks: torch.Tensor,  # 稀疏查询块的数量张量
    sparse_q_indices: torch.Tensor,  # 稀疏查询索引张量
    SPARSE_KV_BLOCK_SIZE: int,  # 稀疏键值块的大小
    SPARSE_Q_BLOCK_SIZE: int,  # 稀疏查询块的大小
    *other_buffers: torch.Tensor,  # 可变数量的其他缓冲区张量
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Defines the functionalization rules for the flex_attention operator.

    Write now we are unwrapping each tensor and then redispatching to the next, however we want to
    guard against any mutations in the score_mod function, to the other_buffers since those
    are free variables.
    """
    # 使用上下文对象的方法对输入张量进行解封装
    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    sparse_kv_num_blocks_unwrapped = ctx.unwrap_tensors(sparse_kv_num_blocks)
    sparse_kv_indices_unwrapped = ctx.unwrap_tensors(sparse_kv_indices)
    sparse_q_num_blocks_unwrapped = ctx.unwrap_tensors(sparse_q_num_blocks)
    sparse_q_indices_unwrapped = ctx.unwrap_tensors(sparse_q_indices)
    other_buffers_unwrapped = ctx.unwrap_tensors(other_buffers)

    # 对解封装后的张量类型进行断言，确保它们的类型符合预期
    assert isinstance(query_unwrapped, torch.Tensor)
    assert isinstance(key_unwrapped, torch.Tensor)
    assert isinstance(value_unwrapped, torch.Tensor)
    assert isinstance(sparse_kv_num_blocks_unwrapped, torch.Tensor)
    assert isinstance(sparse_kv_indices_unwrapped, torch.Tensor)
    assert isinstance(sparse_q_num_blocks_unwrapped, torch.Tensor)
    assert isinstance(sparse_q_indices_unwrapped, torch.Tensor)
    assert isinstance(other_buffers_unwrapped, tuple)
    assert all(isinstance(item, torch.Tensor) for item in other_buffers_unwrapped)

    # 创建一个示例值列表，包含了各种类型的零张量和其他缓冲区的张量
    example_vals = (
        [torch.zeros((), dtype=query.dtype)]  # 使用查询张量的数据类型创建一个零张量
        + [torch.zeros((), dtype=torch.int) for _ in range(4)]  # 创建四个整型的零张量
        + list(other_buffers_unwrapped)  # 添加所有其他缓冲区的张量
    )
    # 使用上下文管理器 ctx.redispatch_to_next()，获取上下文管理器 m
    with ctx.redispatch_to_next() as m:
        # 将 score_mod 函数功能化，返回 functional_score_mod
        functional_score_mod = ctx.functionalize(score_mod)
        # 检查 ctx 是否有 mode 属性，并获取 mode.pre_dispatch 属性
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        # 使用 TransformGetItemToIndex 上下文管理器，执行下面的代码块
        with TransformGetItemToIndex():
            # 检测 functional_score_mod、example_vals、pre_dispatch 是否会导致分支输入突变
            mutates = _has_potential_branch_input_mutation(
                functional_score_mod, example_vals, pre_dispatch
            )
        # 如果发现了突变，抛出异常 UnsupportedAliasMutationException
        if mutates:
            raise UnsupportedAliasMutationException("Mutations detected in score_mod")

        # 调用 flex_attention 函数，传入多个参数执行注意力机制计算
        out = flex_attention(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            functional_score_mod,
            sparse_kv_num_blocks_unwrapped,
            sparse_kv_indices_unwrapped,
            sparse_q_num_blocks_unwrapped,
            sparse_q_indices_unwrapped,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers_unwrapped,
        )
    # 将计算结果 out 使用 ctx.wrap_tensors() 进行包装，返回结果
    return ctx.wrap_tensors(out)  # type: ignore[return-value, arg-type]
# 使用装饰器将函数注册为特定模式下的实现
@flex_attention.py_impl(FakeTensorMode)
def flex_attention_fake_tensor_mode(
    mode: FakeTensorMode,  # 指定模式，用于灵活的注意力机制
    query: torch.Tensor,   # 查询张量
    key: torch.Tensor,     # 键张量
    value: torch.Tensor,   # 值张量
    score_mod: Callable,   # 评分函数
    sparse_kv_num_blocks: torch.Tensor,  # 稀疏键值对应块的数量
    sparse_kv_indices: torch.Tensor,     # 稀疏键值对应的索引
    sparse_q_num_blocks: torch.Tensor,   # 稀疏查询对应块的数量
    sparse_q_indices: torch.Tensor,      # 稀疏查询对应的索引
    SPARSE_KV_BLOCK_SIZE: int,   # 稀疏键值对块大小
    SPARSE_Q_BLOCK_SIZE: int,    # 稀疏查询块大小
    *other_buffers: Tuple[torch.Tensor, ...],  # 其他缓冲区
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 使用指定的模式上下文执行以下代码块
    with mode:
        # 获取查询张量的形状信息
        batch_size, num_heads, seq_len_q, _ = query.shape
        # 创建一个新的空张量，用于存储 logsumexp 结果
        logsumexp = query.new_empty(
            batch_size, num_heads, seq_len_q, dtype=torch.float32
        )
        # 返回与查询张量形状相同的空张量和 logsumexp 结果
        return torch.empty_like(query), logsumexp


# ---------------------------- Autograd Implementation ----------------------------
def create_fw_bw_graph(score_mod, index_values, other_buffers):
    # See Note:[HOP create fw_bw graph]

    # 所有这些导入必须放在此处，以避免循环依赖
    from torch._dispatch.python import suspend_functionalization
    from torch._functorch.aot_autograd import AOTConfig, create_joint
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

    # 创建一个虚拟的 AOT 配置对象，配置自动微分过程的相关参数
    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # 前向编译器，目前未指定
        bw_compiler=None,  # 反向编译器，目前未指定
        partition_fn=None,  # 分区函数，目前未指定
        decompositions={},  # 分解方法，目前为空
        num_params_buffers=0,  # 参数缓冲区数量，目前为零
        aot_id=0,  # AOT ID，目前为零
        keep_inference_input_mutations=False,  # 是否保留推断输入变化，目前为假
    )
    with suspend_functionalization(), disable_functional_mode():
        # 暂时禁用功能化和功能模式

        with disable_proxy_modes_tracing():
            # 禁用代理模式追踪

            def _from_fun(t):
                # 定义一个函数 _from_fun，用于创建一个具有指定属性的空张量
                return torch.empty_strided(
                    t.size(),
                    t.stride(),
                    device=t.device,
                    dtype=t.dtype,
                    requires_grad=t.requires_grad,
                )

            # 如果在默认的编译器后端（"eager"）下运行这个流程
            # 则将用户输入转换为假张量，以避免进行任何实际计算。
            from torch._guards import detect_fake_mode

            # 检测是否处于假张量模式
            fake_mode = detect_fake_mode(index_values)
            if fake_mode is None:
                fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            with fake_mode:
                # 在假张量模式下运行以下代码块

                # 将索引值列表 index_values 映射为具有指定属性的空张量列表 unwrapped_score_mod_indexes
                unwrapped_score_mod_indexes = pytree.tree_map(_from_fun, index_values)
                # 将其他缓冲区列表 other_buffers 映射为具有指定属性的空张量列表 unwrapped_other_buffers
                unwrapped_other_buffers = pytree.tree_map(_from_fun, other_buffers)

            # 断言 unwrapped_score_mod_indexes 和 unwrapped_other_buffers 中的所有元素均为 FakeTensor 类型
            assert all(isinstance(t, FakeTensor) for t in unwrapped_score_mod_indexes)
            assert all(isinstance(t, FakeTensor) for t in unwrapped_other_buffers)

            # 对 score_mod 函数应用到 unwrapped_score_mod_indexes 和 unwrapped_other_buffers 的映射，
            # 得到 example_flat_out，确保输出是张量类型
            example_flat_out = pytree.tree_map(
                _from_fun,
                score_mod(*unwrapped_score_mod_indexes, *unwrapped_other_buffers),
            )
            if not isinstance(example_flat_out, torch.Tensor):
                raise RuntimeError(
                    "Expected output of score_mod to be a tensor."
                    f"Got type {type(example_flat_out)}."
                )
            # 根据 example_flat_out 创建一个与其具有相同属性的空张量 example_grad
            example_grad = _from_fun(example_flat_out)

        # 定义一个 joint_f 函数，用于生成前向计算和梯度的联合函数
        def joint_f(score, b, h, m, n, example_grad, *other_buffers):
            def fw_with_masks(*args):
                # 执行 score_mod 函数并返回输出和是否需要梯度的标志
                fw_out = score_mod(*args)
                out_requires_grad = fw_out.requires_grad
                return ((fw_out,), (out_requires_grad,))

            # 创建一个联合函数 joint，用于前向计算和梯度
            joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
            args = [score, b, h, m, n] + list(other_buffers)
            optional_grad = [example_grad] if example_grad.requires_grad else []
            # 调用联合函数 joint，返回计算得到的梯度 grads
            _, grads = joint(args, optional_grad)

            return grads

        # 使用 make_fx 将 joint_f 转换为一个 FX 函数图，输入包括 unwrapped_score_mod_indexes、example_grad 和 unwrapped_other_buffers
        joint_graph = make_fx(joint_f)(
            *unwrapped_score_mod_indexes, example_grad, *unwrapped_other_buffers
        )
        # 返回 score_mod 函数和生成的函数图 joint_graph
        return score_mod, joint_graph
class FlexAttentionAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        fw_graph,
        joint_graph,
        sparse_kv_num_blocks: torch.Tensor,
        sparse_kv_indices: torch.Tensor,
        sparse_q_num_blocks: torch.Tensor,
        sparse_q_indices: torch.Tensor,
        SPARSE_KV_BLOCK_SIZE: int,
        SPARSE_Q_BLOCK_SIZE: int,
        *other_buffers,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 检查是否有任何缓冲区需要梯度
        any_buffer_requires_grad = any(buffer.requires_grad for buffer in other_buffers)
        assert (
            not any_buffer_requires_grad
        ), "Captured buffers that require grad are not yet supported."
        # 保存一些上下文信息到 ctx 中
        ctx._fw_graph = fw_graph
        ctx._joint_graph = joint_graph
        ctx._SPARSE_KV_BLOCK_SIZE = SPARSE_KV_BLOCK_SIZE
        ctx._SPARSE_Q_BLOCK_SIZE = SPARSE_Q_BLOCK_SIZE
        # 进入下层自动分发层
        with torch._C._AutoDispatchBelowAutograd():
            # 调用灵活注意力机制的前向计算
            out, logsumexp = flex_attention(
                query,
                key,
                value,
                fw_graph,
                sparse_kv_num_blocks,
                sparse_kv_indices,
                sparse_q_num_blocks,
                sparse_q_indices,
                SPARSE_KV_BLOCK_SIZE,
                SPARSE_Q_BLOCK_SIZE,
                *other_buffers,
            )

        # 保存需要用于反向传播的张量
        ctx.save_for_backward(
            query,
            key,
            value,
            out,
            logsumexp,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            *other_buffers,
        )
        return out, logsumexp

    @staticmethod
    def backward(ctx, grad_out, logsumexp_grad):
        # 恢复保存的张量
        fw_args = ctx.saved_tensors
        (
            query,
            key,
            value,
            out,
            logsumexp,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            *other_buffers,
        ) = fw_args
        fw_graph = ctx._fw_graph
        joint_graph = ctx._joint_graph
        SPARSE_KV_BLOCK_SIZE = ctx._SPARSE_KV_BLOCK_SIZE
        SPARSE_Q_BLOCK_SIZE = ctx._SPARSE_Q_BLOCK_SIZE
        # 在前向计算中已经断言了其他缓冲区不需要梯度
        none_grads = [None] * (8 + len(other_buffers))
        # 调用灵活注意力机制的反向计算
        grad_query, grad_key, grad_value = flex_attention_backward(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            fw_graph,
            joint_graph,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers,
        )
        return grad_query, grad_key, grad_value, *none_grads


@flex_attention.py_impl(DispatchKey.Autograd)
def flex_attention_autograd(
    # 定义函数参数，表示注意力机制中的查询张量
    query: torch.Tensor,
    # 定义函数参数，表示注意力机制中的键张量
    key: torch.Tensor,
    # 定义函数参数，表示注意力机制中的值张量
    value: torch.Tensor,
    # 定义函数参数，表示用于修改分数的回调函数
    score_mod: Callable,
    # 定义函数参数，表示稀疏键值张量的块数
    sparse_kv_num_blocks: torch.Tensor,
    # 定义函数参数，表示稀疏键值张量的索引
    sparse_kv_indices: torch.Tensor,
    # 定义函数参数，表示稀疏查询张量的块数
    sparse_q_num_blocks: torch.Tensor,
    # 定义函数参数，表示稀疏查询张量的索引
    sparse_q_indices: torch.Tensor,
    # 定义函数参数，表示稀疏键值对的块大小
    SPARSE_KV_BLOCK_SIZE: int,
    # 定义函数参数，表示稀疏查询的块大小
    SPARSE_Q_BLOCK_SIZE: int,
    # 定义函数参数，表示其他缓冲区的元组
    *other_buffers: Tuple[torch.Tensor, ...],
# 定义一个函数sdpa_dense_backward，实现SDPA（Scaled Dot-Product Attention）的反向传播
@flex_attention_backward.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    fw_graph: Callable,  # fw_graph参数类型为Callable，表示可以调用的函数类型
    joint_graph: Callable,  # joint_graph参数类型为Callable，表示可以调用的函数类型
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,  # other_buffers为可变长度的torch.Tensor参数列表
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 根据query的数据类型确定工作精度为torch.float64或torch.float32
    working_precision = torch.float64 if query.dtype == torch.float64 else torch.float32
    
    # 计算scores，使用query与key的转置进行点积
    scores = (query @ key.transpose(-2, -1)).to(working_precision)
    
    # 根据scores的维度大小，在设备上创建张量b、h、m、n
    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    # 初始化in_dim_buffers为None的元组，长度与other_buffers相同
    in_dim_buffers = (None,) * len(other_buffers)
    
    # 使用torch.vmap对fw_graph进行映射操作，参数的in_dims用于指定映射维度
    score_mod = torch.vmap(fw_graph, in_dims=(0, None, None, None, 0) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None) + in_dim_buffers)
    
    # 使用TransformGetItemToIndex上下文管理器，转换scores为post_mod_scores
    with TransformGetItemToIndex():
        post_mod_scores = score_mod(scores, b, h, m, n, *other_buffers).to(
            working_precision
        )
    
    # 计算softmax_scores，通过减去logsumexp的展开形式获得
    softmax_scores = torch.exp(post_mod_scores - logsumexp.unsqueeze(-1))
    
    # 计算grad_value，使用softmax_scores和grad_out的转置进行矩阵乘法
    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out
    
    # 计算grad_softmax_scores，使用grad_out和value的转置进行矩阵乘法
    grad_softmax_scores = grad_out @ value.transpose(-2, -1)
    
    # 计算sum_scores，通过对out与grad_out的点积在最后一个维度上进行求和
    sum_scores = torch.sum(out * grad_out, -1, keepdim=True)
    grad_score_mod = softmax_scores * (grad_softmax_scores - sum_scores)
    # 计算梯度相对于得分的修改量，使用了 softmax_scores、grad_softmax_scores 和 sum_scores

    # 定义一些维度相关的缓冲区
    in_dim_buffers = (None,) * len(other_buffers)
    # 输出维度设置，初始设置第一个维度为0，其余为None，与其他缓冲区的长度对应
    out_dims = [0, None, None, None, None] + [None] * len(other_buffers)

    # 对 joint_graph 函数进行批处理映射，设置输入和输出的维度
    joint_score_mod = torch.vmap(
        joint_graph,
        in_dims=(0, None, None, None, 0, 0) + in_dim_buffers,
        out_dims=out_dims,
    )

    # 进行多次批处理映射，依次设置不同的输入维度
    joint_score_mod = torch.vmap(
        joint_score_mod,
        in_dims=(0, None, None, 0, None, 0) + in_dim_buffers,
        out_dims=out_dims,
    )
    joint_score_mod = torch.vmap(
        joint_score_mod,
        in_dims=(0, None, 0, None, None, 0) + in_dim_buffers,
        out_dims=out_dims,
    )
    joint_score_mod = torch.vmap(
        joint_score_mod,
        in_dims=(0, 0, None, None, None, 0) + in_dim_buffers,
        out_dims=out_dims,
    )

    # 使用 TransformGetItemToIndex 上下文管理器
    with TransformGetItemToIndex():
        # 调用 joint_score_mod 函数，计算得分、b、h、m、n、grad_score_mod 和其他缓冲区的梯度得分
        grad_scores, *_ = joint_score_mod(
            scores, b, h, m, n, grad_score_mod, *other_buffers
        )
    # 将 grad_scores 转换为与 query 相同的数据类型
    grad_scores = grad_scores.to(query.dtype)

    # 计算 grad_query 和 grad_key
    grad_query = grad_scores @ key
    grad_key = grad_scores.transpose(-2, -1) @ query

    # 返回连续存储的 grad_query、grad_key 和 grad_value
    return grad_query.contiguous(), grad_key.contiguous(), grad_value.contiguous()
# 定义一个函数，用于反向传播灵活注意力机制的计算。此函数是通过 Torch 的代理调度模式执行的。
def trace_flex_attention_backward(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """We already have the forward graph and joint graph from the forward pass, so we create a proxy attach both graphs"""
    # 调用灵活注意力机制的反向传播函数，获取示例输出
    example_out = flex_attention_backward(
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        fw_graph,
        joint_graph,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE,
        *other_buffers,
    )

    # 准备用于前向和反向图的示例值
    fw_example_vals = [
        torch.zeros((), dtype=query.dtype, requires_grad=query.requires_grad)
    ] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
    bw_example_vals = fw_example_vals + [torch.zeros((), dtype=query.dtype)]

    # 使用上下文管理器，将获取项目转换为索引
    with TransformGetItemToIndex():
        # 重新构造前向图，并使用示例值和其他缓冲区参数
        fw_graph = reenter_make_fx(fw_graph)(*fw_example_vals, *other_buffers)
        # 重新构造联合图，并使用示例值和其他缓冲区参数
        joint_graph = reenter_make_fx(joint_graph)(*bw_example_vals, *other_buffers)

    # 在代理模式的追踪器根节点注册前向图和联合图
    proxy_mode.tracer.root.register_module("fw_graph", fw_graph)
    proxy_mode.tracer.root.register_module("joint_graph", joint_graph)

    # 准备节点参数，用于创建代理参数
    node_args = (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        fw_graph,
        joint_graph,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE,
        *other_buffers,
    )

    # 使用代理模式的追踪器解包节点参数，形成代理参数
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)

    # 创建代理，代表调用函数 flex_attention_backward
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        flex_attention_backward,
        proxy_args,
        {},
        name="flex_attention_backward",
    )

    # 返回通过追踪器跟踪的张量树
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


# 定义一个装饰器函数，用于将 ProxyTorchDispatchMode 作为参数传递给灵活注意力机制的反向传播函数实现
@flex_attention_backward.py_impl(ProxyTorchDispatchMode)
def flex_attention_backward_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE,
    SPARSE_Q_BLOCK_SIZE,
):
    # 这个函数实际上并未提供实现细节，而是通过装饰器指定了使用 ProxyTorchDispatchMode 这个调度模式
    pass  # 仅用于装饰器声明，实际逻辑由上面的 trace_flex_attention_backward 函数处理
    *other_buffers: torch.Tensor,
# 定义一个函数 flex_attention_backward_functionalize，用于实现 flex_attention 操作的功能化规则
@flex_attention_backward.py_functionalize_impl
def flex_attention_backward_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    sparse_kv_num_blocks: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_q_num_blocks: torch.Tensor,
    sparse_q_indices: torch.Tensor,
    SPARSE_KV_BLOCK_SIZE: int,
    SPARSE_Q_BLOCK_SIZE: int,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """定义 flex_attention 操作的功能化规则。

    现在我们解包每个张量，然后重新调度到下一个函数，
    因为我们知道前向得分模块函数被确保不会对 other_buffers 进行突变，
    所以我们跳过了突变检查直接进行重新调度。
    """
    # 使用上下文对象解包每个张量
    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    out_unwrapped = ctx.unwrap_tensors(out)
    logsumexp_unwrapped = ctx.unwrap_tensors(logsumexp)
    grad_out_unwrapped = ctx.unwrap_tensors(grad_out)
    sparse_kv_num_blocks_unwrapped = ctx.unwrap_tensors(sparse_kv_num_blocks)
    sparse_kv_indices_unwrapped = ctx.unwrap_tensors(sparse_kv_indices)
    sparse_q_num_blocks_unwrapped = ctx.unwrap_tensors(sparse_q_num_blocks)
    sparse_q_indices_unwrapped = ctx.unwrap_tensors(sparse_q_indices)
    other_buffers_unwrapped = ctx.unwrap_tensors(other_buffers)

    # 确保解包后的对象是 torch.Tensor 类型
    assert isinstance(query_unwrapped, torch.Tensor)
    assert isinstance(key_unwrapped, torch.Tensor)
    assert isinstance(value_unwrapped, torch.Tensor)
    assert isinstance(out_unwrapped, torch.Tensor)
    # 断言确保 logsumexp_unwrapped 是 torch.Tensor 类型
    assert isinstance(logsumexp_unwrapped, torch.Tensor)
    # 断言确保 grad_out_unwrapped 是 torch.Tensor 类型
    assert isinstance(grad_out_unwrapped, torch.Tensor)
    # 断言确保 sparse_kv_num_blocks_unwrapped 是 torch.Tensor 类型
    assert isinstance(sparse_kv_num_blocks_unwrapped, torch.Tensor)
    # 断言确保 sparse_kv_indices_unwrapped 是 torch.Tensor 类型
    assert isinstance(sparse_kv_indices_unwrapped, torch.Tensor)
    # 断言确保 sparse_q_num_blocks_unwrapped 是 torch.Tensor 类型
    assert isinstance(sparse_q_num_blocks_unwrapped, torch.Tensor)
    # 断言确保 sparse_q_indices_unwrapped 是 torch.Tensor 类型
    assert isinstance(sparse_q_indices_unwrapped, torch.Tensor)
    # 断言确保 other_buffers_unwrapped 是一个元组
    assert isinstance(other_buffers_unwrapped, tuple)
    # 断言确保 other_buffers_unwrapped 中的所有元素都是 torch.Tensor 类型
    assert all(isinstance(item, torch.Tensor) for item in other_buffers_unwrapped)

    # 使用上下文管理器，重新分发到下一个模块
    with ctx.redispatch_to_next() as m:
        # 使用上下文对象的 functionalize 方法将 fw_graph 转化为函数图
        functional_fw_graph = ctx.functionalize(fw_graph)
        # 使用上下文对象的 functionalize 方法将 joint_graph 转化为函数图
        functional_joint_graph = ctx.functionalize(joint_graph)

        # 调用 flex_attention_backward 函数进行反向传播计算
        grad_query, grad_key, grad_value = flex_attention_backward(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            out_unwrapped,
            logsumexp_unwrapped,
            grad_out_unwrapped,
            functional_fw_graph,  # 这里指定参数类型为函数图对象，忽略类型检查
            functional_joint_graph,  # 这里指定参数类型为函数图对象，忽略类型检查
            sparse_kv_num_blocks_unwrapped,
            sparse_kv_indices_unwrapped,
            sparse_q_num_blocks_unwrapped,
            sparse_q_indices_unwrapped,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
            *other_buffers_unwrapped,
        )

    # 将计算得到的梯度数据包装成张量，并返回给上层上下文
    return ctx.wrap_tensors((grad_query, grad_key, grad_value))  # 这里指定返回值类型为张量的包装对象，忽略类型检查
# 根据 `flex_attention_backward.py_impl` 装饰器定义一个函数，实现了在 `FakeTensorMode` 模式下的灵活注意力机制的反向传播。
@flex_attention_backward.py_impl(FakeTensorMode)
def flex_attention_backward_fake_tensor_mode(
    mode: FakeTensorMode,  # 定义输入参数 mode，类型为 FakeTensorMode
    query: torch.Tensor,   # 定义输入参数 query，类型为 torch.Tensor
    key: torch.Tensor,     # 定义输入参数 key，类型为 torch.Tensor
    value: torch.Tensor,   # 定义输入参数 value，类型为 torch.Tensor
    out: torch.Tensor,     # 定义输入参数 out，类型为 torch.Tensor
    logsumexp: torch.Tensor,  # 定义输入参数 logsumexp，类型为 torch.Tensor
    grad_out: torch.Tensor,   # 定义输入参数 grad_out，类型为 torch.Tensor
    fw_graph: Union[Callable, GraphModule],  # 定义输入参数 fw_graph，类型为 Callable 或 GraphModule
    joint_graph: GraphModule,  # 定义输入参数 joint_graph，类型为 GraphModule
    sparse_kv_num_blocks: torch.Tensor,  # 定义输入参数 sparse_kv_num_blocks，类型为 torch.Tensor
    sparse_kv_indices: torch.Tensor,     # 定义输入参数 sparse_kv_indices，类型为 torch.Tensor
    sparse_q_num_blocks: torch.Tensor,   # 定义输入参数 sparse_q_num_blocks，类型为 torch.Tensor
    sparse_q_indices: torch.Tensor,      # 定义输入参数 sparse_q_indices，类型为 torch.Tensor
    SPARSE_KV_BLOCK_SIZE: int,   # 定义输入参数 SPARSE_KV_BLOCK_SIZE，类型为 int
    SPARSE_Q_BLOCK_SIZE: int,    # 定义输入参数 SPARSE_Q_BLOCK_SIZE，类型为 int
    *other_buffers: torch.Tensor,  # 定义可变长度参数 other_buffers，类型为 torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回值类型为元组，包含三个 torch.Tensor 对象
    # 进入指定的 mode 上下文环境
    with mode:
        # 初始化与 query 张量相同大小的 grad_query 张量
        grad_query = torch.empty_like(query)
        # 初始化与 key 张量相同大小的 grad_key 张量
        grad_key = torch.empty_like(key)
        # 初始化与 value 张量相同大小的 grad_value 张量
        grad_value = torch.empty_like(value)
        # 返回计算得到的三个梯度张量
        return grad_query, grad_key, grad_value


# 调用函数 flex_attention_backward.py_impl，并传入 DispatchKey.Autograd 作为参数，
# 这里使用 autograd_not_implemented 函数生成一个未实现 autograd 功能的错误，
# 并设置 deferred_error=True 表示这是一个延迟错误，即在使用时才会触发。
flex_attention_backward.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(flex_attention_backward, deferred_error=True)
)
```