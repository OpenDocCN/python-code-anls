# `.\pytorch\torch\distributed\_tensor\experimental\attention.py`

```
import contextlib
import weakref
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Protocol, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
from torch import nn
from torch.distributed._tensor import distribute_module, DTensor, Replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle

aten = torch.ops.aten


def sdpa_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    """
    Handle a specific operation call for scaled dot product attention.

    Args:
        op_call (torch._ops.OpOverload): Operation call to be handled.
        args (Tuple[object, ...]): Arguments passed to the operation.
        kwargs (Dict[str, object]): Keyword arguments passed to the operation.

    Returns:
        object: Result of the operation.
    """
    # extract local tensor and sharding infos to a OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # sharding propagation
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    assert output_sharding is not None, "output sharding should not be None"
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    local_results = _scaled_dot_product_ring_flash_attention(
        op_info.mesh,
        *op_info.local_args,  # type: ignore[arg-type]
        **op_info.local_kwargs,  # type: ignore[arg-type]
    )

    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)


def _merge_sdpa(
    chunks: List[torch.Tensor], logsumexps: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge multiple scaled dot product attention chunks using logsumexp for rescaling.

    Args:
        chunks (List[torch.Tensor]): List of scaled dot product attention chunks.
        logsumexps (List[torch.Tensor]): List of logsumexps corresponding to each chunk.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Merged scaled dot product attention and its logsumexp.
    """
    assert len(chunks) == len(logsumexps)

    # LSE may be padded in the sequence dimension such as with memory efficient attention.
    seq_len = chunks[0].size(2)
    logsumexps = [lse[:, :, :seq_len] for lse in logsumexps]

    softmax_lse = torch.stack([lse.exp() for lse in logsumexps]).sum(dim=0).log_()

    out = []
    for i, (chunk, chunk_lse) in enumerate(zip(chunks, logsumexps)):
        softmax_lse_corrected = torch.exp(chunk_lse - softmax_lse)
        out_corrected = chunk * softmax_lse_corrected.unsqueeze(-1).to(chunk.dtype)
        out.append(out_corrected)
    out = torch.stack(out).sum(dim=0)

    return out, softmax_lse


def _scaled_dot_product_ring_flash_attention(
    mesh: DeviceMesh,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Perform scaled dot product attention operation on a device mesh.

    Args:
        mesh (DeviceMesh): Device mesh for distributing computation.
        query (torch.Tensor): Query tensor.
        key (torch.Tensor): Key tensor.
        value (torch.Tensor): Value tensor.
        dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        is_causal (bool, optional): Whether the attention is causal. Defaults to False.
        return_debug_mask (bool, optional): Whether to return debug mask. Defaults to False.
        scale (Optional[float], optional): Scaling factor. Defaults to None.

    Returns:
        Tuple[torch.Tensor, ...]: Tuple containing attention output tensors.
    """
    # 如果 return_debug_mask 为真，抛出未实现错误，暂不支持返回调试掩码
    if return_debug_mask:
        raise NotImplementedError("return_debug_mask is not supported yet")

    # 调用 _templated_ring_attention 函数，使用 Torch 的自定义操作进行缩放点积闪光注意力计算
    return _templated_ring_attention(
        mesh,                           # 网格参数
        torch.ops.aten._scaled_dot_product_flash_attention,  # 使用 Torch 的自定义操作
        query=query,                    # 查询向量
        key=key,                        # 键向量
        value=value,                    # 值向量
        dropout_p=dropout_p,            # 丢弃率
        is_causal=is_causal,            # 是否因果
        scale=scale,                    # 缩放因子
    )
# 定义一个高效的环形注意力机制，利用缩放点积进行计算
def _scaled_dot_product_ring_efficient_attention(
    mesh: DeviceMesh,            # 设备网格，用于指定计算的设备信息
    query: torch.Tensor,         # 查询张量，用于计算注意力分布
    key: torch.Tensor,           # 键张量，用于计算查询和键之间的相似度
    value: torch.Tensor,         # 值张量，根据注意力分布加权得到最终的输出
    attn_bias: Optional[torch.Tensor] = None,  # 注意力偏置，默认为None，目前不支持
    dropout_p: float = 0.0,       # dropout概率，默认为0.0，用于随机失活
    is_causal: bool = False,      # 是否使用因果注意力，默认为False
    compute_log_sumexp: bool = True,  # 是否计算对数和指数，默认为True
    *,
    scale: Optional[float] = None,  # 缩放因子，用于缩放点积结果的范围，默认为None
) -> Tuple[torch.Tensor, ...]:    # 返回类型为包含多个张量的元组
    if attn_bias is not None:
        raise NotImplementedError("attn_bias is not supported yet")  # 如果传入了注意力偏置，则抛出未实现异常
    if not compute_log_sumexp:
        raise NotImplementedError("compute_log_sumexp must be set")  # 如果未设置计算对数和指数，则抛出未实现异常

    return _templated_ring_attention(
        mesh,
        torch.ops.aten._scaled_dot_product_efficient_attention,  # 使用torch内置操作进行高效缩放点积注意力计算
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        compute_log_sumexp=compute_log_sumexp,
    )


# 定义一个基于cuDNN的环形注意力机制
def _scaled_dot_product_ring_cudnn_attention(
    mesh: DeviceMesh,            # 设备网格，用于指定计算的设备信息
    query: torch.Tensor,         # 查询张量，用于计算注意力分布
    key: torch.Tensor,           # 键张量，用于计算查询和键之间的相似度
    value: torch.Tensor,         # 值张量，根据注意力分布加权得到最终的输出
    attn_bias: Optional[torch.Tensor] = None,  # 注意力偏置，默认为None，目前不支持
    dropout_p: float = 0.0,       # dropout概率，默认为0.0，用于随机失活
    is_causal: bool = False,      # 是否使用因果注意力，默认为False
    return_debug_mask: bool = True,  # 是否返回调试掩码，默认为True
    *,
    scale: Optional[float] = None,  # 缩放因子，用于缩放点积结果的范围，默认为None
) -> Tuple[torch.Tensor, ...]:    # 返回类型为包含多个张量的元组
    if not return_debug_mask:
        raise NotImplementedError("return_debug_mask must be set")  # 如果未设置返回调试掩码，则抛出未实现异常

    return _templated_ring_attention(
        mesh,
        torch.ops.aten._scaled_dot_product_cudnn_attention,  # 使用cuDNN加速的缩放点积注意力计算
        query=query,
        key=key,
        value=value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        return_debug_mask=return_debug_mask,
        scale=scale,
    )


# 实现环形数据块的旋转操作
def _ring_rotate(block: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    rank = dist.get_rank(pg)      # 获取当前进程在进程组中的排名
    size = dist.get_world_size(pg)  # 获取进程组中的总进程数

    # 根据进程的排名和总数，设置输入和输出数据块的分割大小
    input_split_sizes = [0] * size
    input_split_sizes[(rank + 1) % size] = len(block)
    output_split_sizes = [0] * size
    output_split_sizes[(rank - 1) % size] = len(block)

    # 使用所有到所有通信模式，将数据块在进程组中旋转
    out = ft_c.all_to_all_single_autograd(
        block, input_split_sizes, output_split_sizes, pg
    )
    return out


# 定义一个抽象的注意力操作协议
class AttentionOp(Protocol):
    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args: object,
        is_causal: bool = False,
        **kwargs: object,
    ) -> Tuple[torch.Tensor, ...]:
        ...


# 实现一个模板化的环形注意力操作
def _templated_ring_attention(
    mesh: DeviceMesh,            # 设备网格，用于指定计算的设备信息
    op: AttentionOp,             # 注意力操作的实例，遵循AttentionOp协议
    query: torch.Tensor,         # 查询张量，用于计算注意力分布
    key: torch.Tensor,           # 键张量，用于计算查询和键之间的相似度
    value: torch.Tensor,         # 值张量，根据注意力分布加权得到最终的输出
    *args: object,               # 其他位置参数，传递给注意力操作
    is_causal: bool = False,      # 是否使用因果注意力，默认为False
    **kwargs: object,            # 其他关键字参数，传递给注意力操作
) -> Tuple[torch.Tensor, ...]:    # 返回类型为包含多个张量的元组
    """
    This is a generalized ring attention implementation that can support multiple attention ops.

    Parameters
    ----------
    op:
        The attention op to use
    *args:
        additional args are passed to the op
    **kwargs:
        additional kwargs are passed to the op
    """
    # 实际上这里应该调用op对象的__call__方法，但具体实现未给出
    pass  # 此处暂时不实现具体细节，仅作为模板化的环形注意力操作
    if is_causal and (query.size(2) != key.size(2)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )


    # 如果是因果的并且查询（query）的长度与键（key）的长度不同，抛出未实现错误
    if is_causal and (query.size(2) != key.size(2)):
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )



    if isinstance(mesh, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, List[dist.ProcessGroup]] = mesh
    else:
        pg = mesh.get_group()
    assert isinstance(pg, dist.ProcessGroup), "process group must be single dimension"
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)


    # 确定处理组（process group）pg的类型和大小
    if isinstance(mesh, dist.ProcessGroup):
        pg: Union[dist.ProcessGroup, List[dist.ProcessGroup]] = mesh
    else:
        pg = mesh.get_group()
    assert isinstance(pg, dist.ProcessGroup), "process group must be single dimension"
    rank = dist.get_rank(pg)
    size = dist.get_world_size(pg)



    next_kv = None

    chunks = []
    logsumexps = []
    for i in range(size):
        # overlap communication with compute
        if next_kv is not None:
            next_kv = ft_c.wait_tensor(next_kv)
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        if i < (size - 1):
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = _ring_rotate(next_kv, pg)

        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        if is_causal_behavior != _CausalBehavior.SKIP:
            local_results = op(
                query,
                key,
                value,
                *args,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )
            chunks.append(local_results[0])
            logsumexps.append(local_results[1])


    # 初始化变量
    next_kv = None

    chunks = []
    logsumexps = []
    for i in range(size):
        # 重叠通信和计算
        if next_kv is not None:
            next_kv = ft_c.wait_tensor(next_kv)
            key = next_kv[: key.numel()].reshape(key.shape)
            value = next_kv[key.numel() :].reshape(value.shape)

        # 如果不是最后一个进程，进行下一个键值对的环形旋转
        if i < (size - 1):
            next_kv = torch.cat([key.flatten(), value.flatten()])
            next_kv = _ring_rotate(next_kv, pg)

        # 确定因果行为
        is_causal_behavior = _is_causal_behavior(
            rank=rank, world_size=size, i=i, is_causal=is_causal
        )

        # 如果因果行为不是跳过，则执行操作并收集结果
        if is_causal_behavior != _CausalBehavior.SKIP:
            local_results = op(
                query,
                key,
                value,
                *args,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )
            chunks.append(local_results[0])
            logsumexps.append(local_results[1])



    out, softmax_lse = _merge_sdpa(chunks, logsumexps)

    local_results = (out, softmax_lse) + local_results[2:]
    return local_results


    # 合并各个处理器的结果并计算 softmax_lse
    out, softmax_lse = _merge_sdpa(chunks, logsumexps)

    # 将本地结果与全局结果合并
    local_results = (out, softmax_lse) + local_results[2:]
    return local_results
def _scaled_dot_product_chunk_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    size: int,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    This is a single node chunked implementation of
    _scaled_dot_product_ring_flash_attention used for verifying
    the correctness of the backwards pass.
    """

    if return_debug_mask:
        # 如果设置了 return_debug_mask 参数，抛出未实现错误
        raise NotImplementedError("return_debug_mask is not supported yet")

    if is_causal and (query.size(2) != key.size(2)):
        # 如果是因果性注意力，并且查询和键的序列长度不相等，抛出未实现错误
        raise NotImplementedError(
            "is_causal requires the same query and context sequence lengths"
        )

    # 计算每个块的查询长度和上下文长度
    query_len = query.size(2) // size
    ctx_len = key.size(2) // size

    global_out = []  # 存储全局输出
    global_softmax_lse = []  # 存储全局 softmax 的 log-sum-exp

    # 遍历每个块
    for rank in range(size):
        chunks = []  # 存储本地块的结果
        logsumexps = []  # 存储本地 softmax 的 log-sum-exp

        # 提取当前块的查询部分
        chunk_query = query[:, :, rank * query_len : (rank + 1) * query_len]

        # 遍历所有块以处理跨块的注意力
        for i in range(size):
            src_rank = (rank - i) % size
            # 提取当前块的键和值部分
            chunk_key = key[:, :, src_rank * ctx_len : (src_rank + 1) * ctx_len]
            chunk_value = value[:, :, src_rank * ctx_len : (src_rank + 1) * ctx_len]

            # 确定因果性注意力的行为
            is_causal_behavior = _is_causal_behavior(
                rank=rank, world_size=size, i=i, is_causal=is_causal
            )

            # 如果不是跳过行为，执行注意力计算
            if is_causal_behavior != _CausalBehavior.SKIP:
                local_results = torch.ops.aten._scaled_dot_product_flash_attention(
                    chunk_query,
                    chunk_key,
                    chunk_value,
                    dropout_p=dropout_p,
                    is_causal=is_causal_behavior.value,
                    scale=scale,
                )
                chunks.append(local_results[0])  # 存储本地注意力的输出
                logsumexps.append(local_results[1])  # 存储本地 softmax 的 log-sum-exp

        # 合并当前块的所有注意力结果
        out, softmax_lse = _merge_sdpa(chunks, logsumexps)
        global_out.append(out)  # 将全局输出添加到列表中
        global_softmax_lse.append(softmax_lse)  # 将全局 softmax 的 log-sum-exp 添加到列表中

    # 拼接所有块的结果，按维度2进行拼接
    global_out = torch.concat(global_out, dim=2)
    global_softmax_lse = torch.concat(global_softmax_lse, dim=2)

    # 返回全局输出和全局 softmax 的 log-sum-exp
    local_results = (global_out, global_softmax_lse) + local_results[2:]
    return local_results


def sdpa_backward_handler(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    """
    Handler for the backward pass of scaled dot product attention.

    Args:
        op_call: Operation call that triggered backward pass.
        args: Arguments passed to the operation.
        kwargs: Keyword arguments passed to the operation.

    Returns:
        object: Result of handling the backward pass.

    This function redistributes grad_output tensor to match the placement of the output tensor
    and performs sharding propagation for distributed tensors.
    """

    # 将 grad_output 张量重新分布到与 output 张量相同的位置
    args = list(args)
    assert isinstance(args[0], DTensor) and isinstance(args[4], DTensor)
    args[0] = args[0].redistribute(args[4].device_mesh, args[4].placements)
    args = tuple(args)

    # 提取本地张量和分片信息到 OpInfo
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)

    # 执行分片传播
    DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
    output_sharding = op_info.output_sharding
    # 断言确保 output_sharding 不是 None，否则抛出错误信息
    assert output_sharding is not None, "output sharding should not be None"
    # 断言确保 output_sharding 不需要重新分配数据，否则抛出错误信息
    assert not output_sharding.needs_redistribute, "inputs need to be redistributed"

    # 调用 _scaled_dot_product_ring_flash_attention_backward 函数进行计算
    # 使用 op_info 中的 mesh 和 local_args、local_kwargs 作为参数
    local_results = _scaled_dot_product_ring_flash_attention_backward(
        op_info.mesh,
        *op_info.local_args,  # type: ignore[arg-type]
        **op_info.local_kwargs,  # type: ignore[arg-type]
    )

    # 调用 DTensor._op_dispatcher.wrap 函数，将 local_results 包装并返回
    return DTensor._op_dispatcher.wrap(local_results, output_sharding.output_spec)
def _scaled_dot_product_ring_flash_attention_backward(
    mesh: DeviceMesh,
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    philox_seed: torch.Tensor,
    philox_offset: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    # 获取当前进程组
    pg = mesh.get_group()
    # 确保进程组是单维度的
    assert isinstance(pg, dist.ProcessGroup), "must be single dimension"
    # 获取当前进程在进程组中的排名
    rank = dist.get_rank(pg)
    # 获取进程组的总大小
    size = dist.get_world_size(pg)

    # rank 0 发送给 rank 1, rank 1 发送给 rank 2, ..., rank n-1 发送给 rank 0
    right_dsts = list(range(1, size)) + [0]

    next_kv = None

    out_grad_queries = []
    out_grad_keys = []
    out_grad_values = []

    # 将列表中的张量堆叠并求和，以避免累积误差
    out_grad_query = torch.stack(out_grad_queries).sum(dim=0)
    out_grad_key = torch.stack(out_grad_keys).sum(dim=0)
    out_grad_value = torch.stack(out_grad_values).sum(dim=0)

    # 返回三个张量作为元组结果
    return out_grad_query, out_grad_key, out_grad_value


customized_ops = {
    aten._scaled_dot_product_flash_attention.default: sdpa_handler,
    aten._scaled_dot_product_flash_attention_backward.default: sdpa_backward_handler,
}


@contextlib.contextmanager
def attention_context_parallel() -> Generator[None, None, None]:
    """
    This is a context manager that force enables attention context parallel
    optimizations for all scaled_dot_product_attention ops.

    This currently only supports ring attention and the
    SDPBackend.FLASH_ATTENTION backend. See sdpa_kernel.

    Non-flash attention backends will result in incorrect results.
    """
    # 保存旧的操作处理程序
    old_handlers = DTensor._op_dispatcher._custom_op_handlers
    # 更新操作分发器的自定义操作处理程序，加入自定义操作
    DTensor._op_dispatcher._custom_op_handlers = {**old_handlers, **customized_ops}

    yield

    # 恢复操作分发器的旧的操作处理程序
    DTensor._op_dispatcher._custom_op_handlers = old_handlers


class AttentionContextParallel(ParallelStyle):
    """
    Applies context parallel optimizations to the attention layer.

    This will work for nn.MultiHeadedAttention and custom attention layers that
    call F.scaled_dotproduct_attention with a simliar signature.

    This expects the `forward` method consumes either:

    * a single tensor for self attention
    * one argument for each of: query, key, value

    This currently only supports ring attention and the
    SDPBackend.FLASH_ATTENTION backend. See sdpa_kernel.

    Non-flash attention backends will result in incorrect results.
    """

    # 使用弱引用字典存储每个 nn.Module 的上下文管理器
    _CONTEXT_MANAGERS: "weakref.WeakKeyDictionary[nn.Module, Any]" = (
        weakref.WeakKeyDictionary()
    )
    # 将指定模块在给定设备网格上分布
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # 检查设备网格的类型是否为DeviceMesh，如果不是则抛出数值错误异常
        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(
                f"{type(device_mesh)} is not supported by {type(self)} yet."
            )

        # 检查设备网格的维度是否为1，如果不是则抛出数值错误异常
        if not device_mesh.ndim == 1:
            raise ValueError

        # 将模块在设备网格上分布，并返回分布后的模块
        return distribute_module(
            module,
            device_mesh,
            input_fn=self._input_fn,  # 输入函数，类型忽略参数
            output_fn=self._output_fn,  # 输出函数，类型忽略参数
        )

    @classmethod
    def _input_fn(
        cls,
        module: nn.Module,
        inputs: Tuple[Union[torch.Tensor, int, float], ...],
        device_mesh: DeviceMesh,
    ) -> Tuple[Union[torch.Tensor, int, float], ...]:
        # TODO(d4l3k); this should be Shard(2), need to fix Linear layer rules
        # 设置输入的放置策略为Replicate()
        placement = [Replicate()]

        # 定义反向传播的钩子函数，用于管理上下文
        def backward_hook(grad: torch.Tensor) -> None:
            if module in cls._CONTEXT_MANAGERS:
                # 如果模块存在于上下文管理器中，则退出上下文并删除管理器
                cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
                del cls._CONTEXT_MANAGERS[module]

        # 将输入转换为DTensor格式，如果是torch.Tensor并且不是DTensor则转换为本地DTensor
        inp = []
        for input in inputs:
            if isinstance(input, torch.Tensor) and not isinstance(input, DTensor):
                input = DTensor.from_local(
                    input, device_mesh, placement, run_check=False
                )

            # 如果输入是torch.Tensor并且需要梯度，则注册钩子函数
            if isinstance(input, torch.Tensor) and input.requires_grad:
                input.register_hook(backward_hook)

            inp.append(input)

        # 创建并进入并行注意力上下文管理器，并将其存储到类属性中
        manager = attention_context_parallel()
        manager.__enter__()
        cls._CONTEXT_MANAGERS[module] = manager

        # 返回输入的元组
        return tuple(inp)

    @classmethod
    def _output_fn(
        cls,
        module: nn.Module,
        outputs: Union[torch.Tensor, Tuple[Union[torch.Tensor, int, float], ...]],
        device_mesh: DeviceMesh,
    ) -> Union[
        Union[torch.Tensor, int, float], Tuple[Union[torch.Tensor, int, float], ...]
    ]:
        # 退出当前模块的并行注意力上下文管理器，并删除管理器
        cls._CONTEXT_MANAGERS[module].__exit__(None, None, None)
        del cls._CONTEXT_MANAGERS[module]

        # 定义反向传播的钩子函数，用于管理上下文
        def backward_hook(grad: torch.Tensor) -> None:
            if module not in cls._CONTEXT_MANAGERS:
                # 如果模块不在上下文管理器中，则创建新的并行注意力上下文管理器并存储到类属性中
                manager = attention_context_parallel()
                manager.__enter__()
                cls._CONTEXT_MANAGERS[module] = manager

        # 将输出转换为本地张量格式，如果是DTensor则转换为本地张量
        out = []
        for output in [outputs] if isinstance(outputs, torch.Tensor) else outputs:
            output = output.to_local() if isinstance(output, DTensor) else output

            # 如果输出是torch.Tensor并且需要梯度，则注册钩子函数
            if isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(backward_hook)

            out.append(output)

        # 如果输出是torch.Tensor，则返回第一个元素，否则返回元组
        if isinstance(outputs, torch.Tensor):
            return out[0]

        return tuple(out)
# 定义一个枚举 `_CausalBehavior`，表示三种因果行为状态：跳过、非因果、因果
class _CausalBehavior(Enum):
    SKIP = None  # 跳过状态，未定义
    NOT_IS_CAUSAL = False  # 非因果状态，表示不应用因果关系
    IS_CAUSAL = True  # 因果状态，表示应用因果关系


def _is_causal_behavior(
    rank: int, world_size: int, i: int, is_causal: bool
) -> _CausalBehavior:
    """
    Calculate is_causal behavior for each KV block. The attention can either be
    calculated in full, not at all or with the causal mask applied.
    """
    # 如果不需要应用因果关系，直接返回非因果状态
    if not is_causal:
        return _CausalBehavior.NOT_IS_CAUSAL

    # 如果是第一个块，返回因果状态
    if i == 0:
        return _CausalBehavior.IS_CAUSAL

    # 计算源头排名，根据排名判断是否为因果关系
    source_rank = (rank - i) % world_size
    if source_rank < rank:
        return _CausalBehavior.NOT_IS_CAUSAL  # 如果源头排名小于当前排名，返回非因果状态
    else:
        return _CausalBehavior.SKIP  # 否则返回跳过状态
```