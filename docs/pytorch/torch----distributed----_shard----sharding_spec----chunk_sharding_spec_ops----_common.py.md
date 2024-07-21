# `.\pytorch\torch\distributed\_shard\sharding_spec\chunk_sharding_spec_ops\_common.py`

```
# mypy: allow-untyped-defs  # 允许未类型化的定义，这是针对类型检查工具Mypy的设置

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式通信模块
from torch.distributed._shard.sharded_tensor import ShardedTensor  # 导入ShardedTensor类
from torch.distributed._shard.sharded_tensor._ops._common import _sharded_op_common  # 导入_sharded_op_common函数
from torch.distributed._shard.sharding_spec import ChunkShardingSpec  # 导入ChunkShardingSpec类
from torch.distributed._shard.sharding_spec._internals import (
    get_chunk_sharding_params,  # 导入get_chunk_sharding_params函数
    get_chunked_dim_size,  # 导入get_chunked_dim_size函数
    get_split_size,  # 导入get_split_size函数
)
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op  # 导入custom_sharding_spec_op函数
from torch.distributed.nn.functional import (
    _all_gather_base,  # 导入_all_gather_base函数
    all_reduce,  # 导入all_reduce函数
    all_to_all_single,  # 导入all_to_all_single函数
)


def _chunk_sharding_spec_check(spec, op):
    """
    For the given op implementation check if the sharding spec is ChunkShardingSpec.
    给定操作实现，检查是否为ChunkShardingSpec分片规范。
    """
    if not isinstance(spec, ChunkShardingSpec):  # 如果spec不是ChunkShardingSpec的实例
        raise NotImplementedError(
            f"Only ChunkShardingSpec supported for '{op.__name__}'."
        )  # 抛出NotImplementedError，只支持ChunkShardingSpec用于op的名称。


def _register_sharded_op_on_local_tensor(
    op, early_stop_func=None, extra_check=None, customized_func=None
):
    """
    Handles ``__torch_function__`` dispatch for ops which are performed on
    the single local tensor of the sharded tensor such as op like
    ``torch.nn.functional.softmax`` or ``torch.Tensor.view``.

    For more complicated ops, a customized func can be used to generate
    the new local tensor, sharding spec and sharded tensor size.

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.
        customized_func (Callable, optional): the func for customized logic
            to generate the new local tensor, sharding spec and sharded tensor size.
            Default: if ``None``, we simply lower to the real op call with
                the single local tensor of the st.

    Return:
        func (Callable): registered implementation for sharded op for
        ``__torch_function__`` dispatch.
    """
    
    @custom_sharding_spec_op(ChunkShardingSpec, op)
    @_sharded_op_common(op, early_stop_func, extra_check)
    # 定义一个在本地张量上执行分片张量操作的函数，接受类型、参数、关键字参数和进程组作为输入
    def sharded_tensor_op_on_local_tensor(types, args=(), kwargs=None, pg=None):
        # 从参数中获取分片张量对象
        st = args[0]
        # 获取分片规格
        sharding_spec = st.sharding_spec()
        # 如果本地分片数量不等于1，则抛出类型错误异常
        if len(st.local_shards()) != 1:
            raise TypeError(
                f"torch function '{op.__name__}', with args: {args} and "
                f"kwargs: {kwargs} only supported for single local tensor!"
            )
        # 获取分片张量的大小
        st_size = st.size()
        
        # 如果存在自定义函数，则使用自定义函数处理参数和关键字参数
        if customized_func:
            # 调用自定义函数处理参数和关键字参数，获取本地张量、分片规格和分片张量大小
            local_tensor, sharding_spec, st_size = customized_func(args, kwargs, pg)
        else:
            # 否则，将操作应用于分片张量的本地张量和其余参数
            args = (st.local_tensor(), *args[1:])
            local_tensor = op(*args, **kwargs)
        
        # 使用本地张量的连续版本、分片规格、分片张量大小以及可能的进程组初始化分片张量对象
        return ShardedTensor._init_from_local_tensor(
            local_tensor.contiguous(),
            sharding_spec,
            st_size,  # type: ignore[arg-type]
            process_group=pg,
            init_rrefs=st._init_rrefs,
        )
def _handle_col_wise_sharding_base(
    op_func,
    col_dim,
    input,
    world_size,
    weight,
    local_shard,
    pg,
    gathered_inputs,
    mode=None,
    gathered_per_sample_weights=None,
    gathered_offsets=None,
    padding_idx=None,
):
    """
    For col-wise sharding of weight, lots of logic are common.
    So we extract the common logic and put in this function:
    Step 1. To get input from each rank and
    Step 2. To perform the op on the concatenated tensor.
    Step 3. To distribute results to each rank with col rearrangement.
    Step 4. To concatenate all results from all ranks.

    Args:
        op_func: operator which is applied to the input tensor.
        col_dim: dim of result tensor after the operation.
        input: tensor to be applied op on.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: col-wise sharded weight tensor.
        pg: process group.
        gathered_inputs: list of inputs from all ranks. If specified, we
            don't need to communicate with each rank any more.
        mode: aggregation mode of EmbeddingBag.
        gathered_per_sample_weights: per_sample_weights across all ranks.
        gathered_offsets: offsets across all ranks.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.

    Return: final result of input being applied with the op.
    """
    # Iterate over gathered_inputs and apply specified operation function
    results = []
    for i, inp in enumerate(gathered_inputs):
        # Check if the operation function is embedding_bag from torch.nn.functional
        if op_func == torch.nn.functional.embedding_bag:
            result = op_func(
                inp,
                local_shard,
                offsets=gathered_offsets[i] if gathered_offsets is not None else None,
                mode=mode,
                per_sample_weights=gathered_per_sample_weights[i]
                if gathered_per_sample_weights is not None
                else None,
                padding_idx=padding_idx,
            )
        # Check if the operation function is embedding from torch.nn.functional
        elif op_func == torch.nn.functional.embedding:
            result = op_func(
                inp,
                local_shard,
                padding_idx=padding_idx,
            )
        else:
            # Otherwise, apply the operation function with inp and local_shard
            result = op_func(inp, local_shard)
        # Transpose the result tensor along dimension 0 and col_dim
        results.append(torch.transpose(result, 0, col_dim))

    # Distribute results to each rank with column rearrangement using _result_distribute_with_col_rearrange function
    output = _result_distribute_with_col_rearrange(
        results, input, world_size, weight, pg
    )

    # Transpose the output tensor along dimension 0 and col_dim before returning
    return torch.transpose(output, 0, col_dim)


def _result_distribute_with_col_rearrange(results, input, world_size, weight, pg):
    """
    For col-wise sharding of weight, we need to distribute
    """
    # Process results to each rank. We do them in this function.
    # Note that, if the index in the Sharding Spec is not equal to
    # the rank number, we need to do the rearrangement based on the
    # order given by the Sharding Spec (placement).

    Args:
        results: results from ops applied to inputs from all ranks.
            We need to distribute them back to their original ranks.
        input: tensor to be applied op to.
        world_size: number of ranks.
        weight: sharded weight tensor.
        pg: process group.

    Return: column rearranged result.
    """
    # Process results and outputs for all2all.
    # 获取分片维度
    sharding_dim = weight._sharding_spec.dim
    # 获取分片维度的大小
    sharding_dim_size = weight.size(sharding_dim)
    # 复制第一个结果的维度，将第一个维度设为分片维度的大小
    dims = list(results[0].size())
    dims[0] = sharding_dim_size
    # 将所有结果拼接成一个张量
    combined_results = torch.cat(results)
    # 创建一个空张量，形状与dims相同，设备和数据类型与combined_results相同
    output = torch.empty(
        *dims, device=combined_results.device, dtype=combined_results.dtype
    )

    # Compute output splits
    # 计算每个输出分片的大小
    split_size = get_split_size(sharding_dim_size, world_size)
    output_split_sizes = [0] * world_size
    # 遍历权重的分片规格中的排列，设置每个排列对应的输出分片大小
    for idx, placement in enumerate(weight._sharding_spec.placements):
        output_split_sizes[placement.rank()] = get_chunked_dim_size(
            sharding_dim_size, split_size, idx
        )

    # distribute the outputs using all2all.
    # 使用all2all方法分发输出
    output = all_to_all_single(
        output, combined_results, output_split_sizes=output_split_sizes, group=pg
    )

    # Check if we need to rearrange columns appropriately for output.
    # 检查是否需要适当地重新排列输出的列
    rearrange_columns = any(
        idx != placement.rank()
        for idx, placement in enumerate(weight._sharding_spec.placements)
    )
    # 如果不需要重新排列，则直接返回输出
    if not rearrange_columns:
        return output

    # Collect indices for column rearrangement
    # 收集用于列重新排列的索引
    indices = []
    for placement in weight._sharding_spec.placements:
        dim_size = output_split_sizes[placement.rank()]
        start = sum(
            split_size if i < placement.rank() else 0
            for i, split_size in enumerate(output_split_sizes)
        )
        indices += list(range(start, start + dim_size))

    # Perform index select operation to rearrange columns
    # 执行索引选择操作以重新排列列
    return output.index_select(0, torch.tensor(indices, device=output.device))
def _handle_max_norm_col_wise(
    max_norm,
    norm_type,
    local_shard,
    input,
    world_size,
    gathered_inputs,
    pg,
):
    """
    For col-wise sharding of weight, we need to aggregate the
    norm across all ranks before we can perform the proper re-norm.
    Note that, the max_norm logic is only applied to the embedding
    indices that are looked up and not the whole shard.

    Args:
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        local_shard: col-wise shared local weight used for lookup.
        input: tensor to be applied op to.
        world_size: number of ranks.
        gathered_inputs: list of inputs from all ranks.
        pg: process group.

    Return:
        local_shard_norm_renormed: local_shard re-normed to max_norm if the norm is larger
            than it.

    """
    norm_type = norm_type if norm_type is not None else 2.0
    # 获取所有输入的唯一值并进行排重
    unique_inp = torch.unique(torch.cat(gathered_inputs))
    # 计算局部分片的绝对值的 norm_type 次幂之和
    local_shard_sum = torch.sum(
        torch.pow(torch.abs(local_shard), norm_type), dim=1, dtype=local_shard.dtype
    )
    # 对于列分片，首先需要聚合每个秩的幂和，然后计算范数。
    local_shard_sum = all_reduce(local_shard_sum, group=pg)
    # 计算局部分片的范数
    local_shard_norm = torch.pow(local_shard_sum, 1.0 / norm_type)
    # 创建一个与局部分片大小相同的张量，填充为无穷大
    max_norm_tensor = torch.full(
        (local_shard.size(0),),
        float("inf"),
        dtype=local_shard.dtype,
        device=input.device,
    )
    # 将 max_norm 应用于唯一的输入索引
    max_norm_tensor[unique_inp] = max_norm
    # 转置局部分片并确保连续性
    local_shard_t = local_shard.t().contiguous()
    # 根据条件选择归一化张量
    normalized_tensor = torch.where(
        local_shard_norm > max_norm_tensor, max_norm_tensor, local_shard_norm
    )
    # 确保除数不为零
    local_shard_norm[local_shard_norm == 0.0] = 1.0
    # 重新归一化局部分片
    local_shard_norm_renormed = (
        torch.div(torch.mul(local_shard_t, normalized_tensor), local_shard_norm)
        .t()
        .contiguous()
    )
    return local_shard_norm_renormed


def _all_gather_base_input(input, pg):
    """
    Use _all_gather_base to get a concatenated input from each rank.

    Args:
        input: tensor to be applied op on.
        pg: process group.

    Returns:
        gathered_inputs: input gathered from each rank and concat by dim 0.
    """
    # 首先进行 allgather 输入
    gather_inp_size = list(input.size())
    gather_inp_size[0] = input.size(0) * dist.get_world_size(pg)
    gather_inp = torch.empty(gather_inp_size, device=input.device, dtype=input.dtype)
    return _all_gather_base(gather_inp, input, group=pg)


def _handle_row_wise_mask(gather_inp, padding_idx, weight, world_size, rank):
    """
    Mask the input for embedding look-up for IDs which are not stored
    on the current rank. This function also adjust the ``padding_idx``
    """
    """
    This function adjusts the input tensor `gather_inp` based on sharding parameters,
    and computes an adjusted `padding_idx` and `padding_row` for distributed embedding lookup.
    
    Args:
        gather_inp: Tensor containing indices to gather from the embedding table across all ranks.
        padding_idx: Index indicating padding; gradients are not computed for entries at this index.
        weight: Tensor representing the embedding table.
        world_size: Total number of distributed ranks.
        rank: Rank number of the current CUDA process.
    
    Returns:
        lookup_input: Adjusted tensor of indices for masked input.
        padding_idx: Adjusted padding index.
        padding_row: Extra row used during lookup to prevent affecting `max_norm`.
    """
    # Compute the start position and chunk size for the current rank's shard
    (start_pos, chunk_size) = get_chunk_sharding_params(
        weight.size(0), world_size, weight._sharding_spec, rank
    )
    
    # Create a mask to identify indices that fall outside the current rank's shard
    mask = (gather_inp < start_pos) | (gather_inp >= start_pos + chunk_size)
    
    # Adjust `gather_inp` by subtracting `start_pos` to align indices within the current shard
    lookup_input = gather_inp.clone() - start_pos
    
    # Adjust `padding_idx` if it falls within the current shard, otherwise set it to None
    if (
        padding_idx is not None
        and padding_idx >= start_pos
        and padding_idx < (start_pos + chunk_size)
    ):
        padding_idx = padding_idx - start_pos
    else:
        padding_idx = None
    
    # Create a padding row tensor initialized with zeros to prevent `max_norm` affecting it
    padding_row = torch.zeros(
        1, weight.size(1), device=gather_inp.device, dtype=weight.dtype
    )
    
    return lookup_input, padding_idx, padding_row
```