# `.\pytorch\torch\distributed\_shard\sharding_spec\chunk_sharding_spec_ops\embedding_bag.py`

```
# mypy: allow-untyped-defs

# 引入必要的模块和类型注解
from typing import cast, List

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter

# 引入通用的功能函数
from ._common import (
    _all_gather_base_input,
    _handle_col_wise_sharding_base,
    _handle_max_norm_col_wise,
    _handle_row_wise_mask,
)

# 注册自定义的分片规范操作，用于处理embedding_bag函数
@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.embedding_bag)
def sharded_embedding_bag(types, args, kwargs, pg):
    """
    处理 ``torch.nn.functional.embedding_bag`` 的 ``__torch_function__`` 分发。
    该方法计算分片嵌入袋聚合，并具有以下限制:

    1. 仅支持 ``weight`` 的分片。
    2. 仅支持 ``ChunkShardingSpec``。
    3. 每个进程仅支持单个本地分片。
    4. 支持除 scale_grad_by_freq、sparse 等之外的所有规范。

    根据权重分片的维度，有两种算法:

    行分片
    ================
    对于行分片，权重在维度 0 上进行分片。

    算法可以通过以下示例最佳解释。假设输入的维度为 (4 x 6)，权重为 (16 x 17)，并且权重在 4 个 GPU 上分片，创建了 4 个 (4 x 17) 的分片。
    算法如下：

    1. 首先将输入全部聚集到所有进程，因为这是 SPMD，并且输入实际上在所有进程中分片。然后，输入在每个进程上变为 4 (4 x 6) 的张量。例如，如果给定的输入是
       tensor([[6, 5, 2, 9, 6, 3],
               [3, 1, 2, 4, 7, 6],
               [4, 0, 4, 9, 8, 9],
               [8, 6, 6, 4, 6, 1]])
       在进程 0 上。
       然后，在每个进程上，我们将有这个张量。
       如果输入本身已经复制，将不会进行全聚集。
    2. 接下来，我们掩码那些不存储在该进程上的 ID。
       例如，在进程 0 上，我们存储 ID [0, 1, 2]。我们只保留数字集合内的 ID，其余的将被掩码到一个额外的行。
       掩码后的矩阵将用于嵌入查找，如下所示：
       tensor([[4, 4, 2, 4, 4, 4],
               [4, 1, 2, 4, 4, 4],
               [4, 0, 4, 4, 4, 4],
               [4, 4, 4, 4, 4, 1]])
    3. 如果指定了 ``max_norm``，额外的行确保掩码的 ID 不会影响权重重新规范化的行为。

    """
    # 验证输入参数的有效性
    _validate_embedding_bag_param(args, kwargs)

    # 从参数中获取输入、权重、偏移量、每个样本权重、操作模式、最大范数、范数类型、是否包括最后一个偏移量和填充索引
    input = args[0]
    weight = args[1]
    offsets = kwargs.get("offsets")
    per_sample_weights = kwargs.get("per_sample_weights")
    mode = kwargs.get("mode")
    max_norm = kwargs.get("max_norm")
    norm_type = kwargs.get("norm_type")
    include_last_offset = kwargs.get("include_last_offset")
    padding_idx = kwargs.get("padding_idx")

    # 获取本地分片的权重，并确保其连续性
    local_shard = weight.local_tensor().contiguous()

    # 获取权重的分片维度
    sharding_dim = weight._sharding_spec.dim

    # 获取当前进程在分布式环境中的全局大小和排名
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    # 如果包括最后一个偏移量，则去除最后一个偏移量
    if include_last_offset:
        offsets = offsets[:-1]

    # 如果权重在维度1上进行分片
    if sharding_dim == 1:
        # 处理列方向的分片，返回处理后的输出和本地分片
        output, local_shard = _handle_col_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            offsets,
            per_sample_weights,
            mode,
            max_norm,
            norm_type,
            padding_idx,
            pg,
        )

        # 将处理后的本地分片更新回权重对象中的第一个分片
        weight.local_shards()[0].tensor = local_shard

        # 返回处理后的输出
        return output
    # 如果分片维度为0，则调用处理行分片的函数
    elif sharding_dim == 0:
        return _handle_row_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            offsets,
            per_sample_weights,
            mode,
            max_norm,
            norm_type,
            padding_idx,
            rank,
            pg,
        )
    # 如果分片维度不为0，则抛出运行时错误，指明不支持在指定维度上分片权重
    else:
        raise RuntimeError(
            f"nn.EmbeddingBag weight sharded on dim {sharding_dim} not supported!"
        )
def _validate_embedding_bag_param(args, kwargs):
    """
    Validate input params of sharded embeddingBag op.

    Args:
        input: list of ID used for lookup and aggregation.
        weight: sharded weight tensor.
        kwargs: same as normal EmbeddingBag.

    Return: None.
    """

    # Extract input parameters from args and kwargs
    input = args[0]
    weight = args[1]
    offsets = kwargs.get("offsets")
    per_sample_weights = kwargs.get("per_sample_weights")
    mode = kwargs.get("mode")
    max_norm = kwargs.get("max_norm")
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq")
    sparse = kwargs.get("sparse")
    include_last_offset = kwargs.get("include_last_offset")

    # Validate types of input parameters
    if not isinstance(input, torch.Tensor):
        raise TypeError("input need to be torch.Tensor")
    if offsets is not None and not isinstance(offsets, torch.Tensor):
        raise TypeError("offsets need to be torch.Tensor")
    if per_sample_weights is not None and not isinstance(
        per_sample_weights, torch.Tensor
    ):
        raise TypeError("per_sample_weights need to be torch.Tensor")
    if not isinstance(weight, ShardedTensor):
        raise TypeError("weight needs to be ShardedTensor")

    # Validate dimensional constraints
    if len(input.size()) > 2:
        raise ValueError("Input more than 2 dims not supported")
    weight_size = weight.size()
    if len(weight_size) != 2:
        raise ValueError("Weight needs to have exactly 2 dims")

    # Validate index ranges
    if int(torch.min(input).item()) < 0:
        raise ValueError(
            "Index out of range in Input %d %d",
            int(torch.min(input).item()),
            weight_size[1],
        )
    if int(torch.max(input).item()) >= weight_size[0]:
        raise ValueError(
            "Index out of range in Input %d %d",
            int(torch.max(input).item()),
            weight_size[1],
        )

    # Validate conditions based on input and offsets
    if offsets is not None and len(input.size()) != 1:
        raise ValueError("Input dimension needs to be exactly 1 dim")
    if len(input.size()) == 1 and offsets is None:
        raise ValueError("offsets is required for 1D input")

    # Validate per_sample_weights size
    if per_sample_weights is not None and per_sample_weights.size() != input.size():
        raise ValueError(
            f"per_sample_weights size {per_sample_weights.size()} not equal to input size {input.size()}"
        )

    # Set default mode if not provided
    if mode is None:
        mode = "mean"

    # Validate mode value
    if mode not in ["sum", "mean", "max"]:
        raise ValueError(f"mode '{mode}' is not supported")

    # Validate specific flags
    if scale_grad_by_freq:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "scale_grad_by_freq" not supported!'
        )
    if sparse:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "sparse" not supported!'
        )
    if include_last_offset and offsets is None:
        raise ValueError('offsets is required for flag "include_last_offset"!')
    # 如果 include_last_offset 为真且 offsets 的最后一个元素不等于输入张量的大小，
    # 抛出数值错误异常，指示在开启 "include_last_offset" 标志时需要在偏移量末尾具有输入大小！
    if include_last_offset and cast(List[int], offsets)[-1] != input.size(0):
        raise ValueError(
            'offsets need to have the input size in the end when the flag "include_last_offset" is on!'
        )

    # 如果 max_norm 存在且小于等于零，
    # 抛出数值错误异常，指示"max_norm"必须大于零！
    if max_norm and max_norm <= 0.0:
        raise ValueError('"max_norm" must be larger than zero!')

    # 如果权重 weight 的分片规范不是 ChunkShardingSpec 类型，
    # 抛出数值错误异常，表明 ShardedTensor 操作仅支持 ChunkShardingSpec！
    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")

    # 如果权重 weight 的本地分片数量不为1，
    # 抛出数值错误异常，表明仅支持一个本地分片！
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")
# 处理按列分片的逻辑的入口函数，用于embeddingBag的权重按列分片。详细逻辑说明可以在sharded_embedding_bag的注释中找到。

def _handle_col_wise_sharding(
    input,                   # ID列表，用于查找和聚合
    world_size,              # 进程组的数量
    weight,                  # 分片的权重张量
    local_shard,             # 列分片的本地共享权重，用于查找
    offsets,                 # 对于1D输入，每个bag的起始位置列表
    per_sample_weights,      # 加权求和模式下的每个样本权重
    mode,                    # 每个bag的聚合方法
    max_norm,                # 如果给定，每个嵌入向量的范数大于max_norm将被重新归一化为max_norm
    norm_type,               # 计算max_norm选项时的p值
    padding_idx,             # 如果指定，padding_idx位置的条目不会贡献梯度
    pg,                      # 进程组
):
    """
    Entry-point function to handle the logic of col-wise sharding of weight
    for embeddingBag. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding_bag.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: col-wise shared local weight used for lookup.
        offsets: list of start positions of each bag for 1D input.
        per_sample_weights: weights for weighted sum mode.
        mode: aggregation method of each bag.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.
        pg: process group.

    Return:
        output: final result of lookup and aggregation.
        local_shard: col-wise shared local weight used for lookup.
            If max_norm, this will be the renormed weight.
    """

    # 先allgather embedding bag的特殊输入。
    (
        gathered_inputs,
        gathered_per_sample_weights,
        gathered_offsets,
    ) = _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg)

    if max_norm is not None:
        # 如果有max_norm，则对权重进行原地修改
        local_shard = _handle_max_norm_col_wise(
            max_norm, norm_type, local_shard, input, world_size, gathered_inputs, pg
        )

    # 调用基础的按列分片处理函数处理逻辑
    output = _handle_col_wise_sharding_base(
        torch.nn.functional.embedding_bag,
        1,
        input,
        world_size,
        weight,
        local_shard,
        pg,
        gathered_inputs,
        mode=mode,
        gathered_per_sample_weights=gathered_per_sample_weights,
        gathered_offsets=gathered_offsets,
        padding_idx=padding_idx,
    )
    # 返回输出结果和可能被重新归一化的本地共享权重
    return (output, local_shard)


def _handle_row_wise_sharding(
    input,                   # ID列表，用于查找和聚合
    world_size,              # 进程组的数量
    weight,                  # 分片的权重张量
    local_shard,             # 行分片的本地共享权重，用于查找
    offsets,                 # 对于1D输入，每个bag的起始位置列表
    per_sample_weights,      # 加权求和模式下的每个样本权重
    mode,                    # 每个bag的聚合方法
    max_norm,                # 如果给定，每个嵌入向量的范数大于max_norm将被重新归一化为max_norm
    norm_type,               # 计算max_norm选项时的p值
    padding_idx,             # 如果指定，padding_idx位置的条目不会贡献梯度
    rank,                    # 当前rank的索引
    pg,                      # 进程组
):
    """
    Entry-point function to handle the logic of row-wise sharding of weight
    for embeddingBag. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding_bag.)
    """
    Args:
        input: 用于查找和聚合的ID列表。
        world_size: ranks的数量。
        weight: 分片权重张量。
        local_shard: 行共享的本地权重，用于查找。
        offsets: 1D输入每个包的起始位置列表。
        per_sample_weights: 加权求和模式下的样本权重。
        mode: 每个包的聚合方法。
        max_norm: 如果给定，则每个嵌入向量的范数大于max_norm将被重新归一化为max_norm。
                 注意：这将直接修改weight。
        norm_type: max_norm选项中计算的p-norm中的p值。
        padding_idx: 如果指定，则padding_idx位置的条目不贡献梯度；
                    因此，在训练过程中不会更新padding_idx处的嵌入向量，
                    即它保持为固定的“pad”。
                    注意，padding_idx处的嵌入向量不包含在减少中。
        rank: cuda进程的数量。
        pg: 进程组。

    Returns:
        gathered_output: 查找和聚合的最终结果。
    """
    if input.dim() > 1 and per_sample_weights is None:
        # 如果输入维度大于1且没有per_sample_weights，则首先对输入进行allgather，用于非复制张量。
        gather_inp = _all_gather_base_input(input, pg)
    else:
        (
            gathered_inputs,
            gathered_per_sample_weights,
            gathered_offsets,
        ) = _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg)
        cat_dim = 0 if input.dim() != 1 else -1
        gather_inp = torch.cat(gathered_inputs, dim=cat_dim)
        if per_sample_weights is not None:
            per_sample_weights = torch.cat(gathered_per_sample_weights, dim=cat_dim)
        offset_add = 0 if input.dim() > 1 else input.size(0)
        if offsets is not None:
            offsets_list = torch.cat(
                [gathered_offsets[i] + (offset_add * i) for i in range(pg.size())],
                dim=cat_dim,
            )

    # 根据分片规范对输入进行遮罩处理。
    lookup_input, padding_local, padding_row = _handle_row_wise_mask(
        gather_inp, padding_idx, weight, world_size, rank
    )
    if mode == "max":
        padding_row[:] = -float("Inf")

    # 当输入是大张量时，weight的值会改变。
    # 这是目前的一种解决方法。GH问题：#81717。
    if max_norm is not None:
        torch.nn.functional.embedding_bag(
            torch.unique(lookup_input)[:-1],
            local_shard,
            offsets=torch.tensor([0], device=local_shard.device, dtype=torch.long),
            mode=mode,
            per_sample_weights=None,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_local,
        )
        max_norm = None
    # 使用 PyTorch 的 embedding_bag 函数对输入进行嵌入操作，返回结果
    result = torch.nn.functional.embedding_bag(
        lookup_input,
        torch.cat([local_shard, padding_row]),
        offsets=offsets_list if offsets is not None else offsets,  # 如果 offsets 不为 None 则使用 offsets_list，否则使用 offsets；类型标记忽略可能未定义的情况
        mode=mode if mode != "mean" else "sum",  # 如果 mode 不是 "mean" 则使用 mode，否则使用 "sum"
        per_sample_weights=per_sample_weights,
        max_norm=max_norm,
        norm_type=norm_type,
        padding_idx=padding_local,
    )

    # 根据 mode 设置操作类型为求和或最大值
    op = ReduceOp.SUM if mode != "max" else ReduceOp.MAX

    # 将 result 划分为与进程组大小相同的本地分片
    local_shards = result.chunk(pg.size())

    # 使用 reduce_scatter 函数对本地分片进行 reduce 操作，得到结果
    result = reduce_scatter(
        torch.empty_like(local_shards[0]),
        list(local_shards),
        op=op,
        group=pg,
    )

    # 对于 mode 为 "mean" 的情况，需要延迟除法操作，因为各部分的平均值之和并不等于总体的平均值（除数不同）
    if mode == "mean":
        if input.dim() > 1:
            # 计算每个样本非填充部分的大小作为分母
            padding_idx = padding_idx if padding_idx is not None else -1
            split_sizes = torch.sum(
                torch.ne(input, padding_idx), dim=-1, dtype=local_shard.dtype
            )
        else:
            # 对于一维输入，计算每个分段的大小
            split_sizes = torch.cat(
                (
                    offsets[1 : offsets.size(0)] - offsets[0:-1],
                    (input.size(0) - offsets[-1]).unsqueeze(0),
                ),
                dim=-1,
            )
        # 返回 result 与 split_sizes 的逐元素除法结果
        return torch.div(result, split_sizes.unsqueeze(1))

    # 返回适当的本地结果
    return result
def _all_gather
```