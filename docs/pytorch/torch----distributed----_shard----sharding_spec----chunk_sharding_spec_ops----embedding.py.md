# `.\pytorch\torch\distributed\_shard\sharding_spec\chunk_sharding_spec_ops\embedding.py`

```py
# mypy: allow-untyped-defs

# 导入PyTorch库及分布式支持
import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter

# 导入私有函数用于处理分片张量的相关操作
from ._common import (
    _all_gather_base_input,
    _handle_col_wise_sharding_base,
    _handle_max_norm_col_wise,
    _handle_row_wise_mask,
)

# 定义自定义分片规范操作的装饰器，用于处理torch.nn.functional.embedding函数
@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.embedding)
def sharded_embedding(types, args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding``.
    This method computes a sharded embedding lookup and has the following limitations:

    1. Supports only sharding of ``weight``.
    2. Supports only ``ChunkShardingSpec``.
    3. Supports only a single local shard per rank.
    4. Supports all specs except for scale_grad_by_freq, sparse, etc.

    Based on the dimension that the weight is sharded on, there are two
    algorithms:

    ROWWISE SHARDING
    ================
    For row-wise sharding the weight is sharded on dimension 0.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (10 x 17) and W is sharded across
    4 GPUs creating 3 shard of (3 x 17) and 1 shard of (1 x 17).
    The algorithm is as follows:

    1. First the input is all gathered to all ranks, since this is SPMD and
       input is actually sharded across all ranks. The inputs then become a
       4 (4 x 6) tensor on each rank. For example if the given input is
       tensor([[6, 5, 2, 9, 6, 3],
               [3, 1, 2, 4, 7, 6],
               [4, 0, 4, 9, 8, 9],
               [8, 6, 6, 4, 6, 1]])
       on rank 0.
       Then on every rank, we will have this tensor.
       If input itself is already replicated, no all-gather will be done.
    2. Next, we mask the ID which are not stored on that rank.
       For example on rank 0, we store ID [0, 1, 2]. We only keep the ID
       inside the set of numbers. The rest of them will be masked to an extra row.
       The masked matrix will be used for embedding look up and is like:
       tensor([[4, 4, 2, 4, 4, 4],
               [4, 1, 2, 4, 4, 4],
               [4, 0, 4, 4, 4, 4],
               [4, 4, 4, 4, 4, 1]])
       The reason of having an extra row (aka, number 4 in the example) is
       because when max_norm is specified only weight which has looked will
       be re-normed so mask IDs whose embeddings are not stored in current
       rank will to an extra row will ensure max_norm still works as expected.
    3. If max_norm is specified, the extra row guarantees that the mask ID will
       not affect the behavior of weigh re-norm.

    COLWISE SHARDING
    ================
    For col-wise sharding the weight is sharded on dimension 1.
    """
    pass
    """
    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 3 shards of (16 x 5) and 1 shard of (16 x 2).
    The algorithm is as follows:
    
    1. First the input is broadcasted to all ranks, since this is SPMD we
       actually do an all_gather for all the inputs resulting in 4 (4 x 6)
       inputs on each rank.
    2. Next we perform local embedding lookup operation by apply each
       input (4 x 6) with the local shard (16 x 5) ((16 x 2) for the last).
       This results in 4 (5 x 6 x 4) ((2 x 6 x 4) for the last) matrices
       on each rank. We transpose dim 0 and dim 2.
    3. Next, we concat these 4 matrices and perform an all2all to share the
       appropriate (5 x 6 x 4) or (2 x 6 x 4) matrices to each rank.
    4. Now, each rank receives a (17 x 6 x 4) matrix which is basically the
       size of the result we need.
    5. If placements are not in order any appropriate rearrangement of columns
       are done for the (17 x 6 x 4) matrix and finally we transpose the
       dim 0 and dim 2 again.
    6. If max_norm is specified, we manually sum up the norm and renorm. Because
       the renorm must be in place, we need to override the local_shard to mimic
       this behavior.
    """
    
    # Validate input parameters
    _validate_embedding_param(args, kwargs)
    
    # Extract input parameters
    input = args[0]
    weight = args[1]
    max_norm = kwargs.get("max_norm")
    norm_type = kwargs.get("norm_type")
    padding_idx = kwargs.get("padding_idx")
    
    # Obtain the local shard of the weight tensor
    local_shard = weight.local_tensor().contiguous()
    
    # Obtain sharding information
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    
    # Handle different sharding dimensions
    if sharding_dim == 1:
        # Handle column-wise sharding
        output, local_shard = _handle_col_wise_sharding(
            input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, pg
        )
        # Update local shard after processing
        weight.local_shards()[0].tensor = local_shard
        return output
    elif sharding_dim == 0:
        # Handle row-wise sharding
        return _handle_row_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            max_norm,
            norm_type,
            padding_idx,
            rank,
            pg,
        )
    else:
        # Raise an error for unsupported sharding dimension
        raise RuntimeError(
            f"nn.Embedding weight sharded on dim {sharding_dim} not supported!"
        )
# 验证分片嵌入操作的输入参数有效性
def _validate_embedding_param(args, kwargs):
    """
    Validate input params of sharded embedding op.

    Args:
        input: list of ID used for lookup.
        weight: sharded weight tensor.
        kwargs: same as normal Embedding.

    Return: None.
    """

    # 从参数中获取输入和权重信息
    input = args[0]
    weight = args[1]
    max_norm = kwargs.get("max_norm")  # 获取max_norm参数值
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq")  # 获取scale_grad_by_freq参数值
    sparse = kwargs.get("sparse")  # 获取sparse参数值

    # 验证类型
    if not isinstance(input, torch.Tensor):
        raise TypeError("input need to be torch.Tensor")
    if not isinstance(weight, ShardedTensor):
        raise TypeError("weight needs to be ShardedTensor")

    weight_size = weight.size()

    # 验证权重张量维度是否为2
    if len(weight_size) != 2:
        raise ValueError("Weight needs to have exactly 2 dims")

    # 验证输入索引范围是否合法
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

    # 验证是否支持scale_grad_by_freq选项
    if scale_grad_by_freq:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "scale_grad_by_freq" not supported!'
        )

    # 验证是否支持sparse选项
    if sparse:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "sparse" not supported!'
        )

    # 验证max_norm是否大于0
    if max_norm and max_norm <= 0.0:
        raise ValueError('"max_norm" must be larger than zero!')

    # 验证权重分片规格是否为ChunkShardingSpec
    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")

    # 验证本地分片数量是否为1
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")


def _handle_col_wise_sharding(
    input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, pg
):
    """
    Entry-point function to handle the logic of col-wise sharding of weight
    for embedding. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: col-wise shared local weight used for lookup.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
        pg: process group.

    Returns: final result of lookup.
    """
    # 对于非复制张量，首先在所有进程中收集输入数据。
    gathered_inputs = all_gather(input, group=pg)

    if max_norm is not None:
        # 如果设置了最大范数，按列处理权重 inplace 修改
        local_shard = _handle_max_norm_col_wise(
            max_norm, norm_type, local_shard, input, world_size, gathered_inputs, pg
        )

    # 调用基础的按列切片处理函数来处理嵌入矩阵
    output = _handle_col_wise_sharding_base(
        torch.nn.functional.embedding,  # 使用 PyTorch 的嵌入函数
        len(input.size()),  # 输入张量的维度数量
        input,  # 输入张量
        world_size,  # 总进程数量
        weight,  # 权重
        local_shard,  # 本地分片
        pg,  # 进程组
        gathered_inputs,  # 收集的输入数据
        padding_idx=padding_idx,  # 填充索引（可选参数）
    )
    # 返回处理后的输出和本地分片
    return (output, local_shard)
# 定义处理按行分片的函数，用于嵌入权重的按行分片逻辑入口点
def _handle_row_wise_sharding(
    input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, rank, pg
):
    """
    Entry-point function to handle the logic of row-wise sharding of weight
    for embedding. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: row-wise shared local weight used for lookup.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
        rank: # of cuda process.
        pg: process group.

    Returns: final result of lookup.
    """
    # 使用 _all_gather_base_input 函数在非 Replicated Tensor 上首先进行全局收集输入
    gather_inp = _all_gather_base_input(input, pg)

    # 根据分片规格对输入进行掩码处理
    lookup_input, padding_idx, padding_row = _handle_row_wise_mask(
        gather_inp, padding_idx, weight, world_size, rank
    )

    # 当 max_norm 参数不为 None 时，对每个嵌入向量进行归一化处理
    # 这是目前的一个临时解决方案，参见 GitHub 问题: #81717
    if max_norm is not None:
        torch.nn.functional.embedding(
            torch.unique(lookup_input)[:-1],
            local_shard,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
        )
        max_norm = None

    # 使用 torch.nn.functional.embedding 函数进行嵌入查找操作
    local_input_embeddings = torch.nn.functional.embedding(
        lookup_input,
        torch.cat([local_shard, padding_row]),
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
    )

    # TODO: 将结果作为 PartialTensor 处理
    # 使用 reduce_scatter 函数对本地分片进行归约操作
    local_shards = local_input_embeddings.chunk(pg.size())
    return reduce_scatter(
        torch.empty_like(local_shards[0]),
        list(local_shards),
        group=pg,
    )
```